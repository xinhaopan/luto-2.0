"""
10_Water.py
Difference in water yield relative to 2010 by land use and agricultural management.
Positive = more water yield than 2010 baseline (beneficial).
Overview row decomposes the water-yield difference into land-use and climate
components using the CCI (Climate Change Impact) already computed by write.py.
"""
import _path_setup  # noqa: F401

import os
import pandas as pd
import xarray as xr

from tools.two_row_figure import (
    LU_COLORS,
    RENAME_AM_NON_AG,
    classify_land_use,
    export_long_tables,
    get_am_colors,
    input_files,
    load_report_source_csv,
    save_three_row_figure,
)
from tools.data_helper import get_path

VALUE_COL = 'Water Net Yield (ML)'
SOURCE_CSV = 'water_yield_separate_watershed'
CCI_CSV = 'water_yield_limits_and_public_land'
CCI_COL = 'Climate Change Impact (ML)'

OVERVIEW_COLORS = {
    'Land use impact': '#6eabb1',
    'Climate change impact': '#d95f02',
}

WATER_LAND_USE_LEGEND_ORDER = [
    'Dryland cropland and horticulture',
    'Dryland grazing (modified pastures)',
    'Grazing (native vegetation)',
    'Irrigated cropland and horticulture',
    'Unallocated land',
    'Non-agricultural land-use',
    'Irrigated grazing (modified pastures)',
]

def load_water_australia(scenario):
    water = load_report_source_csv(scenario, SOURCE_CSV)
    if water.empty:
        return water
    water = (
        water.groupby(
            ['Water Supply', 'Landuse', 'Type', 'Agricultural Management', 'Year'],
            dropna=False,
            as_index=False,
        )[VALUE_COL]
        .sum()
    )
    water = water.replace(RENAME_AM_NON_AG)
    return water.query('`Water Supply` != "ALL" and Landuse != "ALL"').copy()


def _load_cci_australia(scenario):
    """Load Climate Change Impact summed across all watershed regions per year."""
    cci = load_report_source_csv(scenario, CCI_CSV)
    if cci.empty:
        return {}
    return (
        cci.groupby('Year')[CCI_COL].sum() / 1e3  # ML → GL
    ).to_dict()


def _subtract_2010_baseline(rows_list):
    """Given a list of row dicts (year, scenario, category, value),
    subtract the 2010 value for each (scenario, category) pair."""
    df = pd.DataFrame(rows_list)
    if df.empty:
        return df
    baseline = (
        df[df['year'] == 2010]
        .set_index(['scenario', 'category'])['value']
    )
    def sub_baseline(row):
        key = (row['scenario'], row['category'])
        return row['value'] - baseline.get(key, 0.0)
    df['value'] = df.apply(sub_baseline, axis=1)
    return df


def _load_water_correction_by_category(base_path):
    """
    Attribute the 2010 agricultural water yield of cells converted to non-ag
    land-use back to the original dryland/irrigated land-use categories.

    This mirrors the correction used in 04_indicators.py, but keeps the
    attribution at the detailed land-use legend level used in this figure.
    """
    nc_ag10 = os.path.join(base_path, 'out_2010', 'xr_water_yield_ag_2010.nc')
    if not os.path.exists(nc_ag10):
        return {}

    try:
        ds = xr.open_dataset(nc_ag10)
        data = ds['data'].values
        lu_names = [str(v) for v in ds['lu'].values]
        n_lu = len(lu_names)
        lm_index = {'dry': 1, 'irr': 2}
        category_layers = {}
        for lu_idx, lu_name in enumerate(lu_names):
            if lu_name == 'ALL':
                continue
            for lm_name, lm_idx in lm_index.items():
                category = classify_land_use(lu_name, lm_name)
                if category is None:
                    continue
                layer_idx = lm_idx * n_lu + lu_idx
                category_layers.setdefault(category, []).append(data[:, layer_idx])
        ds.close()
    except Exception:
        return {}

    corrections = {}
    for entry in os.scandir(base_path):
        if not (entry.is_dir() and entry.name.startswith('out_')):
            continue
        try:
            year = int(entry.name[4:])
        except ValueError:
            continue
        if year == 2010:
            corrections[year] = {}
            continue

        nc_nag = os.path.join(entry.path, f'xr_water_yield_non_ag_{year}.nc')
        if not os.path.exists(nc_nag):
            corrections[year] = {}
            continue

        try:
            ds_nag = xr.open_dataset(nc_nag)
            mask = ds_nag['data'].values[:, 0] > 0
            ds_nag.close()
            corrections[year] = {
                category: float(sum(layer[mask].sum() for layer in layers))
                for category, layers in category_layers.items()
            }
        except Exception:
            corrections[year] = {}
    return corrections


def prepare_overview():
    """Overview row: two components — land use impact and climate change impact.
    Land use impact = total water yield change minus CCI (covers Ag LU + AM + Non-ag).
    Climate change impact = CCI (dvar_2010 × Δw_coeff, already a delta from 2010).
    """
    rows = []
    for scenario in input_files:
        water = load_water_australia(scenario)
        if water.empty:
            continue

        cci_by_year = _load_cci_australia(scenario)

        # Total water yield per year (absolute, GL) — all types combined
        total_by_year = (
            water.groupby('Year')[VALUE_COL].sum() / 1e3
        ).to_dict()
        total_baseline = total_by_year.get(2010, 0.0)

        all_years = sorted(set(total_by_year) | set(cci_by_year))
        for year in all_years:
            total_delta = total_by_year.get(year, 0.0) - total_baseline
            cci_val     = cci_by_year.get(year, 0.0)
            lu_impact   = total_delta - cci_val   # all land use change (Ag + AM + Non-ag) minus CCI
            rows.append({
                'year': int(year), 'scenario': scenario,
                'category': 'Land use impact', 'value': lu_impact,
            })
            rows.append({
                'year': int(year), 'scenario': scenario,
                'category': 'Climate change impact', 'value': cci_val,
            })

    return pd.DataFrame(rows)


def prepare_land_use():
    rows = []
    for scenario in input_files:
        water = load_water_australia(scenario)
        if water.empty:
            continue

        water_ag = water.query('Type == "Agricultural land-use"').copy()
        if not water_ag.empty:
            water_ag['category'] = water_ag.apply(
                lambda r: classify_land_use(r['Landuse'], r['Water Supply']), axis=1
            )
            water_ag = water_ag.dropna(subset=['category'])
            water_ag = water_ag.groupby(['Year', 'category'], as_index=False)[VALUE_COL].sum()

        water_non_ag = water.query('Type == "Non-Agricultural Land-use"').copy()
        if not water_non_ag.empty:
            water_non_ag = water_non_ag.groupby('Year', as_index=False)[VALUE_COL].sum()

        ag_values = (
            {} if water_ag.empty else
            water_ag.set_index(['Year', 'category'])[VALUE_COL].to_dict()
        )
        non_ag_values = (
            {} if water_non_ag.empty else
            water_non_ag.set_index('Year')[VALUE_COL].to_dict()
        )
        baseline = {
            category: ag_values.get((2010, category), 0.0)
            for category in LU_COLORS
            if category != 'Non-agricultural land-use'
        }
        baseline_non_ag = non_ag_values.get(2010, 0.0)

        try:
            corrections = _load_water_correction_by_category(get_path(scenario))
        except Exception:
            corrections = {}

        all_years = sorted(
            {year for year, _ in ag_values}
            | set(non_ag_values)
            | set(corrections)
        )
        for year in all_years:
            correction_by_category = corrections.get(int(year), {})
            total_correction = sum(correction_by_category.values())
            for category in LU_COLORS:
                if category == 'Non-agricultural land-use':
                    value = non_ag_values.get(year, 0.0) - baseline_non_ag - total_correction
                else:
                    value = (
                        ag_values.get((year, category), 0.0)
                        - baseline.get(category, 0.0)
                        + correction_by_category.get(category, 0.0)
                    )
                rows.append({
                    'year': int(year),
                    'scenario': scenario,
                    'category': category,
                    'value': value / 1e3,
                })

    return pd.DataFrame(rows)


def prepare_am():
    rows = []
    for scenario in input_files:
        water = load_water_australia(scenario)
        if water.empty:
            continue
        water = water.query('Type == "Agricultural Management"').copy()
        water = water.groupby(['Year', 'Agricultural Management'], as_index=False)[VALUE_COL].sum()
        for _, row in water.iterrows():
            rows.append({
                'year':     int(row['Year']),
                'scenario': scenario,
                'category': row['Agricultural Management'],
                'value':    float(row[VALUE_COL]) / 1e3,
            })
    return _subtract_2010_baseline(rows)


def main():
    overview_df = prepare_overview()
    land_use_df = prepare_land_use()
    am_df       = prepare_am()
    am_colors = {
        label: color
        for label, color in get_am_colors().items()
        if label != 'Other land-use'
    }
    export_long_tables(
        '16_water_long_tables.xlsx',
        overview=overview_df,
        land_use=land_use_df,
        agricultural_management=am_df,
    )
    save_three_row_figure(
        overview_df,
        land_use_df,
        am_df,
        OVERVIEW_COLORS,
        LU_COLORS,
        am_colors,
        'Difference in water yield relative to 2010 (GL yr⁻¹)',
        '16_water.svg',
        y_label_x=-0.006,
        top_legend_order=WATER_LAND_USE_LEGEND_ORDER,
    )


if __name__ == '__main__':
    main()
