"""
10_Water.py
Change in water yield relative to 2010 by land use and agricultural management.
Positive = more water yield than 2010 baseline (beneficial).
Overview row decomposes Ag land-use change into land-use-change and climate-change
components using the CCI (Climate Change Impact) already computed by write.py.
"""
import _path_setup  # noqa: F401

import pandas as pd

from tools.two_row_figure import (
    RENAME_AM_NON_AG,
    export_long_tables,
    get_am_colors,
    input_files,
    load_report_source_csv,
    save_three_row_figure,
)

VALUE_COL = 'Water Net Yield (ML)'
SOURCE_CSV = 'water_yield_separate_watershed'
CCI_CSV = 'water_yield_limits_and_public_land'
CCI_COL = 'Climate Change Impact (ML)'

OVERVIEW_COLORS = {
    'Land use impact': '#6eabb1',
    'Climate change impact': '#d95f02',
}
WATER_LU_COLORS = {
    'Irrigated agriculture': '#2166ac',
    'Dryland agriculture': '#92c5de',
    'Non-agricultural land (plantations)': '#3A7F4A',
}


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
        for ws_raw, ws_label in [('Irrigated', 'Irrigated agriculture'),
                                  ('Dryland',   'Dryland agriculture')]:
            sub = water_ag[water_ag['Water Supply'] == ws_raw].copy()
            if sub.empty:
                continue
            sub = sub.groupby('Year', as_index=False)[VALUE_COL].sum()
            for _, row in sub.iterrows():
                rows.append({
                    'year':     int(row['Year']),
                    'scenario': scenario,
                    'category': ws_label,
                    'value':    float(row[VALUE_COL]) / 1e3,
                })

        water_non_ag = water.query('Type == "Non-Agricultural Land-use"').copy()
        if not water_non_ag.empty:
            water_non_ag = water_non_ag.groupby('Year', as_index=False)[VALUE_COL].sum()
            for _, row in water_non_ag.iterrows():
                rows.append({
                    'year':     int(row['Year']),
                    'scenario': scenario,
                    'category': 'Non-agricultural land (plantations)',
                    'value':    float(row[VALUE_COL]) / 1e3,
                })

    return _subtract_2010_baseline(rows)


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
        '10_water_long_tables.xlsx',
        overview=overview_df,
        land_use=land_use_df,
        agricultural_management=am_df,
    )
    save_three_row_figure(
        overview_df,
        land_use_df,
        am_df,
        OVERVIEW_COLORS,
        WATER_LU_COLORS,
        am_colors,
        'Change in water yield relative to 2010 (GL yr⁻¹)',
        '10_water.svg',
        y_label_x=0.005,
    )


if __name__ == '__main__':
    main()
