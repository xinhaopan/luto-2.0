"""Fig. 4 showing scenario changes from the 2010 baseline.

All six panels use percentages. Water uses absolute national yield totals before
calculating its percentage; land-use change is scaled by total 2010 study area.

Self-contained: the land-use-change extent is computed here directly from the
per-year area NetCDFs (it used to be read from 05_Trade_off.py's synthesis
workbook, which is now retired to archive/).
"""

import _path_setup  # noqa: F401

import io
import os
import zipfile
from pathlib import Path

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.lines as mlines
import matplotlib.ticker as mticker
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tools.parameters import (
    EXCEL_DIR,
    GENERATE_TABLES,
    OUTPUT_DIR,
    SCENARIO_LABELS,
    input_files,
)
from tools.data_helper import get_path, get_zip_info
from tools.two_row_figure import (
    filter_water_detail_rows,
    load_long_tables,
    load_report_source_csv,
    missing_table_error,
)


YEAR = 2050
BASELINE_YEAR = 2010
WORKBOOK = '04_trade_off_percent_threshold.xlsx'
INDICATOR_WORKBOOK = '03_indicators_long_tables.xlsx'

SCENARIO_COLORS = {
    input_files[0]: '#2D688F',
    input_files[1]: '#2F8F5B',
    input_files[2]: '#D9872C',
    input_files[3]: '#B84A4A',
}
SCENARIO_CODES = {
    scenario: scenario.split('_SCN_')[-1]
    for scenario in input_files
}

SOURCE_TABLES = {
    'net_economic_return': (INDICATOR_WORKBOOK, 'net_economic_return'),
    'ghg': (INDICATOR_WORKBOOK, 'ghg'),
    'biodiversity': (INDICATOR_WORKBOOK, 'biodiversity'),
    'food': (INDICATOR_WORKBOOK, 'food'),
    'water': (INDICATOR_WORKBOOK, 'water'),
}


def _year_totals(workbook, sheet):
    data = load_long_tables(workbook, sheet)[sheet]
    required = {'year', 'scenario', 'value'}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f'{workbook}/{sheet} is missing columns: {sorted(missing)}')
    totals = (
        data[data['year'].isin([BASELINE_YEAR, YEAR])]
        .groupby(['scenario', 'year'])['value']
        .sum()
        .unstack('year')
        .reindex(input_files)
    )
    if totals[[BASELINE_YEAR, YEAR]].isna().any().any():
        raise ValueError(
            f'{workbook}/{sheet} does not contain complete '
            f'{BASELINE_YEAR} and {YEAR} values for all scenarios'
        )
    return totals


def _percentage_change(totals, label):
    baseline = totals[BASELINE_YEAR].astype(float)
    if np.isclose(baseline, 0.0).any():
        bad = baseline.index[np.isclose(baseline, 0.0)].tolist()
        raise ValueError(f'{label} has a zero {BASELINE_YEAR} baseline: {bad}')
    return (totals[YEAR].astype(float) / baseline - 1.0) * 100.0


def _load_climate_water_2050():
    path = Path(EXCEL_DIR) / '03_water_climate_impact_split.csv'
    if not path.exists():
        raise missing_table_error(path)
    climate = pd.read_csv(path)
    climate = climate[climate['year'].eq(YEAR)].copy()
    climate['value'] = climate['dry_climate_GL'] + climate['irr_climate_GL']
    values = climate.set_index('scenario')['value'].reindex(input_files)
    if values.isna().any():
        raise ValueError(
            f'Missing {YEAR} climate-water values for: '
            f'{values[values.isna()].index.tolist()}'
        )
    return values.astype(float)


def _load_output_nc(scenario, year, filename):
    """Load an out_{year}/{filename} NetCDF from the run's zip archive or unpacked folder."""
    import xarray as xr

    info = get_zip_info(scenario)
    if info is not None:
        zip_path, prefix = info
        internal_path = f'{prefix}/out_{year}/{filename}'
        with zipfile.ZipFile(zip_path) as z:
            if internal_path not in z.namelist():
                return None
            with z.open(internal_path) as f:
                return xr.load_dataset(io.BytesIO(f.read()))

    try:
        base_path = Path(get_path(scenario))
    except (FileNotFoundError, StopIteration):
        return None
    nc_path = base_path / f'out_{year}' / filename
    if not nc_path.exists():
        return None
    return xr.open_dataset(nc_path)


def _read_landuse_layer_frame(scenario, year, filename, domain):
    """Per-cell × land-use area (ha) from one area NetCDF, as a DataFrame."""
    import cf_xarray as cfxr

    ds = _load_output_nc(scenario, year, filename)
    if ds is None:
        return pd.DataFrame(dtype=float)
    try:
        arr = cfxr.decode_compress_to_multi_index(ds, 'layer')['data']
        if 'layer' in arr.dims:
            arr = arr.unstack('layer')
        if arr.sizes.get('lu', 0) == 0:
            return pd.DataFrame(index=arr['cell'].to_numpy(), dtype=float)
        if 'lm' in arr.dims:
            arr = arr.sel(lm='ALL')
        if 'ALL' in set(arr['lu'].to_numpy()):
            arr = arr.drop_sel(lu='ALL')
        if arr.sizes.get('lu', 0) == 0:
            return pd.DataFrame(index=arr['cell'].to_numpy(), dtype=float)
        arr = arr.transpose('cell', 'lu').fillna(0.0)
        columns = [f'{domain}: {lu}' for lu in arr['lu'].to_numpy()]
        return pd.DataFrame(
            arr.to_numpy().astype(np.float64, copy=False),
            index=arr['cell'].to_numpy(),
            columns=columns,
            dtype=float,
        )
    finally:
        ds.close()


def _read_landuse_area_frame(scenario, year):
    frames = [
        _read_landuse_layer_frame(scenario, year, f'xr_area_agricultural_landuse_{year}.nc', 'ag'),
        _read_landuse_layer_frame(scenario, year, f'xr_area_non_agricultural_landuse_{year}.nc', 'non_ag'),
    ]
    frames = [f for f in frames if not f.empty]
    if not frames:
        return pd.DataFrame(dtype=float)
    return pd.concat(frames, axis=1)


def _load_land_change():
    """Land-use change extent (Mha): half the L1 distance between the 2010 and 2050
    per-cell × land-use area allocations, so each transferred hectare is counted once.

    Ported from the retired 05_Trade_off.py so this figure is self-contained.
    """
    values = {}
    for scenario in input_files:
        base_area = _read_landuse_area_frame(scenario, BASELINE_YEAR)
        target_area = _read_landuse_area_frame(scenario, YEAR)
        if base_area.empty or target_area.empty:
            values[scenario] = np.nan
            continue

        cell_index = base_area.index.union(target_area.index)
        columns = base_area.columns.union(target_area.columns)
        base_matrix = base_area.reindex(index=cell_index, columns=columns, fill_value=0.0).to_numpy(dtype=np.float64)
        target_matrix = target_area.reindex(index=cell_index, columns=columns, fill_value=0.0).to_numpy(dtype=np.float64)

        base_total = float(base_matrix.sum())
        target_total = float(target_matrix.sum())
        if not np.isclose(base_total, target_total, rtol=2e-5, atol=1e3):
            raise ValueError(
                f'{scenario} land-use area is not conserved between '
                f'{BASELINE_YEAR} ({base_total:,.1f} ha) and {YEAR} ({target_total:,.1f} ha)'
            )

        # Half the L1 distance counts each transferred hectare once.
        values[scenario] = float(0.5 * np.abs(target_matrix - base_matrix).sum()) / 1e6

    out = pd.Series(values, index=input_files, dtype=float)
    if out.isna().any():
        raise ValueError('Land-use change is incomplete for one or more scenarios')
    return out


def _load_water_2010_baseline():
    rows = []
    for scenario in input_files:
        water = filter_water_detail_rows(
            load_report_source_csv(scenario, 'water_yield_separate_watershed')
        )
        baseline = (
            water.loc[
                water['Year'].eq(BASELINE_YEAR),
                'Water Net Yield (ML)',
            ].sum()
            / 1e3
        )
        rows.append({'scenario': scenario, 'water_2010_gl': baseline})
    baseline = (
        pd.DataFrame(rows)
        .set_index('scenario')['water_2010_gl']
        .reindex(input_files)
        .astype(float)
    )
    if baseline.isna().any() or np.isclose(baseline, 0.0).any():
        raise ValueError('The 2010 absolute water-yield baseline is incomplete or zero')
    return baseline


def _load_land_area_2010():
    area = load_long_tables('01_area_long_tables.xlsx', 'land_use')['land_use']
    totals = (
        area[area['year'].eq(BASELINE_YEAR)]
        .groupby('scenario')['value']
        .sum()
        .reindex(input_files)
    )
    if totals.isna().any() or np.isclose(totals, 0.0).any():
        raise ValueError('The 2010 land-area baseline is incomplete or zero')
    return totals.astype(float)


def build_summary_table():
    totals = {
        key: _year_totals(*source)
        for key, source in SOURCE_TABLES.items()
    }
    climate_water = _load_climate_water_2050()
    land_change = _load_land_change()
    water_2010 = _load_water_2010_baseline()
    land_area_2010 = _load_land_area_2010()

    if not np.allclose(totals['water'][BASELINE_YEAR], 0.0, atol=1e-9):
        raise ValueError(
            f'{INDICATOR_WORKBOOK}/water must contain zero change in {BASELINE_YEAR}'
        )

    rows = []
    for scenario in input_files:
        water_delta = float(totals['water'].loc[scenario, YEAR])
        water_baseline = float(water_2010.loc[scenario])
        rows.append({
            'scenario': scenario,
            'scenario_code': SCENARIO_CODES[scenario],
            'scenario_label': SCENARIO_LABELS[scenario],
            'food_2010_mt': float(totals['food'].loc[scenario, BASELINE_YEAR]),
            'food_2050_mt': float(totals['food'].loc[scenario, YEAR]),
            'food_change_pct': float(_percentage_change(totals['food'], 'Food').loc[scenario]),
            'ner_2010_baud': float(totals['net_economic_return'].loc[scenario, BASELINE_YEAR]),
            'ner_2050_baud': float(totals['net_economic_return'].loc[scenario, YEAR]),
            'ner_change_pct': float(
                _percentage_change(totals['net_economic_return'], 'NER').loc[scenario]
            ),
            'biodiversity_2010_mha': float(totals['biodiversity'].loc[scenario, BASELINE_YEAR]),
            'biodiversity_2050_mha': float(totals['biodiversity'].loc[scenario, YEAR]),
            'biodiversity_change_pct': float(
                _percentage_change(totals['biodiversity'], 'Biodiversity').loc[scenario]
            ),
            'ghg_2010_mtco2e': float(totals['ghg'].loc[scenario, BASELINE_YEAR]),
            'ghg_2050_mtco2e': float(totals['ghg'].loc[scenario, YEAR]),
            'ghg_change_pct': float(
                _percentage_change(totals['ghg'], 'GHG').loc[scenario]
            ),
            'water_2010_gl': water_baseline,
            'water_2050_gl': water_baseline + water_delta,
            'water_change_2050_gl': water_delta,
            'water_change_pct': water_delta / water_baseline * 100.0,
            'climate_only_water_change_2050_gl': float(climate_water.loc[scenario]),
            'climate_only_water_change_pct': float(
                climate_water.loc[scenario]
                / water_baseline * 100.0
            ),
            'land_area_2010_mha': float(land_area_2010.loc[scenario]),
            'land_use_change_2010_2050_mha': float(land_change.loc[scenario]),
            'land_use_change_pct': float(
                land_change.loc[scenario] / land_area_2010.loc[scenario] * 100.0
            ),
        })

    summary = pd.DataFrame(rows)
    return summary


def definitions_table():
    return pd.DataFrame([
        {
            'panel': 'Food production',
            'metric': '100 * (production_2050 / production_2010 - 1)',
            'benchmark': '0% = each scenario own 2010 production',
            'interpretation': f'2010 and 2050 totals come directly from {INDICATOR_WORKBOOK}/food',
        },
        {
            'panel': 'Net economic return',
            'metric': '100 * (NER_2050 / NER_2010 - 1)',
            'benchmark': '0% = each scenario own 2010 NER',
            'interpretation': f'Totals come directly from {INDICATOR_WORKBOOK}/net_economic_return',
        },
        {
            'panel': 'Biodiversity',
            'metric': '100 * (contribution-weighted area_2050 / area_2010 - 1)',
            'benchmark': '0% = no net loss relative to 2010',
            'interpretation': f'Totals come directly from {INDICATOR_WORKBOOK}/biodiversity',
        },
        {
            'panel': 'Net GHG emissions',
            'metric': '100 * (GHG_2050 / GHG_2010 - 1)',
            'benchmark': '0% = 2010 net GHG emissions',
            'interpretation': f'Totals come directly from {INDICATOR_WORKBOOK}/ghg; below -100% means a net sink',
        },
        {
            'panel': 'Water yield',
            'metric': '100 * indicator_water_change_2050 / absolute_water_yield_2010',
            'benchmark': '0% = 2010 national water yield',
            'interpretation': f'The numerator comes directly from {INDICATOR_WORKBOOK}/water; raw 2010 yield is only the denominator',
        },
        {
            'panel': 'Land-use change',
            'metric': '100 * 0.5 * sum(abs(area_2050 - area_2010)) / total_area_2010',
            'benchmark': '0% = no change from the 2010 land allocation',
            'interpretation': 'Area is dvar * REAL_AREA by NetCDF cell and land use; 0.5 avoids double-counting transfers',
        },
    ])


def save_tables(summary):
    path = Path(EXCEL_DIR) / WORKBOOK
    path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(path, engine='openpyxl') as writer:
        summary.to_excel(writer, sheet_name='summary', index=False)
        definitions_table().to_excel(writer, sheet_name='definitions', index=False)
    print(f'Saved: {path}')
    return path


def load_summary_table():
    path = Path(EXCEL_DIR) / WORKBOOK
    if not path.exists():
        raise missing_table_error(path)
    try:
        summary = pd.read_excel(path, sheet_name='summary')
    except ValueError as exc:
        raise missing_table_error(path, 'summary') from exc
    return summary.set_index('scenario').reindex(input_files).reset_index()


def _set_panel_style(ax, title, panel_letter):
    ax.set_title(
        f'({panel_letter}) {title}',
        loc='center',
        fontsize=19,
        fontweight='bold',
        pad=10,
    )
    ax.xaxis.set_major_formatter(mticker.StrMethodFormatter('{x:,.0f}'))
    ax.grid(axis='x', color='#D9D9D9', linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.tick_params(axis='both', labelsize=15)


def _draw_bars(ax, summary, column, formatter, x_values_extra=(), zero_line=True):
    values = summary[column].astype(float).to_numpy()
    y = np.arange(len(summary))
    colors = [SCENARIO_COLORS[s] for s in summary['scenario']]

    all_values = np.concatenate([values, np.asarray(list(x_values_extra), dtype=float)])
    low = min(0.0, float(np.nanmin(all_values)))
    high = max(0.0, float(np.nanmax(all_values)))
    span = max(high - low, 1.0)
    negative_margin = 0.24 if low < 0 else 0.13
    ax.set_xlim(low - negative_margin * span, high + 0.20 * span)

    bars = ax.barh(y, values, color=colors, height=0.58, zorder=3)
    ax.set_yticks([])
    ax.invert_yaxis()
    if zero_line:
        ax.axvline(0, color='#333333', linewidth=1.1, zorder=2)

    label_pad = 0.018 * (ax.get_xlim()[1] - ax.get_xlim()[0])
    for bar, value in zip(bars, values):
        if value >= 0:
            x, ha = value + label_pad, 'left'
        else:
            x, ha = value - label_pad, 'right'
        ax.text(
            x,
            bar.get_y() + bar.get_height() / 2,
            formatter(value),
            va='center',
            ha=ha,
            fontsize=15,
            color='#222222',
        )
    return y


def plot_figure(summary):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plt.rcParams.update({
        'font.family': 'Arial',
        'font.sans-serif': ['Arial'],
        'axes.linewidth': 0.8,
        'svg.fonttype': 'none',
    })

    # 3 rows x 2 columns — two panels per row.
    fig, axes = plt.subplots(3, 2, figsize=(14.0, 14.5))
    fig.subplots_adjust(
        left=0.055,
        right=0.985,
        top=0.965,
        bottom=0.115,
        wspace=0.20,
        hspace=0.38,
    )

    # Match the indicator order used in 03_indicators.py.
    ax = axes[0, 0]
    _draw_bars(ax, summary, 'ner_change_pct', lambda value: f'{value:+,.1f}%')
    _set_panel_style(ax, 'Net economic returns', 'a')

    ax = axes[0, 1]
    _draw_bars(ax, summary, 'ghg_change_pct', lambda value: f'{value:+,.1f}%')
    _set_panel_style(ax, 'GHG emissions', 'b')

    ax = axes[1, 0]
    _draw_bars(ax, summary, 'biodiversity_change_pct', lambda value: f'{value:+,.1f}%')
    _set_panel_style(ax, 'Biodiversity', 'c')

    ax = axes[1, 1]
    _draw_bars(
        ax,
        summary,
        'food_change_pct',
        lambda value: f'{value:+,.1f}%',
    )
    _set_panel_style(ax, 'Agri-food production', 'd')

    ax = axes[2, 0]
    climate_pct = summary['climate_only_water_change_pct'].astype(float).to_numpy()
    climate_benchmark = float(np.mean(climate_pct))
    _draw_bars(
        ax,
        summary,
        'water_change_pct',
        lambda value: f'{value:+,.1f}%',
        x_values_extra=[climate_benchmark],
    )
    ax.axvline(
        climate_benchmark,
        color='#222222',
        linewidth=1.5,
        linestyle='--',
        zorder=4,
    )
    _set_panel_style(ax, 'Water yield', 'e')

    ax = axes[2, 1]
    _draw_bars(
        ax,
        summary,
        'land_use_change_pct',
        lambda value: f'{value:+,.1f}%',
    )
    _set_panel_style(ax, 'Land-use change', 'f')

    fig.supxlabel(
        'Change from 2010 baseline (%)',
        fontsize=17,
        x=0.5,
        y=0.062,
    )

    scenario_handles = [
        mlines.Line2D(
            [],
            [],
            marker='s',
            linestyle='none',
            markerfacecolor=SCENARIO_COLORS[scenario],
            markeredgecolor=SCENARIO_COLORS[scenario],
            markersize=12,
            label=SCENARIO_LABELS[scenario],
        )
        for scenario in input_files
    ]
    climate_handle = mlines.Line2D(
        [],
        [],
        color='#222222',
        linewidth=1.5,
        linestyle='--',
        label='Climate change impact',
    )
    fig.legend(
        handles=scenario_handles + [climate_handle],
        loc='lower center',
        bbox_to_anchor=(0.5, 0.012),
        ncol=5,
        frameon=False,
        fontsize=15,
        handlelength=1.2,
        columnspacing=2.0,
    )

    svg_path = Path(OUTPUT_DIR) / '04_trade_off_percent_threshold.svg'
    png_path = Path(OUTPUT_DIR) / '04_trade_off_percent_threshold.png'
    fig.savefig(svg_path, dpi=600, bbox_inches='tight', facecolor='white')
    fig.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'Saved: {svg_path}')
    print(f'Saved: {png_path}')
    return svg_path, png_path


def main():
    if GENERATE_TABLES:
        save_tables(build_summary_table())
    summary = load_summary_table()
    plot_figure(summary)


if __name__ == '__main__':
    main()
