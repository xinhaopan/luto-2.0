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
from tools.unit_text import needs_mixed_fonts
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


# ── Panel g: six-axis radar ───────────────────────────────────────────────────
# Clockwise from the top, the same order and the same six quantities as the
# radar in the framework diagram (Extended Data Fig. 1, third block), so a
# reader turning from the framework to the results sees the same object.  The
# abbreviations used there are written out in full here.
#
# (summary column, label, unit, +1 if larger is already better else -1)
RADAR_AXES = [
    ('ghg_2050_mtco2e',       'Net GHG emissions from land',           'Mt CO₂e yr⁻¹', -1),
    ('biodiversity_2050_mha', 'Biodiversity contribution-\nweighted score', 'Mha', +1),
    # Same wording as 03_indicators and 19_Water: "relative to 2010" matters,
    # otherwise -14,000 GL has nothing to be relative to.
    ('water_change_2050_gl',  'Difference in water yield\nrelative to 2010', 'GL yr⁻¹', +1),
    ('ner_2050_baud',         'Net economic returns',                  'billion AU$ yr⁻¹', +1),
    ('food_2050_mt',          'Agri-food production',                  'Mt yr⁻¹',          +1),
    ('land_use_change_2010_2050_mha', 'Land-use change extent',        'Mha',                        -1),
]
# Every axis carries an ordinary scale with round tick values, like the bar
# panels above: a nice step is chosen per axis and the rings sit at whole
# multiples of it.  Because the scale starts below the smallest value rather
# than at it, no scenario is pinned to the centre -- an earlier min-max version
# collapsed System Decline to a bare line.
RADAR_INTERVALS = 5
# Same size as the tick labels and value annotations of panels a-f.
RADAR_FONTSIZE = 15


def _nice_scale(low, high, intervals=RADAR_INTERVALS):
    """A round step and a round starting value covering [low, high].

    Same idea as a normal axis locator, but the number of intervals is fixed so
    that all six axes can share the same rings while each keeps its own units.
    The start is nudged down when the data would otherwise touch the centre.
    """
    if high <= low:
        high = low + 1.0
    magnitude = 10.0 ** np.floor(np.log10((high - low) / intervals))

    # Walk the nice steps upwards and take the first that actually fits, rather
    # than doubling once it fails.  Doubling overshoots badly: net emissions
    # span 175 Mt, and a doubled step gave them a 400-wide axis, so the four
    # scenarios were squeezed into the inner 44% of the radius.
    for decade in (magnitude, magnitude * 10.0, magnitude * 100.0):
        for multiple in (1.0, 2.0, 2.5, 5.0):
            step = multiple * decade
            start = np.floor(low / step) * step
            if low - start < 0.15 * step:   # keep the smallest value off centre
                start -= step
            if start + intervals * step >= high:
                return float(start), float(step)

    step = (high - low) / intervals
    return float(low - step), float(step)


def _radar_tick(value):
    """Tick text: the axes run from 152 Mt to -18,504 GL, so one format cannot
    serve both."""
    if value == 0:
        return '0'          # a step landing on zero can arrive as -0.0
    return f'{value:,.0f}' if abs(value) >= 10.0 else f'{value:,.1f}'


def _unit_font(unit):
    """Arial for plain units, DejaVu Sans when the unit needs a superscript."""
    return 'DejaVu Sans' if needs_mixed_fonts(unit) else 'Arial'



def _draw_radar(ax, summary):
    """Panel g. Every axis is oriented so that further out is better.

    Two of the six are 'smaller is better' (net emissions, and how much land has
    to change hands) and are negated first.  Change in water yield is already
    negative everywhere, and a smaller loss is a larger number, so it needs no
    flip.  Each axis then gets an ordinary scale of its own -- a round step and
    round tick values, chosen by _nice_scale -- so the rings can be read as
    numbers rather than as ranks, and no scenario is pinned to the centre.
    """
    n = len(RADAR_AXES)
    angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)

    ax.set_theta_offset(np.pi / 2.0)     # first axis at the top
    ax.set_theta_direction(-1)           # and then clockwise

    raw = np.array([summary[column].astype(float).to_numpy()
                    for column, _lab, _unit, _sign in RADAR_AXES])
    oriented = raw * np.array([[sign] for _c, _l, _u, sign in RADAR_AXES])

    # One ordinary scale per axis: a round step, a round starting value, and
    # RADAR_INTERVALS rings, so every axis can be read off its own ticks while
    # all six share the same rings.
    starts, steps = [], []
    for i in range(len(RADAR_AXES)):
        start, step = _nice_scale(float(oriented[i].min()), float(oriented[i].max()))
        starts.append(start)
        steps.append(step)
    starts = np.array(starts)[:, None]
    steps = np.array(steps)[:, None]
    scaled = (oriented - starts) / (RADAR_INTERVALS * steps)

    rings = [k / RADAR_INTERVALS for k in range(1, RADAR_INTERVALS + 1)]
    ax.set_ylim(0.0, 1.0)
    ax.set_yticks(rings)
    ax.set_yticklabels([])
    ax.set_xticks(angles)
    ax.set_xticklabels([])
    # The outer circle and the six spokes are the axes of this panel, so they
    # are drawn in the same black and the same weight as the axis lines of the
    # bar panels.  The rings are grid, and stay light, exactly as the vertical
    # grid does in (a)-(f).
    ax.grid(axis='x', color='#222222', linewidth=1.0)
    ax.grid(axis='y', color='#D9D9D9', linewidth=0.8)
    ax.spines['polar'].set_color('#222222')
    ax.spines['polar'].set_linewidth(1.2)
    ax.set_axisbelow(True)

    closed = np.concatenate([angles, angles[:1]])
    # Draw the roomiest polygon first so the tighter ones stay legible on top of
    # it instead of disappearing under its fill.
    order = np.argsort(-scaled.sum(axis=0))
    for rank, j in enumerate(order):
        scenario = summary['scenario'].iloc[j]
        values = np.concatenate([scaled[:, j], scaled[:1, j]])
        color = SCENARIO_COLORS[scenario]
        # Outline only -- four translucent fills stacked on top of each other
        # muddied the colours and hid the rings underneath.
        ax.plot(closed, values, color=color, linewidth=2.2,
                zorder=10 + rank, solid_joinstyle='round')
        # A dot on every vertex, so a scenario sitting low on most axes still
        # shows where it actually is.
        ax.plot(angles, scaled[:, j], linestyle='none', marker='o',
                markersize=5, color=color, zorder=10 + rank)

    # Labels are placed by offsetting from the end of each axis in display
    # space, not by pushing the radius out.  A radial offset moves a label
    # diagonally, so on the side axes the name and the numbers underneath it
    # drift into each other; offsetting in points keeps the numbers squarely
    # below the name whatever direction the axis points.
    for i, (_column, label, unit, _sign) in enumerate(RADAR_AXES):
        angle = angles[i]
        direction = (np.pi / 2.0) - angle          # theta_offset, then clockwise
        dx, dy = np.cos(direction), np.sin(direction)

        # Labels run along the tangent, i.e. square to the axis they belong to,
        # so each block reads as part of its own spoke.  Flip any that would
        # come out upside down.
        rotation = np.degrees(direction) - 90.0
        while rotation > 90.0:
            rotation -= 180.0
        while rotation <= -90.0:
            rotation += 180.0

        def _place(text, pad, fontsize, color, family=None):
            ax.annotate(
                text, xy=(angle, 1.0), xytext=(dx * pad, dy * pad),
                textcoords='offset points', ha='center', va='center',
                rotation=rotation, rotation_mode='anchor',
                fontsize=fontsize, color=color, zorder=5,
                annotation_clip=False,
                **({'fontfamily': family} if family else {}),
            )

        # The unit always sits on the inside, the name outside it, on every
        # axis, so the pair reads the same way all the way round.
        _place(f'({unit})', 42.0, RADAR_FONTSIZE, '#222222', _unit_font(unit))
        _place(label, 64.0, RADAR_FONTSIZE, '#222222')

        # Ordinary tick values for this axis, at the rings.
        sign = RADAR_AXES[i][3]
        for k, ring in enumerate(rings, start=1):
            value = (starts[i, 0] + k * steps[i, 0]) * sign   # undo the flip
            # Horizontal, not tangential: these are the numbers the reader is
            # meant to check, and a rotated number is harder to read.  The white
            # halo keeps them legible where a polygon crosses a ring.
            ax.annotate(
                _radar_tick(value), xy=(angle, ring),
                xytext=(-dy * 10.0, dx * 10.0), textcoords='offset points',
                ha='center', va='center', fontsize=RADAR_FONTSIZE,
                # Under the polygons, not over them: drawn on top, the halo
                # punched holes in the data lines where they crossed a ring.
                color='#222222', zorder=5, annotation_clip=False,
                bbox=dict(boxstyle='round,pad=0.12', facecolor='white',
                          edgecolor='none', alpha=0.85),
            )


def plot_figure(summary):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plt.rcParams.update({
        'font.family': 'Arial',
        'font.sans-serif': ['Arial'],
        'axes.linewidth': 0.8,
        'svg.fonttype': 'none',
    })

    # Three rows of two bar panels, then panel g alone across the fourth row.
    fig = plt.figure(figsize=(14.0, 21.0))
    gs = fig.add_gridspec(
        4, 2,
        height_ratios=[1.0, 1.0, 1.0, 2.20],
        left=0.055, right=0.985, top=0.975, bottom=0.090,
        wspace=0.20, hspace=0.38,
    )
    axes = np.empty((3, 2), dtype=object)
    for row in range(3):
        for col in range(2):
            axes[row, col] = fig.add_subplot(gs[row, col])

    # The radar sits in the middle of the full-width row, leaving the outer
    # thirds for its axis labels.  A thin strip is reserved above it so the
    # topmost axis label has somewhere to go and the panel title can sit above
    # that, centred over the panel exactly as (a)-(f) are.
    radar_cell = gs[3, :].subgridspec(1, 3, width_ratios=[1.0, 2.1, 1.0])
    radar_column = radar_cell[0, 1].subgridspec(2, 1, height_ratios=[0.22, 1.0])
    ax_radar = fig.add_subplot(radar_column[1, 0], polar=True)

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

    _draw_radar(ax_radar, summary)
    # Centred over the panel, same size and weight as the (a)-(f) titles, and
    # anchored to the top of the row so the axis labels pass underneath it.
    radar_cell_box = radar_cell[0, 1].get_position(fig)
    fig.text(radar_cell_box.x0 + radar_cell_box.width / 2.0,
             radar_cell_box.y1 - 0.004,
             '(g) Comparison of the four scenarios',
             fontsize=19, fontweight='bold', color='#222222',
             ha='center', va='top')

    # The shared x label belongs to the six bar panels, not to the radar, so it
    # sits just under the third row rather than at the foot of the figure.
    fig.text(
        0.5, gs[2, 0].get_position(fig).y0 - 0.013,
        'Change from 2010 baseline (%)',
        fontsize=17, ha='center', va='top',
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
