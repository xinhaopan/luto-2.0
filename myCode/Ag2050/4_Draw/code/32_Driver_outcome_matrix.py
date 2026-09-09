"""
32_Driver_outcome_matrix.py -- Extended Data figure

A 9 x 4 small-multiple scatter matrix: the nine time-varying scenario drivers
(rows) against the four 2050 outcomes (columns), one point per scenario.

Answers Referee 2 ("put the scenarios on a two-dimensional plot"; "the model is
deterministic, the results just restate the scenario settings") and Referee 3
("what did we learn from the simulation that we did not already know?") by
letting the reader see directly which outcomes track their drivers and which do
not -- without compressing nine drivers into two axes.

Sources (relative paths, both under EXCEL_DIR)
    12_input_data_long_tables.xlsx, sheet 'series'  -- drivers  (y)
    04_trade_off_percent_threshold.xlsx, sheet 'summary' -- outcomes (x)

Driver rows a-e carry a scenario code directly.  Rows f-i are the feedlot
adjustment ratios, which are keyed by pathway level rather than by scenario, so
LEVEL_MAP below maps each scenario onto its level.  Climate Survival and System
Decline share the 'Medium' feedlot pathway, so their points coincide in rows
f-i.  That is correct and is stated in the caption -- do not jitter it apart.

Each point is the joint result of all drivers and all active constraints, so a
point's position within a panel is NOT the effect of that row's driver alone on
that column's outcome.  Hence no fitted or trend line is drawn anywhere.

Output
    OUTPUT_DIR/32_Driver_outcome_matrix.svg  (svg.fonttype='none')
"""

import _path_setup  # noqa: F401

import os

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

# SCENARIO_LABELS is deliberately not imported: this module keeps its own,
# keyed by scenario code rather than by run id.
from tools.parameters import EXCEL_DIR, GENERATE_TABLES, OUTPUT_DIR, input_files
from tools.text_layout import balanced_wrap
from tools.two_row_figure import export_long_tables, load_long_tables
from tools.unit_text import mixed_xlabel

# Relative paths in tools.parameters are anchored to this directory.
os.chdir(os.path.dirname(os.path.abspath(__file__)))

DRIVER_WORKBOOK = '12_input_data_long_tables.xlsx'
DRIVER_SHEET = 'series'
OUTCOME_WORKBOOK = '04_trade_off_percent_threshold.xlsx'
OUTCOME_SHEET = 'summary'
OUTPUT_NAME = '32_Driver_outcome_matrix.svg'
WORKBOOK = '32_driver_outcome_long_tables.xlsx'   # this figure's cached table

YEAR = 2050
MM = 1.0 / 25.4
FIG_W_MM = 183.0        # Nature double-column width
FIG_H_MM = 240.0

# Taken from 04_Trade_off_percent_threshold.py (Fig. 4). Fixed for the whole
# paper -- do not re-pick.
SCENARIO_COLORS = {
    'AgS1': '#2D688F',
    'AgS2': '#2F8F5B',
    'AgS3': '#D9872C',
    'AgS4': '#B84A4A',
}
SCENARIO_LABELS = {
    'AgS1': 'Regional Ag Capitals',
    'AgS2': 'Landscape Stewardship',
    'AgS3': 'Climate Survival',
    'AgS4': 'System Decline',
}
SCENARIO_ORDER = ['AgS1', 'AgS2', 'AgS3', 'AgS4']
# The cached long tables key on the run id, which is what the shared table
# helpers expect in a 'scenario' column.
RUN_OF = {code: run for code, run in zip(SCENARIO_ORDER, input_files)}

# A second, non-colour cue for scenario identity.  The palette is fixed by the
# rest of the paper and cannot change, but this figure identifies scenarios by
# legend alone, so colour would otherwise be the only cue -- which fails both
# the red/green rule and the greyscale rule.  Shape carries identity as well.
# Areas are tuned per shape so all four read as the same visual weight.
SCENARIO_MARKERS = {
    'AgS1': ('o', 1.00),
    'AgS2': ('s', 0.86),
    'AgS3': ('^', 1.18),
    'AgS4': ('D', 0.86),
}
MARKER_SIZE = 11.0

# Rows a-e are keyed by scenario; rows f-i are keyed by pathway level.
LEVEL_MAP = {'AgS1': 'Very High', 'AgS2': 'High', 'AgS3': 'Medium', 'AgS4': 'Medium'}

# (letter, panel value in the 'panel' column, row title)
DRIVER_ROWS = [
    ('a', 'Beef',          'Beef productivity multiplier'),
    ('b', 'Crop',          'Crop productivity multiplier'),
    ('c', 'Dairy',         'Dairy productivity multiplier'),
    ('d', 'Sheep',         'Sheep productivity multiplier'),
    ('e', 'Labour cost',   'Labour cost multiplier'),
    ('f', 'Revenue ratio', 'Feedlot revenue adjustment ratio'),
    ('g', 'Cost ratio',    'Feedlot cost adjustment ratio'),
    ('h', 'GHG ratio',     'Feedlot GHG adjustment ratio'),
    ('i', 'Water ratio',   'Feedlot water adjustment ratio'),
]
SCENARIO_KEYED_PANELS = {'Beef', 'Crop', 'Dairy', 'Sheep', 'Labour cost'}

# Longest line a rotated row title may have, in characters.  A row is only
# about 20 mm tall, and a rotated line runs along that height.
ROW_TITLE_MAX_CHARS = 18


# (column in the summary sheet, column title with units)
OUTCOME_COLUMNS = [
    ('ghg_2050_mtco2e',      'Net GHG emissions from land\n(Mt CO₂e yr⁻¹)'),
    ('biodiversity_2050_mha', 'Biodiversity contribution-\nweighted score (Mha)'),
    ('food_2050_mt',          'Agri-food production\n(Mt yr⁻¹)'),
    ('water_change_2050_gl',  'Change in water yield\n(GL yr⁻¹)'),
]

# Published values, checked on load so a regenerated workbook cannot silently
# change the figure.  Drivers keyed by scenario; feedlot rows by level.
EXPECTED_DRIVERS = {
    'Beef':          {'AgS1': 1.007, 'AgS2': 1.138, 'AgS3': 0.822, 'AgS4': 0.636},
    'Crop':          {'AgS1': 1.721, 'AgS2': 1.867, 'AgS3': 1.514, 'AgS4': 1.307},
    'Dairy':         {'AgS1': 1.220, 'AgS2': 1.331, 'AgS3': 1.063, 'AgS4': 0.906},
    'Sheep':         {'AgS1': 1.078, 'AgS2': 1.273, 'AgS3': 0.801, 'AgS4': 0.524},
    'Labour cost':   {'AgS1': 2.485, 'AgS2': 2.959, 'AgS3': 2.087, 'AgS4': 1.752},
    'Revenue ratio': {'AgS1': 1.100, 'AgS2': 1.117, 'AgS3': 1.066, 'AgS4': 1.066},
    'Cost ratio':    {'AgS1': 1.071, 'AgS2': 1.083, 'AgS3': 1.047, 'AgS4': 1.047},
    'GHG ratio':     {'AgS1': 0.970, 'AgS2': 0.965, 'AgS3': 0.980, 'AgS4': 0.980},
    'Water ratio':   {'AgS1': 1.070, 'AgS2': 1.081, 'AgS3': 1.046, 'AgS4': 1.046},
}
EXPECTED_OUTCOMES = {
    'ghg_2050_mtco2e':       {'AgS1': 67.12, 'AgS2': -73.72, 'AgS3': 101.47, 'AgS4': 100.51},
    'biodiversity_2050_mha': {'AgS1': 89.39, 'AgS2': 97.87, 'AgS3': 88.47, 'AgS4': 88.84},
    'food_2050_mt':          {'AgS1': 240.36, 'AgS2': 213.57, 'AgS3': 195.84, 'AgS4': 152.23},
    'water_change_2050_gl':  {'AgS1': -16369, 'AgS2': -15254, 'AgS3': -18504, 'AgS4': -17698},
}


def load_drivers() -> dict:
    """Return {panel: {scenario_code: 2050 value}}."""
    path = os.path.join(EXCEL_DIR, DRIVER_WORKBOOK)
    if not os.path.exists(path):
        raise FileNotFoundError(f'{path} not found. Run 12_input_data.py first.')
    series = pd.read_excel(path, sheet_name=DRIVER_SHEET)

    for column in ('panel', 'series_type', 'scenario', 'level', 'year', 'value'):
        if column not in series.columns:
            raise ValueError(f'{DRIVER_WORKBOOK}[{DRIVER_SHEET}] lacks column {column!r}')

    future = series[(series['series_type'] == 'future') & (series['year'] == YEAR)]

    drivers = {}
    for _, panel, _title in DRIVER_ROWS:
        rows = future[future['panel'] == panel]
        if rows.empty:
            raise ValueError(f'No future/{YEAR} rows for driver panel {panel!r}')
        if panel in SCENARIO_KEYED_PANELS:
            lookup = rows.set_index('scenario')['value']
            values = {s: float(lookup[s]) for s in SCENARIO_ORDER}
        else:
            # Feedlot ratios carry a pathway level, not a scenario.
            lookup = rows.set_index('level')['value']
            values = {s: float(lookup[LEVEL_MAP[s]]) for s in SCENARIO_ORDER}
        drivers[panel] = values
    return drivers


def load_outcomes() -> dict:
    """Return {summary column: {scenario_code: 2050 value}}."""
    path = os.path.join(EXCEL_DIR, OUTCOME_WORKBOOK)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f'{path} not found. Run 04_Trade_off_percent_threshold.py first.'
        )
    summary = pd.read_excel(path, sheet_name=OUTCOME_SHEET)
    lookup = summary.set_index('scenario_code')

    outcomes = {}
    for column, _title in OUTCOME_COLUMNS:
        if column not in summary.columns:
            raise ValueError(f'{OUTCOME_WORKBOOK}[{OUTCOME_SHEET}] lacks column {column!r}')
        outcomes[column] = {s: float(lookup.loc[s, column]) for s in SCENARIO_ORDER}
    return outcomes


def verify(drivers: dict, outcomes: dict) -> None:
    problems = []
    for panel, expected in EXPECTED_DRIVERS.items():
        for scenario, want in expected.items():
            got = drivers[panel][scenario]
            if abs(got - want) > 0.0015:
                problems.append(f'  driver {panel:<14} {scenario} expected {want:.3f}, got {got:.3f}')
    for column, expected in EXPECTED_OUTCOMES.items():
        for scenario, want in expected.items():
            got = outcomes[column][scenario]
            tol = 1.0 if abs(want) > 1000 else 0.01
            if abs(got - want) > tol:
                problems.append(f'  outcome {column:<22} {scenario} expected {want}, got {got}')
    if problems:
        raise ValueError('Workbook values do not match the published numbers:\n'
                         + '\n'.join(problems))
    print(f'  drivers and outcomes match the published {YEAR} values')

    coincide = [p for p in EXPECTED_DRIVERS
                if drivers[p]['AgS3'] == drivers[p]['AgS4']]
    print('  rows where Climate Survival and System Decline coincide (expected f-i): '
          + ', '.join(coincide))


def _tidy(drivers, outcomes):
    """Long tables for the cache: one row per (panel, scenario) value."""
    driver_rows = [{'panel': panel, 'scenario': RUN_OF[code], 'value': drivers[panel][code]}
                   for _l, panel, _t in DRIVER_ROWS for code in SCENARIO_ORDER]
    outcome_rows = [{'outcome': column, 'scenario': RUN_OF[code], 'value': outcomes[column][code]}
                    for column, _t in OUTCOME_COLUMNS for code in SCENARIO_ORDER]
    return pd.DataFrame(driver_rows), pd.DataFrame(outcome_rows)


def _from_tables(driver_df, outcome_df):
    """Rebuild the {panel: {code: value}} dicts from the cached long tables."""
    code_of = {run: code for code, run in RUN_OF.items()}
    drivers = {panel: {} for _l, panel, _t in DRIVER_ROWS}
    for row in driver_df.itertuples():
        drivers[row.panel][code_of[row.scenario]] = float(row.value)
    outcomes = {column: {} for column, _t in OUTCOME_COLUMNS}
    for row in outcome_df.itertuples():
        outcomes[row.outcome][code_of[row.scenario]] = float(row.value)
    return drivers, outcomes


def main() -> None:
    # GENERATE_TABLES: pull the values out of the processed workbooks and cache
    # them here; otherwise draw straight from this figure's own cached table.
    if GENERATE_TABLES:
        drivers = load_drivers()
        outcomes = load_outcomes()
        verify(drivers, outcomes)
        driver_df, outcome_df = _tidy(drivers, outcomes)
        export_long_tables(WORKBOOK, drivers=driver_df, outcomes=outcome_df)
    tables = load_long_tables(WORKBOOK, 'drivers', 'outcomes')
    drivers, outcomes = _from_tables(tables['drivers'], tables['outcomes'])
    verify(drivers, outcomes)

    # Straight from the Nature figure spec (static/fragments/backend/python.md).
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans', 'sans-serif'],
        'svg.fonttype': 'none',
        'pdf.fonttype': 42,
        'font.size': 7,
        'axes.spines.right': False,
        'axes.spines.top': False,
        'axes.linewidth': 0.8,
        'legend.frameon': False,
        'xtick.major.width': 0.8,
        'ytick.major.width': 0.8,
    })

    n_rows, n_cols = len(DRIVER_ROWS), len(OUTCOME_COLUMNS)
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(FIG_W_MM * MM, FIG_H_MM * MM),
        sharex='col',   # one x scale down each column
        sharey='row',   # one y scale across each row
    )

    for r, (_letter, panel, row_title) in enumerate(DRIVER_ROWS):
        driver_values = drivers[panel]
        for c, (column, _col_title) in enumerate(OUTCOME_COLUMNS):
            ax = axes[r, c]
            outcome_values = outcomes[column]
            # NO fitted or trend line: a point is the joint result of every
            # driver and constraint, not of this row's driver alone.
            for scenario in SCENARIO_ORDER:
                marker, area_scale = SCENARIO_MARKERS[scenario]
                ax.scatter(
                    outcome_values[scenario], driver_values[scenario],
                    s=MARKER_SIZE * area_scale, c=SCENARIO_COLORS[scenario],
                    marker=marker,
                    edgecolors='white', linewidths=0.4, zorder=3, clip_on=False,
                )
            ax.margins(x=0.22, y=0.30)
            ax.tick_params(axis='both', labelsize=6, width=0.8, length=2.0, pad=1.2)
            ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=3))
            ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=3))
            # Spec: no top or right spine.
            for side in ('top', 'right'):
                ax.spines[side].set_visible(False)
            for side in ('left', 'bottom'):
                ax.spines[side].set_color('#4D4D4D')

            if c == 0:
                # Rotated 90 deg, like a normal y-axis label.  Wrapped by
                # balanced_wrap so a line rarely holds a single word, and
                # at 6.5 pt so the longest line still fits along one row.
                ax.set_ylabel(
                    balanced_wrap(row_title, ROW_TITLE_MAX_CHARS),
                    fontsize=6.5, labelpad=3, rotation=90,
                    ha='center', va='bottom',
                )

    for c, (_column, col_title) in enumerate(OUTCOME_COLUMNS):
        mixed_xlabel(axes[-1, c], col_title, fontsize=7, pad=14.0)

    fig.subplots_adjust(left=0.150, right=0.985, bottom=0.082, top=0.975,
                        wspace=0.34, hspace=0.20)

    handles = [
        mlines.Line2D([], [], color='none', marker=SCENARIO_MARKERS[s][0],
                      linestyle='none',
                      markerfacecolor=SCENARIO_COLORS[s],
                      markeredgecolor='white', markeredgewidth=0.4,
                      markersize=4 * SCENARIO_MARKERS[s][1] ** 0.5,
                      label=SCENARIO_LABELS[s])
        for s in SCENARIO_ORDER
    ]
    fig.legend(
        handles=handles, loc='lower center', ncol=len(SCENARIO_ORDER),
        frameon=False, fontsize=7, handletextpad=0.4, columnspacing=1.8,
        bbox_to_anchor=(0.5, 0.004),
    )

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, OUTPUT_NAME)
    fig.savefig(out_path, format='svg', facecolor='white')
    plt.close(fig)
    print(f'  wrote {out_path}')


if __name__ == '__main__':
    main()
