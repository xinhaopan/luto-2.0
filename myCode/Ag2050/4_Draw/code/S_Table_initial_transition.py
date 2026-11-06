"""
S_Table_initial_transition.py -- Sensitivity of 2050 outcomes to the initial
transition (Referee 1, comment 8).  Rename to S_Table_NN.py once the table has a
number in the manuscript.

Each main run is compared with the same scenario re-run with observed
2010-2023 productivity, area-cost and labour-cost multipliers (the *_actual
sheets; identical to the main inputs from 2024 on):

    Run_1_SCN_AgS1  vs  Run_6_SCN_AgS1_ACT
    Run_2_SCN_AgS2  vs  Run_7_SCN_AgS2_ACT
    Run_3_SCN_AgS3  vs  Run_8_SCN_AgS3_ACT
    Run_4_SCN_AgS4  vs  Run_9_SCN_AgS4_ACT

on the six 2050 indicators of Fig. 4.  The comparison is between the 2050
values themselves, not between changes relative to 2010, because the two runs
of a pair start from different 2010 multipliers.  (Water yield and land-use
change extent are defined against each run's own 2010 in Fig. 4 and are kept
that way, so the rows mean the same as in the main table.)

How the numbers are produced -- same method as S_Table_03.py (the M1 table):
the shared scenario list is pointed at the eight runs and a private cache,
then the Fig_01 / Fig_03 exports and Fig_04.build_summary_table() run
unchanged.  Run_1-Run_4 come out of this path too and must reproduce the
published Fig_04.xlsx summary exactly, or the script stops.

GENERATE_TABLES (tools/parameters.py)
    True   extract all eight runs into the private cache and build the summary.
           Slow: the water split loads each run's Data object.
    False  read the summary already in the private cache.

Outputs
    EXCEL_DIR/S_Table_initial_transition.xlsx
        table          scenario x indicator: main, observed 2010-2023, abs and % difference
        summary_raw    every summary field for all eight runs
        verification   Run_1-4 via this path against the published summary
    EXCEL_DIR/S_Table_initial_transition_cache/
"""

import _path_setup  # noqa: F401

import importlib
import os

import numpy as np
import pandas as pd

# Order matters (see S_Table_03.py): EXCEL_DIR is redirected BEFORE the modules
# import it; input_files is shrunk/extended in place AFTER, because Fig_04
# indexes input_files[0..3] at import.
from tools import parameters as P

PAIRS = {
    'Run_1_SCN_AgS1': 'Run_6_SCN_AgS1_ACT',
    'Run_2_SCN_AgS2': 'Run_7_SCN_AgS2_ACT',
    'Run_3_SCN_AgS3': 'Run_8_SCN_AgS3_ACT',
    'Run_4_SCN_AgS4': 'Run_9_SCN_AgS4_ACT',
}
NAME = 'S_Table_initial_transition'
MAIN_EXCEL_DIR = P.EXCEL_DIR
CACHE = os.path.join(P.EXCEL_DIR, f'{NAME}_cache')

P.EXCEL_DIR = CACHE
for main_run, act_run in PAIRS.items():
    P.SCENARIO_LABELS[act_run] = P.SCENARIO_LABELS[main_run] + ' (observed 2010–2023)'

from tools.parameters import GENERATE_TABLES        # noqa: E402
from tools.data_helper import get_zip_info          # noqa: E402
from tools.two_row_figure import export_long_tables  # noqa: E402

mod01 = importlib.import_module('Fig_01')
mod03 = importlib.import_module('Fig_03')
mod04 = importlib.import_module('Fig_04')
import tools.water_landuse_split as water_split     # noqa: E402

P.input_files[:] = list(PAIRS) + list(PAIRS.values())
mod04.SCENARIO_CODES = {s: s.split('_SCN_')[-1] for s in P.input_files}

import tools.two_row_figure as _trf                # noqa: E402
for _m in (mod01, mod03, mod04, _trf):
    assert _m.input_files is P.input_files, f'{_m.__name__} holds a different list'
for _m in (mod03, mod04, _trf, water_split):
    assert f'{NAME}_cache' in _m.EXCEL_DIR, f'{_m.__name__}.EXCEL_DIR not redirected'

os.chdir(os.path.dirname(os.path.abspath(__file__)))

OUTPUT_NAME = f'{NAME}.xlsx'

# Same indicators, labels and units as S_Table_03.
INDICATORS = [
    ('ner_2050_baud',                 'Net economic returns',                              'billion AU$ yr⁻¹'),
    ('food_2050_mt',                  'Agri-food production',                              'Mt yr⁻¹'),
    ('ghg_2050_mtco2e',               'GHG emissions',                                     'Mt CO₂e yr⁻¹'),
    ('biodiversity_2050_mha',         'Contribution-weighted biodiversity priority score', 'Mha'),
    ('water_change_2050_gl',          'Difference in water yield relative to 2010',        'GL yr⁻¹'),
    ('land_use_change_2010_2050_mha', 'Land-use change extent, 2010–2050',                 'Mha'),
]


def _check_archives():
    missing = [s for s in P.input_files if get_zip_info(s) is None]
    if missing:
        raise FileNotFoundError(
            f'No Run_Archive.zip under output/{P.TASK_ROOT}/ for: {missing}. '
            'The *_ACT runs must be the NCI archives (same input as Run_1-4).'
        )


def build_cache():
    _check_archives()
    os.makedirs(CACHE, exist_ok=True)
    print('  Fig_01 exports ...')
    export_long_tables(
        'Fig_01.xlsx',
        land_use=mod01.prepare_land_use(),
        agricultural_management=mod01.prepare_am(),
    )
    print('  Fig_03 exports (the water split loads each Data object) ...')
    water_split.compute()
    mod03._load_climate_water_impact(force_regenerate=True)
    export_long_tables(
        'Fig_03.xlsx',
        net_economic_return=mod03.prepare_ner(),
        ghg=mod03.prepare_ghg(),
        biodiversity=mod03.prepare_bio(),
        food=mod03.prepare_food(),
        water=mod03.prepare_water(),
    )
    print('  04 summary ...')
    summary = mod04.build_summary_table()
    mod04.save_tables(summary)
    return summary


def verify_against_published(summary):
    """Run_1-4 through this path must reproduce the published summary."""
    published_path = os.path.join(MAIN_EXCEL_DIR, mod04.WORKBOOK)
    if not os.path.exists(published_path):
        raise FileNotFoundError(f'{published_path} not found -- run Fig_04.py first.')
    published = pd.read_excel(published_path, sheet_name='summary').set_index('scenario')
    ours = summary.set_index('scenario')
    cols = [c for c in ours.columns
            if c in published.columns and pd.api.types.is_numeric_dtype(ours[c])]
    rows = []
    for scenario in PAIRS:
        for col in cols:
            a = float(ours.loc[scenario, col])
            b = float(published.loc[scenario, col])
            rows.append({'scenario': scenario, 'field': col, 'this_script': a,
                         'published_summary': b, 'abs_diff': a - b,
                         'match': bool(np.isclose(a, b, rtol=1e-9, atol=1e-9))})
    check = pd.DataFrame(rows)
    bad = check[~check['match']]
    if not bad.empty:
        print(bad.to_string(index=False))
        raise AssertionError(f'{len(bad)} main-run field(s) differ from the published '
                             'summary -- not the same method; do not use these numbers.')
    print(f'  verification: {len(check)} main-run fields reproduce the published summary exactly')
    return check


def make_table(summary):
    s = summary.set_index('scenario')
    rows = []
    for main_run, act_run in PAIRS.items():
        for col, label, unit in INDICATORS:
            main = float(s.loc[main_run, col])
            act = float(s.loc[act_run, col])
            diff = act - main
            rows.append({
                'Scenario': P.SCENARIO_LABELS[main_run].split('\n')[0],
                'Indicator': label,
                'Unit': unit,
                'Main run (projected 2010–2023)': main,
                'Observed 2010–2023': act,
                'Absolute difference': diff,
                'Difference (%)': diff / abs(main) * 100.0 if not np.isclose(main, 0.0) else np.nan,
            })
    return pd.DataFrame(rows)


def main():
    summary = build_cache() if GENERATE_TABLES else mod04.load_summary_table()
    check = verify_against_published(summary)
    table = make_table(summary)

    out = os.path.join(MAIN_EXCEL_DIR, OUTPUT_NAME)
    with pd.ExcelWriter(out, engine='openpyxl') as w:
        table.to_excel(w, sheet_name='table', index=False)
        summary.to_excel(w, sheet_name='summary_raw', index=False)
        check.to_excel(w, sheet_name='verification', index=False)
    print(f'  wrote {out}\n')
    print(table.to_string(index=False, float_format=lambda v: f'{v:,.3f}'))


if __name__ == '__main__':
    main()
