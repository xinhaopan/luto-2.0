"""
S_Table_03.py -- Table S3 | Sensitivity of 2050 outcomes to the agricultural productivity assumption in the Regional Ag Capitals scenario.

Run_1_SCN_AgS1 (Regional Ag Capitals as published, productivity HIGH) against
Run_5_SCN_AgS1_VHP (the same scenario with productivity VERY_HIGH and the
matching very_high area-cost multiplier), on the six 2050 indicators of Fig. 4:

    net economic returns, agri-food production, GHG emissions,
    contribution-weighted biodiversity priority score, change in water yield relative
    to 2010, and land-use change extent.

How the numbers are produced
    Run_5 must go through exactly the pipeline that produced Run_1's published
    numbers, or the comparison is not a comparison.  So this script does not
    re-implement anything: it points the shared scenario list at the two runs
    and a private cache directory, then calls the same export functions
    Fig_01 / Fig_03 use and the same build_summary_table() that
    Fig_04 uses for Fig. 4.

    That also gives a built-in check.  Run_1 comes out of this path too, and it
    must equal Run_1 in the main Fig_04.xlsx summary to
    the last decimal.  If it does, the method is proven identical and Run_5's
    numbers can be trusted; if it does not, the script stops.

    The private cache (EXCEL_DIR/S_Table_03_cache/) keeps the four-scenario caches
    that every other figure reads untouched.

GENERATE_TABLES (tools/parameters.py)
    True   extract both runs from their Run_Archive.zip into the private cache,
           build the summary, write the table.  Slow: the water split loads
           each run's Data object.
    False  read the summary already in the private cache and write the table.

Outputs
    EXCEL_DIR/S_Table_03.xlsx
        S_Table_03     the two-column comparison with absolute and % differences
        summary_raw    every summary field for both runs, as build_summary_table
                       produced them
        verification   Run_1 via this path against Run_1 in the main summary
    EXCEL_DIR/S_Table_03_cache/   the private cache (01/03/04 workbooks for the two runs)
"""

import _path_setup  # noqa: F401

import importlib
import os

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Steer the shared parameters.  Order matters, in two different ways:
#
#   EXCEL_DIR is a string, so every module binds its own copy at import time
#   -> it must be redirected BEFORE the modules are imported.
#
#   input_files is one list object shared by reference, so mutating it in
#   place reaches every module whenever it is done -> but 04 indexes
#   input_files[0..3] at import to build its colour table, so the list must
#   still hold four entries while the modules load, and is cut down AFTER.
# ---------------------------------------------------------------------------
from tools import parameters as P

BASELINE = 'Run_1_SCN_AgS1'
SENSITIVITY = 'Run_5_SCN_AgS1_VHP'
MAIN_EXCEL_DIR = P.EXCEL_DIR                       # where the published caches live
S10_CACHE = os.path.join(P.EXCEL_DIR, 'S_Table_03_cache')

P.EXCEL_DIR = S10_CACHE
P.SCENARIO_LABELS[SENSITIVITY] = 'Regional Ag Capitals (very high productivity)'

from tools.parameters import GENERATE_TABLES        # noqa: E402
from tools.data_helper import get_zip_info          # noqa: E402
from tools.two_row_figure import export_long_tables, load_long_tables  # noqa: E402

mod01 = importlib.import_module('Fig_01')
mod03 = importlib.import_module('Fig_03')
mod04 = importlib.import_module('Fig_04')
import tools.water_landuse_split as water_split     # noqa: E402  (binds the redirected EXCEL_DIR)

# Now the modules are loaded: point the shared list at the two runs.  04's
# SCENARIO_CODES was derived from the old list at import, so re-derive it; its
# SCENARIO_COLORS is only used for plotting and can keep the stale entries.
P.input_files[:] = [BASELINE, SENSITIVITY]
mod04.SCENARIO_CODES = {s: s.split('_SCN_')[-1] for s in P.input_files}

import tools.two_row_figure as _trf                # noqa: E402
for _m in (mod01, mod03, mod04, _trf):
    assert _m.input_files is P.input_files, f'{_m.__name__} holds a different list'
# 01 has no EXCEL_DIR of its own -- its exports go through two_row_figure.
for _m in (mod03, mod04, _trf, water_split):
    assert 'S_Table_03_cache' in _m.EXCEL_DIR, f'{_m.__name__}.EXCEL_DIR not redirected'

# Relative paths in tools.parameters are anchored to this directory.
os.chdir(os.path.dirname(os.path.abspath(__file__)))

OUTPUT_NAME = 'S_Table_03.xlsx'

# (summary column, row label, unit) in the order Table S3 lists them.
INDICATORS = [
    ('ner_2050_baud',                 'Net economic returns',                     'billion AU$ yr⁻¹'),
    ('food_2050_mt',                  'Agri-food production',                     'Mt yr⁻¹'),
    ('ghg_2050_mtco2e',               'GHG emissions',              'Mt CO₂e yr⁻¹'),
    ('biodiversity_2050_mha',         'Contribution-weighted biodiversity priority score', 'Mha'),
    ('water_change_2050_gl',          'Difference in water yield relative to 2010',   'GL yr⁻¹'),
    ('land_use_change_2010_2050_mha', 'Land-use change extent, 2010–2050',       'Mha'),
]


def _check_archives():
    for scenario in (BASELINE, SENSITIVITY):
        if get_zip_info(scenario) is None:
            raise FileNotFoundError(
                f'No Run_Archive.zip for {scenario} under output/{P.TASK_ROOT}/. '
                f'For {SENSITIVITY} this must be the NCI archive (same input as Run_1), '
                'not the local-input control run.'
            )


def build_cache():
    """Run the 01 / 03 exports and the 04 summary for the two runs, into S10_CACHE."""
    _check_archives()
    os.makedirs(S10_CACHE, exist_ok=True)

    print('  Fig_01 exports ...')
    export_long_tables(
        'Fig_01.xlsx',
        land_use=mod01.prepare_land_use(),
        agricultural_management=mod01.prepare_am(),
    )
    print('  Fig_03 exports (the water split loads each Data object) ...')
    # Fig. 3's water row reads the receiving-category split (E_Fig_09 cache),
    # so build it for these two runs in the private cache first.
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


def load_cache():
    return mod04.load_summary_table()


def verify_against_published(summary):
    """Run_1 through this path must reproduce Run_1 in the published summary."""
    published_path = os.path.join(MAIN_EXCEL_DIR, mod04.WORKBOOK)
    if not os.path.exists(published_path):
        raise FileNotFoundError(
            f'{published_path} not found -- run Fig_04.py first, '
            'the check against the published numbers needs it.'
        )
    published = pd.read_excel(published_path, sheet_name='summary').set_index('scenario')
    ours = summary.set_index('scenario')
    cols = [c for c in ours.columns
            if c in published.columns and pd.api.types.is_numeric_dtype(ours[c])]
    rows = []
    for col in cols:
        a = float(ours.loc[BASELINE, col])
        b = float(published.loc[BASELINE, col])
        rows.append({'field': col, 'this_script': a, 'published_summary': b,
                     'abs_diff': a - b, 'match': bool(np.isclose(a, b, rtol=1e-9, atol=1e-9))})
    check = pd.DataFrame(rows)
    bad = check[~check['match']]
    if not bad.empty:
        print(bad.to_string(index=False))
        raise AssertionError(
            f'{len(bad)} field(s) of {BASELINE} differ from the published summary -- '
            'the two paths are not the same method; do not use these numbers.'
        )
    print(f'  verification: {len(check)} Run_1 fields reproduce the published summary exactly')
    return check


def make_table(summary):
    s = summary.set_index('scenario')
    rows = []
    for col, label, unit in INDICATORS:
        high = float(s.loc[BASELINE, col])
        vhp = float(s.loc[SENSITIVITY, col])
        diff = vhp - high
        pct = diff / abs(high) * 100.0 if not np.isclose(high, 0.0) else np.nan
        rows.append({
            'Indicator': label,
            'Unit': unit,
            'Regional Ag Capitals (high)': high,
            'Regional Ag Capitals (very high)': vhp,
            'Absolute difference': diff,
            'Difference (%)': pct,
        })
    return pd.DataFrame(rows)


def main():
    # GENERATE_TABLES: extract both runs into the private cache and build the
    # summary first; otherwise draw the table from the summary already there.
    if GENERATE_TABLES:
        summary = build_cache()
    else:
        summary = load_cache()

    check = verify_against_published(summary)
    table = make_table(summary)

    os.makedirs(MAIN_EXCEL_DIR, exist_ok=True)
    out = os.path.join(MAIN_EXCEL_DIR, OUTPUT_NAME)
    with pd.ExcelWriter(out, engine='openpyxl') as w:
        table.to_excel(w, sheet_name='S_Table_03', index=False)
        summary.to_excel(w, sheet_name='summary_raw', index=False)
        check.to_excel(w, sheet_name='verification', index=False)
    print(f'  wrote {out}')
    print()
    print(table.to_string(index=False, float_format=lambda v: f'{v:,.3f}'))


if __name__ == '__main__':
    main()
