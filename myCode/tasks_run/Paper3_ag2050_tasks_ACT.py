"""
Paper 3 revision -- Referee 1, comment 8: initial-transition sensitivity.

How does the 2010 -> present adjustment shape the 2050 results?  The four main
scenarios start from projected productivity, area-cost and labour-cost
multipliers from 2010 on.  These four runs replace the 2010-2023 part of all
three multipliers with observed values and leave everything from 2024 on
identical to the main run:

    Run_6_SCN_AgS1_ACT   vs  Run_1_SCN_AgS1
    Run_7_SCN_AgS2_ACT   vs  Run_2_SCN_AgS2
    Run_8_SCN_AgS3_ACT   vs  Run_3_SCN_AgS3
    Run_9_SCN_AgS4_ACT   vs  Run_4_SCN_AgS4

Each run moves exactly three settings, all for its own scenario only, each to
the *_actual sheet of the tier the main run used:

    AG2050_PRODUCTIVITY_MAP[s]  e.g. 'HIGH'                -> 'HIGH_ACTUAL'
        yieldincreases_ag_2050.xlsx sheet high_actual
        (luto/data.py accepts the *_ACTUAL names since this revision)
    AG2050_AC_MAP[s]            e.g. 'high'                -> 'high_actual'
        Area_cost.xlsx sheet high_actual
    AG2050_FLC_MAP[s]           e.g. 'FLC_multiplier_high' -> 'FLC_multiplier_high_actual'
        FLC_cost_multipliers.xlsx sheet FLC_multiplier_high_actual

Every *_actual sheet has the same columns as its plain sheet and is identical
to it for 2024-2050 (checked 2026-11-06 for all 12 sheets), so any difference
between a pair of runs comes only from the 2010-2023 inputs.

Everything else is the recorded column of the matching main run, exactly as in
the M1 script (Paper3_ag2050_tasks_VHP.py), including its fidelity audit.

WHERE THE BASELINE COMES FROM
    output/20260714_Paper3_NCI/_original_NCI_grid/grid_search_template.csv,
    the untouched record of the submitted grid.  It under-records two fields;
    see SCALAR_CORRECTIONS.

USAGE (from myCode/tasks_run/)
    python Paper3_ag2050_tasks_ACT.py --dry-run          # generate + audit only
    python Paper3_ag2050_tasks_ACT.py --platform NCI     # submit 4 PBS jobs on gadi
    python Paper3_ag2050_tasks_ACT.py --platform NCI --only Run_8_SCN_AgS3_ACT
"""

import argparse
import ast
import os
import sys

# All helpers below use paths relative to myCode/tasks_run/, so anchor there.
os.chdir(os.path.dirname(os.path.abspath(__file__)))

REPO_ROOT = os.path.abspath(os.path.join(os.getcwd(), '..', '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import pandas as pd
from tools.helpers import create_task_runs, get_settings_df, update_settings


# ---------------------------------------------------------------------------
# What we are building
# ---------------------------------------------------------------------------
TASK_NAME = '20260714_Paper3_NCI'

# new run name -> recorded main run it is compared against
RUNS = {
    'Run_6_SCN_AgS1_ACT': 'Run_1_SCN_AgS1',
    'Run_7_SCN_AgS2_ACT': 'Run_2_SCN_AgS2',
    'Run_8_SCN_AgS3_ACT': 'Run_3_SCN_AgS3',
    'Run_9_SCN_AgS4_ACT': 'Run_4_SCN_AgS4',
}

task_root_dir = f'../../output/{TASK_NAME}'
# The untouched record of the submitted grid (backed up by the M1 script
# before get_settings_df() could rewrite anything in the task folder).
template_csv  = f'{task_root_dir}/_original_NCI_grid/grid_search_template.csv'

# Maps whose entry for the run's own scenario moves to the *_actual sheet.
# PRODUCTIVITY values are upper case (data.py upper-cases, then lower-cases the
# sheet name); AC and FLC values are sheet names as written in the workbooks.
ACTUAL_SUFFIX = {
    'AG2050_PRODUCTIVITY_MAP': '_ACTUAL',
    'AG2050_AC_MAP':           '_actual',
    'AG2050_FLC_MAP':          '_actual',
}

# Corrections to the template CSV, which under-records what ran.  NOT part of
# the sensitivity.
#
#   SOLVE_TIME_LIMIT_SECONDS: the CSV says 14400 (4 h).  The settings.py inside
#   the Run_Archive.zip says 144000 for Run_1/Run_2 and 14400 for Run_3/Run_4.
#   All four get 144000 here: a time limit changes nothing unless a solve hits
#   it, and M1 showed a 4 h limit can kill a run (its 2017 crossover needed
#   ~6 h).  Run_3/Run_4 never hit 4 h, so their answers are unaffected.
#   MEM: the CSV says 200GB; PBS records show Run_1/Run_2 requested 400GB (used
#   302/192GB) and Run_3/Run_4 256GB (used ~207GB).  M1 was OOM-killed at
#   200GB while writing outputs.  400GB for all four.
SCALAR_CORRECTIONS = {
    'SOLVE_TIME_LIMIT_SECONDS': '144000',
    'MEM': '400GB',
}

# update_settings() rewrites these for the machine, so they are expected to
# differ from the record and are excluded from the fidelity audit.
EXPECTED_DIFFS = {
    'INPUT_DIR', 'RAW_DATA', 'NO_GO_VECTORS', 'THREADS', 'JOB_NAME', 'NCPUS', 'MEM',
}


def _to_actual(recorded_value: str, scenario: str, suffix: str) -> tuple[str, str, str]:
    """Return (new map literal, old sheet, new sheet) with one scenario moved to *_actual."""
    mapping = ast.literal_eval(recorded_value)
    if scenario not in mapping:
        raise KeyError(f'{scenario!r} not found in recorded map {mapping!r}')
    old = mapping[scenario]
    if old.lower().endswith('_actual'):
        raise ValueError(f'{scenario} already points at an actual sheet: {old!r}')
    mapping[scenario] = old + suffix
    return str(mapping), old, mapping[scenario]


def build_settings(run_name: str, baseline_col: str,
                   defaults: dict, record: pd.DataFrame) -> dict:
    """Current defaults, overlaid with the recorded run, overlaid with our change."""
    recorded = record[baseline_col].map(str).to_dict()
    added   = sorted(set(defaults) - set(recorded))   # new since the run
    dropped = sorted(set(recorded) - set(defaults))   # retired since the run

    settings_dict = {**defaults, **recorded}
    scenario = settings_dict['AG2050_SCENARIO']
    if settings_dict.get('AG2050_MODE') != 'True' or not scenario:
        raise ValueError(f'{baseline_col} is not an AG2050 run')
    if settings_dict.get('AG2050_USE_FLC_SCENARIO') != 'True':
        raise ValueError(f'{baseline_col} does not read FLC_cost_multipliers.xlsx; '
                         'the FLC *_actual sheet would be ignored')
    if not run_name.split('_SCN_')[1].startswith(scenario):
        raise ValueError(f'{run_name} does not match scenario {scenario} of {baseline_col}')

    print(f'\n{run_name}  <-  {baseline_col}  ({scenario})')
    for key, suffix in ACTUAL_SUFFIX.items():
        settings_dict[key], old, new = _to_actual(settings_dict[key], scenario, suffix)
        print(f'  {key}[{scenario}]: {old!r} -> {new!r}')

    for key, value in SCALAR_CORRECTIONS.items():
        print(f'  {key}: {settings_dict.get(key)!r} -> {value!r}  (correction, not sensitivity)')
        settings_dict[key] = value

    settings_dict = update_settings(settings_dict, run_name)
    settings_dict['JOB_NAME'] = run_name

    _audit(run_name, baseline_col, settings_dict, recorded, added, dropped)
    return settings_dict


def _audit(run_name, baseline_col, settings_dict, recorded, added, dropped) -> None:
    """Print every way this run differs from its recorded main run; stop on surprises."""
    print('-' * 78)
    print(f'AUDIT  {run_name}  vs  {baseline_col}')
    intended = set(ACTUAL_SUFFIX)
    unintended = [
        (k, recorded[k], v) for k, v in settings_dict.items()
        if k not in EXPECTED_DIFFS and k not in intended and k not in SCALAR_CORRECTIONS
        and k in recorded and str(v) != str(recorded[k])
    ]
    for key in sorted(intended):
        print(f'  {key}\n      was: {recorded[key]}\n      now: {settings_dict[key]}')
    if added:
        print(f'  settings added since the run ({len(added)}), filled from today\'s default:')
        for key in added:
            print(f'    {key:<44} = {str(settings_dict[key])[:90]}')
    if dropped:
        print(f'  settings retired since the run ({len(dropped)}), carried from the record:')
        for key in dropped:
            print(f'    {key:<44} = {str(recorded[key])[:90]}')
    if unintended:
        for key, was, now in unintended:
            print(f'  *** {key}\n      recorded: {str(was)[:90]}\n      ours    : {str(now)[:90]}')
        raise RuntimeError(f'{len(unintended)} unintended difference(s) in {run_name}; '
                           'this would not be a clean comparison.')
    print('  unintended differences: none')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dry-run', action='store_true',
                        help='generate the template and audit, but do not create or launch runs')
    parser.add_argument('--platform', default='NCI',
                        choices=['aquila', 'NCI', 'HPC', 'Denethor'],
                        help='NCI/HPC submit one batch job per run; the helper checks the hostname')
    parser.add_argument('--only', nargs='+', choices=list(RUNS),
                        help='build/submit only these runs (e.g. to resubmit one)')
    args = parser.parse_args()

    if not os.path.exists(template_csv):
        raise FileNotFoundError(f'{template_csv} not found; it is the record of the main runs.')

    # Today's defaults (also (re)writes non_str_val.txt, which create_task_runs reads).
    defaults = get_settings_df(task_root_dir).set_index('Name')['Default_run'].to_dict()
    record = pd.read_csv(template_csv).set_index('Name')

    runs = {r: RUNS[r] for r in (args.only or RUNS)}
    for run_name in runs:
        if os.path.exists(f'{task_root_dir}/{run_name}'):
            raise FileExistsError(f'{task_root_dir}/{run_name} already exists; '
                                  'resume it from its checkpoint instead of re-creating it.')

    columns = {r: build_settings(r, b, defaults, record) for r, b in runs.items()}
    template = pd.DataFrame(columns)
    template.index.name = 'Name'
    template = template.reset_index()
    out_csv = f'{task_root_dir}/grid_search_template_ACT.csv'
    if not args.only:
        template.to_csv(out_csv, index=False)
        print(f'\nWrote {out_csv}')

    if args.dry_run:
        print('\n--dry-run: stopping before launch.')
        return

    print(f'\nSubmitting {len(runs)} run(s) on platform: {args.platform}')
    create_task_runs(
        task_root_dir,
        template,
        platform=args.platform,
        n_workers=1,
        use_parallel=False,
    )


if __name__ == '__main__':
    main()
