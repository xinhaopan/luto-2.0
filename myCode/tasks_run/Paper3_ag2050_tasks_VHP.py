"""
Paper 3 revision, M1 -- Referee 3's productivity sensitivity run.

Run_5_SCN_AgS1_VHP = AgS1 "Regional Ag Capitals" with productivity raised from
HIGH to VERY_HIGH, i.e. the "intensified food and fibre production" variant
Referee 3 asked about.  It is a SINGLE-FACTOR sensitivity against
Run_1_SCN_AgS1 of the 20260714_Paper3_NCI production run, so exactly two
settings move and nothing else:

    AG2050_PRODUCTIVITY_MAP['AgS1']   'HIGH' -> 'VERY_HIGH'
        picks the VERY_HIGH sheet of yieldincreases_ag_2050.xlsx
    AG2050_AC_MAP['AgS1']             'high' -> 'very_high'
        picks the very_high sheet of Area_cost.xlsx

The area-cost multiplier is moved with productivity on purpose.  The four main
scenarios tie the two together (AgS1 high/high, AgS2 very_high/very_high), so
holding area cost at 'high' while yields jump to VERY_HIGH would give a run
that is internally inconsistent -- higher output at unchanged establishment and
maintenance cost.  This choice is stated in the appendix alongside Table S10.

Deliberately NOT changed (all verified against the recorded Run_1 column):
    AG2050_GHG_MAP['AgS1']  = 'maintain_historical'   GHG <= 2010 level
    AG2050_BIO_MAP['AgS1']  = 'low'                   restore 5% of top-20% priority areas
    AG2050_FLC_MAP['AgS1']  = 'FLC_multiplier_high'   labour cost unchanged
    NON_AG_LAND_USES        = all on except BECCS
    AG_MANAGEMENTS          = all on except Solar PV / Onshore Wind
    RESFACTOR = 3, SIM_YEARS = 2010..2050, feedlot ratios, demand, water

WHERE THE BASELINE COMES FROM
    Settings are taken from the recorded Run_1_SCN_AgS1 column of
        output/20260714_Paper3_NCI/grid_search_template.csv
    which is the authoritative record of what actually ran -- NOT from
    Paper3_ag2050_tasks_NCI.py, which has drifted since submission (that script
    now says TIME=720:00:00 and SOLVE_TIME_LIMIT_SECONDS=2592000, while the run
    actually used 48:00:00 and 14400).

    Any setting that exists in today's luto/settings.py but was absent from that
    run's template is filled from today's default and reported in the audit
    below, so a silently-added default can never slip into the comparison
    unnoticed.

USAGE (from myCode/tasks_run/)
    python Paper3_ag2050_tasks_VHP.py --dry-run          # generate + audit only
    python Paper3_ag2050_tasks_VHP.py                    # run here (aquila)
    python Paper3_ag2050_tasks_VHP.py --platform NCI     # submit a PBS job on gadi

On NCI the recorded MEM/NCPUS/TIME/QUEUE of the reference run are what the batch
job asks for, so the sensitivity run gets the same resources as the AgS1 run it
is compared against.

Reference: myCode/tasks_run/Paper3_ag2050_tasks_NCI.py
"""

import argparse
import ast
import os
import shutil
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
TASK_NAME     = '20260714_Paper3_NCI'
BASELINE_COL  = 'Run_1_SCN_AgS1'
RUN_NAME      = 'Run_5_SCN_AgS1_VHP'

task_root_dir = f'../../output/{TASK_NAME}'
template_csv  = f'{task_root_dir}/grid_search_template.csv'
backup_dir    = f'{task_root_dir}/_original_NCI_grid'

# The single-factor change: scenario key -> new map value.
PRODUCTIVITY_OVERRIDE = ('AG2050_PRODUCTIVITY_MAP', 'AgS1', 'VERY_HIGH')
AREA_COST_OVERRIDE    = ('AG2050_AC_MAP',           'AgS1', 'very_high')

# update_settings() rewrites these for the local machine, so they are expected
# to differ from the NCI record and are excluded from the fidelity audit.
EXPECTED_DIFFS = {
    'INPUT_DIR', 'RAW_DATA', 'NO_GO_VECTORS', 'THREADS', 'JOB_NAME', 'NCPUS', 'MEM',
}


def _override_scenario_map(recorded_value: str, scenario: str, new_value: str) -> str:
    """Return the recorded map literal with one scenario's value replaced."""
    mapping = ast.literal_eval(recorded_value)
    if scenario not in mapping:
        raise KeyError(f'{scenario!r} not found in recorded map {mapping!r}')
    mapping[scenario] = new_value
    return str(mapping)


def build_settings() -> dict:
    """Current defaults, overlaid with the recorded run, overlaid with our change."""
    if not os.path.exists(template_csv):
        raise FileNotFoundError(
            f'{template_csv} not found. It is the record of the 20260714_Paper3_NCI '
            'run and is required as the baseline for this sensitivity.'
        )

    # Preserve the original grid record before get_settings_df() rewrites
    # non_str_val.txt in the same folder.
    os.makedirs(backup_dir, exist_ok=True)
    for fname in ('grid_search_template.csv', 'grid_search_parameters.csv',
                  'grid_search_parameters_unique.csv', 'non_str_val.txt'):
        src, dst = f'{task_root_dir}/{fname}', f'{backup_dir}/{fname}'
        if os.path.exists(src) and not os.path.exists(dst):
            shutil.copy2(src, dst)
            print(f'  backed up {fname} -> _original_NCI_grid/')

    # 1. Today's defaults, for completeness (also (re)writes non_str_val.txt).
    defaults = get_settings_df(task_root_dir).set_index('Name')['Default_run'].to_dict()

    # 2. The recorded run, for fidelity.
    recorded = pd.read_csv(template_csv).set_index('Name')[BASELINE_COL].map(str).to_dict()

    added   = sorted(set(defaults) - set(recorded))   # new since the run
    dropped = sorted(set(recorded) - set(defaults))   # retired since the run

    settings_dict = {**defaults, **recorded}

    # 3. The single-factor change.
    for key, scenario, new_value in (PRODUCTIVITY_OVERRIDE, AREA_COST_OVERRIDE):
        before = settings_dict[key]
        settings_dict[key] = _override_scenario_map(settings_dict[key], scenario, new_value)
        print(f'  {key}[{scenario}]: {ast.literal_eval(before)[scenario]!r} -> {new_value!r}')

    settings_dict = update_settings(settings_dict, RUN_NAME)
    settings_dict['JOB_NAME'] = RUN_NAME

    _audit(settings_dict, recorded, added, dropped)
    return settings_dict


def _audit(settings_dict: dict, recorded: dict, added: list, dropped: list) -> None:
    """Print every way this run differs from the recorded AgS1 run."""
    print('\n' + '=' * 78)
    print(f'AUDIT  {RUN_NAME}  vs  {BASELINE_COL}  (20260714_Paper3_NCI)')
    print('=' * 78)

    intended = {PRODUCTIVITY_OVERRIDE[0], AREA_COST_OVERRIDE[0]}
    unintended = []
    for key, value in settings_dict.items():
        if key in EXPECTED_DIFFS or key in intended or key not in recorded:
            continue
        if str(value) != str(recorded[key]):
            unintended.append((key, recorded[key], value))

    print('\nIntended changes (the sensitivity itself):')
    for key in sorted(intended):
        print(f'  {key}')
        print(f'      was: {recorded[key]}')
        print(f'      now: {settings_dict[key]}')

    if added:
        print(f'\nSettings added to luto/settings.py since the run ({len(added)}) '
              '-- filled from today\'s default, review these:')
        for key in added:
            print(f'  {key:<44} = {str(settings_dict[key])[:90]}')
    else:
        print('\nSettings added since the run: none.')

    if dropped:
        print(f'\nSettings retired from luto/settings.py since the run ({len(dropped)}) '
              '-- carried through from the record:')
        for key in dropped:
            print(f'  {key:<44} = {str(recorded[key])[:90]}')
    else:
        print('\nSettings retired since the run: none.')

    if unintended:
        print(f'\n*** {len(unintended)} UNINTENDED DIFFERENCE(S) -- these break the '
              'single-factor claim: ***')
        for key, was, now in unintended:
            print(f'  {key}')
            print(f'      recorded: {str(was)[:90]}')
            print(f'      ours    : {str(now)[:90]}')
    else:
        print('\nUnintended differences: none. This is a clean single-factor sensitivity.')

    print('\nKey settings actually going into the run:')
    for key in ('AG2050_MODE', 'AG2050_SCENARIO', 'AG2050_PRODUCTIVITY_MAP',
                'AG2050_AC_MAP', 'AG2050_FLC_MAP', 'AG2050_GHG_MAP', 'AG2050_BIO_MAP',
                'RESFACTOR', 'OBJECTIVE', 'SOLVE_WEIGHT_BETA', 'FEASIBILITY_TOLERANCE',
                'OPTIMALITY_TOLERANCE', 'SOLVE_TIME_LIMIT_SECONDS',
                'REGIONAL_ADOPTION_CONSTRAINTS', 'REGIONAL_ADOPTION_NON_AG_CAP',
                'GBF2_PRIORITY_DEGRADED_AREAS_PERCENTAGE_CUT', 'NON_AG_LAND_USES',
                'AG_MANAGEMENTS', 'THREADS', 'KEEP_OUTPUTS'):
        if key in settings_dict:
            print(f'  {key:<44} = {str(settings_dict[key])[:110]}')
    print('=' * 78 + '\n')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dry-run', action='store_true',
                        help='generate the run folder and audit, but do not launch')
    parser.add_argument('--threads', type=int, default=None,
                        help='override THREADS (default: keep the recorded 50)')
    parser.add_argument('--platform', default='aquila',
                        choices=['aquila', 'NCI', 'HPC', 'Denethor'],
                        help='where to run. aquila runs it in-process; NCI and '
                             'HPC submit a batch job. The helper checks the '
                             'hostname, so this has to match the machine.')
    args = parser.parse_args()

    print(f'Building {RUN_NAME} in {task_root_dir}\n')
    settings_dict = build_settings()

    if args.threads is not None:
        settings_dict['THREADS'] = str(args.threads)
        settings_dict['NCPUS'] = str(args.threads)
        print(f'THREADS/NCPUS overridden to {args.threads}\n')

    template = pd.Series(settings_dict, name=RUN_NAME).to_frame().reset_index(names='Name')
    template.index = template['Name'].values
    template.to_csv(f'{task_root_dir}/grid_search_template_VHP.csv', index=False)
    print(f'Wrote {task_root_dir}/grid_search_template_VHP.csv')

    if args.dry_run:
        print('\n--dry-run: stopping before launch. Re-run without --dry-run to start.')
        return

    # aquila runs `python python_script.py` in the run folder and blocks until
    # the simulation finishes; NCI and HPC hand it to the batch scheduler and
    # return as soon as the job is queued.
    print(f'
Submitting on platform: {args.platform}')
    create_task_runs(
        task_root_dir,
        template,
        platform=args.platform,
        n_workers=1,
        use_parallel=False,
    )


if __name__ == '__main__':
    main()
