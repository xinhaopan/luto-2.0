"""
Paper2 revision runs on NCI (One Earth, ONE-EARTH-D-26-01194R1).

The runs are listed one by one instead of a full grid, so nothing is solved twice by accident.
Every run records the shadow prices of the GHG and GBF2 targets (luto/solvers/tools.py).

Stages
    smoke   one coarse run (RESFACTOR 13, 5-year steps) that checks the uploaded inputs, the solve,
            the shadow price files and the output writing before any full run is paid for
    key     the four scenarios whose shadow prices go in the main text: Reference, highest GHG target,
            highest biodiversity target, and both together. They also measure walltime, memory and
            output size for the stages that follow
    main    the remaining scenarios of the submitted manuscript, with the same run names as
            20260414_Paper2_Results_NCI so myCode/carbonprice/code/tools/config.py keeps working
    sens    sensitivity runs on the four key scenarios: other SSPs, CO2 fertilisation, dynamic price
    amort   optional twins of the key scenarios with upfront costs annualised inside the optimiser
    all     key + main + sens
    resume  show the state of every run that has a folder, and start the interrupted ones again.
            A run continues from its latest checkpoint (luto/simulation.py), or only writes its outputs
            when the solve had finished

Usage, from any folder
    python Paper2_revision_tasks_NCI.py                  list every run and the cost, create nothing
    python Paper2_revision_tasks_NCI.py key              list one stage only
    python Paper2_revision_tasks_NCI.py key --submit     create the run folders and submit the jobs
    python Paper2_revision_tasks_NCI.py resume           show which runs finished, are queued or were interrupted
    python Paper2_revision_tasks_NCI.py resume --submit  start the interrupted runs again

A run whose folder already exists is skipped, so a stage can be submitted again safely.
"""

import os
import re
import sys
import glob
import getpass
import subprocess

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))          # the helpers use paths relative to this folder

import pandas as pd
from tools.helpers import get_settings_df, update_settings, create_task_runs, submit_task


# ----------------------------------------------------------------------------------------------
# What to run
# ----------------------------------------------------------------------------------------------
TASK_NAME = '20261112_Paper2_Revision_NCI'

# True  = Reference, NZlow and NZhigh are solved once for each priority cut, as in the submitted runs.
#         The cut does not enter their optimisation, it only sets the mask used when the GBF2 scores
#         are written, so 12 of these 15 runs repeat a solve. The post-processing needs no change.
# False = each of them is solved once (cut 50). This saves 12 full runs, and the scores under the
#         other cuts must then be derived in post-processing from the all-cell GBF2 output.
REFERENCE_PER_CUT = True

# Restoration target without a GHG target. Cut 50 is the "highest biodiversity target" scenario.
# The five cuts together give the reverse order of stacking for every joint combination
# (restoration first, Reference to NP, then net zero on top, NP to NZ,NP).
NP_ONLY_CUTS = [50, 40, 30, 20, 10]

# Sensitivity variants, each applied to the four key scenarios. RCP and SCENARIO are derived from SSP
# in luto/settings.py, but a run folder gets a flat list of values, so all three are set together here.
SENSITIVITY = {
    'SSP_126': {'SSP': '126', 'RCP': 'rcp2p6', 'SCENARIO': 'SSP1'},
    'SSP_585': {'SSP': '585', 'RCP': 'rcp8p5', 'SCENARIO': 'SSP5'},
    # The inputs hold no rcp7p0 productivity file. SSP370 can only run with the rcp6p0 file:
    # 'SSP_370': {'SSP': '370', 'RCP': 'rcp6p0', 'SCENARIO': 'SSP3'},
    'CO2_on':  {'CO2_FERT': 'on'},
    'DP_on':   {'DYNAMIC_PRICE': True},
}

# Optional twins with establishment and transition costs annualised in the optimiser.
# 60 years and 7% match the amortisation used in myCode/carbonprice/code/0_Preprocess.py.
AMORTISED = {'AMORTISE_UPFRONT_COSTS': True, 'AMORTISATION_PERIOD': 60, 'DISCOUNT_RATE': 0.07}


# ----------------------------------------------------------------------------------------------
# Resources. normalsr charges 2 SU per core hour and counts memory as one core per 5 GB,
# so 18 cores carry up to 90 GB at no extra charge. Only the walltime actually used is charged.
# ----------------------------------------------------------------------------------------------
RESOURCES = {'QUEUE': 'normalsr', 'NCPUS': '18', 'MEM': '90GB', 'TIME': '48:00:00', 'WRITE_THREADS': '2'}
SMOKE_RESOURCES = {'QUEUE': 'normalsr', 'NCPUS': '8', 'MEM': '40GB', 'TIME': '02:00:00', 'WRITE_THREADS': '2'}
EXPECTED_HOURS = 26     # assumption for the cost estimate only: the April runs needed the limit raised from 24 h to 30 h


# ----------------------------------------------------------------------------------------------
# Model settings shared by every run, copied from Paper2_tasks_NCI.py (20260414_Paper2_Results_NCI)
# ----------------------------------------------------------------------------------------------
COMMON = {
    'TASK_NAME': TASK_NAME,
    'KEEP_OUTPUTS': True,                   # False keeps only Run_Archive.zip and the report zip of each run
    'NUMERIC_FOCUS': 3,

    # Checkpoints and solver fallbacks, ported from the upstream model (the other values are in luto/settings.py)
    'CHECKPOINT_INTERVAL': 1,               # checkpoint after every solved year, 0 switches checkpoints off
    'RETRY_ON_SUBOPTIMAL': False,           # False accepts a SUBOPTIMAL year at once, as the April runs did

    'SOLVE_WEIGHT_ALPHA': 1,
    'SOLVE_WEIGHT_BETA': 0.9,
    'OBJECTIVE': 'maxprofit',
    'WRITE_OUTPUT_GEOTIFFS': True,
    'RESFACTOR': 5,
    'SIM_YEARS': list(range(2010, 2051, 1)),

    'GHG_CONSTRAINT_TYPE': 'hard',
    'CARBON_PRICES_FIELD': 'CONSTANT',
    'CARBON_EFFECTS_WINDOW': 60,
    'GHG_TARGETS_DICT': {
        'off': None,
        'low': '1.8C (67%) excl. avoided emis SCOPE1',
        'medium': '1.5C (50%) excl. avoided emis SCOPE1',
        'high': '1.5C (67%) excl. avoided emis SCOPE1',
    },

    'GBF2_CONSTRAINT_TYPE': 'hard',
    'GBF2_TARGETS_DICT': {
        'low': {2030: 0, 2050: 0, 2100: 0},
        'medium': {2030: 0.30, 2050: 0.30, 2100: 0.30},
        'high': {2030: 0.30, 2050: 0.50, 2100: 0.50},
    },
    'BIO_QUALITY_LAYER': 'Suitability',
    'BIODIVERSITY_TARGET_GBF_3': 'off',
    'BIODIVERSITY_TARGET_GBF_4_SNES': 'off',
    'BIODIVERSITY_TARGET_GBF_4_ECNES': 'off',
    'BIODIVERSITY_TARGET_GBF_8': 'off',

    'WATER_STRESS': 0.6,
    'WATER_LIMITS': 'on',
    'WATER_CONSTRAINT_TYPE': 'hard',
    'INCLUDE_WATER_LICENSE_COSTS': 1,
    'WATER_REGION_DEF': 'Drainage Division',
    'WATER_CLIMATE_CHANGE_IMPACT': 'on',

    'DYNAMIC_PRICE': False,
    'DEMAND_CONSTRAINT_TYPE': 'soft',

    'REGIONAL_ADOPTION_CONSTRAINTS': 'off',
    'NON_AG_LAND_USES': {
        'Environmental Plantings': True,
        'Riparian Plantings': True,
        'Sheep Agroforestry': True,
        'Beef Agroforestry': True,
        'Carbon Plantings (Block)': True,
        'Sheep Carbon Plantings (Belt)': True,
        'Beef Carbon Plantings (Belt)': True,
        'BECCS': False,
        'Destocked - natural land': True,
    },
}

# Defaults of the three settings the sensitivity variants change (luto/settings.py lines 59 to 63, 79, 120)
BASELINE = {'SSP': '245', 'RCP': 'rcp4p5', 'SCENARIO': 'SSP2', 'CO2_FERT': 'off',
            'AMORTISE_UPFRONT_COSTS': False}

KEY_SCENARIOS = [('off', 'off', 50), ('high', 'off', 50), ('off', 'high', 50), ('high', 'high', 50)]


# ----------------------------------------------------------------------------------------------
# The run list
# ----------------------------------------------------------------------------------------------
def scenario_settings(ghg, bio, cut):
    return {'GHG_EMISSIONS_LIMITS': ghg, 'BIODIVERSITY_TARGET_GBF_2': bio,
            'GBF2_PRIORITY_DEGRADED_AREAS_PERCENTAGE_CUT': cut}


def build_runs():
    """Every run as a dict(name, stage, settings). The first 25 names equal those of the April task."""
    runs = []
    cuts = [50, 40, 30, 20, 10]

    # Run_01 to Run_25: the order of the April grid (GHG high, low, off; BIO high, off; cut 50 to 10),
    # with GHG off and BIO high left out as it was then.
    idx = 0
    for ghg in ['high', 'low', 'off']:
        for bio in ['high', 'off']:
            for cut in cuts:
                if ghg == 'off' and bio == 'high':
                    continue
                idx += 1
                if bio == 'off' and cut != 50 and not REFERENCE_PER_CUT:
                    continue
                stage = 'key' if (ghg, bio, cut) in KEY_SCENARIOS else 'main'
                runs.append(dict(name=f'Run_{idx:02d}_GHG_{ghg}_BIO_{bio}_CUT_{cut}', stage=stage,
                                 settings=scenario_settings(ghg, bio, cut)))

    # Run_26 to Run_30: restoration target only, the numbers are reserved for cut 50 to 10
    for i, cut in enumerate(cuts):
        if cut in NP_ONLY_CUTS:
            stage = 'key' if ('off', 'high', cut) in KEY_SCENARIOS else 'main'
            runs.append(dict(name=f'Run_{26 + i:02d}_GHG_off_BIO_high_CUT_{cut}', stage=stage,
                             settings=scenario_settings('off', 'high', cut)))

    # Run_31 onwards: sensitivity variants on the four key scenarios
    idx = 30
    for tag, changes in SENSITIVITY.items():
        for ghg, bio, cut in KEY_SCENARIOS:
            idx += 1
            runs.append(dict(name=f'Run_{idx:02d}_GHG_{ghg}_BIO_{bio}_CUT_{cut}_{tag}', stage='sens',
                             settings={**scenario_settings(ghg, bio, cut), **changes}))

    # Run_61 to Run_64: optional twins with annualised upfront costs in the optimiser
    for i, (ghg, bio, cut) in enumerate(KEY_SCENARIOS):
        runs.append(dict(name=f'Run_{61 + i:02d}_GHG_{ghg}_BIO_{bio}_CUT_{cut}_AMORT', stage='amort',
                         settings={**scenario_settings(ghg, bio, cut), **AMORTISED}))

    for run in runs:
        run['settings'] = {**RESOURCES, **run['settings']}
    return runs


def smoke_run():
    settings_ = {**scenario_settings('high', 'high', 50), **SMOKE_RESOURCES,
                 'RESFACTOR': 13, 'SIM_YEARS': list(range(2010, 2051, 5))}
    return [dict(name='Smoke_GHG_high_BIO_high_CUT_50_RES13', stage='smoke', settings=settings_)]


# ----------------------------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------------------------
def effective(run, key):
    return run['settings'].get(key, COMMON.get(key, BASELINE.get(key)))


def hourly_su(run):
    """SU per hour on normalsr: 2 per core, memory counted as one core per 5 GB."""
    ncpus = float(effective(run, 'NCPUS'))
    mem_gb = float(str(effective(run, 'MEM')).upper().replace('GB', ''))
    return 2 * max(ncpus, mem_gb / 5)


def hours(time_str):
    h, m, s = (int(x) for x in str(time_str).split(':'))
    return h + m / 60 + s / 3600


def missing_inputs(run, input_dir):
    """Scenario-dependent input files the run reads at start (luto/data.py lines 587, 1240, 1241)."""
    ssp, rcp = effective(run, 'SSP'), effective(run, 'RCP')
    co2 = str(effective(run, 'CO2_FERT')).upper()
    needed = [f'climate_change_impacts_{rcp}_CO2_FERT_{co2}.h5',
              f'water_yield_ssp{ssp}_2010-2100_dr_ml_ha.h5',
              f'water_yield_ssp{ssp}_2010-2100_sr_ml_ha.h5',
              'demand_projections.h5', 'GHG_targets.xlsx',
              'bio_OVERALL_PRIORITY_RANK_AND_AREA_CONNECTIVITY.h5',
              'BIODIVERSITY_GBF2_conservation_performance.xlsx']
    if effective(run, 'DYNAMIC_PRICE'):
        needed.append('demand_elasticity.csv')
    return [f for f in needed if not os.path.exists(os.path.join(input_dir, f))]


def build_settings_df(task_root_dir, runs):
    """One column of settings per run, in the format `create_task_runs` expects."""
    defaults = get_settings_df(task_root_dir).set_index('Name')['Default_run'].to_dict()   # also writes non_str_val.txt
    columns = []
    for run in runs:
        settings_dict = dict(defaults)
        settings_dict.update({k: str(v) for k, v in COMMON.items()})
        settings_dict.update({k: str(v) for k, v in run['settings'].items()})
        settings_dict = update_settings(settings_dict, run['name'])
        columns.append(pd.Series(settings_dict, name=run['name']))
    df = pd.concat(columns, axis=1).reset_index(names='Name')
    df.index = df['Name'].values
    return df


def print_plan(runs, task_root_dir):
    rows = []
    for run in runs:
        rows.append({
            'run': run['name'], 'stage': run['stage'],
            'SSP': effective(run, 'SSP'), 'RCP': effective(run, 'RCP'), 'demand': effective(run, 'SCENARIO'),
            'CO2': effective(run, 'CO2_FERT'), 'dyn_price': effective(run, 'DYNAMIC_PRICE'),
            'amortise': effective(run, 'AMORTISE_UPFRONT_COSTS'),
            'exists': os.path.isdir(os.path.join(task_root_dir, run['name'])),
        })
    plan = pd.DataFrame(rows)
    with pd.option_context('display.max_rows', None, 'display.width', 250, 'display.max_colwidth', 60):
        print(plan.to_string(index=False))

    print()
    for stage, group in plan.groupby('stage', sort=False):
        stage_runs = [r for r in runs if r['stage'] == stage]
        per_hour = sum(hourly_su(r) for r in stage_runs)
        limit = sum(hourly_su(r) * hours(effective(r, 'TIME')) for r in stage_runs)
        if stage == 'smoke':
            print(f'{stage:6s} {len(stage_runs):3d} runs   at most {limit:,.0f} SU (walltime limit)')
        else:
            print(f'{stage:6s} {len(stage_runs):3d} runs   about {per_hour * EXPECTED_HOURS / 1000:5.1f} KSU at {EXPECTED_HOURS} h per run'
                  f'   at most {limit / 1000:5.1f} KSU (walltime limit)')


def update_template_csv(task_root_dir, settings_df):
    """Keep grid_search_template.csv in step with the runs created so far (0_Check_status.py and 0_write_tasks.py read it)."""
    path = os.path.join(task_root_dir, 'grid_search_template.csv')
    new = settings_df.drop(columns=['Name'])
    if os.path.exists(path):
        old = pd.read_csv(path, index_col=0)
        old = old.drop(columns=[c for c in new.columns if c in old.columns])
        new = pd.concat([old, new.astype(str).reindex(old.index.union(new.index, sort=False))], axis=1)
    new.reset_index(names='Name').to_csv(path, index=False)


# ----------------------------------------------------------------------------------------------
# Resume
# ----------------------------------------------------------------------------------------------
def in_queue(name):
    """True when a PBS job with this name is queued or running."""
    out = subprocess.run(['qselect', '-u', getpass.getuser(), '-N', name], capture_output=True, text=True)
    if out.returncode != 0:
        raise RuntimeError(f'qselect failed for {name}: {out.stderr.strip()}')
    return bool(out.stdout.strip())


def run_state(task_root_dir, name):
    """
    The state of a run and the mode to start it again with (None when there is nothing to start).

    finished      the job script reached its end, or the report of a write-only job is complete
    in the queue  a job with this name is queued or running
    interrupted   the job ended without reaching its end, for example at the walltime limit
    failed        the job script caught an error and wrote "Run failed."
    """
    run_dir = os.path.join(task_root_dir, name)
    if not os.path.isdir(run_dir):
        return 'not created', None, ''

    log_path = os.path.join(run_dir, 'output', 'simulation_log.txt')
    text = open(log_path, encoding='utf-8', errors='replace').read() if os.path.exists(log_path) else ''
    last = text.rsplit('Simulation started', 1)[-1]                 # the log is appended on every start
    out_dirs = sorted(d for d in glob.glob(os.path.join(run_dir, 'output', '*')) if os.path.isdir(d))
    files = [f for d in out_dirs for f in os.listdir(d)]
    checkpoints = sorted(f for f in files if re.fullmatch(r'data_\d{4}\.lz4', f))
    has_final = any(re.fullmatch(r'Data_RES\d+\.lz4', f) for f in files)
    report_done = any(os.path.exists(os.path.join(d, 'DATA_REPORT', 'data', 'map_layers', 'map_water_yield_NonAg.js')) for d in out_dirs)
    year = re.findall(r'Model finished in (\d{4})', last)
    detail = (f'solved to {year[-1]}' if year else '') or (f'checkpoint {checkpoints[-1][5:9]}' if checkpoints else '')

    if 'Total run time' in last or report_done:
        return 'finished', None, detail
    if in_queue(name):
        return 'in the queue', None, detail
    mode = 'Write' if has_final and not checkpoints else 'Run'
    state = 'failed' if 'Run failed.' in last else 'interrupted'
    return state, mode, detail


def resume(submit, retry_failed):
    groups = [(f'../../output/{TASK_NAME}', build_runs()), (f'../../output/{TASK_NAME}_smoke', smoke_run())]
    rows, todo = [], []
    for task_root_dir, runs in groups:
        for run in runs:
            state, mode, detail = run_state(task_root_dir, run['name'])
            if state == 'not created':
                continue
            rows.append({'run': run['name'], 'state': state, 'detail': detail, 'start again as': mode or ''})
            if state == 'interrupted' or (state == 'failed' and retry_failed):
                todo.append((task_root_dir, run['name'], mode))
    if not rows:
        print('No run folder exists yet.')
        return
    with pd.option_context('display.max_rows', None, 'display.width', 250, 'display.max_colwidth', 60):
        print(pd.DataFrame(rows).to_string(index=False))
    if not submit:
        print(f'\n{len(todo)} run(s) would be started again. Nothing was submitted. Add --submit to start them,'
              ' and --retry-failed to include the failed ones.')
        return
    for task_root_dir, name, mode in todo:
        submit_task(task_root_dir, name, 'NCI', 300, mode)
    print(f'\nStarted {len(todo)} run(s) again.')


# ----------------------------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------------------------
def main(argv):
    args = [a for a in argv if not a.startswith('--')]
    stage = args[0] if args else 'list'
    submit = '--submit' in argv
    stages = {'list': ['key', 'main', 'sens', 'amort'], 'all': ['key', 'main', 'sens'],
              'key': ['key'], 'main': ['main'], 'sens': ['sens'], 'amort': ['amort'], 'smoke': ['smoke']}
    if stage == 'resume':
        resume(submit, '--retry-failed' in argv)
        return
    if stage not in stages:
        sys.exit(f'Unknown stage "{stage}". Choose one of: {", ".join(stages)}, resume')
    if stage == 'list' and submit:
        sys.exit('Name a stage to submit: smoke, key, main, sens, amort or all.')

    if stage == 'smoke':
        runs, task_root_dir = smoke_run(), f'../../output/{TASK_NAME}_smoke'
    else:
        runs = [r for r in build_runs() if r['stage'] in stages[stage]]
        task_root_dir = f'../../output/{TASK_NAME}'

    print(f'Task folder: {os.path.abspath(task_root_dir)}\n')
    print_plan(runs, task_root_dir)

    input_dir = os.path.join(REPO_ROOT, 'input')
    problems = {r['name']: missing_inputs(r, input_dir) for r in runs}
    problems = {k: v for k, v in problems.items() if v}
    for name, files in problems.items():
        print(f'\nMISSING INPUT for {name}: {", ".join(files)}')

    if not submit:
        print('\nNothing was created or submitted. Add --submit to a stage to launch it.')
        return
    if problems:
        sys.exit('\nNot submitted: fix the missing input files first.')

    todo = [r for r in runs if not os.path.isdir(os.path.join(task_root_dir, r['name']))]
    for r in runs:
        if r not in todo:
            print(f'Skipping {r["name"]}: its folder already exists.')
    if not todo:
        print('\nEvery run of this stage already has a folder. Nothing submitted.')
        return

    settings_df = build_settings_df(task_root_dir, todo)
    update_template_csv(task_root_dir, settings_df)
    create_task_runs(task_root_dir, settings_df, platform='NCI', n_workers=4, use_parallel=False)
    print(f'\nSubmitted {len(todo)} runs of stage "{stage}".')


if __name__ == '__main__':
    main(sys.argv[1:])
