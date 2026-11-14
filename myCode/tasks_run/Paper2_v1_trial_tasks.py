"""Paper2 revision, trial runs on the current LUTO2 code (branch paper2_v1, from land-use-trade-offs/jinzhu e95e9c0d).

Two runs with the checkout's default settings (GHG_EMISSIONS_LIMITS='low', GBF2_TARGET='high', SSP2-4.5,
SIM_YEARS 2020-2050 in 5-year steps), which is the closest default to the Paper 2 joint NZ + NP scenario:

    Run_RES5_default   quick check that data, solver, duals and report all work in this checkout
    Run_RES1_default   full resolution, to see solve time and memory before the revision runs are set up

Input is a copy of Jinzhu's NCI input (/g/data/jk53/jinzhu/LUTO/luto-2.0/input, 2026-10-22).
climate_change_impacts_rcp7p0_* in that input is the RCP6.0 layer under the rcp7p0 name (no GAEZ RCP7.0 layer).

    cd /g/data/jk53/LUTO_XH/LUTO2_paper2_v1
    PYTHONPATH=$PWD conda run --no-capture-output -n luto python myCode/tasks_run/Paper2_v1_trial_tasks.py            # CSVs only
    PYTHONPATH=$PWD conda run --no-capture-output -n luto python myCode/tasks_run/Paper2_v1_trial_tasks.py --submit   # submit
    ... --only RES5      build / submit only the matching runs
"""
import argparse
from pathlib import Path

import pandas as pd
from luto.tools.create_task_runs.helpers import (
    create_task_runs, get_settings_df, get_grid_search_settings_df,
)

TASK_DIR = Path("/g/data/jk53/LUTO_XH/LUTO2_paper2_v1/output/20261114_Paper2_v1_trial")

COMMON = {
    "QUEUE": "normalsr",
    "SIM_YEARS": [2020, 2025, 2030, 2035, 2040, 2045, 2050],
    "WRITE_OUTPUTS": True,
}

RUNS = {
    "RES5_default": {"RESFACTOR": 5, "NCPUS": 32, "MEM": "128GB", "TIME": "10:00:00",
                     "WRITE_REPORT_MAX_MEM_MB": 100 * 1024},
    # Jinzhu's RES1 run of 2026-10-30 took 3.5-7 h per solve year and peaked near 133 GB
    "RES1_default": {"RESFACTOR": 1, "NCPUS": 104, "MEM": "500GB", "TIME": "48:00:00",
                     "WRITE_REPORT_MAX_MEM_MB": 300 * 1024},
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--submit", action="store_true", help="create run folders and qsub them")
    ap.add_argument("--only", default=None, help="substring of the run name")
    args = ap.parse_args()

    names = [n for n in RUNS if args.only is None or args.only in n]
    TASK_DIR.mkdir(parents=True, exist_ok=True)
    existing = [n for n in names if (TASK_DIR / f"Run_{n}").exists()]
    if existing:
        print(f"Skipping existing run folders: {existing}")
    names = [n for n in names if n not in existing]
    if not names:
        print("Nothing to do.")
        return

    default_df = get_settings_df(str(TASK_DIR))
    rows = []
    for n in names:
        row = {**COMMON, **RUNS[n]}
        rows.append({k: str(v) for k, v in row.items()})
    param_df = pd.DataFrame(rows)
    param_df.insert(0, "run_idx", range(1, len(param_df) + 1))
    param_df.to_csv(TASK_DIR / "grid_search_parameters.csv", index=False)

    settings_df = get_grid_search_settings_df(str(TASK_DIR), default_df, param_df)
    settings_df.columns = ["Name"] + [f"Run_{n}" for n in names]
    for n in names:
        settings_df.loc[settings_df["Name"] == "JOB_NAME", f"Run_{n}"] = f"p2v1_{n}"
    settings_df.to_csv(TASK_DIR / "grid_search_template.csv", index=False)

    for n in names:
        s = settings_df.set_index("Name")[f"Run_{n}"]
        print(f"Run_{n}: RESFACTOR={s['RESFACTOR']} NCPUS={s['NCPUS']} MEM={s['MEM']} TIME={s['TIME']} "
              f"GHG={s['GHG_EMISSIONS_LIMITS']} GBF2={s['GBF2_TARGET']} SSP={s['SSP']} INPUT_DIR={s['INPUT_DIR']}")

    if args.submit:
        create_task_runs(str(TASK_DIR), settings_df, mode="cluster", n_workers=1)
    else:
        print(f"CSV only, written to {TASK_DIR}. Add --submit to create the run folders and qsub.")


if __name__ == "__main__":
    main()
