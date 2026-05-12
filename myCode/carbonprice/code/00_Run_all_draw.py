"""
Run carbonprice plotting scripts for all three discount rates simultaneously.

Execution order per rate:
  1. 01 must complete first
  2. 02-12 run in parallel across all rates
  3. 13 runs after all 02-12 for that rate are done

All three rates run concurrently. A single global thread pool (capped at
cpu_count) avoids spawning too many subprocesses at once.
"""
import sys
import os
import subprocess
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, wait, ALL_COMPLETED

RATES = ["0.03", "0.05", "0.07"]

FIRST_SCRIPT = "01_create_GBF2_targets_csv.py"
PARALLEL_SCRIPTS = [
    "02_draw_original_data.py",
    "03_draw_processed_data.py",
    "04_draw_average_price.py",
    "05_draw_sol_average_price.py",
    "06_make_cost_map.py",
    "07_make_GHG_benefits_map.py",
    "08_make_BIO_benefits_map.py",
    "09_make_price_map.py",
    "10_make_agmgt_map.py",
    "11_make_non_ag_map.py",
    "12_biodiversity_contribution_curve.py",
]
FINAL_SCRIPT = "13_Rename_figure.py"

PYTHON = sys.executable
CWD = os.path.dirname(os.path.abspath(__file__))
MAX_WORKERS = min(len(RATES) * len(PARALLEL_SCRIPTS), os.cpu_count() or 8)


def run_script(script: str, rate: str) -> tuple:
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "CARBON_RATE": rate}
    r = subprocess.run(
        [PYTHON, script], cwd=CWD, env=env,
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    return script, rate, r.returncode, r.stdout + r.stderr


def log(script: str, rate: str, code: int, output: str):
    status = "OK" if code == 0 else f"ERROR (exit {code})"
    print(f"[rate={rate}] [{status}] {script}", flush=True)
    if code != 0:
        print(output[-2000:], flush=True)


print(f"Starting pipelines for rates: {RATES}  (max_workers={MAX_WORKERS})\n")

with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:

    # Step 1: run 01 for all rates in parallel, wait for all
    step1_futures = {pool.submit(run_script, FIRST_SCRIPT, rate): rate for rate in RATES}
    wait(step1_futures, return_when=ALL_COMPLETED)
    for f, rate in step1_futures.items():
        script, rate_, code, output = f.result()
        log(script, rate_, code, output)

    # Step 2: submit 02-12 for all rates at once
    step2_futures = {}
    rate_futures = defaultdict(list)
    for rate in RATES:
        for script in PARALLEL_SCRIPTS:
            f = pool.submit(run_script, script, rate)
            step2_futures[f] = rate
            rate_futures[rate].append(f)

    # As each finishes, log it; when all of a rate's scripts are done, launch 13
    step3_futures = {}
    finished_rates = set()
    remaining = set(step2_futures)

    while remaining:
        done, remaining = wait(remaining, return_when="FIRST_COMPLETED")
        for f in done:
            script, rate, code, output = f.result()
            log(script, rate, code, output)
            if rate not in finished_rates and all(ff.done() for ff in rate_futures[rate]):
                finished_rates.add(rate)
                print(f"[rate={rate}] All parallel scripts done — starting 13", flush=True)
                f13 = pool.submit(run_script, FINAL_SCRIPT, rate)
                step3_futures[f13] = rate

    # Step 3: collect 13 results
    wait(step3_futures, return_when=ALL_COMPLETED)
    for f, rate in step3_futures.items():
        script, _, code, output = f.result()
        if output.strip():
            print(output, flush=True)
        log(script, rate, code, output)

# Step 4: collect all figures into one Paper_figures folder
print("\nCollecting all figures into Paper_figures ...", flush=True)
r = subprocess.run(
    [PYTHON, "14_all_name_change.py"], cwd=CWD,
    capture_output=True, text=True, encoding="utf-8", errors="replace",
)
print(r.stdout + r.stderr, flush=True)
log("14_all_name_change.py", "all", r.returncode, r.stdout + r.stderr)

print("\nAll done.")
