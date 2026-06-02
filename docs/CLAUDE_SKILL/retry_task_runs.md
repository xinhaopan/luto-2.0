# Skill: Retry Task Runs with Adjusted Gurobi Parameters

This skill retries a batch of task runs where one or more years returned a non-optimal
solver status (INFEASIBLE, NUMERIC, SUBOPTIMAL). It unzips the original `Run_Archive.zip`
into a new task directory, overlays fresh source + patched settings, and resubmits via
`redo_checkpoint.py`.

---

## When checkpoint-based retry actually fixes the infeasible year

The retry is only useful if the checkpoint sitting in the archive is from the **year
before** the infeasible one. Whether that holds depends on which version of
`simulation.py` created the archive:

| `simulation.py` version | Checkpoint saved when | Archive contains | Retry fixes infeasible year? |
|---|---|---|---|
| **Old** (unconditional save) | Every year, optimal or not | Infeasible year's state | **No** — infeasible year is the base, not re-solved |
| **New** (save only on optimal) | Last good year only | Year before infeasibility | **Yes** — infeasible year becomes the first target |

The new behaviour (`accepted`-gated checkpoint save + pre-loop base checkpoint) is in
`simulation.py:solve_timeseries`. If the archive was created by old code, a full re-run
from scratch is the only way to fix the infeasible year.

---

## Step 1: Check which year went infeasible and why

```bash
# Scan all PBS stdout logs for solver status messages
for run in /g/data/jk53/jinzhu/LUTO/Custom_runs/<ITER>/Run_G*/; do
    name=$(basename $run)
    pbs_out=$(ls $run/run_G*.o* 2>/dev/null | head -1)
    echo "=== $name ==="
    grep -i "infeasib\|non-optimal\|Solver status" "$pbs_out" 2>/dev/null
done
```

Check the IIS `.err` files if IIS jobs were submitted — "Cannot compute IIS on a
feasible model" means the infeasibility was **numerical** (false infeasible), not
structural. A structurally infeasible model requires constraint relaxation, not solver
tuning.

---

## Step 2: Choose the right `RETRY_PARAMS`

Edit `BASE_GRID["RETRY_PARAMS"]` in `retry_create_task.py`. The default setting that
resolves numerical false-infeasibility:

```python
# Dual simplex as third attempt — avoids false-INFEASIBLE from the homogeneous
# barrier (BARHOMOGENOUS=1), which declares infeasibility too aggressively when
# the feasible region is tight. Dual simplex uses a different code path.
"RETRY_PARAMS": [(0, 2, 0), (3, 2, 0), (3, 1, 0)],
```

Alternative if you suspect barrier stagnation rather than false-infeasibility:

```python
# Barrier with auto crossover — forces a vertex solution from the interior point,
# resolving stagnation at ill-conditioned termination. Can be slow on large models.
"RETRY_PARAMS": [(3, 2, -1)],
```

Method reference: `-1`=auto, `0`=primal simplex, `1`=dual simplex, `2`=barrier,
`3`=concurrent, `4`=deterministic concurrent.

---

## Step 3: Write `retry_create_task.py`

Place it under `jinzhu_inspect_code/<Iteration>/retry_create_task.py`. It mirrors
`create_tasks.py` in structure but `main()` does three things instead of creating
fresh run folders:

1. Builds the settings template (same `BASE_GRID` + `RUN_OVERRIDES` pattern).
2. Unzips each `Run_Archive.zip` from `SOURCE_DIR` into `TASK_DIR/Run_G000X/` — skips
   any run that already has `output/*/data_*.lz4` (idempotent).
3. Calls `create_task_runs(overwrite=True)` to overlay fresh `luto/` source and write
   new `settings.py` + `task_param.py` without submitting.

```python
SOURCE_DIR = Path("/g/data/jk53/jinzhu/LUTO/Custom_runs/<ITER>")
TASK_DIR   = Path("/g/data/jk53/jinzhu/LUTO/Custom_runs/<ITER>_retry")
```

Key points:
- `create_task_runs(overwrite=True)` calls `create_run_folders`, which copies the full
  `luto/` source tree but excludes `output/` (it is in `EXCLUDE_DIRS`), so unzipped
  checkpoints are never touched.
- The per-run `settings.py` written by `write_settings` overrides `RETRY_PARAMS` with
  the value from `BASE_GRID`, regardless of what `luto/settings.py` says.

```python
def main():
    TASK_DIR.mkdir(parents=True, exist_ok=True)
    template = build(TASK_DIR)

    archives = {p.parent.name: p for p in sorted(SOURCE_DIR.glob("Run_G*/Run_Archive.zip"))}
    print(f"\nFound {len(archives)} archives in {SOURCE_DIR.name}\n")

    run_cols = [c for c in template.columns if c != "Name"]
    for col in run_cols:
        run_dir      = TASK_DIR / col
        archive_path = archives.get(col)
        print(f"=== {col} ===")
        if archive_path is None:
            print(f"  [WARN] No archive found — skipping.\n"); continue

        lz4_files = sorted(run_dir.glob("output/*/data_*.lz4")) if run_dir.exists() else []
        if lz4_files:
            print(f"  Already unzipped ({lz4_files[-1].relative_to(run_dir)}) — skipping.\n"); continue

        run_dir.mkdir(parents=True, exist_ok=True)
        print(f"  Unzipping: {archive_path.relative_to(SOURCE_DIR.parent)}")
        with zipfile.ZipFile(archive_path) as zf:
            zf.extractall(run_dir)

        lz4_files = sorted(run_dir.glob("output/*/data_*.lz4"))
        if lz4_files:
            print(f"  Checkpoint: {lz4_files[-1].relative_to(run_dir)}\n")
        else:
            print(f"  [WARN] No checkpoint found — run will start from scratch.\n")

    print("Writing updated settings ...\n")
    create_task_runs(str(TASK_DIR), template, mode="cluster", n_workers=4, overwrite=True)

    print(f"\nDone. To submit:\n  cd {TASK_DIR}\n  python redo_checkpoint.py")
```

---

## Step 4: Run the script

```bash
cd /g/data/jk53/jinzhu/LUTO/luto-2.0
python jinzhu_inspect_code/<Iteration>/retry_create_task.py
```

Verify checkpoint files were found for each run:

```bash
for run in /g/data/jk53/jinzhu/LUTO/Custom_runs/<ITER>_retry/Run_G*/; do
    echo "$(basename $run): $(ls $run/output/*/data_*.lz4 2>/dev/null || echo 'NO CHECKPOINT')"
done
```

---

## Step 5: Submit via `redo_checkpoint.py`

```bash
cd /g/data/jk53/jinzhu/LUTO/Custom_runs/<ITER>_retry
python redo_checkpoint.py --dry-run   # preview — shows checkpoint year per run
python redo_checkpoint.py             # submit
```

`redo_checkpoint.py` classifies run dirs and only submits those with a checkpoint lz4
and no `Run_Archive.zip`. It inherits `MEM`, `NCPUS`, `TIME`, `QUEUE` from each run's
`task_param.py`; override with `--mem`, `--ncpus`, `--time`, `--queue` if needed.

---

## How checkpoint resume re-solves the failed year

`simulation.py:solve_timeseries` with the new checkpoint logic:

```
Before loop:  saves data_{years_to_run[0]}.lz4   (only if file absent — skipped on resume)

Loop step 0:  base=2030, target=2035 → solve 2035
  → OPTIMAL:  save data_2035.lz4, delete data_2030.lz4
  → FAIL:     do NOT save → data_2030.lz4 survives, loop breaks

redo_checkpoint loads data_2030.lz4
  → years_to_run = [2030, 2035, 2040, ...]
  → step 0: base=2030, target=2035  ← re-solves the failed year ✓
```

Edge case — fails on the very first target year (e.g. 2020 on a fresh run):
- Pre-loop saves `data_2010.lz4` (base year, before any solve)
- 2020 fails → `data_2010.lz4` survives
- `redo_checkpoint` resumes from 2010 → re-solves 2020 ✓

---

## Common issues

| Symptom | Cause | Fix |
|---|---|---|
| "Cannot compute IIS on a feasible model" in IIS `.err` | False infeasibility — model is feasible with default tolerances | Use `RETRY_PARAMS` with dual simplex `(3, 1, 0)` |
| Retry still shows infeasible year in results | Archive created by old code — contains infeasible year's checkpoint, not prior year | Full re-run from scratch with new `RETRY_PARAMS` |
| No checkpoint found after unzip | Archive only contains `Data_RES*.lz4` (final state), not `data_YEAR.lz4` | Run was created with very old code; full re-run needed |
| `redo_checkpoint` classifies run as `finished` | `Run_Archive.zip` present in the unzipped dir | Old archive created a zip-inside-zip; remove `Run_Archive.zip` from `TASK_DIR/Run_G*` |
