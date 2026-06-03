# Skill: Debug Species/Community Infeasibility

This skill diagnoses which GBF4 SNES species or ECNES ecological communities have
NRM-region targets that cannot be met, causing solver infeasibility. It works by
building a base model (all species constraints removed), then testing each
species/community individually — maximising its LHS to check whether the RHS target
is physically achievable given all other constraints (GHG, water, land budget, etc.).

## When to use

- A run fails with `GRB.INFEASIBLE` at a specific year and GBF4 SNES or ECNES constraints are active
- You want to identify which species to add to `GBF4_SNES_EXCLUDE_REGION_SPECIES` or `GBF4_ECNES_EXCLUDE_REGION_COMMUNITIES`
- Applies to any USER_DEFINED or NRM-mode SNES/ECNES run

## Key concept

For each species/community constraint `sum(area_sr × dvar_r) >= target`:
- Keep all other constraints active (GHG, water, GBF2, land budget, etc.)
- Maximise the LHS → find the maximum achievable area for this species
- If `max_achievable < target`: the target is impossible even in the best case → **infeasible by construction**
- If `max_achievable >= target`: the target is individually achievable (infeasibility may be from constraint combinations)

## Prerequisites

The MPS file must exist. It is **not** auto-saved by the current code due to a bug
(the `solve()` method crashes on `var.X` before the MPS save block is reached).
Use `build_mps.py` to reconstruct it from a checkpoint instead.

## Scripts (save to the run's inspection directory)

| Script | Purpose |
|--------|---------|
| `build_mps.py` | Load checkpoint → formulate yr N model → save MPS + `constraint_metadata.json` |
| `build_mps.pbs` | PBS submission wrapper for `build_mps.py` (hugemem queue) |
| `find_infeasible_species.py` | Load MPS → build base model → skip already-excluded species → submit per-species jobs |
| `submit_species_checks.pbs` | PBS wrapper for `find_infeasible_species.py` (can chain with `depend=afterok`) |
| `check_one_species.py` | Worker: load base model → maximise one constraint LHS → save result JSON |
| `resubmit_timedout.py` | Re-submit only the jobs whose result JSON has `max_achievable_scaled: null` |

Reference implementation: `jinzhu_inspect_code/Check_NECMA_crash/`

## Steps

### Step 1: Build the MPS from checkpoint

Edit `build_mps.py` for your run:
- `RUN_DIR`: path to the failing run directory (contains its own `luto/` copy)
- `CHECKPOINT`: path to the last successful `data_YYYY.lz4`
- `OUT_DIR`: where to save MPS and metadata
- `BASE_YEAR` / `TARGET_YEAR`: the year transition that failed

Submit:
```bash
qsub build_mps.pbs
```

This saves:
- `debug_model_{BASE}_{TARGET}.mps` — full Gurobi model
- `constraint_metadata.json` — all SNES/ECNES constraints with region, species, scale factor, unscaled RHS

### Step 2: Submit per-species checks

Once `build_mps.pbs` finishes, either chain it with a dependency:
```bash
qsub -W depend=afterok:<build_job_id> submit_species_checks.pbs
```

Or run directly:
```bash
conda run -n luto python find_infeasible_species.py --dry-run   # preview
conda run -n luto python find_infeasible_species.py             # submit jobs
```

`find_infeasible_species.py` will:
1. Load the MPS and extract all SNES/ECNES constraint rows
2. Skip species/communities already in the settings exclusion lists
3. Remove all SNES/ECNES constraints → save `species_checks/base_model.mps`
4. Verify base model is feasible (non-blocking if time-limit hit — yr 2045 solved means base is feasible)
5. Submit one `normalsr` PBS job per remaining constraint

### Step 3: Collect results

```bash
conda run -n luto python find_infeasible_species.py --collect
```

Prints INFEASIBLE/FEASIBLE table and saves `species_feasibility_summary.csv`:

```
======================================================================
SPECIES FEASIBILITY RESULTS  (59 completed, 0 pending)
======================================================================

INFEASIBLE — 3 species/communities cannot meet their target:
  Type    Region                Species/Community              Target ha    Max ha
  SNES    North East            Acacia whanii                    12,500     8,432
  ECNES   Goulburn Broken       Montane bogs and fens            45,000    31,200
  ...

FEASIBLE — 56 species/communities can meet their target individually.
```

### Step 4: Re-submit timed-out jobs (if any)

If some result JSONs have `max_achievable_scaled: null` with `gurobi_status: 9`
(`GRB.TIME_LIMIT`), the solver was cut off before converging. Re-submit using the
existing base model and constraint JSONs — **no rebuild needed**:

```bash
python resubmit_timedout.py --dry-run   # verify count
python resubmit_timedout.py             # submit
```

Then re-run `--collect` once those jobs finish.

### Step 5: Update exclusion lists

Add infeasible species to settings:
```python
# For SNES species:
GBF4_SNES_EXCLUDE_REGION_SPECIES = [
    ('North East', 'Acacia whanii'),
    ...existing entries...
]

# For ECNES communities:
GBF4_ECNES_EXCLUDE_REGION_COMMUNITIES = [
    ('Goulburn Broken', 'Montane bogs and fens'),
    ...existing entries...
]
```

Resubmit the run — the next failing year will hit a different species.

## Known pitfalls

### `max_achievable_scaled: null` with `gurobi_status: 9` (TIME_LIMIT)

**Cause:** `check_one_species.py` originally had `model.Params.TimeLimit = 600` (10 min),
but some constraints take longer to maximise over the full LUTO model. When Gurobi hits
the time limit without proving optimality, the code must check `model.SolCount > 0`
to recover any incumbent found during the run. The correct pattern:

```python
has_solution = (
    status in (GRB.OPTIMAL, GRB.SUBOPTIMAL)
    or (status == GRB.TIME_LIMIT and model.SolCount > 0)
)
max_scaled = model.ObjVal if has_solution else None
```

**Current fix:** `check_one_species.py` no longer sets `TimeLimit` — Gurobi runs to
completion (OPTIMAL or INFEASIBLE). PBS walltime (4 h) acts as the hard cap.

**Do not** set `model.Params.TimeLimit` shorter than the PBS walltime — the mismatch
causes all hard cases to return `null` instead of a usable bound.

### PBS syntax error for species names with parentheses

Species/community names like `Grey Box (Eucalyptus microcarpa) Grassy Woodlands...`
contain `(...)` which bash interprets as a subshell when the path is unquoted in the
PBS script. Always quote all paths in the generated PBS command:

```python
# correct
conda run -n luto python "{WORKER}" "{BASE_MPS}" "{cjson}" "{rjson}"

# wrong — bash treats (...) as subshell
conda run -n luto python {WORKER} {BASE_MPS} {cjson} {rjson}
```

## Notes

- The base model verification may time out (`status=9`) on large RF3/RF5 models — this is treated as a warning not an error, since yr 2045 solving optimally confirms the base is feasible
- All checks share the **same** `base_model.mps` — each job just sets a different objective
- Once `base_model.mps` exists in `species_checks/`, re-submissions never need to rebuild it
- Exclusion loading uses regex on `settings.py` (not `exec()`) to avoid import errors
- The `species` field in `constraint_metadata.json` holds the community name for ECNES rows (uniform field name)
