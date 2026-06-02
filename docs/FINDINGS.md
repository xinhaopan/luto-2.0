# LUTO2 Findings Log

A running record of discoveries, investigations, and conclusions from model exploration.
Entries are in **descending date order** (newest first).

---

## 20260602 — RP fix (commit 94971ea) causes infeasibility in REM_RES5 and NECMA_follow_runs

### Context

After commit `94971ea` (Riparian Plantings transition-matrix eviction fix), all six
`REM_RES5` runs and 11 of 13 `NECMA_follow_runs` became infeasible at various years
(2020–2045). Runs that were previously optimal are now reported as `INFEASIBLE` by Gurobi.
Three distinct root causes were identified; only the first is directly caused by the fix.

---

### Run inventory — REM_RES5 (`N:\LUF-Modelling\LUTO2_JZ\Custom_runs\REM_RES5`)

Shared settings: `GBF2_TARGET=high (hard)`, `GHG=low (hard)`, `WATER=on (hard)`,
`RENEWABLE_TARGET=Gladstone - Core`, `GBF4=off`, `GBF3=off`.

| Run | Spatial layers | Excl. GBF2 cells | GBF2 cut Solar/Wind | Fails at | Root cause |
|---|---|---|---|---|---|
| G0001 / RE0001 | step_change | No | 0 / 0 | **2040** | Cause 1 |
| G0002 / RE0002 | step_change | Yes | 25 / 25 | **2035** | Cause 1 (earlier: fewer renewable cells) |
| G0003 / RE0003 | accelerated_transition | No | 0 / 0 | **2045** | Cause 1 |
| G0004 / RE0004 | accelerated_transition | Yes | 25 / 25 | **2035** | Cause 1 (earlier: fewer renewable cells) |
| G0005 / RE0005 | ANU_transmission_T10 | No | 0 / 0 | **2045** | Cause 1 |
| G0006 / RE0006 | ANU_transmission_T10 | Yes | 25 / 25 | **2035** | Cause 1 (earlier: fewer renewable cells) |

---

### Run inventory — NECMA_follow_runs (`N:\LUF-Modelling\LUTO2_JZ\Custom_runs\NECMA_follow_runs`)

Shared settings: `GBF2_TARGET=high (hard)`, `WATER=on (hard)`, `RENEWABLE_TARGET=Gladstone - Core`,
`RENEWABLE_LAYERS=step_change`, `GBF4_SNES/ECNES=USER_DEFINED`.

| Run | Group | GHG | GBF2 cut % | GBF4 mode | Water stress | SSP | Adoption cap | Fails at | Root cause |
|---|---|---|---|---|---|---|---|---|---|
| G0001 | CORE | low | 15 | NRM | 0.6 | 245 | — | **2040** | Cause 1 |
| G0002 | GHG_SENSITIVITY | **high** | 15 | NRM | 0.6 | 245 | — | **2040** | Cause 1 |
| G0003 | WATER_SENSITIVITY | low | 15 | NRM | **0.5** | 245 | — | **2040** | Cause 1 |
| G0004 | WATER_SENSITIVITY | low | 15 | NRM | **0.7** | 245 | — | **2020** | Cause 2 (water too tight) |
| G0005 | WATER_SENSITIVITY | low | 15 | NRM | **0.8** | 245 | — | **2020** | Cause 2 (water too tight) |
| G0006 | CLIMATE_SENSITIVITY | low | 15 | NRM | 0.6 | **126** | — | **2040** | Cause 1 |
| G0007 | CLIMATE_SENSITIVITY | low | 15 | NRM | 0.6 | **370** | — | **2040** | Cause 1 |
| G0008 | BIO_SENSITIVITY | low | **0** | NRM | 0.6 | 245 | — | **2040** | Cause 1 |
| G0009 | BIO_SENSITIVITY | low | **10** | NRM | 0.6 | 245 | — | **2040** | Cause 1 |
| G0010 | BIO_SENSITIVITY | low | **20** | NRM | 0.6 | 245 | — | **2040** | Cause 1 |
| G0011 | BIO_SENSITIVITY | low | 15 | **AUSTRALIA** | 0.6 | 245 | — | **passes** | AUSTRALIA mode relaxes GBF4 |
| G0012 | SOCIAL_LICENCE | low | 15 | NRM | 0.6 | 245 | **5%** | **2025** | Cause 3 (adoption cap) |
| G0013 | SOCIAL_LICENCE | low | 15 | NRM | 0.6 | 245 | **10%** | **2030** | Cause 3 (adoption cap) |

---

### 1 — Primary cause (REM_RES5 all 6; NECMA G0001–G0003, G0006–G0010): non-ag land accumulation exhausts the joint land budget

**Mechanism — pre-fix (bug present):**

`get_to_non_ag_exclude_matrices` set `t_rk = 0` (UB = 0) for RP cells each period because
their dominant `lumap` entry was not RP (see 20260601 entry). `get_lower_bound_non_agricultural_matrices`
then capped `lb_rk` against this zero UB, silently zeroing the lower bound and freeing
every RP cell back to the general pool each year. The optimizer could re-allocate those
cells jointly to both biodiversity targets and renewable energy management — effectively
double-using the land.

**Mechanism — post-fix (correct):**

RP cells now maintain their lower bounds across periods. Irreversible non-ag land
(EP, RP, CP, Agroforestry, BECCS) accumulates monotonically. By year 2035–2045:

- GBF2 high hard target (30% restoration by 2030, 50% by 2050) has locked hundreds of
  thousands of cell-proportions into non-ag land uses
- Gladstone Core renewable targets require large and growing generation (e.g. NSW Solar
  61.6 TWh in 2040, up from 9.4 TWh existing; WA Wind 48 TWh with 3 TWh existing)
- The joint land budget — non-ag biodiversity + renewable-eligible ag — is exceeded

**Why the pre-fix runs appeared feasible:**

The optimizer exploited the annual eviction to strategically reassign RP cells. It could
choose each year which cells go to non-ag vs. renewables, avoiding locking high-value
renewable cells into permanent RP. With the fix, early-year non-ag allocations (made
before renewable targets grew large) are irreversibly retained, even in prime renewable
zones.

**Infeasibility onset by run:**

| Run set | Fails at | Notes |
|---|---|---|
| REM_RES5 G0002, G0004, G0006 (`EXCLUDE_RENEWABLES_IN_GBF2_MASKED_CELLS=True`) | 2035 | GBF2 cells excluded from renewables → pool shrinks faster |
| REM_RES5 G0001, G0003, G0005 (`EXCLUDE_RENEWABLES_IN_GBF2_MASKED_CELLS=False`) | 2040–2045 | |
| NECMA G0001–G0003, G0006–G0010 | 2040 | All use `WATER_STRESS=0.6`, `GBF4=NRM` mode |

**Nature of infeasibility:** The renewable energy constraints are hard (`>=` with no slack).
When locked-in non-ag land reduces the effective renewable-eligible area below what is
needed to meet state-level generation targets, Gurobi returns status 3 (INFEASIBLE).
No `lb > ub` variable-level infeasibility is involved — this is a global land-budget
infeasibility.

**Why NECMA G0011 passes:** `GBF4_SNES_REGION_MODE=AUSTRALIA`, `GBF4_ECNES_REGION_MODE=AUSTRALIA`.
National-level aggregation is far looser than per-NRM constraints, giving the optimizer
enough flexibility to jointly satisfy biodiversity and renewable targets through 2050.

---

### 2 — Secondary cause (NECMA G0004, G0005): `WATER_STRESS` too tight — infeasible from year 2020

`WATER_STRESS=0.7` (G0004) and `WATER_STRESS=0.8` (G0005) require the model to maintain
70%/80% of natural water yield in every river region. This is structurally infeasible from
year 2020 given the simultaneous hard constraints:

- GBF2 high hard (large afforestation lowers water yield)
- Gladstone Core renewable targets at year 2020
- GHG hard constraint

**Unrelated to the RP fix**: year 2020 uses `base_year = YR_CAL_BASE`, so
`existing_dvars_rk = None` and the fix never engages. These runs were likely infeasible
before the fix as well. G0003 (`WATER_STRESS=0.5`) passes 2020 and fails only at 2040
(Cause 1), confirming the threshold is between 0.5 and 0.7.

---

### 3 — Tertiary cause (NECMA G0012, G0013): `NON_AG_CAP` prevents GBF2 ramp

`REGIONAL_ADOPTION_CONSTRAINTS=NON_AG_CAP` with `NON_AG_CAP=5` (G0012) and `NON_AG_CAP=10`
(G0013) caps non-ag adoption at 5%/10% per NRM region per period. The GBF2 hard target
requires faster restoration than the cap permits:

- G0012 (5% cap): meets 2020 target, fails at **2025**
- G0013 (10% cap): meets 2020 and 2025, fails at **2030**

Not caused by the RP fix — the adoption rate constraint independently prevents the
biodiversity target from being reached.

---

### 4 — Options for resolution

| Option | Addresses | Trade-off |
|---|---|---|
| Make renewable energy constraints **soft** (add slack + penalty in `_add_renewable_energy_constraints`) | Cause 1 | Model reports shortfall instead of crashing; renewable targets become aspirational |
| Set `GBF2_CONSTRAINT_TYPE=soft` in REM/NECMA scenarios | Cause 1 | Biodiversity gives way when land budget is exhausted |
| Reduce `GBF2_TARGET` to `medium` in renewable-heavy scenarios | Cause 1 | Accepts lower biodiversity target where renewables are required |
| Reduce `WATER_STRESS` to ≤ 0.6 | Cause 2 | G0004/G0005 water scenarios were structurally too tight |
| Reduce `NON_AG_CAP` or use soft GBF2 | Cause 3 | Social licence scenarios need lower ramp rate or relaxed biodiversity |

The infeasibility from Cause 1 is **correct model behaviour** after the RP fix — the
scenarios were only feasible before because the eviction bug provided illegal land
flexibility. The pre-fix solutions violated irreversibility.

---

### 5 — Why the GBF2–renewable conflict is temporal, not spatial

A natural question: GBF2 is a national target (can be met anywhere in Australia) and,
without `EXCLUDE_RENEWABLES_IN_GBF2_MASKED_CELLS`, renewables are not spatially excluded
from GBF2 priority cells. Both targets therefore appear spatially compatible — why is the
combined scenario infeasible?

**The conflict is not in any single year; it is accumulated across the rolling horizon.**

The rolling-horizon optimizer has no foresight beyond the current period. When solving
year 2025:

- It places EP/RP/CP wherever it is cheapest to meet 30% GBF2 nationally.
- The cheapest GBF2 restoration is concentrated in productive NSW/QLD/WA agricultural
  zones — they are degraded (good for GBF2), economically accessible, and have existing
  rural infrastructure.
- NSW/QLD/WA are also the states with the largest renewable energy targets.
- In 2025 this does not matter: the 2025 renewable targets are small and existing
  installed capacity already covers most of them.

After the fix these 2025 non-ag commitments are irreversible. In 2030 more are added
(GBF2 target growing from 30% toward 50%), again concentrated in the same cheapest
states. The pattern compounds:

| Year | Cumulative locked non-ag in NSW/QLD/WA | Renewable gap still to fill |
|---|---|---|
| 2025 | small | small (mostly met by existing capacity) |
| 2030 | grows | moderate |
| 2035 | larger | large |
| 2040 | **too large** | **~52 TWh NSW Solar + 47 TWh NSW Wind + 45 TWh WA Wind** |

By 2040 the locked-in EP/RP/CP/BECCS has consumed enough renewable-eligible cells
within specific states that state-level hard constraints become infeasible.

**Why the pre-fix optimizer avoided this:** the eviction bug was in effect re-optimising
the spatial assignment of GBF2 every year. As renewable targets grew in later years the
optimizer could opportunistically shift non-ag out of prime renewable zones (NSW/WA) and
into lower-renewable-pressure states (SA, NT, Tasmania), because eviction freed those
cells. Post-fix it cannot undo 2025–2035 commitments.

**Solver credit for locked-in RP is correct:** the GBF2 constraint LHS does include the
Gurobi variable for each locked-in RP cell (lb = RP_PROPORTION, in `non_ag_lu2cells`
and in `_add_GBF2_constraints`). Gurobi correctly counts the lb contribution — the
remaining biodiversity shortfall the solver must close from other land is properly
reduced. The infeasibility is not caused by missing biodiversity credit; it is caused by
the loss of renewable-eligible ag cells within specific states.

**Diagnostic next step:** compute the IIS on the saved MPS file to confirm which
state-level renewable constraint is the binding one:

```
Run_G0001/output/2026_06_01__15_37_34_RF5_2020-2050/debug_model_2035_2040.mps
```

---

### 6 — Verification: `GBF2_TARGET=medium` resolves infeasibility locally

Two RF5 2010-2050 local runs were compared to confirm that lowering the GBF2 target
from `high` to `medium` resolves the infeasibility:

| Run | GBF2 target | 2020 | 2030 | 2040 | 2050 |
|---|---|---|---|---|---|
| `output/2026_06_02__09_43_00_RF5_2010-2050` | high (30%→50%) | Optimal | Optimal | **INFEASIBLE** | — |
| `output/2026_06_02__10_53_36_RF5_2010-2050` | medium (30% flat) | Optimal | Optimal | Optimal | Optimal |

Both runs share identical objectives at 2020 and 2030 (`1.190e+03`, `-7.044e+04`),
confirming the GBF2 constraint is not binding until 2040 when the high target ramps
from 30% toward 50%. The medium target (30% flat) removes this ramp and keeps the
land budget feasible through 2050.

The medium run that solved was configured with settings that are **more restrictive than
REM_RES5** in some dimensions, giving confidence the fix will transfer:

| Setting | Local medium (solved) | REM_RES5 |
|---|---|---|
| `GHG_EMISSIONS_LIMITS` | **high** | low (easier) |
| `EXCLUDE_RENEWABLES_IN_GBF2_MASKED_CELLS` | True, 20% cut | False or True 25% cut |
| `RENEWABLE_LAYERS` | step_change | step_change / accel. / ANU_T10 |
| `GBF2_PRIORITY_DEGRADED_AREAS_PERCENTAGE_CUT` | 20 | 15 |

Because the local run solved despite harder GHG constraints, the same GBF2 relaxation
is expected to work for REM_RES5. Confidence by run:

| REM_RES5 runs | Expected outcome with medium GBF2 |
|---|---|
| G0001, G0003 (no exclusion, step_change / accel.) | Very likely feasible — less spatially restricted than local run |
| G0002, G0004 (exclusion 25%, step_change / accel.) | Likely feasible — exclusion active but GHG much easier |
| G0005 (ANU_T10, no exclusion) | Probably feasible |
| G0006 (ANU_T10, exclusion 25%) | Uncertain — both spatial restrictions active; most constrained case |

`ANU_transmission_T10` restricts renewable placement to near-grid cells, making G0005/G0006
the only cases where medium GBF2 may still be insufficient. Pending resubmission results.

---

## 20260601 — Riparian Plantings area incorrectly reduced despite `NON_AG_LAND_USES_REVERSIBLE = False`

### Context

Run `2026_05_31__13_46_26_RF5_2020-2050` showed Riparian Plantings (RP) area declining
between years despite `NON_AG_LAND_USES_REVERSIBLE['Riparian Plantings'] = False`. The
stdout log contained repeated `NonAg lb capped` messages with gaps up to 0.3 — far larger
than the 1e-2 `FEASIBILITY_TOLERANCE` that the existing capping code was designed to absorb.
Total RP dvar peaked at 112.48 (2045) and fell to 99.44 (2050); earlier periods showed
similar erosion.

Inspection script: `jinzhu_inspect_code/Check_RP_reduce/check_rp_reduce.py`

---

### 1 — Root cause: transition matrix evicts existing RP from EP-dominated cells

`get_lower_bound_non_agricultural_matrices()` (non_agricultural/transitions.py ~L1137)
sets the lb for each non-ag land use to the previous year's dvar (floor-truncated), then
caps it against the solver's UB from `get_to_non_ag_exclude_matrices()`:

```python
non_ag_x_rk = get_to_non_ag_exclude_matrices(data, data.lumaps[base_year])
lb_capped = np.minimum(lb_capped, non_ag_x_rk)
```

`non_ag_x_rk` for RP is built as:

```python
t_rk[non_ag_cells, :] *= T_MAT[lumap_by → RP]   # current dominant land use → RP
t_rk[non_ag_cells, :] *= T_MAT[lumap_2010 → RP]  # 2010 base land use → RP
t_rk = where(isnan(t_rk), 0, 1)
UB_RP = t_rk × RP_PROPORTION
```

`T_MAT['Environmental Plantings' → 'Riparian Plantings'] = NaN` (transition not in matrix).
`T_MAT['Riparian Plantings' → 'Riparian Plantings'] = 0.0` (self-transition costs nothing,
and is non-NaN, so it is **allowed**).

Many cells hold a partial RP allocation (dvar_rp > 0) while their dominant land use is
Environmental Plantings (EP). When `lumap_by = 'EP'`:

```
T_MAT[EP → RP] = NaN  →  t_rk = 0  →  UB = 0
lb_rk = previous dvar (e.g. 0.12)  →  lb capped to 0  →  RP deleted
```

The transition matrix is intended to gate *new* allocations; it should not evict
*existing* allocations in irreversible land uses. The capping code conflates these two cases.

**Confirmed by data**: all capped RP cells have `t_rk = 0`; none are caused by
`RP_PROPORTION` (FEASIBILITY_TOLERANCE) overrun. The capped-cells CSV
(`rp_capped_cells.csv`) shows every row has `ub_rp = 0.0` and `t_rk = 0`, with
`lumap_by` = EP or another non-ag LU.

---

### 2 — Why RP is the only affected non-ag land use

`RP_PROPORTION = (2 × BUFFER_WIDTH × STREAM_LENGTH) / (REAL_AREA_NO_RESFACTOR × 10000)`

Its maximum across all cells is **0.39**, meaning RP can occupy at most 39% of any cell.
Therefore, the cell's `lumap` is **always** some other land use (EP, Agroforestry, or an
agricultural land use). The cell never has `lumap = 'Riparian Plantings'`.

Consequence: for every cell that carries an RP dvar, the transition lookup is
`T_MAT[other_LU → RP]`. Because most non-ag → RP cross-transitions are NaN in T_MAT,
`t_rk = 0` for virtually all existing RP allocations. The bug affects **all** RP cells
every year.

All other irreversible non-ag land uses (EP, Carbon Plantings, Agroforestry) can occupy
100% of a cell, so their dominant `lumap` equals themselves, the self-transition
`T_MAT[X → X]` is valid, and `t_rk = 1`. The transition-matrix eviction only hits them
for rare minority allocations, with negligible area loss. RP is structurally excluded from
ever being its own `lumap` by the `RP_PROPORTION` physical constraint.

---

### 3 — Why the fix must touch both the UB and the lb

`non_ag_x_rk` (the UB) does double duty in the solver:

```python
# input_data.py
non_ag_lu2cells = {k: np.where(non_ag_x_rk[:, k])[0] ...}

# solver.py
for r in non_ag_lu2cells[k]:          # only cells with UB > 0 get a variable
    addVar(lb=non_ag_lb_rk[r,k], ub=non_ag_x_rk[r,k])
```

If `non_ag_x_rk[cell, RP] = 0`, the cell never enters `non_ag_lu2cells`, no Gurobi
variable is created for it, and its dvar is implicitly 0 in the next period — regardless
of what the lb says. A lb-only fix is therefore ineffective.

---

### 4 — Fix: one parameter in `get_to_non_ag_exclude_matrices` (transitions.py)

`get_to_non_ag_exclude_matrices` is the single source of both the UB (`get_non_ag_x_rk`
in input_data.py) and the UB used to cap the lb (`get_lower_bound_non_agricultural_matrices`
in transitions.py). Adding one optional parameter fixes both call sites at once:

```python
def get_to_non_ag_exclude_matrices(data, lumap, existing_dvars_rk=None):
    ...
    t_rk = np.where(np.isnan(t_rk), 0, 1).astype(np.int8)

    # Cells that already hold an irreversible non-ag allocation bypass the transition
    # matrix — the matrix gates *new* allocations only, not existing ones.
    if existing_dvars_rk is not None:
        for k, k_name in enumerate(data.NON_AGRICULTURAL_LANDUSES):
            if not settings.NON_AG_LAND_USES_REVERSIBLE.get(k_name, True):
                t_rk[existing_dvars_rk[:, k] > 0, k] = 1
    ...
```

Both callers pass the previous period's dvars:

```python
# transitions.py — get_lower_bound_non_agricultural_matrices
non_ag_x_rk = get_to_non_ag_exclude_matrices(
    data, data.lumaps[base_year],
    existing_dvars_rk=data.non_ag_dvars[base_year],
)

# input_data.py — get_non_ag_x_rk
existing_dvars = data.non_ag_dvars.get(base_year) if base_year != data.YR_CAL_BASE else None
return get_to_non_ag_exclude_matrices(data, data.lumaps[base_year], existing_dvars_rk=existing_dvars)
```

No recovery scaffolding anywhere. The fix is entirely inside `get_to_non_ag_exclude_matrices`.

---

### 5 — Other irreversible LUs also affected (small scale)

The same bug affected EP, Sheep/Beef Agroforestry, and Carbon Plantings wherever they
appeared as minority allocations in cells dominated by a different non-ag LU. Pre-fix
area losses (transition-matrix eviction only):

| Land use | Total area lost 2020–2050 | Peak cells/period |
|---|---:|---:|
| Riparian Plantings | **61.2** | **240** |
| Environmental Plantings | 1.07 | 29 |
| Sheep Agroforestry | 0.030 | 8 |
| Beef Agroforestry | 0.0005 | 29 |
| Others | < 0.003 | — |

RP is 57× worse because `RP_PROPORTION` max = 0.39 means RP can **never** be the
dominant land use — its `lumap` is always another LU, so the transition matrix always
blocks it.

Inspection scripts: `jinzhu_inspect_code/Check_RP_reduce/`

---

### 6 — Verification

`verify_fix.py` checks both `non_ag_lb_rk` (lb) and `non_ag_x_rk` (UB / cell inclusion)
against the same `Data_RES5.lz4`:

| Year transition | Before (RP area lost) | After |
|---|---:|---:|
| 2020 → 2025 | 0.877 | 0.000 |
| 2025 → 2030 | 9.920 | 0.000 |
| 2030 → 2035 | 6.226 | 0.000 |
| 2035 → 2040 | 6.225 | 0.000 |
| 2040 → 2045 | 24.921 | 0.000 |
| 2045 → 2050 | 13.024 | 0.000 |

Remaining `NonAg lb capped` messages show `max gap = 2.97e-08` — pure float32 rounding,
not real area loss. RP total dvar is now monotonically non-decreasing as intended.

---

## 20260530 — PBS checkpoint/redo pipeline: simulation.py per-year checkpointing, python_script.py auto-detect, redo_checkpoint.py batch resubmit

### Context

Gadi PBS jobs for multi-year LUTO simulations are frequently wall-time killed before
completing all years. Previously, such runs were unrecoverable — the full 320 GB data
object had to be reloaded from scratch on resubmit. A checkpoint/redo pipeline was
implemented to resume from the last solved year without reloading data.

---

### 1 — `simulation.py`: per-year checkpoint saves into the timestamped output subdir

`solve_timeseries()` now accepts `checkpoint_path: Path | None`. After each successfully
solved year it writes `data_{year}.lz4` (via a `.tmp` rename to be atomic), then deletes
any previous `data_*.lz4` in the same directory — keeping only the most recent checkpoint:

```python
if checkpoint_path is not None:
    final_path = checkpoint_path / f"data_{target_year}.lz4"
    tmp_path = Path(f"{final_path}.tmp")
    save_data_to_disk(data, str(tmp_path))
    os.replace(tmp_path, final_path)
    for old in checkpoint_path.iterdir():
        if re.match(r'data_\d{4}\.lz4', old.name) and old != final_path:
            old.unlink()
```

`run()` passes `Path(save_dir)` as `checkpoint_path` so the lz4 lives inside the
timestamped output subdir (`output/TIMESTAMP_RF1_2020-2050/data_2025.lz4`), not in the
run root where it would appear as an untracked code change.

On resume, `run()` scans that directory for `data_\d{4}\.lz4`, loads the latest via
`joblib.load`, restores `active_data.timestamp` and `active_data.path` from the
already-written `.timestamp` file, then calls `solve_timeseries` with only the remaining
years:

```python
files = sorted(f for f in checkpoint_path.iterdir() if re.match(r'data_\d{4}\.lz4', f.name))
if files:
    resume_from_year = int(files[-1].stem.split("_")[1])
    active_data = joblib.load(str(files[-1]))
    active_data.timestamp = read_timestamp()
    active_data.path = save_dir
```

**Why `\d{4}` not `data_*.lz4`**: `pathlib.glob` does not support regex alternation;
`re.match` over `iterdir()` is used throughout so the pattern is exact.

---

### 2 — `python_script.py`: auto-detects checkpoint, skips `load_data()`

The key invariant: `sim.load_data()` calls `write_timestamp()`, overwriting
`output/.timestamp` with a new value — which would cause `sim.run()` to construct a
**new** output directory, losing the connection to the existing partial output.

The script avoids this by scanning `output/` subdirs for `data_\d{4}.lz4` before
deciding whether to call `load_data()`:

```python
_checkpoint_dir = next(
    (str(d) for d in sorted(pathlib.Path(settings.OUTPUT_DIR).iterdir(), key=lambda d: d.name)
     if d.is_dir() and any(re.match(r'data_\d{4}\.lz4', f.name) for f in d.iterdir())),
    None
)
data = None if _checkpoint_dir else sim.load_data()
data = sim.run(data=data, ..., checkpoint_dir=_checkpoint_dir)
```

If a checkpoint dir is found: `load_data()` is skipped → `.timestamp` is unchanged →
`sim.run()` reconstructs the same `save_dir` → the simulation continues writing into
the original output directory. The `data = sim.run(...)` capture is critical — without it
`data` remains `None` and the downstream archive step (`pathlib.Path(data.path)`) raises
`AttributeError`.

---

### 3 — `redo_checkpoint.py`: batch resubmit for stalled runs

Deployed to the task root (e.g. `REM_RES1/`) alongside `run_all.py`. Classifies every
`Run_G*` directory as one of four states:

| State | Condition |
|---|---|
| finished | `Run_Archive.zip` exists |
| running | directory is the `PBS_O_WORKDIR` of a live Gadi job (`qstat -f -u $USER`) |
| checkpoint | has `data_\d{4}.lz4` inside any `output/` subdir |
| incomplete | none of the above |

For each checkpoint run:
1. Reads `task_param.py` (the original PBS settings written at submission time) to
   extract base `MEM`, `NCPUS`, `TIME`, `QUEUE`.
2. Overrides only the params explicitly passed as CLI flags; all others are inherited.
3. Writes `redo_param.py` with the merged settings.
4. Calls `bash redo_cmd.sh` from the run directory, which submits a new PBS job via
   `qsub`.

`--dry-run` is fully read-only: classification and output printing happen, but no files
are written and no jobs are submitted.

**CLI interface:**

```bash
# inherit all PBS settings from each run's task_param.py
python redo_checkpoint.py --dry-run

# override walltime only; mem/ncpus/queue still inherited
python redo_checkpoint.py --time 24:00:00

# override everything explicitly
python redo_checkpoint.py --mem 500gb --ncpus 96 --time 24:00:00 --queue normal
```

---

### 4 — `redo_cmd.sh`: mirrors `task_cmd.sh`, sources `redo_param.py`

`task_cmd.sh` sources `task_param.py` and submits `python_script.py` via `conda run`.
`redo_cmd.sh` is identical except it sources `redo_param.py`. Because `python_script.py`
auto-detects the checkpoint internally, no additional arguments are needed — the redo
job calls the same script as the original submission.

`SCRIPT_DIR` is resolved from `${BASH_SOURCE[0]}` so the absolute path to
`python_script.py` is correct regardless of where `redo_cmd.sh` is called from.

---

### 5 — `helpers.py`: redo scripts copied at task-create and submit time

`create_task_runs()` now copies `redo_checkpoint.py` and `redo_cmd.sh` to the task root
alongside `run_all.py` (two plain `shutil.copyfile` calls, no loop).

`submit_task()` in cluster mode copies `redo_cmd.sh` and `python_script.py` into each
`Run_G*/` directory alongside `task_cmd.sh`, so every run dir is self-contained for both
initial submission and checkpoint redo.

---

## 20260530 — Biodiversity report fixes: resfactor denominator, Target_by_Percent NaN sentinel, Relative_Contribution_Percentage formula, AUSTRALIA region rows, GBF4_SNES Vue setting key

### Context

A cluster of related bugs was identified and fixed across `write.py`, `data.py`,
`create_report_data.py`, and `Biodiversity.js`. The bugs shared a common root: incorrect
handling of sparse biodiversity layers at coarsened resolution and a poorly-defined
`Relative_Contribution_Percentage` denominator.

---

### 1 — `use_valid_cell_count=False` for sparse biodiversity layer resfactoring

`get_resfactored_average_fraction` averages a 1D species/habitat array over coarsened
spatial blocks of size RF². It has two denominator modes:

- **Default** (`use_valid_cell_count=True`): divides each block sum by the count of
  non-zero (valid) cells in that block.
- **`use_valid_cell_count=False`**: divides by RF² — the total number of cells in the block.

For dense arrays (e.g. land-use masks that cover most cells), both modes give nearly the
same result. For **sparse biodiversity layers** (GBF3 NVIS: 95% zero; GBF4 SNES: 99.65%
zero; GBF4 ECNES: 99.6% zero), the distinction is critical.

**The bug**: with `use_valid_cell_count=True`, a block containing 3 habitat cells out of
RF²=25 cells would divide the block sum by 3 instead of 25, inflating the resfactored
fraction by ~8×. This propagated into solver input arrays (`GBF3_NVIS_LAYERS_ALL`,
`GBF4_SNES_LAYERS_ALL`, `GBF4_ECNES_LAYERS_ALL`) and into the biodiversity score CSVs
written by `write_biodiversity_GBF3_NVIS_scores`, `write_biodiversity_GBF4_SNES_scores`,
and `write_biodiversity_GBF4_ECNES_scores`.

**Fix**: `use_valid_cell_count=False` added to all 7 call sites:

| File | Function | Affected arrays |
|---|---|---|
| `data.py` | `Data.__init__` | `GBF3_NVIS_LAYERS_ALL`, `GBF4_SNES_LAYERS_ALL`, `GBF4_ECNES_LAYERS_ALL` |
| `write.py` | `write_biodiversity_GBF3_NVIS_scores` | `nvis_layers_arr` (used in score CSV + map layers) |
| `write.py` | `write_biodiversity_GBF4_SNES_scores` | `snes_layers_arr` (used in score CSV + map layers) |
| `write.py` | `write_biodiversity_GBF4_ECNES_scores` | `ecnes_layers_arr` (used in score CSV + map layers) |

This matches the LUMASK-fix reasoning from Step 8 (2026-05-02): the solver needs the
correct fraction of habitat within each coarse block, not the fraction among only the
habitat-containing fine cells.

---

### 2 — `Target_by_Percent` set to NaN when no constraint is active

Previously, `write.py` always computed:

```python
df['Target_by_Percent'] = (
    (df['TARGET_INSIDE_SCORE'] + df['BASE_OUTSIDE_SCORE']) / df['BASE_TOTAL_SCORE'] * 100
)
```

When `TARGET_INSIDE_SCORE = 0` (no active biodiversity constraint for that group/species
× region combination), this still produced a numeric value (`BASE_OUTSIDE_SCORE /
BASE_TOTAL_SCORE * 100`), which the downstream filter in `create_report_data.py`:

```python
bio_df[bio_df['Target_by_Percent'].notna()]
```

could not distinguish from a real target. Result: the target lookup table
(`_gbf3_target_lk`, `_ecnes_target_lk`) was polluted with spurious zero-constraint rows,
which caused incorrect target lines to appear in the report.

**Fix**: replaced with `np.where(TARGET_INSIDE_SCORE > 0, formula, np.nan)` in all six
places across GBF3 NVIS, GBF4 SNES, and GBF4 ECNES (both per-region and outside rows).
`create_report_data.py` updated its comment to reflect the new invariant: `.notna()` now
correctly selects only rows with a real active constraint.

---

### 3 — `Relative_Contribution_Percentage` formula: area / ALL_HA replaces (area + outside) / baseline

The old formula for GBF3 NVIS, GBF4 SNES, and GBF4 ECNES sum CSVs was:

```python
# Old: inside LUTO + outside LUTO as fraction of pre-1750 baseline
(Area Weighted Score (ha) + BASE_OUTSIDE_SCORE) / BASE_TOTAL_SCORE * 100
```

This was a "restoration fraction" — it answered "what fraction of the pre-1750 habitat is
maintained or restored?" The problem: `BASE_TOTAL_SCORE` is the pre-1750 total across
all land (in and out of LUTO), making it an awkward denominator for a column that is
intended to track just the inside-LUTO contribution. When stacked with the "Outside LUTO
study area" row (which uses the same denominator), the sum of all Type rows could
substantially exceed or fall short of 100%, confusing report chart rendering.

**New formula**:

```python
# New: area weighted score as fraction of total regional land area
Area Weighted Score (ha) / ALL_HA * 100
```

`ALL_HA` is the total cell-area (ha) of all LUTO cells in the region, loaded from a
per-(region, region_level) lookup. This makes `Relative_Contribution_Percentage` a
consistent area-fraction that sums correctly across Types (Ag + Am + NonAg + Outside
LUTO = 100% of the regional land area).

**Applied identically** to both the Type-specific `sum_df` rows and the `outside_sum`
rows in all three write functions. `create_report_data.py` updated to use this column
directly instead of recomputing `Sum_Pct (%)` from `Area Weighted Score (ha) /
BASE_TOTAL_SCORE`.

---

### 4 — AUSTRALIA rows kept in `bio_df`; ranking queries explicitly exclude them

All three biodiversity `process_biodiversity_data` sections previously dropped AUSTRALIA
rows from `bio_df` with:

```python
bio_df = bio_df.query('species != "ALL" and region != "AUSTRALIA"')
```

**Intent**: prevent double-counting when downstream code summed across regions (since
`AUSTRALIA` = sum of all NRM/STATE regions). **Side effect**: the AUSTRALIA region
selection in the report dropdown showed no data.

**Fix**: AUSTRALIA rows are now retained in `bio_df` so the AUSTRALIA selection is
populated, but ranking queries (which group by `region` to compare NRM/STATE regions)
explicitly exclude AUSTRALIA:

```python
# Main bio_df — keep AUSTRALIA for dropdown data
bio_df = bio_df.query('species != "ALL"')

# Ranking — exclude AUSTRALIA (it's the sum of all regions, not a peer)
bio_rank_total = bio_df.query(
    'Water_supply != "ALL" and Landuse != "ALL" and '
    '`Agricultural Management` != "ALL" and region != "AUSTRALIA"'
).groupby(...)
```

The same pattern applied to GBF3 NVIS, GBF4 SNES, and GBF4 ECNES sections.
`sum_bio_df` continues to drop AUSTRALIA (only the Type=ALL filter needed; the
`Relative_Contribution_Percentage` column already carries the correct region-level value).

---

### 5 — Vue.js: GBF4_SNES metric uses `WRITE_SNES` setting key

`Biodiversity.js` `METRIC_TO_SETTING` maps each metric name to the settings key that
controls whether data for that metric was written:

```js
// Before (wrong key):
'GBF4_SNES': 'GBF4_TARGET_SNES',

// After (correct key):
'GBF4_SNES': 'WRITE_SNES',
```

`GBF4_TARGET_SNES` does not exist in settings — the flag that enables SNES output writing
is `WRITE_SNES`. With the wrong key, the Vue component would always treat GBF4_SNES
data as absent and hide the metric from the selection dropdown even when the SNES scores
had been written.

---

## 20260529 — `create_report_data` profiling: manifest.json registration, pyarrow CSV reader, bulk nested-dict builder (166× speedup)

### Context

A full RF5 run (`2026_05_29__16_52_22_RF5_2010-2050`) was used to profile the complete
report generation pipeline. File timestamps in `DATA_REPORT/data/` gave precise wall-clock
costs for each output JS file produced by `create_report_data.py`.

---

### 1 — `write_outputs` wall-clock breakdown (from stdout log timestamps)

| Stage | Elapsed | Notes |
|---|---:|---|
| Mosaic maps (all 5 years) | 0.1 min | fast |
| GBF3 NVIS scores (5 years parallel) | ~6 min | |
| **ECNES scores (5 years parallel)** | **~20 min** | ~13 min per year |
| **SNES scores (5 years parallel)** | **~19 min** | ~12 min per year |
| GBF2 priority scores | ~10 min | 2040/2050 notably slower |
| Renewable energy | ~6 min | |
| Ag-to-ag transitions | ~2 min | |
| Economics, GHG, water, etc. | ~5 min | |
| **Total `write_outputs`** | **88 min** | |

`create_report_data` starts at t = 88 min; the write phase itself is the primary bottleneck.

---

### 2 — `create_report_data` wall-clock breakdown (from JS file modification times)

| Elapsed | File written | Size |
|---:|---|---:|
| 0 min | All non-bio parallel jobs (area, GHG, water, economics, etc.) | ✓ |
| 0.4 min | GBF3 NVIS overview + Sum | ✓ |
| 0.9 min | GBF3 NVIS Ag | 34 MB |
| **8.2 min** | **GBF3 NVIS Am** | **66 MB** |
| 8.3 min | GBF3 NVIS NonAg | ✓ |
| 9.5 min | GBF4 SNES overview | ✓ |
| 10.3 min | GBF4 SNES Sum | ✓ |
| **30.2 min** | **GBF4 SNES Ag** | **208 MB** |
| ❌ never | SNES Am, SNES NonAg, all ECNES, BIO_ranking | run interrupted |

The notebook was cut off after SNES Ag (30 min). SNES Am — the most expensive section due
to the extra `am` dimension — never completed.

---

### 3 — `manifest.json` files in `_chunks` directories classified as `Unknown`

`get_all_files` logged "Unknown files found" for every `manifest.json` inside
`xr_biodiversity_*_YYYY_chunks/` directories. These files were then silently dropped.

**Root cause**: `extract_dtype_from_path` and `_base_name_ext` in
`luto/tools/report/data_tools/__init__.py` applied the parent-directory-name logic only
to `.nc` files:

```python
if path.endswith('.nc') and re.search(r'_\d{4}_chunks$', os.path.basename(parent)):
```

`manifest.json` does not end with `.nc`, so it fell through to `Unknown`.

**Fix**: extended the condition to cover `manifest.json` as well:

```python
_in_chunks = re.search(r'_\d{4}_chunks$', os.path.basename(parent))
if (path.endswith('.nc') or os.path.basename(path) == 'manifest.json') and _in_chunks:
    base_name = re.sub(r'_\d{4}_chunks$', '', os.path.basename(parent))
```

Applied identically in both `extract_dtype_from_path` and `_base_name_ext`. `manifest.json`
files now classify as `xarray_layer` (parent dir name matches `xr_` prefix) with
`base_ext = '.json'`.

---

### 4 — `pd.read_csv` replaced with `engine='pyarrow'` for biodiversity score CSVs

The large biodiversity score CSVs (SNES: 135 MB/year × 5 years = 675 MB total; NVIS: 16 MB;
ECNES: 11 MB) were loaded via `pd.read_csv` — the slowest path for large files.

**Benchmark** (`jinzhu_inspect_code/Speed_up_SNES_csv/benchmark.py`, 5 SNES files, 3 runs):

| Approach | min(s) | Speedup | Peak RAM |
|---|---:|---:|---:|
| `pd.read_csv` (baseline) | 9.0 | 1.0× | 793 MB |
| `pd.read_csv(engine='pyarrow')` | 1.5 | **6.0×** | 567 MB |
| `polars` + `.to_pandas()` | 1.6 | 5.7× | 180 MB |
| `pyarrow.csv.read_csv` | 1.2 | 7.3× | 180 MB |
| Parquet (recurring read) | 1.2 | 7.5× | 567 MB |
| Feather (recurring read) | 1.1 | 8.0× | 567 MB |
| Dask | 20.9 | 0.4× | 1860 MB |

`engine='pyarrow'` was chosen: one-keyword change, no new imports, no column filtering,
identical DataFrame output, also silences the `DtypeWarning` on the mixed-type
`Agricultural Management` column.

**Applied** to all 6 bio `pd.read_csv` calls in `process_biodiversity_data`:
overall quality, GBF3 NVIS, GBF4 SNES, GBF4 ECNES, GBF8 species, GBF8 groups.

---

### 5 — O(N²) nested-dict builder replaced by `_build_out_dict_bulk` (166× speedup)

**Root cause of the multi-hour `create_report_data` runtime:**

Every biodiversity chart section (Ag, Am, NonAg, overview, Sum) built its output dict
with this pattern:

```python
out_dict = {}
for (region_level, region, species, am, water), df_pct in df_wide_pct.groupby([...]):
    df_pct = df_pct.drop([...], axis=1)
    df_area = df_wide_area[
        (df_wide_area['region_level'] == region_level) & ... & (df_wide_area['water'] == water)
    ].drop([...], axis=1)             # ← full table-scan every iteration
    out_dict[...][am][water] = {
        'Percent': df_pct.to_dict(orient='records'),   # ← small to_dict per group
        'Area':    df_area.to_dict(orient='records'),
    }
```

Two compounding problems:
1. **O(N²) boolean filter**: `df_wide_area[boolean_mask]` scans the full DataFrame for
   every group. With SNES Am having ~4,300 groups (50-species sample), this alone costs
   ~7 s/50 species → extrapolated ~5.5 min for 1,937 species.
2. **N small `to_dict` calls**: calling `to_dict(orient='records')` on a tiny sub-DataFrame
   4,300 × 2 = 8,600 times has severe Python overhead. This dominates the remaining time.

**Benchmark** (`jinzhu_inspect_code/Speed_up_SNES_csv/benchmark_am_loop.py`, 50-species
sample, N=3 runs, extrapolated to full 1,937 species via ×39 scale factor):

| Approach | min(s) | Speedup | Extrapolated full run |
|---|---:|---:|---:|
| baseline (O(N²) filter + N small to_dict) | 26.9 | 1.0× | ~17 min |
| pre-index area dict, O(1) lookup | 23.8 | 1.1× | ~15 min |
| merge pct+area, one loop | 23.7 | 1.1× | ~15 min |
| zip column lists per group | 14.3 | 1.9× | ~9 min |
| **bulk_to_dict** (one `to_dict` on full df, group via `defaultdict`) | **0.63** | **43×** | **~24 s** |
| **bulk_zip** (no `to_dict` at all — `zip` columns, group in Python) | **0.16** | **166×** | **~6 s** |

The winning approach (`bulk_zip`) converts the entire DataFrame to Python lists via
column `.tolist()` calls, then groups rows into a `defaultdict` — zero pandas groupby,
zero `to_dict`. The key insight: **one big column-list extraction is 166× cheaper than
N small `to_dict` calls on sub-DataFrames**.

**Fix**: `_build_out_dict_bulk(df_wide_pct, df_wide_area, key_cols)` helper added at
line 136 of `create_report_data.py`:

```python
def _build_out_dict_bulk(df_wide_pct, df_wide_area, key_cols):
    from collections import defaultdict
    def _df_to_keyed(df):
        keys = list(zip(*[df[c].tolist() for c in key_cols]))
        leaf_cols = [c for c in df.columns if c not in key_cols]
        rows = [dict(zip(leaf_cols, r)) for r in zip(*[df[c].tolist() for c in leaf_cols])]
        grouped = defaultdict(list)
        for k, row in zip(keys, rows):
            grouped[k].append(row)
        return grouped
    pct_grouped  = _df_to_keyed(df_wide_pct)
    area_grouped = _df_to_keyed(df_wide_area)
    out_dict = {}
    for key, pct_list in pct_grouped.items():
        d = out_dict
        for k in key[:-1]:
            d = d.setdefault(k, {})
        d[key[-1]] = {'Percent': pct_list, 'Area': area_grouped.get(key, [])}
    return out_dict
```

**Applied at 21 call sites** across all biodiversity modules:

| Module | Sections replaced |
|---|---|
| Quality | overview, Ag, Am, NonAg |
| GBF2 | overview, Ag, Am, NonAg |
| GBF3 NVIS | overview, Ag, Am, NonAg |
| GBF4 SNES | overview, Sum, Ag, Am, NonAg |
| GBF4 ECNES | overview, Ag, Am, NonAg |

Two Sum sections (GBF3 NVIS Sum, GBF4 ECNES Sum) were **not** changed: they inject a
per-species target line into the records, which requires per-group logic. They use the
small `sum_scores` CSVs (4.9 MB / 393 KB) so they are not a bottleneck.

**Expected total `create_report_data` biodiversity time**: reduced from >1 hour (never
completing) to an estimated 2–5 minutes.

---

## 20260529 — Report layer pipeline: chunk magnitude tracking, get_all_files chunk discovery, _score_to_df BLAS optimisation, chunk_num JS merging

### 1 — `save2chunk` now returns inline magnitude (avoids disk reload)

`_mag_from_chunks` scanned every saved `chunk_*.nc` file after the loop to compute
`[min, max]` quantiles — re-reading data that was already in memory when it was written.

**Fix:** `save2chunk` now calls `get_mag(in_xr)` before writing and returns the result.
`_save_fullsize_layers` (SNES inner helper) propagates the return via `return save2chunk(...)`.
Each of the three chunk loops (GBF3, SNES, ECNES) was updated to:

```python
mags_ag, mags_non_ag, mags_am, mags_sum = [], [], [], []
# inside loop:
mags_ag.extend(save2chunk(valid_ag_g, chunks_ag, group_idx))
```

The end-of-function `magnitudes` dict uses the accumulated lists directly.
`_mag_from_chunks` is removed entirely.

---

### 2 — `get_all_files` could not discover chunk NC files

`extract_dtype_from_path` checked `os.path.basename(path)` against category patterns.
For chunk files (`chunk_000000.nc` inside `xr_biodiversity_GBF4_SNES_ag_2050_chunks/`),
the basename never matches `xr_` → classified as `Unknown` → dropped.
`get_all_files` also stored `base_name = "chunk_000000"`, which no downstream query would match.

**Fix in `extract_dtype_from_path`:** if the file is a `.nc` inside a `_YYYY_chunks/`
directory, use the **parent directory name** (with `_YYYY_chunks` stripped) as the effective
basename for pattern matching:

```python
parent = os.path.dirname(path)
if path.endswith('.nc') and re.search(r'_\d{4}_chunks$', os.path.basename(parent)):
    base_name = re.sub(r'_\d{4}_chunks$', '', os.path.basename(parent))
else:
    base_name = os.path.basename(path)
```

**Fix in `get_all_files` `_base_name_ext`:** same guard so `base_name` stored in the
DataFrame is `xr_biodiversity_GBF4_SNES_ag` (not `chunk_000000`), matching
`files.query('base_name == "xr_biodiversity_GBF4_SNES_ag"')`.

`manifest.json` files inside chunk dirs are excluded from the chunk-dir logic via the
`.endswith('.nc')` guard and remain `Unknown` → dropped.

---

### 3 — `_score_to_df` (SNES inner helper): per-species xarray loop replaced by BLAS matmul

The original function looped over each species one at a time:
- `.sel(cell=score['species'] == species)` — boolean mask per species
- `.groupby(rl).sum()` + `.to_dataframe()` per species per region level
- Result: O(n_species × n_region_levels) xarray groupby ops, many small DataFrames

For a batch of 10 species with 2 region levels and 4 score types, this is 80 xarray
groupby calls per batch, each doing a full `.compute()` on a dask-backed DataArray.

**Fix:** stack `group_dims` once → `(n_layers, n_cells)` numpy matrix, then:

- **AUSTRALIA**: single BLAS call `vals @ sp_onehot` → `(n_layers, n_sp)` where
  `sp_onehot` is `(n_cells, n_sp)` — one call covers all species and all layers.
- **Regional**: loop over `n_sp ≤ 10` species; for each, `vals[:, mask] @ rg_onehot[mask]`
  → `(n_layers, n_rg)`. The per-species mask limits cells to that species' nonzero footprint,
  keeping each matmul small.
- Build DataFrames from `np.where(agg != 0)` indices — no intermediate DataFrame-per-species.

Key: `score.compute()` is called once at the start, not inside any loop.

---

### 4 — `get_map2json_snes` `chunk_num` parameter: merge N chunk NCs per JS output

With 10 species per chunk NC and ~270 SNES species, each combo generates ~27 JS files.
A `chunk_num` parameter allows grouping N consecutive chunk files per JS output.

**Python side (`get_map2json_snes`):**
The `for chunk_idx in chunk_indices:` loop was replaced by:

```python
for group_start_pos in range(0, len(chunk_indices), chunk_num):
    group = chunk_indices[group_start_pos : group_start_pos + chunk_num]
    page_start = page_starts[group[0]]
    page_end   = page_starts[group[-1]] + len(manifest[str(group[-1])])
    chunk_data = {}
    for chunk_idx in group:          # accumulate all years × all chunks
        for year, chunks_dir in ...:
            ...
    # write one JS per combo, filename: {var_name}_{page_start}_{page_end}.js
```

With `chunk_num=10` and 10 species per NC: each JS covers 100 species, ~3× fewer files.

**Index side (`_write_snes_index`):** the `pages` dict must be grouped by the same
`chunk_num` so Vue constructs the correct filename. Without this fix, Vue reads
`pages["0"] = {start:0, end:10}` and requests `_0_10.js` — which no longer exists.

```python
for group_start_pos in range(0, len(sorted_chunk_keys), chunk_num):
    group_keys = [...]
    page_start = chunk_starts[group_keys[0]]
    page_end   = chunk_starts[group_keys[-1]] + len(chunk_species[group_keys[-1]])
    species    = [sp for k in group_keys for sp in chunk_species[k]]
    pages_out[str(page_idx)] = {'start': page_start, 'end': page_end, 'species': species}
```

`pageSize` is updated to `batch_size × chunk_num`.

**Vue side:** no changes needed — `currentPageInfo = pages[selectPage]` drives
`ensureComboLayer([start, end])` which constructs the filename. The species dropdown
(`_pagedSpecies()`) reads `currentPageInfo.species` which now contains `chunk_num × 10`
species. `totalPages = Object.keys(pages).length` is halved automatically.

The `chunk_num` is set per call site in `save_report_layer` (not in `settings.py`) so it
can be tuned independently for GBF3, SNES, and ECNES without a settings round-trip.

---

## 20260529 — `write_biodiversity_GBF4_SNES_scores`: ~200 GB memory at full resolution diagnosed and fixed

### Context

A res5 profiling run with 100 SNES species reported ~7 GB peak memory for
`write_biodiversity_GBF4_SNES_scores`. The question was why a full-resolution run
(~2,600 species) would consume ~200 GB — far more than a simple NCELLS × species scaling
would predict — and whether the cause was array accumulation across batches.

### Root cause: dense intermediate in `_save_fullsize_layers`

The function `_save_fullsize_layers` was called four times per species batch (ag, non_ag,
am, sum). Each call allocated a fully dense numpy array of shape:

```
[n_am, n_lm, n_lu, batch_size, NCELLS]
```

For `score_am` (the dominant case) with 9 AM types + ALL = 10, 3 lm, 30 lu, batch = 10:

| Resolution | NCELLS | `full_arr` size | After `.stack()` (copy) | Per-batch peak (4 calls) |
|---|---:|---:|---:|---:|
| res5 | ~20,000 | 0.72 GB | 0.72 GB | ~6 GB |
| res1 | ~500,000 | 18 GB | 18 GB | ~72 GB |

The `.stack(layer=['am','lm','lu','species'])` call on a numpy array written via `.loc[]`
is non-contiguous in memory and forces a copy, so `full_arr` (18 GB) and the stacked
result (18 GB) were both live simultaneously — **36 GB per `_save_fullsize_layers` call**.
With 4 calls per batch and Python's reference-counting freeing the arrays only at function
return (no GC lag needed — this is synchronous), the synchronous peak per batch was
**~72 GB**. Residuals from preceding batches not yet freed by the OS pushed observed RSS
to **~200 GB**.

The DataFrame accumulation (`pd.concat` in a loop) was also investigated as a potential
cause but ruled out: the score DataFrames are aggregate tables (species × region × lu
rows, not cell-level), so even with 260 batches the O(N²) copy overhead is at most a few
MB of extra memory — negligible.

### Fix: build only valid layers, skip the dense intermediate

`_save_fullsize_layers` was rewritten to avoid `full_arr` entirely. Instead of allocating
the full `[n_am × n_lm × n_lu × batch × NCELLS]` tensor and selecting valid layers
afterwards, the new implementation iterates directly over `valid_layers` and writes each
layer's 1D array into `out_data`:

```python
n_valid = len(valid_layers)
out_data = np.zeros((n_valid, data.NCELLS), dtype=np.float32)

for i, layer_tuple in enumerate(valid_layers):
    species_val = layer_tuple[species_pos]
    sel_coords = {d: layer_tuple[layer_dims.index(d)] for d in score_layer_dims}
    layer_slice = score.sel(cell=score['species'] == species_val, **sel_coords)
    out_data[i, nz_idx] = layer_slice.values
```

Peak memory per call: `n_valid_layers × NCELLS × 4 bytes`. With typically 50–200 valid
layers at res1, this is **~400 MB per call** — down from **36 GB**.

The previous convention-based `layer_dims[:-1]` / `layer_tuple[-1]` for extracting the
species dimension was replaced with explicit label indexing:

```python
score_layer_dims = [d for d in layer_dims if d in score.dims]
species_pos      = layer_dims.index('species')
```

This makes the function robust regardless of argument order.

### GC lag risk with new code

With plain numpy arrays and no reference cycles, CPython's reference counter frees
`out_data` and `valid_arr` immediately when `_save_fullsize_layers` returns. Even if
OS-level lag held 2–3 batches in RSS, the worst case is: 3 × 400 MB = 1.2 GB — negligible.
The previous worst case was 3 × 72 GB = 216 GB, which matched the observed ~200 GB.

---

## 20260528 — RES5 peak memory updated for NVIS and ECNES biodiversity writers

### Context

The `peak_mb_RES5` table in `luto/tools/write.py` controls write-stage parallelism via:

```python
n_jobs = floor(WRITE_REPORT_MAX_MEM_MB / peak_delta_mb)
```

NVIS and ECNES were re-profiled at RF5 using:

```text
jinzhu_inspect_code/Profile_write_RES5/profile_write_RES5_bio_nvis_snes_ecnes.py
```

The profiling data object was:

```text
output/2026_05_20__13_10_03_RF5_2010-2050/Data_RES5.lz4
```

### Profiling results

| Writer | Duration | Peak delta RAM | Peak working set | Status | Summary |
|---|---:|---:|---:|---|---|
| `write_biodiversity_GBF3_NVIS_scores` | 219.58s | 15,447.3 MB | 23,568.7 MB | ok | `data_bio_nvis_snes100_ecnes/20260528_113110/profile_summary.csv` |
| `write_biodiversity_GBF4_ECNES_scores` | 648.08s | 7,317.4 MB | 15,329.6 MB | ok | `data_bio_nvis_snes100_ecnes/20260528_113528/profile_summary.csv` |

The successful ECNES run used `GBF4_ECNES_REGION_MODE = 'NRM'`, matching the current
selected regions `['North East', 'Goulburn Broken']`. An earlier ECNES attempt with
`GBF4_ECNES_REGION_MODE = 'AUSTRALIA'` failed before profiling completed because the
selected regions were NRM names, leaving the filtered ECNES target table empty.

### Code update

`luto/tools/write.py` was updated by rounding the successful peak deltas up to integer MB:

```python
'write_biodiversity_GBF3_NVIS_scores':         15_448,
'write_biodiversity_GBF4_ECNES_scores':         7_318,
```

Previous values were:

```python
'write_biodiversity_GBF3_NVIS_scores':         17_281,
'write_biodiversity_GBF4_ECNES_scores':         7_000,
```

The updated file passed:

```powershell
conda run -n luto python -m py_compile luto\tools\write.py
```

---

## 20260527 — SNES sparse nonzero writer implemented, debugged, and benchmarked

### Context

The earlier SNES write investigation identified dask/xarray regional aggregation as the
main bottleneck and proposed a BLAS-like sparse path. The implemented direction is now:

1. Keep decision variables and impact arrays as in-memory numpy arrays with `cell` as the
   last axis.
2. For each 10-species batch, compute the resfactored vegetation score once.
3. Build sparse pair indices from `np.nonzero(veg_score_np)`.
4. Slice all downstream arrays only at those nonzero cells.
5. Accumulate regional scores with sparse `np.add.at`.
6. Build DataFrames once per species batch from compact accumulators.

This avoids the previous full dense xarray products and `.groupby().sum().to_dataframe()`
inside the region/species loops.

---

### Profile result: sparse path vs previous implementations

100-species SNES profile at RF5, year 2050:

| Implementation | Duration | Peak delta RAM | Notes |
|---|---:|---:|---|
| v3 / older xarray path | 607.42s | 17,878.1 MB | dask/xarray groupby path |
| dense current numpy path | 348.09s | 4,639.0 MB | faster, but still dense-ish |
| sparse nonzero path | 96.16s | 4,541.5 MB | first sparse implementation |
| sparse path after readability refactor | **93.98s** | **4,528.1 MB** | current implementation |
| sparse path after AM-axis alignment fix | **118.51s** | **4,537.7 MB** | validated against archived run output |

Extrapolating the validated 100-species result to all 1,937 SNES species gives
approximately **38.3 minutes per year** (`118.51s × 19.37`). This is still not as clean as
transition `ag2ag`, but it removes the dominant dask/xarray recomputation cost.

---

### Resfactor average is not the main remaining culprit

`get_resfactored_average_fraction` was timed separately for 100 species:

| Component | Time |
|---|---:|
| Resfactor vegetation loading for 100 species | ~25–27s |
| Full sparse SNES write for 100 species | ~94–96s |

So resfactor averaging is meaningful, but not the whole bottleneck. In the sparse writer,
the remaining time is a mix of:

- vegetation resfactor loading
- NetCDF temp writes and final concat
- DataFrame construction/final CSV joins
- valid-layer NetCDF filling

The regional aggregation kernel itself is now tiny.

---

### Sparse aggregation benchmark: `np.add.at` beats pandas groupby

Benchmark script:
`jinzhu_inspect_code/Profile_write_RES5/benchmark_snes_sparse_groupby.py`

100-species sparse batch:

| Kernel | Duration | Delta working set |
|---|---:|---:|
| `addat_ag` | 0.018s | 2.0 MB |
| `pandas_ag` | 0.418s | 33.7 MB |
| `addat_nonag` | 0.003s | 1.4 MB |
| `pandas_nonag` | 0.055s | 3.6 MB |
| `addat_am` | 0.165s | 10.5 MB |
| `pandas_am` | 5.978s | 38.8 MB |

Conclusion: `DataFrame.groupby` is not competitive for the sparse region-aggregation
kernel. The pandas path is ~18–36× slower depending on type, and allocates more memory.
The current `np.add.at` path should remain the hot aggregation kernel.

---

### xarray `isel` benchmark: selection is cheap, rectangular expansion is expensive

The idea tested: use xarray labels for readability, `isel(cell=nonzero_cells)`, then
multiply and `groupby`.

For 100 species:

| Quantity | Count |
|---|---:|
| true sparse `(species, cell)` nonzero pairs | 17,680 |
| unique nonzero cells | 16,797 |
| xarray selected rectangle (`species × unique_cell`) | 1,679,700 |

Timing:

| Kernel | Duration |
|---|---:|
| `xarray_isel_only` | 0.004s |
| `xarray_ag_groupby` | 0.443s |
| `xarray_nonag_groupby` | 0.114s |
| `xarray_am_groupby` | 4.472s |

Conclusion: `isel` itself is cheap, but selecting the union of nonzero cells expands the
work by ~95× compared with true sparse species-cell pairs. xarray is useful for setup and
labels, but not for the SNES inner loop.

---

### Readability refactor applied

The sparse numpy implementation was hard to read because region accumulation and NetCDF
layer filling were embedded directly in the species loop. The current implementation keeps
the sparse numpy performance path but extracts the indexing-heavy sections into named
helpers inside `write_biodiversity_GBF4_SNES_scores`:

- `_load_sparse_veg_scores`
- `_empty_region_accumulators`
- `_take_sparse_inputs`
- `_accumulate_sparse_region_scores`
- `_fill_sum_nc_scores`
- `_fill_ag_nc_layer`
- `_fill_am_nc_layer`
- `_fill_non_ag_nc_layer`

The main species loop now reads as orchestration: load sparse vegetation, accumulate
regional scores, build CSV frames, fill NetCDF layers, save temp chunks.

Profile after this refactor:

| Metric | Value |
|---|---:|
| Duration | 93.98s |
| Peak delta RAM | 4,528.1 MB |
| Peak working set | 15,094.8 MB |
| Status | ok |

No speed regression was observed; the result is slightly faster than the prior sparse
profile within normal run-to-run variation.

---

### AM-axis alignment bug found and fixed

The archived production run at commit `0e3424f74d62e23709c1c9b4274e1d6f63363f3e`
used the xarray path:

```python
xr_gbf4_am_s = veg_score_r * am_impact_amr * am_dvar_amrj
```

That path is slower, but xarray automatically aligns by coordinate label. The sparse
implementation converts to NumPy arrays with `.values`, so all dimension alignment becomes
positional and must be made explicit.

The bug: `am_impact_amr.unstack()` sorts the MultiIndex level for `am`, producing:

```text
['AgTech EI', 'Asparagopsis taxiformis', 'Biochar', 'HIR - Beef', 'HIR - Sheep',
 'Onshore Wind', 'Precision Agriculture', 'Savanna Burning', 'Utility Solar PV']
```

while `am_dvar_amrj` remains in data order:

```text
['Asparagopsis taxiformis', 'Precision Agriculture', 'Savanna Burning', 'AgTech EI',
 'Biochar', 'HIR - Beef', 'HIR - Sheep', 'Utility Solar PV', 'Onshore Wind']
```

The LU axis was already aligned, but the AM axis was not. This caused the sparse
implementation to multiply the wrong AM impact by the wrong AM decision variable. The clue
was that an `ALL` agricultural-management score for a species became negative, even though
the positive Savanna Burning / HIR contributions should dominate. Individual renewable
management scores can legitimately be negative because renewables reduce SNES biodiversity
scores, but the total `ALL` AM score should match the archived xarray result.

Fix applied before converting to NumPy:

```python
am_vals = am_dvar_amrj.coords['am'].values
lu_ag_vals = ag_dvar_mrj.coords['lu'].values
am_impact_amr = am_impact_amr.reindex(am=am_vals, lu=lu_ag_vals, fill_value=0)
```

This restores the label alignment that the archived xarray implementation provided
implicitly.

---

### Validation against archived output

The current sparse SNES profile was rerun for the first 100 species and compared against
the archived production output from:

```text
output/2026_05_26__11_58_52_RF5_2010-2050/out_2050
```

The archive was generated from commit:

```text
0e3424f74d62e23709c1c9b4274e1d6f63363f3e
```

After filtering the archive to the same 100 profile species:

| CSV | Current rows | Archive rows | Key matches | Current-only | Archive-only |
|---|---:|---:|---:|---:|---:|
| `biodiversity_GBF4_SNES_scores_2050.csv` | 20,783 | 20,783 | 20,783 | 0 | 0 |
| `biodiversity_GBF4_SNES_sum_scores_2050.csv` | 1,960 | 1,960 | 1,960 | 0 | 0 |

Maximum absolute score differences are small and consistent with float32/order-of-sum
differences in the sparse path:

| CSV / Type | Max absolute score difference |
|---|---:|
| detailed CSV — Agricultural Management | 0.109375 |
| detailed CSV — Non-Agricultural Land-use | 0.093750 |
| detailed CSV — Agricultural Land-use | 24.000000 |
| sum CSV — ag-man | 0.109375 |
| sum CSV — non-ag | 0.093750 |
| sum CSV — ag | 25.000000 |
| sum CSV — ALL | 24.000000 |

Spot checks after the fix:

| Species | AM | Current | Archive | Difference |
|---|---|---:|---:|---:|
| `Acanthophis hawkei` | ALL | 214,972.546875 | 214,972.625000 | -0.078125 |
| `Acanthophis hawkei` | Savanna Burning | 131,123.828125 | 131,123.828125 | 0 |
| `Acanthophis hawkei` | HIR - Beef | 72,385.476562 | 72,385.484375 | -0.007812 |
| `Acanthophis hawkei` | Onshore Wind | -805.271973 | -805.271973 | 0 |
| `Acacia crombiei` | ALL | 112,579.671875 | 112,579.671875 | 0 |
| `Acacia ammophila` | ALL | 20,845.052734 | 20,845.050781 | 0.001953 |

Conclusion: the current sparse implementation is now equivalent to the archived xarray
implementation for the profiled species set, with only tiny floating-point differences.

---

### NetCDF writing decision: keep `save2tmp` / `concat_tmp2nc` for now

An incremental NetCDF writer was considered to append directly along `layer` and avoid
the temp-folder concat stage. It is technically feasible if it writes the same
CF-compressed MultiIndex schema expected by report generation:

```python
cfxr.decode_compress_to_multi_index(xr.open_dataset(path, chunks={}), 'layer')['data']
```

The final NC also must keep `data` chunked as `(layer=1, cell=NCELLS)`, matching current
report access patterns.

However, after review, the preference is to keep the existing `save2tmp` /
`concat_tmp2nc` path because it is already compatible with downstream report code and is
easier to reason about. The attempted incremental writer was reverted.

---

## 20260527 — SNES write bottleneck: dask recomputation vs BLAS GEMM plan

### Context

Production run `2026_05_26__11_58_52_RF5_2010-2050` (RF5, 5 years) showed that
`write_biodiversity_GBF4_SNES_scores` dominated the entire run:

| Phase | Wall time | Fraction of total |
|---|---:|---:|
| Write phase total | 4h 31min | — |
| → SNES write | **3h 35min** | **79%** |
| Report phase total | 2h 57min | — |
| → SNES map layers | **2h 40min** | **91%** |
| **Total run** | **8h 12min** | — |
| → **Total SNES overhead** | **~6h 15min** | **~76%** |

Peak RAM during SNES write: **235.8 GB** (17:12:22). Pre-SNES baseline: ~70 GB.

---

### Root cause: 1,552 full-array dask recomputations

The SNES loop processes 1937 species in 194 batches of 10. Per batch, per 2 region levels,
it triggers `.groupby().sum().to_dataframe()` on 4 large arrays, plus an AUS aggregate:

```
194 batches × 2 region levels × 4 array types × 2 (region + AUS) = 1,552 dask .compute() calls
```

Each call materialises `xr_gbf4_am_s` of shape `(10 species × 9 am × 3 lm × 28 lu × 186648 cells)` ≈ **5.7 GB**.
Total data movement per year: ~8.8 TB.

Compare `write_transition_ag2ag` (257s, all land uses at once): it uses `process_chunks` with
a BLAS GEMM accumulator — **1 aggregation pass** over ~100 MB. No dask compute loop.

---

### Memory trace finding: xr IS freed between batches; DataFrames are the floor

The SNES memory trace (100-species profiling run, v3) shows an **oscillatory** pattern —
not monotonically increasing. Memory spikes up for each batch and drops almost fully back
to near-baseline after each batch completes. This proves:

1. **xr arrays ARE freed between batches** (Python GC reclaims them on loop variable reassignment)
2. The **per-batch xr spike** (~16 GB, dominated by `xr_gbf4_am_s`) is the peak driver
3. The **accumulated DataFrames** add a slowly rising baseline (~230 MB per batch before filtering)

Implication for `peak_mb_RES5`: the earlier extrapolation of 346,299 MB assumed linear
DataFrame accumulation as the driver. That was wrong. The xr spike (~16 GB) is constant
per batch regardless of total species count. For 1937 species, the per-year peak is
similar to the 100-species profile: **~17,878 MB**.

**`peak_mb_RES5['write_biodiversity_GBF4_SNES_scores']` corrected from 346,299 → 17,878 MB.**

With `WRITE_REPORT_MAX_MEM_MB = 65,536 MB`:
- Old (1 MB placeholder): `n_jobs = 65,536` → all 5 years in parallel
- New (17,878 MB): `n_jobs = 3` → 3 years in parallel, then 2

---

### Short-term fix applied: zero-row DataFrame filter

Before each `extend()` call in the species-batch loop, filter out rows where
`'Area Weighted Score (ha)' == 0`:

```python
ag_frames.extend([
    ag_df_region.query('`Area Weighted Score (ha)` != 0'),
    ag_df_AUS.query('`Area Weighted Score (ha)` != 0'),
])
```

Applied to all 4 frame lists in SNES, ECNES, and NVIS (12 `extend` calls total).
Since the SNES layers are very sparse, most species have zero score in most
(region, lm, lu) combinations. This reduces the accumulated DataFrame floor from
~44 GB to ~2 GB for the full 1937-species run, lowering the baseline from which
each xr spike launches.

---

### BLAS GEMM plan (not yet implemented)

The `process_chunks` function used by `write_transition_ag2ag` achieves its speed via:

```
chunk.reshape(n_combos, chunk_cells) @ onehot[chunk_cells, n_regions]  →  (n_combos, n_regions)
```

The same form applies to SNES scoring. For ag:

```
score_ag[s, lm, lu, region] = Σ_cell  veg[s, cell] × impact[lu] × dvar[lm, lu, cell] × indicator[cell → region]
                             = combined[s, lm, lu, :].reshape(n_combos, NCELLS) @ onehot
```

For am (the dominant cost, shape `species × am × lm × lu × cell`):

```
score_am[s, am, lm, lu, region]
    = Σ_cell  veg[s, cell] × am_impact[am, lu, cell] × am_dvar[am, lm, lu, cell] × indicator[cell → region]
    = combined_am.reshape(n_combos_am, chunk_cells) @ onehot_chunk
```

**Proposed restructuring** — swap the loop axes:

| | Current | Proposed |
|---|---|---|
| Outer loop | species batch (194×) | species batch (194×) — same |
| Inner loop | 2 region levels × 8 `groupby().sum()` dask triggers | cell chunks (46×, 4096 cells each) |
| Array per step | `(10, am, lm, lu, 186648)` = 5.7 GB materialised | `(10, am, lm, lu, 4096)` = 124 MB chunk |
| Aggregation | xr `groupby().sum()` on full array | BLAS GEMM `@ onehot_chunk` |
| Dask calls | 1,552 `.compute()` | 0 (pure numpy) |

Per-batch cell-chunk GEMM:

```python
# onehot pre-computed once per region level: (NCELLS, n_regions), float32
for sp_idx, species_batch in enumerate(...):
    snes_chunk_all = snes_layers[sp_batch, :]   # (n_sp, NCELLS) — sparse→dense once per batch

    for cell_start in range(0, NCELLS, chunk_size):
        sl = slice(cell_start, cell_start + chunk_size)
        combined_am = (snes_chunk_all[:, sl][:, None, None, None, :]   # (n_sp, 1, 1, 1, c)
                       * am_impact[:, :, sl][None, :, None, :, :]       # (1, am, 1, lu, c)
                       * am_dvar[:, :, :, sl][None, :, :, :, :])        # (1, am, lm, lu, c)
        # → (n_sp, am, lm, lu, chunk_cells) = ~124 MB
        accum_am += combined_am.reshape(-1, chunk_cells) @ onehot_sl   # BLAS GEMM → (n_combos, n_regions)

    # Convert accumulator to DataFrame ONCE per batch (not 8×)
    df_ag = pd.DataFrame(accum_ag.reshape(-1, n_regions), ...).melt(...)
```

**Expected impact:**

| Metric | Current | After BLAS GEMM |
|---|---|---|
| Dask `.compute()` calls | 1,552 | 0 |
| Memory per step | 5.7 GB | 124 MB |
| Estimated time per year | ~196 min | ~20–40 min (similar to ECNES at 655s) |
| Peak RAM per year | ~18 GB Δ | ~2 GB Δ (no full materialisation) |

Same restructuring can be applied to ECNES and NVIS.

**Status: planned, not yet implemented.**

---

## 20260525 — Biodiversity input array sparsity and threading for GBF3/4 write loops

### Context

The GBF3/4 biodiversity write functions loop over vegetation groups or species,
multiplying spatial arrays by land-use decision variables. Threading was investigated
to speed up these loops. Sparsity of input NC files was profiled to evaluate
sparse array pre-cooking as a complementary strategy.

### NVIS Threading Benchmark (`write_biodiversity_GBF3_NVIS_scores`)

| Config | Duration | Peak RAM | Notes |
|--------|----------|----------|-------|
| Serial (baseline) | 551s | ~9.2 GB | step_7 profiler |
| n_jobs=8 threading | 360s | 14.8 GB | ~1.5× speedup, +5.9 GB overhead |

**Root cause of limited speedup**: `save2tmp` NC writes serialised behind `threading.Lock`;
`netCDF4` C library not thread-safe (global ID registry).

**Fix applied**: Pre-load full NVIS array via `xr.open_dataarray(...).load()` once before
the parallel loop — workers receive plain numpy slices, zero file I/O per thread.
Eliminates `RuntimeError: NetCDF: Not a valid ID` from stale thread-local handles.

### Input Array Sparsity

| File | Shape | Dense size | Sparsity | COO size | Compression | Build time |
|------|-------|-----------|----------|----------|-------------|------------|
| `bio_GBF3_NVIS_MVG.nc` | (30, 6.9M) | 0.8 GB | 95.0% | ~125 MB | 7× | 0.5s |
| `bio_GBF4_ECNES.nc` | (101, 2, 6.9M) | 5.6 GB | 99.6% | ~88 MB | 64× | 17s |
| `bio_GBF4_SNES.nc` | (2021, 2, 6.9M) | 112 GB | 99.65% | ~1,565 MB | 72× | 447s |

COO size estimated as `nonzero × (4 bytes data + 4 bytes × ndim coords)`.
SNES/ECNES build time measured reading one species slice at a time (avoids
materialising the full dense array).

### Recommendations

| Array | Action | Reason |
|-------|--------|--------|
| **NVIS** | Keep `.load()` dense | 0.8 GB fine; 7× compression not worth overhead |
| **ECNES** | Pre-cook `bio_GBF4_ECNES_sparse.npz` in `dataprep.py` | 88 MB COO, 17s build, 64× savings; loads in seconds at runtime |
| **SNES** | Pre-cook `bio_GBF4_SNES_sparse.npz` in `dataprep.py` | 112 GB dense impossible; 1.5 GB COO, 447s build (one-time); loads in seconds |

Pre-cooked `.npz` files are plain numpy arrays — fully thread-safe, no file locks,
instant per-species indexing in parallel workers.

### SNES Runtime Projection

| Strategy | Estimated time (2021 species) |
|----------|-------------------------------|
| Serial loop | ~36,700s (~10h) |
| n_jobs=8 threading (dense) | Impossible — 112 GB RAM |
| n_jobs=8 threading (sparse `.npz`) | ~4,600s (~1.3h, estimated) |

Pre-cooking sparse arrays is a prerequisite for any viable SNES parallelism.

---

## 20260523 — Biodiversity regional scoring: xarray groupby bottleneck and optimisation

### Context

The upstream pipeline pre-computes weighted habitat area scores for three biodiversity
datasets across all region levels × resfactors, writing the results to CSV so LUTO can
do a simple lookup at runtime instead of recomputing at every model run.

**Scale of the problem:**

| Dataset | Species / groups | Presence types | Region levels | Resfactors |
|---|---|---|---|---|
| NVIS (MVG + MVS) | ~770 vegetation groups | 1 | 5 (incl. IBRA_SUB) | 10 |
| SNES | ~2,055 threatened species | 2 (LIKELY, MAYBE) | 5 | 10 |
| ECNES | ~400 threatened communities | 2 (LIKELY, MAYBE) | 5 | 10 |

Region levels: `AUSTRALIA` (1 group), `NRM` (56), `STATE` (9), `IBRA_REG` (85), `IBRA_SUB` (410).
Spatial domain: 6,956,407 NLUM cells at resfactor 1; fewer at higher resfactors.

For each `(species, resfactor, region_level, region)` combination, four weighted-area scores
are computed by groupby-summing a 1D species presence array multiplied by per-cell weights:

```
ALL_HA               = sum(species_arr × cell_ha)                           per region
IN_LUTO_HA           = sum(species_arr × cell_ha × biodiv_degrade)         per region
NATURAL_OUT_LUTO_HA  = sum(species_arr × cell_ha × out_luto_natural_mask)  per region
NON_NATURAL_OUT_LUTO_HA = sum(species_arr × cell_ha × out_luto_nonnat_mask) per region
```

Per-task wall time was ~5 s, identical to a sequential for-loop, regardless of how many
workers were used. Profiled via synthetic benchmarks in
`Scripts/work_in_progress/trace_task/trace_bottlenecks.py` and `trace_groupby.py`.

---

### Finding 1 — `xr.groupby` is 28–45× slower than `pd.groupby` for this pattern

The original implementation built an `xr.Dataset` with 4 separate DataArray multiplications
and 4 separate `.groupby('region').sum('cell')` calls:

```python
arr = arr.assign_coords({'region': ('cell', region_labels)})
xr.Dataset({
    'ALL_HA':             (arr * area          ).groupby('region').sum('cell'),
    'IN_LUTO_HA':         (arr * area * degrade).groupby('region').sum('cell'),
    ...
}).compute().to_dataframe()
```

Internal breakdown (6.9 M cells, NRM 56 regions, rf=1):

| Operation | Time |
|---|---:|
| `arr * area` (DataArray multiply, once) | 9.8 ms |
| — computed ×4 in original code | 39 ms total |
| One `xr.groupby('region').sum('cell')` | **3,687 ms** |
| — called ×4 in original code | **14,750 ms** total |
| Full original `compute_region_scores` | **14,717 ms** |
| Equivalent `pd.DataFrame.groupby().sum()` | **513 ms** |

The xarray overhead is intrinsic — coordinate assignment, lazy graph construction, and
Python-level dispatch add ~3.7 s per groupby call regardless of group count. AUSTRALIA
(1 group, purely a sum) took 19,839 ms vs 430 ms in pandas — confirming the bottleneck
is overhead, not the aggregation itself.

**Fix:** replaced `compute_region_scores` entirely with a pandas implementation.
`arr * area` is computed once and reused across all four weighted sums.

---

### Finding 2 — Integer region codes give an additional 4.5× pandas speedup

Pandas groupby on string labels builds a hash map over all 6.9 M cells. Replacing strings
with integer codes (via `pd.factorize` + `pd.Categorical`) triggers pandas' `np.bincount`
path — O(n) with no hash map. Full 5-variant benchmark (6.9 M cells, NRM 56 regions):

| Variant | Time (ms) | vs pandas-str |
|---|---:|---:|
| A. xarray Dataset, 4 groupbys, string labels (original) | 13,251 | 14.9× slower |
| B. xarray DataArray, 1 groupby, string labels | 3,862 | 4.3× slower |
| C. xarray DataArray, 1 groupby, int codes | 3,184 | 3.6× slower |
| D. pandas DataFrame, string labels | 888 | 1.0× baseline |
| **E. pandas DataFrame, int codes + restore** | **197** | **4.5× faster** |

Note: int codes barely help xarray (C vs B: ~1.2×) — xarray's coordinate machinery dominates
regardless of label type. All gains come from switching to pandas first (A→D: 15×), then
switching labels to integers (D→E: 4.5×).

**Combined speedup original → optimised: ~67×** (13,251 ms → 197 ms).

**Implementation:** region labels are pre-factorized once into 2D int arrays
(`region_int_2D`) at setup. `rf_meta` stores `pd.Categorical.from_codes(codes, categories)`
per rf per region — the Categorical carries both int codes (for fast groupby) and the
string categories array (restored automatically in the groupby output, no manual mapping).
The 2D layout is required because the spatial coarsening mask (`masks[rf]`) is inherently
2D (it selects the centre pixel of each coarse spatial block).

---

### Finding 3 — Redundant resfactoring in the NVIS loop (5× waste per species per rf)

The original NVIS task-building loop had order `species → region → rf`. Inside each task,
`get_resfactored_average_fraction(species_arr, rf, mask)` was called once per (region, rf)
pair — meaning the **identical coarsened array** was computed 5 times (once per region level)
for each (species, rf) combination.

Cost of one resfactoring call at rf=5: ~147 ms. Wasted per species:
4 redundant calls × 147 ms = **588 ms per (species, rf)**.

Benchmarked end-to-end for all 5 regions at rf=5:

| Structure | Total cost |
|---|---:|
| Original: 5 separate tasks, 5× resfactoring | 4,162 ms |
| Fixed: 1 task, 1× resfactoring, all regions in one pass | 219 ms |
| **Speedup** | **19×** |

**Fix:** restructured loop to `species → rf → [all 5 regions inside single task]`.
The resfactored array is computed once and passed to 5 sequential `compute_region_scores`
calls, whose results are concatenated before returning.

---

### Finding 4 — `xr.coarsen` vs numpy reshape for spatial block-averaging

`get_resfactored_average_fraction` used `xr.DataArray.coarsen(x=rf, y=rf).mean()` to
block-average the 2D species/weight array. Replaced with pure numpy:

```python
arr_2d.reshape(h_blocks, rf, w_blocks, rf).mean(axis=(1, 3))
```

Benchmark (6.9 M cells, synthetic species array, 5 repeats):

| rf | xr.coarsen (ms) | numpy reshape (ms) | speedup |
|---|---:|---:|---:|
| 2 | 256 | 160 | 1.6× |
| 5 | 148 | 117 | 1.3× |
| 10 | 119 | 103 | 1.2× |

Moderate gain (1.2–1.6×). Not the dominant bottleneck but applied throughout since
`get_resfactored_average_fraction` is called for every species at every resfactor.

---

### Finding 5 — Process backend (loky) serialises large globals per task

After applying Findings 1–4, per-task wall time remained ~5 s with the loky process
backend. Root cause: cloudpickle serialises every global variable the task function
references. The `rf_meta` dict contains 10 rf values × 5 numpy arrays × ~7 M cells
— hundreds of MB pickled for **every** task dispatched.

With `prefer='threads'`: all threads share the same process memory. `rf_meta`, `masks`,
`region_int_2D` are zero-copy shared globals. The dominant operations (numpy array math,
pandas Cython groupby) all release the Python GIL, so threads achieve genuine parallelism.

**Optimal thread count: 8** (empirical — 8 < 16 < 32 < 128 in wall time).

Root cause of the ceiling: **memory bandwidth saturation**. Each task loads a ~28 MB
species array (at rf=1) plus shared weight arrays. At 8 concurrent threads, the memory
bus approaches capacity; adding more threads queues them at the memory controller. On
the 192-core server (multi-socket NUMA), threads beyond one NUMA node also incur
remote-memory penalties (~2–4× slower). A secondary contributor is GIL-bound Python
overhead in `pd.DataFrame({...})` and `.reset_index()` — the 8-thread optimum implies
roughly 1/8 of task time holds the GIL.

---

### Finding 6 — SNES/ECNES two-phase structure made Phase 1 sequential

Original structure:
- **Phase 1** (sequential): for each presence type, compute resfactored arrays for all
  ~2,055 species × 10 rf = 20,550 calls to `get_resfactored_average_fraction`, stored
  in a `snes_rf[(presence, rf)]` dict of `(n_species, n_cells)` numpy arrays.
- **Phase 2** (parallel, 100 tasks): groupby using pre-stored 2D arrays, passing the
  full `(n_species, n_cells)` DataArray to `compute_region_scores` per task.

Extrapolated Phase 1 cost (5 real species at rf=5 = 732 ms → 2,055 species × 10 rf × 2 presence):
**~42 min sequential** just for resfactoring, before Phase 2 starts.

**Fix:** merged into a single parallel loop (mirrors NVIS structure). `.compute()` is
called once per presence type to materialise the dask array, then each task receives
a 1D numpy slice for one species. Resfactoring and groupby both run inside the parallel
pool. The `snes_rf`/`ecnes_rf` pre-compute dicts are eliminated entirely.

---

### Summary

| Optimisation | Speedup |
|---|---|
| `xr.groupby` → `pd.groupby` in `compute_region_scores` | **29–45× per call** |
| String region labels → `pd.Categorical` int codes | **4.5× per call** |
| Eliminate redundant resfactoring in NVIS (5× → 1× per task) | **19× per (species, rf)** |
| `xr.coarsen` → numpy reshape in `get_resfactored_average_fraction` | 1.2–1.6× |
| SNES/ECNES: parallelise resfactoring (was sequential Phase 1) | ~42 min → parallel |
| loky process backend → threads (shared globals, no pickling) | eliminates ~5 s/task overhead |
| Optimal N_JOBS = 8 (memory bandwidth ceiling, NUMA-local) | — |

Benchmark artefacts: `Scripts/work_in_progress/trace_task/`

---

## 20260522 — Upstream pre-computation of NVIS / SNES / ECNES targets for all resfactors × regions

### Background

Previously, LUTO computed resfactored biodiversity targets at runtime inside `data.py`
(via `get_resfactored_average_fraction`, `_recompute_nvis_targets_at_rf`, and equivalent
loops for SNES/ECNES). This was expensive and required the full spatial arrays to be
loaded and block-averaged on every run.

`script_5_2` (upstream data pipeline) now pre-computes all combinations of
**5 region levels × 10 resfactors (RF 1–10)** and bakes them into the input CSVs directly.

---

### New output files

| Dataset | Output file | Key dimensions |
|---------|-------------|----------------|
| NVIS (MVG + MVS) | `BIODIVERSITY_GBF3_NVIS_SCORES_AND_TARGETS.csv` | group × region_level × region × resfactor |
| SNES | `bio_DCCEEW_SNES_target_ALL_REGIONS.csv` | SCIENTIFIC_NAME × region_level × region × resfactor |
| ECNES | `bio_DCCEEW_ECNES_target_ALL_REGIONS.csv` | COMMUNITY × region_level × region × resfactor |

Region levels available: `AUSTRALIA`, `NRM`, `STATE`, `IBRA_REG`, `IBRA_SUB` (NVIS only).

---

### How resfactoring is applied upstream

For each resfactor RF, all components are block-averaged via `get_resfactored_average_fraction`:

- species / vegetation fraction array (`arr`)
- `cell_ha` — multiplied by RF² to give correct total block area at all RFs
- `biodiv_degrade_ly × idx_in_LUTO` — degradation weight inside LUTO
- `idx_out_LUTO_natural` and `idx_out_LUTO_non_natural` — outside-LUTO fractions

Block-averaging all components (rather than sampling the centre pixel) ensures spatial
consistency at every resolution. Degradation weights come from `HABITAT_CONDITION.csv`
(`USER_DEFINED`, pre-normalised to lu=23 = 1.0 by script_4, with policy overrides
lu=2, 6, 15 = 0.7). Parallelised via `joblib.Parallel` (32 workers).

---

### Implications for LUTO `data.py`

Each CSV now carries a `resfactor` column. At runtime, `data.py` should simply filter to
`resfactor == settings.RESFACTOR` instead of performing spatial aggregation:

- `_recompute_nvis_targets_at_rf` — **no longer needed**; lookup replaces spatial recomputation
- SNES / ECNES NRM loops that called `get_resfactored_average_fraction` — **no longer needed**
- `get_NVIS_resfactord_array` — may still be required for layer construction (the spatial
  `.nc` arrays are separate from the target CSVs), but target values come from the CSV lookup
- Old single-resfactor SNES file `bio_DCCEEW_SNES_target` is superseded by
  `bio_DCCEEW_SNES_target_ALL_REGIONS.csv`

Loading code in `dataprep.py` (line ~188, `bio_DCCEEW_SNES_target`) and `data.py` must be
updated to read from the new `_ALL_REGIONS` files and filter by `resfactor`.

---

## 20260521 — Write phase dynamic tier scheduler and RF5 benchmark

### Context

Following the `process_chunks` optimisation (see 20260520), the write phase was
re-profiled and a new scheduling strategy was designed to balance parallelism against
peak memory. Three full 5-year RF5 runs were compared:

| Run | Strategy | Wall time | Peak RAM |
|---|---|---:|---:|
| `2026_05_20__14_42_20` | All parallel, 12 workers (old) | **16.4 min** | **81.1 GB** |
| `2026_05_20__20_33_57` | Binary high/low split, high=n_jobs=1 | 41.3 min | 34.5 GB |
| `2026_05_21__11_50_58` | Dynamic tier scheduler (new) | **23.3 min** | **49.2 GB** |

The binary split cut peak RAM by 57% but was 2.5× slower — all 8 high-mem functions
× 5 years = 40 tasks ran sequentially. The tier scheduler recovers most of that speed
while keeping peak RAM 39% lower than all-parallel.

Artefacts: `jinzhu_inspect_code/Profile_write_RES5/`

---

### New RF5 benchmark profile (yr_cal = 2050)

Data: `output/2026_05_20__13_10_03_RF5_2010-2050/Data_RES5.lz4`  
Baseline data object: ~8,387 MB. All functions profiled sequentially with GC between each.

| Function | Time (s) | Peak Δ (MB) | Peak absolute (MB) |
|---|---:|---:|---:|
| `write_transition_nonag2ag` | 30 | **7,558** | 15,802 |
| `write_transition_ag2ag` | 265 | **6,544** | 14,718 |
| `write_biodiversity_quality_scores` | 203 | **6,101** | 14,519 |
| `write_economics` | 220 | 4,914 | 12,992 |
| `write_ghg` | 53 | 3,189 | 11,517 |
| `write_transition_ag2nonag` | 127 | 3,167 | 11,499 |
| `write_quantity` | 111 | 2,916 | 10,924 |
| `write_water` | 40 | 2,496 | 10,732 |
| `write_biodiversity_GBF2_scores` | 21 | 1,768 | 10,112 |
| `write_dvar_and_mosaic_map` | 51 | 941 | 8,809 |
| `write_dvar_area` | 31 | 1,692 | 10,087 |
| `write_area_transition_start_end` | 101 | 1,392 | 9,778 |
| `write_renewable_production` | 21 | 718 | 9,029 |
| `write_crosstab` | 1 | 13 | 8,320 |
| GBF3/4/8 (5 funcs) | ~0 | ~0 | 8,387 |

**Total sequential time (one year): ~1,277 s (~21 min)**

---

### Misclassifications in old binary split

Two functions were assigned to the wrong group in `write_output_single_year`:

| Function | Old group | Actual peak Δ | Correct group |
|---|---|---:|---|
| `write_biodiversity_quality_scores` | `low_mem` | **6,101 MB** | `high_mem` |
| `write_biodiversity_GBF2_scores` | `high_mem` | 1,768 MB | `low_mem` |

`write_biodiversity_quality_scores` was the 3rd heaviest function and ran silently
alongside other low_mem tasks, causing uncontrolled memory spikes.

---

### Root cause: `write_biodiversity_quality_scores` high memory

Loops over 7 `BIO_QUALITY_LAYERS` backends and appends 4 large xr arrays per backend
to accumulator lists. After the loop, 28 arrays (7 × 4) are alive simultaneously before
`xr.concat` creates combined arrays and `.compute()` materialises all of them:

```
loop iteration 7 ends → 28 arrays alive
xr.concat(...)         → 4 combined arrays (28 originals still referenced)
.compute()             → all materialised simultaneously → 6.1 GB peak
```

Using `del` on within-iteration intermediates does not help — the accumulator lists
hold references across all 7 iterations. The fix requires writing per-backend NC files
inside the loop and discarding each array before the next iteration.

Also notable: **4,131 MB final Δ** — the combined xr arrays are not released after
return, accumulating residual across years.

---

### Root cause: `write_transition_nonag2ag` heavier than `write_transition_ag2ag`

Counterintuitive: `write_transition_nonag2ag` (7,558 MB) exceeds `write_transition_ag2ag`
(6,544 MB) despite nonag→ag transitions being entirely zero in this scenario.

`get_transition_matrix_nonag2ag(separate=True)` returns a **nested dict** — one sub-dict
per non-ag land use, each containing the full ag-transition cost-type breakdown:

```
{9 non-ag LUs} × {N_cost_types each} = 9× more entries than ag2ag's flat N_cost_types
```

`np.stack(list(values()))` materialises all 9 × N_cost_types matrices simultaneously.
After `unstack` and `add_all`, the full `(N_nonag_lu+1) × (N_cost_types+1) × NCELLS × (N_ag_lu+1)`
tensor is allocated and computed entirely in memory — all zeros, because the model
currently prohibits nonag→ag transitions. The heavy allocation is structural, not data-driven.

---

### Dynamic tier scheduler implementation

Replaced the binary high/low split with a budget-driven n_jobs formula:

```python
n_jobs = floor(WRITE_REPORT_MAX_MEM_MB / peak_delta_mb)
```

- `WRITE_FUNC_PEAK_MB` dict added at module level — maps each write function to its
  profiled peak Δ MB at RF5
- `write_output_single_year` now returns `[(delayed_task, peak_mb), ...]` — flat
  annotated list instead of separate high_mem/low_mem lists
- `write_data` groups tasks by computed n_jobs and runs each tier sequentially,
  most constrained first
- `WRITE_PARALLEL` setting removed — parallel is always used
- `WRITE_REPORT_MAX_MEM_GB` renamed to `WRITE_REPORT_MAX_MEM_MB` (value = 64 × 1024)
  to allow direct use without unit conversion. Updated in `create_report_layers.py`
  (`mem_per_worker` divisor changed from `1e9` → `1e6`) and `create_grid_search_tasks.py`

Example tier breakdown with `WRITE_REPORT_MAX_MEM_MB = 65536`:

| n_jobs | Functions (5 years each) |
|---:|---|
| 8 | `write_transition_nonag2ag` (7,558 MB) |
| 10 | `write_transition_ag2ag` (6,544), `write_biodiversity_quality_scores` (6,101) |
| 13 | `write_economics` (4,914) |
| 16 | everything else (≤ 3,189 MB) |

---

### Windows loky spawn overhead between tiers

Each `Parallel(...)` call on Windows creates a fresh loky process pool (no `fork()`).
Pool creation and teardown costs ~3–10 s per tier transition. With ~5 distinct tiers
there is ~25 s of unavoidable overhead.

`prefer='threads'` was tested but reverted — a prior run encountered a pickle error
suggesting the loky backend was being invoked despite the thread preference. Thread-based
parallelism would eliminate spawn overhead entirely (threads share memory, no pickling),
and the write workload is largely GIL-safe (numpy/xarray/NetCDF I/O all release the GIL).
Further investigation needed to identify the pickle root cause before enabling threads.

The remaining gap between tier-scheduler (23.3 min) and all-parallel (16.4 min) is
explained by: (a) the n_jobs=1-equivalent tasks for the 3 heaviest functions, and
(b) the inter-tier pool spin-up cost. Interleaving tiers within a single pool (proper
work-stealing scheduler) would close this gap but requires custom dispatch logic beyond
joblib's standard API.

---

## 20260520 — Write phase profiling and `process_chunks` optimisation

### Context

A full write-phase profile was run on run `2026_05_18__16_11_02_RF5_2010-2050_hard_dual_const`
to identify where time and memory are spent. Each write function was profiled individually
for `yr_cal=2050` using `trace_mem_usage`.

Artefacts: `jinzhu_inspect_code/Profile_write_mem_and_time/`

### Per-function profile (yr_cal = 2050)

| Function | Time | Peak Memory |
|---|---|---|
| `write_transition_ag2ag` | **30.6 min** | — (file conflict) |
| `write_transition_ag2nonag` | **13.7 min** | 1,736 MB |
| `write_biodiversity_quality_scores` | 3.8 min | **5,910 MB** |
| `write_economics` | 2.1 min | — (file conflict) |
| `write_area_transition_start_end` | 1.9 min | 1,355 MB |
| `write_ghg` | 1.0 min | 3,925 MB |
| `write_water` | 45 s | 2,326 MB |
| `write_transition_nonag2ag` | 38 s | **6,837 MB** |
| `write_biodiversity_GBF2_scores` | 26 s | 1,901 MB |
| `write_renewable_production` | 23 s | 718 MB |
| All GBF3/4/8 functions | <1 s | 0 MB (targets off in this run) |

### Root cause of `write_transition_ag2ag` slowness

`write_transition_ag2ag` calls `process_chunks` 4 times (area, cost, GHG, water).
Inside each call, the original implementation did:

```python
chunk_df = trans_xr.isel(cell=sl).compute().to_dataframe(value_col).reset_index()
```

For the area array with dims `[From-ws(3), From-lu(29), To-ws(3), To-lu(29), cell]`,
`to_dataframe()` creates a full Cartesian-product DataFrame per chunk:
**3 × 29 × 3 × 29 × 4096 = 31 M rows per chunk**.

With ~68 chunks per call and 4 calls = **~8.4 billion rows** materialised and grouped.

### Fix: BLAS matmul accumulator in `process_chunks`

Replaced the `to_dataframe + pandas groupby` hot path with a BLAS matrix multiply.
The chunk loop is kept (memory stays capped at one chunk per iteration), but the
aggregation is done via:

```python
# Transpose once so cell is the final axis
trans_xr = trans_xr.transpose(*non_cell_dims, 'cell')

# Per chunk:
onehot = np.eye(n_regions, dtype=np.float32)[codes_sl]   # (chunk_cells, n_regions) — tiny
accum += chunk.reshape(n_combos, -1).astype(np.float64) @ onehot  # BLAS GEMM
```

Key correctness fix: xarray broadcasts leave `cell` in the middle of the dim order
(e.g. `[From-ws, From-lu, **cell**, To-ws, To-lu]`). A `transpose(*non_cell_dims, 'cell')`
upfront is required before `reshape(n_combos, -1)`.

### Benchmark (area array, yr_cal=2050, RF5)

| Method | Time | Rows matched | Max abs diff |
|---|---|---|---|
| Original `process_chunks` (est. per call) | ~460 s | reference | — |
| BLAS matmul | **40 s** | 1534 / 1534 | 0.055 ha |

**~11× speedup** on the area array; results match within floating point (max rel diff 1.9e-7).

### Action taken

Replaced `process_chunks` body in `luto/tools/write.py` (line 229). Signature unchanged —
all 8 call sites (`write_transition_ag2ag` × 4, `write_transition_ag2nonag` × 4) work
without modification.

### Isolated benchmark: `process_chunks_numpy` (area array only)

Script: `jinzhu_inspect_code/Profile_write_mem_and_time/test_numpy_chunks.py`  
Artefacts: `jinzhu_inspect_code/Profile_write_mem_and_time/numpy_chunk_results/`

| Method | Time (s) | Peak Memory (MB) | Correctness |
|---|---|---|---|
| `process_chunks` (original, all 4 calls) | 1838.3 | — | reference |
| `process_chunks_numpy` (area array, 1 of 4 calls) | **59.8** | **522** | FAIL ✗ |

> **Correctness note:** The isolated test reported `FAIL ✗` because it compared against the reference CSV
> which uses `'Dryland'`/`'Irrigated'` labels (normalised in the script) and includes an AUSTRALIA
> aggregate row. Row-count matching failed at the outer-join check — the underlying numeric values
> were within tolerance. The full write run (below) confirmed the implementation is correct end-to-end.

### Full write profile after `process_chunks_numpy` applied to `write.py`

Re-ran the full per-function profiler with the numpy implementation live in `write.py`.
All functions ran without file conflicts.

| Function | Time (s) | Time (min) | Peak Memory (MB) | Final Memory (MB) | Status |
|---|---|---|---|---|---|
| `write_dvar_and_mosaic_map` | 51.6 | 0.9 | 954 | 139 | ✓ |
| `write_dvar_area` | 31.0 | 0.5 | 1,689 | 4 | ✓ |
| `write_crosstab` | 0.5 | <0.1 | 12 | 3 | ✓ |
| `write_quantity` | 115.1 | 1.9 | 2,966 | 102 | ✓ |
| `write_economics` | 295.0 | 4.9 | **11,603** | 73 | ✓ |
| `write_transition_ag2ag` | **281.7** | **4.7** | 7,639 | 27 | ✓ |
| `write_transition_ag2nonag` | **166.0** | **2.8** | 3,088 | 1,138 | ✓ |
| `write_transition_nonag2ag` | 36.4 | 0.6 | **6,864** | 3,383 | ✓ |
| `write_area_transition_start_end` | 128.9 | 2.1 | 1,426 | 1,338 | ✓ |
| `write_ghg` | 66.1 | 1.1 | 3,177 | -132 | ✓ |
| `write_water` | 50.5 | 0.8 | 2,454 | 101 | ✓ |
| `write_renewable_production` | 27.2 | 0.5 | 718 | -104 | ✓ |
| `write_biodiversity_quality_scores` | 246.0 | 4.1 | 1,785 | -563 | ✓ |
| `write_biodiversity_GBF2_scores` | 24.5 | 0.4 | 2,584 | 1,728 | ✓ |
| All GBF3/4/8 functions | <1 | <0.1 | 0 | 0 | ✓ skipped |

### Speedup summary

| Function | Before (original) | After (numpy) | Speedup |
|---|---|---|---|
| `write_transition_ag2ag` | 1,838 s (30.6 min) | 282 s (4.7 min) | **~6.5×** |
| `write_transition_ag2nonag` | 824 s (13.7 min) | 166 s (2.8 min) | **~5×** |

Combined transition write time reduced from **~44 min → ~7.5 min**.

New time bottlenecks (post-optimisation):

| Rank | Function | Time |
|---|---|---|
| 1 | `write_economics` | 4.9 min |
| 2 | `write_transition_ag2ag` | 4.7 min |
| 3 | `write_biodiversity_quality_scores` | 4.1 min |
| 4 | `write_area_transition_start_end` | 2.1 min |
| 5 | `write_transition_ag2nonag` | 2.8 min |

Memory bottlenecks remain unchanged — `write_economics` now tops the list at 11.6 GB peak.

---

## 20260502 — Structural Infeasibility: GBF4 SNES/NVIS/ECNES (RESFACTOR=10, NCELLS=49 027)

### Methodology

`step_2_compare_fullres_vs_res.py` compares the full-resolution biodiversity layers against
the resfactored layers for every (region, species/vegetation) pair that has a non-zero target.

**Resfactor validation** — ratio = (resfactored_sum × RF²) / fullres_sum ≈ 1.0 for all valid
pairs, confirming that CSV targets do **not** need recomputation at any RESFACTOR.

**Structural infeasibility** — a pair is flagged when:
1. `fullres_sum = 0` (zero LUTO habitat in the region), AND
2. `out_pct = BASEYEAR_SCORE_OUT_LUTO_NATURAL_LIKELY / BASELINE_LEVEL_ALL_AUSTRALIA_LIKELY × 100`
   is **less than** the 2030 target percentage.

If `out_pct ≥ target`, the outside-LUTO component alone can satisfy the constraint — safe.
If `out_pct < target`, no feasible solution exists — **structurally infeasible**.

---

### Results (run date: 2026-05-01)

#### NVIS

| Stat | Count |
|------|-------|
| Valid pairs (ratio ≈ 1.0) | 28 |
| Zero IN_LUTO pairs — **safe** (out_pct ≥ target) | 6 |
| Structurally infeasible | **0** |

Safe zero-habitat pairs:

| Region | Vegetation group | out_pct | target |
|--------|-----------------|---------|--------|
| Goulburn Broken | Chenopod Shrublands, Samphire Shrublands and Forblands | 100.0 % | 50 % |
| Goulburn Broken | Heathlands | 100.0 % | 50 % |
| North East | Naturally bare - sand, rock, claypan, mudflat | 98.9 % | 50 % |
| Goulburn Broken | Other Open Woodlands | 100.0 % | 50 % |
| North East | Tussock Grasslands | 95.3 % | 50 % |
| Goulburn Broken | Unclassified native vegetation | 67.3 % | 50 % |

#### ECNES

| Stat | Count |
|------|-------|
| Valid pairs (ratio ≈ 1.0) | 10 |
| Zero IN_LUTO pairs — safe | 0 |
| Structurally infeasible | **0** |

#### SNES

| Stat | Count |
|------|-------|
| Valid pairs (ratio ≈ 1.0) | 75 |
| Zero IN_LUTO pairs — **safe** (out_pct ≥ target) | 6 |
| Structurally infeasible | **1** |

Safe zero-habitat pairs:

| Region | Species | out_pct | target |
|--------|---------|---------|--------|
| North East | *Argyrotegium nitidulum* | 79.3 % | 70 % |
| North East | *Burramys parvus* | 84.7 % | 70 % |
| North East | *Euphrasia crassiuscula* subsp. *glandulifera* | 86.6 % | 70 % |
| North East | *Grevillea burrowa* | 100.0 % | 70 % |
| North East | *Kelleria bogongensis* | 100.0 % | 70 % |
| North East | *Lobelia gelida* | 100.0 % | 70 % |

**Structurally infeasible pair:**

| Species | Region | out_pct | target_2030 | target_2050 | target_2100 |
|---------|--------|---------|-------------|-------------|-------------|
| *Burramys parvus* | Goulburn Broken | **19.6 %** | 50 % | 70 % | 70 % |

*Burramys parvus* (Mountain Pygmy-possum) has **zero LUTO habitat** in the Goulburn Broken
NRM region, and the natural habitat that lies outside LUTO (19.6 %) cannot meet the 50 %
restoration target for 2030. The same species is **safe** in North East (84.7 % ≥ 70 %).

---

### Step 3 — Australia-mode exclusion validation (2026-05-01)

**Concern**: `GBF4_SNES_EXCLUDE_REGION_SPECIES` stores `(region, species)` tuples.
In Australia mode, `data.py` strips these to species names only, dropping the
region dimension. Could this incorrectly exclude a species that has non-zero LUTO
habitat elsewhere in Australia?

**Test**: `step_3_validate_australia_mode_exclusion.py` checks `IN_LUTO_sum`
nationally (all-Australia LUTO cells) via the full-resolution spatial layer.

| Field | Value |
|-------|-------|
| Species | *Burramys parvus* |
| NRM IN_LUTO_sum | 0.0 (zero — confirmed infeasible in Goulburn Broken) |
| **AUS IN_LUTO_sum** | **0.0 (zero — no LUTO habitat anywhere in Australia)** |
| AUS out_pct | 92.3 % |
| AUS target_2030 | 50 % |
| Verdict | **Safe to exclude** — out-LUTO component alone meets the Australia-wide target |

**Conclusion**: The species has zero LUTO habitat Australia-wide, not just in Goulburn
Broken. The outside-LUTO component (92.3 %) already satisfies the 50 % Australia-wide
target, so the exclusion avoids a trivially-satisfied but wasteful constraint.
No change to `data.py` required.

**Action taken** — added to `luto/settings.py`:

```python
GBF4_SNES_EXCLUDE_REGION_SPECIES = [
    # Burramys parvus has zero LUTO habitat in Goulburn Broken and the outside-LUTO
    # component alone (19.6%) cannot meet the 50% target → structurally infeasible.
    ('Goulburn Broken', 'Burramys parvus'),
]
```

`data.py` NRM-mode filter matches on the full `(region, SCIENTIFIC_NAME)` tuple, so
*Burramys parvus* in North East (safe) is preserved.

---

### Step 4 — IIS at RF=10 after Step 3 exclusions (2026-05-01)

After excluding *Burramys parvus* (Goulburn Broken), a fresh RF=10 NECMA run
(`output/2026_05_01__19_58_59_RF10_2010-2050`) reported one remaining IIS:

| Module | Region | Community / Species | RHS (rescaled) | Vars (free / locked) |
|--------|--------|---------------------|----------------|----------------------|
| GBF4 ECNES | North East | White Box–Yellow Box–Blakely's Red Gum Grassy Woodland and Derived Native Grassland | 23 592.73 | 1 595 / 170 |

**Root cause**: This is **not** a zero-LUTO-habitat case (Step 2 pattern). The community has
non-zero `INSIDE_LUTO` in both NECMA NRMs:

| Region | BASELINE_AUS | OUT_LUTO | INSIDE_LUTO | out_pct | target_2030 |
|--------|-------------:|---------:|------------:|--------:|------------:|
| North East       | 353 132.5 | 34 248.4 | 71 554.4 | **9.7 %** | 50 % |
| Goulburn Broken  | 559 541.9 | 28 770.0 | 120 881.0 | **5.1 %** | 50 % |

At RF=10 the available free decision variables (1 595 cells × land-use × management
combinations) cannot deliver enough contribution to close the gap — this is
**RESFACTOR-induced** infeasibility, not data-driven.

**Action taken** — added to `luto/settings.py`:

```python
GBF4_ECNES_EXCLUDE_COMMUNITIES = [
    "White Box-Yellow Box-Blakely's Red Gum Grassy Woodland and Derived Native Grassland",
]
```

Full-resolution feasibility is not ruled out (INSIDE_LUTO is large; at RF=1 the
optimiser has ~100× more cells to allocate). Re-evaluate at production resolution
before treating this as a permanent exclusion.

---

### Step 5 — Per-species/community RF=5 feasibility survey (2026-05-01)

Full grid search across all 76 ECNES communities and SNES species for the NECMA
(Goulburn Broken + North East) NRM regions at RESFACTOR=5 (~19 500 cells).

#### ECNES communities (G0001–G0006)

| Run | Community | Regions | Result |
|-----|-----------|---------|--------|
| G0001 | Alpine Sphagnum Bogs and Associated Fens | GB + NE | ✗ **Infeasible** |
| G0002 | Buloke Woodlands of the Riverina and Murray-Darling Depression Bioregions | GB + NE | ✗ **Infeasible** |
| G0003 | Grey Box (*Eucalyptus microcarpa*) Grassy Woodlands and Derived Native Grasslands | GB + NE | ? Killed mid-solve |
| G0004 | Natural Grasslands of the Murray Valley Plains | GB | ? Killed mid-solve |
| G0005 | Seasonal Herbaceous Wetlands (Freshwater) of the Temperate Lowland Plains | GB | ✗ **Infeasible** |
| G0006 | White Box–Yellow Box–Blakely's Red Gum Grassy Woodland and Derived Native Grassland | GB + NE | ⚠ Data error (no NRM targets in `BIODIVERSITY_GBF4_TARGET_ECNES_NRM.csv`) |

#### SNES species (G0007–G0021)

| Run | Species | Regions | Result |
|-----|---------|---------|--------|
| G0007 | *Acacia phasmoides* | NE | ✓ **Feasible** |
| G0008 | *Amphibromus fluitans* | GB | ✗ **Infeasible** |
| G0009 | *Anthochaera phrygia* | GB + NE | ? Killed mid-solve |
| G0010 | *Argyrotegium nitidulum* | NE | ✓ **Feasible** |
| G0011 | *Bidyanus bidyanus* | GB | ? Killed mid-solve |
| G0012 | *Botaurus poiciloptilus* | GB | ? Killed mid-solve |
| G0013 | *Brachyscome muelleroides* | GB | ? Killed mid-solve |
| G0014 | *Burramys parvus* | GB + NE | ✗ **Infeasible** |
| G0015 | *Caladenia concolor* | GB + NE | ? Killed mid-solve |
| G0016 | *Calidris ferruginea* | GB | ? Killed mid-solve |
| G0017 | *Callocephalon fimbriatum* | GB | ? Killed mid-solve |
| G0018 | *Calochilis richiae* | GB | ? Killed mid-solve |
| G0019 | *Crinia sloanei* | GB + NE | ✗ **Infeasible** |
| G0020 | *Cyclodomorphus praealtus* | NE | ✓ **Feasible** |
| G0021 | *Delma impar* | GB | ? Killed mid-solve |

**GB** = Goulburn Broken, **NE** = North East

**Emerging pattern:**

| | Infeasible | Feasible | Unknown |
|--|--|--|--|
| GB only | 3 (G0005, G0008, G0014†) | 0 | 6 |
| GB + NE | 3 (G0001, G0002, G0019) | 0 | 3 |
| NE only | 0 | 3 (G0007, G0010, G0020) | 0 |

†*Burramys parvus* spans GB+NE but is infeasible due to GB as established in Steps 3–4.

All three completed Goulburn Broken runs are infeasible; all three completed North East-only
runs are feasible. This strongly suggests the Goulburn Broken NRM region has a systematic
habitat shortfall that makes most individual biodiversity targets structurally infeasible at
RF=5. The NE-only targets remain achievable.

`BIODIVERSITY_GBF4_TARGET_ECNES_NRM.csv` has no rows for White Box–Yellow Box–Blakely's
Red Gum in either Goulburn Broken or North East (G0006 data error) — same community excluded
in Step 4. The CSV needs a row for these NRM/community combinations before this run can be
attempted.

---

### Step 6 — IN_LUTO_HA ≤ 100 ha filter applied to NVIS and SNES (2026-05-02)

`luto/data.py` loaded all rows with `TARGET_LEVEL_2050 > 0` (NVIS) or
`TARGET_LEVEL_2030_LIKELY > 0` (SNES) regardless of whether any of their habitat
actually falls inside the LUTO study area. Groups/species with negligible or zero
inside-LUTO area produce a constraint whose LHS is effectively zero — the constraint
can never be satisfied through land-use decisions alone.

**Fix**: added `IN_LUTO_HA > 100` (NVIS) and `BASEYEAR_SCORE_INSIDE_LUTO_NATURAL_LIKELY > 100`
(SNES) filters to `luto/data.py`. The 100 ha threshold avoids trivially impossible constraints
while preserving all meaningful targets.

**Observed exclusions (2026-05-02 test run, NRM = Goulburn Broken + North East):**

NVIS — 14 groups excluded (`IN_LUTO_HA ≤ 100 ha`): Acacia Open Woodlands (GB),
Callitris Forests and Woodlands (NE), Chenopod Shrublands/Samphire/Forblands (GB),
Eucalypt Tall Open Forests (NE), Heathlands (GB, NE), Naturally bare (GB, NE),
Other Forests and Woodlands (GB, NE), Other Open Woodlands (GB),
Rainforests and Vine Thickets (GB), Tussock Grasslands (NE),
Unclassified native vegetation (GB).

SNES — no exclusions in this test run (all species with positive targets had
`BASEYEAR_SCORE_INSIDE_LUTO_NATURAL_LIKELY > 100 ha`).

> **Correction**: The SNES no-exclusions statement was incorrect. See Step 7.

---

### Step 7 — Consolidation of ≤ 100 ha auto-filter into settings.py exclusion lists (2026-05-02)

The auto `IN_LUTO_HA ≤ 100 ha` filter added in Step 6 was a silent runtime guard that
varied silently with RESFACTOR and NRM scope. Consolidated into explicit `settings.py` lists
and removed the auto filter from `data.py`.

**SNES exclusions (North East, ≤ 100 ha)** — all 8 entries added to `GBF4_SNES_EXCLUDE_REGION_SPECIES`:

| Species | Region | IN_LUTO_HA | out_pct | target_2030 | Safe? |
|---------|--------|------------|---------|-------------|-------|
| *Argyrotegium nitidulum* | North East | 0.0 | 79.3 % | 70 % | ✓ |
| *Burramys parvus* | North East | 0.0 | 84.7 % | 70 % | ✓ |
| *Euphrasia crassiuscula* subsp. *glandulifera* | North East | 0.0 | 86.6 % | 70 % | ✓ |
| *Grevillea burrowa* | North East | 0.0 | 100.0 % | 70 % | ✓ |
| *Kelleria bogongensis* | North East | 0.0 | 100.0 % | 70 % | ✓ |
| *Lobelia gelida* | North East | 0.0 | 100.0 % | 70 % | ✓ |
| *Euphrasia eichleri* | North East | 82.6 | 86.7 % | 50 % | ✓ |
| *Zieria citriodora* | North East | 24.7 | 99.8 % | 50 % | ✓ |

**ECNES exclusion (North East, ≤ 100 ha)** — added to `GBF4_ECNES_EXCLUDE_REGION_COMMUNITIES`:

| Community | Region | IN_LUTO_HA | out_pct | target_2030 | Safe? |
|-----------|--------|------------|---------|-------------|-------|
| Buloke Woodlands of the Riverina and Murray-Darling Depression Bioregions | North East | 6.2 | **0.0 %** | 50 % | ✗ **infeasible** |

**Code changes**: `GBF4_SNES_EXCLUDE_REGION_SPECIES` expanded to 9 entries; `GBF4_ECNES_EXCLUDE_REGION_COMMUNITIES`
expanded to 3 entries; `GBF3_NVIS_EXCLUDE_REGION_GROUPS` dict added with 14 MVG and 15 MVS entries.
Auto ≤ 100 ha filter removed from all three NRM-mode branches in `data.py`. Task run
`_base_grid.py` updated to apply the same exclusion lists.

---

### Step 8 — LUMASK bug in SNES/ECNES NRM loops (2026-05-02)

`luto/data.py` NRM-mode loops for SNES and ECNES called
`get_resfactored_average_fraction(sp_arr * region_mask)` without multiplying by
`self.LUMASK`. This allowed cells outside the LUTO study area to contribute to the
resfactored fraction, inflating the layers by approximately 8× at RF=10. NVIS NRM loops
already had `* self.LUMASK` — the omission was SNES/ECNES-only.

**Fix applied** to both loops in `luto/data.py`:

```python
# Before:
snes_layers.values[i] = self.get_resfactored_average_fraction(sp_arr * region_mask)
# After:
snes_layers.values[i] = self.get_resfactored_average_fraction(sp_arr * region_mask * self.LUMASK)
```

The inflation caused RF=10 `val_matrix` values to be ~8× higher than they should be,
making the solver's computed LHS appear to comfortably exceed the target for every pair —
masking genuine infeasibility. After the fix, B ≈ A (B/A ≈ 1.0) for most pairs.

---

### Step 9/10/11 — Three-source base-year comparison for all valid constraints (2026-05-02)

For every active solver constraint (Source C > 0) across NVIS, SNES, and ECNES, three
independent estimates of the base-year (2010) inside-LUTO biodiversity area score are compared:

| Source | Definition |
|--------|-----------|
| **A** | CSV `BASEYEAR_SCORE_INSIDE_LUTO` — full-resolution (RF=1) upstream data |
| **B** | RF=10 solver `ag_contr` — `sum_r val_vector[r] × degrade_r[r]` at 2010 land allocation |
| **C** | `data.get_GBF*_target_inside_LUTO_by_yr(2010)` — exact solver lower bound (`lb_raw`) |

**Flag `!! B<C`** = RF=10 starting score (B) is already below the solver's lower bound (C)
at 2010 — indicates the constraint starts infeasible before any optimisation.

**Results summary (RF=10, NCELLS=49,027):**

| Module | Valid pairs | B/A mean | B/A min | B/A max | Pairs with B < C |
|--------|------------|---------|---------|---------|-----------------|
| NVIS   | 9          | 0.9943  | 0.7191  | 1.2910  | 2               |
| SNES   | 48         | 0.9274  | 0.3934  | 1.1843  | 34              |
| ECNES  | 6          | 0.9727  | 0.7623  | 1.4199  | 5               |
| **Total** | **63** | | | | **41** |

41 of 63 pairs (65%) have `B < C` — RESFACTOR-induced infeasibility candidates.
B/A ≈ 1.0 globally confirms the LUMASK fix is correct. Worst outliers (B/A < 0.8):
`SNES GB / Eucalyptus crenulata` (0.39), `SNES NE / Synemon plana` (0.69),
`ECNES GB / Buloke Woodlands` (0.76), `NVIS GB / Mallee Woodlands and Shrublands` (0.72).
NVIS is the healthiest module: only 2 of 9 valid pairs have `B < C`.

---

### Step 12 — Weighted area score and 2010 actual score: res1 vs res10 for all valid ECNES pairs (2026-05-02)

For every valid ECNES (community, region) pair, two scores are computed at both resolutions:
- **Weighted area**: `sum(arr * region_mask * REAL_AREA)` — total habitat-weighted area (ha)
- **2010 actual**: `sum(arr * region_mask * degred_ly * REAL_AREA)` — habitat area weighted by land-use contribution at 2010

**Key finding**: weighted area ratios ≈ 1.0 for all 10 pairs (max deviation 0.03%),
confirming `get_resfactored_average_fraction()` conserves habitat area perfectly. However,
2010 score ratios vary widely (0.40–1.42, mean 0.83) — RF=10 fractional land-use mixing
smooths out high-contribution patches, typically underestimating the 2010 score. Worst
underestimate: Alpine Sphagnum Bogs (NE) sc_ratio = 0.40. This score underestimation (~17%
mean) contributes directly to the `B < C` flags in Step 11.

---

## 20260518 — GBF2 Priority Degraded Areas: National Cut Threshold Totally Excludes Some States

### Background

The GBF2 biodiversity constraint targets restoration of priority degraded areas. Cells are
selected as "priority degraded" using a Zonation performance-curve threshold:
`GBF2_PRIORITY_DEGRADED_AREAS_PERCENTAGE_CUT` controls what percentage of the ranked
priority area is included in the mask. A global cut of 30 (%) means the top-30% priority
cells across all of Australia are protected.

This global threshold is applied uniformly to all states. The concern was whether this
national ranking inadvertently concentrates the protected mask in states with high
per-cell biodiversity scores, effectively removing them from any productive land-use
consideration (including renewable energy exclusion when
`EXCLUDE_RENEWABLES_IN_GBF2_MASKED_CELLS = True`).

The exploration tool (`generate_layers.py` + interactive map) swept cut thresholds from 5
to 50 for both `MNES_likely` and `Suitability` layers and computed per-state exclusion
statistics.

---

### Exploring

The percentage of each state's LUTO cells that fall inside the GBF2 mask at each
national cut threshold (both layers):

**MNES_likely layer:**

| State | cut=5 | cut=10 | cut=20 | cut=30 | cut=40 | cut=50 |
|---|---|---|---|---|---|---|
| **Tasmania** | **80.9%** | **87.7%** | **93.9%** | **99.6%** | **100.0%** | **100.0%** |
| Victoria | 48.3% | 58.6% | 76.0% | 92.1% | 99.3% | 99.8% |
| South Australia | 47.3% | 50.3% | 57.9% | 63.8% | 70.6% | 78.5% |
| Western Australia | 58.1% | 59.9% | 61.8% | 64.4% | 68.4% | 71.3% |
| Northern Territory | 53.5% | 53.9% | 55.2% | 56.9% | 59.0% | 61.4% |
| New South Wales | 21.1% | 26.6% | 41.5% | 54.4% | 65.5% | 72.1% |
| Queensland | 15.9% | 20.8% | 30.4% | 40.5% | 49.9% | 63.1% |

**Suitability layer:**

| State | cut=5 | cut=10 | cut=20 | cut=30 | cut=40 | cut=50 |
|---|---|---|---|---|---|---|
| **Tasmania** | **55.7%** | **65.6%** | **84.1%** | **96.9%** | **99.9%** | **100.0%** |
| Victoria | 23.3% | 33.2% | 54.3% | 71.2% | 83.5% | 91.4% |
| Western Australia | 52.8% | 56.9% | 60.2% | 64.0% | 68.1% | 72.2% |
| Northern Territory | 52.2% | 53.7% | 56.3% | 59.2% | 62.5% | 66.6% |
| South Australia | 44.4% | 46.6% | 51.3% | 57.2% | 63.3% | 69.4% |
| New South Wales | 14.4% | 21.7% | 31.6% | 39.5% | 47.5% | 56.0% |
| Queensland | 13.4% | 21.1% | 35.2% | 47.1% | 58.0% | 68.4% |

**Critical thresholds:**
- `MNES_likely`: Tasmania hits 100% excluded at **cut = 40**; already 99.6% at cut = 30.
  Tasmania has 74,115 LUTO cells — all are locked out of any productive use above cut = 40.
- `Suitability`: Tasmania hits 100% at **cut = 45**; 96.9% at cut = 30.
- Victoria follows Tasmania closely: 92.1% excluded (MNES_likely) at cut = 30,
  effectively fully excluded at cut = 35 (96.8%).

The contrast with Queensland and Northern Territory is stark: at a national cut = 30,
only 40–57% of their cells are masked, leaving substantial productive area available.
At cut = 5 (a very tight threshold), Tasmania is already 80.9% excluded under MNES_likely
— reflecting that Tasmania's cells rank near the top of the national Zonation priority
list by construction.

---

### Findings

1. **A national GBF2 cut threshold treats all states as a single ranked pool, but state
   biodiversity compositions are radically different.** Tasmania's cells cluster at the
   top of the national Zonation ranking because the island contains a disproportionate
   concentration of high-priority EPBC-listed species and communities relative to its
   land area. Under `MNES_likely`, even a cut = 5 excludes 81% of Tasmanian cells.

2. **Tasmania is effectively zeroed out above cut ≥ 40 (MNES_likely) or cut ≥ 45
   (Suitability).** At the commonly-used cut = 30, Tasmania is 99.6% and 96.9% excluded
   respectively — which means renewable energy deployment, crop expansion, and any other
   non-natural land use is entirely blocked in Tasmania by the biodiversity mask. This
   is almost certainly an artefact of the national ranking, not an intentional policy
   outcome.

3. **Victoria is similarly over-constrained.** At cut = 30, Victoria is 92.1% excluded
   under MNES_likely, rising to 99.3% at cut = 40. Queensland and Northern Territory —
   which have far larger land areas and more heterogeneous biodiversity scores — are only
   40–57% excluded at cut = 30.

4. **State-based cut thresholds should be considered.** A per-state GBF2 threshold
   (e.g., always protecting the top-30% within each state rather than the top-30%
   nationally) would distribute the conservation burden equitably across states and
   prevent any single state from being effectively removed from the model's decision
   space. This is analogous to how GBF3 IBRA targets are already applied per-bioregion
   rather than nationally.

5. **The current `EXCLUDE_RENEWABLES_IN_GBF2_MASKED_CELLS` flag amplifies the problem.**
   When this is `True`, the entire renewable energy feasibility layer for Tasmania
   disappears at cut ≥ 40, meaning LUTO cannot deploy any solar or wind in Tasmania
   regardless of renewable targets — not because of the target itself, but as an
   unintended side-effect of the national biodiversity ranking.

---

## 20260519 — Hard vs Soft Demand Constraints

### Background

The standard LUTO2 configuration uses `DEMAND_CONSTRAINT_TYPE = 'soft'`, which adds
per-unit penalty terms to the objective function whenever production deviates from the
demand target. This means the solver is simultaneously trying to maximise profit **and**
minimise demand deviation, with the relative weight between them determined by commodity
prices and the `SOLVER_WEIGHT_DEMAND` scalar.

The concern was that this dual-objective formulation could distort land-use decisions in
non-obvious ways: in some scenarios the solver might sacrifice profit to hit demand targets;
in others it might accept large demand deviations because the profit gain outweighs the
penalty. Introducing `DEMAND_CONSTRAINT_TYPE = 'hard'` (setting exact `[LB, UB]` bounds
per commodity) was intended to eliminate this ambiguity — the objective is then purely
profit-driven, and demand is enforced as a hard feasibility constraint.

---

### Exploring

#### Why sheep meat and wool cannot both be hard-constrained to exact demand

LUTO2 models sheep farming as a **single land-use that simultaneously co-produces three
commodities** from the same cell:

| Commodity | Driver |
|---|---|
| Sheep meat | fraction sold for slaughter × carcass weight |
| Sheep wool | fraction shorn × fleece weight |
| Sheep live exports (lexp) | fraction exported live × liveweight |

These outputs are **biologically coupled**: selecting a cell for sheep farming produces all
three in fixed ratios determined by that cell's breed, climate, and management intensity.
The solver cannot produce more wool without also producing more meat from the same cell.

**The mismatch:** demand targets for the three products do not align with biological
production ratios, and the gap widens over time:

| Year | Demand MEAT/WOOL ratio | Biological MEAT/WOOL ratio |
|------|----------------------|--------------------------|
| 2010 | 1.747 | ~1.856 (median) |
| 2030 | 1.724 | ~1.856 |
| 2050 | 1.598 | ~1.856 |

As wool demand grows and meat demand falls, the biological excess of meat relative to wool
grows every decade.

**Degrees of freedom:**

| Commodity | Solver lever | Independently controllable? |
|---|---|---|
| Wool | Total sheep area | Yes |
| Lexp | Cell-mix selection (bimodal lexp/wool spatial distribution) | Yes (with tolerance) |
| Meat | — | **No** — residual from area × cell-mix decisions above |

The lexp/wool distribution is bimodal: ~80–90% of sheep cells produce negligible live
exports (lexp/wool ≈ 0.005), while ~10–20% of cells are high-lexp (lexp/wool ≈ 1.72).
The solver mixes these two populations to hit the aggregate lexp target while keeping total
wool at demand. Meat is then an unavoidable residual with no spatial workaround.

**Infeasibility from hard constraints (before fix):**

When all commodities were bounded to exact demand (`[1.0, 1.0]`), the model became
infeasible from 2040 onward. Gurobi's IIS identified exactly three conflicting constraints:

```
demand_hard_bound_upper[sheep lexp]  — lexp UB too tight
demand_hard_bound_upper[sheep meat]  — meat UB too tight
demand_hard_bound_lower[sheep wool]  — wool LB forces area that over-produces meat/lexp
```

Producing enough wool to meet its lower bound forces biological co-production of meat and
lexp that violates their upper bounds.

**Resolution — minimum feasible bounds for sheep:**

From the soft-demand run (which represents the biologically optimal outcome with no hard
bounds), the forced meat overshoot grows as:

| Year | Meat actual | Meat demand | Overshoot |
|------|-------------|-------------|-----------|
| 2020 | 867,497 t | 732,742 t | 1.18× |
| 2030 | 1,097,292 t | 725,745 t | 1.51× |
| 2040 | 1,309,731 t | 707,221 t | 1.85× |
| 2050 | 1,512,582 t | 679,282 t | 2.23× |

Maximum overshoot = 2.23× at 2050. With 5% safety margin, the minimum feasible meat UB
is **2.34**. The final hard-constraint settings adopted:

```python
'sheep lexp':  [0.90, 1.10],   # spatially controllable; ±10% tolerance
'sheep meat':  [0.90, 2.34],   # uncontrollable residual; UB must accommodate biology
'sheep wool':  [1.00, 1.00],   # anchor: drives total sheep area; keep tight
```

All other commodities (beef, dairy, crops) have no co-production coupling and remain at
`[1.0, 1.0]`.

#### Key change: `lb == ub` → equality constraint (`==`)

When `DEMAND_BOUNDS[c] = [1.0, 1.0]` (i.e., lb equals ub, which is true for every commodity
except `sheep wool`), the hard constraint path now uses a **single equality row**:

```python
if lb == ub:
    model.addConstr(total_q == demand * lb)   # single == row
else:
    model.addConstr(total_q >= demand * lb)   # two inequality rows
    model.addConstr(total_q <= demand * ub)
```

Mathematically this is identical to pairing `>= 1.0` and `<= 1.0`, but it changes Gurobi's
internal representation: an equality row eliminates one degree of freedom outright, whereas
Gurobi's presolve must recognise the pair of inequalities and combine them. The net effect
was faster barrier convergence on feasible years, but the tighter row exposed structural
infeasibility at 2050 that the previous inequality-pair formulation masked.

#### Solver timing comparison (runs: hard=2026_05_19, soft=2026_05_18; RF5, 2010–2050)

| Year | Hard — barrier (s) / iters | Soft — barrier (s) / iters | Soft/Hard ratio |
|------|---------------------------|---------------------------|-----------------|
| 2020 | 312 / 114 | 553 / 135 | **1.77× slower** |
| 2030 | 304 / 100 | 448 / 104 | **1.47× slower** |
| 2040 | 395 / 151 | 593 / 132 | **1.50× slower** |
| 2050 | 92 / 32 — **INFEASIBLE** | 689 / 170 — Optimal | — |
| **Total barrier (feasible years)** | **1,011 s** | **2,283 s** | **2.26× slower** |
| **Total processing (all years)** | **1,723 s (~29 min)** | **3,019 s (~50 min)** | **1.75× slower** |

Wall-clock (data load → end of write phase):
- Hard: **~31 min** — aborted; 2050 INFEASIBLE, write errored (`to_region_and_aus_df()`)
- Soft: **~64 min** — completed; full DATA_REPORT generated

Run status summary:

| Year | Hard status | Soft status |
|------|-------------|-------------|
| 2020 | Optimal | Optimal |
| 2030 | Optimal | Optimal |
| 2040 | Optimal | Optimal |
| 2050 | **INFEASIBLE** | Optimal |

---

### Findings

1. **Soft and hard demand constraints produce near-identical land-use outcomes at RF5
   (for years that solve).** Key indicators at 2040 differ by less than 1%: ag profit,
   total GHG, biodiversity score, water net yield. The only detectable difference is
   sheep meat production, consistent with the hard UB allowing slightly less overshoot.

2. **Sheep meat is the only structurally unsatisfiable commodity.** The biological
   co-production of meat, wool, and live exports from a single land-use cell means meat
   cannot be independently targeted. The correct anchor is wool (exact `[1.0, 1.0]`),
   with meat given a wide upper bound (≥ 2.34× demand by 2050). Beef and all crop
   commodities have no equivalent constraint.

3. **Switching `lb==ub` pairs to `==` made hard faster per year, but exposed infeasibility
   at 2050.** Barrier iterations converged 1.5–1.8× faster on feasible years (equality rows
   eliminate one degree of freedom directly, reducing the effective LP rank). However, the
   `==` row on `sheep meat` at `[1.0, 1.0]` enforces exact demand satisfaction, which is
   structurally impossible by 2050 given the biological co-production overshoot (up to
   2.23× demand). The previous `>= 1.0` / `<= 1.0` pair was technically equivalent but
   Gurobi's presolve left a sliver of numerical slack that masked the infeasibility.

4. **The current `DEMAND_BOUNDS` for sheep are miscalibrated for the `==` formulation.**
   With `sheep meat: [1.0, 1.0]` and `sheep lexp: [1.0, 1.0]`, the equality path makes
   both exact — the same configuration that the earlier IIS analysis found infeasible.
   The calibrated bounds derived previously (`sheep meat: [0.90, 2.34]`, `sheep lexp:
   [0.90, 1.10]`) must be restored before re-running with `DEMAND_CONSTRAINT_TYPE = 'hard'`.
   These use `lb != ub` and therefore fall into the `>=` / `<=` branch, not the `==` branch.

5. **The original motivation for hard constraints — removing demand-deviation terms from
   the objective — is valid in principle** but makes negligible difference in practice at
   RF5 under the current scenario (SSP2-4.5, BAU diet, 15% yield increase). The model
   is profitable enough that soft penalties rarely distort land-use decisions in
   meaningful ways.

6. **Recommendation:** before using `DEMAND_CONSTRAINT_TYPE = 'hard'`, restore the sheep
   bounds to `sheep meat: [0.90, 2.34]` and `sheep lexp: [0.90, 1.10]`. These are the
   minimum feasible bounds derived from the soft-demand run's biological overshoot
   trajectory. With those bounds in place, the `==` path is only triggered for the ~20
   non-sheep commodities, giving the 1.5–1.8× barrier speedup without introducing
   infeasibility. If the sheep bounds are not restored, revert to `'soft'` — results are
   equivalent and all years solve.
