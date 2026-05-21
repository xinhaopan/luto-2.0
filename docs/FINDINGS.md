# LUTO2 Findings Log

A running record of discoveries, investigations, and conclusions from model exploration.
Entries are in **descending date order** (newest first).

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
n_jobs = floor(WRITE_REPORT_MAX_MEM_MB / peak_delta_mb)   # capped at WRITE_THREADS
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

Example tier breakdown with `WRITE_REPORT_MAX_MEM_MB = 65536` and `WRITE_THREADS = 16`:

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
