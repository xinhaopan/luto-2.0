# LUTO2 Findings Log

A running record of discoveries, investigations, and conclusions from model exploration.

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

#### Solver timing comparison

The hard and soft runs were both executed at RF5 (2010–2050, 5 time steps) on the same
machine (AMD EPYC 9654P).

| Year | Hard — Barrier (s) | Soft — Barrier (s) | Hard/Soft |
|------|-------------------|--------------------|-----------|
| 2020 | 553 | 360 | 1.54× slower |
| 2030 | 448 | 315 | 1.42× slower |
| 2040 | 593 | 395 | 1.50× slower |
| 2050 | 689 | 486 | 1.42× slower |
| **Total processing** | **3,019 s (~50 min)** | **2,284 s (~38 min)** | **1.32× slower** |

Wall-clock (start → data writing complete):
- Hard: **56 min 38 s**
- Soft: **44 min 54 s**

---

### Findings

1. **Soft and hard demand constraints produce near-identical land-use outcomes at RF5.**
   Key indicators at 2050 differ by less than 1%: ag profit (+0.1%), total GHG (−0.4%),
   biodiversity score (ALL types, MNES_likely: +0.0%), water net yield (+0.0%). The only
   detectable difference is sheep meat production (−0.8 pp of demand), consistent with
   the hard UB allowing slightly less overshoot.

2. **Sheep meat is the only structurally unsatisfiable commodity.** The biological
   co-production of meat, wool, and live exports from a single land-use cell means meat
   cannot be independently targeted. The correct anchor is wool (exact `[1.0, 1.0]`),
   with meat given a wide upper bound (≥ 2.34× demand by 2050). Beef and all crop
   commodities have no equivalent constraint.

3. **Hard constraints are slower, not faster.** The hard formulation adds strict equality
   and inequality rows that shrink the feasible region, forcing Gurobi's barrier method
   to work ~40–55% more iterations per year to reach optimality. The soft formulation's
   quadratic penalty terms add a smooth bowl to the objective surface that the barrier
   method navigates more efficiently.

4. **The original motivation for hard constraints — removing demand-deviation terms from
   the objective — is valid in principle** but makes negligible difference in practice at
   RF5 under the current scenario (SSP2-4.5, BAU diet, 15% yield increase). The model
   is profitable enough that soft penalties rarely distort land-use decisions in
   meaningful ways.

5. **Recommendation:** retain `DEMAND_CONSTRAINT_TYPE = 'hard'` with the calibrated sheep
   bounds `[0.90, 2.34]` for meat and `[0.90, 1.10]` for lexp, to keep the objective
   function clean. Accept the ~30% runtime penalty as the cost of structural clarity.
   If runtime is the binding constraint, revert to `'soft'` — results are equivalent.
