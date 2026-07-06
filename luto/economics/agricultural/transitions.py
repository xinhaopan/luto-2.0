# Copyright 2025 Bryan, B.A., Williams, N., Archibald, C.L., de Haan, F., Wang, J., 
# van Schoten, N., Hadjikakou, M., Sanson, J.,  Zyngier, R., Marcos-Martinez, R.,  
# Navarro, J.,  Gao, L., Aghighi, H., Armstrong, T., Bohl, H., Jaffe, P., Khan, M.S., 
# Moallemi, E.A., Nazari, A., Pan, X., Steyl, D., and Thiruvady, D.R.
#
# This file is part of LUTO2 - Version 2 of the Australian Land-Use Trade-Offs model
#
# LUTO2 is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# LUTO2 is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with
# LUTO2. If not, see <https://www.gnu.org/licenses/>.



"""
Data about transitions costs.
"""

import numpy as np
import luto.tools as tools
import luto.data as Data
import luto.economics.agricultural.ghg as ag_ghg

from functools import lru_cache
from luto import settings
from typing import Dict
from luto.data import Data
from luto.economics.agricultural.water import get_wreq_matrices



@lru_cache(maxsize=1)
def get_base_dvar_mj_cell_map(data: Data, base_year: int) -> dict:
    """Slice the base-year ag dvar by each (from_m, from_j), returning {(from_m, from_j): cell_idx}
    for every source combo whose dvar fraction exceeds `threshold` in at least one cell.

    ★ THIS IS THE KEY DESIGN FOR THE DELTA TRANSITION COST. We slice the base-year dvar by the same
    source (from_m, from_j) — e.g. dry-Apples. All cells in one slice share the same transition
    costs (the cost of leaving dry-Apples for each target). The solver then creates, for each slice,
    the same number of delta variables (delta >= 0, positive-increment Gurobi vars) — one per sliced
    cell — that represent the TRUE transition flow out of that source on those cells. Transition cost
    is then trans_cost = delta_dvars * cost_cells, and the objective minimises sum(trans_cost). This 
    is the per-source basis of the delta transition model (no single dominant-LU cost per cell).

    Uses settings.EXACT_REACHABILITY_MIN_FRACTION as the single fraction threshold (not a parameter,
    so every caller — and the exact GHG cell-slicing in ghg.py — shares one value). This map is the
    single source of truth for BOTH (a) the solver's per-source delta/cost slices and (b) target
    eligibility (`get_to_ag_exclude_matrices`). Keeping one threshold keeps the two consistent — a
    source below the threshold is dropped from both (it gets no flow-out var), and its sliver of land
    is conserved by the 'stay' fallback (`get_ag2ag_lb` locks it in place as itself), so nothing is lost.

    Cached (maxsize=1): the exclude builder and all exact cost functions call this for the same
    (data, base_year) pair within one solve step, so subsequent calls are free.
    """
    threshold = settings.EXACT_REACHABILITY_MIN_FRACTION
    base_dvar_mrj = data.ag_dvars[base_year]
    return {
        (m, j): np.where(base_dvar_mrj[m, :, j] > threshold)[0]
        for m in range(data.NLMS)
        for j in range(data.N_AG_LUS)
        if (base_dvar_mrj[m, :, j] > threshold).any()
    }


def get_to_ag_exclude_matrices(data: Data, base_year: int) -> np.ndarray:
    """To-ag target-eligibility (exclude) matrix (NLMS, NCELLS, N_AG_LUS) via per-source reachability.

    Covers BOTH ag→ag and non-ag→ag reachability. A cell r is eligible for ag target tj iff:

    - SOME land use present at r above `EXACT_REACHABILITY_MIN_FRACTION` — whether an agricultural
      source or a non-agricultural source — can transition to tj (finite T_MAT entry); AND
    - tj is spatially allowed (EXCLUDE) and not a no-go LU there.

    Notes:

    - The present sources come from `get_base_dvar_mj_cell_map` (ag) and
      `get_base_nonag_dvar_k_cell_map` (nonag) — the SAME maps that drive the solver's per-source
      flow-out slices, so eligibility and the flow vars share one threshold.
    - `ag_lu2cells` then derives straight from this matrix and cannot diverge from the solver's
      direct `ag_x_mrj[m, r, j]` reads.
    """
    # Lazy import to avoid the agricultural <-> non_agricultural transitions import cycle.
    from luto.economics.non_agricultural.transitions import get_base_nonag_dvar_k_cell_map

    mj_cell_map = get_base_dvar_mj_cell_map(data, base_year)
    k_cell_map  = get_base_nonag_dvar_k_cell_map(data, base_year)

    # Binary T_MAT allow/disallow matrices (finite → True, NaN → False)
    t_ag2ag_jj = ~np.isnan(
        data.T_MAT.sel(from_lu=data.AGRICULTURAL_LANDUSES, to_lu=data.AGRICULTURAL_LANDUSES).values
    )   # (N_AG_LUS_from, N_AG_LUS_to)
    t_nonag2ag_kj = ~np.isnan(
        data.T_MAT.sel(from_lu=data.NON_AGRICULTURAL_LANDUSES, to_lu=data.AGRICULTURAL_LANDUSES).values
    )   # (N_NON_AG_LUS, N_AG_LUS_to)

    # Per-source reachability union: mark every target reachable from any present source at r.
    reach_rj = np.zeros((data.NCELLS, data.N_AG_LUS), dtype=bool)
    for (_fm, fj), cells in mj_cell_map.items():
        if cells.size:
            reach_rj[cells] |= t_ag2ag_jj[fj]
    for k, cells in k_cell_map.items():
        if cells.size:
            reach_rj[cells] |= t_nonag2ag_kj[k]

    t_rj = reach_rj.astype(np.int8)  # (NCELLS, N_AG_LUS)

    # Spatial exclusion and no-go zones
    x_mrj = data.EXCLUDE.copy().astype(np.int8)

    no_go_x_mrj = np.ones_like(data.AG_L_MRJ)
    if settings.EXCLUDE_NO_GO_LU:
        for no_go_x_r, no_go_desc in zip(data.NO_GO_REGION_AG, data.NO_GO_LANDUSE_AG):
            no_go_x_mrj[:, :, data.DESC2AGLU[no_go_desc]] = no_go_x_r

    return (x_mrj * t_rj[np.newaxis, :, :] * no_go_x_mrj).astype(np.int8)


def get_ag2ag_ub(data: Data, base_year: int) -> np.ndarray:
    """ag→ag TARGET upper bound (NLMS, NCELLS, N_AG_LUS), FRACTIONAL.

    `ub[to_m, r, to_j]` is the product of four factors:

    - T_MAT: binary allow/disallow per (from_j → to_j) — `finite → 1`, `NaN → 0`.
    - fraction: reachable land share = `Σ_{from_j : T_MAT[from_j→to_j] finite} Σ_m base_dvar[m, r, from_j]`
      (base-year fractions of every source LU that can reach to_j; overlapping sources summed, not OR-ed).
    - no-go: user-defined LUs banned in specific regions.
    - spatial exclusion (`data.EXCLUDE`): LU never present in the SA2 region in 2010 → banned there.

    Example: cell is 0.2 Apples + 0.8 (reachable LUs); if Apples↛to_j then ub[to_j] = 0.8.
    ag-source component only — nonag2ag adds its own fraction in the combined ag ub (later step).
    """
    ag_dvar    = data.ag_dvars[base_year]                                                              # (NLMS, NCELLS, N_AG_LUS)
    
    # Transition exclusion (T_MAT): binary allow/disallow per (from_j → to_j).
    t_ag2ag_jj = (~np.isnan(
        data.T_MAT.sel(from_lu=data.AGRICULTURAL_LANDUSES, to_lu=data.AGRICULTURAL_LANDUSES).values
    )).astype(np.float32)                                                                              # (from_j, to_j)
    
    # Reachable land share: sum the base-year fractions of every source LU that can reach to_j.
    ag_frac_rj    = ag_dvar.sum(axis=0)                                                                # (NCELLS, from_j)
    reach_frac_rj = (ag_frac_rj @ t_ag2ag_jj).astype(np.float32)                                       # (NCELLS, to_j)

    # No-go exclusion: user-defined LUs banned in specific regions.
    no_go = np.ones((data.NLMS, data.NCELLS, data.N_AG_LUS), dtype=np.float32)
    if settings.EXCLUDE_NO_GO_LU:
        for no_go_x_r, no_go_desc in zip(data.NO_GO_REGION_AG, data.NO_GO_LANDUSE_AG):
            no_go[:, :, data.DESC2AGLU[no_go_desc]] = no_go_x_r

    # Spatial exclusion (data.EXCLUDE): LU never present in the SA2 region in 2010 → banned there.
    x_mrj = data.EXCLUDE.astype(np.float32)
    return (x_mrj * reach_frac_rj[np.newaxis, :, :] * no_go).astype(np.float32)


def get_ag2ag_lb(data: Data, base_year: int) -> np.ndarray:
    """ag→ag TARGET lower bound (NLMS, NCELLS, N_AG_LUS): the 'stay' floor for sub-threshold slivers.

    - In principle lb SHOULD be all 0s — any ag cell can be fully cleared and re-allocated, so no
      land use is ever forced to persist.
    - In practice we exempt tiny dvar fractions from the transition machinery: θ =
      EXACT_REACHABILITY_MIN_FRACTION is the cutoff below which a source is too small to be worth
      creating flow-dvars for. Sources with fraction ≤ θ are skipped from the real transition
      considering/processing and instead just "stay the same".
    - This lb pins those skipped slivers in place — `lb[m,r,j] = x_old[m,r,j]` — so their land is
      conserved instead of vanishing (`Σ x_old = ag_mask` stays satisfied).
    - Example: a cell of 0.01 Apples + 0.90 Beef + 0.09 Citrus with θ = 0.05 — Apples (< θ) just
      stays as itself, while only Beef and Citrus get flow-dvars and may transition.

    Floor-truncated at ROUND_DECIMALS; get_feasible_ag_cells_mrj unions in lb>0 cells so the stay
    var exists. NOT capped at the raw ag2ag ub: the final box is safe regardless (get_dvar_lb_ag
    clamps lb ≤ base, get_dvar_ub_ag raises ub ≥ base), and capping here would zero the pin exactly
    where EXCLUDE/no-go bans a held sliver — deleting its X var and making Σ X = ag_mask infeasible.
    """
    x_old = data.ag_dvars[base_year]
    noise = 10 ** (-settings.ROUND_DECIMALS)
    sliver = (x_old > noise) & (x_old <= settings.EXACT_REACHABILITY_MIN_FRACTION)

    scale = 10 ** settings.ROUND_DECIMALS
    lb = np.where(sliver, np.floor(x_old * scale) / scale, 0.0).astype(np.float32)

    # Report the lock-in instead of passing silently: pinned slivers cannot move until they exceed θ
    # (which the pin itself prevents), so this is the land θ takes out of the transition market. Pins
    # sitting on EXCLUDE/no-go-banned entries are flagged separately — there the pin is load-bearing
    # (without it the holding has no X var and Σ X = ag_mask goes infeasible).
    if np.any(lb > 0):
        pinned_ha = float((lb.sum(axis=(0, 2)) * data.REAL_AREA).sum())
        banned = int(((lb > 0) & (get_ag2ag_ub(data, base_year) == 0)).sum())
        msg = f"  └── Ag lb sliver pin (θ={settings.EXACT_REACHABILITY_MIN_FRACTION}): {int((lb > 0).sum()):,} entries, {pinned_ha:,.1f} ha locked in place"
        if banned:
            msg += f"; {banned:,} on EXCLUDE/no-go-banned entries (kept alive by the pin)"
        print(msg, flush=True)
    return lb


def get_transition_matrices_ag2ag(data: Data, yr_idx: int, from_m: int, from_j: int, cells=None, separate=False):
    """Source-parameterised ag2ag transition-cost primitive.

    Answers ONE question: transitioning FROM a single source (from_m, from_j) TO every target
    (to_m, to_j), on `cells` (default all NCELLS), what is the cost/water/GHG per cell? Returns
    (NLMS, len(cells), N_AG_LUS) [to_m, r, to_j] (separate=True → {component: same-shape array}).

    UNMASKED — no exclude / no "not-staying" mask. The diagonal ((to_m,to_j)==(from_m,from_j)) is
    naturally 0 (T_MAT[j→j]=0, zero water-delta, zero GHG), and target eligibility is the solver's job
    (a flow/delta var is created only for a valid transition).

    This is THE per-source cost primitive of the delta transition model. Callers select which cells to
    evaluate per source: the ag2ag/ag2nonag cost builds it on each source's dvar>θ cells
    (`get_base_dvar_mj_cell_map`, θ = EXACT_REACHABILITY_MIN_FRACTION); the nonag→ag levers call it for a
    uniform livestock source (cells=None → all cells). The year-invariant water / T_MAT terms are
    derived internally each call (water via the lru_cached `get_wreq_matrices`, so per-source calls in
    a dict loop stay a single compute per step).

    Components:
      - Establishment  : amortise(T_MAT[from_j → to_j] × mult) × REAL_AREA  (both target lm equal)
      - Water license  : amortised [(target req − source (from_m,from_j) req) × price + irrigation setup/teardown]
      - GHG (× carbon price) : natural→modified / unallocated-natural→livestock-natural release
    """
    yr_cal = data.YR_CAL_BASE + yr_idx
    if cells is None:
        cells = np.arange(data.NCELLS)
    n = len(cells)
    N_AG = data.N_AG_LUS
    area = data.REAL_AREA[cells]                                                    # (n,)

    # ── Establishment: source only enters via the T_MAT[from_j] row ──
    t_ij  = data.T_MAT.sel(from_lu=data.AGRICULTURAL_LANDUSES, to_lu=data.AGRICULTURAL_LANDUSES).values * data.TRANS_COST_MULTS[yr_cal]
    t_row = np.nan_to_num(t_ij[from_j]).astype(np.float32)                          # (N_AG,)
    e_rj  = tools.amortise(np.tile(t_row, (n, 1))) * area[:, None]                  # (n, N_AG)
    e_mrj = np.stack([e_rj, e_rj], axis=0).astype(np.float32)                       # (NLMS, n, N_AG)

    # ── Water licence delta (source-parameterised), amortised like Establishment above ──
    w_mrj       = get_wreq_matrices(data, yr_idx)                                   # <ML/cell> (lru_cached)
    w_raw_mrj   = tools.get_ag_to_ag_water_delta_matrix(data, from_m, from_j, cells, w_mrj, yr_idx)
    w_delta_mrj = tools.amortise(w_raw_mrj).astype(np.float32)

    # ── GHG release ($ = carbon price × raw emissions, amortised): source-parameterised ──
    price   = data.get_carbon_price_by_yr_idx(yr_idx)
    ghg_raw = ag_ghg.get_ghg_transition_emissions(data, from_m, from_j, cells, separate=True)   # raw t/cell
    ghg     = {k: tools.amortise(v * price).astype(np.float32) for k, v in ghg_raw.items()}

    if separate:
        return {'Establishment cost': e_mrj, 'Water license cost': w_delta_mrj, **ghg}
    return (e_mrj + w_delta_mrj + sum(ghg.values())).astype(np.float32)


def get_asparagopsis_effect_t_mrj(data: Data):
    """
    Gets the transition costs of asparagopsis taxiformis, which are none.
    Transition/establishment costs are handled in the costs matrix.
    """
    land_uses = settings.AG_MANAGEMENTS_TO_LAND_USES["Asparagopsis taxiformis"]
    return np.zeros((data.NLMS, data.NCELLS, len(land_uses))).astype(np.float32)


def get_precision_agriculture_effect_t_mrj(data: Data):
    """
    Gets the effects on transition costs of precision agriculture, which are none.
    Transition/establishment costs are handled in the costs matrix.
    """
    land_uses = settings.AG_MANAGEMENTS_TO_LAND_USES['Precision Agriculture']
    return np.zeros((data.NLMS, data.NCELLS, len(land_uses))).astype(np.float32)


def get_ecological_grazing_effect_t_mrj(data: Data):
    """
    Gets the effects on transition costs of ecological grazing, which are none.
    Transition/establishment costs are handled in the costs matrix.
    """
    land_uses = settings.AG_MANAGEMENTS_TO_LAND_USES['Ecological Grazing']
    return np.zeros((data.NLMS, data.NCELLS, len(land_uses))).astype(np.float32)


def get_savanna_burning_effect_t_mrj(data: Data):
    """
    Gets the effects on transition costs of savanna burning, which are none.
    Transition/establishment costs are handled in the costs matrix.
    """
    land_uses = settings.AG_MANAGEMENTS_TO_LAND_USES['Savanna Burning']
    return np.zeros((data.NLMS, data.NCELLS, len(land_uses))).astype(np.float32)


def get_agtech_ei_effect_t_mrj(data: Data):
    """
    Gets the effects on transition costs of AgTech EI, which are none.
    Transition/establishment costs are handled in the costs matrix.
    """
    land_uses = settings.AG_MANAGEMENTS_TO_LAND_USES['AgTech EI']
    return np.zeros((data.NLMS, data.NCELLS, len(land_uses))).astype(np.float32)


def get_biochar_effect_t_mrj(data: Data):
    """
    Gets the effects on transition costs of Biochar, which are none.
    Transition/establishment costs are handled in the costs matrix.
    """
    land_uses = settings.AG_MANAGEMENTS_TO_LAND_USES['Biochar']
    return np.zeros((data.NLMS, data.NCELLS, len(land_uses))).astype(np.float32)


def get_beef_hir_effect_t_mrj(data: Data):
    land_uses = settings.AG_MANAGEMENTS_TO_LAND_USES['HIR - Beef']
    return np.zeros((data.NLMS, data.NCELLS, len(land_uses))).astype(np.float32)


def get_sheep_hir_effect_t_mrj(data: Data):
    land_uses = settings.AG_MANAGEMENTS_TO_LAND_USES['HIR - Sheep']
    return np.zeros((data.NLMS, data.NCELLS, len(land_uses))).astype(np.float32)


def get_utility_solar_pv_effect_t_mrj(data: Data, yr_idx):
    """
    Returns zeros — CAPEX for Utility Solar PV has been moved to
    get_utility_solar_pv_effect_c_mrj (cost.py) as an amortised annual cost.
    """
    solar_lus = settings.AG_MANAGEMENTS_TO_LAND_USES['Utility Solar PV']
    return np.zeros((data.NLMS, data.NCELLS, len(solar_lus)), dtype=np.float32)


def get_onshore_wind_effect_t_mrj(data: Data, yr_idx):
    """
    Returns zeros — CAPEX for Onshore Wind has been moved to
    get_onshore_wind_effect_c_mrj (cost.py) as an amortised annual cost.
    """
    wind_lus = settings.AG_MANAGEMENTS_TO_LAND_USES['Onshore Wind']
    return np.zeros((data.NLMS, data.NCELLS, len(wind_lus)), dtype=np.float32)


def get_agricultural_management_transition_matrices(data: Data, yr_idx) -> Dict[str, np.ndarray]:
    
    asparagopsis_data = get_asparagopsis_effect_t_mrj(data)                     
    precision_agriculture_data = get_precision_agriculture_effect_t_mrj(data)   
    eco_grazing_data = get_ecological_grazing_effect_t_mrj(data)                
    sav_burning_data = get_savanna_burning_effect_t_mrj(data)                   
    agtech_ei_data = get_agtech_ei_effect_t_mrj(data)                           
    biochar_data = get_biochar_effect_t_mrj(data)                               
    beef_hir_data = get_beef_hir_effect_t_mrj(data)                             
    sheep_hir_data = get_sheep_hir_effect_t_mrj(data)
    utility_solar_pv_data = get_utility_solar_pv_effect_t_mrj(data, yr_idx)
    onshore_wind_data = get_onshore_wind_effect_t_mrj(data, yr_idx)
                  
    return {
        'Asparagopsis taxiformis': asparagopsis_data,
        'Precision Agriculture': precision_agriculture_data,
        'Ecological Grazing': eco_grazing_data,
        'Savanna Burning': sav_burning_data,
        'AgTech EI': agtech_ei_data,
        'Biochar': biochar_data,
        'HIR - Beef': beef_hir_data,
        'HIR - Sheep': sheep_hir_data,
        'Utility Solar PV': utility_solar_pv_data,
        'Onshore Wind': onshore_wind_data
    }


def get_asparagopsis_adoption_limits(data: Data, yr_idx):
    """
    Gets the adoption limit of Asparagopsis taxiformis for each possible land use.
    """
    if not settings.AG_MANAGEMENTS['Asparagopsis taxiformis']:
        return {data.DESC2AGLU[lu]: 0 for lu in settings.AG_MANAGEMENTS_TO_LAND_USES['Asparagopsis taxiformis']}
    
    asparagopsis_limits = {}
    yr_cal = data.YR_CAL_BASE + yr_idx
    for lu in settings.AG_MANAGEMENTS_TO_LAND_USES['Asparagopsis taxiformis']:
        j = data.DESC2AGLU[lu]
        asparagopsis_limits[j] = min(data.ASPARAGOPSIS_DATA[lu].loc[yr_cal, 'Technical_Adoption'] * settings.TECH_ADOPT_MULT, 1)

    return asparagopsis_limits


def get_precision_agriculture_adoption_limit(data: Data, yr_idx):
    """
    Gets the adoption limit of precision agriculture for each possible land use.
    """
    if not settings.AG_MANAGEMENTS['Precision Agriculture']:
        return {data.DESC2AGLU[lu]: 0 for lu in settings.AG_MANAGEMENTS_TO_LAND_USES['Precision Agriculture']}
    
    prec_agr_limits = {}
    yr_cal = data.YR_CAL_BASE + yr_idx
    for lu in settings.AG_MANAGEMENTS_TO_LAND_USES['Precision Agriculture']:
        j = data.DESC2AGLU[lu]
        prec_agr_limits[j] = min(data.PRECISION_AGRICULTURE_DATA[settings.LU2TYPE[lu]].loc[yr_cal, 'Technical_Adoption'] * settings.TECH_ADOPT_MULT, 1)

    return prec_agr_limits


def get_ecological_grazing_adoption_limit(data: Data, yr_idx):
    """
    Gets the adoption limit of ecological grazing for each possible land use.
    """
    if not settings.AG_MANAGEMENTS['Ecological Grazing']:
        return {data.DESC2AGLU[lu]: 0 for lu in settings.AG_MANAGEMENTS_TO_LAND_USES['Ecological Grazing']}
    
    eco_grazing_limits = {}
    yr_cal = data.YR_CAL_BASE + yr_idx
    for lu in settings.AG_MANAGEMENTS_TO_LAND_USES['Ecological Grazing']:
        j = data.DESC2AGLU[lu]
        eco_grazing_limits[j] = data.ECOLOGICAL_GRAZING_DATA[lu].loc[yr_cal, 'Feasible Adoption (%)']

    return eco_grazing_limits


def get_savanna_burning_adoption_limit(data: Data):
    """
    Gets the adoption limit of Savanna Burning for each possible land use
    """
    if not settings.AG_MANAGEMENTS['Savanna Burning']:
        return {data.DESC2AGLU[lu]: 0 for lu in settings.AG_MANAGEMENTS_TO_LAND_USES['Savanna Burning']}
    
    sav_burning_limits = {}
    for lu in settings.AG_MANAGEMENTS_TO_LAND_USES['Savanna Burning']:
        j = data.DESC2AGLU[lu]
        sav_burning_limits[j] = 1

    return sav_burning_limits


def get_agtech_ei_adoption_limit(data: Data, yr_idx):
    """
    Gets the adoption limit of AgTech EI for each possible land use.
    """
    if not settings.AG_MANAGEMENTS['AgTech EI']:
        return {data.DESC2AGLU[lu]: 0 for lu in settings.AG_MANAGEMENTS_TO_LAND_USES['AgTech EI']}
    
    agtech_ei_limits = {}
    yr_cal = data.YR_CAL_BASE + yr_idx
    for lu in settings.AG_MANAGEMENTS_TO_LAND_USES['AgTech EI']:
        j = data.DESC2AGLU[lu]
        agtech_ei_limits[j] = min(data.AGTECH_EI_DATA[settings.LU2TYPE[lu]].loc[yr_cal, 'Technical_Adoption'] * settings.TECH_ADOPT_MULT, 1)

    return agtech_ei_limits


def get_biochar_adoption_limit(data: Data, yr_idx):
    """
    Gets the adoption limit of Biochar for each possible land use.
    """
    if not settings.AG_MANAGEMENTS['Biochar']:
        return {data.DESC2AGLU[lu]: 0 for lu in settings.AG_MANAGEMENTS_TO_LAND_USES['Biochar']}
    
    biochar_limits = {}
    yr_cal = data.YR_CAL_BASE + yr_idx
    for lu in settings.AG_MANAGEMENTS_TO_LAND_USES['Biochar']:
        j = data.DESC2AGLU[lu]
        biochar_limits[j] = min(data.BIOCHAR_DATA[settings.LU2TYPE[lu]].loc[yr_cal, 'Technical_Adoption'] * settings.TECH_ADOPT_MULT, 1)

    return biochar_limits


def get_beef_hir_adoption_limit(data: Data):
    """
    Gets the adoption limit of HIR - Beef for each possible land use.
    """
    if not settings.AG_MANAGEMENTS['HIR - Beef']:
        return {data.DESC2AGLU[lu]: 0 for lu in settings.AG_MANAGEMENTS_TO_LAND_USES['HIR - Beef']}
    hir_limits = {}
    for lu in settings.AG_MANAGEMENTS_TO_LAND_USES['HIR - Beef']:
        j = data.DESC2AGLU[lu]
        hir_limits[j] = 1

    return hir_limits


def get_sheep_hir_adoption_limit(data: Data):
    """
    Gets the adoption limit of HIR - Sheep for each possible land use.
    """
    if not settings.AG_MANAGEMENTS['HIR - Sheep']:
        return {data.DESC2AGLU[lu]: 0 for lu in settings.AG_MANAGEMENTS_TO_LAND_USES['HIR - Sheep']}
    hir_limits = {}
    for lu in settings.AG_MANAGEMENTS_TO_LAND_USES['HIR - Sheep']:
        j = data.DESC2AGLU[lu]
        hir_limits[j] = 1

    return hir_limits

def get_utility_solar_pv_adoption_limit(data: Data):
    """
    Gets the adoption limit of Utility Solar PV for each possible land use.
    """
    if not settings.AG_MANAGEMENTS['Utility Solar PV']:
        return {data.DESC2AGLU[lu]: 0 for lu in settings.AG_MANAGEMENTS_TO_LAND_USES['Utility Solar PV']}
    solar_pv_limits = {}
    for lu in settings.AG_MANAGEMENTS_TO_LAND_USES['Utility Solar PV']:
        j = data.DESC2AGLU[lu]
        solar_pv_limits[j] = settings.RENEWABLES_ADOPTION_LIMITS['Utility Solar PV']

    return solar_pv_limits

def get_onshore_wind_adoption_limit(data: Data):
    """
    Gets the adoption limit of Onshore Wind for each possible land use.
    """
    if not settings.AG_MANAGEMENTS['Onshore Wind']:
        return {data.DESC2AGLU[lu]: 0 for lu in settings.AG_MANAGEMENTS_TO_LAND_USES['Onshore Wind']}
    
    wind_limits = {}
    for lu in settings.AG_MANAGEMENTS_TO_LAND_USES['Onshore Wind']:
        j = data.DESC2AGLU[lu]
        wind_limits[j] = settings.RENEWABLES_ADOPTION_LIMITS['Onshore Wind']

    return wind_limits

def get_agricultural_management_adoption_limits(data: Data, yr_idx) -> Dict[str, dict]:
    """
    An adoption limit represents the maximum percentage of cells (for each land use) that can utilise
    each agricultural management option.
    """
    ag_management_data = {}

    ag_management_data['Asparagopsis taxiformis'] = get_asparagopsis_adoption_limits(data, yr_idx)
    ag_management_data['Precision Agriculture'] = get_precision_agriculture_adoption_limit(data, yr_idx)
    ag_management_data['Ecological Grazing'] = get_ecological_grazing_adoption_limit(data, yr_idx)
    ag_management_data['Savanna Burning'] = get_savanna_burning_adoption_limit(data)
    ag_management_data['AgTech EI'] = get_agtech_ei_adoption_limit(data, yr_idx)
    ag_management_data['Biochar'] = get_biochar_adoption_limit(data, yr_idx)
    ag_management_data['HIR - Beef'] = get_beef_hir_adoption_limit(data)
    ag_management_data['HIR - Sheep'] = get_sheep_hir_adoption_limit(data)
    ag_management_data['Utility Solar PV'] = get_utility_solar_pv_adoption_limit(data)
    ag_management_data['Onshore Wind'] = get_onshore_wind_adoption_limit(data)
   
    return ag_management_data


def get_lower_bound_agricultural_management_matrices(data: Data, base_year) -> dict[str, dict]:
    """
    Returns per-am lower bounds (shape: NLMS × NCELLS × N_AG_LUS) for the next solve.

    Each am_lb[am][m, r, j] is the floor-truncated min(am_dvar, ag_dvar).  The clamp
    against ag_dvar corrects for FeasibilityTol: the solver enforces am ≤ ag as a linear
    constraint (not a variable bound), so the reported am_dvar can exceed ag_dvar by up to
    FeasibilityTol.  The correct "true" am value is min(am_dvar, ag_dvar) — equivalent to
    what a variable upper-bound on ag_dvar would have enforced exactly.
    """

    if base_year == data.YR_CAL_BASE or base_year not in data.ag_man_dvars:
        return {
            am: np.zeros((data.NLMS, data.NCELLS, data.N_AG_LUS), dtype=np.float32)
            for am in settings.AG_MANAGEMENTS_TO_LAND_USES
            if settings.AG_MANAGEMENTS[am]
        }

    ag_dvar = data.ag_dvars[base_year].astype(np.float32)  # (NLMS, NCELLS, N_AG_LUS)

    result = {}
    for am in settings.AG_MANAGEMENTS_TO_LAND_USES:
        if not settings.AG_MANAGEMENTS[am]:
            continue
        am_dvar = data.ag_man_dvars[base_year][am].astype(np.float32)
        am_dvar_true = tools.clamp_dvar_bound(am_dvar, 0.0, ag_dvar, f'Ag man lb clamped [{am}]')   # am cannot exceed its host ag
        am_lb = np.divide(
            np.floor(am_dvar_true * 10 ** settings.ROUND_DECIMALS),
            10 ** settings.ROUND_DECIMALS,
        )
        result[am] = am_lb
    return result


def get_regional_adoption_limits(data: Data, yr_cal: int):
    """
    Build per-region adoption caps for the solver.

    Returns
    -------
    ag_reg_adoption_constrs : list[[reg_id, lu_code, lu_name, reg_ind, area_limit_ha]]
        Per-(region, ag-landuse) caps from regional_adoption_zones.xlsx. 'on' mode only.
    non_ag_reg_adoption_constrs : list[[reg_id, lu_code, lu_name, reg_ind, area_limit_ha]]
        Per-(region, non-ag-landuse) caps from regional_adoption_zones.xlsx. 'on' mode only.
    non_ag_reg_adoption_sum_constrs : list[[reg_id, reg_ind, area_limit_ha]]
        Per-region SUM-of-all-non-ag caps. 'NON_AG_CAP' mode only.
    """
    if settings.REGIONAL_ADOPTION_CONSTRAINTS == "off":
        return [], [], []

    ag_reg_adoption_constrs = []
    non_ag_reg_adoption_constrs = []

    # Per-LU (ag + non-ag) caps from xlsx — only populated in 'on' mode
    for reg_id, lu_name, area_limit_ha in data.get_regional_adoption_limit_ha_by_year(yr_cal):
        reg_ind = np.where(data.REGIONAL_ADOPTION_ZONES == reg_id)[0]

        if lu_name in data.DESC2AGLU:
            lu_code = data.DESC2AGLU[lu_name]
            ag_reg_adoption_constrs.append([reg_id, lu_code, lu_name, reg_ind, area_limit_ha])
        elif lu_name in data.DESC2NONAGLU:
            lu_code = data.DESC2NONAGLU[lu_name] - settings.NON_AGRICULTURAL_LU_BASE_CODE
            non_ag_reg_adoption_constrs.append([reg_id, lu_code, lu_name, reg_ind, area_limit_ha])
        else:
            raise ValueError(f"Regional adoption constraint exists for unrecognised land use: {lu_name}")

    # SUM-of-non-ag per-region cap — only populated in 'NON_AG_CAP' mode
    non_ag_reg_adoption_sum_constrs = [
        [reg_id, reg_ind, area_limit_ha]
        for reg_id, reg_ind, area_limit_ha
        in data.get_regional_adoption_non_ag_sum_limit_ha_by_year(yr_cal)
    ]

    return ag_reg_adoption_constrs, non_ag_reg_adoption_constrs, non_ag_reg_adoption_sum_constrs
