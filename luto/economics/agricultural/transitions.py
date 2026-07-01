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
from joblib import Parallel, delayed
from luto import settings
from typing import Dict
from luto.data import Data
from luto.economics.agricultural.water import get_wreq_matrices



@lru_cache(maxsize=1)
def get_base_dvar_mj_cell_map(data: Data, base_year: int) -> dict:
    """Slice the base-year ag dvar by each (from_m, from_j), returning {(from_m, from_j): cell_idx}
    for every source combo whose dvar fraction exceeds `threshold` in at least one cell. Only used
    in exact mode.

    ★ THIS IS THE KEY DESIGN FOR EXACT TRANSITION COST. We slice the base-year dvar by the same
    source (from_m, from_j) — e.g. dry-Apples. All cells in one slice share the same transition
    costs (the cost of leaving dry-Apples for each target). The solver then creates, for each slice,
    the same number of delta variables (delta >= 0, positive-increment Gurobi vars) — one per sliced
    cell — that represent the TRUE transition flow out of that source on those cells. Transition cost
    is then trans_cost = delta_dvars * cost_cells, and the objective minimises sum(trans_cost). This
    replaces the crisp/blend approximation where a whole cell carried a single dominant-LU cost.

    Uses settings.EXACT_REACHABILITY_MIN_FRACTION as the single fraction threshold (not a parameter,
    so every caller — and the exact GHG cell-slicing in ghg.py — shares one value). This map is the
    single source of truth for BOTH (a) the solver's per-source delta/cost slices and (b) target
    eligibility (`get_to_ag_exclude_matrices_exact`). Keeping one threshold keeps the two consistent — a
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


def get_to_ag_exclude_matrices_crisp(data: Data, lumap: np.ndarray) -> np.ndarray:
    """Return x_mrj exclude matrices (NLMS, NCELLS, N_AG_LUS) based on the dominant LU (crisp/blend).

    Derived as the union of the two per-source TO-ag upper bounds: ag2ag (ag-dominant cells reach ag
    targets from their dominant ag LU) and nonag2ag (non-ag-dominant cells reach ag targets from their
    current non-ag LU AND 2010 ag status). Since cell dominance partitions ag vs non-ag cells, the two
    components are disjoint and `(ag2ag_ub + nonag2ag_ub) > 0` reproduces the old monolithic exclude
    bit-for-bit — for the real base-year map AND the cost primitive's synthetic single-source maps.
    Single source of truth: the crisp exclude logic now lives only in the two ub builders.
    """
    # Lazy import to avoid the agricultural <-> non_agricultural transitions import cycle.
    from luto.economics.non_agricultural.transitions import get_nonag2ag_ub_crisp

    to_ag_ub = get_ag2ag_ub_crisp(data, lumap) + get_nonag2ag_ub_crisp(data, lumap)          # (NLMS, NCELLS, N_AG_LUS)
    return (to_ag_ub > 0).astype(np.int8)


def get_to_ag_exclude_matrices_exact(data: Data, base_year: int) -> np.ndarray:
    """Return x_mrj exclude matrices (NLMS, NCELLS, N_AG_LUS) via per-source reachability (exact mode).

    A cell r is eligible for ag target tj if and only if SOME land use present at r above
    `EXACT_REACHABILITY_MIN_FRACTION` can transition to tj (finite T_MAT entry), AND tj is
    spatially allowed (EXCLUDE) and not a no-go LU there. The present sources come from
    `get_base_dvar_mj_cell_map` (ag) and `get_base_nonag_dvar_k_cell_map` (nonag) — the SAME maps
    that drive the solver's per-source flow-out slices, so eligibility and the flow vars share one
    threshold. `ag_lu2cells` then derives straight from this matrix and cannot diverge from the
    solver's direct `ag_x_mrj[m, r, j]` reads.
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

    # Spatial exclusion and no-go zones (same logic as crisp)
    x_mrj = data.EXCLUDE.copy().astype(np.int8)

    no_go_x_mrj = np.ones_like(data.AG_L_MRJ)
    if settings.EXCLUDE_NO_GO_LU:
        for no_go_x_r, no_go_desc in zip(data.NO_GO_REGION_AG, data.NO_GO_LANDUSE_AG):
            no_go_x_mrj[:, :, data.DESC2AGLU[no_go_desc]] = no_go_x_r

    return (x_mrj * t_rj[np.newaxis, :, :] * no_go_x_mrj).astype(np.int8)


def get_to_ag_exclude_matrices(data: Data, lumap: np.ndarray = None, base_year: int = None) -> np.ndarray:
    """Dispatch to the appropriate exclude matrix builder based on TRANSITION_MODE.

    crisp / blend: uses dominant-LU lumap  (requires lumap).
    exact:         uses base-year dvar reachability (requires base_year).
    """
    if settings.TRANSITION_MODE in ('crisp', 'blend'):
        return get_to_ag_exclude_matrices_crisp(data, lumap)
    elif settings.TRANSITION_MODE == 'exact':
        return get_to_ag_exclude_matrices_exact(data, base_year)
    else:
        raise ValueError(f"Unknown TRANSITION_MODE: {settings.TRANSITION_MODE!r}")


def get_ag2ag_ub_crisp(data: Data, lumap: np.ndarray) -> np.ndarray:
    """ag→ag TARGET upper bound (NLMS, NCELLS, N_AG_LUS), 0/1, from the dominant LU in `lumap`.

    TO-view bound (which ag target a cell may become), ag-source component only. For a cell whose
    dominant LU is ag LU j, target to_j is allowed (1) iff T_MAT[j→to_j] is finite, spatially
    allowed (data.EXCLUDE), and not no-go. Non-ag-dominant cells contribute 0 here (their reach to
    ag targets is the nonag2ag ub). crisp/blend share this (blend's exclusion is lumap-based too).

    Parametrised by `lumap` (not base_year) so it serves BOTH the TO-view (lumap = real base-year map,
    passed by the dispatcher) AND the cost primitive's synthetic single-source maps (all-sheep, etc.).
    Together with get_nonag2ag_ub_crisp it reconstructs get_to_ag_exclude_matrices_crisp exactly.
    """
    t_ij  = data.T_MAT.sel(from_lu=data.AGRICULTURAL_LANDUSES, to_lu=data.AGRICULTURAL_LANDUSES).values  # (from_j, to_j)
    ag_cells, _ = tools.get_ag_and_non_ag_cells(lumap)

    # Transition exclusion (T_MAT): target to_j reachable from the cell's dominant ag LU?
    # NaN default → non-ag cells (and forbidden ag transitions) collapse to 0; allowed ag → 1.
    t_rj = np.full((data.NCELLS, data.N_AG_LUS), np.nan, dtype=np.float32)
    t_rj[ag_cells] = t_ij[lumap[ag_cells]]
    reach_rj = np.where(np.isnan(t_rj), 0.0, 1.0).astype(np.float32)

    # No-go exclusion: user-defined LUs banned in specific regions.
    no_go = np.ones((data.NLMS, data.NCELLS, data.N_AG_LUS), dtype=np.float32)
    if settings.EXCLUDE_NO_GO_LU:
        for no_go_x_r, no_go_desc in zip(data.NO_GO_REGION_AG, data.NO_GO_LANDUSE_AG):
            no_go[:, :, data.DESC2AGLU[no_go_desc]] = no_go_x_r

    # Spatial exclusion (data.EXCLUDE): LU never present in the SA2 region in 2010 → banned there.
    x_mrj = data.EXCLUDE.astype(np.float32)
    return (x_mrj * reach_rj[np.newaxis, :, :] * no_go).astype(np.float32)


def get_ag2ag_ub_exact(data: Data, base_year: int) -> np.ndarray:
    """ag→ag TARGET upper bound (NLMS, NCELLS, N_AG_LUS), FRACTIONAL.

    ub[to_m, r, to_j] = (fraction of cell r's ag land that may reach to_j) × data.EXCLUDE × no-go,
    where the reachable fraction = Σ_{from_j : T_MAT[from_j→to_j] finite} Σ_m base_dvar[m, r, from_j].
    Example: cell is 0.2 Apples + 0.8 (reachable LUs); if Apples↛to_j then ub[to_j] = 0.8. This is the
    fractional generalisation of the crisp 0/1 (overlapping sources summed, not OR-ed). ag-source
    component only — nonag2ag adds its own fraction in the combined ag ub (later step).
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


def get_ag2ag_ub(data: Data, base_year: int) -> np.ndarray:
    """Dispatch the ag→ag target upper bound by TRANSITION_MODE.

    crisp / blend: 0/1 from the dominant base LU (get_ag2ag_ub_crisp).
    exact:         fractional reachable-land share (get_ag2ag_ub_exact).
    """
    if settings.TRANSITION_MODE in ('crisp', 'blend'):
        return get_ag2ag_ub_crisp(data, data.lumaps[base_year])
    elif settings.TRANSITION_MODE == 'exact':
        return get_ag2ag_ub_exact(data, base_year)
    else:
        raise ValueError(f"Unknown TRANSITION_MODE: {settings.TRANSITION_MODE!r}")


def get_ag2ag_lb(data: Data, base_year: int) -> np.ndarray:
    """ag→ag TARGET lower bound (NLMS, NCELLS, N_AG_LUS): the exact-mode 'stay' fallback (Q4, source side).

    θ = EXACT_REACHABILITY_MIN_FRACTION filters which dvars are eligible for exact flow transition.
    E.g. a cell of 0.01 Apples + 0.90 Beef + 0.09 Citrus: with θ = 0.05, Apples (< θ) just "stays the
    same", while only Beef and Citrus are given flow-dvars and may transition. This lb pins those stayed
    slivers — lb[m,r,j] = x_old[m,r,j] — so their land is conserved instead of vanishing (Σx_old = ag_mask).

    Exact only (crisp/blend have no slivers). Floor-truncated at ROUND_DECIMALS and capped at the exact
    ub so lb ≤ ub by construction; get_feasible_ag_cells_mrj unions in lb>0 cells so the stay var exists.
    """
    if settings.TRANSITION_MODE != 'exact':
        return np.zeros((data.NLMS, data.NCELLS, data.N_AG_LUS), dtype=np.float32)

    x_old = data.ag_dvars[base_year]
    noise = 10 ** (-settings.ROUND_DECIMALS)
    sliver = (x_old > noise) & (x_old <= settings.EXACT_REACHABILITY_MIN_FRACTION)

    scale = 10 ** settings.ROUND_DECIMALS
    lb = np.where(sliver, np.floor(x_old * scale) / scale, 0.0).astype(np.float32)

    # Cap at the exact ag2ag ub so a stay floor can never exceed the cell's allowable bound: the
    # sliver's own diagonal is part of ag2ag_ub, and lb ≤ ag2ag_ub ≤ combined dvar_ub_ag, so lb ≤ ub.
    ub_ag2ag = get_ag2ag_ub_exact(data, base_year)
    return np.minimum(lb, ub_ag2ag).astype(np.float32)

# Even been used multiple times for getting sheep/beef agroforestry, this function is not cacheable
# because using np.array as input (np.array is not hashable, and hashable is required for caching).
def get_transition_matrices_ag2ag_base(data: Data, yr_idx: int, base_lumap: np.ndarray, base_lmmap: np.ndarray, separate=False, w_mrj=None, t_ij=None):
    """
    Base ag2ag transition-cost primitive: flat (NLMS, NCELLS, N_AG_LUS), the cost for each cell
    to switch to (to_m, to_j) given its base_lumap/base_lmmap source. This is the single shared
    building block all three modes build on (called with a uniform single-source map to get
    "cost from one source"):
      - crisp : base(actual lumap/lmmap) grouped by dominant (from_m, from_j)
      - blend : base(uniform source) weighted by the source's base-year dvar fraction
      - exact : base(uniform source) sliced to the source's cells (get_base_dvar_mj_cell_map)
    It is also reused directly by the nonag->ag cost functions ("cost from one source").

    Calculate the transition matrices for land-use and land management transitions.
    Args:
        data (Data object): The data object containing the necessary input data.
        yr_idx (int): The index of the current year.
        base_lumap (np.ndarray): Land use map of the base year for the transitions.
        base_lmmap (np.ndarray): Land management map of the base year for the transitions.
        separate (bool, optional): Whether to return separate cost matrices for each cost component.
                                   Defaults to False.
        w_mrj (np.ndarray, optional): Precomputed water requirement matrices, to avoid recomputation
                                   when this function is called repeatedly for the same yr_idx.
        t_ij (np.ndarray, optional): Precomputed lexicographical transition-cost matrix, to avoid
                                   recomputation when this function is called repeatedly for the same yr_idx.
    Returns:
            numpy.ndarray or dict: The transition matrices for land-use and land management transitions.
                               If `separate` is False, returns a numpy array representing the total costs.
                               If `separate` is True, returns a dictionary with separate cost matrices for
                               establishment costs, Water license cost, and carbon releasing costs.
    """
    yr_cal = data.YR_CAL_BASE + yr_idx

    # Return l_mrj (Boolean) for current land-use and land management
    l_mrj = tools.lumap2ag_l_mrj(base_lumap, base_lmmap)
    l_mrj_not = np.logical_not(l_mrj)

    # Get the exclusion matrix
    x_mrj = get_to_ag_exclude_matrices_crisp(data, base_lumap)

    ag_cells, _ = tools.get_ag_and_non_ag_cells(base_lumap)

    n_ag_lms, ncells, n_ag_lus = data.AG_L_MRJ.shape

    # -------------------------------------------------------------- #
    # Transition costs (upfront, amortised to annual, per cell).     #
    # -------------------------------------------------------------- #

    # Raw transition-cost matrix is in $/ha and lexigraphically ordered (shape: land-use x land-use).
    if t_ij is None:
        t_ij = data.T_MAT.sel(from_lu=data.AGRICULTURAL_LANDUSES, to_lu=data.AGRICULTURAL_LANDUSES).values * data.TRANS_COST_MULTS[yr_cal]

    # Non-irrigation related transition costs for cell r to change to land-use j calculated based on lumap (in $/ha).
    # Only consider for cells currently being used for agriculture.
    e_rj = np.zeros((ncells, n_ag_lus)).astype(np.float32)
    e_rj[ag_cells, :] = t_ij[base_lumap[ag_cells]]

    # Amortise upfront costs to annualised costs and converted to $ per cell via REAL_AREA
    e_rj = tools.amortise(e_rj) * data.REAL_AREA[:, None]

    # Repeat the transition costs into dryland and irrigated land management types
    e_mrj = np.stack([e_rj, e_rj], axis=0)

    # Update the cost matrix with exclude matrices; the transition cost for a cell that remain the same is 0.
    e_mrj = np.einsum('mrj,mrj,mrj->mrj', e_mrj, x_mrj, l_mrj_not).astype(np.float32)
    e_mrj = np.nan_to_num(e_mrj)

    # -------------------------------------------------------------- #
    # Water license cost (upfront, amortised to annual, per cell).   #
    # -------------------------------------------------------------- #

    if w_mrj is None:
        w_mrj = get_wreq_matrices(data, yr_idx)                                                 # <unit: ML/cell>
    w_delta_mrj = tools.get_ag_to_ag_water_delta_matrix(w_mrj, l_mrj, data, yr_idx)
    w_delta_mrj = np.einsum('mrj,mrj,mrj->mrj', w_delta_mrj, x_mrj, l_mrj_not).astype(np.float32)

    # -------------------------------------------------------------- #
    # Carbon costs of transitioning cells.                           #
    # -------------------------------------------------------------- #

    # Apply the cost of carbon released by transitioning natural land to modified land
    ghg_transition = ag_ghg.get_ghg_transition_emissions(data, base_lumap, separate=True)       # <unit: t/ha>
        
    ghg_transition = {
        k:np.einsum('mrj,mrj,mrj->mrj', v, x_mrj, l_mrj_not).astype(np.float32)                 # No GHG penalty for cells that remain the same, or are prohibited from transitioning
        for k, v in ghg_transition.items()
    }
    
    ghg_transition = {
        k:tools.amortise(v * data.get_carbon_price_by_yr_idx(yr_idx))                           # Amortise the GHG penalties
        for k,v in ghg_transition.items()
    }
    
    ghg_t_types = ghg_transition.keys()
    ghg_t_smrj = np.stack([ghg_transition[t] for t in ghg_t_types], axis=0)                     # s: ghg_t_types, m: land management, r: cell, j: land use
    ghg_t_mrj = np.einsum('smrj->mrj', ghg_t_smrj)

    
    # TODO: add cost of biodiversity loss/gain from land-use transitions.
    
    # -------------------------------------------------------------- #
    # Total costs.                                                   #
    # -------------------------------------------------------------- #

    if separate:
        return {'Establishment cost': e_mrj, 'Water license cost': w_delta_mrj, **ghg_transition}
    else:
        t_mrj = e_mrj + w_delta_mrj + ghg_t_mrj
        return t_mrj


def get_transition_matrices_ag2ag_crisp(data: Data, yr_idx: int, base_lumap: np.ndarray, base_lmmap: np.ndarray, separate=False, w_mrj=None, t_ij=None):
    """Crisp ag2ag transition cost in flow (from_m, from_j) format.

    Returns dict[(from_m, from_j)] -> ndarray(NLMS, ncells_src, N_AG_LUS)  [to_m, r, to_j]
    (separate=True -> the leaf is {component: same-shape array}), where ncells_src is the number
    of cells whose DOMINANT base-year LU is (from_m, from_j).

    crisp assigns each cell exactly one dominant (from_m, from_j) via the integerised
    lumap/lmmap, so the flat building block `flat[:, r, :]` IS the cost from cell r's dominant
    source. We compute the flat matrix once (get_transition_matrices_ag2ag_base) and slice
    it per source — no recompute. Collapsing this dict back (scatter each slice into its cells)
    reproduces the flat matrix exactly on ag-dominant cells. Non-ag-dominant cells are not keyed
    here — their ag transition is the nonag2ag direction.
    """
    flat = get_transition_matrices_ag2ag_base(
        data, yr_idx, base_lumap, base_lmmap, separate, w_mrj=w_mrj, t_ij=t_ij
    )
    result = {}
    for from_m in range(data.NLMS):
        for from_j in range(data.N_AG_LUS):
            cells = np.where((base_lmmap == from_m) & (base_lumap == from_j))[0]
            if cells.size == 0:
                continue
            if separate:
                result[(from_m, from_j)] = {k: v[:, cells, :] for k, v in flat.items()}
            else:
                result[(from_m, from_j)] = flat[:, cells, :]
    return result


def get_transition_matrices_ag2ag_blend(data: Data, yr_idx: int, base_year: int, separate=False):
    """Blend ag2ag transition cost in flow (from_m, from_j) format.

    Per-source contribution = (source's base-year dvar fraction) × (`base` cost from that source)
    / (eligible-source denominator) — i.e. exactly one term of the fractional-dvar-weighted blend
    sum, kept unsummed. Scatter-summing the dict over sources reproduces the flat blend to float
    tolerance; sources whose fraction is below the rounding threshold are dropped (their
    frac × cost contribution is negligible).

    Returns dict[(from_m, from_j)] -> ndarray(NLMS, ncells_src, N_AG_LUS)  [to_m, r, to_j]
    (separate=True -> the leaf is {component: same-shape array}), with cells = the source's
    present cells (base-year dvar fraction above the rounding threshold).
    """
    yr_cal   = data.YR_CAL_BASE + yr_idx
    ag_X_mrj = data.ag_dvars[base_year]

    # Hoist invariants shared across all sources (same as the flat blend).
    w_mrj = get_wreq_matrices(data, yr_idx)
    t_ij  = data.T_MAT.sel(from_lu=data.AGRICULTURAL_LANDUSES, to_lu=data.AGRICULTURAL_LANDUSES).values * data.TRANS_COST_MULTS[yr_cal]

    valid_mask       = ~np.isnan(t_ij)
    ag_frac_per_lu   = ag_X_mrj.sum(axis=0)
    eligible_rj      = ag_frac_per_lu @ valid_mask
    eligible_rj_safe = np.where(eligible_rj > 10 ** (-settings.ROUND_DECIMALS), eligible_rj, 1.0)

    threshold = 10 ** (-settings.ROUND_DECIMALS)
    present   = [(m, j) for m in range(data.NLMS) for j in range(data.N_AG_LUS)
                 if (ag_X_mrj[m, :, j] > threshold).any()]

    def _compute_from(m, j):
        cells       = np.where(ag_X_mrj[m, :, j] > threshold)[0]
        all_j_lumap = np.full(data.NCELLS, j, dtype=np.int64)
        all_m_lumap = np.full(data.NCELLS, m, dtype=np.int64)
        base = get_transition_matrices_ag2ag_base(
            data, yr_idx, all_j_lumap, all_m_lumap, separate, w_mrj=w_mrj, t_ij=t_ij
        )
        # per-source factor[r, to_j] = frac[m,r,j] / eligible[r, to_j]  (broadcast over to_m)
        factor = ag_X_mrj[m, cells, j][:, None] / eligible_rj_safe[cells, :]
        if separate:
            return (m, j), {k: (v[:, cells, :] * factor[None, :, :]).astype(np.float32) for k, v in base.items()}
        return (m, j), (base[:, cells, :] * factor[None, :, :]).astype(np.float32)

    result = {}
    n_jobs = settings.TRANSITION_MODE_N_JOBS
    for i in range(0, len(present), n_jobs):
        batch = present[i:i + n_jobs]
        for key, arr in Parallel(n_jobs=n_jobs, backend="threading")(delayed(_compute_from)(m, j) for m, j in batch):
            result[key] = arr
    return result


def get_transition_matrices_ag2ag_exact(data: Data, base_year: int, target_year: int, mj_cell_map: dict, separate=False):
    """
    Create exact transition matrices for every (from_m, from_j) combo in mj_cell_map.

    Cost components (all $/cell/yr, amortised):
      - Establishment cost   — T_MAT lookup × area
      - Water licence delta  — (target – base water need) × price
      - GHG penalty          — lvstk-natural → modified only

    Parameters
    ----------
    mj_cell_map : dict returned by get_exact_mj_cell_map(data, base_year)

    Returns
    -------
    dict[(from_m, from_j)] → ndarray (NLMS, len(cell_idx), N_AG_LUS) in $/cell/yr
    OR, if separate=True:
    dict[(from_m, from_j)] → {'Establishment cost', 'Water license cost',
                               'Livestock natural to modified'}  each same shape
    """
    yr_idx = target_year - data.YR_CAL_BASE

    # Hoist invariants outside the per-(from_m, from_j) loop
    #   (N_AG_LUS, N_AG_LUS); NaN = prohibited transition
    t_ij = (
        data.T_MAT.sel(from_lu=data.AGRICULTURAL_LANDUSES, to_lu=data.AGRICULTURAL_LANDUSES).values
        * data.TRANS_COST_MULTS[target_year]
    )
    # Single target-year water matrix — mirrors crisp: w_r = (w_mrj * l_mrj).sum() uses
    # target-year req at the base-year LU, so exact does the same via w_mrj[from_m, cell_idx, from_j]
    w_req_mrj = get_wreq_matrices(data, yr_idx)  # (NLMS, NCELLS, N_AG_LUS), ML/cell

    result = {}

    for (from_m, from_j), cell_idx in mj_cell_map.items():

        # --- Establishment cost: $/cell/yr ---
        # NaN in t_ij = prohibited transition → 0 cost (exclude enforced in solver)
        t_row = np.nan_to_num(t_ij[from_j]).astype(np.float32)                        # (N_AG_LUS,)
        t_rj  = tools.amortise(np.tile(t_row, (len(cell_idx), 1))) * data.REAL_AREA[cell_idx, None]
        t_mrj = np.stack([t_rj, t_rj], axis=0).astype(np.float32)                    # (NLMS, ncells*, N_AG_LUS)

        # --- Water licence delta cost: $/cell/yr ---
        # w_base: target-yr req at from_LU — matches crisp's w_r = (w_mrj * l_mrj).sum()
        w_target  = w_req_mrj[:, cell_idx, :] * settings.INCLUDE_WATER_LICENSE_COSTS
        w_base    = w_req_mrj[from_m, cell_idx, from_j]                              # (ncells*,)
        w_cost    = (w_target - w_base[None, :, None]) * data.WATER_LICENCE_PRICE[cell_idx, None] * data.WATER_LICENSE_COST_MULTS[target_year]
        irrig_cell = settings.NEW_IRRIG_COST    * data.IRRIG_COST_MULTS[target_year] * data.REAL_AREA[cell_idx, None]
        deirr_cell = settings.REMOVE_IRRIG_COST * data.IRRIG_COST_MULTS[target_year] * data.REAL_AREA[cell_idx, None]
        if from_m == 0:    # was dryland → setup cost when switching to irrigated (m=1)
            w_cost[1] += irrig_cell
        else:              # was irrigated → teardown cost when switching to dryland (m=0)
            w_cost[0] += deirr_cell
        w_cost_mrj = tools.amortise(w_cost).astype(np.float32)                       # (NLMS, ncells*, N_AG_LUS)

        # --- GHG transition cost: $/cell/yr (all three source-type penalties) ---
        carbon_price    = data.get_carbon_price_by_yr_idx(yr_idx)
        ghg_lvstk_mod   = tools.amortise(ag_ghg.get_ghg_lvstk_natural_to_modified_exact(data, base_year, from_m, from_j)      * carbon_price).astype(np.float32)
        ghg_unall_lvstk = tools.amortise(ag_ghg.get_ghg_unall_natural_to_lvstk_natural_exact(data, base_year, from_m, from_j) * carbon_price).astype(np.float32)
        ghg_unall_mod   = tools.amortise(ag_ghg.get_ghg_unall_natural_to_modified_exact(data, base_year, from_m, from_j)      * carbon_price).astype(np.float32)

        if separate:
            result[(from_m, from_j)] = {
                'Establishment cost':                       t_mrj,
                'Water license cost':                       w_cost_mrj,
                'Livestock natural to unallocated natural': np.zeros_like(ghg_lvstk_mod),
                'Unallocated natural to livestock natural': ghg_unall_lvstk,
                'Livestock natural to modified':            ghg_lvstk_mod,
                'Unallocated natural to modified':          ghg_unall_mod,
            }
        else:
            result[(from_m, from_j)] = t_mrj + w_cost_mrj + ghg_lvstk_mod + ghg_unall_lvstk + ghg_unall_mod

    return result


def get_transition_matrices_ag2ag_from_base_year(data: Data, yr_idx, base_year, separate=False):
    """Dispatch to the from-based ag2ag transition cost (dict[(from_m, from_j)]) by TRANSITION_MODE."""
    if settings.TRANSITION_MODE == 'crisp':
        return get_transition_matrices_ag2ag_crisp(data, yr_idx, data.lumaps[base_year], data.lmmaps[base_year], separate)
    elif settings.TRANSITION_MODE == 'blend':
        return get_transition_matrices_ag2ag_blend(data, yr_idx, base_year, separate)
    elif settings.TRANSITION_MODE == 'exact':
        yr_cal = data.YR_CAL_BASE + yr_idx
        mj_cell_map = get_base_dvar_mj_cell_map(data, base_year)
        return get_transition_matrices_ag2ag_exact(data, base_year, yr_cal, mj_cell_map, separate)
    else:
        raise ValueError(f"Unknown TRANSITION_MODE: {settings.TRANSITION_MODE}")


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
        am_dvar_true = np.minimum(am_dvar, ag_dvar)          # clamp: am cannot exceed its host ag
        am_lb = np.divide(
            np.floor(am_dvar_true * 10 ** settings.ROUND_DECIMALS),
            10 ** settings.ROUND_DECIMALS,
        )
        clamped = am_dvar_true < am_dvar
        if clamped.any():
            gap = am_dvar[clamped] - am_dvar_true[clamped]
            print(
                f"  └── Ag man lb clamped [{am}]: {clamped.sum()} cells, max gap={gap.max():.2e}, mean gap={gap.mean():.2e}"
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
