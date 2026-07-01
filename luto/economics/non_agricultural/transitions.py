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

import numpy as np
from functools import lru_cache

import luto.tools as tools
import luto.economics.agricultural.water as ag_water
import luto.economics.agricultural.ghg as ag_ghg

from luto import settings
from luto.data import Data
from luto.economics.agricultural.transitions import (
    get_base_dvar_mj_cell_map,
    get_transition_matrices_ag2ag_base,
)



# TODO: Ag to Non-Ag GHG transition costs are omitted; 
#   Need to think about whether to include them, and if so, how to calculate them (e.g., using ag_ghg.get_ghg_costs_from_ag_to_nonag()).
def get_env_plant_transitions_from_ag_crisp(data: Data, base_year: int, target_year: int, separate=False) -> dict:
    """EP transition cost (ag → Environmental Plantings) in flow (from_m, from_j) format.

    Returns dict[(from_m, from_j)] -> ndarray(ncells_src,) in $/cell/yr
    (separate=True -> the leaf is {component: ndarray(ncells_src,)}), where ncells_src is the
    number of cells whose DOMINANT base-year LU is (from_m, from_j).

    crisp assigns each cell exactly one dominant (from_m, from_j) via the integerised lumap/lmmap,
    so the flat per-cell cost `flat[r]` IS the cost from cell r's dominant source. We compute the
    flat per-cell cost once and slice it per source — no recompute. Collapsing the dict back
    (scatter each slice into its cells) reproduces the flat crisp cost on ag-dominant cells.
    Mirrors get_env_plant_transitions_from_ag_exact's shape.
    """
    lumap = data.lumaps[base_year]
    lmmap = data.lmmaps[base_year]
    cells = np.isin(lumap, np.array(list(data.AGLU2DESC.keys())))

    # ── Flat per-cell components (cost from each cell's actual dominant source) ──
    # Establishment costs
    est_costs_r = tools.amortise(data.EP_EST_COST_HA * data.REAL_AREA * data.EST_COST_MULTS[target_year]).astype(np.float32)
    est_costs_r[~cells] = 0.0

    # Water costs; Assume EP is dryland
    w_rm_irrig_cost_r = (np.where(lmmap == 1, settings.REMOVE_IRRIG_COST * data.IRRIG_COST_MULTS[target_year], 0) * data.REAL_AREA).astype(np.float32)
    w_rm_irrig_cost_r[~cells] = 0.0

    # Transition costs
    ag_to_ep_j = data.T_MAT.sel(from_lu=data.AGRICULTURAL_LANDUSES, to_lu='Environmental Plantings').values
    ag_to_ep_t_r = np.vectorize(dict(enumerate(ag_to_ep_j)).get, otypes=['float32'])(lumap)
    ag_to_ep_t_r = np.nan_to_num(ag_to_ep_t_r)
    ag_to_ep_t_r = tools.amortise(ag_to_ep_t_r * data.REAL_AREA)
    ag_to_ep_t_r[~cells] = 0.0

    if separate:
        flat = {
            'Establishment cost (Ag2Non-Ag)':     est_costs_r,
            'Transition cost (Ag2Non-Ag)':        ag_to_ep_t_r,
            'Remove irrigation cost (Ag2Non-Ag)': w_rm_irrig_cost_r,
        }
    else:
        flat = est_costs_r + ag_to_ep_t_r + w_rm_irrig_cost_r

    # ── Slice the flat cost by dominant (from_m, from_j) source ──
    result = {}
    for from_m in range(data.NLMS):
        for from_j in range(data.N_AG_LUS):
            src_cells = np.where((lmmap == from_m) & (lumap == from_j))[0]
            if src_cells.size == 0:
                continue
            if separate:
                result[(from_m, from_j)] = {k: v[src_cells] for k, v in flat.items()}
            else:
                result[(from_m, from_j)] = flat[src_cells]
    return result


def get_env_plant_transitions_from_ag_blend(data: Data, base_year: int, target_year: int, separate=False) -> dict:
    """Blended EP transition cost (ag → EP) in flow (from_m, from_j) format.

    Per-source contribution = one term of the flat blend's weighted average, kept unsummed.
    Scatter-summing the dict over sources reproduces the flat blend on cells with ag fraction
    above the rounding threshold. The flat blend's three components use *different*
    normalisations, so they are decomposed separately:
      - Establishment & T_MAT: weighted by frac_s / ag_frac (sum → whole-cell est rate / the
        frac-weighted-average T_MAT respectively).
      - Remove-irrigation: weighted by frac_s for irrigated sources only (sum → irr_frac × rate),
        NOT divided by ag_frac — matching the flat blend.

    Returns dict[(from_m, from_j)] -> ndarray(ncells_src,) (separate=True -> {component: ...}),
    cells = the source's present cells (base-year dvar fraction above the rounding threshold).

    NB: unlike the flat blend, this omits cells whose total ag fraction is ≤ threshold (no ag
    source to flow from) — the flat blend charged the whole-cell EP establishment there
    regardless. In flow terms ag→EP cannot originate from a non-ag cell, so the omission is
    correct (and matches the crisp-flow treatment).
    """
    dvars          = data.ag_dvars[base_year]                       # (NLMS, NCELLS, N_AG_LUS)
    threshold      = 10 ** (-settings.ROUND_DECIMALS)
    ag_frac_r      = dvars.sum(axis=(0, 2))                         # (NCELLS,) total ag fraction
    ag_frac_r_safe = np.where(ag_frac_r > threshold, ag_frac_r, 1.0)

    # Whole-cell rates (source-independent)
    est_whole_r     = tools.amortise(data.EP_EST_COST_HA * data.REAL_AREA * data.EST_COST_MULTS[target_year]).astype(np.float32)
    rm_irrig_rate_r = (settings.REMOVE_IRRIG_COST * data.IRRIG_COST_MULTS[target_year] * data.REAL_AREA).astype(np.float32)  # not amortised (matches flat)
    t_j             = data.T_MAT.sel(from_lu=data.AGRICULTURAL_LANDUSES, to_lu='Environmental Plantings').values  # (N_AG_LUS,), NaN = prohibited

    result = {}
    for from_m in range(data.NLMS):
        for from_j in range(data.N_AG_LUS):
            frac_s = dvars[from_m, :, from_j]                       # (NCELLS,)
            cells  = np.where(frac_s > threshold)[0]
            if cells.size == 0:
                continue
            w_norm = (frac_s[cells] / ag_frac_r_safe[cells]).astype(np.float32)          # est & T_MAT weight
            est_s  = (est_whole_r[cells] * w_norm).astype(np.float32)
            t_s    = (tools.amortise(np.nan_to_num(t_j[from_j]) * data.REAL_AREA[cells]).astype(np.float32) * w_norm).astype(np.float32)
            w_s    = ((rm_irrig_rate_r[cells] * frac_s[cells]).astype(np.float32)
                      if from_m == 1 else np.zeros(cells.size, dtype=np.float32))
            if separate:
                result[(from_m, from_j)] = {
                    'Establishment cost (Ag2Non-Ag)':     est_s,
                    'Transition cost (Ag2Non-Ag)':        t_s,
                    'Remove irrigation cost (Ag2Non-Ag)': w_s,
                }
            else:
                result[(from_m, from_j)] = est_s + t_s + w_s
    return result


def get_env_plant_transitions_from_ag_exact(
    data: Data, target_year: int, mj_cell_map: dict, separate=False
) -> dict:
    """Exact EP transition costs per-(from_m, from_j) combo over cells with ag dvar > threshold.

    Parameters
    ----------
    mj_cell_map : dict returned by get_base_dvar_mj_cell_map(data, base_year)

    Returns
    -------
    dict[(from_m, from_j)] → ndarray (len(cell_idx),) in $/cell/yr
    OR, if separate=True:
    dict[(from_m, from_j)] → {'Establishment cost (Ag2Non-Ag)', 'Transition cost (Ag2Non-Ag)',
                               'Remove irrigation cost (Ag2Non-Ag)'} each ndarray (len(cell_idx),)
    """

    # Hoist invariants outside the per-combo loop
    t_j        = data.T_MAT.sel(from_lu=data.AGRICULTURAL_LANDUSES, to_lu='Environmental Plantings').values  # (N_AG_LUS,), NaN = prohibited
    est_mults  = data.EST_COST_MULTS[target_year]   # scalar
    irr_scalar = settings.REMOVE_IRRIG_COST * data.IRRIG_COST_MULTS[target_year]   # scalar

    result = {}
    for (from_m, from_j), cell_idx in mj_cell_map.items():
        est_costs  = tools.amortise(data.EP_EST_COST_HA[cell_idx] * est_mults * data.REAL_AREA[cell_idx]).astype(np.float32)
        t_cost     = tools.amortise(np.nan_to_num(t_j[from_j]) * data.REAL_AREA[cell_idx]).astype(np.float32)
        w_rm_irrig = (
            (irr_scalar * data.REAL_AREA[cell_idx]).astype(np.float32)
            if from_m == 1
            else np.zeros(len(cell_idx), dtype=np.float32)
        )

        if separate:
            result[(from_m, from_j)] = {
                'Establishment cost (Ag2Non-Ag)':     est_costs,
                'Transition cost (Ag2Non-Ag)':        t_cost,
                'Remove irrigation cost (Ag2Non-Ag)': w_rm_irrig,
            }
        else:
            result[(from_m, from_j)] = est_costs + t_cost + w_rm_irrig

    return result


def get_env_plant_transitions_from_ag_from_base_year(
    data: Data, base_year: int, target_year: int, separate=False
) -> np.ndarray | dict:
    """Dispatcher: exact, blended or crisp-lumap EP transition costs."""
    if settings.TRANSITION_MODE == 'exact':
        mj_cell_map = get_base_dvar_mj_cell_map(data, base_year)
        return get_env_plant_transitions_from_ag_exact(data, target_year, mj_cell_map, separate)
    elif settings.TRANSITION_MODE == 'blend':
        return get_env_plant_transitions_from_ag_blend(data, base_year, target_year, separate)
    elif settings.TRANSITION_MODE == 'crisp':
        return get_env_plant_transitions_from_ag_crisp(data, base_year, target_year, separate)
    else:
        raise ValueError(f"Unknown TRANSITION_MODE {settings.TRANSITION_MODE!r}; expected 'exact', 'blend', or 'crisp'.")


def get_rip_plant_transitions_from_ag_crisp(data: Data, base_year: int, target_year: int, separate=False) -> dict:
    """RP transition cost (ag → Riparian Plantings) in flow (from_m, from_j) format.

    Returns dict[(from_m, from_j)] -> ndarray(ncells_src,) in $/cell/yr
    (separate=True -> the leaf is {component: ndarray(ncells_src,)}), where ncells_src is the
    number of cells whose DOMINANT base-year LU is (from_m, from_j).

    crisp assigns each cell exactly one dominant (from_m, from_j) via the integerised lumap/lmmap,
    so the flat per-cell cost `flat[r]` IS the cost from cell r's dominant source. Computes the
    flat per-cell cost once and slices it per source. Collapsing back reproduces the flat crisp
    cost on ag-dominant cells. Mirrors get_rip_plant_transitions_from_ag_exact's shape.
    """
    lumap = data.lumaps[base_year]
    lmmap = data.lmmaps[base_year]
    cells = np.isin(lumap, np.array(list(data.AGLU2DESC.keys())))

    # ── Flat per-cell components (cost from each cell's actual dominant source) ──
    # Establishment costs
    est_costs_r = tools.amortise(data.RP_EST_COST_HA * data.REAL_AREA * data.EST_COST_MULTS[target_year]).astype(np.float32)
    est_costs_r[~cells] = 0.0

    # Transition costs
    ag_to_ep_j = data.T_MAT.sel(from_lu=data.AGRICULTURAL_LANDUSES, to_lu='Riparian Plantings').values
    ag_to_ep_t_r = np.vectorize(dict(enumerate(ag_to_ep_j)).get, otypes=['float32'])(lumap)
    ag_to_ep_t_r = np.nan_to_num(ag_to_ep_t_r)
    ag_to_ep_t_r = tools.amortise(ag_to_ep_t_r * data.REAL_AREA)
    ag_to_ep_t_r[~cells] = 0.0

    # Water costs; Assume riparian plantings are dryland
    w_rm_irrig_cost_r = (np.where(lmmap == 1, settings.REMOVE_IRRIG_COST * data.IRRIG_COST_MULTS[target_year], 0) * data.REAL_AREA).astype(np.float32)
    w_rm_irrig_cost_r[~cells] = 0.0

    # Fencing costs
    fencing_cost_r = (
        data.RP_FENCING_LENGTH
        * settings.FENCING_COST_PER_M
        * data.FENCE_COST_MULTS[target_year]
        * data.REAL_AREA
    ).astype(np.float32)
    fencing_cost_r[~cells] = 0.0

    if separate:
        flat = {
            'Establishment cost (Ag2Non-Ag)':     est_costs_r,
            'Transition cost (Ag2Non-Ag)':        ag_to_ep_t_r,
            'Remove irrigation cost (Ag2Non-Ag)': w_rm_irrig_cost_r,
            'Fencing cost (Ag2Non-Ag)':           fencing_cost_r,
        }
    else:
        flat = est_costs_r + ag_to_ep_t_r + w_rm_irrig_cost_r + fencing_cost_r

    # ── Slice the flat cost by dominant (from_m, from_j) source ──
    result = {}
    for from_m in range(data.NLMS):
        for from_j in range(data.N_AG_LUS):
            src_cells = np.where((lmmap == from_m) & (lumap == from_j))[0]
            if src_cells.size == 0:
                continue
            if separate:
                result[(from_m, from_j)] = {k: v[src_cells] for k, v in flat.items()}
            else:
                result[(from_m, from_j)] = flat[src_cells]
    return result


def get_rip_plant_transitions_from_ag_blend(data: Data, base_year: int, target_year: int, separate=False) -> dict:
    """Blended RP transition cost in flow (from_m, from_j) format.

    Per present source: own-source est/T_MAT/fence weighted by frac_s/ag_frac (so scatter-sum
    reproduces the flat blend), remove-irrigation by frac_s for irrigated sources only.
    """
    dvars         = data.ag_dvars[base_year]
    thr           = 10 ** (-settings.ROUND_DECIMALS)
    area          = data.REAL_AREA
    ag_frac       = dvars.sum(axis=(0, 2))
    ag_frac_safe  = np.where(ag_frac > thr, ag_frac, 1.0)
    est_whole     = tools.amortise(data.RP_EST_COST_HA * area * data.EST_COST_MULTS[target_year]).astype(np.float32)
    fence_whole   = (data.RP_FENCING_LENGTH * settings.FENCING_COST_PER_M * data.FENCE_COST_MULTS[target_year] * area).astype(np.float32)
    rm_irrig_rate = (settings.REMOVE_IRRIG_COST * data.IRRIG_COST_MULTS[target_year] * area).astype(np.float32)
    t_j           = data.T_MAT.sel(from_lu=data.AGRICULTURAL_LANDUSES, to_lu='Riparian Plantings').values

    result = {}
    for from_m in range(data.NLMS):
        for from_j in range(data.N_AG_LUS):
            frac_s = dvars[from_m, :, from_j]
            cells  = np.where(frac_s > thr)[0]
            if cells.size == 0:
                continue
            w_norm  = (frac_s[cells] / ag_frac_safe[cells]).astype(np.float32)
            est_s   = (est_whole[cells] * w_norm).astype(np.float32)
            t_s     = (tools.amortise(np.nan_to_num(t_j[from_j]) * area[cells]).astype(np.float32) * w_norm).astype(np.float32)
            fence_s = (fence_whole[cells] * w_norm).astype(np.float32)
            w_s     = ((rm_irrig_rate[cells] * frac_s[cells]).astype(np.float32)
                       if from_m == 1 else np.zeros(cells.size, dtype=np.float32))
            if separate:
                result[(from_m, from_j)] = {
                    'Establishment cost (Ag2Non-Ag)':     est_s,
                    'Transition cost (Ag2Non-Ag)':        t_s,
                    'Remove irrigation cost (Ag2Non-Ag)': w_s,
                    'Fencing cost (Ag2Non-Ag)':           fence_s,
                }
            else:
                result[(from_m, from_j)] = est_s + t_s + w_s + fence_s
    return result


def get_rip_plant_transitions_from_ag_exact(
    data: Data, target_year: int, mj_cell_map: dict, separate=False
) -> dict:
    """Exact RP transition costs per-(from_m, from_j) combo."""
    t_j        = data.T_MAT.sel(from_lu=data.AGRICULTURAL_LANDUSES, to_lu='Riparian Plantings').values
    est_mults  = data.EST_COST_MULTS[target_year]
    irr_scalar = settings.REMOVE_IRRIG_COST * data.IRRIG_COST_MULTS[target_year]
    fence_scalar = settings.FENCING_COST_PER_M * data.FENCE_COST_MULTS[target_year]

    result = {}
    for (from_m, from_j), cell_idx in mj_cell_map.items():
        est_costs    = tools.amortise(data.RP_EST_COST_HA[cell_idx] * est_mults * data.REAL_AREA[cell_idx]).astype(np.float32)
        t_cost       = tools.amortise(np.nan_to_num(t_j[from_j]) * data.REAL_AREA[cell_idx]).astype(np.float32)
        w_rm_irrig   = (
            (irr_scalar * data.REAL_AREA[cell_idx]).astype(np.float32)
            if from_m == 1 else np.zeros(len(cell_idx), dtype=np.float32)
        )
        fencing_cost = (data.RP_FENCING_LENGTH[cell_idx] * fence_scalar * data.REAL_AREA[cell_idx]).astype(np.float32)

        if separate:
            result[(from_m, from_j)] = {
                'Establishment cost (Ag2Non-Ag)':     est_costs,
                'Transition cost (Ag2Non-Ag)':        t_cost,
                'Remove irrigation cost (Ag2Non-Ag)': w_rm_irrig,
                'Fencing cost (Ag2Non-Ag)':           fencing_cost,
            }
        else:
            result[(from_m, from_j)] = est_costs + t_cost + w_rm_irrig + fencing_cost
    return result


def get_rip_plant_transitions_from_ag_from_base_year(
    data: Data, base_year: int, target_year: int, separate=False
) -> np.ndarray | dict:
    """Dispatcher: exact, blended or crisp-lumap RP transition costs."""
    if settings.TRANSITION_MODE == 'exact':
        mj_cell_map = get_base_dvar_mj_cell_map(data, base_year)
        return get_rip_plant_transitions_from_ag_exact(data, target_year, mj_cell_map, separate)
    elif settings.TRANSITION_MODE == 'blend':
        return get_rip_plant_transitions_from_ag_blend(data, base_year, target_year, separate)
    elif settings.TRANSITION_MODE == 'crisp':
        return get_rip_plant_transitions_from_ag_crisp(data, base_year, target_year, separate)
    else:
        raise ValueError(f"Unknown TRANSITION_MODE {settings.TRANSITION_MODE!r}; expected 'exact', 'blend', or 'crisp'.")


def get_sheep_agroforestry_transitions_from_ag_crisp(
    data: Data, base_year: int, target_year: int, separate=False
):
    """
    Get the base transition costs from agricultural land uses to Sheep Agroforestry for each cell.

    Returns
    -------
    np.ndarray
        (separate = False) 1-D array, indexed by cell. 
    dict
        (separate = True) Dict of separated transition costs.
    """
    
    lumap = data.lumaps[base_year]
    lmmap = data.lmmaps[base_year]
    cells = np.isin(lumap, np.array(list(data.AGLU2DESC.keys())))

    # Establishment costs
    est_costs_r = tools.amortise(data.AF_EST_COST_HA * data.REAL_AREA * data.EST_COST_MULTS[target_year]).astype(np.float32)
    est_costs_r[~cells] = 0.0
    est_costs_r *= settings.AF_PROPORTION

    # Transition costs
    ag_to_agroforestry_j = data.T_MAT.sel(from_lu=data.AGRICULTURAL_LANDUSES, to_lu='Sheep Agroforestry').values
    ag_to_agroforestry_t_r = np.vectorize(dict(enumerate(ag_to_agroforestry_j)).get, otypes=['float32'])(lumap)
    ag_to_agroforestry_t_r = np.nan_to_num(ag_to_agroforestry_t_r)
    ag_to_agroforestry_t_r = tools.amortise(ag_to_agroforestry_t_r * data.REAL_AREA)
    ag_to_agroforestry_t_r[~cells] = 0.0
    ag_to_agroforestry_t_r *= settings.AF_PROPORTION

    ag_to_sheep_j = data.T_MAT.sel(from_lu=data.AGRICULTURAL_LANDUSES, to_lu='Sheep - modified land').values    # Only consider ag to sheep-modified land here; Ag to sheep-natural is handled in the destocked-natural module
    ag_to_sheep_t_r = np.vectorize(dict(enumerate(ag_to_sheep_j)).get, otypes=['float32'])(lumap)
    ag_to_sheep_t_r = np.nan_to_num(ag_to_sheep_t_r)
    ag_to_sheep_t_r = tools.amortise(ag_to_sheep_t_r * data.REAL_AREA)
    ag_to_sheep_t_r[~cells] = 0.0
    ag_to_sheep_t_r *= (1 - settings.AF_PROPORTION)

    # Water costs; Assume AF is dryland so no need to multiply by AF_PROPORTION here
    w_rm_irrig_cost_r = (np.where(lmmap == 1, settings.REMOVE_IRRIG_COST * data.IRRIG_COST_MULTS[target_year], 0) * data.REAL_AREA).astype(np.float32)
    w_rm_irrig_cost_r[~cells] = 0.0

    # Fencing costs
    fencing_cost_r = (
        settings.AF_FENCING_LENGTH_HA
        * settings.FENCING_COST_PER_M
        * data.FENCE_COST_MULTS[target_year]
        * data.REAL_AREA
    ).astype(np.float32)
    fencing_cost_r[~cells] = 0.0
    
    flat = {
        'Establishment cost (Ag2Non-Ag)':     est_costs_r,
        'Transition cost (Ag2Non-Ag)':        ag_to_agroforestry_t_r,
        'Transition cost (Ag2AF-Sheep)':      ag_to_sheep_t_r,
        'Remove irrigation cost (Ag2Non-Ag)': w_rm_irrig_cost_r,
        'Fencing cost (Ag2Non-Ag)':           fencing_cost_r,
    } if separate else (est_costs_r + ag_to_agroforestry_t_r + ag_to_sheep_t_r + w_rm_irrig_cost_r + fencing_cost_r)
    result = {}
    for from_m in range(data.NLMS):
        for from_j in range(data.N_AG_LUS):
            src_cells = np.where((lmmap == from_m) & (lumap == from_j))[0]
            if src_cells.size == 0:
                continue
            if separate:
                result[(from_m, from_j)] = {k: v[src_cells] for k, v in flat.items()}
            else:
                result[(from_m, from_j)] = flat[src_cells]
    return result


def get_sheep_agroforestry_transitions_from_ag_blend(data: Data, base_year: int, target_year: int, separate=False) -> dict:
    """Blended Sheep AF transition cost in flow (from_m, from_j) format.

    Own-source est/T_MAT(AF & sheep)/fence weighted by frac_s/ag_frac; remove-irrigation by frac_s
    for irrigated sources. AF tree share × AF_PROPORTION, livestock share × (1-AF_PROPORTION).
    """
    AFP           = settings.AF_PROPORTION
    dvars         = data.ag_dvars[base_year]
    thr           = 10 ** (-settings.ROUND_DECIMALS)
    area          = data.REAL_AREA
    ag_frac       = dvars.sum(axis=(0, 2))
    ag_frac_safe  = np.where(ag_frac > thr, ag_frac, 1.0)
    est_whole     = (tools.amortise(data.AF_EST_COST_HA * area * data.EST_COST_MULTS[target_year]) * AFP).astype(np.float32)
    fence_whole   = (settings.AF_FENCING_LENGTH_HA * settings.FENCING_COST_PER_M * data.FENCE_COST_MULTS[target_year] * area).astype(np.float32)
    rm_irrig_rate = (settings.REMOVE_IRRIG_COST * data.IRRIG_COST_MULTS[target_year] * area).astype(np.float32)
    t_af_j        = data.T_MAT.sel(from_lu=data.AGRICULTURAL_LANDUSES, to_lu='Sheep Agroforestry').values
    t_sheep_j     = data.T_MAT.sel(from_lu=data.AGRICULTURAL_LANDUSES, to_lu='Sheep - modified land').values

    result = {}
    for from_m in range(data.NLMS):
        for from_j in range(data.N_AG_LUS):
            frac_s = dvars[from_m, :, from_j]
            cells  = np.where(frac_s > thr)[0]
            if cells.size == 0:
                continue
            w_norm  = (frac_s[cells] / ag_frac_safe[cells]).astype(np.float32)
            est_s   = (est_whole[cells] * w_norm).astype(np.float32)
            t_af_s  = (tools.amortise(np.nan_to_num(t_af_j[from_j])    * area[cells]).astype(np.float32) * AFP       * w_norm).astype(np.float32)
            t_shp_s = (tools.amortise(np.nan_to_num(t_sheep_j[from_j]) * area[cells]).astype(np.float32) * (1 - AFP) * w_norm).astype(np.float32)
            fence_s = (fence_whole[cells] * w_norm).astype(np.float32)
            w_s     = ((rm_irrig_rate[cells] * frac_s[cells]).astype(np.float32)
                       if from_m == 1 else np.zeros(cells.size, dtype=np.float32))
            if separate:
                result[(from_m, from_j)] = {
                    'Establishment cost (Ag2Non-Ag)':     est_s,
                    'Transition cost (Ag2Non-Ag)':        t_af_s,
                    'Transition cost (Ag2AF-Sheep)':      t_shp_s,
                    'Remove irrigation cost (Ag2Non-Ag)': w_s,
                    'Fencing cost (Ag2Non-Ag)':           fence_s,
                }
            else:
                result[(from_m, from_j)] = est_s + t_af_s + t_shp_s + w_s + fence_s
    return result


def get_sheep_agroforestry_transitions_from_ag_exact(
    data: Data, target_year: int, mj_cell_map: dict, separate=False
) -> dict:
    """Exact Sheep AF transition costs per-(from_m, from_j) combo."""
    t_af_j       = data.T_MAT.sel(from_lu=data.AGRICULTURAL_LANDUSES, to_lu='Sheep Agroforestry').values
    t_sheep_j    = data.T_MAT.sel(from_lu=data.AGRICULTURAL_LANDUSES, to_lu='Sheep - modified land').values
    est_mults    = data.EST_COST_MULTS[target_year]
    irr_scalar   = settings.REMOVE_IRRIG_COST * data.IRRIG_COST_MULTS[target_year]
    fence_scalar = settings.AF_FENCING_LENGTH_HA * settings.FENCING_COST_PER_M * data.FENCE_COST_MULTS[target_year]

    result = {}
    for (from_m, from_j), cell_idx in mj_cell_map.items():
        est_costs           = (tools.amortise(data.AF_EST_COST_HA[cell_idx] * est_mults * data.REAL_AREA[cell_idx]) * settings.AF_PROPORTION).astype(np.float32)
        ag_to_af_t          = (tools.amortise(np.nan_to_num(t_af_j[from_j]) * data.REAL_AREA[cell_idx]) * settings.AF_PROPORTION).astype(np.float32)
        ag_to_sheep_t       = (tools.amortise(np.nan_to_num(t_sheep_j[from_j]) * data.REAL_AREA[cell_idx]) * (1 - settings.AF_PROPORTION)).astype(np.float32)
        w_rm_irrig          = (
            (irr_scalar * data.REAL_AREA[cell_idx]).astype(np.float32)
            if from_m == 1 else np.zeros(len(cell_idx), dtype=np.float32)
        )
        fencing_cost        = (fence_scalar * data.REAL_AREA[cell_idx]).astype(np.float32)

        if separate:
            result[(from_m, from_j)] = {
                'Establishment cost (Ag2Non-Ag)':     est_costs,
                'Transition cost (Ag2Non-Ag)':        ag_to_af_t,
                'Transition cost (Ag2AF-Sheep)':      ag_to_sheep_t,
                'Remove irrigation cost (Ag2Non-Ag)': w_rm_irrig,
                'Fencing cost (Ag2Non-Ag)':           fencing_cost,
            }
        else:
            result[(from_m, from_j)] = est_costs + ag_to_af_t + ag_to_sheep_t + w_rm_irrig + fencing_cost
    return result


def get_sheep_agroforestry_transitions_from_ag_from_base_year(
    data: Data, base_year: int, target_year: int, separate=False
) -> np.ndarray | dict:
    """Dispatcher: exact, blended or crisp-lumap Sheep AF transition costs."""
    if settings.TRANSITION_MODE == 'exact':
        mj_cell_map = get_base_dvar_mj_cell_map(data, base_year)
        return get_sheep_agroforestry_transitions_from_ag_exact(data, target_year, mj_cell_map, separate)
    elif settings.TRANSITION_MODE == 'blend':
        return get_sheep_agroforestry_transitions_from_ag_blend(data, base_year, target_year, separate)
    elif settings.TRANSITION_MODE == 'crisp':
        return get_sheep_agroforestry_transitions_from_ag_crisp(data, base_year, target_year, separate)
    else:
        raise ValueError(f"Unknown TRANSITION_MODE {settings.TRANSITION_MODE!r}; expected 'exact', 'blend', or 'crisp'.")


def get_beef_agroforestry_transitions_from_ag_crisp(
    data: Data, base_year: int, target_year: int, separate=False
):
    """
    Get the base transition costs from agricultural land uses to Beef Agroforestry for each cell.

    Returns
    -------
    np.ndarray
        (separate = False) 1-D array, indexed by cell. 
    dict
        (separate = True) Dict of separated transition costs.
    """
    
    lumap = data.lumaps[base_year]
    lmmap = data.lmmaps[base_year]
    cells = np.isin(lumap, np.array(list(data.AGLU2DESC.keys())))

    # Establishment costs
    est_costs_r = tools.amortise(data.AF_EST_COST_HA * data.REAL_AREA * data.EST_COST_MULTS[target_year]).astype(np.float32)
    est_costs_r[~cells] = 0.0
    est_costs_r *= settings.AF_PROPORTION

    # Transition costs
    ag_to_agroforestry_j = data.T_MAT.sel(from_lu=data.AGRICULTURAL_LANDUSES, to_lu='Beef Agroforestry').values
    ag_to_agroforestry_t_r = np.vectorize(dict(enumerate(ag_to_agroforestry_j)).get, otypes=['float32'])(lumap)
    ag_to_agroforestry_t_r = np.nan_to_num(ag_to_agroforestry_t_r)
    ag_to_agroforestry_t_r = tools.amortise(ag_to_agroforestry_t_r * data.REAL_AREA)
    ag_to_agroforestry_t_r[~cells] = 0.0
    ag_to_agroforestry_t_r *= settings.AF_PROPORTION

    ag_to_beef_j = data.T_MAT.sel(from_lu=data.AGRICULTURAL_LANDUSES, to_lu='Beef - modified land').values    # Only consider ag to beef-modified land here; Ag to beef-natural is handled in the destocked-natural module
    ag_to_beef_t_r = np.vectorize(dict(enumerate(ag_to_beef_j)).get, otypes=['float32'])(lumap)
    ag_to_beef_t_r = np.nan_to_num(ag_to_beef_t_r)
    ag_to_beef_t_r = tools.amortise(ag_to_beef_t_r * data.REAL_AREA)
    ag_to_beef_t_r[~cells] = 0.0
    ag_to_beef_t_r *= (1 - settings.AF_PROPORTION)

    # Water costs; Assume AF is dryland so no need to multiply by AF_PROPORTION here
    w_rm_irrig_cost_r = (np.where(lmmap == 1, settings.REMOVE_IRRIG_COST * data.IRRIG_COST_MULTS[target_year], 0) * data.REAL_AREA).astype(np.float32)
    w_rm_irrig_cost_r[~cells] = 0.0

    # Fencing costs
    fencing_cost_r = (
        settings.AF_FENCING_LENGTH_HA
        * settings.FENCING_COST_PER_M
        * data.FENCE_COST_MULTS[target_year]
        * data.REAL_AREA
    ).astype(np.float32)
    fencing_cost_r[~cells] = 0.0
    
    flat = {
        'Establishment cost (Ag2Non-Ag)':     est_costs_r,
        'Transition cost (Ag2Non-Ag)':        ag_to_agroforestry_t_r,
        'Transition cost (Ag2AF-Beef)':       ag_to_beef_t_r,
        'Remove irrigation cost (Ag2Non-Ag)': w_rm_irrig_cost_r,
        'Fencing cost (Ag2Non-Ag)':           fencing_cost_r,
    } if separate else (est_costs_r + ag_to_agroforestry_t_r + ag_to_beef_t_r + w_rm_irrig_cost_r + fencing_cost_r)
    result = {}
    for from_m in range(data.NLMS):
        for from_j in range(data.N_AG_LUS):
            src_cells = np.where((lmmap == from_m) & (lumap == from_j))[0]
            if src_cells.size == 0:
                continue
            if separate:
                result[(from_m, from_j)] = {k: v[src_cells] for k, v in flat.items()}
            else:
                result[(from_m, from_j)] = flat[src_cells]
    return result


def get_beef_agroforestry_transitions_from_ag_blend(data: Data, base_year: int, target_year: int, separate=False) -> dict:
    """Blended Beef AF transition cost in flow (from_m, from_j) format (AF tree × AF_PROPORTION,
    beef share × (1-AF_PROPORTION); est/T_MAT/fence weighted by frac_s/ag_frac, water by frac_s)."""
    AFP           = settings.AF_PROPORTION
    dvars         = data.ag_dvars[base_year]
    thr           = 10 ** (-settings.ROUND_DECIMALS)
    area          = data.REAL_AREA
    ag_frac       = dvars.sum(axis=(0, 2))
    ag_frac_safe  = np.where(ag_frac > thr, ag_frac, 1.0)
    est_whole     = (tools.amortise(data.AF_EST_COST_HA * area * data.EST_COST_MULTS[target_year]) * AFP).astype(np.float32)
    fence_whole   = (settings.AF_FENCING_LENGTH_HA * settings.FENCING_COST_PER_M * data.FENCE_COST_MULTS[target_year] * area).astype(np.float32)
    rm_irrig_rate = (settings.REMOVE_IRRIG_COST * data.IRRIG_COST_MULTS[target_year] * area).astype(np.float32)
    t_af_j        = data.T_MAT.sel(from_lu=data.AGRICULTURAL_LANDUSES, to_lu='Beef Agroforestry').values
    t_beef_j      = data.T_MAT.sel(from_lu=data.AGRICULTURAL_LANDUSES, to_lu='Beef - modified land').values

    result = {}
    for from_m in range(data.NLMS):
        for from_j in range(data.N_AG_LUS):
            frac_s = dvars[from_m, :, from_j]
            cells  = np.where(frac_s > thr)[0]
            if cells.size == 0:
                continue
            w_norm  = (frac_s[cells] / ag_frac_safe[cells]).astype(np.float32)
            est_s   = (est_whole[cells] * w_norm).astype(np.float32)
            t_af_s  = (tools.amortise(np.nan_to_num(t_af_j[from_j])   * area[cells]).astype(np.float32) * AFP       * w_norm).astype(np.float32)
            t_bf_s  = (tools.amortise(np.nan_to_num(t_beef_j[from_j]) * area[cells]).astype(np.float32) * (1 - AFP) * w_norm).astype(np.float32)
            fence_s = (fence_whole[cells] * w_norm).astype(np.float32)
            w_s     = ((rm_irrig_rate[cells] * frac_s[cells]).astype(np.float32)
                       if from_m == 1 else np.zeros(cells.size, dtype=np.float32))
            if separate:
                result[(from_m, from_j)] = {
                    'Establishment cost (Ag2Non-Ag)':     est_s,
                    'Transition cost (Ag2Non-Ag)':        t_af_s,
                    'Transition cost (Ag2AF-Beef)':       t_bf_s,
                    'Remove irrigation cost (Ag2Non-Ag)': w_s,
                    'Fencing cost (Ag2Non-Ag)':           fence_s,
                }
            else:
                result[(from_m, from_j)] = est_s + t_af_s + t_bf_s + w_s + fence_s
    return result


def get_beef_agroforestry_transitions_from_ag_exact(
    data: Data, target_year: int, mj_cell_map: dict, separate=False
) -> dict:
    """Exact Beef AF transition costs per-(from_m, from_j) combo."""
    t_af_j       = data.T_MAT.sel(from_lu=data.AGRICULTURAL_LANDUSES, to_lu='Beef Agroforestry').values
    t_beef_j     = data.T_MAT.sel(from_lu=data.AGRICULTURAL_LANDUSES, to_lu='Beef - modified land').values
    est_mults    = data.EST_COST_MULTS[target_year]
    irr_scalar   = settings.REMOVE_IRRIG_COST * data.IRRIG_COST_MULTS[target_year]
    fence_scalar = settings.AF_FENCING_LENGTH_HA * settings.FENCING_COST_PER_M * data.FENCE_COST_MULTS[target_year]

    result = {}
    for (from_m, from_j), cell_idx in mj_cell_map.items():
        est_costs    = (tools.amortise(data.AF_EST_COST_HA[cell_idx] * est_mults * data.REAL_AREA[cell_idx]) * settings.AF_PROPORTION).astype(np.float32)
        ag_to_af_t   = (tools.amortise(np.nan_to_num(t_af_j[from_j]) * data.REAL_AREA[cell_idx]) * settings.AF_PROPORTION).astype(np.float32)
        ag_to_beef_t = (tools.amortise(np.nan_to_num(t_beef_j[from_j]) * data.REAL_AREA[cell_idx]) * (1 - settings.AF_PROPORTION)).astype(np.float32)
        w_rm_irrig   = (
            (irr_scalar * data.REAL_AREA[cell_idx]).astype(np.float32)
            if from_m == 1 else np.zeros(len(cell_idx), dtype=np.float32)
        )
        fencing_cost = (fence_scalar * data.REAL_AREA[cell_idx]).astype(np.float32)

        if separate:
            result[(from_m, from_j)] = {
                'Establishment cost (Ag2Non-Ag)':     est_costs,
                'Transition cost (Ag2Non-Ag)':        ag_to_af_t,
                'Transition cost (Ag2AF-Beef)':       ag_to_beef_t,
                'Remove irrigation cost (Ag2Non-Ag)': w_rm_irrig,
                'Fencing cost (Ag2Non-Ag)':           fencing_cost,
            }
        else:
            result[(from_m, from_j)] = est_costs + ag_to_af_t + ag_to_beef_t + w_rm_irrig + fencing_cost
    return result


def get_beef_agroforestry_transitions_from_ag_from_base_year(
    data: Data, base_year: int, target_year: int, separate=False
) -> np.ndarray | dict:
    """Dispatcher: exact, blended or crisp-lumap Beef AF transition costs."""
    if settings.TRANSITION_MODE == 'exact':
        mj_cell_map = get_base_dvar_mj_cell_map(data, base_year)
        return get_beef_agroforestry_transitions_from_ag_exact(data, target_year, mj_cell_map, separate)
    elif settings.TRANSITION_MODE == 'blend':
        return get_beef_agroforestry_transitions_from_ag_blend(data, base_year, target_year, separate)
    elif settings.TRANSITION_MODE == 'crisp':
        return get_beef_agroforestry_transitions_from_ag_crisp(data, base_year, target_year, separate)
    else:
        raise ValueError(f"Unknown TRANSITION_MODE {settings.TRANSITION_MODE!r}; expected 'exact', 'blend', or 'crisp'.")


def get_carbon_plantings_block_from_ag_crisp(data: Data, base_year: int, target_year: int, separate=False) -> np.ndarray | dict:
    """
    Get transition costs from agricultural land uses to carbon plantings (block) for each cell.

    Returns
    -------
    np.ndarray
        1-D array, indexed by cell.
    """
    lumap = data.lumaps[base_year]
    lmmap = data.lmmaps[base_year]
    cells = np.isin(lumap, np.array(list(data.AGLU2DESC.keys())))

    # Establishment costs
    est_costs_CP_r = tools.amortise(data.CP_EST_COST_HA * data.REAL_AREA * data.EST_COST_MULTS[target_year]).astype(np.float32)
    est_costs_CP_r[~cells] = 0.0

    # Transition costs
    ag_to_cp_j = data.T_MAT.sel(from_lu=data.AGRICULTURAL_LANDUSES, to_lu='Carbon Plantings (Block)').values
    ag_to_cp_t_r = np.vectorize(dict(enumerate(ag_to_cp_j)).get, otypes=['float32'])(lumap)
    ag_to_cp_t_r = np.nan_to_num(ag_to_cp_t_r)
    ag_to_cp_t_r = tools.amortise(ag_to_cp_t_r * data.REAL_AREA)
    ag_to_cp_t_r[~cells] = 0.0

    # Water costs; Assume CP is dryland
    w_rm_irrig_cost_r = (np.where(lmmap == 1, settings.REMOVE_IRRIG_COST * data.IRRIG_COST_MULTS[target_year], 0) * data.REAL_AREA).astype(np.float32)
    w_rm_irrig_cost_r[~cells] = 0.0

    flat = {
        'Establishment cost (Ag2Non-Ag)':     est_costs_CP_r,
        'Transition cost (Ag2Non-Ag)':        ag_to_cp_t_r,
        'Remove irrigation cost (Ag2Non-Ag)': w_rm_irrig_cost_r,
    } if separate else (est_costs_CP_r + ag_to_cp_t_r + w_rm_irrig_cost_r)
    result = {}
    for from_m in range(data.NLMS):
        for from_j in range(data.N_AG_LUS):
            src_cells = np.where((lmmap == from_m) & (lumap == from_j))[0]
            if src_cells.size == 0:
                continue
            if separate:
                result[(from_m, from_j)] = {k: v[src_cells] for k, v in flat.items()}
            else:
                result[(from_m, from_j)] = flat[src_cells]
    return result


def get_carbon_plantings_block_from_ag_blend(data: Data, base_year: int, target_year: int, separate=False) -> dict:
    """Blended CP Block transition cost in flow (from_m, from_j) format (est/T_MAT weighted by
    frac_s/ag_frac, remove-irrigation by frac_s for irrigated sources)."""
    dvars         = data.ag_dvars[base_year]
    thr           = 10 ** (-settings.ROUND_DECIMALS)
    area          = data.REAL_AREA
    ag_frac       = dvars.sum(axis=(0, 2))
    ag_frac_safe  = np.where(ag_frac > thr, ag_frac, 1.0)
    est_whole     = tools.amortise(data.CP_EST_COST_HA * area * data.EST_COST_MULTS[target_year]).astype(np.float32)
    rm_irrig_rate = (settings.REMOVE_IRRIG_COST * data.IRRIG_COST_MULTS[target_year] * area).astype(np.float32)
    t_j           = data.T_MAT.sel(from_lu=data.AGRICULTURAL_LANDUSES, to_lu='Carbon Plantings (Block)').values

    result = {}
    for from_m in range(data.NLMS):
        for from_j in range(data.N_AG_LUS):
            frac_s = dvars[from_m, :, from_j]
            cells  = np.where(frac_s > thr)[0]
            if cells.size == 0:
                continue
            w_norm = (frac_s[cells] / ag_frac_safe[cells]).astype(np.float32)
            est_s  = (est_whole[cells] * w_norm).astype(np.float32)
            t_s    = (tools.amortise(np.nan_to_num(t_j[from_j]) * area[cells]).astype(np.float32) * w_norm).astype(np.float32)
            w_s    = ((rm_irrig_rate[cells] * frac_s[cells]).astype(np.float32)
                      if from_m == 1 else np.zeros(cells.size, dtype=np.float32))
            if separate:
                result[(from_m, from_j)] = {
                    'Establishment cost (Ag2Non-Ag)':     est_s,
                    'Transition cost (Ag2Non-Ag)':        t_s,
                    'Remove irrigation cost (Ag2Non-Ag)': w_s,
                }
            else:
                result[(from_m, from_j)] = est_s + t_s + w_s
    return result


def get_carbon_plantings_block_from_ag_exact(
    data: Data, target_year: int, mj_cell_map: dict, separate=False
) -> dict:
    """Exact CP Block transition costs per-(from_m, from_j) combo."""
    t_j        = data.T_MAT.sel(from_lu=data.AGRICULTURAL_LANDUSES, to_lu='Carbon Plantings (Block)').values
    est_mults  = data.EST_COST_MULTS[target_year]
    irr_scalar = settings.REMOVE_IRRIG_COST * data.IRRIG_COST_MULTS[target_year]

    result = {}
    for (from_m, from_j), cell_idx in mj_cell_map.items():
        est_costs  = tools.amortise(data.CP_EST_COST_HA[cell_idx] * est_mults * data.REAL_AREA[cell_idx]).astype(np.float32)
        t_cost     = tools.amortise(np.nan_to_num(t_j[from_j]) * data.REAL_AREA[cell_idx]).astype(np.float32)
        w_rm_irrig = (
            (irr_scalar * data.REAL_AREA[cell_idx]).astype(np.float32)
            if from_m == 1 else np.zeros(len(cell_idx), dtype=np.float32)
        )

        if separate:
            result[(from_m, from_j)] = {
                'Establishment cost (Ag2Non-Ag)':     est_costs,
                'Transition cost (Ag2Non-Ag)':        t_cost,
                'Remove irrigation cost (Ag2Non-Ag)': w_rm_irrig,
            }
        else:
            result[(from_m, from_j)] = est_costs + t_cost + w_rm_irrig
    return result


def get_carbon_plantings_block_from_ag_from_base_year(
    data: Data, base_year: int, target_year: int, separate=False
) -> np.ndarray | dict:
    """Dispatcher: exact, blended or crisp-lumap CP Block transition costs."""
    if settings.TRANSITION_MODE == 'exact':
        mj_cell_map = get_base_dvar_mj_cell_map(data, base_year)
        return get_carbon_plantings_block_from_ag_exact(data, target_year, mj_cell_map, separate)
    elif settings.TRANSITION_MODE == 'blend':
        return get_carbon_plantings_block_from_ag_blend(data, base_year, target_year, separate)
    elif settings.TRANSITION_MODE == 'crisp':
        return get_carbon_plantings_block_from_ag_crisp(data, base_year, target_year, separate)
    else:
        raise ValueError(f"Unknown TRANSITION_MODE {settings.TRANSITION_MODE!r}; expected 'exact', 'blend', or 'crisp'.")


def get_sheep_carbon_plantings_belt_from_ag_crisp(
    data: Data, base_year: int, target_year: int, separate=False
):
    """
    Get the transition costs from agricultural land uses to Sheep Carbon Plantings (belt) for each cell.

    Returns
    -------
    np.ndarray
        (separate = False) 1-D array, indexed by cell. 
    dict
        (separate = True) Dict of separated transition costs.
    """
    
    lumap = data.lumaps[base_year]
    lmmap = data.lmmaps[base_year]
    cells = np.isin(lumap, np.array(list(data.AGLU2DESC.keys())))

    # Establishment costs
    est_costs_CP_r = tools.amortise(data.CP_EST_COST_HA * data.REAL_AREA * data.EST_COST_MULTS[target_year]).astype(np.float32)
    est_costs_CP_r[~cells] = 0.0
    est_costs_CP_r *= settings.CP_BELT_PROPORTION

    # Transition costs
    ag_to_cp_j = data.T_MAT.sel(from_lu=data.AGRICULTURAL_LANDUSES, to_lu='Sheep Carbon Plantings (Belt)').values
    ag_to_cp_t_r = np.vectorize(dict(enumerate(ag_to_cp_j)).get, otypes=['float32'])(lumap)
    ag_to_cp_t_r = np.nan_to_num(ag_to_cp_t_r)
    ag_to_cp_t_r = tools.amortise(ag_to_cp_t_r * data.REAL_AREA)
    ag_to_cp_t_r[~cells] = 0.0
    ag_to_cp_t_r *= settings.CP_BELT_PROPORTION

    ag_to_sheep_j = data.T_MAT.sel(from_lu=data.AGRICULTURAL_LANDUSES, to_lu='Sheep - modified land').values    # Only consider sheep-modified land here; Ag to sheep-natural is handled in the destocked-natural module
    ag_to_sheep_t_r = np.vectorize(dict(enumerate(ag_to_sheep_j)).get, otypes=['float32'])(lumap)
    ag_to_sheep_t_r = np.nan_to_num(ag_to_sheep_t_r)
    ag_to_sheep_t_r = tools.amortise(ag_to_sheep_t_r * data.REAL_AREA)
    ag_to_sheep_t_r[~cells] = 0.0
    ag_to_sheep_t_r *= (1 - settings.CP_BELT_PROPORTION)

    # Water costs; Assume CP is dryland
    w_rm_irrig_cost_r = (np.where(lmmap == 1, settings.REMOVE_IRRIG_COST * data.IRRIG_COST_MULTS[target_year], 0) * data.REAL_AREA).astype(np.float32)
    w_rm_irrig_cost_r[~cells] = 0.0

    fencing_cost_r = (
        settings.CP_BELT_FENCING_LENGTH
        * settings.FENCING_COST_PER_M
        * data.FENCE_COST_MULTS[target_year]
        * data.REAL_AREA
    ).astype(np.float32)
    fencing_cost_r[~cells] = 0.0

    flat = {
        'Establishment cost (Ag2Non-Ag)':     est_costs_CP_r,
        'Transition cost (Ag2Non-Ag)':        ag_to_cp_t_r,
        'Transition cost (Ag2CP-Sheep)':      ag_to_sheep_t_r,
        'Remove irrigation cost (Ag2Non-Ag)': w_rm_irrig_cost_r,
        'Fencing cost (Ag2Non-Ag)':           fencing_cost_r,
    } if separate else (est_costs_CP_r + ag_to_cp_t_r + ag_to_sheep_t_r + w_rm_irrig_cost_r + fencing_cost_r)
    result = {}
    for from_m in range(data.NLMS):
        for from_j in range(data.N_AG_LUS):
            src_cells = np.where((lmmap == from_m) & (lumap == from_j))[0]
            if src_cells.size == 0:
                continue
            if separate:
                result[(from_m, from_j)] = {k: v[src_cells] for k, v in flat.items()}
            else:
                result[(from_m, from_j)] = flat[src_cells]
    return result


def get_sheep_carbon_plantings_belt_from_ag_blend(data: Data, base_year: int, target_year: int, separate=False) -> dict:
    """Blended Sheep CP Belt transition cost in flow (from_m, from_j) format (CP tree × CP_BELT_PROPORTION,
    sheep share × (1-CP_BELT_PROPORTION); est/T_MAT/fence weighted by frac_s/ag_frac, water by frac_s)."""
    CPP           = settings.CP_BELT_PROPORTION
    dvars         = data.ag_dvars[base_year]
    thr           = 10 ** (-settings.ROUND_DECIMALS)
    area          = data.REAL_AREA
    ag_frac       = dvars.sum(axis=(0, 2))
    ag_frac_safe  = np.where(ag_frac > thr, ag_frac, 1.0)
    est_whole     = (tools.amortise(data.CP_EST_COST_HA * area * data.EST_COST_MULTS[target_year]) * CPP).astype(np.float32)
    fence_whole   = (settings.CP_BELT_FENCING_LENGTH * settings.FENCING_COST_PER_M * data.FENCE_COST_MULTS[target_year] * area).astype(np.float32)
    rm_irrig_rate = (settings.REMOVE_IRRIG_COST * data.IRRIG_COST_MULTS[target_year] * area).astype(np.float32)
    t_cp_j        = data.T_MAT.sel(from_lu=data.AGRICULTURAL_LANDUSES, to_lu='Sheep Carbon Plantings (Belt)').values
    t_sheep_j     = data.T_MAT.sel(from_lu=data.AGRICULTURAL_LANDUSES, to_lu='Sheep - modified land').values

    result = {}
    for from_m in range(data.NLMS):
        for from_j in range(data.N_AG_LUS):
            frac_s = dvars[from_m, :, from_j]
            cells  = np.where(frac_s > thr)[0]
            if cells.size == 0:
                continue
            w_norm  = (frac_s[cells] / ag_frac_safe[cells]).astype(np.float32)
            est_s   = (est_whole[cells] * w_norm).astype(np.float32)
            t_cp_s  = (tools.amortise(np.nan_to_num(t_cp_j[from_j])    * area[cells]).astype(np.float32) * CPP       * w_norm).astype(np.float32)
            t_shp_s = (tools.amortise(np.nan_to_num(t_sheep_j[from_j]) * area[cells]).astype(np.float32) * (1 - CPP) * w_norm).astype(np.float32)
            fence_s = (fence_whole[cells] * w_norm).astype(np.float32)
            w_s     = ((rm_irrig_rate[cells] * frac_s[cells]).astype(np.float32)
                       if from_m == 1 else np.zeros(cells.size, dtype=np.float32))
            if separate:
                result[(from_m, from_j)] = {
                    'Establishment cost (Ag2Non-Ag)':     est_s,
                    'Transition cost (Ag2Non-Ag)':        t_cp_s,
                    'Transition cost (Ag2CP-Sheep)':      t_shp_s,
                    'Remove irrigation cost (Ag2Non-Ag)': w_s,
                    'Fencing cost (Ag2Non-Ag)':           fence_s,
                }
            else:
                result[(from_m, from_j)] = est_s + t_cp_s + t_shp_s + w_s + fence_s
    return result


def get_sheep_carbon_plantings_belt_from_ag_exact(
    data: Data, target_year: int, mj_cell_map: dict, separate=False
) -> dict:
    """Exact Sheep CP Belt transition costs per-(from_m, from_j) combo."""
    t_cp_j       = data.T_MAT.sel(from_lu=data.AGRICULTURAL_LANDUSES, to_lu='Sheep Carbon Plantings (Belt)').values
    t_sheep_j    = data.T_MAT.sel(from_lu=data.AGRICULTURAL_LANDUSES, to_lu='Sheep - modified land').values
    est_mults    = data.EST_COST_MULTS[target_year]
    irr_scalar   = settings.REMOVE_IRRIG_COST * data.IRRIG_COST_MULTS[target_year]
    fence_scalar = settings.CP_BELT_FENCING_LENGTH * settings.FENCING_COST_PER_M * data.FENCE_COST_MULTS[target_year]

    result = {}
    for (from_m, from_j), cell_idx in mj_cell_map.items():
        est_costs     = (tools.amortise(data.CP_EST_COST_HA[cell_idx] * est_mults * data.REAL_AREA[cell_idx]) * settings.CP_BELT_PROPORTION).astype(np.float32)
        ag_to_cp_t    = (tools.amortise(np.nan_to_num(t_cp_j[from_j]) * data.REAL_AREA[cell_idx]) * settings.CP_BELT_PROPORTION).astype(np.float32)
        ag_to_sheep_t = (tools.amortise(np.nan_to_num(t_sheep_j[from_j]) * data.REAL_AREA[cell_idx]) * (1 - settings.CP_BELT_PROPORTION)).astype(np.float32)
        w_rm_irrig    = (
            (irr_scalar * data.REAL_AREA[cell_idx]).astype(np.float32)
            if from_m == 1 else np.zeros(len(cell_idx), dtype=np.float32)
        )
        fencing_cost  = (fence_scalar * data.REAL_AREA[cell_idx]).astype(np.float32)

        if separate:
            result[(from_m, from_j)] = {
                'Establishment cost (Ag2Non-Ag)':     est_costs,
                'Transition cost (Ag2Non-Ag)':        ag_to_cp_t,
                'Transition cost (Ag2CP-Sheep)':      ag_to_sheep_t,
                'Remove irrigation cost (Ag2Non-Ag)': w_rm_irrig,
                'Fencing cost (Ag2Non-Ag)':           fencing_cost,
            }
        else:
            result[(from_m, from_j)] = est_costs + ag_to_cp_t + ag_to_sheep_t + w_rm_irrig + fencing_cost
    return result


def get_sheep_carbon_plantings_belt_from_ag_from_base_year(
    data: Data, base_year: int, target_year: int, separate=False
) -> np.ndarray | dict:
    """Dispatcher: exact, blended or crisp-lumap Sheep CP Belt transition costs."""
    if settings.TRANSITION_MODE == 'exact':
        mj_cell_map = get_base_dvar_mj_cell_map(data, base_year)
        return get_sheep_carbon_plantings_belt_from_ag_exact(data, target_year, mj_cell_map, separate)
    elif settings.TRANSITION_MODE == 'blend':
        return get_sheep_carbon_plantings_belt_from_ag_blend(data, base_year, target_year, separate)
    elif settings.TRANSITION_MODE == 'crisp':
        return get_sheep_carbon_plantings_belt_from_ag_crisp(data, base_year, target_year, separate)
    else:
        raise ValueError(f"Unknown TRANSITION_MODE {settings.TRANSITION_MODE!r}; expected 'exact', 'blend', or 'crisp'.")


def get_beef_carbon_plantings_belt_from_ag_crisp(
    data: Data, base_year: int, target_year: int, separate=False
):
    """
    Get the base transition costs from agricultural land uses to Beef Carbon Plantings (belt) for each cell.

    Returns
    -------
    np.ndarray
        (separate = False) 1-D array, indexed by cell. 
    dict
        (separate = True) Dict of separated transition costs.
    """
    
    lumap = data.lumaps[base_year]
    lmmap = data.lmmaps[base_year]
    cells = np.isin(lumap, np.array(list(data.AGLU2DESC.keys())))

    # Establishment costs
    est_costs_CP_r = tools.amortise(data.CP_EST_COST_HA * data.REAL_AREA * data.EST_COST_MULTS[target_year]).astype(np.float32)
    est_costs_CP_r[~cells] = 0.0
    est_costs_CP_r *= settings.CP_BELT_PROPORTION

    # Transition costs
    ag_to_cp_j = data.T_MAT.sel(from_lu=data.AGRICULTURAL_LANDUSES, to_lu='Beef Carbon Plantings (Belt)').values
    ag_to_cp_t_r = np.vectorize(dict(enumerate(ag_to_cp_j)).get, otypes=['float32'])(lumap)
    ag_to_cp_t_r = np.nan_to_num(ag_to_cp_t_r)
    ag_to_cp_t_r = tools.amortise(ag_to_cp_t_r * data.REAL_AREA)
    ag_to_cp_t_r[~cells] = 0.0
    ag_to_cp_t_r *= settings.CP_BELT_PROPORTION

    ag_to_sheep_j = data.T_MAT.sel(from_lu=data.AGRICULTURAL_LANDUSES, to_lu='Beef - modified land').values    # Only consider beef-modified land here; Ag to beef-natural is handled in the destocked-natural module
    ag_to_sheep_t_r = np.vectorize(dict(enumerate(ag_to_sheep_j)).get, otypes=['float32'])(lumap)
    ag_to_sheep_t_r = np.nan_to_num(ag_to_sheep_t_r)
    ag_to_sheep_t_r = tools.amortise(ag_to_sheep_t_r * data.REAL_AREA)
    ag_to_sheep_t_r[~cells] = 0.0
    ag_to_sheep_t_r *= (1 - settings.CP_BELT_PROPORTION)

    # Water costs; Assume CP is dryland
    w_rm_irrig_cost_r = (np.where(lmmap == 1, settings.REMOVE_IRRIG_COST * data.IRRIG_COST_MULTS[target_year], 0) * data.REAL_AREA).astype(np.float32)
    w_rm_irrig_cost_r[~cells] = 0.0

    fencing_cost_r = (
        settings.CP_BELT_FENCING_LENGTH
        * settings.FENCING_COST_PER_M
        * data.FENCE_COST_MULTS[target_year]
        * data.REAL_AREA
    ).astype(np.float32)
    fencing_cost_r[~cells] = 0.0

    flat = {
        'Establishment cost (Ag2Non-Ag)':     est_costs_CP_r,
        'Transition cost (Ag2Non-Ag)':        ag_to_cp_t_r,
        'Transition cost (Ag2CP-Beef)':       ag_to_sheep_t_r,
        'Remove irrigation cost (Ag2Non-Ag)': w_rm_irrig_cost_r,
        'Fencing cost (Ag2Non-Ag)':           fencing_cost_r,
    } if separate else (est_costs_CP_r + ag_to_cp_t_r + ag_to_sheep_t_r + w_rm_irrig_cost_r + fencing_cost_r)
    result = {}
    for from_m in range(data.NLMS):
        for from_j in range(data.N_AG_LUS):
            src_cells = np.where((lmmap == from_m) & (lumap == from_j))[0]
            if src_cells.size == 0:
                continue
            if separate:
                result[(from_m, from_j)] = {k: v[src_cells] for k, v in flat.items()}
            else:
                result[(from_m, from_j)] = flat[src_cells]
    return result


def get_beef_carbon_plantings_belt_from_ag_blend(data: Data, base_year: int, target_year: int, separate=False) -> dict:
    """Blended Beef CP Belt transition cost in flow (from_m, from_j) format (CP tree × CP_BELT_PROPORTION,
    beef share × (1-CP_BELT_PROPORTION); est/T_MAT/fence weighted by frac_s/ag_frac, water by frac_s)."""
    CPP           = settings.CP_BELT_PROPORTION
    dvars         = data.ag_dvars[base_year]
    thr           = 10 ** (-settings.ROUND_DECIMALS)
    area          = data.REAL_AREA
    ag_frac       = dvars.sum(axis=(0, 2))
    ag_frac_safe  = np.where(ag_frac > thr, ag_frac, 1.0)
    est_whole     = (tools.amortise(data.CP_EST_COST_HA * area * data.EST_COST_MULTS[target_year]) * CPP).astype(np.float32)
    fence_whole   = (settings.CP_BELT_FENCING_LENGTH * settings.FENCING_COST_PER_M * data.FENCE_COST_MULTS[target_year] * area).astype(np.float32)
    rm_irrig_rate = (settings.REMOVE_IRRIG_COST * data.IRRIG_COST_MULTS[target_year] * area).astype(np.float32)
    t_cp_j        = data.T_MAT.sel(from_lu=data.AGRICULTURAL_LANDUSES, to_lu='Beef Carbon Plantings (Belt)').values
    t_beef_j      = data.T_MAT.sel(from_lu=data.AGRICULTURAL_LANDUSES, to_lu='Beef - modified land').values

    result = {}
    for from_m in range(data.NLMS):
        for from_j in range(data.N_AG_LUS):
            frac_s = dvars[from_m, :, from_j]
            cells  = np.where(frac_s > thr)[0]
            if cells.size == 0:
                continue
            w_norm  = (frac_s[cells] / ag_frac_safe[cells]).astype(np.float32)
            est_s   = (est_whole[cells] * w_norm).astype(np.float32)
            t_cp_s  = (tools.amortise(np.nan_to_num(t_cp_j[from_j])   * area[cells]).astype(np.float32) * CPP       * w_norm).astype(np.float32)
            t_bf_s  = (tools.amortise(np.nan_to_num(t_beef_j[from_j]) * area[cells]).astype(np.float32) * (1 - CPP) * w_norm).astype(np.float32)
            fence_s = (fence_whole[cells] * w_norm).astype(np.float32)
            w_s     = ((rm_irrig_rate[cells] * frac_s[cells]).astype(np.float32)
                       if from_m == 1 else np.zeros(cells.size, dtype=np.float32))
            if separate:
                result[(from_m, from_j)] = {
                    'Establishment cost (Ag2Non-Ag)':     est_s,
                    'Transition cost (Ag2Non-Ag)':        t_cp_s,
                    'Transition cost (Ag2CP-Beef)':       t_bf_s,
                    'Remove irrigation cost (Ag2Non-Ag)': w_s,
                    'Fencing cost (Ag2Non-Ag)':           fence_s,
                }
            else:
                result[(from_m, from_j)] = est_s + t_cp_s + t_bf_s + w_s + fence_s
    return result


def get_beef_carbon_plantings_belt_from_ag_exact(
    data: Data, target_year: int, mj_cell_map: dict, separate=False
) -> dict:
    """Exact Beef CP Belt transition costs per-(from_m, from_j) combo."""
    t_cp_j       = data.T_MAT.sel(from_lu=data.AGRICULTURAL_LANDUSES, to_lu='Beef Carbon Plantings (Belt)').values
    t_beef_j     = data.T_MAT.sel(from_lu=data.AGRICULTURAL_LANDUSES, to_lu='Beef - modified land').values
    est_mults    = data.EST_COST_MULTS[target_year]
    irr_scalar   = settings.REMOVE_IRRIG_COST * data.IRRIG_COST_MULTS[target_year]
    fence_scalar = settings.CP_BELT_FENCING_LENGTH * settings.FENCING_COST_PER_M * data.FENCE_COST_MULTS[target_year]

    result = {}
    for (from_m, from_j), cell_idx in mj_cell_map.items():
        est_costs    = (tools.amortise(data.CP_EST_COST_HA[cell_idx] * est_mults * data.REAL_AREA[cell_idx]) * settings.CP_BELT_PROPORTION).astype(np.float32)
        ag_to_cp_t   = (tools.amortise(np.nan_to_num(t_cp_j[from_j]) * data.REAL_AREA[cell_idx]) * settings.CP_BELT_PROPORTION).astype(np.float32)
        ag_to_beef_t = (tools.amortise(np.nan_to_num(t_beef_j[from_j]) * data.REAL_AREA[cell_idx]) * (1 - settings.CP_BELT_PROPORTION)).astype(np.float32)
        w_rm_irrig   = (
            (irr_scalar * data.REAL_AREA[cell_idx]).astype(np.float32)
            if from_m == 1 else np.zeros(len(cell_idx), dtype=np.float32)
        )
        fencing_cost = (fence_scalar * data.REAL_AREA[cell_idx]).astype(np.float32)

        if separate:
            result[(from_m, from_j)] = {
                'Establishment cost (Ag2Non-Ag)':     est_costs,
                'Transition cost (Ag2Non-Ag)':        ag_to_cp_t,
                'Transition cost (Ag2CP-Beef)':       ag_to_beef_t,
                'Remove irrigation cost (Ag2Non-Ag)': w_rm_irrig,
                'Fencing cost (Ag2Non-Ag)':           fencing_cost,
            }
        else:
            result[(from_m, from_j)] = est_costs + ag_to_cp_t + ag_to_beef_t + w_rm_irrig + fencing_cost
    return result


def get_beef_carbon_plantings_belt_from_ag_from_base_year(
    data: Data, base_year: int, target_year: int, separate=False
) -> np.ndarray | dict:
    """Dispatcher: exact, blended or crisp-lumap Beef CP Belt transition costs."""
    if settings.TRANSITION_MODE == 'exact':
        mj_cell_map = get_base_dvar_mj_cell_map(data, base_year)
        return get_beef_carbon_plantings_belt_from_ag_exact(data, target_year, mj_cell_map, separate)
    elif settings.TRANSITION_MODE == 'blend':
        return get_beef_carbon_plantings_belt_from_ag_blend(data, base_year, target_year, separate)
    elif settings.TRANSITION_MODE == 'crisp':
        return get_beef_carbon_plantings_belt_from_ag_crisp(data, base_year, target_year, separate)
    else:
        raise ValueError(f"Unknown TRANSITION_MODE {settings.TRANSITION_MODE!r}; expected 'exact', 'blend', or 'crisp'.")


def get_beccs_from_ag(data: Data, base_year: int, target_year: int, separate=False) -> np.ndarray | dict:
    """Transition costs from agricultural land uses to BECCS (same as EP)."""
    return get_env_plant_transitions_from_ag_from_base_year(data, base_year, target_year, separate)


def get_destocked_from_ag_crisp(
    data: Data, base_year: int, target_year: int, separate=False
) -> np.ndarray | dict:
    """
    Get transition costs from agricultural land uses to destocked land for each cell, including
     - Fixed cost of removing livestock
     - Establishment cost, proportional to the increase of biodiversity/habitat contribution jumped
     - Water costs of removing irrigation (if previously irrigated)

    The establishment cost is just calculated as `proportional of biodiversity benefit` * `Environmental Planting
    establishment cost per ha`, not meaning any actual planting is done.


    Returns
    -------
    if separate == False:
        np.ndarray
            1-D array, indexed by cell.
    if separate == True:
        dict
            Separated dictionary of transition cost arrays.
    """
    lumap = data.lumaps[base_year]
    lmmap = data.lmmaps[base_year]
    cells = np.isin(lumap, data.LU_LVSTK_NATURAL)
    
    # Fixed cost of removing livestock
    ag2destock_j = data.T_MAT.sel(from_lu=data.AGRICULTURAL_LANDUSES, to_lu='Destocked - natural land').values
    trans_cost_r = np.vectorize(dict(enumerate(ag2destock_j)).get, otypes=['float32'])(lumap)
    trans_cost_r = np.nan_to_num(trans_cost_r)
    trans_cost_r = tools.amortise(trans_cost_r * data.REAL_AREA)
    trans_cost_r[~cells] = 0.0
            
    # Establishment cost, proportional to the `increase of biodiversity/habitat contribution` * `Environmental Planting establishment cost per ha`
    # Note: This is just a notional cost, not meaning any actual planting is done
    HCAS_benefit_mult = {lu:1 - data.BIO_HABITAT_CONTRIBUTION_LOOK_UP[lu] for lu in data.LU_LVSTK_NATURAL}
    removal_cost_r = np.vectorize(HCAS_benefit_mult.get, otypes=[np.float32])(lumap) * data.EP_EST_COST_HA
    removal_cost_r = np.nan_to_num(removal_cost_r)
    removal_cost_r = tools.amortise(removal_cost_r * data.REAL_AREA)
 
    est_costs_r = removal_cost_r + trans_cost_r
        
    # Water costs; Assume destocked land is dryland
    w_rm_irrig_cost_r = (np.where(lmmap == 1, settings.REMOVE_IRRIG_COST * data.IRRIG_COST_MULTS[target_year], 0) * data.REAL_AREA).astype(np.float32)
    w_rm_irrig_cost_r[~cells] = 0.0

    flat = {
        'Establishment cost (Ag2Non-Ag)':     est_costs_r,
        'Remove irrigation cost (Ag2Non-Ag)': w_rm_irrig_cost_r,
    } if separate else (est_costs_r + w_rm_irrig_cost_r)
    result = {}
    for from_m in range(data.NLMS):
        for from_j in range(data.N_AG_LUS):
            src_cells = np.where((lmmap == from_m) & (lumap == from_j))[0]
            if src_cells.size == 0:
                continue
            if separate:
                result[(from_m, from_j)] = {k: v[src_cells] for k, v in flat.items()}
            else:
                result[(from_m, from_j)] = flat[src_cells]
    return result


def get_destocked_from_ag_blend(data: Data, base_year: int, target_year: int, separate=False) -> dict:
    """Blended Destocked transition cost in flow (from_m, from_j) format.

    Only livestock-natural sources (T_MAT[from_j -> Destocked] finite) are eligible. Per eligible
    source: own T_MAT(destock) + HCAS removal weighted by frac_s / eligible_frac (NOT ag_frac, so
    scatter-sum reproduces the flat blend's eligible-pool normalisation); remove-irrigation by
    frac_s for irrigated eligible sources.
    """
    dvars         = data.ag_dvars[base_year]
    thr           = 10 ** (-settings.ROUND_DECIMALS)
    area          = data.REAL_AREA
    HCAS          = {lu: 1 - data.BIO_HABITAT_CONTRIBUTION_LOOK_UP[lu] for lu in data.LU_LVSTK_NATURAL}
    t_j_raw       = data.T_MAT.sel(from_lu=data.AGRICULTURAL_LANDUSES, to_lu='Destocked - natural land').values
    eligible_j    = ~np.isnan(t_j_raw)
    rm_irrig_rate = (settings.REMOVE_IRRIG_COST * data.IRRIG_COST_MULTS[target_year] * area).astype(np.float32)

    # Normalise over the eligible (lvstk-natural) pool only — matches the flat blend.
    eligible_frac      = np.zeros(data.NCELLS, dtype=np.float32)
    for j in np.where(eligible_j)[0]:
        eligible_frac += dvars[:, :, j].sum(axis=0)
    eligible_frac_safe = np.where(eligible_frac > thr, eligible_frac, 1.0)

    result = {}
    for from_m in range(data.NLMS):
        for from_j in range(data.N_AG_LUS):
            frac_s = dvars[from_m, :, from_j]
            cells  = np.where(frac_s > thr)[0]
            if cells.size == 0:
                continue
            w_norm     = (frac_s[cells] / eligible_frac_safe[cells]).astype(np.float32)
            t_s        = (tools.amortise(np.nan_to_num(t_j_raw[from_j]) * area[cells]).astype(np.float32) * w_norm).astype(np.float32)
            removal_s  = (tools.amortise(HCAS.get(from_j, 0.0) * data.EP_EST_COST_HA[cells] * area[cells]).astype(np.float32) * w_norm).astype(np.float32)
            est_s      = (t_s + removal_s).astype(np.float32)
            w_s        = ((rm_irrig_rate[cells] * frac_s[cells]).astype(np.float32)
                          if (from_m == 1 and eligible_j[from_j]) else np.zeros(cells.size, dtype=np.float32))
            if separate:
                result[(from_m, from_j)] = {
                    'Establishment cost (Ag2Non-Ag)':     est_s,
                    'Remove irrigation cost (Ag2Non-Ag)': w_s,
                }
            else:
                result[(from_m, from_j)] = est_s + w_s
    return result


def get_destocked_from_ag_exact(
    data: Data, target_year: int, mj_cell_map: dict, separate=False
) -> dict:
    """Exact Destocked transition costs per-(from_m, from_j) combo."""
    t_j_raw      = data.T_MAT.sel(from_lu=data.AGRICULTURAL_LANDUSES, to_lu='Destocked - natural land').values
    irr_scalar   = settings.REMOVE_IRRIG_COST * data.IRRIG_COST_MULTS[target_year]
    HCAS_benefit_mult = {lu: 1 - data.BIO_HABITAT_CONTRIBUTION_LOOK_UP[lu] for lu in data.LU_LVSTK_NATURAL}

    result = {}
    for (from_m, from_j), cell_idx in mj_cell_map.items():
        is_eligible  = not np.isnan(t_j_raw[from_j])   # NaN means non-lvstk-natural, zero cost
        hcas_mult    = HCAS_benefit_mult.get(from_j, 0.0)  # LU_LVSTK_NATURAL contains int codes

        t_cost       = tools.amortise(np.nan_to_num(t_j_raw[from_j]) * data.REAL_AREA[cell_idx]).astype(np.float32)
        removal_cost = tools.amortise(hcas_mult * data.EP_EST_COST_HA[cell_idx] * data.REAL_AREA[cell_idx]).astype(np.float32)
        est_costs    = t_cost + removal_cost

        w_rm_irrig   = (
            (irr_scalar * data.REAL_AREA[cell_idx]).astype(np.float32)
            if (from_m == 1 and is_eligible) else np.zeros(len(cell_idx), dtype=np.float32)
        )

        if separate:
            result[(from_m, from_j)] = {
                'Establishment cost (Ag2Non-Ag)':     est_costs,
                'Remove irrigation cost (Ag2Non-Ag)': w_rm_irrig,
            }
        else:
            result[(from_m, from_j)] = est_costs + w_rm_irrig
    return result


def get_destocked_from_ag_from_base_year(
    data: Data, base_year: int, target_year: int, separate=False
) -> np.ndarray | dict:
    """Dispatcher: exact, blended or crisp-lumap Destocked transition costs."""
    if settings.TRANSITION_MODE == 'exact':
        mj_cell_map = get_base_dvar_mj_cell_map(data, base_year)
        return get_destocked_from_ag_exact(data, target_year, mj_cell_map, separate)
    elif settings.TRANSITION_MODE == 'blend':
        return get_destocked_from_ag_blend(data, base_year, target_year, separate)
    elif settings.TRANSITION_MODE == 'crisp':
        return get_destocked_from_ag_crisp(data, base_year, target_year, separate)
    else:
        raise ValueError(f"Unknown TRANSITION_MODE {settings.TRANSITION_MODE!r}; expected 'exact', 'blend', or 'crisp'.")


def get_transition_matrix_ag2nonag(data: Data, base_year: int, target_year: int, separate: bool = False) -> dict:
    """Assemble ag→non-ag transition costs in flow format: dict[lu_name -> dict[(from_m, from_j) -> ndarray]].

    Mode-independent: each per-lever `_from_base_year` router already dispatches crisp/blend/exact
    and returns the same from-based dict, so this just assembles them by non-ag target LU. (The old
    per-mode `_crisp`/`_blend`/`_exact` assemblers behind a second mode dispatch are gone — they were
    structurally identical once every lever returned the uniform flow dict.) input_data selects the
    per-source diagonal `dict[k]` for each lu_name.
    """
    return {
        'Environmental Plantings':       get_env_plant_transitions_from_ag_from_base_year(data, base_year, target_year, separate),
        'Riparian Plantings':            get_rip_plant_transitions_from_ag_from_base_year(data, base_year, target_year, separate),
        'Sheep Agroforestry':            get_sheep_agroforestry_transitions_from_ag_from_base_year(data, base_year, target_year, separate),
        'Beef Agroforestry':             get_beef_agroforestry_transitions_from_ag_from_base_year(data, base_year, target_year, separate),
        'Carbon Plantings (Block)':      get_carbon_plantings_block_from_ag_from_base_year(data, base_year, target_year, separate),
        'Sheep Carbon Plantings (Belt)': get_sheep_carbon_plantings_belt_from_ag_from_base_year(data, base_year, target_year, separate),
        'Beef Carbon Plantings (Belt)':  get_beef_carbon_plantings_belt_from_ag_from_base_year(data, base_year, target_year, separate),
        'BECCS':                         get_beccs_from_ag(data, base_year, target_year, separate),
        'Destocked - natural land':      get_destocked_from_ag_from_base_year(data, base_year, target_year, separate),
    }



@lru_cache(maxsize=1)
def get_base_nonag_dvar_k_cell_map(data: Data, base_year: int, threshold: float = None) -> dict:
    """Slice the base-year non-ag dvar by each source k, returning {k: cell_idx} for every nonag LU
    whose dvar fraction exceeds `threshold`. Only used in exact mode.

    ★ THIS IS THE KEY DESIGN FOR EXACT TRANSITION COST (non-ag source side). We slice the base-year
    dvar by the same source k. All cells in one slice share the same transition costs (the cost of
    leaving source k for each target). The solver then creates, for each slice, the same number of
    delta variables (delta >= 0, positive-increment Gurobi vars) — one per sliced cell — that
    represent the TRUE transition flow out of source k on those cells. Transition cost is then
    trans_cost = delta_cells * cost_cells, and the objective minimises sum(trans_cost). This replaces
    the crisp/blend approximation where a whole cell carried a single dominant-LU cost.

    `threshold` defaults to settings.EXACT_REACHABILITY_MIN_FRACTION — the same fraction used by
    `get_base_dvar_mj_cell_map` and exact-mode eligibility, so the nonag source slices, cost dicts,
    and `get_to_ag_exclude_matrices_exact` reachability all agree on one threshold.

    Cached (maxsize=1): all exact nonag→ag functions call this for the same (data, base_year)
    pair within one solve step, so subsequent calls are free.
    """
    threshold = settings.EXACT_REACHABILITY_MIN_FRACTION if threshold is None else threshold
    base_dvar_rk = data.non_ag_dvars[base_year]
    return {
        k: np.where(base_dvar_rk[:, k] > threshold)[0]
        for k in range(data.N_NON_AG_LUS)
        if (base_dvar_rk[:, k] > threshold).any()
    }


def get_nonag2ag_ub_crisp(data: Data, lumap: np.ndarray) -> np.ndarray:
    """nonag→ag TARGET upper bound (NLMS, NCELLS, N_AG_LUS), 0/1 — non-ag-source component.

    TO-view bound, non-ag-source side. For a non-ag-dominant cell, ag target to_j is allowed (1) iff
    BOTH its current non-ag LU and its 2010 ag status can transition to to_j (mirrors the non-ag
    branch of get_to_ag_exclude_matrices_crisp), spatially allowed (data.EXCLUDE), and not no-go.
    Ag-dominant cells contribute 0 here (their reach to ag is get_ag2ag_ub). Combined with
    get_ag2ag_ub_crisp this reconstructs get_to_ag_exclude_matrices_crisp exactly. crisp/blend share this.

    Parametrised by `lumap` (not base_year): the dispatcher passes the real base-year map, while a
    synthetic pure-ag map (e.g. all-sheep) yields zero here (no non-ag cells) — exactly what the cost
    primitive's per-source exclude needs.
    """
    t_ij  = data.T_MAT.loc[:, data.AGRICULTURAL_LANDUSES]                # all from_lu → ag (xr; NaN=forbidden)
    ag_cells, non_ag_cells = tools.get_ag_and_non_ag_cells(lumap)
    lumap2desc = np.vectorize(data.ALLLU2DESC.get, otypes=[str])

    # Transition exclusion (T_MAT): current non-ag LU → to_j AND 2010 ag status → to_j must both allow.
    # NaN default → ag cells (and forbidden transitions) collapse to 0; allowed → 1.
    t_rj = np.full((data.NCELLS, data.N_AG_LUS), np.nan, dtype=np.float32)
    if len(non_ag_cells):
        t_rj[non_ag_cells] = (
            t_ij.sel(from_lu=lumap2desc(lumap[non_ag_cells])).values
            * t_ij.sel(from_lu=lumap2desc(data.LUMAP[non_ag_cells])).values
        )
    reach_rj = np.where(np.isnan(t_rj), 0.0, 1.0).astype(np.float32)

    # No-go exclusion: user-defined LUs banned in specific regions.
    no_go = np.ones((data.NLMS, data.NCELLS, data.N_AG_LUS), dtype=np.float32)
    if settings.EXCLUDE_NO_GO_LU:
        for no_go_x_r, no_go_desc in zip(data.NO_GO_REGION_AG, data.NO_GO_LANDUSE_AG):
            no_go[:, :, data.DESC2AGLU[no_go_desc]] = no_go_x_r

    # Spatial exclusion (data.EXCLUDE): LU never present in the SA2 region in 2010 → banned there.
    x_mrj = data.EXCLUDE.astype(np.float32)
    return (x_mrj * reach_rj[np.newaxis, :, :] * no_go).astype(np.float32)


def get_nonag2ag_ub_exact(data: Data, base_year: int) -> np.ndarray:
    """nonag→ag TARGET upper bound (NLMS, NCELLS, N_AG_LUS), FRACTIONAL — non-ag-source component.

    ub[to_m, r, to_j] = (fraction of cell r's non-ag land that may reach to_j) × data.EXCLUDE × no-go,
    where the reachable fraction = Σ_{k : T_MAT[k→to_j] finite} base_nonag_dvar[r, k]. ag-source share
    is added separately in the combined ag ub (later step).
    """
    non_ag_dvar = data.non_ag_dvars[base_year]                          # (NCELLS, N_NON_AG_LUS)
    # Transition exclusion (T_MAT): binary allow per (nonag k → to_j).
    t_kj = (~np.isnan(
        data.T_MAT.sel(from_lu=data.NON_AGRICULTURAL_LANDUSES, to_lu=data.AGRICULTURAL_LANDUSES).values
    )).astype(np.float32)                                               # (k, to_j)
    # Reachable land share: sum base-year fractions of every non-ag source that can reach to_j.
    reach_frac_rj = (non_ag_dvar @ t_kj).astype(np.float32)             # (NCELLS, to_j)

    # No-go exclusion: user-defined LUs banned in specific regions.
    no_go = np.ones((data.NLMS, data.NCELLS, data.N_AG_LUS), dtype=np.float32)
    if settings.EXCLUDE_NO_GO_LU:
        for no_go_x_r, no_go_desc in zip(data.NO_GO_REGION_AG, data.NO_GO_LANDUSE_AG):
            no_go[:, :, data.DESC2AGLU[no_go_desc]] = no_go_x_r

    # Spatial exclusion (data.EXCLUDE).
    x_mrj = data.EXCLUDE.astype(np.float32)
    return (x_mrj * reach_frac_rj[np.newaxis, :, :] * no_go).astype(np.float32)


def get_nonag2ag_ub(data: Data, base_year: int) -> np.ndarray:
    """Dispatch the nonag→ag target upper bound by TRANSITION_MODE.

    crisp / blend: 0/1 from the dominant non-ag LU (+ 2010 ag status) — get_nonag2ag_ub_crisp.
    exact:         fractional reachable non-ag-land share — get_nonag2ag_ub_exact.
    """
    if settings.TRANSITION_MODE in ('crisp', 'blend'):
        return get_nonag2ag_ub_crisp(data, data.lumaps[base_year])
    elif settings.TRANSITION_MODE == 'exact':
        return get_nonag2ag_ub_exact(data, base_year)
    else:
        raise ValueError(f"Unknown TRANSITION_MODE: {settings.TRANSITION_MODE!r}")


def get_nonag2ag_lb(data: Data, base_year: int) -> np.ndarray:
    """nonag→ag TARGET lower bound (NLMS, NCELLS, N_AG_LUS) — all zeros for now (placeholder)."""
    return np.zeros((data.NLMS, data.NCELLS, data.N_AG_LUS), dtype=np.float32)



def get_env_plantings_to_ag_base(data: Data, base_year: int, target_year: int, separate=False) -> np.ndarray | dict:
    """Flat EP→ag transition-cost primitive: (NLMS, NCELLS, N_AG_LUS), cost for each non-ag cell to
    convert to ag target (to_m, to_j); AG cells are zeroed (the cost lives only on non-ag cells).

    Reused directly as a flat array by the sheep/beef agroforestry and CP-belt _to_ag functions
    (which combine it with the sheep/beef ag2ag cost). The from-based get_env_plantings_to_ag_crisp
    slices this by NON-AG dominant source k.
    """
    lumap = data.lumaps[base_year]
    lmmap = data.lmmaps[base_year]
    ag_cells, nonag_cells = tools.get_ag_and_non_ag_cells(lumap)

    base_ep_to_ag_t = data.EP2AG_TRANSITION_COSTS_HA * data.TRANS_COST_MULTS[target_year]
    base_ep_to_ag_t_mrj = np.broadcast_to(base_ep_to_ag_t, (data.NLMS, data.NCELLS, base_ep_to_ag_t.shape[0]))
    base_ep_to_ag_t_mrj = tools.amortise(base_ep_to_ag_t_mrj).copy()
    base_ep_to_ag_t_mrj[:, ag_cells, :] = 0

    l_mrj = tools.lumap2ag_l_mrj(lumap, lmmap)
    w_mrj = ag_water.get_wreq_matrices(data, target_year - data.YR_CAL_BASE)
    w_delta_mrj = tools.get_ag_to_ag_water_delta_matrix(w_mrj, l_mrj, data, target_year - data.YR_CAL_BASE)
    w_delta_mrj[:, ag_cells, :] = 0
    if len(nonag_cells):
        new_irrig = tools.amortise(settings.NEW_IRRIG_COST * data.IRRIG_COST_MULTS[target_year] * data.REAL_AREA[nonag_cells]).astype(np.float32)
        w_delta_mrj[1, nonag_cells, :] += new_irrig[:, np.newaxis]

    base_cost_mrj  = np.nan_to_num(base_ep_to_ag_t_mrj * data.REAL_AREA[np.newaxis, :, np.newaxis])
    water_cost_mrj = np.nan_to_num(w_delta_mrj)

    if separate:
        return {
            'Transition cost (Non-Ag2Ag)': base_cost_mrj,
            'Water license cost (Non-Ag2Ag)': water_cost_mrj,
        }
    return base_cost_mrj + water_cost_mrj


def get_env_plantings_to_ag_crisp(data: Data, base_year: int, target_year: int, separate=False) -> dict:
    """EP→ag transition cost in flow format, keyed by NON-AG source k (crisp mode).

    Returns dict[k] -> ndarray(NLMS, ncells_k, N_AG_LUS)  [to_m, r, to_j]
    (separate=True -> the leaf is {component: same-shape array}), where ncells_k is the number of
    cells whose DOMINANT base-year LU is non-ag LU k. The source is dry (non-ag land is always
    dryland); the NLMS axis is the TARGET land management. Computes the flat base once and slices
    it by non-ag dominant source (each non-ag cell in exactly one k). Mirrors
    get_env_plantings_to_ag_exact's shape; collapsing back reproduces the flat base (ag cells,
    already zeroed there, are not keyed here).
    """
    flat      = get_env_plantings_to_ag_base(data, base_year, target_year, separate)
    lumap     = data.lumaps[base_year]
    base_code = settings.NON_AGRICULTURAL_LU_BASE_CODE

    result = {}
    for k in range(data.N_NON_AG_LUS):
        cells = np.where(lumap == base_code + k)[0]
        if cells.size == 0:
            continue
        if separate:
            result[k] = {key: v[:, cells, :] for key, v in flat.items()}
        else:
            result[k] = flat[:, cells, :]
    return result


def get_env_plantings_to_ag_blend(data: Data, base_year: int, target_year: int, separate=False) -> dict:
    """EP→ag transition cost in flow format, keyed by NON-AG source k (blend mode).

    Returns {k: ndarray(NLMS, ncells_k, N_AG_LUS)} for every non-ag source present in the base-year
    dvar (> rounding threshold), each carrying the generic non-ag→ag per-unit cost over k's cells
    (the flow var supplies the actual fraction). Source is dry; the NLMS axis is the target lm.
    Same structure as get_env_plantings_to_ag_exact, but cell selection uses the blend (rounding)
    threshold rather than the exact reachability fraction.
    """
    thr             = 10 ** (-settings.ROUND_DECIMALS)
    base_ep_to_ag_t = data.EP2AG_TRANSITION_COSTS_HA * data.TRANS_COST_MULTS[target_year]   # (N_AG_LUS,)
    w_mrj           = ag_water.get_wreq_matrices(data, target_year - data.YR_CAL_BASE)
    irr_scalar      = settings.NEW_IRRIG_COST * data.IRRIG_COST_MULTS[target_year]
    non_ag_dvars    = data.non_ag_dvars[base_year]

    result = {}
    for k in range(data.N_NON_AG_LUS):
        cells = np.where(non_ag_dvars[:, k] > thr)[0]
        if cells.size == 0:
            continue
        base_cost_rj  = tools.amortise(np.nan_to_num(base_ep_to_ag_t) * data.REAL_AREA[cells, np.newaxis]).astype(np.float32)
        base_cost_mrj = np.stack([base_cost_rj, base_cost_rj], axis=0)
        water_cost_mrj = np.nan_to_num(
            w_mrj[:, cells, :]
            * data.WATER_LICENCE_PRICE[cells, np.newaxis]
            * data.WATER_LICENSE_COST_MULTS[target_year]
            * settings.INCLUDE_WATER_LICENSE_COSTS
        ).astype(np.float32)
        new_irrig_r = tools.amortise(irr_scalar * data.REAL_AREA[cells]).astype(np.float32)
        water_cost_mrj[1] += new_irrig_r[:, np.newaxis]
        if separate:
            result[k] = {
                'Transition cost (Non-Ag2Ag)':    base_cost_mrj,
                'Water license cost (Non-Ag2Ag)': water_cost_mrj,
            }
        else:
            result[k] = base_cost_mrj + water_cost_mrj
    return result


def get_env_plantings_to_ag_exact(
    data: Data, target_year: int, k_cell_map: dict, separate=False
) -> dict:
    """Exact EP→ag costs per nonag LU k, indexed to dvar-active cells only.

    Parameters
    ----------
    k_cell_map : dict returned by get_base_nonag_dvar_k_cell_map(data, base_year)

    Returns
    -------
    dict[k] → ndarray (NLMS, len(cell_idx), N_AG_LUS) in $/cell/yr
    OR, if separate=True:
    dict[k] → {'Transition cost (Non-Ag2Ag)', 'Water license cost (Non-Ag2Ag)'}
               each ndarray (NLMS, len(cell_idx), N_AG_LUS)
    """

    # Hoist invariants outside the per-k loop
    base_ep_to_ag_t = data.EP2AG_TRANSITION_COSTS_HA * data.TRANS_COST_MULTS[target_year]  # (N_AG_LUS,)
    w_mrj = ag_water.get_wreq_matrices(data, target_year - data.YR_CAL_BASE)
    irr_scalar = settings.NEW_IRRIG_COST * data.IRRIG_COST_MULTS[target_year]

    result = {}
    for k, cell_idx in k_cell_map.items():
        # Base transition cost: same for both lm slices (EP source is always dryland)
        base_cost_rj = tools.amortise(np.nan_to_num(base_ep_to_ag_t) * data.REAL_AREA[cell_idx, np.newaxis]).astype(np.float32)  # (n, N_AG_LUS)
        base_cost_mrj = np.stack([base_cost_rj, base_cost_rj], axis=0)  # (NLMS, n, N_AG_LUS)

        # Water license: source water req = 0 (nonag), so delta = full target req
        water_cost_mrj = np.nan_to_num(
            w_mrj[:, cell_idx, :]                          # (NLMS, n, N_AG_LUS)
            * data.WATER_LICENCE_PRICE[cell_idx, np.newaxis]  # (n, 1) → broadcasts
            * data.WATER_LICENSE_COST_MULTS[target_year]
            * settings.INCLUDE_WATER_LICENSE_COSTS
        ).astype(np.float32)

        # New irrigation setup: nonag source is always dryland → lm=1 always pays this
        new_irrig_r = tools.amortise(irr_scalar * data.REAL_AREA[cell_idx]).astype(np.float32)  # (n,)
        water_cost_mrj[1] += new_irrig_r[:, np.newaxis]

        if separate:
            result[k] = {
                'Transition cost (Non-Ag2Ag)':    base_cost_mrj,
                'Water license cost (Non-Ag2Ag)': water_cost_mrj,
            }
        else:
            result[k] = base_cost_mrj + water_cost_mrj

    return result


def get_env_plantings_to_ag(
    data: Data, base_year: int, target_year: int, separate=False
) -> np.ndarray | dict:
    """Dispatcher: exact, blend, or crisp EP→ag transition costs.

    exact  → dict{k: ndarray(NLMS, len(cell_idx), N_AG_LUS)}  (solver integration: future)
    blend  → ndarray(NLMS, NCELLS, N_AG_LUS), full per-unit rate everywhere
    crisp  → ndarray(NLMS, NCELLS, N_AG_LUS), nonag cells only (ag cells zeroed)
    """
    if settings.TRANSITION_MODE == 'exact':
        k_cell_map = get_base_nonag_dvar_k_cell_map(data, base_year)
        return get_env_plantings_to_ag_exact(data, target_year, k_cell_map, separate)
    elif settings.TRANSITION_MODE == 'blend':
        return get_env_plantings_to_ag_blend(data, base_year, target_year, separate)
    elif settings.TRANSITION_MODE == 'crisp':
        return get_env_plantings_to_ag_crisp(data, base_year, target_year, separate)
    else:
        raise ValueError(
            f"Unknown TRANSITION_MODE {settings.TRANSITION_MODE!r}; expected 'exact', 'blend', or 'crisp'."
        )


def get_rip_plantings_to_ag(data: Data, base_year: int, target_year: int, separate=False):
    # Same as EP — get_transition_matrix_nonag2ag assigns ep_to_ag directly; this stub is unused.
    pass


def get_agroforestry_to_ag_base(data: Data, base_year: int, target_year: int, separate=False):
    # Same as EP — callers now call get_env_plantings_to_ag directly; this stub is unused.
    pass


def get_sheep_agroforestry_to_ag_crisp(
    data: Data, base_year: int, target_year: int, agroforestry_x_r, separate=False
) -> dict:
    """Sheep AF→ag cost in flow format, keyed by the Sheep Agroforestry source k.

    Returns {sheep_af_k: ndarray(NLMS, ncells_k, N_AG_LUS)} (separate=True -> {sheep_af_k: {comp: ...}}),
    where ncells_k = cells whose dominant base-year LU is Sheep Agroforestry. combined =
    x_r*EP_base + (1-x_r)*sheep_base, sliced to those cells. Mirrors
    get_sheep_agroforestry_to_ag_exact but with crisp's per-cell agroforestry_x_r and
    dominant-LU cell detection (no cell_mask needed — the slice keeps only the AF cells).
    """
    lumap      = data.lumaps[base_year]
    yr_idx     = target_year - data.YR_CAL_BASE
    sheep_af_k = data.NON_AGRICULTURAL_LANDUSES.index('Sheep Agroforestry')
    cells      = tools.get_sheep_agroforestry_cells(lumap)

    sheep_j         = tools.get_sheep_code(data)
    all_sheep_lumap = np.full(data.NCELLS, sheep_j, dtype=np.int8)
    all_dry_lmmap   = np.zeros(data.NCELLS, dtype=np.float32)
    sheep_tcosts    = get_transition_matrices_ag2ag_base(data, yr_idx, all_sheep_lumap, all_dry_lmmap, separate)

    agroforestry_tcosts = get_env_plantings_to_ag_base(data, base_year, target_year, separate)
    x_r = agroforestry_x_r[np.newaxis, :, np.newaxis]  # AF proportion broadcast over (m, r, j)

    if separate:
        all_keys = set(agroforestry_tcosts) | set(sheep_tcosts)
        zeros = np.zeros((data.NLMS, data.NCELLS, data.N_AG_LUS), dtype=np.float32)
        return {sheep_af_k: {key: (x_r * agroforestry_tcosts.get(key, zeros) + (1 - x_r) * sheep_tcosts.get(key, zeros))[:, cells, :].astype(np.float32) for key in all_keys}}
    full = (x_r * agroforestry_tcosts + (1 - x_r) * sheep_tcosts).astype(np.float32)
    return {sheep_af_k: full[:, cells, :]}


def get_sheep_agroforestry_to_ag_blend(
    data: Data, base_year: int, target_year: int, separate=False
) -> dict:
    """Blend Sheep AF→ag cost in flow format: {sheep_af_k: ndarray(NLMS, ncells_k, N_AG_LUS)}.

    Per-unit combined cost (af_prop·EP_base + (1-af_prop)·sheep_base) sliced to cells with any
    Sheep-AF dvar presence (> rounding threshold). Same as the exact variant in flow form.
    """
    thr        = 10 ** (-settings.ROUND_DECIMALS)
    sheep_af_k = data.NON_AGRICULTURAL_LANDUSES.index('Sheep Agroforestry')
    cells      = np.where(data.non_ag_dvars[base_year][:, sheep_af_k] > thr)[0]
    yr_idx     = target_year - data.YR_CAL_BASE

    sheep_j         = tools.get_sheep_code(data)
    all_sheep_lumap = np.full(data.NCELLS, sheep_j, dtype=np.int8)
    all_dry_lmmap   = np.zeros(data.NCELLS, dtype=np.float32)
    sheep_tcosts    = get_transition_matrices_ag2ag_base(data, yr_idx, all_sheep_lumap, all_dry_lmmap, separate)

    agroforestry_tcosts = get_env_plantings_to_ag_base(data, base_year, target_year, separate)
    af_prop = settings.AF_PROPORTION

    if separate:
        all_keys = set(agroforestry_tcosts) | set(sheep_tcosts)
        zeros = np.zeros((data.NLMS, data.NCELLS, data.N_AG_LUS), dtype=np.float32)
        return {sheep_af_k: {key: (af_prop * agroforestry_tcosts.get(key, zeros) + (1 - af_prop) * sheep_tcosts.get(key, zeros))[:, cells, :].astype(np.float32) for key in all_keys}}
    full = (af_prop * agroforestry_tcosts + (1 - af_prop) * sheep_tcosts).astype(np.float32)
    return {sheep_af_k: full[:, cells, :]}


def get_sheep_agroforestry_to_ag_exact(
    data: Data, base_year: int, target_year: int, separate=False
) -> dict:
    """Exact Sheep AF→ag costs: {sheep_af_k: array(NLMS, n_active_cells, N_AG_LUS)}.

    Per-unit costs are identical to crisp (all-dry-sheep maps); the sparse dict
    format lets the solver multiply by actual dvar values for exact weighting.
    """
    threshold  = 10 ** (-settings.ROUND_DECIMALS)
    sheep_af_k = data.NON_AGRICULTURAL_LANDUSES.index('Sheep Agroforestry')
    cell_idx   = np.where(data.non_ag_dvars[base_year][:, sheep_af_k] > threshold)[0]
    yr_idx     = target_year - data.YR_CAL_BASE

    sheep_j         = tools.get_sheep_code(data)
    all_sheep_lumap = np.full(data.NCELLS, sheep_j, dtype=np.int8)
    all_dry_lmmap   = np.zeros(data.NCELLS, dtype=np.float32)
    sheep_tcosts    = get_transition_matrices_ag2ag_base(data, yr_idx, all_sheep_lumap, all_dry_lmmap, separate)

    agroforestry_tcosts = get_env_plantings_to_ag_base(data, base_year, target_year, separate)
    af_prop = settings.AF_PROPORTION

    if separate:
        all_keys = set(agroforestry_tcosts) | set(sheep_tcosts)
        zeros = np.zeros((data.NLMS, data.NCELLS, data.N_AG_LUS), dtype=np.float32)
        return {sheep_af_k: {key: (af_prop * agroforestry_tcosts.get(key, zeros) + (1 - af_prop) * sheep_tcosts.get(key, zeros))[:, cell_idx, :].astype(np.float32) for key in all_keys}}
    full = (af_prop * agroforestry_tcosts + (1 - af_prop) * sheep_tcosts).astype(np.float32)
    return {sheep_af_k: full[:, cell_idx, :]}


def get_sheep_agroforestry_to_ag(
    data: Data, base_year: int, target_year: int, agroforestry_x_r, separate=False
) -> np.ndarray | dict:
    """Router: dispatch to exact, blend, or crisp Sheep AF→ag transition costs."""
    if settings.TRANSITION_MODE == 'exact':
        return get_sheep_agroforestry_to_ag_exact(data, base_year, target_year, separate)
    elif settings.TRANSITION_MODE == 'blend':
        return get_sheep_agroforestry_to_ag_blend(data, base_year, target_year, separate)
    elif settings.TRANSITION_MODE == 'crisp':
        return get_sheep_agroforestry_to_ag_crisp(data, base_year, target_year, agroforestry_x_r, separate)
    else:
        raise ValueError(f"Unknown TRANSITION_MODE {settings.TRANSITION_MODE!r}; expected 'exact', 'blend', or 'crisp'.")


def get_beef_agroforestry_to_ag_crisp(
    data: Data, base_year: int, target_year: int, agroforestry_x_r, separate=False
) -> dict:
    """Beef AF→ag cost in flow format, keyed by the Beef Agroforestry source k.

    {beef_af_k: ndarray(NLMS, ncells_k, N_AG_LUS)} (separate -> {beef_af_k: {comp: ...}}),
    combined = x_r*EP_base + (1-x_r)*beef_base, sliced to beef-AF dominant cells. Mirrors
    get_beef_agroforestry_to_ag_exact with crisp's per-cell x_r and dominant-LU cell detection.
    """
    lumap     = data.lumaps[base_year]
    yr_idx    = target_year - data.YR_CAL_BASE
    beef_af_k = data.NON_AGRICULTURAL_LANDUSES.index('Beef Agroforestry')
    cells     = tools.get_beef_agroforestry_cells(lumap)

    beef_j         = tools.get_beef_code(data)
    all_beef_lumap = np.full(data.NCELLS, beef_j, dtype=np.int8)
    all_dry_lmmap  = np.zeros(data.NCELLS, dtype=np.float32)
    beef_tcosts    = get_transition_matrices_ag2ag_base(data, yr_idx, all_beef_lumap, all_dry_lmmap, separate)

    agroforestry_tcosts = get_env_plantings_to_ag_base(data, base_year, target_year, separate)
    x_r = agroforestry_x_r[np.newaxis, :, np.newaxis]

    if separate:
        all_keys = set(agroforestry_tcosts) | set(beef_tcosts)
        zeros = np.zeros((data.NLMS, data.NCELLS, data.N_AG_LUS), dtype=np.float32)
        return {beef_af_k: {key: (x_r * agroforestry_tcosts.get(key, zeros) + (1 - x_r) * beef_tcosts.get(key, zeros))[:, cells, :].astype(np.float32) for key in all_keys}}
    full = (x_r * agroforestry_tcosts + (1 - x_r) * beef_tcosts).astype(np.float32)
    return {beef_af_k: full[:, cells, :]}


def get_beef_agroforestry_to_ag_blend(
    data: Data, base_year: int, target_year: int, separate=False
) -> dict:
    """Blend Beef AF→ag cost in flow format: {beef_af_k: ndarray(NLMS, ncells_k, N_AG_LUS)}.

    Per-unit combined cost (af_prop·EP_base + (1-af_prop)·beef_base) sliced to cells with any
    Beef-AF dvar presence (> rounding threshold).
    """
    thr       = 10 ** (-settings.ROUND_DECIMALS)
    beef_af_k = data.NON_AGRICULTURAL_LANDUSES.index('Beef Agroforestry')
    cells     = np.where(data.non_ag_dvars[base_year][:, beef_af_k] > thr)[0]
    yr_idx    = target_year - data.YR_CAL_BASE

    beef_j         = tools.get_beef_code(data)
    all_beef_lumap = np.full(data.NCELLS, beef_j, dtype=np.int8)
    all_dry_lmmap  = np.zeros(data.NCELLS, dtype=np.float32)
    beef_tcosts    = get_transition_matrices_ag2ag_base(data, yr_idx, all_beef_lumap, all_dry_lmmap, separate)

    agroforestry_tcosts = get_env_plantings_to_ag_base(data, base_year, target_year, separate)
    af_prop = settings.AF_PROPORTION

    if separate:
        all_keys = set(agroforestry_tcosts) | set(beef_tcosts)
        zeros = np.zeros((data.NLMS, data.NCELLS, data.N_AG_LUS), dtype=np.float32)
        return {beef_af_k: {key: (af_prop * agroforestry_tcosts.get(key, zeros) + (1 - af_prop) * beef_tcosts.get(key, zeros))[:, cells, :].astype(np.float32) for key in all_keys}}
    full = (af_prop * agroforestry_tcosts + (1 - af_prop) * beef_tcosts).astype(np.float32)
    return {beef_af_k: full[:, cells, :]}


def get_beef_agroforestry_to_ag_exact(
    data: Data, base_year: int, target_year: int, separate=False
) -> dict:
    """Exact Beef AF→ag costs: {beef_af_k: array(NLMS, n_active_cells, N_AG_LUS)}."""
    threshold = 10 ** (-settings.ROUND_DECIMALS)
    beef_af_k = data.NON_AGRICULTURAL_LANDUSES.index('Beef Agroforestry')
    cell_idx  = np.where(data.non_ag_dvars[base_year][:, beef_af_k] > threshold)[0]
    yr_idx    = target_year - data.YR_CAL_BASE

    beef_j         = tools.get_beef_code(data)
    all_beef_lumap = np.full(data.NCELLS, beef_j, dtype=np.int8)
    all_dry_lmmap  = np.zeros(data.NCELLS, dtype=np.float32)
    beef_tcosts    = get_transition_matrices_ag2ag_base(data, yr_idx, all_beef_lumap, all_dry_lmmap, separate)

    agroforestry_tcosts = get_env_plantings_to_ag_base(data, base_year, target_year, separate)
    af_prop = settings.AF_PROPORTION

    if separate:
        all_keys = set(agroforestry_tcosts) | set(beef_tcosts)
        zeros = np.zeros((data.NLMS, data.NCELLS, data.N_AG_LUS), dtype=np.float32)
        return {beef_af_k: {key: (af_prop * agroforestry_tcosts.get(key, zeros) + (1 - af_prop) * beef_tcosts.get(key, zeros))[:, cell_idx, :].astype(np.float32) for key in all_keys}}
    full = (af_prop * agroforestry_tcosts + (1 - af_prop) * beef_tcosts).astype(np.float32)
    return {beef_af_k: full[:, cell_idx, :]}


def get_beef_agroforestry_to_ag(
    data: Data, base_year: int, target_year: int, agroforestry_x_r, separate=False
) -> np.ndarray | dict:
    """Router: dispatch to exact, blend, or crisp Beef AF→ag transition costs."""
    if settings.TRANSITION_MODE == 'exact':
        return get_beef_agroforestry_to_ag_exact(data, base_year, target_year, separate)
    elif settings.TRANSITION_MODE == 'blend':
        return get_beef_agroforestry_to_ag_blend(data, base_year, target_year, separate)
    elif settings.TRANSITION_MODE == 'crisp':
        return get_beef_agroforestry_to_ag_crisp(data, base_year, target_year, agroforestry_x_r, separate)
    else:
        raise ValueError(f"Unknown TRANSITION_MODE {settings.TRANSITION_MODE!r}; expected 'exact', 'blend', or 'crisp'.")


def get_carbon_plantings_block_to_ag(data: Data, base_year: int, target_year: int, separate=False):
    # Same as EP — get_transition_matrix_nonag2ag assigns ep_to_ag directly; this stub is unused.
    pass


def get_carbon_plantings_belt_to_ag_base(data: Data, base_year: int, target_year: int, separate=False) -> np.ndarray|dict:
    # Same as EP — callers now call get_env_plantings_to_ag directly; this stub is unused.
    pass


def get_sheep_carbon_plantings_belt_to_ag_crisp(
    data: Data, base_year: int, target_year: int, cp_belt_x_r, separate=False
) -> dict:
    """Sheep CP Belt→ag cost in flow format, keyed by the Sheep CP-Belt source k.

    {sheep_cpb_k: ndarray(NLMS, ncells_k, N_AG_LUS)} (separate -> {sheep_cpb_k: {comp: ...}}),
    combined = x_r*EP_base + (1-x_r)*sheep_base, sliced to sheep-CP-belt dominant cells.
    """
    lumap       = data.lumaps[base_year]
    yr_idx      = target_year - data.YR_CAL_BASE
    sheep_cpb_k = data.NON_AGRICULTURAL_LANDUSES.index('Sheep Carbon Plantings (Belt)')
    cells       = tools.get_sheep_carbon_plantings_belt_cells(lumap)

    sheep_j         = tools.get_sheep_code(data)
    all_sheep_lumap = np.full(data.NCELLS, sheep_j, dtype=np.int8)
    all_dry_lmmap   = np.zeros(data.NCELLS, dtype=np.float32)
    sheep_tcosts    = get_transition_matrices_ag2ag_base(data, yr_idx, all_sheep_lumap, all_dry_lmmap, separate)

    cp_belt_tcosts = get_env_plantings_to_ag_base(data, base_year, target_year, separate)
    x_r = cp_belt_x_r[np.newaxis, :, np.newaxis]

    if separate:
        all_keys = set(cp_belt_tcosts) | set(sheep_tcosts)
        zeros = np.zeros((data.NLMS, data.NCELLS, data.N_AG_LUS), dtype=np.float32)
        return {sheep_cpb_k: {key: (x_r * cp_belt_tcosts.get(key, zeros) + (1 - x_r) * sheep_tcosts.get(key, zeros))[:, cells, :].astype(np.float32) for key in all_keys}}
    full = (x_r * cp_belt_tcosts + (1 - x_r) * sheep_tcosts).astype(np.float32)
    return {sheep_cpb_k: full[:, cells, :]}


def get_sheep_carbon_plantings_belt_to_ag_blend(
    data: Data, base_year: int, target_year: int, separate=False
) -> dict:
    """Blend Sheep CP Belt→ag cost in flow format: {sheep_cpb_k: ndarray(NLMS, ncells_k, N_AG_LUS)}.

    Per-unit combined cost (cp_prop·EP_base + (1-cp_prop)·sheep_base) sliced to cells with any
    Sheep-CP-belt dvar presence (> rounding threshold).
    """
    thr         = 10 ** (-settings.ROUND_DECIMALS)
    sheep_cpb_k = data.NON_AGRICULTURAL_LANDUSES.index('Sheep Carbon Plantings (Belt)')
    cells       = np.where(data.non_ag_dvars[base_year][:, sheep_cpb_k] > thr)[0]
    yr_idx      = target_year - data.YR_CAL_BASE

    sheep_j         = tools.get_sheep_code(data)
    all_sheep_lumap = np.full(data.NCELLS, sheep_j, dtype=np.int8)
    all_dry_lmmap   = np.zeros(data.NCELLS, dtype=np.float32)
    sheep_tcosts    = get_transition_matrices_ag2ag_base(data, yr_idx, all_sheep_lumap, all_dry_lmmap, separate)

    cp_belt_tcosts = get_env_plantings_to_ag_base(data, base_year, target_year, separate)
    cp_prop = settings.CP_BELT_PROPORTION

    if separate:
        all_keys = set(cp_belt_tcosts) | set(sheep_tcosts)
        zeros = np.zeros((data.NLMS, data.NCELLS, data.N_AG_LUS), dtype=np.float32)
        return {sheep_cpb_k: {key: (cp_prop * cp_belt_tcosts.get(key, zeros) + (1 - cp_prop) * sheep_tcosts.get(key, zeros))[:, cells, :].astype(np.float32) for key in all_keys}}
    full = (cp_prop * cp_belt_tcosts + (1 - cp_prop) * sheep_tcosts).astype(np.float32)
    return {sheep_cpb_k: full[:, cells, :]}


def get_sheep_carbon_plantings_belt_to_ag_exact(
    data: Data, base_year: int, target_year: int, separate=False
) -> dict:
    """Exact Sheep CP Belt→ag costs: {sheep_cpb_k: array(NLMS, n_active_cells, N_AG_LUS)}."""
    threshold   = 10 ** (-settings.ROUND_DECIMALS)
    sheep_cpb_k = data.NON_AGRICULTURAL_LANDUSES.index('Sheep Carbon Plantings (Belt)')
    cell_idx    = np.where(data.non_ag_dvars[base_year][:, sheep_cpb_k] > threshold)[0]
    yr_idx      = target_year - data.YR_CAL_BASE

    sheep_j         = tools.get_sheep_code(data)
    all_sheep_lumap = np.full(data.NCELLS, sheep_j, dtype=np.int8)
    all_dry_lmmap   = np.zeros(data.NCELLS, dtype=np.float32)
    sheep_tcosts    = get_transition_matrices_ag2ag_base(data, yr_idx, all_sheep_lumap, all_dry_lmmap, separate)

    cp_belt_tcosts = get_env_plantings_to_ag_base(data, base_year, target_year, separate)
    cp_prop = settings.CP_BELT_PROPORTION

    if separate:
        all_keys = set(cp_belt_tcosts) | set(sheep_tcosts)
        zeros = np.zeros((data.NLMS, data.NCELLS, data.N_AG_LUS), dtype=np.float32)
        return {sheep_cpb_k: {key: (cp_prop * cp_belt_tcosts.get(key, zeros) + (1 - cp_prop) * sheep_tcosts.get(key, zeros))[:, cell_idx, :].astype(np.float32) for key in all_keys}}
    full = (cp_prop * cp_belt_tcosts + (1 - cp_prop) * sheep_tcosts).astype(np.float32)
    return {sheep_cpb_k: full[:, cell_idx, :]}


def get_sheep_carbon_plantings_belt_to_ag(
    data: Data, base_year: int, target_year: int, cp_belt_x_r, separate=False
) -> np.ndarray | dict:
    """Router: dispatch to exact, blend, or crisp Sheep CP Belt→ag transition costs."""
    if settings.TRANSITION_MODE == 'exact':
        return get_sheep_carbon_plantings_belt_to_ag_exact(data, base_year, target_year, separate)
    elif settings.TRANSITION_MODE == 'blend':
        return get_sheep_carbon_plantings_belt_to_ag_blend(data, base_year, target_year, separate)
    elif settings.TRANSITION_MODE == 'crisp':
        return get_sheep_carbon_plantings_belt_to_ag_crisp(data, base_year, target_year, cp_belt_x_r, separate)
    else:
        raise ValueError(f"Unknown TRANSITION_MODE {settings.TRANSITION_MODE!r}; expected 'exact', 'blend', or 'crisp'.")


def get_beef_carbon_plantings_belt_to_ag_crisp(
    data: Data, base_year: int, target_year: int, cp_belt_x_r, separate=False
) -> dict:
    """Beef CP Belt→ag cost in flow format, keyed by the Beef CP-Belt source k.

    {beef_cpb_k: ndarray(NLMS, ncells_k, N_AG_LUS)} (separate -> {beef_cpb_k: {comp: ...}}),
    combined = x_r*EP_base + (1-x_r)*beef_base, sliced to beef-CP-belt dominant cells.
    """
    lumap      = data.lumaps[base_year]
    yr_idx     = target_year - data.YR_CAL_BASE
    beef_cpb_k = data.NON_AGRICULTURAL_LANDUSES.index('Beef Carbon Plantings (Belt)')
    cells      = tools.get_beef_carbon_plantings_belt_cells(lumap)

    beef_j         = tools.get_beef_code(data)
    all_beef_lumap = np.full(data.NCELLS, beef_j, dtype=np.int8)
    all_dry_lmmap  = np.zeros(data.NCELLS, dtype=np.float32)
    beef_tcosts    = get_transition_matrices_ag2ag_base(data, yr_idx, all_beef_lumap, all_dry_lmmap, separate)

    cp_belt_tcosts = get_env_plantings_to_ag_base(data, base_year, target_year, separate)
    x_r = cp_belt_x_r[np.newaxis, :, np.newaxis]

    if separate:
        all_keys = set(cp_belt_tcosts) | set(beef_tcosts)
        zeros = np.zeros((data.NLMS, data.NCELLS, data.N_AG_LUS), dtype=np.float32)
        return {beef_cpb_k: {key: (x_r * cp_belt_tcosts.get(key, zeros) + (1 - x_r) * beef_tcosts.get(key, zeros))[:, cells, :].astype(np.float32) for key in all_keys}}
    full = (x_r * cp_belt_tcosts + (1 - x_r) * beef_tcosts).astype(np.float32)
    return {beef_cpb_k: full[:, cells, :]}


def get_beef_carbon_plantings_belt_to_ag_blend(
    data: Data, base_year: int, target_year: int, separate=False
) -> dict:
    """Blend Beef CP Belt→ag cost in flow format: {beef_cpb_k: ndarray(NLMS, ncells_k, N_AG_LUS)}.

    Per-unit combined cost (cp_prop·EP_base + (1-cp_prop)·beef_base) sliced to cells with any
    Beef-CP-belt dvar presence (> rounding threshold).
    """
    thr        = 10 ** (-settings.ROUND_DECIMALS)
    beef_cpb_k = data.NON_AGRICULTURAL_LANDUSES.index('Beef Carbon Plantings (Belt)')
    cells      = np.where(data.non_ag_dvars[base_year][:, beef_cpb_k] > thr)[0]
    yr_idx     = target_year - data.YR_CAL_BASE

    beef_j         = tools.get_beef_code(data)
    all_beef_lumap = np.full(data.NCELLS, beef_j, dtype=np.int8)
    all_dry_lmmap  = np.zeros(data.NCELLS, dtype=np.float32)
    beef_tcosts    = get_transition_matrices_ag2ag_base(data, yr_idx, all_beef_lumap, all_dry_lmmap, separate)

    cp_belt_tcosts = get_env_plantings_to_ag_base(data, base_year, target_year, separate)
    cp_prop = settings.CP_BELT_PROPORTION

    if separate:
        all_keys = set(cp_belt_tcosts) | set(beef_tcosts)
        zeros = np.zeros((data.NLMS, data.NCELLS, data.N_AG_LUS), dtype=np.float32)
        return {beef_cpb_k: {key: (cp_prop * cp_belt_tcosts.get(key, zeros) + (1 - cp_prop) * beef_tcosts.get(key, zeros))[:, cells, :].astype(np.float32) for key in all_keys}}
    full = (cp_prop * cp_belt_tcosts + (1 - cp_prop) * beef_tcosts).astype(np.float32)
    return {beef_cpb_k: full[:, cells, :]}


def get_beef_carbon_plantings_belt_to_ag_exact(
    data: Data, base_year: int, target_year: int, separate=False
) -> dict:
    """Exact Beef CP Belt→ag costs: {beef_cpb_k: array(NLMS, n_active_cells, N_AG_LUS)}."""
    threshold  = 10 ** (-settings.ROUND_DECIMALS)
    beef_cpb_k = data.NON_AGRICULTURAL_LANDUSES.index('Beef Carbon Plantings (Belt)')
    cell_idx   = np.where(data.non_ag_dvars[base_year][:, beef_cpb_k] > threshold)[0]
    yr_idx     = target_year - data.YR_CAL_BASE

    beef_j         = tools.get_beef_code(data)
    all_beef_lumap = np.full(data.NCELLS, beef_j, dtype=np.int8)
    all_dry_lmmap  = np.zeros(data.NCELLS, dtype=np.float32)
    beef_tcosts    = get_transition_matrices_ag2ag_base(data, yr_idx, all_beef_lumap, all_dry_lmmap, separate)

    cp_belt_tcosts = get_env_plantings_to_ag_base(data, base_year, target_year, separate)
    cp_prop = settings.CP_BELT_PROPORTION

    if separate:
        all_keys = set(cp_belt_tcosts) | set(beef_tcosts)
        zeros = np.zeros((data.NLMS, data.NCELLS, data.N_AG_LUS), dtype=np.float32)
        return {beef_cpb_k: {key: (cp_prop * cp_belt_tcosts.get(key, zeros) + (1 - cp_prop) * beef_tcosts.get(key, zeros))[:, cell_idx, :].astype(np.float32) for key in all_keys}}
    full = (cp_prop * cp_belt_tcosts + (1 - cp_prop) * beef_tcosts).astype(np.float32)
    return {beef_cpb_k: full[:, cell_idx, :]}


def get_beef_carbon_plantings_belt_to_ag(
    data: Data, base_year: int, target_year: int, cp_belt_x_r, separate=False
) -> np.ndarray | dict:
    """Router: dispatch to exact, blend, or crisp Beef CP Belt→ag transition costs."""
    if settings.TRANSITION_MODE == 'exact':
        return get_beef_carbon_plantings_belt_to_ag_exact(data, base_year, target_year, separate)
    elif settings.TRANSITION_MODE == 'blend':
        return get_beef_carbon_plantings_belt_to_ag_blend(data, base_year, target_year, separate)
    elif settings.TRANSITION_MODE == 'crisp':
        return get_beef_carbon_plantings_belt_to_ag_crisp(data, base_year, target_year, cp_belt_x_r, separate)
    else:
        raise ValueError(f"Unknown TRANSITION_MODE {settings.TRANSITION_MODE!r}; expected 'exact', 'blend', or 'crisp'.")


def get_beccs_to_ag(data: Data, target_year, lumap, lmmap, separate=False) -> np.ndarray|dict:
    # Same as EP — get_transition_matrix_nonag2ag assigns ep_to_ag directly; this stub is unused.
    pass
    

def get_destocked_to_ag_base(
    data: Data, target_year: int, cell_idx: np.ndarray, separate: bool = False
) -> dict:
    """Per-cell Destocked→ag transition cost over `cell_idx` (the Destocked source cells).

    Components (all $/cell/yr, amortised), mirroring the EP→ag structure plus a carbon-release term:
      - Transition cost: reverting destocked land to ag LU to_j, from Destocked's OWN
        T_MAT[Destocked→to_j] row (NaN/prohibited → 0). The source is dryland, so both target lm
        slices share it.
      - Water license cost: destocked source is dryland (req 0) → delta = full target req; plus the
        new-irrigation setup on the irrigated (lm=1) slice. Same as EP→ag.
      - Carbon release: destocked land is natural-equivalent (holds carbon); reverting to modified /
        livestock-natural land releases it. Reuses the ag2ag transition-emissions for a synthetic
        all-Unallocated-natural source, UNMASKED — the solver's nonag2ag eligibility (keyed on
        T_MAT[Destocked→to_j]) gates which flows exist.

    Fixes the previous model, which borrowed the whole ag2ag-from-Unallocated-natural cost: that used
    Unallocated-natural's T_MAT row AND its transition-exclude, which zeroed every component for
    Destocked→Unallocated-modified (Unalloc-nat→Unalloc-mod is prohibited) even though
    Destocked→Unalloc-mod is allowed (T_MAT=10390) and the cells hold carbon.

    Shared by crisp/blend/exact — they differ only in which cells are selected. Non-ag→ag cost is
    per-unit in every mode (the flow var supplies the fraction), so there is no fractional weighting.
    Returns {component: ndarray(NLMS, len(cell_idx), N_AG_LUS)} if separate else the summed array.
    """
    yr_idx = target_year - data.YR_CAL_BASE

    # --- Transition cost: Destocked's OWN T_MAT row; dryland source (both target lm equal) ---
    t_j = data.T_MAT.sel(from_lu='Destocked - natural land', to_lu=data.AGRICULTURAL_LANDUSES).values  # (N_AG_LUS,)
    t_j = np.nan_to_num(t_j) * data.TRANS_COST_MULTS[target_year]
    trans_rj  = tools.amortise(t_j[np.newaxis, :] * data.REAL_AREA[cell_idx, np.newaxis]).astype(np.float32)  # (n, N_AG_LUS)
    trans_mrj = np.stack([trans_rj, trans_rj], axis=0)                                                        # (NLMS, n, N_AG_LUS)

    # --- Water license cost: destocked source dryland (req 0) → delta = full target req ---
    w_mrj = ag_water.get_wreq_matrices(data, yr_idx)
    water_mrj = np.nan_to_num(
        w_mrj[:, cell_idx, :]
        * data.WATER_LICENCE_PRICE[cell_idx, np.newaxis]
        * data.WATER_LICENSE_COST_MULTS[target_year]
        * settings.INCLUDE_WATER_LICENSE_COSTS
    ).astype(np.float32)
    new_irrig_r = tools.amortise(
        settings.NEW_IRRIG_COST * data.IRRIG_COST_MULTS[target_year] * data.REAL_AREA[cell_idx]
    ).astype(np.float32)
    water_mrj[1] += new_irrig_r[:, np.newaxis]

    # --- Carbon release: natural-equivalent land → modified/lvstk-natural, UNMASKED ---
    unallocated_j     = tools.get_unallocated_natural_land_code(data)
    all_unalloc_lumap = np.full(data.NCELLS, unallocated_j, dtype=np.int8)
    ghg = ag_ghg.get_ghg_transition_emissions(data, all_unalloc_lumap, separate=False)      # (NLMS, NCELLS, N_AG_LUS) t/cell
    carbon_mrj = tools.amortise(
        ghg[:, cell_idx, :] * data.get_carbon_price_by_yr_idx(yr_idx)
    ).astype(np.float32)

    if separate:
        return {
            'Transition cost (Non-Ag2Ag)':     trans_mrj,
            'Water license cost (Non-Ag2Ag)':  water_mrj,
            'Carbon release cost (Non-Ag2Ag)': carbon_mrj,
        }
    return trans_mrj + water_mrj + carbon_mrj


def get_destocked_to_ag_crisp(
    data: Data, base_year: int, target_year: int, separate: bool = False
) -> dict:
    """Destocked→ag cost {destocked_k: ndarray(NLMS, ncells_k, N_AG_LUS)}, crisp dominant cells."""
    destocked_k = data.NON_AGRICULTURAL_LANDUSES.index('Destocked - natural land')
    cells       = tools.get_destocked_land_cells(data.lumaps[base_year])
    return {destocked_k: get_destocked_to_ag_base(data, target_year, cells, separate)}


def get_destocked_to_ag_blend(
    data: Data, base_year: int, target_year: int, separate: bool = False
) -> dict:
    """Destocked→ag cost {destocked_k: ...}, cells with any Destocked dvar presence (> rounding thr)."""
    destocked_k = data.NON_AGRICULTURAL_LANDUSES.index('Destocked - natural land')
    cells       = np.where(data.non_ag_dvars[base_year][:, destocked_k] > 10 ** (-settings.ROUND_DECIMALS))[0]
    return {destocked_k: get_destocked_to_ag_base(data, target_year, cells, separate)}


def get_destocked_to_ag_exact(
    data: Data, base_year: int, target_year: int, separate: bool = False
) -> dict:
    """Destocked→ag cost {destocked_k: ...}, cells with Destocked dvar > EXACT_REACHABILITY_MIN_FRACTION
    (matches get_base_nonag_dvar_k_cell_map so the cost dict and the exact source map agree)."""
    destocked_k = data.NON_AGRICULTURAL_LANDUSES.index('Destocked - natural land')
    cell_idx    = np.where(data.non_ag_dvars[base_year][:, destocked_k] > settings.EXACT_REACHABILITY_MIN_FRACTION)[0]
    return {destocked_k: get_destocked_to_ag_base(data, target_year, cell_idx, separate)}


def get_destocked_to_ag(
    data: Data, base_year: int, target_year: int, separate: bool = False
) -> np.ndarray | dict:
    """Router: dispatch to exact, blend, or crisp Destocked→ag transition costs."""
    if settings.TRANSITION_MODE == 'exact':
        return get_destocked_to_ag_exact(data, base_year, target_year, separate)
    elif settings.TRANSITION_MODE == 'blend':
        return get_destocked_to_ag_blend(data, base_year, target_year, separate)
    elif settings.TRANSITION_MODE == 'crisp':
        return get_destocked_to_ag_crisp(data, base_year, target_year, separate)
    else:
        raise ValueError(f"Unknown TRANSITION_MODE {settings.TRANSITION_MODE!r}; expected 'exact', 'blend', or 'crisp'.")




def get_crisp_irreversible_nonag_cell_mask(data: Data, base_year) -> np.ndarray | None:
    """
    Boolean (NCELLS,) mask of cells that already hold an irreversible non-ag LU.

    A cell is flagged if any irreversible non-ag LU (NON_AG_LAND_USES_REVERSIBLE is
    False) has a base-year dvar fraction >= FEASIBILITY_TOLERANCE. The same tolerance
    and ordering are used by get_non_ag_ub_matrices so the lock-in stays consistent.

    Returns None at the base year (or when no dvars exist), where no lock-in applies.
    """
    if base_year is None or base_year == data.YR_CAL_BASE or base_year not in data.non_ag_dvars:
        return None

    dvars_rk = data.non_ag_dvars[base_year]
    mask_r = np.zeros(data.NCELLS, dtype=bool)
    for k, k_name in enumerate(data.NON_AGRICULTURAL_LANDUSES):
        if not settings.NON_AG_LAND_USES_REVERSIBLE[k_name]:
            mask_r |= dvars_rk[:, k] >= settings.FEASIBILITY_TOLERANCE
    return mask_r


def get_transition_matrix_nonag2ag(data: Data, base_year: int, target_year: int, separate=False) -> dict:
    """Assemble non-ag→ag transition costs in flow format: dict[lu_name -> dict[k -> ndarray(NLMS, ncells_k, N_AG_LUS)]].

    Mode-independent: each per-lever `_to_ag` router already dispatches crisp/blend/exact and
    returns the same {k: ...} flow dict (keyed by the non-ag source k, dry source, target lm on the
    NLMS axis), so this just assembles them by non-ag source LU. (The old per-mode _crisp/_blend/_exact
    assemblers behind a second mode dispatch are gone — identical once every lever returned the
    uniform flow dict; the old blend `add.reduce` and crisp irreversible-cell mask are dropped, with
    irreversibility handled by the solver lb-lock.) EP/RP/CP-block/BECCS share the EP cost dict;
    input_data selects the per-source diagonal `dict[k]` for each lu_name.

    `agroforestry_x_r` / `cp_belt_x_r` (per-cell AF/CP-belt proportions) are forwarded for crisp
    mode; the blend/exact branches of those routers ignore them.
    """
    lumap            = data.lumaps[base_year]
    agroforestry_x_r = tools.get_exclusions_agroforestry_base(data, lumap)
    cp_belt_x_r      = tools.get_exclusions_carbon_plantings_belt_base(data, lumap)
    ep_to_ag         = get_env_plantings_to_ag(data, base_year, target_year, separate)
    return {
        'Environmental Plantings':       ep_to_ag,
        'Riparian Plantings':            ep_to_ag,
        'Sheep Agroforestry':            get_sheep_agroforestry_to_ag(data, base_year, target_year, agroforestry_x_r, separate),
        'Beef Agroforestry':             get_beef_agroforestry_to_ag(data, base_year, target_year, agroforestry_x_r, separate),
        'Carbon Plantings (Block)':      ep_to_ag,
        'Sheep Carbon Plantings (Belt)': get_sheep_carbon_plantings_belt_to_ag(data, base_year, target_year, cp_belt_x_r, separate),
        'Beef Carbon Plantings (Belt)':  get_beef_carbon_plantings_belt_to_ag(data, base_year, target_year, cp_belt_x_r, separate),
        'BECCS':                         ep_to_ag,
        'Destocked - natural land':      get_destocked_to_ag(data, base_year, target_year, separate),
    }


def get_nonag2nonag_transition_matrix(data: Data) -> np.ndarray:
    """
    Get the matrix that contains transition costs for non-agricultural land uses. 
    Currently, nonag is not allowed to transition to other nonag land uses, so the matrix is filled with zeros.
    
    Parameters
        data (object): The data object containing information about the model.
    
    Returns
        np.ndarray: The transition cost matrix, filled with zeros.
    """
    return np.zeros((data.NCELLS, data.N_NON_AG_LUS)).astype(np.float32)




def get_non_ag_ub_matrices_crisp(data: Data, lumap, base_dvar_nonag_rk, base_dvar_ag_mrj) -> np.ndarray:
    """
    CRISP / BLEND non-ag TARGET upper bound, shape (NCELLS, N_NON_AG_LUS). Each entry is the
    maximum cell proportion the solver may allocate to that non-ag LU. Transition eligibility
    (step 1) is decided from the **dominant** base-year LU per cell (`lumap`) — a 0/1 crisp test.
    Serves both crisp and blend (blend's exclusion is lumap-based too); the only mode-dependent
    piece is the Destocked cap (step 4), which is off for crisp and on for blend.

    Rules that set the UB:

    1. New allocations — transition matrix decide whether a cell is allowed to take
       on a non-ag LU for the first time. UB = 0 blocks the transition; UB = 1.

    2. No-go zones for non-ag→ag transitions — when EXCLUDE_NO_GO_LU is True, specified 
       no-go LU columns are set to UB=0 for all cells in the specified no-go regions. 
       This is applied after the above rules so it can only further restrict UB.

    3. Existing irreversible allocations — when base_dvar_nonag_rk is supplied,
       any cell that already holds an irreversible non-ag LU is forced to UB = 1,
       bypassing the transition-matrix check.  This guarantees the cell stays in
       feasible_non_ag_cells every solve so its lower bound (lock-in) remains active.
       Without this override, a T_MAT block in year Y would give dvar = 0, which
       propagates lb = 0 in year Y+1 and silently breaks irreversibility.

    4. Blended Destocked cap (base_dvar_ag_mrj supplied + TRANSITION_MODE != 'crisp') —
       the Destocked column is capped at (livestock-natural ag fraction + existing destocked
       fraction).  Existing destocked area can always remain destocked regardless of
       reversibility; reversibility only governs the lower bound (0 vs existing_dvar).
       
    5. RP_PROPORTION cap — the Riparian Plantings column is capped at the stream-buffer 
       fraction of each cell. This is applied last so it is the hard ceiling regardless 
       of what earlier steps set. Existing RP dvar ≤ RP_PROPORTION by construction, so 
       lb ≤ UB is always satisfied.
    """

    # 1. Existing T_MAT block gives initial UB=0/1 for new allocations based on from/to LU.  
    # This applies to all cells regardless of existing dvar.
    t_ik = data.T_MAT.sel(to_lu=data.NON_AGRICULTURAL_LANDUSES).copy()
    lumap2desc = np.vectorize(data.ALLLU2DESC.get, otypes=[str])
    ag_cells, non_ag_cells = tools.get_ag_and_non_ag_cells(lumap)

    t_rk = np.ones((data.NCELLS, len(data.NON_AGRICULTURAL_LANDUSES))).astype(np.float32)
    t_rk[ag_cells, :]     = t_ik[lumap[ag_cells]]
    t_rk[non_ag_cells, :] *= t_ik.sel(from_lu=lumap2desc(lumap[non_ag_cells]))
    t_rk[non_ag_cells, :] *= t_ik.sel(from_lu=lumap2desc(data.LUMAP[non_ag_cells]))
    t_rk = np.where(np.isnan(t_rk), 0, 1).astype(np.float32)

    # 2. No-go zones for non-ag→ag transitions set UB=0 for the affected non-ag LUs in those cells.  
    # This is applied to all cells regardless of existing dvar, but only affects the specified no-go LU columns.
    no_go_x_rk = np.ones((data.NCELLS, data.N_NON_AG_LUS))
    if settings.EXCLUDE_NO_GO_LU:
        for no_go_x_r, no_go_desc in zip(data.NO_GO_REGION_NON_AG, data.NO_GO_LANDUSE_NON_AG):
            no_go_j = data.NON_AGRICULTURAL_LANDUSES.index(no_go_desc)
            no_go_x_rk[:, no_go_j] = no_go_x_r
    t_rk = (t_rk * no_go_x_rk).astype(np.float32)

    # 3. Existing irreversible non-ag allocations override the above rules to ensure lock-in is 
    # never silently broken by a T_MAT block or no-go zone.
    for k, k_name in enumerate(data.NON_AGRICULTURAL_LANDUSES):
        if not settings.NON_AG_LAND_USES_REVERSIBLE[k_name]:
            t_rk[base_dvar_nonag_rk[:, k] >= settings.FEASIBILITY_TOLERANCE, k] = 1

    # 4. Blended Destocked cap: UB ≤ livestock-natural fraction + already-destocked fraction.
    #    Values below FEASIBILITY_TOLERANCE are rounding noise from prior solves and are
    #    zeroed to avoid creating near-degenerate destock variables.
    if settings.TRANSITION_MODE != 'crisp':
        destock_k = data.NON_AGRICULTURAL_LANDUSES.index('Destocked - natural land')
        eligible_frac_r = (
            base_dvar_ag_mrj[:, :, data.LU_LVSTK_NATURAL].sum(axis=(0, 2))
            + base_dvar_nonag_rk[:, destock_k]
        ).astype(np.float32)
        # Exclude rounding-noise values so cells with no real livestock-natural or
        # destocked area are treated as ineligible (same threshold as step 3 lb override).
        eligible_frac_r[eligible_frac_r < settings.FEASIBILITY_TOLERANCE] = 0.0
        t_rk[:, destock_k] *= eligible_frac_r

    # 5. Riparian Plantings is physically bounded by the stream-buffer fraction of each cell.
    # Applied last so it is the hard ceiling regardless of what earlier steps set.
    # Existing RP dvar <= RP_PROPORTION by construction, so lb <= UB is always satisfied.
    RP_j = data.NON_AGRICULTURAL_LANDUSES.index('Riparian Plantings')
    t_rk[:, RP_j] *= data.RP_PROPORTION

    return t_rk


def get_non_ag_ub_matrices_exact(data: Data, base_dvar_nonag_rk, base_dvar_ag_mrj) -> np.ndarray:
    """
    EXACT non-ag TARGET upper bound, shape (NCELLS, N_NON_AG_LUS), FRACTIONAL.

    Same five rules as the crisp version, but step 1 (transition eligibility) is the fractional
    reachable share of the cell instead of a 0/1 test on the dominant LU — mirroring
    get_ag2ag_ub_exact / get_nonag2ag_ub_exact. For target non-ag LU k at cell r:

        reach[r,k] = Σ_{ag from_j : T_MAT[from_j→k] finite} Σ_m frac_ag[m,r,from_j]   (ag2nonag)
                   + Σ_{nonag k'  : T_MAT[k'→k]      finite}      frac_nonag[r,k']      (nonag2nonag)

    i.e. the proportion of the cell currently held by *any* source LU that may legally become k.
    Bounded by 1 because per-cell ag + non-ag fractions sum to ≤ 1. Steps 2–5 (no-go, irreversible
    lock-in override, Destocked cap, RP cap) are identical in intent; the two physical caps use
    np.minimum (the exact generalisation of the crisp 0/1 × cap, to which it reduces).
    """
    # 1. Transition exclusion (T_MAT), FRACTIONAL: reachable land share per non-ag target.
    t_jk = (~np.isnan(
        data.T_MAT.sel(from_lu=data.AGRICULTURAL_LANDUSES, to_lu=data.NON_AGRICULTURAL_LANDUSES).values
    )).astype(np.float32)                                                                # (ag_from_j, nonag_k)
    t_kk = (~np.isnan(
        data.T_MAT.sel(from_lu=data.NON_AGRICULTURAL_LANDUSES, to_lu=data.NON_AGRICULTURAL_LANDUSES).values
    )).astype(np.float32)                                                                # (nonag_from_k, nonag_k)
    ag_frac_rj = base_dvar_ag_mrj.sum(axis=0)                                            # (NCELLS, ag_j)
    t_rk = (ag_frac_rj @ t_jk + base_dvar_nonag_rk @ t_kk).astype(np.float32)            # (NCELLS, nonag_k)

    # 2. No-go zones for non-ag LUs set UB=0 for the affected columns in the no-go regions.
    if settings.EXCLUDE_NO_GO_LU:
        for no_go_x_r, no_go_desc in zip(data.NO_GO_REGION_NON_AG, data.NO_GO_LANDUSE_NON_AG):
            no_go_j = data.NON_AGRICULTURAL_LANDUSES.index(no_go_desc)
            t_rk[:, no_go_j] *= no_go_x_r

    # 3. Existing irreversible non-ag allocations forced to UB=1 so their lock-in (lb) survives.
    for k, k_name in enumerate(data.NON_AGRICULTURAL_LANDUSES):
        if not settings.NON_AG_LAND_USES_REVERSIBLE[k_name]:
            t_rk[base_dvar_nonag_rk[:, k] >= settings.FEASIBILITY_TOLERANCE, k] = 1

    # 4. Destocked physical cap: UB ≤ livestock-natural ag fraction + already-destocked fraction.
    #    (min, not ×, because the exact reach is already fractional.)
    destock_k = data.NON_AGRICULTURAL_LANDUSES.index('Destocked - natural land')
    eligible_frac_r = (
        base_dvar_ag_mrj[:, :, data.LU_LVSTK_NATURAL].sum(axis=(0, 2))
        + base_dvar_nonag_rk[:, destock_k]
    ).astype(np.float32)
    eligible_frac_r[eligible_frac_r < settings.FEASIBILITY_TOLERANCE] = 0.0
    t_rk[:, destock_k] = np.minimum(t_rk[:, destock_k], eligible_frac_r)

    # 5. Riparian Plantings physical cap: UB ≤ stream-buffer fraction of the cell.
    RP_j = data.NON_AGRICULTURAL_LANDUSES.index('Riparian Plantings')
    t_rk[:, RP_j] = np.minimum(t_rk[:, RP_j], data.RP_PROPORTION)

    return t_rk.astype(np.float32)


def get_non_ag_ub_matrices(data: Data, lumap, base_dvar_nonag_rk, base_dvar_ag_mrj) -> np.ndarray:
    """Dispatch the non-ag TARGET upper bound by TRANSITION_MODE.

    crisp / blend: 0/1 transition test on the dominant base LU (get_non_ag_ub_matrices_crisp).
    exact:         fractional reachable-land share (get_non_ag_ub_matrices_exact).
    """
    if settings.TRANSITION_MODE in ('crisp', 'blend'):
        return get_non_ag_ub_matrices_crisp(data, lumap, base_dvar_nonag_rk, base_dvar_ag_mrj)
    elif settings.TRANSITION_MODE == 'exact':
        return get_non_ag_ub_matrices_exact(data, base_dvar_nonag_rk, base_dvar_ag_mrj)
    else:
        raise ValueError(f"Unknown TRANSITION_MODE: {settings.TRANSITION_MODE!r}")


def get_non_ag_lb_matrices(data: Data, base_year) -> np.ndarray:
    """
    Returns the lower-bound (LB) matrix for non-agricultural land uses,
    shape (NCELLS, N_NON_AG_LUS).  Each entry is the minimum cell proportion
    the solver must maintain — i.e. the lock-in floor for irreversible LUs.

    At the base year (or before any allocation exists) the matrix is all zeros.
    For subsequent years, the LB for each irreversible LU is set to the
    floor-truncated dvar value from the previous period, preventing the solver
    from reducing already-committed irreversible land.

    This function has no knowledge of the transition matrix or UB.  The pairing
    with get_non_ag_ub_matrices (which forces UB=1 for existing irreversible
    cells) ensures Gurobi always receives a valid lb <= ub.
    """

    if base_year == data.YR_CAL_BASE or base_year not in data.non_ag_dvars:
        return np.zeros((data.NCELLS, len(settings.NON_AG_LAND_USES))).astype(np.float32)

    prev_dvars = data.non_ag_dvars[base_year].astype(np.float32)
    scale = 10 ** settings.ROUND_DECIMALS

    # Only lock in non-reversible LUs; reversible LUs keep lb = 0.
    lb_rk = np.zeros_like(prev_dvars)
    for k, k_name in enumerate(data.NON_AGRICULTURAL_LANDUSES):
        if not settings.NON_AG_LAND_USES_REVERSIBLE[k_name]:
            # Floor-truncate to prevent float32 upward rounding from inflating lb
            lb_rk[:, k] = np.floor(prev_dvars[:, k] * scale) / scale

    # NonAg lb can not exceed the available ag-mask proportion of the cell
    lb_capped = np.minimum(lb_rk, data.AG_MASK_PROPORTION_R[:, np.newaxis]).astype(np.float32)

    lb_update = lb_capped < lb_rk
    if lb_update.any():
        gap = lb_rk[lb_update] - lb_capped[lb_update]
        print(
            f"  └── NonAg lb capped: {lb_update.sum()} cells updated,"
            f" max gap={gap.max():.2e}, mean gap={gap.mean():.2e}"
        )

    return lb_capped
