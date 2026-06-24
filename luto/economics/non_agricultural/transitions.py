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

from luto import settings
from luto.data import Data
import luto.tools as tools
import luto.economics.agricultural.water as ag_water
import luto.economics.agricultural.ghg as ag_ghg
import luto.economics.agricultural.transitions as ag_transitions


def get_env_plant_transitions_from_ag(data: Data, yr_idx, lumap, lmmap, separate=False) -> np.ndarray|dict:
    """
    Calculate the transition costs for transitioning from agricultural land to environmental plantings.

    Args:
        data (object): The data object containing relevant information.
        yr_idx (int): The index of the year.
        lumap (np.ndarray): The land use map.
        lmmap (np.ndarray): The water supply map.
        separate (bool, optional): Whether to return separate costs or the total cost. Defaults to False.

    Returns
        np.ndarray|dict: The transition costs as either a numpy array or a dictionary, depending on the value of `separate`.
    """
    yr_cal = data.YR_CAL_BASE + yr_idx
    cells = np.isin(lumap, np.array(list(data.AGLU2DESC.keys())))

    # Establishment costs
    est_costs_r = tools.amortise(data.EP_EST_COST_HA * data.REAL_AREA * data.EST_COST_MULTS[yr_cal]).astype(np.float32)
    est_costs_r[~cells] = 0.0
    
    # Water costs; Assume EP is dryland
    w_rm_irrig_cost_r = np.where(lmmap == 1, settings.REMOVE_IRRIG_COST * data.IRRIG_COST_MULTS[yr_cal], 0) * data.REAL_AREA
    w_rm_irrig_cost_r[~cells] = 0.0
    
    # Transition costs
    ag_to_ep_j = data.T_MAT.sel(from_lu=data.AGRICULTURAL_LANDUSES, to_lu='Environmental Plantings').values
    ag_to_ep_t_r = np.vectorize(dict(enumerate(ag_to_ep_j)).get, otypes=['float32'])(lumap)
    ag_to_ep_t_r = np.nan_to_num(ag_to_ep_t_r)
    ag_to_ep_t_r = tools.amortise(ag_to_ep_t_r * data.REAL_AREA)
    ag_to_ep_t_r[~cells] = 0.0
    
    if separate:
        return {
            'Establishment cost (Ag2Non-Ag)': est_costs_r,
            'Transition cost (Ag2Non-Ag)': ag_to_ep_t_r,
            'Remove irrigation cost (Ag2Non-Ag)': w_rm_irrig_cost_r
        }
    else:
        return est_costs_r + ag_to_ep_t_r + w_rm_irrig_cost_r


def get_env_plant_transitions_from_ag_blended(data: Data, yr_idx, base_year, separate=False) -> np.ndarray | dict:
    """Blended EP transition costs: weighted-average T_MAT over source LUs, normalised to whole-cell cost."""
    yr_cal    = data.YR_CAL_BASE + yr_idx
    dvars     = data.ag_dvars[base_year]
    dvar_base = tools.ag_mrj_to_xr(data, dvars, threshold=0)

    threshold      = 10 ** (-settings.ROUND_DECIMALS)
    ag_frac_r      = dvar_base.sum(['lm', 'lu']).values
    ag_frac_r_safe = np.where(ag_frac_r > threshold, ag_frac_r, 1.0)
    irr_frac_r     = dvar_base.sel(lm='irr').sum('lu').values

    # Establishment cost — full whole-cell rate; solver's X_nonag handles fraction
    est_costs_r = tools.amortise(data.EP_EST_COST_HA * data.REAL_AREA * data.EST_COST_MULTS[yr_cal]).astype(np.float32)

    # WATER — proportional to irrigated fraction of cell; dvar handles actual allocation
    w_rm_irrig_cost_r = (irr_frac_r * settings.REMOVE_IRRIG_COST * data.IRRIG_COST_MULTS[yr_cal] * data.REAL_AREA).astype(np.float32)

    # T_MAT — normalised weighted average over source LUs
    t_r = np.zeros(data.NCELLS, dtype=np.float32)
    for j_idx, j_name in enumerate(data.AGRICULTURAL_LANDUSES):
        trans_cost = data.T_MAT.sel(from_lu=j_name, to_lu='Environmental Plantings').item()
        if np.isnan(trans_cost):
            continue
        t_r += trans_cost * dvars[:, :, j_idx].sum(axis=0)
    ag_to_ep_t_r = tools.amortise((t_r / ag_frac_r_safe) * data.REAL_AREA).astype(np.float32)

    if separate:
        return {
            'Establishment cost (Ag2Non-Ag)': est_costs_r,
            'Transition cost (Ag2Non-Ag)': ag_to_ep_t_r,
            'Remove irrigation cost (Ag2Non-Ag)': w_rm_irrig_cost_r
        }
    else:
        return est_costs_r + ag_to_ep_t_r + w_rm_irrig_cost_r


def get_env_plant_transitions_from_ag_from_base_year(
    data: Data, yr_idx, base_year, lumap, lmmap, separate=False
) -> np.ndarray | dict:
    """Dispatcher: blended (RESFACTOR>1 mixed cells) or crisp-lumap EP transition costs."""
    if settings.BLENDED_TRANSITION_COSTS:
        return get_env_plant_transitions_from_ag_blended(data, yr_idx, base_year, separate)
    else:
        return get_env_plant_transitions_from_ag(data, yr_idx, lumap, lmmap, separate)


def get_rip_plant_transitions_from_ag(data: Data, yr_idx, lumap, lmmap, separate=False) -> np.ndarray|dict:
    """
    Get transition costs from agricultural land uses to riparian plantings for each cell.

    Returns
    -------
    np.ndarray
        1-D array, indexed by cell.
    """
    yr_cal = data.YR_CAL_BASE + yr_idx
    cells = np.isin(lumap, np.array(list(data.AGLU2DESC.keys())))

    # Establishment costs
    est_costs_r = tools.amortise(data.RP_EST_COST_HA * data.REAL_AREA * data.EST_COST_MULTS[yr_cal]).astype(np.float32)
    est_costs_r[~cells] = 0.0
    # est_costs_r *= data.RP_PROPORTION
    
    # Transition costs
    ag_to_ep_j = data.T_MAT.sel(from_lu=data.AGRICULTURAL_LANDUSES, to_lu='Riparian Plantings').values
    ag_to_ep_t_r = np.vectorize(dict(enumerate(ag_to_ep_j)).get, otypes=['float32'])(lumap)
    ag_to_ep_t_r = np.nan_to_num(ag_to_ep_t_r)
    ag_to_ep_t_r = tools.amortise(ag_to_ep_t_r * data.REAL_AREA)
    ag_to_ep_t_r[~cells] = 0.0
    # ag_to_ep_t_r *= data.RP_PROPORTION
    
    
    # Water costs; Assume riparian plantings are dryland
    w_rm_irrig_cost_r = np.where(lmmap == 1, settings.REMOVE_IRRIG_COST * data.IRRIG_COST_MULTS[yr_cal], 0) * data.REAL_AREA
    w_rm_irrig_cost_r[~cells] = 0.0

    # Fencing costs
    fencing_cost_r = (
        data.RP_FENCING_LENGTH 
        * settings.FENCING_COST_PER_M
        * data.FENCE_COST_MULTS[yr_cal]
        * data.REAL_AREA
    ).astype(np.float32)
    fencing_cost_r[~cells] = 0.0
    # fencing_cost_r *= data.RP_PROPORTION
    
    
    if separate:
        return {
            'Establishment cost (Ag2Non-Ag)': est_costs_r,
            'Transition cost (Ag2Non-Ag)': ag_to_ep_t_r,
            'Remove irrigation cost (Ag2Non-Ag)': w_rm_irrig_cost_r,
            'Fencing cost (Ag2Non-Ag)': fencing_cost_r
        }
    else:
        return est_costs_r + ag_to_ep_t_r + w_rm_irrig_cost_r + fencing_cost_r


def get_rip_plant_transitions_from_ag_blended(data: Data, yr_idx, base_year, separate=False) -> np.ndarray | dict:
    """Blended RP transition costs: weighted-average T_MAT over source LUs, normalised to whole-cell cost."""
    yr_cal    = data.YR_CAL_BASE + yr_idx
    dvars     = data.ag_dvars[base_year]
    dvar_base = tools.ag_mrj_to_xr(data, dvars, threshold=0)

    threshold      = 10 ** (-settings.ROUND_DECIMALS)
    ag_frac_r      = dvar_base.sum(['lm', 'lu']).values
    ag_frac_r_safe = np.where(ag_frac_r > threshold, ag_frac_r, 1.0)
    irr_frac_r     = dvar_base.sel(lm='irr').sum('lu').values

    # EST — full whole-cell rate
    est_costs_r = tools.amortise(data.RP_EST_COST_HA * data.REAL_AREA * data.EST_COST_MULTS[yr_cal]).astype(np.float32)

    # T_MAT — normalised weighted average
    t_r = np.zeros(data.NCELLS, dtype=np.float32)
    for j_idx, j_name in enumerate(data.AGRICULTURAL_LANDUSES):
        trans_cost = data.T_MAT.sel(from_lu=j_name, to_lu='Riparian Plantings').item()
        if np.isnan(trans_cost):
            continue
        t_r += trans_cost * dvars[:, :, j_idx].sum(axis=0)
    ag_to_ep_t_r = tools.amortise((t_r / ag_frac_r_safe) * data.REAL_AREA).astype(np.float32)

    # WATER — expected irrigated fraction among the ag pool
    w_rm_irrig_cost_r = (irr_frac_r * settings.REMOVE_IRRIG_COST * data.IRRIG_COST_MULTS[yr_cal] * data.REAL_AREA).astype(np.float32)

    # FENCING — full whole-cell rate
    fencing_cost_r = (data.RP_FENCING_LENGTH * settings.FENCING_COST_PER_M * data.FENCE_COST_MULTS[yr_cal] * data.REAL_AREA).astype(np.float32)

    if separate:
        return {
            'Establishment cost (Ag2Non-Ag)': est_costs_r,
            'Transition cost (Ag2Non-Ag)': ag_to_ep_t_r,
            'Remove irrigation cost (Ag2Non-Ag)': w_rm_irrig_cost_r,
            'Fencing cost (Ag2Non-Ag)': fencing_cost_r
        }
    else:
        return est_costs_r + ag_to_ep_t_r + w_rm_irrig_cost_r + fencing_cost_r


def get_rip_plant_transitions_from_ag_from_base_year(
    data: Data, yr_idx, base_year, lumap, lmmap, separate=False
) -> np.ndarray | dict:
    """Dispatcher: blended or crisp-lumap RP transition costs."""
    if settings.BLENDED_TRANSITION_COSTS:
        return get_rip_plant_transitions_from_ag_blended(data, yr_idx, base_year, separate)
    else:
        return get_rip_plant_transitions_from_ag(data, yr_idx, lumap, lmmap, separate)


def get_sheep_agroforestry_transitions_from_ag(
    data: Data, yr_idx, lumap, lmmap, separate=False
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
    
    yr_cal = data.YR_CAL_BASE + yr_idx
    cells = np.isin(lumap, np.array(list(data.AGLU2DESC.keys())))

    # Establishment costs
    est_costs_r = tools.amortise(data.AF_EST_COST_HA * data.REAL_AREA * data.EST_COST_MULTS[yr_cal]).astype(np.float32)
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
    w_rm_irrig_cost_r = np.where(lmmap == 1, settings.REMOVE_IRRIG_COST * data.IRRIG_COST_MULTS[yr_cal], 0) * data.REAL_AREA
    w_rm_irrig_cost_r[~cells] = 0.0

    # Fencing costs
    fencing_cost_r = (
        settings.AF_FENCING_LENGTH_HA
        * settings.FENCING_COST_PER_M
        * data.FENCE_COST_MULTS[yr_cal]
        * data.REAL_AREA 
    ).astype(np.float32)
    fencing_cost_r[~cells] = 0.0
    
    if separate:
        return {
            'Establishment cost (Ag2Non-Ag)': est_costs_r,
            'Transition cost (Ag2Non-Ag)': ag_to_agroforestry_t_r,
            'Transition cost (Ag2AF-Sheep)': ag_to_sheep_t_r,
            'Remove irrigation cost (Ag2Non-Ag)': w_rm_irrig_cost_r,
            'Fencing cost (Ag2Non-Ag)': fencing_cost_r
        }
    else:
        return est_costs_r + ag_to_agroforestry_t_r + ag_to_sheep_t_r + w_rm_irrig_cost_r + fencing_cost_r


def get_sheep_agroforestry_transitions_from_ag_blended(data: Data, yr_idx, base_year, separate=False) -> np.ndarray | dict:
    """Blended Sheep AF transition costs: normalised weighted-average T_MAT, whole-cell EST/FENCING."""
    yr_cal    = data.YR_CAL_BASE + yr_idx
    dvars     = data.ag_dvars[base_year]
    dvar_base = tools.ag_mrj_to_xr(data, dvars, threshold=0)

    threshold      = 10 ** (-settings.ROUND_DECIMALS)
    ag_frac_r      = dvar_base.sum(['lm', 'lu']).values
    ag_frac_r_safe = np.where(ag_frac_r > threshold, ag_frac_r, 1.0)
    irr_frac_r     = dvar_base.sel(lm='irr').sum('lu').values

    # EST — full whole-cell rate × AF_PROPORTION
    est_costs_r = (tools.amortise(data.AF_EST_COST_HA * data.REAL_AREA * data.EST_COST_MULTS[yr_cal]) * settings.AF_PROPORTION).astype(np.float32)

    # T_MAT (AF) — normalised weighted average × AF_PROPORTION
    t_af_r = np.zeros(data.NCELLS, dtype=np.float32)
    for j_idx, j_name in enumerate(data.AGRICULTURAL_LANDUSES):
        trans_cost = data.T_MAT.sel(from_lu=j_name, to_lu='Sheep Agroforestry').item()
        if np.isnan(trans_cost):
            continue
        t_af_r += trans_cost * dvars[:, :, j_idx].sum(axis=0)
    ag_to_agroforestry_t_r = (tools.amortise((t_af_r / ag_frac_r_safe) * data.REAL_AREA) * settings.AF_PROPORTION).astype(np.float32)

    # T_MAT (Sheep) — normalised weighted average × (1-AF_PROPORTION)
    t_sheep_r = np.zeros(data.NCELLS, dtype=np.float32)
    for j_idx, j_name in enumerate(data.AGRICULTURAL_LANDUSES):
        trans_cost = data.T_MAT.sel(from_lu=j_name, to_lu='Sheep - modified land').item()
        if np.isnan(trans_cost):
            continue
        t_sheep_r += trans_cost * dvars[:, :, j_idx].sum(axis=0)
    ag_to_sheep_t_r = (tools.amortise((t_sheep_r / ag_frac_r_safe) * data.REAL_AREA) * (1 - settings.AF_PROPORTION)).astype(np.float32)

    # WATER — expected irrigated fraction among the ag pool
    w_rm_irrig_cost_r = (irr_frac_r * settings.REMOVE_IRRIG_COST * data.IRRIG_COST_MULTS[yr_cal] * data.REAL_AREA).astype(np.float32)

    # FENCING — full whole-cell rate
    fencing_cost_r = (settings.AF_FENCING_LENGTH_HA * settings.FENCING_COST_PER_M * data.FENCE_COST_MULTS[yr_cal] * data.REAL_AREA).astype(np.float32)

    if separate:
        return {
            'Establishment cost (Ag2Non-Ag)': est_costs_r,
            'Transition cost (Ag2Non-Ag)': ag_to_agroforestry_t_r,
            'Transition cost (Ag2AF-Sheep)': ag_to_sheep_t_r,
            'Remove irrigation cost (Ag2Non-Ag)': w_rm_irrig_cost_r,
            'Fencing cost (Ag2Non-Ag)': fencing_cost_r
        }
    else:
        return est_costs_r + ag_to_agroforestry_t_r + ag_to_sheep_t_r + w_rm_irrig_cost_r + fencing_cost_r


def get_sheep_agroforestry_transitions_from_ag_from_base_year(
    data: Data, yr_idx, base_year, lumap, lmmap, separate=False
) -> np.ndarray | dict:
    """Dispatcher: blended or crisp-lumap Sheep AF transition costs."""
    if settings.BLENDED_TRANSITION_COSTS:
        return get_sheep_agroforestry_transitions_from_ag_blended(data, yr_idx, base_year, separate)
    else:
        return get_sheep_agroforestry_transitions_from_ag(data, yr_idx, lumap, lmmap, separate)


def get_beef_agroforestry_transitions_from_ag(
    data: Data, yr_idx, lumap, lmmap, separate=False
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
    
    yr_cal = data.YR_CAL_BASE + yr_idx
    cells = np.isin(lumap, np.array(list(data.AGLU2DESC.keys())))

    # Establishment costs
    est_costs_r = tools.amortise(data.AF_EST_COST_HA * data.REAL_AREA * data.EST_COST_MULTS[yr_cal]).astype(np.float32)
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
    w_rm_irrig_cost_r = np.where(lmmap == 1, settings.REMOVE_IRRIG_COST * data.IRRIG_COST_MULTS[yr_cal], 0) * data.REAL_AREA
    w_rm_irrig_cost_r[~cells] = 0.0
    
    # Fencing costs
    fencing_cost_r = (
        settings.AF_FENCING_LENGTH_HA
        * settings.FENCING_COST_PER_M
        * data.FENCE_COST_MULTS[data.YR_CAL_BASE + yr_idx]
        * data.REAL_AREA
    ).astype(np.float32)
    fencing_cost_r[~cells] = 0.0
    
    if separate:
        return {
            'Establishment cost (Ag2Non-Ag)': est_costs_r,
            'Transition cost (Ag2Non-Ag)': ag_to_agroforestry_t_r,
            'Transition cost (Ag2AF-Beef)': ag_to_beef_t_r,
            'Remove irrigation cost (Ag2Non-Ag)': w_rm_irrig_cost_r,
            'Fencing cost (Ag2Non-Ag)': fencing_cost_r
        }
    else:
        return est_costs_r + ag_to_agroforestry_t_r + ag_to_beef_t_r + w_rm_irrig_cost_r + fencing_cost_r


def get_beef_agroforestry_transitions_from_ag_blended(data: Data, yr_idx, base_year, separate=False) -> np.ndarray | dict:
    """Blended Beef AF transition costs: normalised weighted-average T_MAT, whole-cell EST/FENCING."""
    yr_cal    = data.YR_CAL_BASE + yr_idx
    dvars     = data.ag_dvars[base_year]
    dvar_base = tools.ag_mrj_to_xr(data, dvars, threshold=0)

    threshold      = 10 ** (-settings.ROUND_DECIMALS)
    ag_frac_r      = dvar_base.sum(['lm', 'lu']).values
    ag_frac_r_safe = np.where(ag_frac_r > threshold, ag_frac_r, 1.0)
    irr_frac_r     = dvar_base.sel(lm='irr').sum('lu').values

    # EST — full whole-cell rate × AF_PROPORTION
    est_costs_r = (tools.amortise(data.AF_EST_COST_HA * data.REAL_AREA * data.EST_COST_MULTS[yr_cal]) * settings.AF_PROPORTION).astype(np.float32)

    # T_MAT (AF) — normalised weighted average × AF_PROPORTION
    t_af_r = np.zeros(data.NCELLS, dtype=np.float32)
    for j_idx, j_name in enumerate(data.AGRICULTURAL_LANDUSES):
        trans_cost = data.T_MAT.sel(from_lu=j_name, to_lu='Beef Agroforestry').item()
        if np.isnan(trans_cost):
            continue
        t_af_r += trans_cost * dvars[:, :, j_idx].sum(axis=0)
    ag_to_agroforestry_t_r = (tools.amortise((t_af_r / ag_frac_r_safe) * data.REAL_AREA) * settings.AF_PROPORTION).astype(np.float32)

    # T_MAT (Beef) — normalised weighted average × (1-AF_PROPORTION)
    t_beef_r = np.zeros(data.NCELLS, dtype=np.float32)
    for j_idx, j_name in enumerate(data.AGRICULTURAL_LANDUSES):
        trans_cost = data.T_MAT.sel(from_lu=j_name, to_lu='Beef - modified land').item()
        if np.isnan(trans_cost):
            continue
        t_beef_r += trans_cost * dvars[:, :, j_idx].sum(axis=0)
    ag_to_beef_t_r = (tools.amortise((t_beef_r / ag_frac_r_safe) * data.REAL_AREA) * (1 - settings.AF_PROPORTION)).astype(np.float32)

    # WATER — expected irrigated fraction among the ag pool
    w_rm_irrig_cost_r = (irr_frac_r * settings.REMOVE_IRRIG_COST * data.IRRIG_COST_MULTS[yr_cal] * data.REAL_AREA).astype(np.float32)

    # FENCING — full whole-cell rate
    fencing_cost_r = (settings.AF_FENCING_LENGTH_HA * settings.FENCING_COST_PER_M * data.FENCE_COST_MULTS[yr_cal] * data.REAL_AREA).astype(np.float32)

    if separate:
        return {
            'Establishment cost (Ag2Non-Ag)': est_costs_r,
            'Transition cost (Ag2Non-Ag)': ag_to_agroforestry_t_r,
            'Transition cost (Ag2AF-Beef)': ag_to_beef_t_r,
            'Remove irrigation cost (Ag2Non-Ag)': w_rm_irrig_cost_r,
            'Fencing cost (Ag2Non-Ag)': fencing_cost_r
        }
    else:
        return est_costs_r + ag_to_agroforestry_t_r + ag_to_beef_t_r + w_rm_irrig_cost_r + fencing_cost_r


def get_beef_agroforestry_transitions_from_ag_from_base_year(
    data: Data, yr_idx, base_year, lumap, lmmap, separate=False
) -> np.ndarray | dict:
    """Dispatcher: blended or crisp-lumap Beef AF transition costs."""
    if settings.BLENDED_TRANSITION_COSTS:
        return get_beef_agroforestry_transitions_from_ag_blended(data, yr_idx, base_year, separate)
    else:
        return get_beef_agroforestry_transitions_from_ag(data, yr_idx, lumap, lmmap, separate)


def get_carbon_plantings_block_from_ag(data: Data, yr_idx, lumap, lmmap, separate=False) -> np.ndarray|dict:
    """
    Get transition costs from agricultural land uses to carbon plantings (block) for each cell.

    Returns
    -------
    np.ndarray
        1-D array, indexed by cell.
    """
    yr_cal = data.YR_CAL_BASE + yr_idx
    cells = np.isin(lumap, np.array(list(data.AGLU2DESC.keys())))

    # Establishment costs
    est_costs_CP_r = tools.amortise(data.CP_EST_COST_HA * data.REAL_AREA * data.EST_COST_MULTS[yr_cal]).astype(np.float32)
    est_costs_CP_r[~cells] = 0.0

    # Transition costs
    ag_to_cp_j = data.T_MAT.sel(from_lu=data.AGRICULTURAL_LANDUSES, to_lu='Carbon Plantings (Block)').values
    ag_to_cp_t_r = np.vectorize(dict(enumerate(ag_to_cp_j)).get, otypes=['float32'])(lumap)
    ag_to_cp_t_r = np.nan_to_num(ag_to_cp_t_r)
    ag_to_cp_t_r = tools.amortise(ag_to_cp_t_r * data.REAL_AREA)
    ag_to_cp_t_r[~cells] = 0.0

    # Water costs; Assume CP is dryland
    w_rm_irrig_cost_r = np.where(lmmap == 1, settings.REMOVE_IRRIG_COST * data.IRRIG_COST_MULTS[yr_cal], 0) * data.REAL_AREA
    w_rm_irrig_cost_r[~cells] = 0.0

    if separate:
        return {
            'Establishment cost (Ag2Non-Ag)': est_costs_CP_r,
            'Transition cost (Ag2Non-Ag)': ag_to_cp_t_r,
            'Remove irrigation cost (Ag2Non-Ag)': w_rm_irrig_cost_r
        }
    else:
        return est_costs_CP_r + ag_to_cp_t_r + w_rm_irrig_cost_r


def get_carbon_plantings_block_from_ag_blended(data: Data, yr_idx, base_year, separate=False) -> np.ndarray | dict:
    """Blended CP Block transition costs: normalised weighted-average T_MAT, whole-cell EST."""
    yr_cal    = data.YR_CAL_BASE + yr_idx
    dvars     = data.ag_dvars[base_year]
    dvar_base = tools.ag_mrj_to_xr(data, dvars, threshold=0)

    threshold      = 10 ** (-settings.ROUND_DECIMALS)
    ag_frac_r      = dvar_base.sum(['lm', 'lu']).values
    ag_frac_r_safe = np.where(ag_frac_r > threshold, ag_frac_r, 1.0)
    irr_frac_r     = dvar_base.sel(lm='irr').sum('lu').values

    # EST — full whole-cell rate
    est_costs_CP_r = tools.amortise(data.CP_EST_COST_HA * data.REAL_AREA * data.EST_COST_MULTS[yr_cal]).astype(np.float32)

    # T_MAT — normalised weighted average
    t_r = np.zeros(data.NCELLS, dtype=np.float32)
    for j_idx, j_name in enumerate(data.AGRICULTURAL_LANDUSES):
        trans_cost = data.T_MAT.sel(from_lu=j_name, to_lu='Carbon Plantings (Block)').item()
        if np.isnan(trans_cost):
            continue
        t_r += trans_cost * dvars[:, :, j_idx].sum(axis=0)
    ag_to_cp_t_r = tools.amortise((t_r / ag_frac_r_safe) * data.REAL_AREA).astype(np.float32)

    # WATER — expected irrigated fraction among the ag pool
    w_rm_irrig_cost_r = (irr_frac_r * settings.REMOVE_IRRIG_COST * data.IRRIG_COST_MULTS[yr_cal] * data.REAL_AREA).astype(np.float32)

    if separate:
        return {
            'Establishment cost (Ag2Non-Ag)': est_costs_CP_r,
            'Transition cost (Ag2Non-Ag)': ag_to_cp_t_r,
            'Remove irrigation cost (Ag2Non-Ag)': w_rm_irrig_cost_r
        }
    else:
        return est_costs_CP_r + ag_to_cp_t_r + w_rm_irrig_cost_r


def get_carbon_plantings_block_from_ag_from_base_year(
    data: Data, yr_idx, base_year, lumap, lmmap, separate=False
) -> np.ndarray | dict:
    """Dispatcher: blended or crisp-lumap CP Block transition costs."""
    if settings.BLENDED_TRANSITION_COSTS:
        return get_carbon_plantings_block_from_ag_blended(data, yr_idx, base_year, separate)
    else:
        return get_carbon_plantings_block_from_ag(data, yr_idx, lumap, lmmap, separate)


def get_sheep_carbon_plantings_belt_from_ag(
    data: Data, yr_idx, lumap, lmmap, separate=False
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
    
    yr_cal = data.YR_CAL_BASE + yr_idx
    cells = np.isin(lumap, np.array(list(data.AGLU2DESC.keys())))

    # Establishment costs
    est_costs_CP_r = tools.amortise(data.CP_EST_COST_HA * data.REAL_AREA * data.EST_COST_MULTS[yr_cal]).astype(np.float32)
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
    w_rm_irrig_cost_r = np.where(lmmap == 1, settings.REMOVE_IRRIG_COST * data.IRRIG_COST_MULTS[yr_cal], 0) * data.REAL_AREA
    w_rm_irrig_cost_r[~cells] = 0.0

    fencing_cost_r = (
        settings.CP_BELT_FENCING_LENGTH
        * settings.FENCING_COST_PER_M
        * data.FENCE_COST_MULTS[data.YR_CAL_BASE + yr_idx]
        * data.REAL_AREA
    ).astype(np.float32)
    fencing_cost_r[~cells] = 0.0

    if separate:
        return {
            'Establishment cost (Ag2Non-Ag)': est_costs_CP_r,
            'Transition cost (Ag2Non-Ag)': ag_to_cp_t_r,
            'Transition cost (Ag2CP-Sheep)': ag_to_sheep_t_r,
            'Remove irrigation cost (Ag2Non-Ag)': w_rm_irrig_cost_r,
            'Fencing cost (Ag2Non-Ag)': fencing_cost_r
        }
    else:
        return est_costs_CP_r + ag_to_cp_t_r + ag_to_sheep_t_r + w_rm_irrig_cost_r + fencing_cost_r


def get_sheep_carbon_plantings_belt_from_ag_blended(data: Data, yr_idx, base_year, separate=False) -> np.ndarray | dict:
    """Blended Sheep CP Belt transition costs: normalised weighted-average T_MAT, whole-cell EST/FENCING."""
    yr_cal    = data.YR_CAL_BASE + yr_idx
    dvars     = data.ag_dvars[base_year]
    dvar_base = tools.ag_mrj_to_xr(data, dvars, threshold=0)

    threshold      = 10 ** (-settings.ROUND_DECIMALS)
    ag_frac_r      = dvar_base.sum(['lm', 'lu']).values
    ag_frac_r_safe = np.where(ag_frac_r > threshold, ag_frac_r, 1.0)
    irr_frac_r     = dvar_base.sel(lm='irr').sum('lu').values

    # EST — full whole-cell rate × CP_BELT_PROPORTION
    est_costs_CP_r = (tools.amortise(data.CP_EST_COST_HA * data.REAL_AREA * data.EST_COST_MULTS[yr_cal]) * settings.CP_BELT_PROPORTION).astype(np.float32)

    # T_MAT (CP Belt) — normalised weighted average × CP_BELT_PROPORTION
    t_cp_r = np.zeros(data.NCELLS, dtype=np.float32)
    for j_idx, j_name in enumerate(data.AGRICULTURAL_LANDUSES):
        trans_cost = data.T_MAT.sel(from_lu=j_name, to_lu='Sheep Carbon Plantings (Belt)').item()
        if np.isnan(trans_cost):
            continue
        t_cp_r += trans_cost * dvars[:, :, j_idx].sum(axis=0)
    ag_to_cp_t_r = (tools.amortise((t_cp_r / ag_frac_r_safe) * data.REAL_AREA) * settings.CP_BELT_PROPORTION).astype(np.float32)

    # T_MAT (Sheep) — normalised weighted average × (1-CP_BELT_PROPORTION)
    t_sheep_r = np.zeros(data.NCELLS, dtype=np.float32)
    for j_idx, j_name in enumerate(data.AGRICULTURAL_LANDUSES):
        trans_cost = data.T_MAT.sel(from_lu=j_name, to_lu='Sheep - modified land').item()
        if np.isnan(trans_cost):
            continue
        t_sheep_r += trans_cost * dvars[:, :, j_idx].sum(axis=0)
    ag_to_sheep_t_r = (tools.amortise((t_sheep_r / ag_frac_r_safe) * data.REAL_AREA) * (1 - settings.CP_BELT_PROPORTION)).astype(np.float32)

    # WATER — expected irrigated fraction among the ag pool
    w_rm_irrig_cost_r = (irr_frac_r * settings.REMOVE_IRRIG_COST * data.IRRIG_COST_MULTS[yr_cal] * data.REAL_AREA).astype(np.float32)

    # FENCING — full whole-cell rate
    fencing_cost_r = (settings.CP_BELT_FENCING_LENGTH * settings.FENCING_COST_PER_M * data.FENCE_COST_MULTS[yr_cal] * data.REAL_AREA).astype(np.float32)

    if separate:
        return {
            'Establishment cost (Ag2Non-Ag)': est_costs_CP_r,
            'Transition cost (Ag2Non-Ag)': ag_to_cp_t_r,
            'Transition cost (Ag2CP-Sheep)': ag_to_sheep_t_r,
            'Remove irrigation cost (Ag2Non-Ag)': w_rm_irrig_cost_r,
            'Fencing cost (Ag2Non-Ag)': fencing_cost_r
        }
    else:
        return est_costs_CP_r + ag_to_cp_t_r + ag_to_sheep_t_r + w_rm_irrig_cost_r + fencing_cost_r


def get_sheep_carbon_plantings_belt_from_ag_from_base_year(
    data: Data, yr_idx, base_year, lumap, lmmap, separate=False
) -> np.ndarray | dict:
    """Dispatcher: blended or crisp-lumap Sheep CP Belt transition costs."""
    if settings.BLENDED_TRANSITION_COSTS:
        return get_sheep_carbon_plantings_belt_from_ag_blended(data, yr_idx, base_year, separate)
    else:
        return get_sheep_carbon_plantings_belt_from_ag(data, yr_idx, lumap, lmmap, separate)


def get_beef_carbon_plantings_belt_from_ag(
    data: Data,  yr_idx, lumap, lmmap, separate=False
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
    
    yr_cal = data.YR_CAL_BASE + yr_idx
    cells = np.isin(lumap, np.array(list(data.AGLU2DESC.keys())))

    # Establishment costs
    est_costs_CP_r = tools.amortise(data.CP_EST_COST_HA * data.REAL_AREA * data.EST_COST_MULTS[yr_cal]).astype(np.float32)
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
    w_rm_irrig_cost_r = np.where(lmmap == 1, settings.REMOVE_IRRIG_COST * data.IRRIG_COST_MULTS[yr_cal], 0) * data.REAL_AREA
    w_rm_irrig_cost_r[~cells] = 0.0

    fencing_cost_r = (
        settings.CP_BELT_FENCING_LENGTH
        * settings.FENCING_COST_PER_M
        * data.FENCE_COST_MULTS[data.YR_CAL_BASE + yr_idx]
        * data.REAL_AREA
    ).astype(np.float32)
    fencing_cost_r[~cells] = 0.0

    if separate:
        return {
            'Establishment cost (Ag2Non-Ag)': est_costs_CP_r,
            'Transition cost (Ag2Non-Ag)': ag_to_cp_t_r,
            'Transition cost (Ag2CP-Beef)': ag_to_sheep_t_r,
            'Remove irrigation cost (Ag2Non-Ag)': w_rm_irrig_cost_r,
            'Fencing cost (Ag2Non-Ag)': fencing_cost_r
        }
    else:
        return est_costs_CP_r + ag_to_cp_t_r + ag_to_sheep_t_r + w_rm_irrig_cost_r + fencing_cost_r


def get_beef_carbon_plantings_belt_from_ag_blended(data: Data, yr_idx, base_year, separate=False) -> np.ndarray | dict:
    """Blended Beef CP Belt transition costs: normalised weighted-average T_MAT, whole-cell EST/FENCING."""
    yr_cal    = data.YR_CAL_BASE + yr_idx
    dvars     = data.ag_dvars[base_year]
    dvar_base = tools.ag_mrj_to_xr(data, dvars, threshold=0)

    threshold      = 10 ** (-settings.ROUND_DECIMALS)
    ag_frac_r      = dvar_base.sum(['lm', 'lu']).values
    ag_frac_r_safe = np.where(ag_frac_r > threshold, ag_frac_r, 1.0)
    irr_frac_r     = dvar_base.sel(lm='irr').sum('lu').values

    # EST — full whole-cell rate × CP_BELT_PROPORTION
    est_costs_CP_r = (tools.amortise(data.CP_EST_COST_HA * data.REAL_AREA * data.EST_COST_MULTS[yr_cal]) * settings.CP_BELT_PROPORTION).astype(np.float32)

    # T_MAT (CP Belt) — normalised weighted average × CP_BELT_PROPORTION
    t_cp_r = np.zeros(data.NCELLS, dtype=np.float32)
    for j_idx, j_name in enumerate(data.AGRICULTURAL_LANDUSES):
        trans_cost = data.T_MAT.sel(from_lu=j_name, to_lu='Beef Carbon Plantings (Belt)').item()
        if np.isnan(trans_cost):
            continue
        t_cp_r += trans_cost * dvars[:, :, j_idx].sum(axis=0)
    ag_to_cp_t_r = (tools.amortise((t_cp_r / ag_frac_r_safe) * data.REAL_AREA) * settings.CP_BELT_PROPORTION).astype(np.float32)

    # T_MAT (Beef) — normalised weighted average × (1-CP_BELT_PROPORTION)
    t_beef_r = np.zeros(data.NCELLS, dtype=np.float32)
    for j_idx, j_name in enumerate(data.AGRICULTURAL_LANDUSES):
        trans_cost = data.T_MAT.sel(from_lu=j_name, to_lu='Beef - modified land').item()
        if np.isnan(trans_cost):
            continue
        t_beef_r += trans_cost * dvars[:, :, j_idx].sum(axis=0)
    ag_to_beef_t_r = (tools.amortise((t_beef_r / ag_frac_r_safe) * data.REAL_AREA) * (1 - settings.CP_BELT_PROPORTION)).astype(np.float32)

    # WATER — expected irrigated fraction among the ag pool
    w_rm_irrig_cost_r = (irr_frac_r * settings.REMOVE_IRRIG_COST * data.IRRIG_COST_MULTS[yr_cal] * data.REAL_AREA).astype(np.float32)

    # FENCING — full whole-cell rate
    fencing_cost_r = (settings.CP_BELT_FENCING_LENGTH * settings.FENCING_COST_PER_M * data.FENCE_COST_MULTS[yr_cal] * data.REAL_AREA).astype(np.float32)

    if separate:
        return {
            'Establishment cost (Ag2Non-Ag)': est_costs_CP_r,
            'Transition cost (Ag2Non-Ag)': ag_to_cp_t_r,
            'Transition cost (Ag2CP-Beef)': ag_to_beef_t_r,
            'Remove irrigation cost (Ag2Non-Ag)': w_rm_irrig_cost_r,
            'Fencing cost (Ag2Non-Ag)': fencing_cost_r
        }
    else:
        return est_costs_CP_r + ag_to_cp_t_r + ag_to_beef_t_r + w_rm_irrig_cost_r + fencing_cost_r


def get_beef_carbon_plantings_belt_from_ag_from_base_year(
    data: Data, yr_idx, base_year, lumap, lmmap, separate=False
) -> np.ndarray | dict:
    """Dispatcher: blended or crisp-lumap Beef CP Belt transition costs."""
    if settings.BLENDED_TRANSITION_COSTS:
        return get_beef_carbon_plantings_belt_from_ag_blended(data, yr_idx, base_year, separate)
    else:
        return get_beef_carbon_plantings_belt_from_ag(data, yr_idx, lumap, lmmap, separate)


def get_beccs_from_ag(data, yr_idx, base_year, lumap, lmmap, separate=False) -> np.ndarray|dict:
    """
    Get transition costs from agricultural land uses to BECCS for each cell.

    Returns
    -------
    np.ndarray
        1-D array, indexed by cell.
    """

    return get_env_plant_transitions_from_ag_from_base_year(data, yr_idx, base_year, lumap, lmmap, separate)


def get_destocked_from_ag(
    data: Data, yr_idx, lumap, lmmap, separate=False
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
    yr_cal = data.YR_CAL_BASE + yr_idx
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
    w_rm_irrig_cost_r = np.where(lmmap == 1, settings.REMOVE_IRRIG_COST * data.IRRIG_COST_MULTS[yr_cal], 0) * data.REAL_AREA
    w_rm_irrig_cost_r[~cells] = 0.0
    
    if separate:
        return {
            'Establishment cost (Ag2Non-Ag)': est_costs_r,
            'Remove irrigation cost (Ag2Non-Ag)': w_rm_irrig_cost_r
        }
    else:
        return est_costs_r + w_rm_irrig_cost_r


def get_destocked_from_ag_blended(data: Data, yr_idx, base_year, separate=False) -> np.ndarray | dict:
    """Blended Destocked transition costs: normalised weighted-average T_MAT and HCAS removal, whole-cell rates."""
    yr_cal    = data.YR_CAL_BASE + yr_idx
    dvars     = data.ag_dvars[base_year]
    dvar_base = tools.ag_mrj_to_xr(data, dvars, threshold=0)

    threshold  = 10 ** (-settings.ROUND_DECIMALS)
    irr_frac_r = dvar_base.sel(lm='irr').sum('lu').values

    HCAS_benefit_mult = {lu: 1 - data.BIO_HABITAT_CONTRIBUTION_LOOK_UP[lu] for lu in data.LU_LVSTK_NATURAL}

    # T_MAT and HCAS removal — only accumulate for j in LU_LVSTK_NATURAL (others are NaN in T_MAT).
    # Normalise by eligible_frac_r (lv-nat only) so the per-unit cost equals a pure lv-nat cell
    # regardless of how much non-eligible ag land is co-located in the same cell.
    t_destock_r     = np.zeros(data.NCELLS, dtype=np.float32)
    hcas_cost_r     = np.zeros(data.NCELLS, dtype=np.float32)
    eligible_frac_r = np.zeros(data.NCELLS, dtype=np.float32)
    for j_idx, j_name in enumerate(data.AGRICULTURAL_LANDUSES):
        trans_cost = data.T_MAT.sel(from_lu=j_name, to_lu='Destocked - natural land').item()
        if np.isnan(trans_cost):
            continue
        frac_j = dvars[:, :, j_idx].sum(axis=0)   # sum over lm
        t_destock_r     += trans_cost * frac_j
        eligible_frac_r += frac_j
        hcas_mult = HCAS_benefit_mult.get(j_name, 0.0)
        hcas_cost_r += hcas_mult * frac_j

    eligible_frac_r_safe = np.where(eligible_frac_r > threshold, eligible_frac_r, 1.0)
    trans_cost_r   = tools.amortise((t_destock_r / eligible_frac_r_safe) * data.REAL_AREA).astype(np.float32)
    removal_cost_r = tools.amortise((hcas_cost_r / eligible_frac_r_safe) * data.EP_EST_COST_HA * data.REAL_AREA).astype(np.float32)
    est_costs_r = removal_cost_r + trans_cost_r

    # WATER — expected irrigated fraction among the ag pool
    w_rm_irrig_cost_r = (irr_frac_r * settings.REMOVE_IRRIG_COST * data.IRRIG_COST_MULTS[yr_cal] * data.REAL_AREA).astype(np.float32)

    if separate:
        return {
            'Establishment cost (Ag2Non-Ag)': est_costs_r,
            'Remove irrigation cost (Ag2Non-Ag)': w_rm_irrig_cost_r
        }
    else:
        return est_costs_r + w_rm_irrig_cost_r


def get_destocked_from_ag_from_base_year(
    data: Data, yr_idx, base_year, lumap, lmmap, separate=False
) -> np.ndarray | dict:
    """Dispatcher: blended or crisp-lumap Destocked transition costs."""
    if settings.BLENDED_TRANSITION_COSTS:
        return get_destocked_from_ag_blended(data, yr_idx, base_year, separate)
    else:
        return get_destocked_from_ag(data, yr_idx, lumap, lmmap, separate)


def get_transition_matrix_ag2nonag(
    data: Data,
    yr_idx: int,
    lumap: np.ndarray,
    lmmap: np.ndarray,
    base_year: int = None,
    separate: bool = False,
) -> np.ndarray|dict:
    """
    Get the matrix containing transition costs from agricultural land uses to non-agricultural land uses.

    Parameters
    ----------
    data : object
        The data object containing information about the model.
    yr_idx : int
        The index of the year.
    lumap : dict
        The land use map.
    lmmap : dict
        The land management map.
    base_year : int, optional
        The calendar year of the previous solve step (used for blended transition costs).
    separate : bool, optional
        If True, return a dictionary containing the transition costs for each non-agricultural land use.
        If False, return a 2-D array indexed by (r, k) where r is cell and k is non-agricultural land usage.

    Returns
    -------
    np.ndarray or dict
        If separate is False, returns a 2-D array indexed by (r, k) where r is cell and k is non-agricultural land usage.
        If separate is True, returns a dictionary containing the transition costs for each non-agricultural land use.
    """

    env_plant_transitions_from_ag = get_env_plant_transitions_from_ag_from_base_year(data, yr_idx, base_year, lumap, lmmap, separate)
    rip_plant_transitions_from_ag = get_rip_plant_transitions_from_ag_from_base_year(data, yr_idx, base_year, lumap, lmmap, separate)
    sheep_agroforestry_transitions_from_ag = get_sheep_agroforestry_transitions_from_ag_from_base_year(data, yr_idx, base_year, lumap, lmmap, separate)
    beef_agroforestry_transitions_from_ag = get_beef_agroforestry_transitions_from_ag_from_base_year(data, yr_idx, base_year, lumap, lmmap, separate)
    carbon_plantings_block_transitions_from_ag = get_carbon_plantings_block_from_ag_from_base_year(data, yr_idx, base_year, lumap, lmmap, separate)
    sheep_carbon_plantings_belt_transitions_from_ag = get_sheep_carbon_plantings_belt_from_ag_from_base_year(data, yr_idx, base_year, lumap, lmmap, separate)
    beef_carbon_plantings_belt_transitions_from_ag = get_beef_carbon_plantings_belt_from_ag_from_base_year(data, yr_idx, base_year, lumap, lmmap, separate)
    beccs_transitions_from_ag = get_beccs_from_ag(data, yr_idx, base_year, lumap, lmmap, separate)
    destocked_from_ag = get_destocked_from_ag_from_base_year(data, yr_idx, base_year, lumap, lmmap, separate)

    if separate:
        # IMPORTANT: The order of the keys in the dictionary must match the order of the non-agricultural land uses
        return {
            'Environmental Plantings': env_plant_transitions_from_ag,
            'Riparian Plantings': rip_plant_transitions_from_ag,
            'Sheep Agroforestry': sheep_agroforestry_transitions_from_ag,
            'Beef Agroforestry': beef_agroforestry_transitions_from_ag,
            'Carbon Plantings (Block)': carbon_plantings_block_transitions_from_ag,
            'Sheep Carbon Plantings (Belt)': sheep_carbon_plantings_belt_transitions_from_ag,
            'Beef Carbon Plantings (Belt)': beef_carbon_plantings_belt_transitions_from_ag,
            'BECCS': beccs_transitions_from_ag,
            'Destocked - natural land': destocked_from_ag,
        }
        
    else:
        return np.array([
            env_plant_transitions_from_ag,
            rip_plant_transitions_from_ag,
            sheep_agroforestry_transitions_from_ag,
            beef_agroforestry_transitions_from_ag,
            carbon_plantings_block_transitions_from_ag,
            sheep_carbon_plantings_belt_transitions_from_ag,
            beef_carbon_plantings_belt_transitions_from_ag,
            beccs_transitions_from_ag,
            destocked_from_ag,
        ]).T.astype(np.float32)


def get_env_plantings_to_ag(data: Data, yr_idx, lumap, lmmap, separate=False) -> np.ndarray|dict:
    """
    Get transition costs from environmental plantings to agricultural land uses for each cell.

    Returns
    -------
    np.ndarray
        3-D array, indexed by (m, r, j).
    """
    yr_cal = data.YR_CAL_BASE + yr_idx
    l_mrj = tools.lumap2ag_l_mrj(lumap, lmmap)
    l_mrj_not = np.logical_not(l_mrj)           # This ensures the lu remains the same has 0 cost
    ag_cells, _ = tools.get_ag_and_non_ag_cells(lumap)

    # Get base transition costs: add cost of installing irrigation
    base_ep_to_ag_t = data.EP2AG_TRANSITION_COSTS_HA * data.TRANS_COST_MULTS[yr_cal]
    base_ep_to_ag_t_mrj = np.broadcast_to(base_ep_to_ag_t, (data.NLMS, data.NCELLS, base_ep_to_ag_t.shape[0]))
    base_ep_to_ag_t_mrj = tools.amortise(base_ep_to_ag_t_mrj).copy()
    base_ep_to_ag_t_mrj[:, ag_cells, :] = 0

    # Get water license price and costs of installing/removing irrigation where appropriate
    w_mrj = ag_water.get_wreq_matrices(data, yr_idx)
    w_delta_mrj = tools.get_ag_to_ag_water_delta_matrix(w_mrj, l_mrj, data, yr_idx)
    w_delta_mrj[:, ag_cells, :] = 0
    

    if separate:
        return {'Transition cost (Non-Ag2Ag)':np.nan_to_num(np.einsum('mrj,mrj,r->mrj', base_ep_to_ag_t_mrj, l_mrj_not, data.REAL_AREA)), 
                'Water license cost (Non-Ag2Ag)': np.nan_to_num(np.einsum('mrj,mrj->mrj', w_delta_mrj, l_mrj_not))}
        
    # base_ep_to_ag_t_mrj is $/ha → convert to $/cell via REAL_AREA; w_delta_mrj is already
    # $/cell (see tools.get_ag_to_ag_water_delta_matrix), so it must NOT be multiplied by
    # REAL_AREA again. This mirrors the ag→ag path (agricultural/transitions.py:126,140-141).
    ep_to_ag_t_mrj = (
        base_ep_to_ag_t_mrj * data.REAL_AREA[np.newaxis, :, np.newaxis] + w_delta_mrj
    ) * l_mrj_not
    return np.nan_to_num(ep_to_ag_t_mrj) 


def get_rip_plantings_to_ag(data: Data, yr_idx, lumap, lmmap, separate=False) -> np.ndarray|dict:
    """
    Get transition costs from riparian plantings to agricultural land uses for each cell.
    
    Note: this is the same as for environmental plantings.

    Returns
    -------
    np.ndarray
        3-D array, indexed by (m, r, j).
    """
    if separate:
        return get_env_plantings_to_ag(data, yr_idx, lumap, lmmap, separate)
    else:
        return get_env_plantings_to_ag(data, yr_idx, lumap, lmmap)


def get_agroforestry_to_ag_base(data: Data, yr_idx, lumap, lmmap, separate) -> np.ndarray|dict:
    """
    Get transition costs from agroforestry to agricultural land uses for each cell.
    
    Note: this is the same as for environmental plantings.

    Returns
    -------
    np.ndarray
        3-D array, indexed by (m, r, j).
    """
    if separate:
        return get_env_plantings_to_ag(data, yr_idx, lumap, lmmap, separate)
    else:
        return get_env_plantings_to_ag(data, yr_idx, lumap, lmmap)


def get_sheep_to_ag_base(data: Data, yr_idx: int, lumap, separate=False) -> np.ndarray|dict:
    """
    Get sheep contribution to transition costs to agricultural land uses.
    Used for getting transition costs for Sheep Agroforestry and CP (Belt).

    Returns
    -------
    np.ndarray separate = False
        3-D array, indexed by (m, r, j).
    dict (separate = True)
        Dictionary of separated out transition costs.
    ------
    """
    yr_cal = data.YR_CAL_BASE + yr_idx
    sheep_j = tools.get_sheep_code(data)

    all_sheep_lumap = (np.ones(data.NCELLS) * sheep_j).astype(np.int8)
    all_dry_lmmap = np.zeros(data.NCELLS).astype(np.float32)
    l_mrj = tools.lumap2ag_l_mrj(all_sheep_lumap, all_dry_lmmap)
    l_mrj_not = np.logical_not(l_mrj)

    t_ij = data.AG_TMATRIX * data.TRANS_COST_MULTS[yr_cal]
    x_mrj = ag_transitions.get_to_ag_exclude_matrices(data, all_sheep_lumap)

    # Calculate sheep contribution to transition costs
    # Establishment costs
    ag_cells = tools.get_ag_cells(lumap)

    e_rj = np.zeros((data.NCELLS, data.N_AG_LUS)).astype(np.float32)
    e_rj[ag_cells, :] = t_ij[all_sheep_lumap[ag_cells]]

    e_rj = tools.amortise(e_rj) * data.REAL_AREA[:, np.newaxis]
    e_rj_dry = np.einsum('rj,r->rj', e_rj, all_sheep_lumap == 0)
    e_rj_irr = np.einsum('rj,r->rj', e_rj, all_dry_lmmap == 1)
    e_mrj = np.stack([e_rj_dry, e_rj_irr], axis=0)
    e_mrj = np.einsum('mrj,mrj,mrj->mrj', e_mrj, x_mrj, l_mrj_not)

    # Water license cost
    w_mrj = ag_water.get_wreq_matrices(data, yr_idx)
    w_delta_mrj = tools.get_ag_to_ag_water_delta_matrix(w_mrj, l_mrj, data, yr_idx)
    w_delta_mrj = np.einsum('mrj,mrj,mrj->mrj', w_delta_mrj, x_mrj, l_mrj_not)

    # Carbon costs
    ghg_t_mrj = ag_ghg.get_ghg_transition_emissions(data, all_sheep_lumap)               # <unit: t/ha>      
    ghg_t_mrj_cost = tools.amortise(ghg_t_mrj * data.get_carbon_price_by_yr_idx(yr_idx))     
    ghg_t_mrj_cost = np.einsum('mrj,mrj,mrj->mrj', ghg_t_mrj_cost, x_mrj, l_mrj_not)

    # Ensure transition costs are zero for all agricultural cells 
    e_mrj[:, ag_cells, :] = np.zeros((data.NLMS, ag_cells.shape[0], data.N_AG_LUS)).astype(np.float32)
    w_delta_mrj[:, ag_cells, :] = np.zeros((data.NLMS, ag_cells.shape[0], data.N_AG_LUS)).astype(np.float32)
    ghg_t_mrj_cost[:, ag_cells, :] = np.zeros((data.NLMS, ag_cells.shape[0], data.N_AG_LUS)).astype(np.float32)

    if separate:
        return {
            'Establishment cost (Non-Ag2Ag)': np.nan_to_num(e_mrj), 
            'Water license cost (Non-Ag2Ag)': np.nan_to_num(w_delta_mrj), 
            'GHG emissions cost (Non-Ag2Ag)': np.nan_to_num(ghg_t_mrj_cost)
        }
    
    else:
        return np.nan_to_num(e_mrj + w_delta_mrj + ghg_t_mrj_cost)


def get_beef_to_ag_base(data: Data, yr_idx, lumap, separate) -> np.ndarray|dict:
    """
    Get beef contribution to transition costs to agricultural land uses.
    Used for getting transition costs for Beef Agroforestry and CP (Belt).

    Returns
    -------
    np.ndarray separate = False
        3-D array, indexed by (m, r, j).
    dict (separate = True)
        Dictionary of separated out transition costs.
    """
    yr_cal = data.YR_CAL_BASE + yr_idx
    beef_j = tools.get_beef_code(data)

    all_beef_lumap = (np.ones(data.NCELLS) * beef_j).astype(np.int8)
    all_dry_lmmap = np.zeros(data.NCELLS).astype(np.float32)
    l_mrj = tools.lumap2ag_l_mrj(all_beef_lumap, all_dry_lmmap)
    l_mrj_not = np.logical_not(l_mrj)

    t_ij = data.AG_TMATRIX * data.TRANS_COST_MULTS[yr_cal]
    x_mrj = ag_transitions.get_to_ag_exclude_matrices(data, all_beef_lumap)

    # Establishment costs
    ag_cells = tools.get_ag_cells(lumap)

    e_rj = np.zeros((data.NCELLS, data.N_AG_LUS)).astype(np.float32)
    e_rj[ag_cells, :] = t_ij[all_beef_lumap[ag_cells]]

    e_rj = tools.amortise(e_rj) * data.REAL_AREA[:, np.newaxis]
    e_mrj = np.stack([e_rj] * data.NLMS, axis=0)
    e_mrj = np.einsum('mrj,mrj,mrj->mrj', e_mrj, x_mrj, l_mrj_not)

    # Water license cost
    w_mrj = ag_water.get_wreq_matrices(data, yr_idx)
    w_delta_mrj = tools.get_ag_to_ag_water_delta_matrix(w_mrj, l_mrj, data, yr_idx)
    w_delta_mrj = np.einsum('mrj,mrj,mrj->mrj', w_delta_mrj, x_mrj, l_mrj_not)

    # Carbon costs
    ghg_t_mrj = ag_ghg.get_ghg_transition_emissions(data, all_beef_lumap)               # <unit: t/ha>      
    ghg_t_mrj_cost = tools.amortise(ghg_t_mrj * data.get_carbon_price_by_yr_idx(yr_idx))     
    ghg_t_mrj_cost = np.einsum('mrj,mrj,mrj->mrj', ghg_t_mrj_cost, x_mrj, l_mrj_not)

    beef_af_cells = tools.get_beef_agroforestry_cells(lumap)
    non_beef_af_cells = np.array([r for r in range(data.NCELLS) if r not in beef_af_cells])

    # Ensure transition costs are zero for all agricultural cells 
    e_mrj[:, ag_cells, :] = np.zeros((data.NLMS, ag_cells.shape[0], data.N_AG_LUS)).astype(np.float32)
    w_delta_mrj[:, ag_cells, :] = np.zeros((data.NLMS, ag_cells.shape[0], data.N_AG_LUS)).astype(np.float32)
    ghg_t_mrj_cost[:, ag_cells, :] = np.zeros((data.NLMS, ag_cells.shape[0], data.N_AG_LUS)).astype(np.float32)

    if separate:
        return {
            'Establishment cost (Non-Ag2Ag)': np.nan_to_num(e_mrj), 
            'Water license cost (Non-Ag2Ag)': np.nan_to_num(w_delta_mrj), 
            'GHG emissions cost (Non-Ag2Ag)': np.nan_to_num(ghg_t_mrj_cost)
        }
    
    else:
        t_mrj = e_mrj + w_delta_mrj + ghg_t_mrj_cost
        # Set all costs for non-beef-agroforestry cells to zero
        t_mrj[:, non_beef_af_cells, :] = 0
        return np.nan_to_num(t_mrj)


def get_sheep_agroforestry_to_ag(
    data: Data, yr_idx, lumap, lmmap, agroforestry_x_r, separate=False
) -> np.ndarray|dict:
    """
    Get transition costs of Sheep Agroforestry to all agricultural land uses.

    Returns
    -------
    np.ndarray separate = False
        3-D array, indexed by (m, r, j).
    dict (separate = True)
        Dictionary of separated out transition costs.
    """
    sheep_tcosts = get_sheep_to_ag_base(data, yr_idx, lumap, separate)
    agroforestry_tcosts = get_agroforestry_to_ag_base(data, yr_idx, lumap, lmmap, separate)

    if separate:
        # Combine and return separated costs
        combined_costs = {}
        for key, array in agroforestry_tcosts.items():
            combined_costs[key] = np.zeros(array.shape).astype(np.float32)
            for m in range(data.NLMS):
                for j in range(data.N_AG_LUS):
                    combined_costs[key][m, :, j] = array[m, :, j] * agroforestry_x_r

        for key, array in sheep_tcosts.items():
            if key not in combined_costs:
                combined_costs[key] = np.zeros(array.shape).astype(np.float32)
            for m in range(data.NLMS):
                for j in range(data.N_AG_LUS):
                    combined_costs[key][m, :, j] += array[m, :, j] * (1 - agroforestry_x_r)

        return combined_costs
    
    else:
        sheep_contr = np.zeros((data.NLMS, data.NCELLS, data.N_AG_LUS)).astype(np.float32)
        for m in range(data.NLMS):
            for j in range(data.N_AG_LUS):
                sheep_contr[m, :, j] = (1 - agroforestry_x_r) * sheep_tcosts[m, :, j]

        agroforestry_contr = np.zeros((data.NLMS, data.NCELLS, data.N_AG_LUS)).astype(np.float32)
        for m in range(data.NLMS):
            for j in range(data.N_AG_LUS):
                agroforestry_contr[m, :, j] = agroforestry_x_r * agroforestry_tcosts[m, :, j]

        return sheep_contr + agroforestry_contr


def get_beef_agroforestry_to_ag(
    data: Data, yr_idx, lumap, lmmap, agroforestry_x_r, separate=False
) -> np.ndarray|dict:
    """
    Get transition costs of Beef Agroforestry to all agricultural land uses.
    
    Returns
    -------
    np.ndarray separate = False
        3-D array, indexed by (m, r, j).
    dict (separate = True)
        Dictionary of separated out transition costs.
    """
    beef_tcosts = get_beef_to_ag_base(data, yr_idx, lumap, separate)
    agroforestry_tcosts = get_agroforestry_to_ag_base(data, yr_idx, lumap, lmmap, separate)

    if separate:
        # Combine and return separated costs
        combined_costs = {}
        for key, array in agroforestry_tcosts.items():
            combined_costs[key] = np.zeros(array.shape).astype(np.float32)
            for m in range(data.NLMS):
                for j in range(data.N_AG_LUS):
                    combined_costs[key][m, :, j] = array[m, :, j] * agroforestry_x_r

        for key, array in beef_tcosts.items():
            if key not in combined_costs:
                combined_costs[key] = np.zeros(array.shape).astype(np.float32)
            for m in range(data.NLMS):
                for j in range(data.N_AG_LUS):
                    combined_costs[key][m, :, j] += array[m, :, j] * (1 - agroforestry_x_r)

        return combined_costs
    
    else:
        beef_contr = np.zeros((data.NLMS, data.NCELLS, data.N_AG_LUS)).astype(np.float32)
        for m in range(data.NLMS):
            for j in range(data.N_AG_LUS):
                beef_contr[m, :, j] = (1 - agroforestry_x_r) * beef_tcosts[m, :, j]

        agroforestry_contr = np.zeros((data.NLMS, data.NCELLS, data.N_AG_LUS)).astype(np.float32)
        for m in range(data.NLMS):
            for j in range(data.N_AG_LUS):
                agroforestry_contr[m, :, j] = agroforestry_x_r * agroforestry_tcosts[m, :, j]

        return beef_contr + agroforestry_contr


def get_carbon_plantings_block_to_ag(data: Data, yr_idx, lumap, lmmap, separate=False):
    """
    Get transition costs from carbon plantings (block) to agricultural land uses for each cell.
    
    Note: this is the same as for environmental plantings.

    Returns
    -------
    np.ndarray
        3-D array, indexed by (m, r, j).
    """
    return get_env_plantings_to_ag(data, yr_idx, lumap, lmmap, separate)


def get_carbon_plantings_belt_to_ag_base(data, yr_idx, lumap, lmmap, separate=False) -> np.ndarray|dict:
    """
    Get transition costs from carbon plantings (belt) to agricultural land uses for each cell.
    
    Note: this is the same as for environmental plantings.

    Returns
    -------
    np.ndarray
        3-D array, indexed by (m, r, j).
    """
    return get_env_plantings_to_ag(data, yr_idx, lumap, lmmap, separate)


def get_sheep_carbon_plantings_belt_to_ag(
    data: Data, yr_idx, lumap, lmmap, cp_belt_x_r, separate
) -> np.ndarray|dict:
    """
    Get transition costs of Sheep Carbon Plantings (Belt) to all agricultural land uses.
    
    Returns
    -------
    np.ndarray separate = False
        3-D array, indexed by (m, r, j).
    dict (separate = True)
        Dictionary of separated out transition costs.
    """
    sheep_tcosts = get_sheep_to_ag_base(data, yr_idx, lumap, separate)
    cp_belt_tcosts = get_carbon_plantings_belt_to_ag_base(data, yr_idx, lumap, lmmap, separate)

    if separate:
        # Combine and return separated costs
        combined_costs = {}
        for key, array in cp_belt_tcosts.items():
            combined_costs[key] = np.zeros(array.shape).astype(np.float32)
            for m in range(data.NLMS):
                for j in range(data.N_AG_LUS):
                    combined_costs[key][m, :, j] = array[m, :, j] * cp_belt_x_r

        for key, array in sheep_tcosts.items():
            if key not in combined_costs:
                combined_costs[key] = np.zeros(array.shape).astype(np.float32)
            for m in range(data.NLMS):
                for j in range(data.N_AG_LUS):
                    combined_costs[key][m, :, j] += array[m, :, j] * (1 - cp_belt_x_r)

        return combined_costs
    
    else:
        sheep_contr = np.zeros((data.NLMS, data.NCELLS, data.N_AG_LUS)).astype(np.float32)
        for m in range(data.NLMS):
            for j in range(data.N_AG_LUS):
                sheep_contr[m, :, j] = (1 - cp_belt_x_r) * sheep_tcosts[m, :, j]

        cp_belt_contr = np.zeros((data.NLMS, data.NCELLS, data.N_AG_LUS)).astype(np.float32)
        for m in range(data.NLMS):
            for j in range(data.N_AG_LUS):
                cp_belt_contr[m, :, j] = cp_belt_x_r * cp_belt_tcosts[m, :, j]

        return sheep_contr + cp_belt_contr
    

def get_beef_carbon_plantings_belt_to_ag(
    data: Data, yr_idx, lumap, lmmap, cp_belt_x_r, separate
) -> np.ndarray|dict:
    """
    Get transition costs of Beef Carbon Plantings (Belt) to all agricultural land uses.
    
    Returns
    -------
    np.ndarray separate = False
        3-D array, indexed by (m, r, j).
    dict (separate = True)
        Dictionary of separated out transition costs.
    """
    beef_tcosts = get_beef_to_ag_base(data, yr_idx, lumap, separate)
    cp_belt_tcosts = get_carbon_plantings_belt_to_ag_base(data, yr_idx, lumap, lmmap, separate)

    if separate:
        # Combine and return separated costs
        combined_costs = {}
        for key, array in cp_belt_tcosts.items():
            combined_costs[key] = np.zeros(array.shape).astype(np.float32)
            for m in range(data.NLMS):
                for j in range(data.N_AG_LUS):
                    combined_costs[key][m, :, j] = array[m, :, j] * cp_belt_x_r

        for key, array in beef_tcosts.items():
            if key not in combined_costs:
                combined_costs[key] = np.zeros(array.shape).astype(np.float32)
            for m in range(data.NLMS):
                for j in range(data.N_AG_LUS):
                    combined_costs[key][m, :, j] += array[m, :, j] * (1 - cp_belt_x_r)

        return combined_costs
    
    else:
        beef_contr = np.zeros((data.NLMS, data.NCELLS, data.N_AG_LUS)).astype(np.float32)
        for m in range(data.NLMS):
            for j in range(data.N_AG_LUS):
                beef_contr[m, :, j] = (1 - cp_belt_x_r) * beef_tcosts[m, :, j]

        cp_belt_contr = np.zeros((data.NLMS, data.NCELLS, data.N_AG_LUS)).astype(np.float32)
        for m in range(data.NLMS):
            for j in range(data.N_AG_LUS):
                cp_belt_contr[m, :, j] = cp_belt_x_r * cp_belt_tcosts[m, :, j]

        return beef_contr + cp_belt_contr


def get_beccs_to_ag(data: Data, yr_idx, lumap, lmmap, separate=False) -> np.ndarray|dict:
    """
    Get transition costs from BECCS to agricultural land uses for each cell.
    
    Note: this is the same as for environmental plantings.

    Returns
    -------
    np.ndarray
        3-D array, indexed by (m, r, j).
    """
    if separate:
        return get_env_plantings_to_ag(data, yr_idx, lumap, lmmap, separate)
    else:
        return get_env_plantings_to_ag(data, yr_idx, lumap, lmmap)
    

def get_destocked_to_ag(data: Data, yr_idx: int, lumap: np.ndarray, lmmap: np.ndarray, separate: bool = False) -> np.ndarray:
    """
    Get transition costs from destocked land to agricultural land uses for each cell.
    Transition costs are based on the transition costs of unallocated natural land to agricultural land.
    
    Returns
    -------
    np.ndarray
        3-D array, indexed by (m, r, j).
    """
    unallocated_j = tools.get_unallocated_natural_land_code(data)
    all_unallocated_lumap = (np.ones(data.NCELLS) * unallocated_j).astype(np.int8)
    all_dry_lmmap = (np.zeros(data.NCELLS)).astype(np.int8)

    destocked_cells = tools.get_destocked_land_cells(lumap)
    if destocked_cells.size == 0 and separate == False:
        return np.zeros((data.NLMS, data.NCELLS, data.N_AG_LUS))
    
    # Get transition costs from destocked cells by using transition costs from unallocated land
    unallocated_t_mrj = ag_transitions.get_transition_matrices_ag2ag(
        data, yr_idx, all_unallocated_lumap, all_dry_lmmap, separate=separate
    )

    if separate == False:
        destocked_t_mrj = np.zeros((data.NLMS, data.NCELLS, data.N_AG_LUS))
        destocked_t_mrj[:, destocked_cells, :] = unallocated_t_mrj[:, destocked_cells, :]
        return destocked_t_mrj
    
    elif separate == True:
        sep_destocked_trans = {k: np.zeros((data.NLMS, data.NCELLS, data.N_AG_LUS)) for k in unallocated_t_mrj}
        if destocked_cells.size == 0:
            return sep_destocked_trans

        for k, v in unallocated_t_mrj.items():
            sep_destocked_trans[k][:, destocked_cells, :] = v[:, destocked_cells, :]
        return sep_destocked_trans

    raise ValueError(
        f"Incorrect value for 'separate' when calling get_destocked_from_ag: {separate}. "
        f"should be either True or False."
    )


def get_irreversible_non_ag_cell_mask(data: Data, base_year) -> np.ndarray | None:
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


def get_transition_matrix_nonag2ag(data: Data, yr_idx, lumap, lmmap, separate=False, base_year=None) -> np.ndarray|dict:
    """
    Get the matrix containing transition costs from non-agricultural land uses to agricultural land uses.

    Parameters
    ----------
    data : np.ndarray
        The input data array.
    yr_idx : int
        The index of the year.
    lumap : dict
        The land use mapping dictionary.
    lmmap : dict
        The land management mapping dictionary.
    separate : bool, optional
        If True, returns a dictionary of transition matrices for each land use category.
        If False, returns a single aggregated transition matrix.
    base_year : int, optional
        The base year whose non-ag dvars define the irreversible lock-in. When supplied,
        cells already holding an irreversible non-ag LU are excluded from the non-ag→ag
        transition cost (see note below). Defaults to None (no exclusion).

    Returns
    -------
    np.ndarray or dict
        If `separate` is True, returns a dictionary of transition matrices, where the keys are the land use categories.
        If `separate` is False, returns a single aggregated transition matrix.

    """
    
    non_ag_to_agr_t_matrices = {lu: np.zeros((data.NLMS, data.NCELLS, data.N_AG_LUS)).astype(np.float32) for lu in settings.NON_AG_LAND_USES}

    agroforestry_x_r = tools.get_exclusions_agroforestry_base(data, lumap)
    cp_belt_x_r = tools.get_exclusions_carbon_plantings_belt_base(data, lumap)

    # reshape each non-agricultural matrix to be indexed (r, k) and concatenate on the k indexing
    non_ag_to_agr_t_matrices['Environmental Plantings'] = get_env_plantings_to_ag(data, yr_idx, lumap, lmmap, separate)
    non_ag_to_agr_t_matrices['Riparian Plantings'] = get_rip_plantings_to_ag(data, yr_idx, lumap, lmmap, separate)
    non_ag_to_agr_t_matrices['Sheep Agroforestry'] = get_sheep_agroforestry_to_ag(data, yr_idx, lumap, lmmap, agroforestry_x_r, separate)
    non_ag_to_agr_t_matrices['Beef Agroforestry'] = get_beef_agroforestry_to_ag(data, yr_idx, lumap, lmmap, agroforestry_x_r, separate)
    non_ag_to_agr_t_matrices['Carbon Plantings (Block)'] = get_carbon_plantings_block_to_ag(data, yr_idx, lumap, lmmap, separate)
    non_ag_to_agr_t_matrices['Sheep Carbon Plantings (Belt)'] = get_sheep_carbon_plantings_belt_to_ag(data, yr_idx, lumap, lmmap, cp_belt_x_r, separate)
    non_ag_to_agr_t_matrices['Beef Carbon Plantings (Belt)'] = get_beef_carbon_plantings_belt_to_ag(data, yr_idx, lumap, lmmap, cp_belt_x_r, separate)
    non_ag_to_agr_t_matrices['BECCS'] = get_beccs_to_ag(data, yr_idx, lumap, lmmap, separate)
    non_ag_to_agr_t_matrices['Destocked - natural land'] = get_destocked_to_ag(data, yr_idx, lumap, lmmap, separate)

    # Cells that already hold an irreversible non-ag LU cannot truly revert that locked
    # fraction back to agriculture (irreversibility is enforced via the non-ag lb-lock).
    # Because transition costs are integerised-lumap based, the model would otherwise
    # charge a spurious non-ag→ag transition cost on the free fraction of these cells.
    # Zero the cost there so the solver (and cost reporting) never see it.
    irrev_mask_r = get_irreversible_non_ag_cell_mask(data, base_year)
    if irrev_mask_r is not None and irrev_mask_r.any():
        for mat in non_ag_to_agr_t_matrices.values():
            if isinstance(mat, dict):
                for src_arr in mat.values():
                    src_arr[:, irrev_mask_r, :] = 0
            else:
                mat[:, irrev_mask_r, :] = 0

    if separate:
        # Note: The order of the keys in the dictionary must match the order of the non-agricultural land uses
        return non_ag_to_agr_t_matrices
            
    non_ag_to_agr_t_matrices = list(non_ag_to_agr_t_matrices.values())
    return np.add.reduce(non_ag_to_agr_t_matrices)


def get_non_ag_to_non_ag_transition_matrix(data: Data) -> np.ndarray:
    """
    Get the matrix that contains transition costs for non-agricultural land uses. 
    There are no transition costs for non-agricultural land uses, therefore the matrix is filled with zeros.
    
    Parameters
        data (object): The data object containing information about the model.
    
    Returns
        np.ndarray: The transition cost matrix, filled with zeros.
    """
    return np.zeros((data.NCELLS, data.N_NON_AG_LUS)).astype(np.float32)




def get_non_ag_ub_matrices(data: Data, lumap, base_dvar_nonag_rk, base_dvar_ag_mrj) -> np.ndarray:
    """
    Returns the upper-bound (UB) matrix for non-agricultural land uses,
    shape (NCELLS, N_NON_AG_LUS).  Each entry is the maximum cell proportion
    the solver may allocate to that non-ag LU.

    Two rules set the UB:

    1. New allocations — transition matrix decide whether a cell is allowed to take 
       on a non-ag LU for the first time. UB = 0 blocks the transition; UB = 1.
       
    2. No-go zones for non-ag→ag transitions — when EXCLUDE_NO_GO_LU is True, specified 
       no-go LU columns are set to UB=0 for all cells in the specified no-go regions. 
       This is applied after the above rules so it can only further restrict UB.

    3. Existing irreversible allocations — when base_dvar_nonag_rk is supplied,
       any cell that already holds an irreversible non-ag LU is forced to UB = 1,
       bypassing the transition-matrix check.  This guarantees the cell stays in
       non_ag_lu2cells every solve so its lower bound (lock-in) remains active.
       Without this override, a T_MAT block in year Y would give dvar = 0, which
       propagates lb = 0 in year Y+1 and silently breaks irreversibility.

    4. Blended Destocked cap (base_dvar_ag_mrj supplied + BLENDED_TRANSITION_COSTS) —
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
    if settings.BLENDED_TRANSITION_COSTS:
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
