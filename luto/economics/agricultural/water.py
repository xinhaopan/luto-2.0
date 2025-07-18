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
Pure functions to calculate water net yield by lm, lu and water limits.
"""

import numpy as np
import pandas as pd

from typing import Optional
import luto.settings as settings
from luto.economics.agricultural.quantity import get_yield_pot, lvs_veg_types



def get_wreq_matrices(data, yr_idx):
    """
    Return water requirement (water use by irrigation and livestock drinking water) matrices
    by land management, cell, and land-use type.

    Parameters
        data (object): The data object containing the required data.
        yr_idx (int): The index of the year.

    Returns
        numpy.ndarray: The w_mrj <unit: ML/cell> water requirement matrices, indexed (m, r, j).
    """

    # Stack water requirements data
    w_req_mrj = np.stack((data.WREQ_DRY_RJ, data.WREQ_IRR_RJ))    # <unit: ML/head|ha>

    # Covert water requirements units from ML/head to ML/ha
    for j, lu in enumerate(data.AGRICULTURAL_LANDUSES):
        if lu in data.LU_LVSTK:
            lvs, veg = lvs_veg_types(lu)
            w_req_mrj[0, :, j] = w_req_mrj[0, :, j] * get_yield_pot(data, lvs, veg, 'dry', yr_idx)  # Water reqs depend on current stocking rate for drinking water
            w_req_mrj[1, :, j] = w_req_mrj[1, :, j] * get_yield_pot(data, lvs, veg, 'irr', 0)       # Water reqs depend on initial stocking rate for irrigation

    # Convert to ML per cell via REAL_AREA
    w_req_mrj *= data.REAL_AREA[:, np.newaxis]                      # <unit: ML/ha> * <unit: ha/cell> -> <unit: ML/cell>

    return w_req_mrj


def get_wyield_matrices(
    data, 
    yr_idx:int, 
    water_dr_yield: Optional[np.ndarray] = None,
    water_sr_yield: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Return water yield matrices for YR_CAL_BASE (2010) by land management, cell, and land-use type.

    Parameters
        data (object): The data object containing the required data.
        yr_idx (int): The index of the year.
        water_dr_yield (ndarray, <unit:ML/cell>): The water yield for deep-rooted vegetation.
        water_sr_yield (ndarray, <unit:ML/cell>): The water yield for shallow-rooted vegetation.

    Returns
        numpy.ndarray: The w_mrj <unit: ML/cell> water yield matrices, indexed (m, r, j).
    """
    w_yield_mrj = np.zeros((data.NLMS, data.NCELLS, data.N_AG_LUS)).astype(np.float32)

    w_yield_dr = data.WATER_YIELD_DR_FILE[yr_idx] if water_dr_yield is None else water_dr_yield
    w_yield_sr = data.WATER_YIELD_SR_FILE[yr_idx] if water_sr_yield is None else water_sr_yield
    w_yield_nl = data.get_water_nl_yield_for_yr_idx(yr_idx, w_yield_dr, w_yield_sr)


    for j in range(data.N_AG_LUS):
        if j in data.LU_SHALLOW_ROOTED:
            for m in range(data.NLMS):
                w_yield_mrj[m, :, j] = w_yield_sr * data.REAL_AREA

        elif j in data.LU_DEEP_ROOTED:
            for m in range(data.NLMS):
                w_yield_mrj[m, :, j] = w_yield_dr * data.REAL_AREA

        elif j in data.LU_NATURAL:
            for m in range(data.NLMS):
                w_yield_mrj[m, :, j] = w_yield_nl * data.REAL_AREA

        else:
            raise ValueError(
                f"Land use {j} ({data.AGLU2DESC[j]}) missing from all of "
                f"data.LU_SHALLOW_ROOTED, data.LU_DEEP_ROOTED, data.LU_NATURAL "
                f"(requires root definition)."
            )

    return w_yield_mrj


def get_water_net_yield_matrices(
    data, 
    yr_idx, 
    water_dr_yield: Optional[np.ndarray] = None,
    water_sr_yield: Optional[np.ndarray] = None
    ) -> np.ndarray:
    """
    Return water net yield matrices by land management, cell, and land-use type.
    The resulting array is used as the net yield w_mrj array in the input data of the solver.

    Parameters
        data (object): The data object containing the required data.
        yr_idx (int): The index of the year.
        water_dr_yield (ndarray, <unit:ML/cell>): The water yield for deep-rooted vegetation.
        water_sr_yield (ndarray, <unit:ML/cell>): The water yield for shallow-rooted vegetation.
        
    Notes:
        Provides the `water_dr_yield` or `water_sr_yield` will make the `yr_idx` useless because
        the water net yield will be calculated based on the provided water yield data regardless of the year.

    Returns
        numpy.ndarray: The w_mrj <unit: ML/cell> water net yield matrices, indexed (m, r, j).
    """
    return get_wyield_matrices(data, yr_idx, water_dr_yield, water_sr_yield) - get_wreq_matrices(data, yr_idx)


def get_asparagopsis_effect_w_mrj(data, yr_idx):
    """
    Applies the effects of using asparagopsis to the water net yield data
    for all relevant agr. land uses.

    Args:
        data (object): The data object containing relevant information.
        w_mrj (ndarray, <unit:ML/cell>): The water net yield data for all land uses.
        yr_idx (int): The index of the year.

    Returns
        ndarray <unit:ML/cell>: The updated water net yield data with the effects of using asparagopsis.

    Notes:
        Asparagopsis taxiformis has no effect on the water required.
    """

    land_uses = settings.AG_MANAGEMENTS_TO_LAND_USES["Asparagopsis taxiformis"]
    lu_codes = np.array([data.DESC2AGLU[lu] for lu in land_uses])
    yr_cal = data.YR_CAL_BASE + yr_idx

    wreq_mrj = get_wreq_matrices(data, yr_idx)

    # Set up the effects matrix
    w_mrj_effect = np.zeros((data.NLMS, data.NCELLS, len(land_uses))).astype(np.float32)

    # Update values in the new matrix using the correct multiplier for each LU
    for lu_idx, lu in enumerate(land_uses):
        multiplier = data.ASPARAGOPSIS_DATA[lu].loc[yr_cal, "Water_use"]
        if multiplier != 1:
            j = lu_codes[lu_idx]
            # The effect is: new value = old value * multiplier - old value
            # E.g. a multiplier of .95 means a 5% reduction in water used.
            # Since the effect applies to water use, it effects the net yield negatively.
            w_mrj_effect[:, :, lu_idx] = wreq_mrj[:, :, j] * (1- multiplier)

    return w_mrj_effect


def get_precision_agriculture_effect_w_mrj(data, yr_idx):
    """
    Applies the effects of using precision agriculture to the water net yield data
    for all relevant agricultural land uses.

    Parameters
    - data: The data object containing relevant information for the calculation.
    - w_mrj <unit:ML/cell>: The original water net yield data for different land uses.
    - yr_idx: The index representing the year for which the calculation is performed.

    Returns
    - w_mrj_effect <unit:ML/cell>: The updated water net yield data after applying precision agriculture effects.
    """

    land_uses = settings.AG_MANAGEMENTS_TO_LAND_USES['Precision Agriculture']
    lu_codes = np.array([data.DESC2AGLU[lu] for lu in land_uses])
    yr_cal = data.YR_CAL_BASE + yr_idx

    wreq_mrj = get_wreq_matrices(data, yr_idx)

    # Set up the effects matrix
    w_mrj_effect = np.zeros((data.NLMS, data.NCELLS, len(land_uses))).astype(np.float32)

    # Update values in the new matrix using the correct multiplier for each land use
    for lu_idx, lu in enumerate(land_uses):
        multiplier = data.PRECISION_AGRICULTURE_DATA[lu].loc[yr_cal, "Water_use"]
        if multiplier != 1:
            j = lu_codes[lu_idx]
            # The effect is: new value = old value * multiplier - old value
            # E.g. a multiplier of .95 means a 5% reduction in water used.
            # Since the effect applies to water use, it effects the net yield negatively.
            w_mrj_effect[:, :, lu_idx] = wreq_mrj[:, :, j] * (1- multiplier)

    return w_mrj_effect


def get_ecological_grazing_effect_w_mrj(data, yr_idx):
    """
    Applies the effects of using ecological grazing to the water net yield data
    for all relevant agricultural land uses.

    Parameters
    - data: The data object containing relevant information.
    - w_mrj <unit:ML/cell>: The water net yield data for different land uses.
    - yr_idx: The index of the year.

    Returns
    - w_mrj_effect <unit:ML/cell>: The updated water net yield data after applying ecological grazing effects.
    """

    land_uses = settings.AG_MANAGEMENTS_TO_LAND_USES['Ecological Grazing']
    lu_codes = np.array([data.DESC2AGLU[lu] for lu in land_uses])
    yr_cal = data.YR_CAL_BASE + yr_idx

    wreq_mrj = get_wreq_matrices(data, yr_idx)

    # Set up the effects matrix
    w_mrj_effect = np.zeros((data.NLMS, data.NCELLS, len(land_uses))).astype(np.float32)

    # Update values in the new matrix using the correct multiplier for each land use
    for lu_idx, lu in enumerate(land_uses):
        multiplier = data.ECOLOGICAL_GRAZING_DATA[lu].loc[yr_cal, "INPUT-wrt_water-required"]
        if multiplier != 1:
            j = lu_codes[lu_idx]
            # The effect is: new value = old value * multiplier - old value
            # E.g. a multiplier of .95 means a 5% reduction in water used.
            # Since the effect applies to water use, it effects the net yield negatively.
            w_mrj_effect[:, :, lu_idx] = wreq_mrj[:, :, j] * (1- multiplier)

    return w_mrj_effect


def get_savanna_burning_effect_w_mrj(data):
    """
    Applies the effects of using savanna burning to the water net yield data
    for all relevant agr. land uses.

    Savanna burning does not affect water usage, so return an array of zeros.

    Parameters
    - data: The input data object containing information about land uses and water net yield.

    Returns
    - An array of zeros with dimensions (NLMS, NCELLS, nlus), where:
        - NLMS: Number of land management systems
        - NCELLS: Number of cells
        - nlus: Number of land uses affected by savanna burning
    """

    nlus = len(settings.AG_MANAGEMENTS_TO_LAND_USES['Savanna Burning'])
    return np.zeros((data.NLMS, data.NCELLS, nlus)).astype(np.float32)


def get_agtech_ei_effect_w_mrj(data, yr_idx):
    """
    Applies the effects of using AgTech EI to the water net yield data
    for all relevant agr. land uses.

    Parameters
    - data: The data object containing relevant information.
    - w_mrj <unit:ML/cell>: The water net yield data for all land uses.
    - yr_idx: The index of the year.

    Returns
    - w_mrj_effect <unit:ML/cell>: The updated water net yield data with AgTech EI effects applied.
    """

    land_uses = settings.AG_MANAGEMENTS_TO_LAND_USES['AgTech EI']
    lu_codes = np.array([data.DESC2AGLU[lu] for lu in land_uses])
    yr_cal = data.YR_CAL_BASE + yr_idx

    wreq_mrj = get_wreq_matrices(data, yr_idx)

    # Set up the effects matrix
    w_mrj_effect = np.zeros((data.NLMS, data.NCELLS, len(land_uses))).astype(np.float32)

    # Update values in the new matrix using the correct multiplier for each LU
    for lu_idx, lu in enumerate(land_uses):
        multiplier = data.AGTECH_EI_DATA[lu].loc[yr_cal, "Water_use"]
        if multiplier != 1:
            j = lu_codes[lu_idx]
            # The effect is: new value = old value * multiplier - old value
            # E.g. a multiplier of .95 means a 5% reduction in water used.
            # Since the effect applies to water use, it effects the net yield negatively.
            w_mrj_effect[:, :, lu_idx] = wreq_mrj[:, :, j] * (1- multiplier)


    return w_mrj_effect


def get_biochar_effect_w_mrj(data, yr_idx):
    """
    Applies the effects of using Biochar to the water net yield data
    for all relevant agr. land uses.

    Parameters
    - data: The data object containing relevant information.
    - w_mrj <unit:ML/cell>: The water net yield data for all land uses.
    - yr_idx: The index of the year.

    Returns
    - w_mrj_effect <unit:ML/cell>: The updated water net yield data with Biochar applied.
    """

    land_uses = settings.AG_MANAGEMENTS_TO_LAND_USES['Biochar']
    lu_codes = np.array([data.DESC2AGLU[lu] for lu in land_uses])
    yr_cal = data.YR_CAL_BASE + yr_idx

    wreq_mrj = get_wreq_matrices(data, yr_idx)

    # Set up the effects matrix
    w_mrj_effect = np.zeros((data.NLMS, data.NCELLS, len(land_uses))).astype(np.float32)

    # Update values in the new matrix using the correct multiplier for each LU
    for lu_idx, lu in enumerate(land_uses):
        multiplier = data.BIOCHAR_DATA[lu].loc[yr_cal, "Water_use"]
        if multiplier != 1:
            j = lu_codes[lu_idx]
            w_mrj_effect[:, :, lu_idx] = wreq_mrj[:, :, j] * (1- multiplier)

    return w_mrj_effect


def get_beef_hir_effect_w_mrj(data, yr_idx):
    """
    Applies the effects of using HIR to the water net yield data
    for the natural beef land use.

    Parameters:
    - data: The data object containing relevant information.
    - w_mrj <unit:ML/cell>: The water net yield data for all land uses.
    - yr_idx: The index of the year.

    Returns:
    - w_mrj_effects <unit:ML/cell>: The updated water net yield data with Biochar applied.
    """
    land_uses = settings.AG_MANAGEMENTS_TO_LAND_USES['HIR - Beef']

    w_mrj_effects = np.zeros((data.NLMS, data.NCELLS, len(land_uses))).astype(np.float32)
    base_w_req_mrj = np.stack(( data.WREQ_DRY_RJ, data.WREQ_IRR_RJ ))

    for lu_idx, lu in enumerate(land_uses):
        j = data.DESC2AGLU[lu]

        multiplier = 1 - settings.HIR_PRODUCTIVITY_CONTRIBUTION

        # Reduce water requirements due to drop in yield potential (increase in net yield)
        if lu in data.LU_LVSTK:
            lvs, veg = lvs_veg_types(lu)

            w_mrj_effects[0, :, lu_idx] = (
                (multiplier - 1) * base_w_req_mrj[0, :, j] * get_yield_pot(data, lvs, veg, 'dry', yr_idx)
            )
            w_mrj_effects[1, :, lu_idx] = (
                (multiplier - 1) * base_w_req_mrj[1, :, j] * get_yield_pot(data, lvs, veg, 'irr', 0)
            )

    return w_mrj_effects


def get_sheep_hir_effect_w_mrj(data, yr_idx):
    """
    Applies the effects of using HIR to the water net yield data
    for the natural sheep land use.

    Parameters:
    - data: The data object containing relevant information.
    - w_mrj <unit:ML/cell>: The water net yield data for all land uses.
    - yr_idx: The index of the year.

    Returns:
    - w_mrj_effects <unit:ML/cell>: The updated water net yield data with Biochar applied.
    """
    land_uses = settings.AG_MANAGEMENTS_TO_LAND_USES['HIR - Sheep']

    w_mrj_effects = np.zeros((data.NLMS, data.NCELLS, len(land_uses))).astype(np.float32)
    base_w_req_mrj = np.stack(( data.WREQ_DRY_RJ, data.WREQ_IRR_RJ ))

    for lu_idx, lu in enumerate(land_uses):
        j = data.DESC2AGLU[lu]

        multiplier = 1 - settings.HIR_PRODUCTIVITY_CONTRIBUTION

        # Reduce water requirements due to drop in yield potential (increase in net yield)
        if lu in data.LU_LVSTK:
            lvs, veg = lvs_veg_types(lu)

            w_mrj_effects[0, :, lu_idx] = (
                multiplier * base_w_req_mrj[0, :, j] * get_yield_pot(data, lvs, veg, 'dry', yr_idx)
            )
            w_mrj_effects[1, :, lu_idx] = (
                multiplier * base_w_req_mrj[1, :, j] * get_yield_pot(data, lvs, veg, 'irr', 0)
            )

        # TODO - potential effects of increased water yield of HIR areas?

    return w_mrj_effects


def get_agricultural_management_water_matrices(data, yr_idx) -> dict[str, np.ndarray]:
    
    ag_mam_w_mrj ={}
    
    ag_mam_w_mrj['Asparagopsis taxiformis'] = get_asparagopsis_effect_w_mrj(data, yr_idx)           
    ag_mam_w_mrj['Precision Agriculture'] = get_precision_agriculture_effect_w_mrj(data, yr_idx)    
    ag_mam_w_mrj['Ecological Grazing'] = get_ecological_grazing_effect_w_mrj(data, yr_idx)          
    ag_mam_w_mrj['Savanna Burning'] = get_savanna_burning_effect_w_mrj(data)                        
    ag_mam_w_mrj['AgTech EI'] = get_agtech_ei_effect_w_mrj(data, yr_idx)                            
    ag_mam_w_mrj['Biochar'] = get_biochar_effect_w_mrj(data, yr_idx)                                
    ag_mam_w_mrj['HIR - Beef'] = get_beef_hir_effect_w_mrj(data, yr_idx)                            
    ag_mam_w_mrj['HIR - Sheep'] = get_sheep_hir_effect_w_mrj(data, yr_idx)                          

    return ag_mam_w_mrj


def get_water_outside_luto_study_area(data, yr_cal:int) ->  dict[int, float]:
    """
    Return water yield from the outside regions of LUTO study area.

    Parameters
        data (object): The data object containing the required data.
        yr_cal (int): The year for which the water yield is calculated.

    Returns
        dict[int, dict[int, float]]: <unit: ML/cell> dictionary of water yield amounts.
            The first key is year and the second key is region ID.
    """
    if settings.WATER_REGION_DEF == 'River Region':
        water_yield_arr = data.WATER_OUTSIDE_LUTO_RR

    elif settings.WATER_REGION_DEF == 'Drainage Division':
        water_yield_arr = data.WATER_OUTSIDE_LUTO_DD

    else:
        raise ValueError(
            f"Invalid value for setting WATER_REGION_DEF: '{settings.WATER_REGION_DEF}' "
            f"(must be either 'River Region' or 'Drainage Division')."
        )

    return water_yield_arr.loc[yr_cal].to_dict()


def get_water_outside_luto_study_area_from_hist_level(data) -> dict[int, float]:
    """
    Return water yield from the outside regions of LUTO study area based on historical levels.

    Parameters
        data (object): The data object containing the required data.

    Returns
        dict[int, float]: <unit: ML/cell> dictionary of water yield amounts.
    """
    if settings.WATER_REGION_DEF == 'River Region':
        water_yield_arr = data.WATER_OUTSIDE_LUTO_RR_HIST

    elif settings.WATER_REGION_DEF == 'Drainage Division':
        water_yield_arr = data.WATER_OUTSIDE_LUTO_DD_HIST

    else:
        raise ValueError(
            f"Invalid value for setting WATER_REGION_DEF: '{settings.WATER_REGION_DEF}' "
            f"(must be either 'River Region' or 'Drainage Division')."
        )

    return water_yield_arr



def calc_water_net_yield_inside_LUTO_BASE_YR_hist_water_lyr(data) -> dict[int, float]:
    """
    Calculate the water net yield for the base year (2010) for all regions.

    Parameters
    - data: The data object containing the necessary input data.

    Returns
    - dict[int, float]: A dictionary with the following structure:
        - key: region ID
        - value: water net yield for this region (ML)
    """
    if data.water_yield_regions_BASE_YR is not None:
        return data.water_yield_regions_BASE_YR
    
    # Get the water yield matrices
    w_mrj = get_water_net_yield_matrices(data, 0, data.WATER_YIELD_HIST_DR, data.WATER_YIELD_HIST_SR)
    
    # Get the ag decision variables for the base year
    ag_dvar_mrj = data.AG_L_MRJ

    # Multiply the water yield by the decision variables
    ag_w_r = np.einsum('mrj,mrj->r', w_mrj, ag_dvar_mrj)
    
    # Get water net yield for each region
    wny_inside_LUTO_regions = np.bincount(data.WATER_REGION_ID, ag_w_r)
    wny_inside_LUTO_regions = {region: wny for region, wny in enumerate(wny_inside_LUTO_regions) if region != 0}

    return wny_inside_LUTO_regions



def get_wny_for_watershed_regions_for_base_yr(data):
    """
    Return water net yield for watershed regions at the BASE_YR.

    Parameters
        data (object): The data object containing the required data.

    Returns
        dict[int, float]: A dictionary with the following structure:
        - key: region ID
        - value: water net yield for this region (ML)
    """
    wny_inside_mrj = get_water_net_yield_matrices(data, data.YR_CAL_BASE, data.WATER_YIELD_HIST_DR, data.WATER_YIELD_HIST_SR)
    wny_base_yr_inside_r = np.einsum('mrj,mrj->r', wny_inside_mrj, data.AG_L_MRJ)
    
    wny_base_yr_inside_regions = {k:v for k,v in enumerate(np.bincount(data.WATER_REGION_ID, wny_base_yr_inside_r))}
    wny_base_yr_outside_regions = get_water_outside_luto_study_area_from_hist_level(data)
    w_use_domestic_regions = data.WATER_USE_DOMESTIC
    
    wny_sum = {}
    for reg_id in wny_base_yr_outside_regions:
        wny_sum[reg_id] = (
            wny_base_yr_inside_regions[reg_id] +
            wny_base_yr_outside_regions[reg_id] -
            w_use_domestic_regions[reg_id]
        )
        
    return wny_sum


def get_water_net_yield_limit_for_regions_inside_LUTO(data):
    """
    Calculate the water net yield limit for each region based on historical levels.
    
    The limit is calculated as:
        - Water net yield limit for the whole region
        - Plus domestic water use
        - Minus water net yield from outside the LUTO study area
        
    Parameters
        data (object): The data object containing the required data.
        
    Returns
        dict[int, float]: A dictionary with the following structure:
        - key: region ID
        - value: water net yield limit for this region (ML)
    """
    
    wny_outside_LUTO_regions = get_water_outside_luto_study_area_from_hist_level(data)
    
    # Get the water net yield limit INSIDE LUTO study area
    wny_limit_stress = {
        reg_idx: (                            
            hist_level * settings.WATER_STRESS      # Water net yield limit for the whole region
            + data.WATER_USE_DOMESTIC[reg_idx]      # Domestic water use
            - wny_outside_LUTO_regions[reg_idx]     # Water net yield from outside the LUTO study area
        )
        for reg_idx, hist_level in data.WATER_REGION_HIST_LEVEL.items()
    }
    
    # Get the water net yield for the base year (2010)
    wny_limit_base_yr = calc_water_net_yield_inside_LUTO_BASE_YR_hist_water_lyr(data)
    
    # Update the water net yield limit for each region
    for reg_idx, limit in wny_limit_base_yr.items():
        if limit < wny_limit_stress[reg_idx]:       # If the base year water net yield is lower than the historical stress limit
            print(
                f"\t"
                f"Water net yield limit was relaxed to {limit:10.2f} ML (from {wny_limit_stress[reg_idx]:10.2f} ML) for {data.WATER_REGION_NAMES[reg_idx]}"
            )
            wny_limit_stress[reg_idx] = limit

    return wny_limit_stress
        
    


"""
Water logic *** a little outdated but maybe still useful ***

The limits are related to the pre-European inflows into rivers. As a proxy
for these inflows are used the flows that would result if all cells had
deeply-rooted vegetation. The values from 1985 are used for this as these
do not incorporate climate change corrections on rainfall. So the limit is
a _lower_ limit, it is a bottom, not a cap.

Performance relative to the cap is then composed of two parts:
    1. Water used for irrigation or as livestock drinking water, and
    2. Water retained in the soil by vegetation.
The former (1) is calculated using the water requirements (WR) data. This
water use effectively raises the lower limit, i.e. is added to it. The latter
is computed from the water yields data. The water yield data state the
inflows from each cell based on which type of vegetation (deeply or shallowly
rooted) and which SSP projection.

The first approach is to try to limit water stress to below 40% of the
pre-European inflows. This means the limit is to have _at least_ 60% of
the 1985 inflows if all cells had deeply rooted vegetation. If these 1985
inflows are called L, then inflows need to be >= .6L. Inflows are computed
using the water yield data based on the vegetation the simulation wants to
plant -- i.e. deeply or shallowly rooted, corresponding to trees and crops,
roughly. Subtracted from this is then the water use for irrigation. Since
plants do not fully use the irrigated water, some of the irrigation actually
also adds to the inflows. This fraction is the _complement_ of the irrigation
efficiency. So either the irrigation efficiency corrected water use is added
to the lower limit, or the complement of it (the irrigation running off) is
added to the inflow.
"""
