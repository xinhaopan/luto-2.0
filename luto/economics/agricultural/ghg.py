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
Pure functions to calculate greenhouse gas emissions by lm, lu.
"""

import itertools
import numpy as np
import pandas as pd

from luto.data import Data
from luto import settings
from luto.economics.agricultural.quantity import get_yield_pot, lvs_veg_types
from functools import lru_cache


def get_ghg_crop(data:Data, lu, lm, aggregate):
    """Return crop GHG emissions <unit: t/cell>  of `lu`+`lm` in `yr_idx` 
    as (np array|pd.DataFrame) depending on aggregate (True|False).

    Args:
        data (object/module): Data object or module. Assumes fields like in `luto.data`.
        lu (str): Land use (e.g. 'Winter cereals' or 'Beef - natural land').
        lm (str): Land management (e.g. 'dry', 'irr').
        aggregate (bool): True -> return GHG emission as np.array, False -> return GHG emission as pd.DataFrame.

    Returns
        np.array or pd.DataFrame: Crop GHG emissions <unit: t/cell>  of `lu`+`lm` in `yr_idx`.

    Crop GHG emissions include:
        - 'CO2E_KG_HA_CHEM_APPL'
        - 'CO2E_KG_HA_CROP_MGT'
        - 'CO2E_KG_HA_CULTIV'
        - 'CO2E_KG_HA_FERT_PROD'
        - 'CO2E_KG_HA_HARVEST'
        - 'CO2E_KG_HA_IRRIG'
        - 'CO2E_KG_HA_PEST_PROD'
        - 'CO2E_KG_HA_SOIL'
        - 'CO2E_KG_HA_SOWING'
    """
    
    # Process GHG_crop only if the land-use (lu) and land management (lm) combination exists (e.g., dryland Pears/Rice do not occur)
    if lu in data.AGGHG_CROPS['CO2E_KG_HA_CHEM_APPL', lm].columns:

        # Get the data column {ghg_rs: r -> each pixel,  s -> each GHG source}
        if settings.USE_GHG_SCOPE_1:
            ghg_rs = data.AGGHG_CROPS.loc[:, (data.AGGHG_CROPS.columns.get_level_values(0).isin(settings.CROP_GHG_SCOPE_1)) & 
                                             (data.AGGHG_CROPS.columns.get_level_values(1) == lm) & 
                                             (data.AGGHG_CROPS.columns.get_level_values(2) == lu)]
        else:
            ghg_rs = data.AGGHG_CROPS.loc[:, (slice(None), lm, lu)]

        # Convert kg CO2e per ha to tonnes. 
        ghg_rs /= 1000

        # Convert tonnes CO2 per ha to tonnes CO2 per cell including resfactor
        ghg_rs *= data.REAL_AREA[:, np.newaxis]

        # Convert to MultiIndex with levels [source, lm, lu]
        ghg_rs.columns = pd.MultiIndex.from_tuples([[col[0], lm, lu] for col in ghg_rs.columns])

        # Reset the dataframe index
        ghg_rs.reset_index(drop=True, inplace=True)

        # Return greenhouse gas emissions by individual source or summed over all sources (default)
        return ghg_rs if aggregate == False else ghg_rs.sum(axis=1).values



def get_ghg_lvstk( data:Data    # Data object.
                 , lu           # Land use.
                 , lm           # Land management.
                 , yr_idx       # Number of years post base-year ('YR_CAL_BASE').
                 , aggregate):  # GHG calculated as a total (for the solver) or by individual source (for writing outputs)
    """Return livestock GHG emissions <unit: t/cell>  of `lu`+`lm` in `yr_idx`
            as (np array|pd.DataFrame) depending on aggregate (True|False).

    `data`: data object/module -- assumes fields like in `luto.data`.
    `lu`: land use (e.g. 'Winter cereals' or 'Beef - natural land').
    `lm`: land management (e.g. 'dry', 'irr').
    `yr_idx`: number of years from base year, counting from zero.
    `aggregate`: True -> return GHG emission as np.array 
                 False -> return GHG emission as pd.DataFrame.
    
    Livestock GHG emissions include:    
                  'CO2E_KG_HEAD_DUNG_URINE',
                  'CO2E_KG_HEAD_ELEC',
                  'CO2E_KG_HEAD_ENTERIC',
                  'CO2E_KG_HEAD_FODDER',
                  'CO2E_KG_HEAD_FUEL',
                  'CO2E_KG_HEAD_IND_LEACH_RUNOFF',
                  'CO2E_KG_HEAD_MANURE_MGT',
                  'CO2E_KG_HEAD_SEED',
    """
    
    lvstype, vegtype = lvs_veg_types(lu)

    # Get the yield potential, i.e. the total number of livestock head per hectare.
    yield_pot = get_yield_pot(data, lvstype, vegtype, lm, yr_idx)

    # Get GHG emissions by source in kg CO2e per head of livestock.  settings.LVSTK_GHG_SCOPE_1
    # Note: ghg_rs (r -> each cell, s -> each GHG source)
    if settings.USE_GHG_SCOPE_1:
        ghg_raw = data.AGGHG_LVSTK.loc[:, (data.AGGHG_LVSTK.columns.get_level_values(0) == lvstype) &
                                          (data.AGGHG_LVSTK.columns.get_level_values(1).isin(settings.LVSTK_GHG_SCOPE_1))]
    else:
        ghg_raw = data.AGGHG_LVSTK.loc[:, (lvstype, slice(None)) ]

    # Get the names for each GHG source
    ghg_name_s = [ i[1] for i in ghg_raw.columns ]

    # Calculate the GHG emissions (kgCO2/head * head/ha = kgCO/ha)
    ghg_rs = ghg_raw * yield_pot[:,np.newaxis]


    # Add pasture irrigation emissions.
    if lm == 'irr':
        ghg_lvstk_irr = data.AGGHG_IRRPAST
        ghg_lvstk_irr_cols = [i for i in ghg_lvstk_irr.columns if 'CO2E' in i]
        
        ghg_rs = pd.concat([ghg_rs, ghg_lvstk_irr[ghg_lvstk_irr_cols]], axis = 1)
        ghg_name_s += ghg_lvstk_irr_cols
        

    # Convert to tonnes of CO2e per ha. 
    ghg_rs = ghg_rs / 1000

    # Convert to tonnes CO2e per cell including resfactor
    ghg_rs *= data.REAL_AREA[:, np.newaxis]

    # Convert to MultiIndex with levels [source, lm, lu]
    ghg_rs = pd.DataFrame(ghg_rs)
    ghg_rs.columns = pd.MultiIndex.from_tuples( [(ghg, lm, lu) for ghg in ghg_name_s ])

    # Reset dataframe index
    ghg_rs.reset_index(drop = True, inplace = True)
    
    # Return the full dataframe if Aggregate == False otherwise return the sum over all GHG sources
    return ghg_rs if aggregate == False else ghg_rs.sum(axis = 1).values
       


def get_ghg(data:Data, lu, lm, yr_idx, aggregate):
    """Return GHG emissions [tCO2e/cell] of `lu`+`lm` in `yr_idx` 
    as (np array|pd.DataFrame) depending on aggregate (True|False).

    Args:
        data (object/module): Data object or module. Assumes fields like in `luto.data`.
        lu (str): Land use (e.g. 'Winter cereals').
        lm (str): Land management (e.g. 'dry', 'irr').
        yr_idx (int): Number of years from base year, counting from zero.
        aggregate (bool): True -> return GHG emission as np.array, False -> return GHG emission as pd.DataFrame.

    Returns
        np.array or pd.DataFrame: GHG emissions [tCO2e/cell] of `lu`+`lm` in `yr_idx`.

    Raises:
        KeyError: If land use `lu` is not found in `data.LANDUSES`.
    """

    # If it is a crop, it is known how to get GHG emissions.
    if lu in data.LU_CROPS:
        return get_ghg_crop(data, lu, lm, aggregate)
    elif lu in data.LU_LVSTK:
        return get_ghg_lvstk(data, lu, lm, yr_idx, aggregate)
    elif lu in data.AGRICULTURAL_LANDUSES:
        if aggregate:
            return np.zeros(data.NCELLS, dtype=np.float32)
        else:
            return pd.DataFrame({('CO2E_KG_HA_CHEM_APPL', lm, lu): np.zeros(data.NCELLS, dtype=np.float32)})
    else:
        raise KeyError(f"Land use '{lu}' not found in data.LANDUSES")



def get_ghg_matrix(data:Data, lm, yr_idx, aggregate):
    """
    Return g_rj matrix <unit: t/cell> per lu under `lm` in `yr_idx`.

    Parameters
    - data: The data object containing the necessary information.
    - lm: The land use model.
    - yr_idx: The index of the year.
    - aggregate: A boolean indicating whether to aggregate the results or not.

    Returns
    - If `aggregate` is True, returns a numpy array of shape (NCELLS, len(data.AGRICULTURAL_LANDUSES)).
    - If `aggregate` is False, returns a pandas DataFrame with columns corresponding to each agricultural land use.

    """
    if aggregate == True: 
        g_rj = np.zeros((data.NCELLS, len(data.AGRICULTURAL_LANDUSES)), dtype=np.float32)
        for j, lu in enumerate(data.AGRICULTURAL_LANDUSES):
            g_rj[:, j] = get_ghg(data, lu, lm, yr_idx, aggregate)
            
        # Make sure all NaNs are replaced by zeroes.
        g_rj = np.nan_to_num(g_rj)
    
        return g_rj
    
    elif aggregate == False:     
        return pd.concat([get_ghg(data, lu, lm, yr_idx, aggregate) 
                          for lu in data.AGRICULTURAL_LANDUSES],axis=1)
        


@lru_cache(maxsize=1)
def get_ghg_matrices(data:Data, yr_idx, aggregate=True):
    """
    Return g_mrj matrix <unit: t/cell> as 3D Numpy array.
    
    Parameters
        data (object): The data object containing the necessary information.
        yr_idx (int): The index of the year.
        aggregate (bool, optional): Whether to aggregate the results. Defaults to True.
    
    Returns
        numpy.ndarray or pandas.DataFrame: The GHG emissions matrix as a 3D Numpy array if aggregate is True,
        or as a pandas DataFrame if aggregate is False.
    """
    
    if aggregate == True:  
        return np.stack(
            tuple(
                get_ghg_matrix(data, lm, yr_idx, aggregate)
                for lm in data.LANDMANS
            )
        )
    elif aggregate == False:
        ghg_df = pd.concat([get_ghg_matrix(data, lu, yr_idx, aggregate) for lu in data.LANDMANS], axis=1).replace('CO2E_KG_HA','TCO2E')
        column_rename = [(i[0].replace('CO2E_KG_HA','TCO2E'),i[1],i[2]) for i in ghg_df.columns]
        column_rename = [(i[0].replace('CO2E_KG_HEAD','TCO2E'),i[1],i[2]) for i in column_rename]
        ghg_df.columns = pd.MultiIndex.from_tuples(column_rename)
        return ghg_df


# ---------------------------------------------------------------------------
# GHG transition: unallocated natural → livestock natural
# ---------------------------------------------------------------------------

def get_ghg_unall_natural_to_lvstk_natural_crisp(data: Data, base_lumap) -> np.ndarray:
    """GHG penalties for unall-natural→lvstk-natural using dominant lumap cell assignment."""
    ghg_rj = np.zeros((data.NCELLS, data.N_AG_LUS), dtype=np.float32)
    un_allow_code = data.DESC2AGLU["Unallocated - natural land"]
    cells = base_lumap == un_allow_code

    for to_lu in data.LU_LVSTK_NATURAL:
        ghg_rj[cells, to_lu] = (
            data.CO2E_STOCK_UNALL_NATURAL_TCO2_HA_PER_YR[cells]
            * (1 - data.BIO_HABITAT_CONTRIBUTION_LOOK_UP[to_lu])
            * data.REAL_AREA[cells]
        )

    return np.stack([ghg_rj] * data.NLMS)


def get_ghg_unall_natural_to_lvstk_natural_blend(data: Data, ag_X_mrj: np.ndarray) -> np.ndarray:
    """GHG penalties for unall-natural→lvstk-natural weighted by fractional dvar composition."""
    threshold = 10 ** (-settings.ROUND_DECIMALS)
    un_allow_code = data.DESC2AGLU["Unallocated - natural land"]
    all_from_lumap = np.full(data.NCELLS, un_allow_code, dtype=np.int8)
    penalty_mrj = get_ghg_unall_natural_to_lvstk_natural_crisp(data, all_from_lumap)

    frac_r = ag_X_mrj[:, :, un_allow_code].sum(axis=0)
    frac_r[frac_r < settings.FEASIBILITY_TOLERANCE] = 0.0
    frac_r_safe = np.where(frac_r > threshold, frac_r, 1.0)

    return penalty_mrj * frac_r[np.newaxis, :, np.newaxis] / frac_r_safe[np.newaxis, :, np.newaxis]


def get_ghg_unall_natural_to_lvstk_natural_exact(data: Data, base_year: int, from_m: int, from_j: int) -> np.ndarray:
    """GHG penalties for unall-natural→lvstk-natural using exact cell selection per (from_m, from_j)."""
    threshold = settings.EXACT_REACHABILITY_MIN_FRACTION  # MUST match get_base_dvar_mj_cell_map
    cell_idx = np.where(data.ag_dvars[base_year][from_m, :, from_j] > threshold)[0]
    result_arr = np.zeros((data.NLMS, len(cell_idx), data.N_AG_LUS), dtype=np.float32)

    if from_j != data.DESC2AGLU["Unallocated - natural land"]:
        return result_arr

    for to_j in data.LU_LVSTK_NATURAL:
        penalty = (
            data.CO2E_STOCK_UNALL_NATURAL_TCO2_HA_PER_YR[cell_idx]
            * (1 - data.BIO_HABITAT_CONTRIBUTION_LOOK_UP[to_j])
            * data.REAL_AREA[cell_idx]
        )
        result_arr[:, :, to_j] = penalty[np.newaxis, :]

    return result_arr


def get_ghg_unall_natural_to_lvstk_natural(data: Data, base_lumap, ag_X_mrj=None) -> np.ndarray:
    """Dispatch to _crisp (crisp / no dvar) or _blend (blend & exact GHG account).

    The transition-GHG ACCOUNT is target-indexed (added to the ongoing GHG and applied to X_new in the
    GHG constraint), so blend AND exact both use the base-year-composition frac-weighting. exact's
    per-source GHG is carried by the transition COST (get_ghg_*_exact via get_transition_matrices_ag2ag_exact).
    """
    if settings.TRANSITION_MODE == 'crisp' or ag_X_mrj is None:
        return get_ghg_unall_natural_to_lvstk_natural_crisp(data, base_lumap)
    return get_ghg_unall_natural_to_lvstk_natural_blend(data, ag_X_mrj)


# ---------------------------------------------------------------------------
# GHG transition: livestock natural → modified
# ---------------------------------------------------------------------------

def get_ghg_lvstk_natural_to_modified_crisp(data: Data, base_lumap) -> np.ndarray:
    """GHG penalties for lvstk-natural→modified transitions using the dominant lumap cell assignment."""
    ghg_rj = np.zeros((data.NCELLS, data.N_AG_LUS), dtype=np.float32)

    for from_lu, to_lu in itertools.product(data.LU_LVSTK_NATURAL, data.LU_MODIFIED_LAND):
        cells = base_lumap == from_lu
        ghg_rj[cells, to_lu] = (
            data.CO2E_STOCK_UNALL_NATURAL_TCO2_HA_PER_YR[cells]
            * (1 - data.BIO_HABITAT_CONTRIBUTION_LOOK_UP[to_lu])
            * data.REAL_AREA[cells]
        )

    return np.stack([ghg_rj] * data.NLMS)


def get_ghg_lvstk_natural_to_modified_blend(data: Data, ag_X_mrj: np.ndarray) -> np.ndarray:
    """GHG penalties for lvstk-natural→modified transitions weighted by fractional dvar composition."""
    threshold = 10 ** (-settings.ROUND_DECIMALS)
    result_mrj    = np.zeros((data.NLMS, data.NCELLS, data.N_AG_LUS), dtype=np.float32)
    total_frac_r  = np.zeros(data.NCELLS, dtype=np.float32)

    for from_lu in data.LU_LVSTK_NATURAL:
        all_from_lumap = np.full(data.NCELLS, from_lu, dtype=np.int8)
        penalty_mrj = get_ghg_lvstk_natural_to_modified_crisp(data, all_from_lumap)

        frac_r = ag_X_mrj[:, :, from_lu].sum(axis=0)
        frac_r[frac_r < settings.FEASIBILITY_TOLERANCE] = 0.0

        result_mrj   += penalty_mrj * frac_r[np.newaxis, :, np.newaxis]
        total_frac_r += frac_r

    total_frac_r_safe = np.where(total_frac_r > threshold, total_frac_r, 1.0)
    return result_mrj / total_frac_r_safe[np.newaxis, :, np.newaxis]


def get_ghg_lvstk_natural_to_modified_exact(data: Data, base_year: int, from_m: int, from_j: int) -> np.ndarray:
    """GHG penalties for lvstk-natural→modified using exact cell selection per (from_m, from_j)."""
    threshold = settings.EXACT_REACHABILITY_MIN_FRACTION  # MUST match get_base_dvar_mj_cell_map
    cell_idx = np.where(data.ag_dvars[base_year][from_m, :, from_j] > threshold)[0]
    result_arr = np.zeros((data.NLMS, len(cell_idx), data.N_AG_LUS), dtype=np.float32)

    if from_j not in data.LU_LVSTK_NATURAL:
        return result_arr

    for to_j in data.LU_MODIFIED_LAND:
        result_arr[:, :, to_j] = (
            data.CO2E_STOCK_UNALL_NATURAL_TCO2_HA_PER_YR[cell_idx]
            * (1 - data.BIO_HABITAT_CONTRIBUTION_LOOK_UP[to_j])
            * data.REAL_AREA[cell_idx]
        )[np.newaxis, :]

    return result_arr


def get_ghg_lvstk_natural_to_modified(data: Data, base_lumap, ag_X_mrj=None) -> np.ndarray:
    """Dispatch to _crisp (crisp / no dvar) or _blend (blend & exact GHG account). See
    get_ghg_unall_natural_to_lvstk_natural for why exact shares the blend (frac-weighted) account."""
    if settings.TRANSITION_MODE == 'crisp' or ag_X_mrj is None:
        return get_ghg_lvstk_natural_to_modified_crisp(data, base_lumap)
    return get_ghg_lvstk_natural_to_modified_blend(data, ag_X_mrj)


# ---------------------------------------------------------------------------
# GHG transition: unallocated natural → modified
# ---------------------------------------------------------------------------

def get_ghg_unall_natural_to_modified_crisp(data: Data, base_lumap) -> np.ndarray:
    """GHG penalties for unall-natural→modified using dominant lumap cell assignment."""
    ghg_rj = np.zeros((data.NCELLS, data.N_AG_LUS), dtype=np.float32)
    un_allow_code = data.DESC2AGLU["Unallocated - natural land"]
    cells = base_lumap == un_allow_code

    for to_lu in data.LU_MODIFIED_LAND:
        ghg_rj[cells, to_lu] = (
            data.CO2E_STOCK_UNALL_NATURAL_TCO2_HA_PER_YR[cells]
            * (1 - data.BIO_HABITAT_CONTRIBUTION_LOOK_UP[to_lu])
            * data.REAL_AREA[cells]
        )

    return np.stack([ghg_rj] * data.NLMS)


def get_ghg_unall_natural_to_modified_blend(data: Data, ag_X_mrj: np.ndarray) -> np.ndarray:
    """GHG penalties for unall-natural→modified weighted by fractional dvar composition."""
    threshold = 10 ** (-settings.ROUND_DECIMALS)
    un_allow_code = data.DESC2AGLU["Unallocated - natural land"]
    all_from_lumap = np.full(data.NCELLS, un_allow_code, dtype=np.int8)
    penalty_mrj = get_ghg_unall_natural_to_modified_crisp(data, all_from_lumap)

    frac_r = ag_X_mrj[:, :, un_allow_code].sum(axis=0)
    frac_r[frac_r < settings.FEASIBILITY_TOLERANCE] = 0.0
    frac_r_safe = np.where(frac_r > threshold, frac_r, 1.0)

    return penalty_mrj * frac_r[np.newaxis, :, np.newaxis] / frac_r_safe[np.newaxis, :, np.newaxis]


def get_ghg_unall_natural_to_modified_exact(data: Data, base_year: int, from_m: int, from_j: int) -> np.ndarray:
    """GHG penalties for unall-natural→modified using exact cell selection per (from_m, from_j)."""
    threshold = settings.EXACT_REACHABILITY_MIN_FRACTION  # MUST match get_base_dvar_mj_cell_map
    cell_idx = np.where(data.ag_dvars[base_year][from_m, :, from_j] > threshold)[0]
    result_arr = np.zeros((data.NLMS, len(cell_idx), data.N_AG_LUS), dtype=np.float32)

    if from_j != data.DESC2AGLU["Unallocated - natural land"]:
        return result_arr

    for to_j in data.LU_MODIFIED_LAND:
        result_arr[:, :, to_j] = (
            data.CO2E_STOCK_UNALL_NATURAL_TCO2_HA_PER_YR[cell_idx]
            * (1 - data.BIO_HABITAT_CONTRIBUTION_LOOK_UP[to_j])
            * data.REAL_AREA[cell_idx]
        )[np.newaxis, :]

    return result_arr


def get_ghg_unall_natural_to_modified(data: Data, base_lumap, ag_X_mrj=None) -> np.ndarray:
    """Dispatch to _crisp (crisp / no dvar) or _blend (blend & exact GHG account). See
    get_ghg_unall_natural_to_lvstk_natural for why exact shares the blend (frac-weighted) account."""
    if settings.TRANSITION_MODE == 'crisp' or ag_X_mrj is None:
        return get_ghg_unall_natural_to_modified_crisp(data, base_lumap)
    return get_ghg_unall_natural_to_modified_blend(data, ag_X_mrj)


def get_ghg_transition_emissions(data:Data, base_lumap, separate=False, cells=None, ag_X_mrj=None) -> np.ndarray:
    """
    Get the one-off greenhouse gas penalties for transitioning between land uses.

    Parameters
    ----------
      data (object): The data object containing relevant information.
      base_lumap (np.ndarray): The base_lumap object containing land use mapping.
      separate (bool): Whether to return the penalties for each transition separately.
      cells (np.ndarray, optional): The cells for which to calculate penalties.
      ag_X_mrj (np.ndarray, optional): Fractional ag dvar array (nlms, ncells, n_ag_lus);
          required when TRANSITION_MODE != 'crisp'.

    Returns
    -------
      GHG penalties (dict[np.ndarray]|np.ndarray): The greenhouse gas transition penalties.
    """

    ghg_lvstck_natural_to_unall_natural = np.zeros_like(data.AG_L_MRJ, dtype=np.float32)   # No land can transited to unall-natural, here use a full zero array for consistency
    ghg_lvstck_natural_to_modified = get_ghg_lvstk_natural_to_modified(data, base_lumap, ag_X_mrj)
    ghg_unall_natural_to_lvstck_natural = get_ghg_unall_natural_to_lvstk_natural(data, base_lumap, ag_X_mrj)
    ghg_unall_natural_to_modified = get_ghg_unall_natural_to_modified(data, base_lumap, ag_X_mrj)
    
    if separate:
        ghg_trainsition_penalties = {
            'Livestock natural to unallocated natural': ghg_lvstck_natural_to_unall_natural,
            'Unallocated natural to livestock natural': ghg_unall_natural_to_lvstck_natural,
            'Livestock natural to modified': ghg_lvstck_natural_to_modified,
            'Unallocated natural to modified': ghg_unall_natural_to_modified
        }
    else:
        ghg_trainsition_penalties = ghg_lvstck_natural_to_unall_natural \
            + ghg_unall_natural_to_lvstck_natural \
            + ghg_lvstck_natural_to_modified \
            + ghg_unall_natural_to_modified \
    
    return ghg_trainsition_penalties
    
    


def get_ghg_transition_emissions_from_base_year(data: Data, base_year: int) -> dict:
    """Source-keyed ag2ag transition GHG EMISSIONS (t CO2/cell): dict[(from_m, from_j)] -> arr[to_m, cells, to_j].

    The physical-emissions parallel of get_transition_matrices_ag2ag_from_base_year (the $ transition
    COST). It uses the SAME per-mode source-keying and weighting as the cost, so the two share the flow
    var F in Phase 3: the GHG constraint will sum `Σ flow_ghg·F` (replacing the target-based
    ag_ghg_t_mrj / get_ag_ghg_t_mrj) exactly as the objective sums `Σ flow_cost·F`. Emissions are RAW
    t CO2 — no carbon price, no amortise (that priced+amortised version is the GHG component of the
    transition cost). Built alongside flow_cost; NOT yet consumed by the solver (wired in Phase 3).

    Modes (mirroring the cost):
      crisp — slice the dominant-source flat emissions by (from_m, from_j).
      blend — per present source: uniform-source emissions weighted by frac_s / eligible_rj (the SAME
              factor as the blend transition cost, so cost and GHG share the flow var consistently).
      exact — sum the three per-source get_ghg_*_exact components on the source's θ cells.
    """
    if settings.TRANSITION_MODE == 'crisp':
        return ghg_transition_emissions_ag2ag_crisp(data, base_year)
    elif settings.TRANSITION_MODE == 'blend':
        return ghg_transition_emissions_ag2ag_blend(data, base_year)
    elif settings.TRANSITION_MODE == 'exact':
        return ghg_transition_emissions_ag2ag_exact(data, base_year)
    raise ValueError(f"Unknown TRANSITION_MODE {settings.TRANSITION_MODE!r}")


def ghg_transition_emissions_ag2ag_crisp(data: Data, base_year: int) -> dict:
    """Crisp: slice the dominant-source flat emissions (each ag cell has one dominant source)."""
    lumap = data.lumaps[base_year]
    lmmap = data.lmmaps[base_year]
    flat = get_ghg_transition_emissions(data, lumap)                    # (NLMS, NCELLS, N_AG_LUS) raw t CO2
    result = {}
    for fm in range(data.NLMS):
        for fj in range(data.N_AG_LUS):
            cells = np.where((lmmap == fm) & (lumap == fj))[0]
            if cells.size:
                result[(fm, fj)] = flat[:, cells, :].astype(np.float32)
    return result


def ghg_transition_emissions_ag2ag_exact(data: Data, base_year: int) -> dict:
    """Exact: per-source physical emissions (the three components), sliced to the source's θ cells.
    Same cell set as get_base_dvar_mj_cell_map / the exact transition cost (all use
    EXACT_REACHABILITY_MIN_FRACTION)."""
    threshold = settings.EXACT_REACHABILITY_MIN_FRACTION
    ag_X = data.ag_dvars[base_year]
    result = {}
    for fm in range(data.NLMS):
        for fj in range(data.N_AG_LUS):
            cells = np.where(ag_X[fm, :, fj] > threshold)[0]
            if cells.size == 0:
                continue
            result[(fm, fj)] = (
                get_ghg_lvstk_natural_to_modified_exact(data, base_year, fm, fj)
                + get_ghg_unall_natural_to_lvstk_natural_exact(data, base_year, fm, fj)
                + get_ghg_unall_natural_to_modified_exact(data, base_year, fm, fj)
            ).astype(np.float32)
    return result


def ghg_transition_emissions_ag2ag_blend(data: Data, base_year: int) -> dict:
    """Blend: per present source, the uniform-source emissions weighted by frac_s / eligible_rj —
    the SAME eligible-target normaliser as the blend transition cost, so GHG and cost share one factor."""
    ag_X = data.ag_dvars[base_year]
    thr = 10 ** (-settings.ROUND_DECIMALS)
    t_ij = data.T_MAT.sel(from_lu=data.AGRICULTURAL_LANDUSES, to_lu=data.AGRICULTURAL_LANDUSES).values
    valid = ~np.isnan(t_ij)                                             # (N_AG_LUS_from, N_AG_LUS_to)
    eligible_rj = ag_X.sum(axis=0) @ valid                             # (NCELLS, N_AG_LUS)
    eligible_safe = np.where(eligible_rj > thr, eligible_rj, 1.0)
    result = {}
    for fm in range(data.NLMS):
        for fj in range(data.N_AG_LUS):
            cells = np.where(ag_X[fm, :, fj] > thr)[0]
            if cells.size == 0:
                continue
            uniform_lumap = np.full(data.NCELLS, fj, dtype=np.int64)   # emissions from an all-fj source
            base_ghg = get_ghg_transition_emissions(data, uniform_lumap)   # ag_X_mrj=None -> _crisp path
            factor = ag_X[fm, cells, fj][:, None] / eligible_safe[cells, :]
            result[(fm, fj)] = (base_ghg[:, cells, :] * factor[np.newaxis, :, :]).astype(np.float32)
    return result


def get_asparagopsis_effect_g_mrj(data:Data, yr_idx):
    """
    Applies the effects of using asparagopsis to the GHG data
    for all relevant agricultural land uses.

    Parameters
    - data: The input data containing GHG and land use information.
    - yr_idx: The index of the year to calculate the effects for.

    Returns
    - new_g_mrj: The matrix <unit: t/cell> containing the updated GHG data with the effects of using asparagopsis.

    Note: This function relies on other helper functions such as lvs_veg_types and get_yield_pot to calculate
    the reduction amount for each land use and management type.
    """
    land_uses = settings.AG_MANAGEMENTS_TO_LAND_USES["Asparagopsis taxiformis"]
    yr_cal = data.YR_CAL_BASE + yr_idx

    # Set up the effects matrix
    new_g_mrj = np.zeros((data.NLMS, data.NCELLS, len(land_uses)), dtype=np.float32)

    if not settings.AG_MANAGEMENTS['Asparagopsis taxiformis']:
        return new_g_mrj

    # Update values in the new matrix, taking into account the CH4 reduction of asparagopsis
    for lu_idx, lu in enumerate(land_uses):
        ch4_reduction_perc = 1 - data.ASPARAGOPSIS_DATA[lu].loc[yr_cal, "CO2E_KG_HEAD_ENTERIC"]

        if ch4_reduction_perc != 0:
            for lm in data.LANDMANS:
                m = 0 if lm == 'irr' else 1
                # Subtract enteric fermentation emissions multiplied by reduction multiplier
                lvstype, vegtype = lvs_veg_types(lu)

                yield_pot = get_yield_pot(data, lvstype, vegtype, lm, yr_idx)

                reduction_amnt = (
                    data.AGGHG_LVSTK[lvstype, "CO2E_KG_HEAD_ENTERIC"].to_numpy()
                    * yield_pot
                    * ch4_reduction_perc
                    / 1000            # convert to tonnes
                    * data.REAL_AREA  # adjust for resfactor
                )
                new_g_mrj[m, :, lu_idx] = -reduction_amnt

    return new_g_mrj


def get_precision_agriculture_effect_g_mrj(data:Data, yr_idx):
    """
    Applies the effects of using precision agriculture to the GHG data
    for all relevant agr. land uses.

    Parameters
    - data: The input data containing the necessary information.
    - yr_idx: The index of the year to calculate the effects for.

    Returns
    - new_g_mrj: The matrix <unit: t/cell> containing the updated GHG data after applying the effects of precision agriculture.
    """

    land_uses = settings.AG_MANAGEMENTS_TO_LAND_USES['Precision Agriculture']
    yr_cal = data.YR_CAL_BASE + yr_idx

    # Set up the effects matrix
    new_g_mrj = np.zeros((data.NLMS, data.NCELLS, len(land_uses))).astype(np.float32)

    if not settings.AG_MANAGEMENTS['Precision Agriculture']:
        return new_g_mrj

    # Update values in the new matrix
    for lu_idx, lu in enumerate(land_uses):
        lu_data = data.PRECISION_AGRICULTURE_DATA[settings.LU2TYPE[lu]]

        for lm in data.LANDMANS:
            m = 0 if lm == 'dry' else 1
            for co2e_type in [
                'CO2E_KG_HA_CHEM_APPL',
                'CO2E_KG_HA_CROP_MGT',
                'CO2E_KG_HA_PEST_PROD',
                'CO2E_KG_HA_SOIL',
            ]:
                # Check if land-use/land management combination exists (e.g., dryland Pears/Rice do not occur), if not use zeros
                if lu not in data.AGGHG_CROPS[data.AGGHG_CROPS.columns[0][0], lm].columns:
                    continue

                reduction_perc = 1 - lu_data.loc[yr_cal, co2e_type]

                if reduction_perc != 0:
                    reduction_amnt = (
                        np.nan_to_num(data.AGGHG_CROPS[co2e_type, lm, lu].to_numpy().copy(), 0)
                        * reduction_perc
                        / 1000            # convert to tonnes
                        * data.REAL_AREA  # adjust for resfactor
                    )
                    new_g_mrj[m, :, lu_idx] -= reduction_amnt

    if np.isnan(new_g_mrj).any():
        raise ValueError("Error in data: NaNs detected in agricultural management options' GHG effect matrix.")

    return new_g_mrj


def get_ecological_grazing_effect_g_mrj(data:Data, yr_idx):
    """
    Applies the effects of using ecological grazing to the GHG data
    for all relevant agricultural land uses.

    Parameters
    - data: The input data containing relevant information for calculations.
    - yr_idx: The index of the year for which the calculations are performed.

    Returns
    - new_g_mrj: The matrix <unit: t/cell> containing the updated GHG data after applying ecological grazing effects.
    """

    land_uses = settings.AG_MANAGEMENTS_TO_LAND_USES['Ecological Grazing']
    yr_cal = data.YR_CAL_BASE + yr_idx

    # Set up the effects matrix
    new_g_mrj = np.zeros((data.NLMS, data.NCELLS, len(land_uses))).astype(np.float32)

    if not settings.AG_MANAGEMENTS['Ecological Grazing']:
        return new_g_mrj

    # Update values in the new matrix
    for lu_idx, lu in enumerate(land_uses):
        lu_data = data.ECOLOGICAL_GRAZING_DATA[lu]

        for lm in data.LANDMANS:
            m = 0 if lm == 'dry' else 1
            # Subtract leach runoff carbon benefit
            leach_reduction_perc = 1 - lu_data.loc[yr_cal, 'CO2E_KG_HEAD_IND_LEACH_RUNOFF']
            if leach_reduction_perc != 0:
                lvstype, vegtype = lvs_veg_types(lu)
                yield_pot = get_yield_pot(data, lvstype, vegtype, lm, yr_idx)

                leach_reduction_amnt = (
                    data.AGGHG_LVSTK[lvstype, 'CO2E_KG_HEAD_IND_LEACH_RUNOFF'].to_numpy()
                    * yield_pot       # convert to HAs
                    * leach_reduction_perc
                    / 1000            # convert to tonnes
                    * data.REAL_AREA  # adjust for resfactor
                )
                new_g_mrj[m, :, lu_idx] -= leach_reduction_amnt

            # Subtract soil carbon benefit
            soil_multiplier = lu_data.loc[yr_cal, 'IMPACTS_soil_carbon'] - 1
            if soil_multiplier != 0:
                soil_reduction_amnt = (
                    data.SOIL_CARBON_AVG_T_CO2_HA_PER_YR
                    * soil_multiplier
                    * data.REAL_AREA 
                )
                new_g_mrj[m, :, lu_idx] -= soil_reduction_amnt

    return new_g_mrj


def get_savanna_burning_effect_g_mrj(data:Data):
    """
    Applies the effects of using savanna burning to the GHG data
    for all relevant agr. land uses.

    Parameters
    - data: The input data containing relevant information.
    - g_mrj: The savanna burning factor.

    Returns
    - sb_g_mrj: The GHG data <unit: t/cell> with the effects of savanna burning applied.
    """
    nlus = len(settings.AG_MANAGEMENTS_TO_LAND_USES["Savanna Burning"])
    sb_g_mrj = np.zeros((data.NLMS, data.NCELLS, nlus)).astype(np.float32)

    if not settings.AG_MANAGEMENTS['Savanna Burning']:
        return sb_g_mrj

    for m, j in itertools.product(range(data.NLMS), range(nlus)):
        # sb_g_mrj[m, :, j] = -data.SAVBURN_TOTAL_TCO2E_HA * data.REAL_AREA
        sb_g_mrj[m, :, j] = np.where( data.SAVBURN_ELIGIBLE, 
                                     -data.SAVBURN_TOTAL_TCO2E_HA * data.REAL_AREA, 
                                      0
                                    )
    return sb_g_mrj


def get_agtech_ei_effect_g_mrj(data:Data, yr_idx):
    """
    Applies the effects of using AgTech EI to the GHG data
    for all relevant agr. land uses.

    Parameters
    - data: The input data containing the necessary information.
    - yr_idx: The index of the year to calculate the effects for.

    Returns
    - new_g_mrj: The matrix <unit: t/cell> containing the updated GHG data after applying the AgTech EI effects.
    """
    land_uses = settings.AG_MANAGEMENTS_TO_LAND_USES['AgTech EI']
    yr_cal = data.YR_CAL_BASE + yr_idx

    # Set up the effects matrix
    new_g_mrj = np.zeros((data.NLMS, data.NCELLS, len(land_uses))).astype(np.float32)

    if not settings.AG_MANAGEMENTS['AgTech EI']:
        return new_g_mrj

    # Update values in the new matrix
    for lu_idx, lu in enumerate(land_uses):
        lu_data = data.AGTECH_EI_DATA[settings.LU2TYPE[lu]]

        for lm in data.LANDMANS:
            m = 0 if lm == 'dry' else 1
            for co2e_type in [
                'CO2E_KG_HA_CHEM_APPL',
                'CO2E_KG_HA_CROP_MGT',
                'CO2E_KG_HA_PEST_PROD',
                'CO2E_KG_HA_SOIL'
            ]:    
                # Check if land-use/land management combination exists (e.g., dryland Pears/Rice do not occur), if not use zeros
                if lu not in data.AGGHG_CROPS[data.AGGHG_CROPS.columns[0][0], lm].columns:
                    continue

                reduction_perc = 1 - lu_data.loc[yr_cal, co2e_type]

                if reduction_perc != 0:
                    reduction_amnt = (
                        np.nan_to_num(data.AGGHG_CROPS[co2e_type, lm, lu].to_numpy().copy(), 0) 
                        * reduction_perc
                        / 1000            # convert to tonnes
                        * data.REAL_AREA  # adjust for resfactor
                    )
                    new_g_mrj[m, :, lu_idx] -= reduction_amnt

            # Subtract extra 'CO2e_KG_HA_IRRIG' carbon for irrigated land uses
            if m == 1:
                if lu not in data.AGGHG_CROPS[data.AGGHG_CROPS.columns[0][0], lm].columns:
                    continue

                # Columns names for irrig. CO2e are inconsistent across sheets
                irrig_co2e_col = 'CO2e_KG_HA_IRRIG'
                if 'CO2E_KG_HA_IRRIG' in lu_data.columns:
                    irrig_co2e_col = 'CO2E_KG_HA_IRRIG'

                reduction_perc = 1 - lu_data.loc[yr_cal, irrig_co2e_col]

                if reduction_perc != 0:
                    reduction_amnt = (
                        np.nan_to_num(data.AGGHG_CROPS['CO2E_KG_HA_IRRIG', lm, lu].copy().to_numpy(), 0) 
                        * reduction_perc
                        / 1000            # convert to tonnes
                        * data.REAL_AREA  # adjust for resfactor
                    )
                    new_g_mrj[m, :, lu_idx] -= reduction_amnt

    return new_g_mrj


def get_biochar_effect_g_mrj(data:Data, yr_idx):
    """
    Applies the effects of using Biochar to the GHG data
    for all relevant agr. land uses.

    Parameters
    - data: The input data containing the necessary information.
    - yr_idx: The index of the year to calculate the effects for.

    Returns
    - new_g_mrj: The matrix <unit: t/cell> containing the updated GHG data after applying the Biochar effects.
    """
    land_uses = settings.AG_MANAGEMENTS_TO_LAND_USES['Biochar']
    yr_cal = data.YR_CAL_BASE + yr_idx

    # Set up the effects matrix
    new_g_mrj = np.zeros((data.NLMS, data.NCELLS, len(land_uses))).astype(np.float32)

    if not settings.AG_MANAGEMENTS['Biochar']:
        return new_g_mrj

    # Update values in the new matrix
    for lu_idx, lu in enumerate(land_uses):
        lu_data = data.BIOCHAR_DATA[settings.LU2TYPE[lu]]

        for lm in data.LANDMANS:
            m = 0 if lm == 'dry' else 1
            for co2e_type in [
                'CO2E_KG_HA_CROP_MGT',
                'CO2E_KG_HA_SOIL',
            ]:
                # Check if land-use/land management combination exists (e.g., dryland Pears/Rice do not occur), if not use zeros
                if lu not in data.AGGHG_CROPS[data.AGGHG_CROPS.columns[0][0], lm].columns:
                    continue

                reduction_perc = 1 - lu_data.loc[yr_cal, co2e_type]

                if reduction_perc != 0:
                    reduction_amnt = (
                        np.nan_to_num(data.AGGHG_CROPS[co2e_type, lm, lu].to_numpy().copy(), 0) 
                        * reduction_perc
                        / 1000            # convert to tonnes
                        * data.REAL_AREA  # adjust for resfactor
                    )
                    new_g_mrj[m, :, lu_idx] -= reduction_amnt

            # Subtract soil carbon benefit
            soil_multiplier = lu_data.loc[yr_cal, 'IMPACTS_soil_carbon'] - 1
            if soil_multiplier != 0:
                soil_reduction_amnt = (
                    data.SOIL_CARBON_AVG_T_CO2_HA_PER_YR
                    * soil_multiplier
                    * data.REAL_AREA
                )
                new_g_mrj[m, :, lu_idx] -= soil_reduction_amnt

    return new_g_mrj


def get_beef_hir_effect_g_mrj(data: Data, yr_idx):
    land_uses = settings.AG_MANAGEMENTS_TO_LAND_USES['HIR - Beef']
    g_mrj_effect = np.zeros((data.NLMS, data.NCELLS, len(land_uses)), dtype=np.float32)
    
    # GHG abatement from Land Use Change 
    for j_idx, lu in enumerate(land_uses):
        g_mrj_effect[:, :, j_idx] -= (
            (
                data.CO2E_STOCK_UNALL_NATURAL_TCO2_HA_PER_YR
                * (1 - data.BIO_HABITAT_CONTRIBUTION_LOOK_UP[data.DESC2AGLU[lu]])
            )
            - (
                data.CO2E_STOCK_UNALL_NATURAL_TCO2_HA_PER_YR
                * (1 - settings.HIR_CEILING_PERCENTAGE)
            )
        ) * data.REAL_AREA 

    # GHG abatement from livestock density reduction
    for lm_idx, lm in enumerate(data.LANDMANS):         
        for j_idx, lu in enumerate(land_uses):
            g_mrj_effect[lm_idx, :, j_idx] -= get_ghg_lvstk(data, lu, lm, yr_idx, True) * settings.HIR_PRODUCTIVITY_CONTRIBUTION

    return g_mrj_effect


def get_sheep_hir_effect_g_mrj(data: Data, yr_idx):
    land_uses = settings.AG_MANAGEMENTS_TO_LAND_USES['HIR - Sheep']
    g_mrj_effect = np.zeros((data.NLMS, data.NCELLS, len(land_uses)), dtype=np.float32)
    
    # GHG abatement from Land Use Change
    for j_idx, lu in enumerate(land_uses):
        g_mrj_effect[:, :, j_idx] -= (
            (
                data.CO2E_STOCK_UNALL_NATURAL_TCO2_HA_PER_YR
                * (1 - data.BIO_HABITAT_CONTRIBUTION_LOOK_UP[data.DESC2AGLU[lu]])
            )
            - (
                data.CO2E_STOCK_UNALL_NATURAL_TCO2_HA_PER_YR
                * (1 - settings.HIR_CEILING_PERCENTAGE)
            )
        ) * data.REAL_AREA 
        
    # GHG abatement from livestock density reduction
    for lm_idx, lm in enumerate(data.LANDMANS):
        for j_idx, lu in enumerate(land_uses):
            g_mrj_effect[lm_idx, :, j_idx] -= get_ghg_lvstk(data, lu, lm, yr_idx, True) * settings.HIR_PRODUCTIVITY_CONTRIBUTION


    return g_mrj_effect

def get_utility_solar_pv_effect_g_mrj(data: Data) -> np.ndarray:
    """
    Applies the effects of using solar PV to the GHG data
    for all relevant agricultural land uses.
    
    Returns zero impact as solar PV installation has no direct 
    impact on farm emissions - displacement benefits are handled 
    by AusTIMES integration.
    """
    land_uses = settings.AG_MANAGEMENTS_TO_LAND_USES['Utility Solar PV']
    
    # Set up the effects matrix - all zeros for no impact
    new_g_mrj = np.zeros((data.NLMS, data.NCELLS, len(land_uses))).astype(np.float32)
    
    if not settings.AG_MANAGEMENTS['Utility Solar PV']:
        return new_g_mrj
    
    # Return zeros - no direct emissions impact from solar installation
    return new_g_mrj

def get_onshore_wind_effect_g_mrj(data:Data) -> np.ndarray:
    """
    Applies the effects of using onshore wind to the GHG data
    for all relevant agricultural land uses.
    
    Returns zero impact as onshore wind installation has no direct 
    impact on farm emissions - displacement benefits are handled 
    by AusTIMES integration.
    """
    land_uses = settings.AG_MANAGEMENTS_TO_LAND_USES['Onshore Wind']
    
    # Set up the effects matrix - all zeros for no impact
    new_g_mrj = np.zeros((data.NLMS, data.NCELLS, len(land_uses))).astype(np.float32)
    
    if not settings.AG_MANAGEMENTS['Onshore Wind']:
        return new_g_mrj
    
    # Return zeros - no direct emissions impact from wind installation
    return new_g_mrj

def get_agricultural_management_ghg_matrices(data:Data, yr_idx) -> dict[str, np.ndarray]:
    """
    Calculate the greenhouse gas (GHG) matrices for different agricultural management practices.

    Args:
        data: The input data for the calculations.
        yr_idx: The year index.

    Returns
        A dictionary containing the GHG matrices <unit: t/cell> for different agricultural management practices.
        The keys of the dictionary represent the management practices, and the values are numpy arrays.

    """
    asparagopsis_data = get_asparagopsis_effect_g_mrj(data, yr_idx)                         
    precision_agriculture_data = get_precision_agriculture_effect_g_mrj(data, yr_idx)       
    eco_grazing_data = get_ecological_grazing_effect_g_mrj(data, yr_idx)                    
    sav_burning_ghg_impact = get_savanna_burning_effect_g_mrj(data)                         
    agtech_ei_ghg_impact = get_agtech_ei_effect_g_mrj(data, yr_idx)                         
    biochar_ghg_impact = get_biochar_effect_g_mrj(data, yr_idx)                             
    beef_hir_ghg_impact = get_beef_hir_effect_g_mrj(data, yr_idx)                                   
    sheep_hir_ghg_impact = get_sheep_hir_effect_g_mrj(data, yr_idx)
    utility_solar_ghg_impact = get_utility_solar_pv_effect_g_mrj(data)
    onshore_wind_ghg_impact = get_onshore_wind_effect_g_mrj(data)                                 

    return {
        'Asparagopsis taxiformis': asparagopsis_data,
        'Precision Agriculture': precision_agriculture_data,
        'Ecological Grazing': eco_grazing_data,
        'Savanna Burning': sav_burning_ghg_impact,
        'AgTech EI': agtech_ei_ghg_impact,
        'Biochar': biochar_ghg_impact,
        'HIR - Beef': beef_hir_ghg_impact,
        'HIR - Sheep': sheep_hir_ghg_impact,
        'Utility Solar PV': utility_solar_ghg_impact,
        'Onshore Wind': onshore_wind_ghg_impact
    }
