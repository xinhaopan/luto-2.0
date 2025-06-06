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
import pandas as pd
import luto.settings as settings

from luto.data import Data
from luto import tools



def get_ghg_env_plantings(data: Data, aggregate) -> np.ndarray|pd.DataFrame:
    """
    Parameters
    ----------
    data: Data object.

    Returns
    -------
    if aggregate == True (default)  -> np.ndarray
       aggregate == False           -> pd.DataFrame
    
    Greenhouse gas emissions of environmental plantings for each cell.
    Since environmental plantings reduces carbon in the air, each value will be <= 0.
    """
    
    # Tonnes of CO2e per ha, adjusted for resfactor
    if aggregate==True:
        return -data.EP_BLOCK_AVG_T_CO2_HA * data.REAL_AREA
    elif aggregate==False:
        return pd.DataFrame(-data.EP_BLOCK_AVG_T_CO2_HA * data.REAL_AREA, columns=['ENV_PLANTINGS'])
    else:
        # If the aggregate arguments is not in [True,False]. That must be someting wrong
        raise KeyError(f"Aggregate '{aggregate} can be only specified as [True,False]" )


def get_ghg_rip_plantings(data: Data, aggregate) -> np.ndarray|pd.DataFrame:
    """
    Parameters
    ----------
    data: Data object.

    Returns
    -------
    if aggregate == True (default)  -> np.ndarray
       aggregate == False           -> pd.DataFrame
    
    Greenhouse gas emissions of Riparian Plantings for each cell. Same as environmental plantings.
    Since riparian plantings reduces carbon in the air, each value will be <= 0.
    """

    # Tonnes of CO2e per ha, adjusted for resfactor
    if aggregate==True:
        return -data.EP_RIP_AVG_T_CO2_HA * data.REAL_AREA
    elif aggregate==False:
        return pd.DataFrame(-data.EP_RIP_AVG_T_CO2_HA * data.REAL_AREA,columns=['RIP_PLANTINGS'])
    else:
        raise KeyError(f"Aggregate '{aggregate} can be only specified as [True,False]" )


def get_ghg_agroforestry_base(data: Data) -> np.ndarray:
    """
    Parameters
    ----------
    data: Data object.

    Returns
    -------
    np.ndarray
    
    Greenhouse gas emissions of agroforestry for each cell.
    Since agroforestry reduces carbon in the air, each value will be <= 0.
    """
    
    # Tonnes of CO2e per ha, adjusted for resfactor
    return -data.EP_BELT_AVG_T_CO2_HA * data.REAL_AREA
    

def get_ghg_sheep_agroforestry(
    data: Data,
    ag_g_mrj: np.ndarray, 
    agroforestry_x_r: np.ndarray,
    aggregate: bool,
) -> np.ndarray|pd.DataFrame:
    """
    Parameters
    ------
    data: Data object.
    aggregate: boolean governing whether to aggregate data or not.
    ag_g_mrj: agricultural GHG matrix.
    agroforestry_x_r: Agroforestry exclude matrix.

    Returns
    ------
    Numpy array indexed by r
    """
    sheep_j = tools.get_sheep_code(data)

    # Only use the dryland version of sheep
    sheep_cost = ag_g_mrj[0, :, sheep_j]
    base_agroforestry_cost = get_ghg_agroforestry_base(data)

    # Calculate contributions and return the sum
    agroforestry_contr = base_agroforestry_cost * agroforestry_x_r
    sheep_contr = sheep_cost * (1 - agroforestry_x_r)
    ghg_total = agroforestry_contr + sheep_contr

    if aggregate==True:
        return ghg_total
    elif aggregate==False:
        return pd.DataFrame(ghg_total, columns=['SHEEP_AGROFORESTRY'])
    else:
        raise KeyError(f"Aggregate '{aggregate} can be only specified as [True,False]" )
    

def get_ghg_beef_agroforestry(
    data: Data,
    ag_g_mrj: np.ndarray, 
    agroforestry_x_r: np.ndarray,
    aggregate: bool,
) -> np.ndarray|pd.DataFrame:
    """
    Parameters
    ------
    data: Data object.
    aggregate: boolean governing whether to aggregate data or not.
    ag_g_mrj: agricultural GHG matrix.
    agroforestry_x_r: Agroforestry exclude matrix.

    Returns
    ------
    Numpy array indexed by r
    """
    beef_j = tools.get_beef_code(data)

    # Only use the dryland version of beef
    beef_cost = ag_g_mrj[0, :, beef_j]
    base_agroforestry_cost = get_ghg_agroforestry_base(data)

    # Calculate contributions and return the sum
    agroforestry_contr = base_agroforestry_cost * agroforestry_x_r
    beef_contr = beef_cost * (1 - agroforestry_x_r)
    ghg_total = agroforestry_contr + beef_contr

    if aggregate==True:
        return ghg_total
    elif aggregate==False:
        return pd.DataFrame(ghg_total, columns=['BEEF_AGROFORESTRY'])
    else:
        raise KeyError(f"Aggregate '{aggregate} can be only specified as [True,False]" )


def get_ghg_carbon_plantings_block(data, aggregate) -> np.ndarray|pd.DataFrame:
    """
    Parameters
    ----------
    data: object/module
        Data object or module with fields like in `luto.data`.

    Returns
    -------
    if aggregate == True (default)  -> np.ndarray
       aggregate == False           -> pd.DataFrame
    
    Greenhouse gas emissions of carbon plantings (block) for each cell.
    Since carbon plantings reduces carbon in the air, each value will be <= 0.
    """
    
    # Tonnes of CO2e per ha, adjusted for resfactor
    if aggregate==True:
        return -data.CP_BLOCK_AVG_T_CO2_HA * data.REAL_AREA
    elif aggregate==False:
        return pd.DataFrame(-data.CP_BLOCK_AVG_T_CO2_HA * data.REAL_AREA,columns=['CARBON_PLANTINGS_BLOCK'])
    else:
        raise KeyError(f"Aggregate '{aggregate} can be only specified as [True,False]" )
    

def get_ghg_carbon_plantings_belt_base(data) -> np.ndarray:
    """
    Parameters
    ----------
    data: object/module
        Data object or module with fields like in `luto.data`.

    Returns
    -------
    np.ndarray
    
    Greenhouse gas emissions of carbon plantings (belt) for each cell.
    Since carbon plantings reduces carbon in the air, each value will be <= 0.
    """
    # Tonnes of CO2e per ha, adjusted for resfactor
    return -data.CP_BELT_AVG_T_CO2_HA * data.REAL_AREA
    

def get_ghg_sheep_carbon_plantings_belt(
    data: Data,
    ag_g_mrj: np.ndarray, 
    cp_belt_x_r: np.ndarray,
    aggregate: bool,
) -> np.ndarray|pd.DataFrame:
    """
    Parameters
    ------
    data: Data object.
    aggregate: boolean governing whether to aggregate data or not.
    ag_g_mrj: agricultural GHG matrix.
    cp_belt_x_r: Carbon plantings belt exclude matrix.

    Returns
    ------
    Numpy array indexed by r
    """
    sheep_j = tools.get_sheep_code(data)

    # Only use the dryland version of sheep
    sheep_cost = ag_g_mrj[0, :, sheep_j]
    base_cp_cost = get_ghg_carbon_plantings_belt_base(data)

    # Calculate contributions and return the sum
    cp_contr = base_cp_cost * cp_belt_x_r
    sheep_contr = sheep_cost * (1 - cp_belt_x_r)
    ghg_total = cp_contr + sheep_contr

    if aggregate==True:
        return ghg_total
    elif aggregate==False:
        return pd.DataFrame(ghg_total,columns=['SHEEP_CARBON_PLANTINGS_BELT'])
    else:
        raise KeyError(f"Aggregate '{aggregate} can be only specified as [True,False]" )


def get_ghg_beef_carbon_plantings_belt(
    data: Data,
    ag_g_mrj: np.ndarray, 
    cp_belt_x_r: np.ndarray,
    aggregate: bool,
) -> np.ndarray|pd.DataFrame:
    """
    Parameters
    ------
    data: Data object.
    aggregate: boolean governing whether to aggregate data or not.
    ag_g_mrj: agricultural GHG matrix.
    cp_belt_x_r: Carbon plantings belt exclude matrix.

    Returns
    ------
    Numpy array indexed by r
    """
    beef_j = tools.get_beef_code(data)

    # Only use the dryland version of beef
    beef_cost = ag_g_mrj[0, :, beef_j]
    base_cp_cost = get_ghg_carbon_plantings_belt_base(data)

    # Calculate contributions and return the sum
    cp_contr = base_cp_cost * cp_belt_x_r
    beef_contr = beef_cost * (1 - cp_belt_x_r)
    ghg_total = cp_contr + beef_contr

    if aggregate==True:
        return ghg_total
    elif aggregate==False:
        return pd.DataFrame(ghg_total,columns=['SHEEP_CARBON_PLANTINGS_BELT'])
    else:
        raise KeyError(f"Aggregate '{aggregate} can be only specified as [True,False]" )


def get_ghg_beccs(data, aggregate) -> np.ndarray|pd.DataFrame:
    """
    Calculate the greenhouse gas emissions of BECCS for each cell.
    
    Parameters
    ----------
    data: object/module
        Data object or module with fields like in `luto.data`.

    Returns
    -------
    if aggregate == True (default)  -> np.ndarray
       aggregate == False           -> pd.DataFrame
    """

    # Tonnes of CO2e per ha, adjusted for resfactor
    if aggregate==True:
        return -np.nan_to_num(data.BECCS_TCO2E_HA_YR) * data.REAL_AREA
    elif aggregate==False:
        return pd.DataFrame(-np.nan_to_num(data.BECCS_TCO2E_HA_YR) * data.REAL_AREA, columns=['BECCS'])
    else:
        raise KeyError(f"Aggregate '{aggregate} can be only specified as [True,False]" )
    

def get_ghg_destocked_land(
    data: Data,
    lumap: np.ndarray,
    aggregate: bool = True,
) -> np.ndarray:
    """
    Calculate the greenhouse gas emissions of Destocked land for each cell. Note, only cells that are 
    livestock - natural land in the BASE_YR (2010) and then converted into unallowcated - natural land 
    in the target year are 'destocked'.
    
    Parameters
    ----------
    data: object/module
        Data object or module with fields like in `luto.data`.

    Returns
    -------
    if aggregate == True (default)  -> np.ndarray
       aggregate == False           -> pd.DataFrame
    """
    # Only cells that are livestock - natural land in the BASE_YR (2010) and then converted into unallowcated - natural land in the target year are 'destocked'.
    lumap_BASE_YR = data.lumaps[data.YR_CAL_BASE]
    penalty_ghg_r = np.zeros(data.NCELLS)
    
    for from_lu in data.LU_LVSTK_NATURAL:
        penalty_ghg_r[lumap_BASE_YR == from_lu] = (
            data.CO2E_STOCK_UNALL_NATURAL[lumap_BASE_YR == from_lu]
            * (data.BIO_HABITAT_CONTRIBUTION_LOOK_UP[from_lu] - 1)
            * data.REAL_AREA[lumap_BASE_YR == from_lu]
            / settings.HIR_EFFECT_YEARS    # Annualise carbon sequestration capacity to align the full grwoth span of a tree
        )

    if aggregate==True:
        return penalty_ghg_r
    elif aggregate==False:
        return pd.DataFrame(penalty_ghg_r, columns=['DESTOCKED_LAND'])
    else:
        # If the aggregate arguments is not in [True,False]. That must be someting wrong
        raise KeyError(f"Aggregate '{aggregate} can be only specified as [True,False]" )



def get_ghg_matrix(data: Data, ag_g_mrj, lumap, aggregate=True) -> np.ndarray:
    """
    Get the g_rk matrix containing non-agricultural greenhouse gas emissions.

    Parameters
    - data: The input data for calculating greenhouse gas emissions.
    - aggregate: A boolean flag indicating whether to aggregate the matrices or not. Default is True.

    Returns
    - If aggregate is True, returns a numpy ndarray representing the aggregated g_rk matrix.
    - If aggregate is False, returns a pandas DataFrame representing the g_rk matrix.

    Raises:
    - KeyError: If the aggregate argument is not a boolean value.

    Note:
    - The function internally calls several other functions to calculate different components of the g_rk matrix.
    """

    agroforestry_x_r = tools.get_exclusions_agroforestry_base(data, lumap)
    cp_belt_x_r = tools.get_exclusions_carbon_plantings_belt_base(data, lumap)

    non_agr_ghg_matrices = {}

    # reshape each non-agricultural matrix to be indexed (r, k) and concatenate on the k indexing
    non_agr_ghg_matrices['Environmental Plantings'] = get_ghg_env_plantings(data, aggregate)                                                
    non_agr_ghg_matrices['Riparian Plantings'] = get_ghg_rip_plantings(data, aggregate)                                                     
    non_agr_ghg_matrices['Sheep Agroforestry'] = get_ghg_sheep_agroforestry(data, ag_g_mrj, agroforestry_x_r, aggregate)                    
    non_agr_ghg_matrices['Beef Agroforestry'] = get_ghg_beef_agroforestry(data, ag_g_mrj, agroforestry_x_r, aggregate)                      
    non_agr_ghg_matrices['Carbon Plantings (Block)'] = get_ghg_carbon_plantings_block(data, aggregate)                                      
    non_agr_ghg_matrices['Sheep Carbon Plantings (Belt)'] = get_ghg_sheep_carbon_plantings_belt(data, ag_g_mrj, cp_belt_x_r, aggregate) 
    non_agr_ghg_matrices['Beef Carbon Plantings (Belt)'] = get_ghg_beef_carbon_plantings_belt(data, ag_g_mrj, cp_belt_x_r, aggregate)       
    non_agr_ghg_matrices['BECCS'] = get_ghg_beccs(data, aggregate)                                                                          
    non_agr_ghg_matrices['Destocked - natural land'] = get_ghg_destocked_land(data, lumap, aggregate)                                       
      
    if aggregate==True:
        # reshape each non-agricultural matrix to be indexed (r, k) and concatenate on the k indexing
        non_agr_ghg_matrices = [
            non_agr_ghg_matrix.reshape((data.NCELLS, 1)) for non_agr_ghg_matrix in non_agr_ghg_matrices.values()
        ]
        return np.concatenate(non_agr_ghg_matrices, axis=1)
    
    elif aggregate==False:
        return pd.concat(list(non_agr_ghg_matrices.values()), axis=1)
    else:
        # If the aggregate arguments is not in [True,False]. That must be someting wrong
        raise KeyError(f"Aggregate '{aggregate} can be only specified as [True,False]" )
