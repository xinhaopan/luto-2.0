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
Pure helper functions and other tools.
"""

import sys
import os.path
import traceback
import functools

import pandas as pd
import numpy as np
import xarray as xr
import numpy_financial as npf
import luto.settings as settings

from typing import Tuple
from datetime import datetime

import luto.economics.agricultural.water as ag_water
import luto.economics.non_agricultural.water as non_ag_water


def write_timestamp():
    timestamp = datetime.now().strftime('%Y_%m_%d__%H_%M_%S')
    timestamp_path = os.path.join(settings.OUTPUT_DIR, '.timestamp')
    with open(timestamp_path, 'w') as f: f.write(timestamp)
    return timestamp

def read_timestamp():
    timestamp_path = os.path.join(settings.OUTPUT_DIR, '.timestamp')
    if os.path.exists(timestamp_path):
        with open(timestamp_path, 'r') as f: timestamp = f.read()
    else:
        raise FileNotFoundError(f"Timestamp file not found at {timestamp_path}")
    return timestamp


def amortise(cost, rate=settings.DISCOUNT_RATE, horizon=settings.AMORTISATION_PERIOD):
    """Return NPV of future `cost` amortised to annual value at discount `rate` over `horizon` years."""
    if settings.AMORTISE_UPFRONT_COSTS: return -1 * npf.pmt(rate, horizon, pv=cost, fv=0, when='begin')
    else: return cost


def lumap2ag_l_mrj(lumap, lmmap):
    """
    Return land-use maps in decision-variable (X_mrj) format.
    Where 'm' is land mgt, 'r' is cell, and 'j' is agricultural land-use.

    Cells used for non-agricultural land uses will have value 0 for all agricultural
    land uses, i.e. all r.
    """
    # Set up a container array of shape m, r, j.
    x_mrj = np.zeros((2, lumap.shape[0], 28), dtype=bool)   # TODO - remove 2

    # Populate the 3D land-use, land mgt mask.
    for j in range(28):
        # One boolean map for each land use.
        jmap = np.where(lumap == j, True, False).astype(bool)
        # Keep only dryland version.
        x_mrj[0, :, j] = np.where(lmmap == False, jmap, False)
        # Keep only irrigated version.
        x_mrj[1, :, j] = np.where(lmmap == True, jmap, False)

    return x_mrj.astype(bool)


def lumap2non_ag_l_mk(lumap, num_non_ag_land_uses: int):
    """
    Convert the land-use map to a decision variable X_rk, where 'r' indexes cell and
    'k' indexes non-agricultural land use.

    Cells used for agricultural purposes have value 0 for all k.
    """
    base_code = settings.NON_AGRICULTURAL_LU_BASE_CODE
    non_ag_lu_codes = list(range(base_code, base_code + num_non_ag_land_uses))

    # Set up a container array of shape r, k.
    x_rk = np.zeros((lumap.shape[0], num_non_ag_land_uses), dtype=bool)

    for i,k in enumerate(non_ag_lu_codes):
        kmap = np.where(lumap == k, True, False)
        x_rk[:, i] = kmap

    return x_rk.astype(bool)


def get_ag_and_non_ag_cells(lumap) -> Tuple[np.ndarray, np.ndarray]:
    """
    Splits the index of cells based on whether that cell is used for agricultural
    land, given the lumap.

    Returns
    -------
    ( np.ndarray, np.ndarray )
        Two numpy arrays containing the split cell index.
    """
    non_ag_base = settings.NON_AGRICULTURAL_LU_BASE_CODE
    all_cells = np.array(range(lumap.shape[0]))

    # get all agricultural and non agricultural cells
    non_agricultural_cells = np.nonzero(lumap >= non_ag_base)[0]
    agricultural_cells = np.nonzero(
        ~np.isin(all_cells, non_agricultural_cells)
    )[0]

    return agricultural_cells, non_agricultural_cells


def get_env_plantings_cells(lumap) -> np.ndarray:
    """
    Get an array with cells used for environmental plantings
    """
    return np.nonzero(lumap == settings.NON_AGRICULTURAL_LU_BASE_CODE + 0)[0]


def get_riparian_plantings_cells(lumap) -> np.ndarray:
    """
    Get an array with cells used for riparian plantings
    """
    return np.nonzero(lumap == settings.NON_AGRICULTURAL_LU_BASE_CODE + 1)[0]


def get_sheep_agroforestry_cells(lumap) -> np.ndarray:
    """
    Get an array with cells used for riparian plantings
    """
    return np.nonzero(lumap == settings.NON_AGRICULTURAL_LU_BASE_CODE + 2)[0]


def get_beef_agroforestry_cells(lumap) -> np.ndarray:
    """
    Get an array with cells used for riparian plantings
    """
    return np.nonzero(lumap == settings.NON_AGRICULTURAL_LU_BASE_CODE + 3)[0]


def get_agroforestry_cells(lumap) -> np.ndarray:
    """
    Get an array with cells that currently use agroforestry (either sheep or beef)
    """
    agroforestry_lus = [settings.NON_AGRICULTURAL_LU_BASE_CODE + 2, settings.NON_AGRICULTURAL_LU_BASE_CODE + 3]
    return np.nonzero(np.isin(lumap, agroforestry_lus))[0]


def get_carbon_plantings_block_cells(lumap) -> np.ndarray:
    """
    Get an array with all cells being used for carbon plantings (block)
    """
    return np.nonzero(lumap == settings.NON_AGRICULTURAL_LU_BASE_CODE + 4)[0]


def get_sheep_carbon_plantings_belt_cells(lumap) -> np.ndarray:
    """
    Get an array with all cells being used for sheep carbon plantings (belt)
    """
    return np.nonzero(lumap == settings.NON_AGRICULTURAL_LU_BASE_CODE + 5)[0]


def get_beef_carbon_plantings_belt_cells(lumap) -> np.ndarray:
    """
    Get an array with all cells being used for beef carbon plantings (belt)
    """
    return np.nonzero(lumap == settings.NON_AGRICULTURAL_LU_BASE_CODE + 6)[0]


def get_carbon_plantings_belt_cells(lumap) -> np.ndarray:
    """
    Get an array with cells used that currently use carbon plantings belt (either sheep or beef)
    """

    cp_belt_lus = [settings.NON_AGRICULTURAL_LU_BASE_CODE + 5, settings.NON_AGRICULTURAL_LU_BASE_CODE + 6]
    return np.nonzero(np.isin(lumap, cp_belt_lus))[0]


def get_beccs_cells(lumap) -> np.ndarray:
    """
    Get an array with all cells being used for carbon plantings (block)
    """
    return np.nonzero(lumap == settings.NON_AGRICULTURAL_LU_BASE_CODE + 7)[0]


def get_unallocated_natural_lu_cells(data, lumap) -> np.ndarray:
    """
    Gets all cells being used for unallocated natural land uses.
    """
    return np.nonzero(np.isin(lumap, data.DESC2AGLU["Unallocated - natural land"]))[0]

def get_lvstk_natural_lu_cells(data, lumap) -> np.ndarray:
    """
    Gets all cells being used for livestock natural land uses.
    """
    return np.nonzero(np.isin(lumap, data.LU_LVSTK_NATURAL))[0]


def get_non_ag_natural_lu_cells(data, lumap) -> np.ndarray:
    """
    Gets all cells being used for non-agricultural natural land uses.
    """
    return np.nonzero(np.isin(lumap, data.NON_AG_LU_NATURAL))[0]


def get_ag_and_non_ag_natural_lu_cells(data, lumap) -> np.ndarray:
    """
    Gets all cells being used for natural land uses, both agricultural and non-agricultural.
    """
    return np.nonzero(np.isin(lumap, data.LU_NATURAL + data.NON_AG_LU_NATURAL))[0]


def get_ag_cells(lumap) -> np.ndarray:
    """
    Get an array containing the index of all non-agricultural cells
    """
    return np.nonzero(lumap < settings.NON_AGRICULTURAL_LU_BASE_CODE)[0]


def get_non_ag_cells(lumap) -> np.ndarray:
    """
    Get an array containing the index of all non-agricultural cells
    """
    return np.nonzero(lumap >= settings.NON_AGRICULTURAL_LU_BASE_CODE)[0]


def get_ag_to_ag_water_delta_matrix(w_mrj, l_mrj, data, yr_idx):
    """
    Gets the water delta matrix ($/cell) that applies the cost of installing/removing irrigation to
    base transition costs. Includes the costs of water license fees.

    Parameters
     w_mrj (numpy.ndarray, <unit:ML/cell>): Water requirements matrix for target year.
     l_mrj (numpy.ndarray): Land-use and land management matrix for the base_year.
     data (object): Data object containing necessary information.

    Returns
     w_delta_mrj (numpy.ndarray, <unit:$/cell>).
    """
    yr_cal = data.YR_CAL_BASE + yr_idx
    
    # Get water requirements from current agriculture, converting water requirements for LVSTK from ML per head to ML per cell (inc. REAL_AREA).
    # Sum total water requirements of current land-use and land management
    w_r = (w_mrj * l_mrj).sum(axis=0).sum(axis=1)

    # Net water requirements calculated as the diff in water requirements between current land-use and all other land-uses j.
    w_net_mrj = w_mrj - w_r[:, np.newaxis]

    # Water license cost calculated as net water requirements (ML/cell) x licence price ($/ML).
    w_delta_mrj = w_net_mrj * data.WATER_LICENCE_PRICE[:, np.newaxis] * data.WATER_LICENSE_COST_MULTS[yr_cal] * settings.INCLUDE_WATER_LICENSE_COSTS

    # When land-use changes from dryland to irrigated add <settings.NEW_IRRIG_COST> per hectare for establishing irrigation infrastructure
    new_irrig = (
        settings.NEW_IRRIG_COST
        * data.IRRIG_COST_MULTS[yr_cal]
        * data.REAL_AREA[:, np.newaxis]  # <unit:$/cell>
    )
    w_delta_mrj[1] = np.where(l_mrj[0], w_delta_mrj[1] + new_irrig, w_delta_mrj[1])

    # When land-use changes from irrigated to dryland add <settings.REMOVE_IRRIG_COST> per hectare for removing irrigation infrastructure
    remove_irrig = (
        settings.REMOVE_IRRIG_COST
        * data.IRRIG_COST_MULTS[yr_cal]
        * data.REAL_AREA[:, np.newaxis]  # <unit:$/cell>
    )
    w_delta_mrj[0] = np.where(l_mrj[1], w_delta_mrj[0] + remove_irrig, w_delta_mrj[0])
    
    # Amortise upfront costs to annualised costs
    w_delta_mrj = amortise(w_delta_mrj)
    
    return w_delta_mrj  # <unit:$/cell>

def get_ag_to_non_ag_water_delta_matrix(data, yr_idx, lumap, lmmap)->tuple[np.ndarray, np.ndarray]:
    """
    Gets the water delta matrix ($/cell) that applies the cost of installing/removing irrigation to
    base transition costs. Includes the costs of water license fees.
    
    Parameters
     data (object): Data object containing necessary information.
     yr_idx (int): Index of the target year.
     lumap (numpy.ndarray): Land-use map.
     lmmap (numpy.ndarray): Land management map.
    
    Returns
     w_license_cost_r (numpy.ndarray) : Water license cost for each cell.
     w_rm_irrig_cost_r (numpy.ndarray) : Cost of removing irrigation for each cell.
     
     
    """
    
    yr_cal = data.YR_CAL_BASE + yr_idx
    l_mrj = lumap2ag_l_mrj(lumap, lmmap)
    non_ag_cells = get_non_ag_cells(lumap)
    
    w_req_mrj = ag_water.get_wreq_matrices(data, yr_idx).astype(np.float32)     # <unit: ML/CELL>
    w_req_r = (w_req_mrj * l_mrj).sum(axis=0).sum(axis=1)
    w_yield_r = non_ag_water.get_w_net_yield_matrix_env_planting(data, yr_idx)  # <unit: ML/CELL>
    w_delta_r = - (w_req_r + w_yield_r)
    
    w_license_cost_r = w_delta_r * data.WATER_LICENCE_PRICE * data.WATER_LICENSE_COST_MULTS[yr_cal] * settings.INCLUDE_WATER_LICENSE_COSTS     # <unit: $/CELL>
    w_rm_irrig_cost_r = np.where(lmmap == 1, settings.REMOVE_IRRIG_COST * data.IRRIG_COST_MULTS[yr_cal], 0) * data.REAL_AREA                   # <unit: $/CELL>
    
    # Amortise upfront costs to annualised costs
    w_license_cost_r = amortise(w_license_cost_r)
    w_rm_irrig_cost_r = amortise(w_rm_irrig_cost_r)
    
    return w_license_cost_r, w_rm_irrig_cost_r


def am_name_snake_case(am_name):
    """Get snake_case version of the AM name"""
    return am_name.lower().replace(' ', '_')


def get_exclusions_for_excluding_all_natural_cells(data, lumap) -> np.ndarray:
    """
    A number of non-agricultural land uses can only be applied to cells that
    don't already utilise a natural land use. This function gets the exclusion
    matrix for all such non-ag land uses, returning an array valued 0 at the 
    indices of cells that use natural land uses, and 1 everywhere else.

    Parameters
     data: The data object containing information about the cells.
     lumap: The land use map.

    Returns
     exclude: An array of shape (NCELLS,) with values 0 at the indices of cells
               that use natural land uses, and 1 everywhere else.
    """
    exclude = np.ones(data.NCELLS)

    natural_lu_cells = get_ag_and_non_ag_natural_lu_cells(data, lumap)
    exclude[natural_lu_cells] = 0

    return exclude


def get_exclusions_agroforestry_base(data, lumap) -> np.ndarray:
    """
    Return a 1-D array indexed by r that represents how much agroforestry can possibly 
    be done at each cell.

    Parameters
     data: The data object containing information about the landscape.
     lumap: The land use map.

    Returns
     exclude: A 1-D array.
    """
    exclude = (np.ones(data.NCELLS) * settings.AF_PROPORTION).astype(np.float32)

    # Ensure cells being used for agroforestry may retain that LU
    exclude[get_agroforestry_cells(lumap)] = settings.AF_PROPORTION

    return exclude


def get_exclusions_carbon_plantings_belt_base(data, lumap) -> np.ndarray:
    """
    Return a 1-D array indexed by r that represents how much carbon plantings (belt) can possibly 
    be done at each cell.

    Parameters
     data (Data): The data object containing information about the cells.
     lumap (np.ndarray): The land use map.

    Returns
     exclude: A 1-D array
    """
    exclude = (np.ones(data.NCELLS) * settings.CP_BELT_PROPORTION).astype(np.float32)

    # Ensure cells being used for carbon plantings (belt) may retain that LU
    exclude[get_carbon_plantings_belt_cells(lumap)] = settings.CP_BELT_PROPORTION

    return exclude


def get_sheep_code(data):
    """
    Get the land use code (j) for 'Sheep - modified land'
    """
    return data.DESC2AGLU['Sheep - modified land']


def get_beef_code(data):
    """
    Get the land use code (j) for 'Beef - modified land'
    """
    return data.DESC2AGLU['Beef - modified land']

def ag_mrj_to_xr(data, arr):
    return xr.DataArray(
        arr,
        dims=['lm', 'cell', 'lu'],
        coords={'lm': data.LANDMANS,
                'cell': np.arange(data.NCELLS),
                'lu': data.AGRICULTURAL_LANDUSES}
    )

def non_ag_rk_to_xr(data, arr):
    return xr.DataArray(
        arr,
        dims=['cell', 'lu'],
        coords={'cell': np.arange(data.NCELLS),
                'lu': data.NON_AGRICULTURAL_LANDUSES}
    )

def am_mrj_to_xr(data, am_mrj_dict):
    emp_arr_xr = xr.DataArray(
        np.full((len(settings.AG_MANAGEMENTS_TO_LAND_USES), len(data.LANDMANS), data.NCELLS, len(data.AGRICULTURAL_LANDUSES)), np.nan),
        dims=['am', 'lm', 'cell', 'lu'],
        coords={'am': list(settings.AG_MANAGEMENTS_TO_LAND_USES.keys()),
                'lm': data.LANDMANS,
                'cell': np.arange(data.NCELLS),
                'lu': data.AGRICULTURAL_LANDUSES}
    )

    for am,lu in settings.AG_MANAGEMENTS_TO_LAND_USES.items():
        if emp_arr_xr.loc[am, :, :, lu].shape == am_mrj_dict[am].shape:
            # If the shape is the same, just assign the value
            emp_arr_xr.loc[am, :, :, lu] = am_mrj_dict[am]  
        else:
            # Otherwise, assign the array at index of the land use
            lu_idx = [data.DESC2AGLU[i] for i in settings.AG_MANAGEMENTS_TO_LAND_USES[am]]
            emp_arr_xr.loc[am, :, :, lu] = am_mrj_dict[am][:,:, lu_idx]   
    return emp_arr_xr


def map_desc_to_dvar_index(category: str,
                           desc2idx: dict,
                           dvar_arr: np.ndarray):
    '''Input:
        category: str, the category of the dvar, e.g., 'Agriculture/Non-Agriculture',
        desc2idx: dict, the mapping between lu_desc and dvar index, e.g., {'Apples': 0 ...},
        dvar_arr: np.ndarray, the dvar array with shape (r,{j|k}), where r is the number of pixels,
                  and {j|k} is the dimension of ag-landuses or non-ag-landuses.

    Return:
        pd.DataFrame, with columns of ['Category','lu_desc','dvar_idx','dvar'].'''

    df = pd.DataFrame({'Category': category,
                       'lu_desc': desc2idx.keys(),
                       'dvar_idx': desc2idx.values()})

    df['dvar'] = [dvar_arr[:, j] for j in df['dvar_idx']]

    return df.reindex(columns=['Category', 'lu_desc', 'dvar_idx', 'dvar'])


def get_out_resfactor(dvar_path:str):
    # Get the number of pixels in the output decision variable
    dvar = np.load(dvar_path)
    dvar_size = dvar.shape[1]

    # Get the number of pixels in full resolution LUMASK
    full_lumap = pd.read_hdf(os.path.join(settings.INPUT_DIR, "lumap.h5")).to_numpy() 
    full_size = (full_lumap > 0).sum()

    # Return the resfactor
    return round((full_size/dvar_size)**0.5)


    
def calc_water(
    data, 
    ind:np.ndarray, 
    ag_w_mrj:np.ndarray, 
    non_ag_w_rk:np.ndarray, 
    ag_man_w_mrj:np.ndarray, 
    ag_dvar:np.ndarray, 
    non_ag_dvar:np.ndarray, 
    am_dvar:np.ndarray
) -> pd.DataFrame:
    
    '''
    Note
        This function is only used for the `write_water` in the `luto.tools.write` module.
        Calculate water yields for year given the index. 
    
    Return
     pd.DataFrame, the water yields for year and region.
    '''
    
    # Calculate water yields for year and region.
    index_levels = ['Landuse Type', 'Landuse', 'Water_supply',  'Water Net Yield (ML)']

    # Agricultural contribution
    ag_mrj = ag_w_mrj[:, ind, :] * ag_dvar[:, ind, :]   
    ag_jm = np.einsum('mrj->jm', ag_mrj)                             
    ag_df = pd.DataFrame(
        ag_jm.reshape(-1).tolist(),
        index=pd.MultiIndex.from_product(
            [['Agricultural Landuse'],
                data.AGRICULTURAL_LANDUSES,
                data.LANDMANS])).reset_index()
    ag_df.columns = index_levels

    # Non-agricultural contribution
    non_ag_rk = non_ag_w_rk[ind, :] * non_ag_dvar[ind, :]  # Non-agricultural contribution
    non_ag_k = np.einsum('rk->k', non_ag_rk)                             # Sum over cells
    non_ag_df = pd.DataFrame(
        non_ag_k, 
        index= pd.MultiIndex.from_product([
                ['Non-agricultural Landuse'],
                settings.NON_AG_LAND_USES.keys() ,
                ['dry']  # non-agricultural land is always dry
    ])).reset_index()
    non_ag_df.columns = index_levels

    # Agricultural managements contribution
    AM_dfs = []
    for am, am_lus in settings.AG_MANAGEMENTS_TO_LAND_USES.items():  # Agricultural managements contribution
        am_j = np.array([data.DESC2AGLU[lu] for lu in am_lus])
        am_mrj = ag_man_w_mrj[am][:, ind, :] * am_dvar[am][:, ind, :][:, :, am_j] 
        am_jm = np.einsum('mrj->jm', am_mrj)
        # water yields for each agricultural management in long dataframe format
        df_am = pd.DataFrame(
            am_jm.reshape(-1).tolist(),
            index=pd.MultiIndex.from_product([
                ['Agricultural Management'],
                am_lus,
                data.LANDMANS
                ])).reset_index()
        df_am.columns = index_levels
        AM_dfs.append(df_am)
    AM_df = pd.concat(AM_dfs)
    
    return pd.concat([ag_df, non_ag_df, AM_df])


def calc_major_vegetation_group_ag_area_for_year(
    GBF3_raw_MVG_area_vr: np.ndarray, lumap: np.ndarray, biodiv_contr_ag_rj: dict[int, float]
) -> dict[int, float]:
    prod_data = {}

    ag_biodiv_impacts_j = {j: 1 - x for j, x in biodiv_contr_ag_rj.items()}

    # Numpy magic
    ag_biodiv_degr_r = np.vectorize(ag_biodiv_impacts_j.get)(lumap)
    
    for v in range(GBF3_raw_MVG_area_vr.shape[0]):
        prod_data[v] = GBF3_raw_MVG_area_vr[v, :] * ag_biodiv_degr_r

    return prod_data


def calc_species_ag_area_for_year(
    GBF4_raw_species_area_sr: np.ndarray, lumap: np.ndarray, biodiv_contr_ag_rj: dict[int, float]
) -> dict[int, float]:
    prod_data = {}

    ag_biodiv_impacts_j = {j: 1 - x for j, x in biodiv_contr_ag_rj.items()}

    # Numpy magic
    ag_biodiv_degr_r = np.vectorize(ag_biodiv_impacts_j.get)(lumap)
    
    for s in range(GBF4_raw_species_area_sr.shape[0]):
        prod_data[s] = GBF4_raw_species_area_sr[s, :] * ag_biodiv_degr_r

    return prod_data


def calc_nes_ag_area_for_year(
    nes_xr: np.ndarray, lumap: np.ndarray, ag_biodiv_degr_j: dict[int, float]
) -> dict[int, float]:
    prod_data = {}

    ag_biodiv_impacts_j = {j: 1 - x for j, x in ag_biodiv_degr_j.items()}

    # Numpy magic
    ag_biodiv_degr_r = np.vectorize(ag_biodiv_impacts_j.get)(lumap)
    
    for x in range(nes_xr.shape[0]):
        prod_data[x] = nes_xr[x, :] * ag_biodiv_degr_r

    return prod_data


class LogToFile:
    def __init__(self, log_path, mode:str='w'):
        self.log_path_stdout = f"{log_path}_stdout.log"
        self.log_path_stderr = f"{log_path}_stderr.log"
        self.mode = mode

    def __call__(self, func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Open files for writing here, ensuring they're only created upon function call
            with open(self.log_path_stdout, self.mode) as file_stdout, open(self.log_path_stderr, self.mode) as file_stderr:
                original_stdout = sys.stdout
                original_stderr = sys.stderr
                try:
                    sys.stdout = self.StreamToLogger(file_stdout, original_stdout)
                    sys.stderr = self.StreamToLogger(file_stderr, original_stderr)
                    return func(*args, **kwargs)
                except Exception as e:
                    # Capture the full traceback
                    exc_info = traceback.format_exc()
                    # Log the traceback to stderr log before re-raising the exception
                    sys.stderr.write(exc_info + '\n')
                    raise  # Re-raise the caught exception to propagate it
                finally:
                    # Reset stdout and stderr
                    sys.stdout = original_stdout
                    sys.stderr = original_stderr
        return wrapper

    class StreamToLogger(object):
        def __init__(self, file, orig_stream=None):
            self.file = file
            self.orig_stream = orig_stream

        def write(self, buf):
            if buf.strip():  # Check if buf is just whitespace/newline
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                formatted_buf = f"{timestamp} - {buf}"
            else:
                formatted_buf = buf  # If buf is just a newline/whitespace, don't prepend timestamp

            # Write to the original stream if it exists
            if self.orig_stream:
                self.orig_stream.write(formatted_buf)
            
            # Write to the log file
            self.file.write(formatted_buf)

        def flush(self):
            # Ensure content is written to disk
            self.file.flush()