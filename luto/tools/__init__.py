# Copyright 2022 Fjalar J. de Haan and Brett A. Bryan at Deakin University
#
# This file is part of LUTO 2.0.
#
# LUTO 2.0 is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# LUTO 2.0 is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with
# LUTO 2.0. If not, see <https://www.gnu.org/licenses/>.

"""
Pure helper functions and other tools.
"""

from datetime import datetime
import functools
import sys
import time
import os.path
import traceback
from typing import Tuple
from contextlib import redirect_stderr, redirect_stdout

import pandas as pd
import numpy as np
import numpy_financial as npf

import luto.economics.agricultural.quantity as ag_quantity
import luto.economics.non_agricultural.quantity as non_ag_quantity
import luto.settings as settings
from luto.ag_managements import AG_MANAGEMENTS_TO_LAND_USES


def amortise(cost, rate=settings.DISCOUNT_RATE, horizon=settings.AMORTISATION_PERIOD):
    """Return NPV of future `cost` amortised to annual value at discount `rate` over `horizon` years."""
    if settings.AMORTISE_UPFRONT_COSTS: return -1 * npf.pmt(rate, horizon, pv=cost, fv=0, when='begin')
    else: return cost


# def show_map(yr_cal):
#     """Show a plot of the lumap of `yr_cal`."""
#     plotmap(lumaps[yr_cal], labels=bdata.AGLU2DESC)


def get_production(data, yr_cal, ag_X_mrj, non_ag_X_rk, ag_man_X_mrj):
    """Return total production of commodities for a specific year...

    'data' is a sim.data or bdata like object, 'yr_cal' is calendar year, and X_mrj 
    is land-use and land management in mrj decision-variable format.

    Can return base year production (e.g., year = 2010) or can return production for 
    a simulated year if one exists (i.e., year = 2030) check sim.info()).

    Includes the impacts of land-use change, productivity increases, and 
    climate change on yield."""

    # Calculate year index (i.e., number of years since 2010)
    yr_idx = yr_cal - data.YR_CAL_BASE

    # Get the quantity of each commodity produced by agricultural land uses
    # Get the quantity matrices. Quantity array of shape m, r, p
    ag_q_mrp = ag_quantity.get_quantity_matrices(data, yr_idx)

    # Convert map of land-use in mrj format to mrp format
    ag_X_mrp = np.stack([ag_X_mrj[:, :, j] for p in range(data.NPRS)
                         for j in range(data.N_AG_LUS)
                         if data.LU2PR[p, j]
                         ], axis=2)

    # Sum quantities in product (PR/p) representation.
    ag_q_p = np.sum(ag_q_mrp * ag_X_mrp, axis=(0, 1), keepdims=False)

    # Transform quantities to commodity (CM/c) representation.
    ag_q_c = [sum(ag_q_p[p] for p in range(data.NPRS) if data.PR2CM[c, p])
              for c in range(data.NCMS)]

    # Get the quantity of each commodity produced by non-agricultural land uses
    # Quantity matrix in the shape of c, r, k
    q_crk = non_ag_quantity.get_quantity_matrix(data)
    non_ag_q_c = [(q_crk[c, :, :] * non_ag_X_rk).sum()
                  for c in range(data.NCMS)]

    # Get quantities produced by agricultural management options
    j2p = {j: [p for p in range(data.NPRS) if data.LU2PR[p, j]]
           for j in range(data.N_AG_LUS)}
    ag_man_q_mrp = ag_quantity.get_agricultural_management_quantity_matrices(
        data, ag_q_mrp, yr_idx)
    ag_man_q_c = np.zeros(data.NCMS)

    for am, am_lus in AG_MANAGEMENTS_TO_LAND_USES.items():
        am_j_list = [data.DESC2AGLU[lu] for lu in am_lus]
        current_ag_man_X_mrp = np.zeros(ag_q_mrp.shape, dtype=np.float32)

        for j in am_j_list:
            for p in j2p[j]:
                current_ag_man_X_mrp[:, :, p] = ag_man_X_mrj[am][:, :, j]

        ag_man_q_p = np.sum(
            ag_man_q_mrp[am] * current_ag_man_X_mrp, axis=(0, 1), keepdims=False)

        for c in range(data.NCMS):
            ag_man_q_c[c] += sum(ag_man_q_p[p]
                                 for p in range(data.NPRS) if data.PR2CM[c, p])

    # Return total commodity production as numpy array.
    total_q_c = [ag_q_c[c] + non_ag_q_c[c] + ag_man_q_c[c]
                 for c in range(data.NCMS)]
    return np.array(total_q_c)


# def get_production_copy(sim, yr_cal):
#     """Return total production of commodities for a specific year...

#     Can return base year production (e.g., year = 2010) or can return production for 
#     a simulated year if one exists (i.e., year = 2030) check sim.info()).

#     Includes the impacts of land-use change, productivity increases, and 
#     climate change on yield."""

#     # Calculate year index (i.e., number of years since 2010)
#     yr_idx = yr_cal - sim.data.YR_CAL_BASE

#     # Acquire local names for matrices and shapes.
#     lu2pr_pj = sim.data.LU2PR
#     pr2cm_cp = sim.data.PR2CM
#     nlus = sim.data.N_AG_LUS
#     nprs = sim.data.NPRS
#     ncms = sim.data.NCMS

#     # Get the maps in decision-variable format. 0/1 array of shape j, r
#     X_dry = sim.dvars[yr_cal][0].T
#     X_irr = sim.dvars[yr_cal][1].T

#     # Get the quantity matrices. Quantity array of shape m, r, p
#     q_mrp = get_quantity_matrices(sim.data, yr_idx)

#     X_dry_pr = [X_dry[j] for p in range(nprs) for j in range(nlus)
#                 if lu2pr_pj[p, j]]
#     X_irr_pr = [X_irr[j] for p in range(nprs) for j in range(nlus)
#                 if lu2pr_pj[p, j]]

#     # Quantities in product (PR/p) representation by land management (dry/irr).
#     q_dry_p = [q_mrp[0, :, p] @ X_dry_pr[p] for p in range(nprs)]
#     q_irr_p = [q_mrp[1, :, p] @ X_irr_pr[p] for p in range(nprs)]

#     # Transform quantities to commodity (CM/c) representation by land management (dry/irr).
#     q_dry_c = [sum(q_dry_p[p] for p in range(nprs) if pr2cm_cp[c, p])
#                for c in range(ncms)]
#     q_irr_c = [sum(q_irr_p[p] for p in range(nprs) if pr2cm_cp[c, p])
#                for c in range(ncms)]

#     # Total quantities in commodity (CM/c) representation.
#     q_c = [q_dry_c[c] + q_irr_c[c] for c in range(ncms)]

#     # Return total commodity production.
#     return np.array(q_c)


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


def get_base_am_vars(ncells, ncms, n_ag_lus):
    """
    Get the 2010 agricultural management option vars.
    It is assumed that no agricultural management options were used in 2010, 
    so get zero arrays in the correct format.
    """
    am_vars = {}
    for am in AG_MANAGEMENTS_TO_LAND_USES:
        am_vars[am] = np.zeros((ncms, ncells, n_ag_lus))

    return am_vars


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


def get_agroforestry_cells(lumap) -> np.ndarray:
    """
    Get an array with cells used for riparian plantings
    """
    return np.nonzero(lumap == settings.NON_AGRICULTURAL_LU_BASE_CODE + 2)[0]


def get_ag_natural_lu_cells(data, lumap) -> np.ndarray:
    """
    Gets all cells being used for agricultural natural land uses.
    """
    return np.isin(lumap, data.LU_NATURAL)


def get_non_ag_natural_lu_cells(data, lumap) -> np.ndarray:
    """
    Gets all cells being used for non-agricultural natural land uses.
    """
    return np.isin(lumap, data.NON_AG_LU_NATURAL)


def get_ag_and_non_ag_natural_lu_cells(data, lumap) -> np.ndarray:
    """
    Gets all cells being used for natural land uses, both agricultural and non-agricultural.
    """
    return np.isin(lumap, data.LU_NATURAL + data.NON_AG_LU_NATURAL)


# def get_ag_natural_lu_cells(sim, yr_cal) -> np.ndarray::
#     """
#     Gets all cells being used for natural land uses.
#     """
#     dvar = sim.ag_dvars[yr_cal]
#     dvar_nat = dvar[:,:,sim.data.LU_NATURAL]

#     dvar_nat_val = np.einsum('mrj -> r', dvar_nat)
#     dvar_nat_idx = np.nonzero(dvar_nat_val)[0]

#     return dvar_nat_idx, dvar_nat_val[dvar_nat_idx]



def timethis(function, *args, **kwargs):
    """Generic wrapper to time functions."""

    # Start the wall clock.
    start = time.time()
    start_time = time.localtime()
    start_time_str = time.strftime("%Y-%m-%d %H:%M:%S", start_time)

    # Call the function.
    return_value = function(*args, **kwargs)

    # Stop the wall clock.
    stop = time.time()
    stop_time = time.localtime()
    stop_time_str = time.strftime("%Y-%m-%d %H:%M:%S", stop_time)

    print()
    print("Start time: %s" % start_time_str)
    print("Stop time: %s" % stop_time_str)
    print("Elapsed time: %d seconds." % (stop - start))

    return return_value


def mergeorderly(dict1, dict2):
    """Return merged dictionary with keys in alphabetic order."""
    list1 = list(dict1.keys())
    list2 = list(dict2.keys())
    lst = sorted(list1 + list2)
    merged = {}
    for key in lst:
        if key in list1:
            merged[key] = dict1[key]
        else:
            merged[key] = dict2[key]
    return merged


def crosstabulate(before, after, labels):
    """Return a cross-tabulation matrix as a Pandas DataFrame."""

    # The base DataFrame
    ct = pd.crosstab(before, after, margins=True)

    # Replace the numbered row labels with corresponding entries in `labels`.
    rows = [labels[int(i)] for i in ct.index if i != 'All']
    # The sum row was excluded, now add it back.
    rows.append('All')

    # Replace the numbered column labels with corresponding entries in `labels`.
    cols = [labels[int(i)] for i in ct.columns if i != 'All']
    # The sum column was excluded, now add it back.
    cols.append('All')

    # Replace the row and column labels.
    ct.index = rows
    ct.columns = cols

    return ct


def ctabnpystocsv(before, after, labels):
    """Write cross-tabulation of `before` and `after` from .npys to .csv."""

    # Load the .npy arrays.
    pre = np.load(before)
    post = np.load(after)

    # Get the file names as strings without extension.
    prename = os.path.splitext(os.path.basename(before))[0]
    postname = os.path.splitext(os.path.basename(after))[0]
    dirname = os.path.dirname(before)

    # Produce the cross tabulation and write the file to dir of first file.
    ct = crosstabulate(pre, post, labels)
    ct.to_csv(os.path.join(prename + str(2) + postname + ".csv"))


def inspect(lumap, highpos, d_j, q_rj, c_rj, landuses):

    # Prepare the pandas.DataFrame.
    columns = ["Pre Cells [#]", "Pre Costs [AUD]", "Pre Deviation [units]", "Pre Deviation [%]", "Post Cells [#]", "Post Costs [AUD]", "Post Deviation [units]", "Post Deviation [%]",
               "Delta Total Cells [#]", "Delta Total Cells [%]", "Delta Costs [AUD]", "Delta Costs [%]", "Delta Moved Cells [#]", "Delta Deviation [units]", "Delta Deviation [%]"]
    index = landuses
    df = pd.DataFrame(index=index, columns=columns)

    precost = 0
    postcost = 0

    # Reduced land-use list with half of the land-uses.
    lus = [lu for lu in landuses if '_dry' in lu]

    for j, lu in enumerate(lus):
        k = 2*j
        prewhere_dry = np.where(lumap == k, 1, 0)
        prewhere_irr = np.where(lumap == k+1, 1, 0)
        precount_dry = prewhere_dry.sum()
        precount_irr = prewhere_irr.sum()
        prequantity_dry = prewhere_dry @ q_rj.T[k]
        prequantity_irr = prewhere_irr @ q_rj.T[k+1]

        precost_dry = prewhere_dry @ c_rj.T[k]
        precost_irr = prewhere_irr @ c_rj.T[k+1]

        postwhere_dry = np.where(highpos == k, 1, 0)
        postwhere_irr = np.where(highpos == k+1, 1, 0)
        postcount_dry = postwhere_dry.sum()
        postcount_irr = postwhere_irr.sum()
        postquantity_dry = postwhere_dry @ q_rj.T[k]
        postquantity_irr = postwhere_irr @ q_rj.T[k+1]

        postcost_dry = postwhere_dry @ c_rj.T[k]
        postcost_irr = postwhere_irr @ c_rj.T[k+1]

        predeviation = prequantity_dry + prequantity_irr - d_j[j]
        predevfrac = predeviation / d_j[j]

        postdeviation = postquantity_dry + postquantity_irr - d_j[j]
        postdevfrac = postdeviation / d_j[j]

        df.loc[lu] = [precount_dry, precost_dry, predeviation, 100 * predevfrac, postcount_dry, postcost_dry, postdeviation, 100 * postdevfrac, postcount_dry - precount_dry, 100 * (postcount_dry - precount_dry) / precount_dry, postcost_dry - precost_dry, 100 * (postcost_dry - precost_dry) / precost_dry, np.sum(postwhere_dry - (prewhere_dry*postwhere_dry)), np.abs(postdeviation) - np.abs(predeviation), 100 * (np.abs(postdeviation) - np.abs(predeviation)
                                                                                                                                                                                                                                                                                                                                                                                                                            / predeviation)]

        lu_irr = lu.replace('_dry', '_irr')
        df.loc[lu_irr] = [precount_irr, precost_irr, predeviation, 100 * predevfrac, postcount_irr, postcost_irr, postdeviation, 100 * postdevfrac, postcount_irr - precount_irr, 100 * (postcount_irr - precount_irr) / precount_irr, postcost_irr - precost_irr, 100 * (postcost_irr - precost_irr) / precost_irr, np.sum(postwhere_irr - (prewhere_irr*postwhere_irr)), np.abs(postdeviation) - np.abs(predeviation), 100 * (np.abs(postdeviation) - np.abs(predeviation)
                                                                                                                                                                                                                                                                                                                                                                                                                                / predeviation)]

        precost += prewhere_dry @ c_rj.T[k] + prewhere_irr @ c_rj.T[k+1]
        postcost += postwhere_dry @ c_rj.T[k] + postwhere_irr @ c_rj.T[k+1]

    df.loc['total'] = [df[col].sum() for col in df.columns]

    return df


def get_water_delta_matrix(w_mrj, l_mrj, data):
    """
    Gets the water delta matrix that applies the cost of installing/removing irrigation to
    base transition costs. Includes the costs of water license fees.
    """
    # Get water requirements from current agriculture, converting water requirements for LVSTK from ML per head to ML per cell (inc. REAL_AREA).
    # Sum total water requirements of current land-use and land management
    w_r = (w_mrj * l_mrj).sum(axis=0).sum(axis=1)

    # Net water requirements calculated as the diff in water requirements between current land-use and all other land-uses j.
    w_net_mrj = w_mrj - w_r[:, np.newaxis]

    # Water license cost calculated as net water requirements (ML/ha) x licence price ($/ML).
    w_delta_mrj = w_net_mrj * data.WATER_LICENCE_PRICE[:, np.newaxis]

    # When land-use changes from dryland to irrigated add $7.5k per hectare for establishing irrigation infrastructure
    new_irrig_cost = 7500 * data.REAL_AREA[:, np.newaxis]
    w_delta_mrj[1] = np.where(
        l_mrj[0], w_delta_mrj[1] + new_irrig_cost, w_delta_mrj[1])

    # When land-use changes from irrigated to dryland add $3k per hectare for removing irrigation infrastructure
    remove_irrig_cost = 3000 * data.REAL_AREA[:, np.newaxis]
    w_delta_mrj[0] = np.where(
        l_mrj[1], w_delta_mrj[0] + remove_irrig_cost, w_delta_mrj[0])

    # Amortise upfront costs to annualised costs
    w_delta_mrj = amortise(w_delta_mrj)
    return w_delta_mrj


def am_name_snake_case(am_name):
    """Get snake_case version of the AM name"""
    return am_name.lower().replace(' ', '_')


def summarize_ghg_separate_df(in_array, column_level, lu_desc):
    '''Function to summarize the in_array to a df
    Arguments:
        in_array: a n-d np.array with the first dimension to be pixels/rows (dimension r)

        column_level: The levels of the in_array being reshaped to (r,-1). For example, if
                      the in_array has a shape of (r,2,3), then the levels could be a tuple 
                      of list as below. Note here add a ['Agricultural Landuse] as an extra
                      level to indicate the origin of this array.

                      (['Agricultural Landuse],
                       ['dry','irri']),
                       ['chemical_co2_emission','transportation_co2_emission']).

        lu_desc: The description of each pixel. 

    Return:
        pd.DataFrame: A multilevel (column-wise) df.
    '''

    # warp the array back to a df
    df = pd.DataFrame(in_array.reshape(
        (in_array.shape[0], -1)), columns=pd.MultiIndex.from_product(column_level))

    # add landuse describtion
    df['lu'] = lu_desc

    # sumarize the column
    df_summary = df.groupby('lu').sum(0).reset_index()
    df_summary = df_summary.set_index('lu')

    # add SUM row/index
    df_summary.loc['SUM'] = df_summary.sum(axis=0)
    df_summary['SUM'] = df_summary.sum(axis=1)

    # remove column/index names
    df_summary.columns = pd.MultiIndex.from_tuples(df_summary.columns.tolist())
    df_summary.index = df_summary.index.tolist()

    return df_summary


# function to create mapping table between lu_desc and dvar index
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



class LogToFile:
    def __init__(self, log_path):
        self.log_path_stdout = f"{log_path}_stdout.log"
        self.log_path_stderr = f"{log_path}_stderr.log"

    def __call__(self, func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Open files for writing here, ensuring they're only created upon function call
            with open(self.log_path_stdout, 'w') as file_stdout, open(self.log_path_stderr, 'w') as file_stderr:
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
            if buf.strip():  # Only prepend timestamp to non-newline content
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                formatted_buf = f"{timestamp} - {buf}"
            else:
                formatted_buf = buf  # If buf is just a newline/whitespace, don't prepend timestamp
            
            if self.orig_stream:
                self.orig_stream.write(formatted_buf)  # Write to the original stream if it exists
            self.file.write(formatted_buf)  # Write to the log file

        def flush(self):
            self.file.flush()  # Ensure content is written to disk

        def __init__(self, file, orig_stream=None):
            self.file = file
            self.orig_stream = orig_stream

        def write(self, buf):
            if buf.strip():  # Check if buf is not just whitespace/newline
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