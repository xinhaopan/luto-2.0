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
Writes model output and statistics to files.
"""


import os, re
import shutil
import threading
import numpy as np
import pandas as pd
import rasterio
import xarray as xr

from joblib import Parallel, delayed

from luto import settings
from luto import tools
from luto.data import Data
from luto.tools.spatializers import create_2d_map
from luto.tools.compmap import lumap_crossmap, lmmap_crossmap, crossmap_irrstat, crossmap_amstat

import luto.economics.agricultural.quantity as ag_quantity                      # ag_quantity has already been calculated and stored in <sim.prod_data>
import luto.economics.agricultural.revenue as ag_revenue
import luto.economics.agricultural.cost as ag_cost
import luto.economics.agricultural.transitions as ag_transitions
import luto.economics.agricultural.ghg as ag_ghg
import luto.economics.agricultural.water as ag_water
import luto.economics.agricultural.biodiversity as ag_biodiversity

import luto.economics.non_agricultural.quantity as non_ag_quantity              # non_ag_quantity has already been calculated and stored in <sim.prod_data>
import luto.economics.non_agricultural.revenue as non_ag_revenue
import luto.economics.non_agricultural.cost as non_ag_cost
import luto.economics.non_agricultural.transitions as non_ag_transitions
import luto.economics.non_agricultural.ghg as non_ag_ghg
import luto.economics.non_agricultural.water as non_ag_water
import luto.economics.non_agricultural.biodiversity as non_ag_biodiversity

from luto.settings import NON_AG_LAND_USES
from luto.tools.report.create_report_data import save_report_data
from luto.tools.report.create_html import data2html
from luto.tools.report.create_static_maps import TIF2MAP


# Global timestamp for the run
timestamp = tools.write_timestamp()
          
def write_outputs(data: Data):
    """Write model outputs to file"""

   # Start recording memory usage
    stop_event = threading.Event()
    memory_thread = threading.Thread(target=tools.log_memory_usage, args=(settings.OUTPUT_DIR, 'a',1, stop_event))
    memory_thread.start()
    
    try:
        write_data(data)
        move_logs(data)
    except Exception as e:
        print(f"An error occurred while writing outputs: {e}")
        raise e
    finally:
        # Ensure the memory logging thread is stopped
        stop_event.set()
        memory_thread.join()



@tools.LogToFile(f"{settings.OUTPUT_DIR}/write_{timestamp}")
def write_data(data: Data):

    years = [i for i in settings.SIM_YEARS if i<=data.last_year]
    data.set_path()
    paths = [f"{data.path}/out_{yr}" for yr in years]

    write_settings(data.path)
    write_area_transition_start_end(data, f'{data.path}/out_{years[-1]}', data.last_year)

    # Wrap write to a list of delayed jobs
    jobs = [delayed(write_output_single_year)(data, yr, path_yr, None) for (yr, path_yr) in zip(years, paths)]
    jobs += [delayed(write_output_single_year)(data, years[-1], f"{data.path_begin_end_compare}/out_{years[-1]}", years[0])]

    # Parallel write the outputs for each year
    num_jobs = (
        min(len(jobs), settings.WRITE_THREADS) 
        if settings.PARALLEL_WRITE 
        else 1   
    )
    Parallel(n_jobs=num_jobs)(jobs)

    # Copy the base-year outputs to the path_begin_end_compare
    shutil.copytree(f"{data.path}/out_{years[0]}", f"{data.path_begin_end_compare}/out_{years[0]}", dirs_exist_ok = True)
    
    # Create the report HTML and png maps
    TIF2MAP(data.path) if settings.WRITE_OUTPUT_GEOTIFFS else None
    save_report_data(data.path)
    data2html(data.path)



def move_logs(data: Data):
    
    logs = [
        f"{settings.OUTPUT_DIR}/run_{timestamp}_stdout.log",
        f"{settings.OUTPUT_DIR}/run_{timestamp}_stderr.log",
        f"{settings.OUTPUT_DIR}/write_{timestamp}_stdout.log",
        f"{settings.OUTPUT_DIR}/write_{timestamp}_stderr.log",
        f'{settings.OUTPUT_DIR}/RES_{settings.RESFACTOR}_mem_log.txt',
        f'{settings.OUTPUT_DIR}/.timestamp'
    ]

    for log in logs:
        try: shutil.move(log, f"{data.path}/{os.path.basename(log)}")
        except: pass



def write_output_single_year(data: Data, yr_cal, path_yr, yr_cal_sim_pre=None):
    """Write outputs for simulation 'sim', calendar year, demands d_c, and path"""

    years = sorted(list(data.lumaps.keys()))

    if not os.path.isdir(path_yr):
        os.makedirs(path_yr, exist_ok=True)

    write_files(data, yr_cal, path_yr)
    write_files_separate(data, yr_cal, path_yr) if settings.WRITE_OUTPUT_GEOTIFFS else None
    write_dvar_area(data, yr_cal, path_yr)
    write_crosstab(data, yr_cal, path_yr, yr_cal_sim_pre)
    write_quantity(data, yr_cal, path_yr, yr_cal_sim_pre)
    write_revenue_cost_ag(data, yr_cal, path_yr)
    write_revenue_cost_ag_management(data, yr_cal, path_yr)
    write_revenue_cost_non_ag(data, yr_cal, path_yr)
    write_cost_transition(data, yr_cal, path_yr)
    write_water(data, yr_cal, path_yr)
    write_ghg(data, yr_cal, path_yr)
    write_ghg_separate(data, yr_cal, path_yr)
    write_ghg_offland_commodity(data, yr_cal, path_yr)
    write_biodiversity_overall_priority_scores(data, yr_cal, path_yr)
    write_biodiversity_GBF2_scores(data, yr_cal, path_yr)
    write_biodiversity_GBF3_scores(data, yr_cal, path_yr)
    write_biodiversity_GBF4_SNES_scores(data, yr_cal, path_yr)
    write_biodiversity_GBF4_ECNES_scores(data, yr_cal, path_yr)
    write_biodiversity_GBF8_scores_groups(data, yr_cal, path_yr)
    write_biodiversity_GBF8_scores_species(data, yr_cal, path_yr)

    write_quantity_separate(data, yr_cal, path_yr)
    write_biodiversity(data, yr_cal, path_yr)
    write_npy(data, yr_cal, path_yr)

    print(f"Finished writing {yr_cal} out of {years[0]}-{years[-1]} years\n")


def get_settings(setting_path:str):
    with open(setting_path, 'r') as file:
        lines = file.readlines()
        # Regex patterns that matches variable assignments from settings
        parameter_reg = re.compile(r"^(\s*[A-Z].*?)\s*=")
        settings_order = [match[1].strip() for line in lines if (match := parameter_reg.match(line))]
        
        settings_dict = {i: getattr(settings, i) for i in dir(settings) if i.isupper()}
        settings_dict = {i: settings_dict[i] for i in settings_order if i in settings_dict}
        
    return settings_dict



def write_settings(path):
    settings_dict = get_settings('luto/settings.py')
    with open(os.path.join(path, 'model_run_settings.txt'), 'w') as f:
        f.writelines(f'{k}:{v}\n' for k, v in settings_dict.items())
        


def write_files(data: Data, yr_cal, path):
    """Writes numpy arrays and geotiffs to file"""

    print(f'Writing numpy arrays and geotiff outputs for {yr_cal}')

    # Save raw agricultural decision variables (float array).
    ag_X_mrj_fname = f'ag_X_mrj_{yr_cal}.npy'
    np.save(os.path.join(path, ag_X_mrj_fname), data.ag_dvars[yr_cal])

    # Save raw non-agricultural decision variables (float array).
    non_ag_X_rk_fname = f'non_ag_X_rk_{yr_cal}.npy'
    np.save(os.path.join(path, non_ag_X_rk_fname), data.non_ag_dvars[yr_cal])

    # Save raw agricultural management decision variables (float array).
    for am in data.AG_MAN_DESC:
        snake_case_am = tools.am_name_snake_case(am)
        am_X_mrj_fname = f'ag_man_X_mrj_{snake_case_am}_{yr_cal}.npy'
        np.save(os.path.join(path, am_X_mrj_fname), data.ag_man_dvars[yr_cal][am])

    # Write out raw numpy arrays for land-use and land management
    lumap_fname = f'lumap_{yr_cal}.npy'
    lmmap_fname = f'lmmap_{yr_cal}.npy'
    np.save(os.path.join(path, lumap_fname), data.lumaps[yr_cal])
    np.save(os.path.join(path, lmmap_fname), data.lmmaps[yr_cal])

    # Get the Agricultural Management applied to each pixel
    ag_man_dvar = np.stack([np.einsum('mrj -> r', v) for _,v in data.ag_man_dvars[yr_cal].items()]).T   # (r, am)
    ag_man_dvar_mask = ag_man_dvar.sum(1) > 0.01            # Meaning that they have at least 1% of agricultural management applied
    ag_man_dvar = np.argmax(ag_man_dvar, axis=1) + 1        # Start from 1
    ag_man_dvar_argmax = np.where(ag_man_dvar_mask, ag_man_dvar, 0).astype(np.float32)

    # Get the non-agricultural landuse for each pixel
    non_ag_dvar = data.non_ag_dvars[yr_cal]                 # (r, k)
    non_ag_dvar_mask = non_ag_dvar.sum(1) > 0.01            # Meaning that they have at least 1% of non-agricultural landuse applied
    non_ag_dvar = np.argmax(non_ag_dvar, axis=1) + settings.NON_AGRICULTURAL_LU_BASE_CODE    # Start from 100
    non_ag_dvar_argmax = np.where(non_ag_dvar_mask, non_ag_dvar, 0).astype(np.float32)

    with rasterio.open(os.path.join(path, f'lumap_{yr_cal}.tiff'), 'w+', **data.GEO_META) as dst_lumap,\
         rasterio.open(os.path.join(path, f'lmmap_{yr_cal}.tiff'), 'w+', **data.GEO_META) as dst_lmmap,\
         rasterio.open(os.path.join(path, f'ammap_{yr_cal}.tiff'), 'w+', **data.GEO_META) as dst_ammap,\
         rasterio.open(os.path.join(path, f'non_ag_{yr_cal}.tiff'), 'w+', **data.GEO_META) as dst_non_ag:
             
        dst_lumap.write_band(1, create_2d_map(data, data.lumaps[yr_cal]))
        dst_lmmap.write_band(1, create_2d_map(data, data.lmmaps[yr_cal]))
        dst_ammap.write_band(1, create_2d_map(data, ag_man_dvar_argmax))
        dst_non_ag.write_band(1, create_2d_map(data, non_ag_dvar_argmax))




def write_files_separate(data: Data, yr_cal, path):
    '''Write raw decision variables to separate GeoTiffs'''

    print(f'Write raw decision variables to separate GeoTiffs for {yr_cal}')

    # Collapse the land management dimension (m -> [dry, irr])
    ag_dvar_rj = np.einsum('mrj -> rj', data.ag_dvars[yr_cal])    
    ag_dvar_rm = np.einsum('mrj -> rm', data.ag_dvars[yr_cal])    
    non_ag_rk = np.einsum('rk -> rk', data.non_ag_dvars[yr_cal])  
    ag_man_rj_dict = {am: np.einsum('mrj -> rj', ammap) for am, ammap in data.ag_man_dvars[yr_cal].items()}

    # Get the desc2dvar table.
    ag_dvar_map = pd.DataFrame({'Category': 'Ag_LU','lu_desc': data.AGRICULTURAL_LANDUSES,'dvar_idx': range(data.N_AG_LUS)}
        ).assign(dvar=[ag_dvar_rj[:, j] for j in range(data.N_AG_LUS)]
        ).reindex(columns=['Category', 'lu_desc', 'dvar_idx', 'dvar'])
    non_ag_dvar_map = pd.DataFrame({'Category': 'Non-Ag_LU','lu_desc': data.NON_AGRICULTURAL_LANDUSES,'dvar_idx': range(data.N_NON_AG_LUS)}
        ).assign(dvar=[non_ag_rk[:, k] for k in range(data.N_NON_AG_LUS)]
        ).reindex(columns=['Category', 'lu_desc', 'dvar_idx', 'dvar'])
    lm_dvar_map = pd.DataFrame({'Category': 'Land_Mgt','lu_desc': data.LANDMANS,'dvar_idx': range(data.NLMS)}
        ).assign(dvar=[ag_dvar_rm[:, j] for j in range(data.NLMS)]
        ).reindex(columns=['Category', 'lu_desc', 'dvar_idx', 'dvar'])
    ag_man_map = pd.concat([
        pd.DataFrame({'Category': 'Ag_Mgt', 'lu_desc': am, 'dvar_idx': [0]}
        ).assign(dvar=[np.einsum('rj -> r', am_dvar_rj)]
        ).reindex(columns=['Category', 'lu_desc', 'dvar_idx', 'dvar'])
        for am, am_dvar_rj in ag_man_rj_dict.items()
    ])

    # Export to GeoTiff
    desc2dvar_df = pd.concat([ag_dvar_map, ag_man_map, non_ag_dvar_map, lm_dvar_map])
    lucc_separate_dir = os.path.join(path, 'lucc_separate')
    os.makedirs(lucc_separate_dir, exist_ok=True)
    for _, row in desc2dvar_df.iterrows():
        category = row['Category']
        dvar_idx = row['dvar_idx']
        desc = row['lu_desc']
        dvar = create_2d_map(data, row['dvar'].astype(np.float32))
        fname = f'{category}_{dvar_idx:02}_{desc}_{yr_cal}.tiff'
        lucc_separate_path = os.path.join(lucc_separate_dir, fname)
        
        with rasterio.open(lucc_separate_path, 'w+', **data.GEO_META) as dst:
            dst.write_band(1, dvar)




def write_quantity(data: Data, yr_cal, path, yr_cal_sim_pre=None):
    '''Write quantity comparison between base year and target year.'''

    print(f'Writing quantity outputs for {yr_cal}')

    simulated_year_list = sorted(list(data.lumaps.keys()))
    yr_idx = yr_cal - data.YR_CAL_BASE
    yr_idx_sim = sorted(list(data.lumaps.keys())).index(yr_cal)
    yr_cal_sim_pre = simulated_year_list[yr_idx_sim - 1] if yr_cal_sim_pre is None else yr_cal_sim_pre

    # Calculate data for quantity comparison between base year and target year
    if yr_cal > data.YR_CAL_BASE:
        # Check if yr_cal_sim_pre meets the requirement
        assert data.YR_CAL_BASE <= yr_cal_sim_pre < yr_cal, f"yr_cal_sim_pre ({yr_cal_sim_pre}) must be >= {data.YR_CAL_BASE} and < {yr_cal}"

        # Get commodity production quantities produced in base year and target year
        prod_base = np.array(data.prod_data[yr_cal_sim_pre]['Production'])
        prod_targ = np.array(data.prod_data[yr_cal]['Production'])
        demands = data.D_CY[yr_idx]  # Get commodity demands for target year

        # Calculate differences
        abs_diff = prod_targ - demands
        prop_diff = (prod_targ / demands) * 100

        # Create pandas dataframe
        df = pd.DataFrame({
            'Commodity': [i[0].capitalize() + i[1:] for i in data.COMMODITIES],
            'Prod_base_year (tonnes, KL)': prod_base,
            'Prod_targ_year (tonnes, KL)': prod_targ,
            'Demand (tonnes, KL)': demands,
            'Abs_diff (tonnes, KL)': abs_diff,
            'Prop_diff (%)': prop_diff
        })

        # Save files to disk
        df['Year'] = yr_cal
        df.to_csv(os.path.join(path, f'quantity_comparison_{yr_cal}.csv'), index=False)

        # Write the production of each year to disk
        production_years = pd.DataFrame({yr_cal: data.prod_data[yr_cal]['Production']})
        production_years.insert(0, 'Commodity', [i[0].capitalize() + i[1:] for i in data.COMMODITIES])
        production_years = production_years.rename(columns={2011: 'Value (tonnes, KL)'})
        production_years['Year'] = yr_cal
        production_years.to_csv(os.path.join(path, f'quantity_production_kt_{yr_cal}.csv'), index=False)

        # --------------------------------------------------------------------------------------------
        # NOTE: non_ag_quantity is already calculated and stored in <sim.prod_data>
        # --------------------------------------------------------------------------------------------



def write_quantity_separate(data: Data, yr_cal, path):
    index_levels = ['Landuse Type', 'Landuse subtype', 'Landuse', 'Land management', 'Production (tonnes, KL)']
    if yr_cal == data.YR_CAL_BASE:
        ag_X_mrj = data.AG_L_MRJ
        non_ag_X_rk = data.NON_AG_L_RK
        ag_man_X_mrj = data.AG_MAN_L_MRJ_DICT

    else:
        ag_X_mrj = tools.lumap2ag_l_mrj(data.lumaps[yr_cal], data.lmmaps[yr_cal])
        non_ag_X_rk = tools.lumap2non_ag_l_mk(data.lumaps[yr_cal], len(settings.NON_AG_LAND_USES.keys()))
        ag_man_X_mrj = data.ag_man_dvars[yr_cal]

    # Calculate year index (i.e., number of years since 2010)
    yr_idx = yr_cal - data.YR_CAL_BASE
    ag_q_mrp = ag_quantity.get_quantity_matrices(data, yr_idx)
    # Convert map of land-use in mrj format to mrp format using vectorization
    ag_X_mrp = np.einsum('mrj,pj->mrp', ag_X_mrj, data.LU2PR.astype(bool))

    # Sum quantities in product (PR/p) representation.
    ag_qu_mrp = np.einsum('mrp,mrp->mrp', ag_q_mrp, ag_X_mrp)
    ag_qu_mrj = np.einsum('mrp,pj->mrj', ag_qu_mrp, data.LU2PR.astype(bool))
    ag_jm = np.einsum('mrj->jm', ag_qu_mrj)
    ag_df = pd.DataFrame(
        ag_jm.reshape(-1).tolist(),
        index=pd.MultiIndex.from_product(
            [['Agricultural Landuse'],
             ['Agricultural Landuse'],
             data.AGRICULTURAL_LANDUSES,
             data.LANDMANS])).reset_index()
    ag_df.columns = index_levels

    # Get the quantity by non-agricultural land uses----------------------------------------------------------------
    q_crk = non_ag_quantity.get_quantity_matrix(data, ag_q_mrp, data.LUMAP)
    non_ag_k = np.einsum('crk,rk->k', q_crk, non_ag_X_rk)
    non_ag_df = pd.DataFrame(
        non_ag_k,
        index=pd.MultiIndex.from_product(
            [['Non-agricultural Landuse'],
             ['Non-Agricultural Landuse'],
             settings.NON_AG_LAND_USES.keys(),
             ['None']])).reset_index()
    non_ag_df.columns = index_levels

    # Get the quantity of  by agricultural management-----------------------------------------------------------------
    ag_man_q_mrp = ag_quantity.get_agricultural_management_quantity_matrices(data, ag_q_mrp, yr_idx)
    j2p = {j: [p for p in range(data.NPRS) if data.LU2PR[p, j]]
           for j in range(data.N_AG_LUS)}
    # 创建一个字典存储结果
    ag_man_q_mrj_dict = {}
    for am, am_lus in settings.AG_MANAGEMENTS_TO_LAND_USES.items():
        if not settings.AG_MANAGEMENTS[am]:
            continue
        am_j_list = [data.DESC2AGLU[lu] for lu in am_lus]
        current_ag_man_X_mrp = np.zeros(ag_q_mrp.shape, dtype=np.float32)
        for j in am_j_list:
            for p in j2p[j]:
                current_ag_man_X_mrp[:, :, p] = ag_man_X_mrj[am][:, :, j]

        ag_man_qu_mrp = np.einsum('mrp,mrp->mrp', ag_man_q_mrp[am], current_ag_man_X_mrp)
        # print(am,np.sum(ag_man_qu_mrp),np.sum(ag_man_q_mrp[am]),np.sum(current_ag_man_X_mrp))
        ag_man_qu_mrj = np.einsum('mrp,pj->mrj', ag_man_qu_mrp, data.LU2PR.astype(bool))
        ag_man_q_mrj_dict[am] = ag_man_qu_mrj

    AM_dfs = []
    for am, am_lus in settings.AG_MANAGEMENTS_TO_LAND_USES.items():  # Agricultural managements contribution
        if not settings.AG_MANAGEMENTS[am]:
            continue
        am_mrj = ag_man_q_mrj_dict[am]
        am_jm = np.einsum('mrj->jm', am_mrj)
        df_am = pd.DataFrame(
            am_jm.reshape(-1).tolist(),
            index=pd.MultiIndex.from_product([
                ['Agricultural Management'],
                [am],
                data.AGRICULTURAL_LANDUSES,
                data.LANDMANS
            ])).reset_index()
        df_am.columns = index_levels
        AM_dfs.append(df_am)
    AM_df = pd.concat(AM_dfs)


    df = pd.concat([ag_df, non_ag_df, AM_df])
    df.to_csv(os.path.join(path, f'quantity_production_kt_separate_{yr_cal}.csv'), index=False)


def write_revenue_cost_ag(data: Data, yr_cal, path):
    """Calculate agricultural revenue. Takes a simulation object, a target calendar
       year (e.g., 2030), and an output path as input."""

    print(f'Writing agricultural revenue_cost outputs for {yr_cal}')

    yr_idx = yr_cal - data.YR_CAL_BASE
    ag_dvar_mrj = data.ag_dvars[yr_cal]

    # Get agricultural revenue/cost for year in mrjs format
    ag_rev_df_rjms = ag_revenue.get_rev_matrices(data, yr_idx, aggregate=False)
    ag_cost_df_rjms = ag_cost.get_cost_matrices(data, yr_idx, aggregate=False)

    # Expand the original df with zero values to convert it to a **mrjs** array
    ag_rev_rjms = ag_rev_df_rjms.reindex(columns=pd.MultiIndex.from_product(ag_rev_df_rjms.columns.levels), fill_value=0).values.reshape(-1, *ag_rev_df_rjms.columns.levshape)
    ag_cost_rjms = ag_cost_df_rjms.reindex(columns=pd.MultiIndex.from_product(ag_cost_df_rjms.columns.levels), fill_value=0).values.reshape(-1, *ag_cost_df_rjms.columns.levshape)

    # Multiply the ag_dvar_mrj with the ag_rev_mrj to get the ag_rev_jm
    ag_rev_jms = np.einsum('mrj,rjms -> jms', ag_dvar_mrj, ag_rev_rjms)
    ag_cost_jms = np.einsum('mrj,rjms -> jms', ag_dvar_mrj, ag_cost_rjms)

    # Put the ag_rev_jms into a dataframe
    df_rev = pd.DataFrame(ag_rev_jms.reshape(ag_rev_jms.shape[0],-1),
                          columns=pd.MultiIndex.from_product(ag_rev_df_rjms.columns.levels[1:]),
                          index=ag_rev_df_rjms.columns.levels[0])

    df_cost = pd.DataFrame(ag_cost_jms.reshape(ag_cost_jms.shape[0],-1),
                           columns=pd.MultiIndex.from_product(ag_cost_df_rjms.columns.levels[1:]),
                           index=ag_cost_df_rjms.columns.levels[0])

    # Reformat the revenue/cost matrix into a long dataframe
    df_rev = df_rev.melt(ignore_index=False).reset_index()
    df_rev.columns = ['Land-use', 'Water_supply', 'Type', 'Value ($)']
    df_rev['Year'] = yr_cal
    df_cost = df_cost.melt(ignore_index=False).reset_index()
    df_cost.columns = ['Land-use', 'Water_supply', 'Type', 'Value ($)']
    df_cost['Year'] = yr_cal

    # Save to file
    df_rev = df_rev.replace({'dry':'Dryland', 'irr':'Irrigated'})
    df_cost = df_cost.replace({'dry':'Dryland', 'irr':'Irrigated'})

    df_rev.to_csv(os.path.join(path, f'revenue_agricultural_commodity_{yr_cal}.csv'), index=False)
    df_cost.to_csv(os.path.join(path, f'cost_agricultural_commodity_{yr_cal}.csv'), index=False)


def write_revenue_cost_ag_management(data: Data, yr_cal, path):
    """Calculate agricultural management revenue and cost."""

    print(f'Writing agricultural management revenue_cost outputs for {yr_cal}')

    yr_idx = yr_cal - data.YR_CAL_BASE

    # Get the revenue/cost matirces for each agricultural land-use
    ag_rev_mrj = ag_revenue.get_rev_matrices(data, yr_idx)
    ag_cost_mrj = ag_cost.get_cost_matrices(data, yr_idx)

    # Get the revenuecost matrices for each agricultural management
    am_revenue_mat = ag_revenue.get_agricultural_management_revenue_matrices(data, ag_rev_mrj, yr_idx)
    am_cost_mat = ag_cost.get_agricultural_management_cost_matrices(data, ag_cost_mrj, yr_idx)

    revenue_am_dfs = []
    cost_am_dfs = []
    # Loop through the agricultural managements
    for am, am_desc in data.AG_MAN_LU_DESC.items():

        # Get the land use codes for the agricultural management
        am_code = [data.DESC2AGLU[desc] for desc in am_desc]

        # Get the revenue/cost matrix for the agricultural management
        am_rev = np.nan_to_num(am_revenue_mat[am])  # Replace NaNs with 0
        am_cost = np.nan_to_num(am_cost_mat[am])  # Replace NaNs with 0

        # Get the decision variable for each agricultural management
        am_dvar = data.ag_man_dvars[yr_cal][am][:, :, am_code]

        # Multiply the decision variable by revenue matrix
        am_rev_yr = np.einsum('mrj,mrj->jm', am_dvar, am_rev)
        am_cost_yr = np.einsum('mrj,mrj->jm', am_dvar, am_cost)

        # Reformat the revenue/cost matrix into a dataframe
        am_rev_yr_df = pd.DataFrame(am_rev_yr, columns=data.LANDMANS)
        am_rev_yr_df['Land-use'] = am_desc
        am_rev_yr_df = am_rev_yr_df.melt(id_vars='Land-use',
                                         value_vars=data.LANDMANS,
                                         var_name='Water_supply',
                                         value_name='Value ($)')
        am_rev_yr_df['Year'] = yr_cal
        am_rev_yr_df['Management Type'] = am

        am_cost_yr_df = pd.DataFrame(am_cost_yr, columns=data.LANDMANS)
        am_cost_yr_df['Land-use'] = am_desc
        am_cost_yr_df = am_cost_yr_df.melt(id_vars='Land-use',
                                           value_vars=data.LANDMANS,
                                           var_name='Water_supply',
                                           value_name='Value ($)')
        am_cost_yr_df['Year'] = yr_cal
        am_cost_yr_df['Management Type'] = am

        # Store the revenue/cost dataframes
        revenue_am_dfs.append(am_rev_yr_df)
        cost_am_dfs.append(am_cost_yr_df)

    # Concatenate the revenue/cost dataframes
    revenue_am_df = pd.concat(revenue_am_dfs)
    cost_am_df = pd.concat(cost_am_dfs)

    revenue_am_df = revenue_am_df.replace({'dry':'Dryland', 'irr':'Irrigated'})
    cost_am_df = cost_am_df.replace({'dry':'Dryland', 'irr':'Irrigated'})

    revenue_am_df.to_csv(os.path.join(path, f'revenue_agricultural_management_{yr_cal}.csv'), index=False)
    cost_am_df.to_csv(os.path.join(path, f'cost_agricultural_management_{yr_cal}.csv'), index=False)



def write_cost_transition(data: Data, yr_cal, path, yr_cal_sim_pre=None):
    """Calculate transition cost."""

    print(f'Writing transition cost outputs for {yr_cal}')

    # Retrieve list of simulation years (e.g., [2010, 2050] for snapshot or [2010, 2011, 2012] for timeseries)
    simulated_year_list = sorted(list(data.lumaps.keys()))
    # Get index of yr_cal in timeseries (e.g., if yr_cal is 2050 then yr_idx = 40)
    yr_idx = yr_cal - data.YR_CAL_BASE
    
    # Get index of yr_cal in simulated_year_list (e.g., if yr_cal is 2050 then yr_idx_sim = 2 if snapshot)
    yr_idx_sim = simulated_year_list.index(yr_cal)
    # Get index of year previous to yr_cal in simulated_year_list (e.g., if yr_cal is 2050 then yr_cal_sim_pre = 2010 if snapshot)
    yr_cal_sim_pre = simulated_year_list[yr_idx_sim - 1] if yr_cal_sim_pre is None else yr_cal_sim_pre


    # Get the decision variables for agricultural land-use
    ag_dvar = data.ag_dvars[yr_cal]                          # (m,r,j)
    # Get the non-agricultural decision variable
    non_ag_dvar = data.non_ag_dvars[yr_cal]                  # (r,k)


    #---------------------------------------------------------------------
    #              Agricultural land-use transition costs
    #---------------------------------------------------------------------
    
    # Get the base_year mrj matirx
    base_mrj = tools.lumap2ag_l_mrj(data.lumaps[yr_cal_sim_pre], data.lmmaps[yr_cal_sim_pre])

    # Get the transition cost matrices for agricultural land-use
    if yr_idx == 0:
        ag_transitions_cost_mat = {'Establishment cost': np.zeros((data.NLMS, data.NCELLS, data.N_AG_LUS)).astype(np.float32)}
    else:
        # Get the transition cost matrices for agricultural land-use
        ag_transitions_cost_mat = ag_transitions.get_transition_matrices_ag2ag_from_base_year(data, yr_idx, yr_cal_sim_pre, separate=True)

    # Convert the transition cost matrices to a DataFrame
    cost_dfs = []
    for from_lu_desc,from_lu_idx in data.DESC2AGLU.items():
        for from_lm_idx,from_lm in enumerate(data.LANDMANS):
            for cost_type in ag_transitions_cost_mat.keys():

                base_lu_arr = base_mrj[from_lm_idx, :, from_lu_idx]
                if base_lu_arr.sum() == 0: continue
                    
                arr_dvar = ag_dvar[:, base_lu_arr, :]                                   # Get the decision variable of the from land-use % from water-supply (mr*j) 
                arr_trans = ag_transitions_cost_mat[cost_type][:, base_lu_arr, :]       # Get the transition cost matrix of the from land-use % from water-supply (mr*j) 
                cost_arr = np.einsum('mrj,mrj->mj', arr_dvar, arr_trans).flatten()      # Calculate the cost array (mj flatten)

                arr_df = pd.DataFrame(
                        cost_arr,
                        index=pd.MultiIndex.from_product([data.LANDMANS, data.AGRICULTURAL_LANDUSES],
                        names=['To water-supply', 'To land-use']),
                        columns=['Cost ($)']
                ).reset_index()
                
                arr_df.insert(0, 'Type', cost_type)
                arr_df.insert(1, 'From water-supply', data.LANDMANS[from_lm_idx])
                arr_df.insert(2, 'From land-use', from_lu_desc)
                arr_df.insert(3, 'Year', yr_cal)
                
                cost_dfs.append(arr_df) 

    # Save the cost DataFrames
    cost_df = pd.concat(cost_dfs, axis=0)
    cost_df = cost_df.replace({'dry':'Dryland', 'irr':'Irrigated'})
    cost_df.to_csv(os.path.join(path, f'cost_transition_ag2ag_{yr_cal}.csv'), index=False)



    #---------------------------------------------------------------------
    #              Agricultural management transition costs
    #---------------------------------------------------------------------

    # The agricultural management transition cost are all zeros, so skip the calculation here
    # am_cost = ag_transitions.get_agricultural_management_transition_matrices(data)




    #--------------------------------------------------------------------
    #              Non-agricultural land-use transition costs (from ag to non-ag)
    #--------------------------------------------------------------------

    # Get the transition cost matirces for non-agricultural land-use
    if yr_idx == 0:
        non_ag_transitions_cost_mat = {
            k:{'Transition cost':np.zeros(data.NCELLS).astype(np.float32)}
            for k in NON_AG_LAND_USES.keys()
        }
    else:
        non_ag_transitions_cost_mat = non_ag_transitions.get_transition_matrix_ag2nonag(
            data, yr_idx, data.lumaps[yr_cal_sim_pre], data.lmmaps[yr_cal_sim_pre], separate=True
        )
    
    # Get all land use decision variables
    desc2lu_all = {**data.DESC2AGLU, **data.DESC2NONAGLU}
    
    cost_dfs = []
    for from_lu in desc2lu_all.keys():
        for from_lm in data.LANDMANS:
            for to_lu in NON_AG_LAND_USES.keys():
                for cost_type in non_ag_transitions_cost_mat[to_lu].keys():
                    
                    lu_idx = data.lumaps[yr_cal_sim_pre] == desc2lu_all[from_lu]                          # Get the land-use index of the from land-use (r)
                    lm_idx = data.lmmaps[yr_cal_sim_pre] == data.LANDMANS.index(from_lm)                  # Get the land-management index of the from land-management (r)
                    from_lu_idx = lu_idx & lm_idx                                                         # Get the land-use index of the from land-use (r*)
                    
                    arr_dvar = non_ag_dvar[from_lu_idx, data.NON_AGRICULTURAL_LANDUSES.index(to_lu)]      # Get the decision variable of the from land-use (r*) 
                    arr_trans = non_ag_transitions_cost_mat[to_lu][cost_type][from_lu_idx]                # Get the transition cost matrix of the unchanged land-use (r) 
            
                    if arr_dvar.size == 0:
                        continue
                    
                    cost_arr = np.einsum('r,r->', arr_trans, arr_dvar)                                    # Calculate the cost array
                    arr_df = pd.DataFrame([{
                        'From land-use': from_lu,
                        'From water-supply': from_lm,
                        'To land-use': to_lu,
                        'Cost type': cost_type,
                        'Cost ($)': cost_arr,
                        'Year': yr_cal
                    }])
                    
                    cost_dfs.append(arr_df)

    # Save the cost DataFrames
    if len(cost_dfs) == 0:
        # This is to avoid an error when concatenating an empty list
        cost_df = pd.DataFrame(columns=['From land-use', 'From water-supply', 'To land-use', 'Cost type', 'Cost ($)', 'Year'])
        cost_df.loc[0,'Year'] = yr_cal
    else:
        cost_df = pd.concat(cost_dfs, axis=0)
        cost_df = cost_df.replace({'dry':'Dryland', 'irr':'Irrigated'})
    cost_df.to_csv(os.path.join(path, f'cost_transition_ag2non_ag_{yr_cal}.csv'), index=False)



    #--------------------------------------------------------------------
    #              Non-agricultural land-use transition costs (from non-ag to ag)
    #--------------------------------------------------------------------

    # Get the transition cost matirces for non-agricultural land-use
    if yr_idx == 0:
        non_ag_transitions_cost_mat = {k:{'Transition cost':np.zeros((data.NLMS, data.NCELLS, data.N_AG_LUS)).astype(np.float32)}
                                        for k in NON_AG_LAND_USES.keys()}
    else:
        non_ag_transitions_cost_mat = non_ag_transitions.get_transition_matrix_nonag2ag(data,
                                                                                    yr_idx,
                                                                                    data.lumaps[yr_cal_sim_pre],
                                                                                    data.lmmaps[yr_cal_sim_pre],
                                                                                    separate=True)

    cost_dfs = []
    for non_ag_type in non_ag_transitions_cost_mat:
        for cost_type in non_ag_transitions_cost_mat[non_ag_type]:

            arr = non_ag_transitions_cost_mat[non_ag_type][cost_type]          # Get the transition cost matrix
            arr = np.einsum('mrj,mrj->mj', arr, ag_dvar)                       # Multiply the transition cost matrix by the cost of non-agricultural land-use


            arr_df = pd.DataFrame(arr.flatten(),
                                index=pd.MultiIndex.from_product([data.LANDMANS, data.AGRICULTURAL_LANDUSES],names=['Water supply', 'To land-use']),
                                columns=['Cost ($)']).reset_index()
            arr_df.insert(0, 'From land-use', non_ag_type)
            arr_df.insert(1, 'Cost type', cost_type)
            arr_df.insert(2, 'Year', yr_cal)
            cost_dfs.append(arr_df)

    # Save the cost DataFrames
    cost_df = pd.concat(cost_dfs, axis=0)
    cost_df = cost_df.replace({'dry':'Dryland', 'irr':'Irrigated'})
    cost_df.to_csv(os.path.join(path, f'cost_transition_non_ag2_ag_{yr_cal}.csv'), index=False)


def write_revenue_cost_non_ag(data: Data, yr_cal, path):
    """Calculate non_agricultural cost. """

    print(f'Writing non agricultural management cost outputs for {yr_cal}')
    non_ag_dvar = data.non_ag_dvars[yr_cal]
    yr_idx = yr_cal - data.YR_CAL_BASE

    # Get the non-agricultural revenue/cost matrices
    ag_r_mrj = ag_revenue.get_rev_matrices(data, yr_idx)
    ag_c_mrj = ag_cost.get_cost_matrices(data, yr_idx)
    non_ag_rev_mat = non_ag_revenue.get_rev_matrix(data, yr_cal, ag_r_mrj, data.lumaps[yr_cal])    # rk
    non_ag_cost_mat = non_ag_cost.get_cost_matrix(data, ag_c_mrj, data.lumaps[yr_cal], yr_cal)     # rk
    non_ag_rev_mat = np.nan_to_num(non_ag_rev_mat)
    non_ag_cost_mat = np.nan_to_num(non_ag_cost_mat)

    # Calculate the non-agricultural revenue and cost
    rev_non_ag = np.einsum('rk,rk->k', non_ag_dvar, non_ag_rev_mat)
    cost_non_ag = np.einsum('rk,rk->k', non_ag_dvar, non_ag_cost_mat)

    # Reformat the revenue/cost matrix into a dataframe
    rev_non_ag_df = pd.DataFrame(rev_non_ag.reshape(-1,1), columns=['Value ($)'])
    rev_non_ag_df['Year'] = yr_cal
    rev_non_ag_df['Land-use'] = NON_AG_LAND_USES.keys()

    cost_non_ag_df = pd.DataFrame(cost_non_ag.reshape(-1,1), columns=['Value ($)'])
    cost_non_ag_df['Year'] = yr_cal
    cost_non_ag_df['Land-use'] = NON_AG_LAND_USES.keys()

    # Save to disk
    rev_non_ag_df.to_csv(os.path.join(path, f'revenue_non_ag_{yr_cal}.csv'), index = False)
    cost_non_ag_df.to_csv(os.path.join(path, f'cost_non_ag_{yr_cal}.csv'), index = False)




def write_dvar_area(data: Data, yr_cal, path):

    # Reprot the process
    print(f'Writing area calculated from dvars for {yr_cal}')

    # Get the decision variables for the year, multiply them by the area of each pixel,
    # and sum over the landuse dimension (j/k)
    ag_area = np.einsum('mrj,r -> mj', data.ag_dvars[yr_cal], data.REAL_AREA)
    non_ag_area = np.einsum('rk,r -> k', data.non_ag_dvars[yr_cal], data.REAL_AREA)
    ag_man_area_dict = {
        am: np.einsum('mrj,r -> mj', ammap, data.REAL_AREA)
        for am, ammap in data.ag_man_dvars[yr_cal].items()
    }

    # Agricultural landuse
    df_ag_area = pd.DataFrame(ag_area.reshape(-1),
                                index=pd.MultiIndex.from_product([[yr_cal],
                                                                data.LANDMANS,
                                                                data.AGRICULTURAL_LANDUSES],
                                                                names=['Year', 'Water_supply','Land-use']),
                                columns=['Area (ha)']).reset_index()
    # Non-agricultural landuse
    df_non_ag_area = pd.DataFrame(non_ag_area.reshape(-1),
                                index=pd.MultiIndex.from_product([[yr_cal],
                                                                ['dry'],
                                                                NON_AG_LAND_USES.keys()],
                                                                names=['Year', 'Water_supply', 'Land-use']),
                                columns=['Area (ha)']).reset_index()

    # Agricultural management
    am_areas = []
    for am, am_arr in ag_man_area_dict.items():
        df_am_area = pd.DataFrame(am_arr.reshape(-1),
                                index=pd.MultiIndex.from_product([[yr_cal],
                                                                [am],
                                                                data.LANDMANS,
                                                                data.AGRICULTURAL_LANDUSES],
                                                                names=['Year', 'Type', 'Water_supply','Land-use']),
                                columns=['Area (ha)']).reset_index()
        am_areas.append(df_am_area)

    # Concatenate the dataframes
    df_am_area = pd.concat(am_areas)

    # Save to file
    df_ag_area = df_ag_area.replace({'dry':'Dryland', 'irr':'Irrigated'})
    df_non_ag_area = df_non_ag_area.replace({'dry':'Dryland', 'irr':'Irrigated'})
    df_am_area = df_am_area.replace({'dry':'Dryland', 'irr':'Irrigated'})

    df_ag_area.to_csv(os.path.join(path, f'area_agricultural_landuse_{yr_cal}.csv'), index = False)
    df_non_ag_area.to_csv(os.path.join(path, f'area_non_agricultural_landuse_{yr_cal}.csv'), index = False)
    df_am_area.to_csv(os.path.join(path, f'area_agricultural_management_{yr_cal}.csv'), index = False)


def write_area_transition_start_end(data: Data, path, yr_cal_end):

    print(f'Save transition matrix between start and end year\n')

    # Get the end year
    yr_cal_start = data.YR_CAL_BASE

    # Get the decision variables for the start year
    dvar_base = tools.lumap2ag_l_mrj(data.lumaps[yr_cal_start], data.lmmaps[yr_cal_start])

    # Calculate the transition matrix for agricultural land uses (start) to agricultural land uses (end)
    transitions_ag2ag = []
    for lu_idx, lu in enumerate(data.AGRICULTURAL_LANDUSES):
        dvar_target = data.ag_dvars[yr_cal_end][:,:,lu_idx]
        trans = np.einsum('mrj, mr, r -> j', dvar_base, dvar_target, data.REAL_AREA)
        trans_df = pd.DataFrame({lu:trans.flatten()}, index=data.AGRICULTURAL_LANDUSES)
        transitions_ag2ag.append(trans_df)
    transition_ag2ag = pd.concat(transitions_ag2ag, axis=1)

    # Calculate the transition matrix for agricultural land uses (start) to non-agricultural land uses (end)
    trainsitions_ag2non_ag = []
    for lu_idx, lu in enumerate(NON_AG_LAND_USES.keys()):
        dvar_target = data.non_ag_dvars[yr_cal_end][:,lu_idx]
        trans = np.einsum('mrj, r, r -> j', dvar_base, dvar_target, data.REAL_AREA)
        trans_df = pd.DataFrame({lu:trans.flatten()}, index=data.AGRICULTURAL_LANDUSES)
        trainsitions_ag2non_ag.append(trans_df)
    transition_ag2non_ag = pd.concat(trainsitions_ag2non_ag, axis=1)

    # Concatenate the two transition matrices
    transition = pd.concat([transition_ag2ag, transition_ag2non_ag], axis=1)
    transition = transition.stack().reset_index()
    transition.columns = ['From land-use','To land-use','Area (ha)']

    # Write the transition matrix to a csv file
    os.makedirs(path, exist_ok=True)
    transition.to_csv(os.path.join(path, f'transition_matrix_{yr_cal_start}_{yr_cal_end}.csv'), index=False)



def write_crosstab(data: Data, yr_cal, path, yr_cal_sim_pre=None):
    """Write out land-use and production data"""

    print(f'Writing area transition outputs for {yr_cal}')

    simulated_year_list = sorted(list(data.lumaps.keys()))
    yr_idx_sim = simulated_year_list.index(yr_cal)
    yr_cal_sim_pre = simulated_year_list[yr_idx_sim - 1] if yr_cal_sim_pre is None else yr_cal_sim_pre


    # Only perform the calculation if the yr_cal is not the base year
    if yr_cal > data.YR_CAL_BASE:

        # Check if yr_cal_sim_pre meets the requirement
        assert yr_cal_sim_pre >= data.YR_CAL_BASE and yr_cal_sim_pre < yr_cal,\
            f"yr_cal_sim_pre ({yr_cal_sim_pre}) must be >= {data.YR_CAL_BASE} and < {yr_cal}"

        print(f'Writing crosstab data for {yr_cal}')

        # LUS = ['Non-agricultural land'] + data.AGRICULTURAL_LANDUSES + NON_AG_LAND_USES.keys()
        ctlu, swlu = lumap_crossmap( data.lumaps[yr_cal_sim_pre]
                                   , data.lumaps[yr_cal]
                                   , data.AGRICULTURAL_LANDUSES
                                   , NON_AG_LAND_USES.keys()
                                   , data.REAL_AREA)

        ctlm, swlm = lmmap_crossmap( data.lmmaps[yr_cal_sim_pre]
                                   , data.lmmaps[yr_cal]
                                   , data.REAL_AREA
                                   , data.LANDMANS)

        cthp, swhp = crossmap_irrstat( data.lumaps[yr_cal_sim_pre]
                                     , data.lmmaps[yr_cal_sim_pre]
                                     , data.lumaps[yr_cal], data.lmmaps[yr_cal]
                                     , data.AGRICULTURAL_LANDUSES
                                     , NON_AG_LAND_USES.keys()
                                     , data.REAL_AREA)


        ctass = {}
        swass = {}
        for am in data.AG_MAN_DESC:
            ctas, swas = crossmap_amstat( am
                                        , data.lumaps[yr_cal_sim_pre]
                                        , data.ammaps[yr_cal_sim_pre][am]
                                        , data.lumaps[yr_cal]
                                        , data.ammaps[yr_cal][am]
                                        , data.AGRICULTURAL_LANDUSES
                                        , NON_AG_LAND_USES.keys()
                                        , data.REAL_AREA)
            ctass[am] = ctas
            swass[am] = swas

        ctlu['Year'] = yr_cal
        ctlm['Year'] = yr_cal
        cthp['Year'] = yr_cal

        ctlu.to_csv(os.path.join(path, f'crosstab-lumap_{yr_cal}.csv'), index=False)
        ctlm.to_csv(os.path.join(path, f'crosstab-lmmap_{yr_cal}.csv'), index=False)
        swlu.to_csv(os.path.join(path, f'switches-lumap_{yr_cal}.csv'), index=False)
        swlm.to_csv(os.path.join(path, f'switches-lmmap_{yr_cal}.csv'), index=False)
        cthp.to_csv(os.path.join(path, f'crosstab-irrstat_{yr_cal}.csv'), index=False)
        swhp.to_csv(os.path.join(path, f'switches-irrstat_{yr_cal}.csv'), index=False)

        for am in data.AG_MAN_DESC:
            am_snake_case = tools.am_name_snake_case(am).replace("_", "-")
            ctass[am]['Year'] = yr_cal
            ctass[am].to_csv(os.path.join(path, f'crosstab-amstat-{am_snake_case}_{yr_cal}.csv'), index=False)
            swass[am].to_csv(os.path.join(path, f'switches-amstat-{am_snake_case}_{yr_cal}.csv'), index=False)



def write_water(data: Data, yr_cal, path):
    """Calculate water yield totals. Takes a Data Object, a calendar year (e.g., 2030), and an output path as input."""

    print(f'Writing water outputs for {yr_cal}')

    yr_idx = yr_cal - data.YR_CAL_BASE
    
    # Get water water yield historical level, and the domestic water use
    w_limit_inside_luto = ag_water.get_water_net_yield_limit_for_regions_inside_LUTO(data)
    domestic_water_use = data.WATER_USE_DOMESTIC

    # Get the decision variables
    ag_dvar_mrj = tools.ag_mrj_to_xr(data, data.ag_dvars[yr_cal])
    non_ag_dvar_rj = tools.non_ag_rk_to_xr(data, data.non_ag_dvars[yr_cal])
    am_dvar_mrj = tools.am_mrj_to_xr(data, data.ag_man_dvars[yr_cal])

    # Get water use without climate change impact; i.e., providing 'water_dr_yield' and 'water_sr_yield' as with historical layers
    ag_w_mrj_base_yr = tools.ag_mrj_to_xr(data, ag_water.get_water_net_yield_matrices(data, yr_idx, data.WATER_YIELD_HIST_DR, data.WATER_YIELD_HIST_SR))
    non_ag_w_rk_base_yr = tools.non_ag_rk_to_xr(data, non_ag_water.get_w_net_yield_matrix(data, ag_w_mrj_base_yr.values, data.lumaps[yr_cal], yr_idx, data.WATER_YIELD_HIST_DR, data.WATER_YIELD_HIST_SR))
    wny_outside_luto_study_area_base_yr = xr.DataArray(
        list(ag_water.get_water_outside_luto_study_area_from_hist_level(data).values()),
        dims=['region'],
        coords={'region': [data.WATER_REGION_NAMES[k] for k in ag_water.get_water_outside_luto_study_area_from_hist_level(data)]},
    )

    # Get water use under climate change impact; i.e., not providing 'water_dr_yield' and 'water_sr_yield' arguments
    ag_w_mrj_CCI = ag_water.get_water_net_yield_matrices(data, yr_idx) - ag_w_mrj_base_yr
    non_ag_w_rk_CCI = non_ag_water.get_w_net_yield_matrix(data, ag_w_mrj_CCI.values, data.lumaps[yr_cal], yr_idx) - non_ag_w_rk_base_yr
    wny_outside_luto_study_area_CCI = np.array(list(ag_water.get_water_outside_luto_study_area(data, yr_cal).values())) - wny_outside_luto_study_area_base_yr

    # Water yield from agricultural management is a multiple agricultural water use, not affected by the climate change impact.
    ag_man_w_mrj = tools.am_mrj_to_xr(data, ag_water.get_agricultural_management_water_matrices(data, yr_idx) )

    water_yields_inside_luto = pd.DataFrame()
    water_other_records = pd.DataFrame()
    for reg_idx, w_limit_inside in w_limit_inside_luto.items():
        

        ind = data.WATER_REGION_INDEX_R[reg_idx]
        reg_name = data.WATER_REGION_NAMES[reg_idx]

        # Get the water net yield for ag, non-ag, and ag-man
        ag_wny = (ag_w_mrj_base_yr.isel(cell=ind) * ag_dvar_mrj.isel(cell=ind)
            ).sum('cell'
            ).to_dataframe('Water Net Yield (ML)'
            ).reset_index(
            ).assign(Type='Agricultural Landuse', Year=yr_cal, Region=reg_name)
        non_ag_wny = (non_ag_w_rk_base_yr.isel(cell=ind) * non_ag_dvar_rj.isel(cell=ind)
            ).sum('cell'
            ).to_dataframe('Water Net Yield (ML)'
            ).reset_index(
            ).assign(Type='Non-Agricultural Landuse', Year=yr_cal, Region=reg_name)
        am_wny = (am_dvar_mrj.isel(cell=ind) * ag_man_w_mrj.isel(cell=ind)).sum('cell').to_dataframe('Water Net Yield (ML)'
            ).reset_index(
            ).assign(Type='Agricultural Management', Year=yr_cal, Region=reg_name)
            
        water_yields_inside_luto = pd.concat([water_yields_inside_luto, ag_wny, non_ag_wny, am_wny], ignore_index=True)
            
        # Get the climate change impact, limit, and outside water yield for the region
        CCI_impact = (
            (ag_w_mrj_CCI.isel(cell=ind) * ag_dvar_mrj.isel(cell=ind)).sum() 
            + (non_ag_w_rk_CCI.isel(cell=ind) * non_ag_dvar_rj.isel(cell=ind)).sum()
            + wny_outside_luto_study_area_CCI.sel(region=reg_name)
        )
        

        wny_sum = (
             water_yields_inside_luto.query('Region == @reg_name')['Water Net Yield (ML)'].sum() 
            + wny_outside_luto_study_area_base_yr.sel(region=reg_name).values
            - domestic_water_use[reg_idx]
        )
        
        wny_limit_region = (
            w_limit_inside
            + wny_outside_luto_study_area_base_yr.sel(region=reg_name).values
            - domestic_water_use[reg_idx]   
        )
        
        water_other_records = pd.concat([water_other_records, pd.DataFrame([{
            'Year': yr_cal,
            'Region': reg_name,
            'Water yield outside LUTO (ML)': wny_outside_luto_study_area_base_yr.sel(region=reg_name).values,
            'Climate Change Impact (ML)': CCI_impact.values,
            'Domestic Water Use (ML)': domestic_water_use[reg_idx],
            'Water Yield Limit (ML)': wny_limit_region,
            'Water Net Yield (ML)': wny_sum,
        }])], ignore_index=True)
        
        
    # Save the water yield data
    water_yields_inside_luto.rename(columns={
        'lu':'Landuse',
        'am':'Agri-Management',
        'lm':'Water Supply'}
        ).replace({'dry':'Dryland', 'irr':'Irrigated'}
        ).dropna(axis=0, how='all'
        ).to_csv(os.path.join(path, f'water_yield_separate_{yr_cal}.csv'), index=False)
        
    water_other_records.to_csv(os.path.join(path, f'water_yield_limits_and_public_land_{yr_cal}.csv'), index=False)

def write_biodiversity(data: Data, yr_cal, path):
    """
    Write biodiversity info for a given year ('yr_cal'), simulation ('sim')
    and output path ('path').
    """
    if settings.BIODIVERSITY_TARGET_GBF_2 == 'off':
        return

    # Check biodiversity limits and report
    biodiv_limit = data.get_GBF2_target_for_yr_cal(yr_cal)

    print(f'Writing biodiversity outputs for {yr_cal}')

    # Get biodiversity score from model
    if yr_cal >= data.YR_CAL_BASE + 1:
        biodiv_score = data.prod_data[yr_cal]["BIO (GBF2) value (ha)"]
    else:
        # Return the base year biodiversity score
        biodiv_score = data.get_GBF2_target_for_yr_cal(data.YR_CAL_BASE)

    # Add to dataframe
    df = pd.DataFrame({
            'Variable':['Biodiversity score limit',
                        'Solve biodiversity score'],
            'Score':[biodiv_limit, biodiv_score]
            })

    # Save to file
    df['Year'] = yr_cal
    df.to_csv(os.path.join(path, f'biodiversity_targets_{yr_cal}.csv'), index = False)


def write_biodiversity_overall_priority_scores(data: Data, yr_cal, path):
    
    print(f'Writing biodiversity priority scores for {yr_cal}')
    
    yr_cal_previouse = sorted(data.lumaps.keys())[sorted(data.lumaps.keys()).index(yr_cal) - 1]
    yr_idx = yr_cal - data.YR_CAL_BASE
    
    # Get the decision variables for the year
    ag_dvar_mrj = tools.ag_mrj_to_xr(data, data.ag_dvars[yr_cal])
    ag_mam_dvar_mrj =  tools.am_mrj_to_xr(data, data.ag_man_dvars[yr_cal])
    non_ag_dvar_rk = tools.non_ag_rk_to_xr(data, data.non_ag_dvars[yr_cal])

    # Get the biodiversity scores b_mrj
    bio_ag_priority_mrj =  tools.ag_mrj_to_xr(data, ag_biodiversity.get_bio_overall_priority_score_matrices_mrj(data))   
    bio_am_priority_tmrj = tools.am_mrj_to_xr(data, ag_biodiversity.get_agricultural_management_biodiversity_matrices(data, bio_ag_priority_mrj.values, yr_idx))
    bio_non_ag_priority_rk = tools.non_ag_rk_to_xr(data, non_ag_biodiversity.get_breq_matrix(data,bio_ag_priority_mrj.values, data.lumaps[yr_cal_previouse]))

    # Calculate the biodiversity scores
    base_yr_score = np.einsum('j,mrj->', ag_biodiversity.get_ag_biodiversity_contribution(data), data.AG_L_MRJ)

    priority_ag = (ag_dvar_mrj * bio_ag_priority_mrj
        ).sum(['cell','lm']
        ).to_dataframe('Area Weighted Score (ha)'
        ).reset_index(
        ).assign(Relative_Contribution_Percentage = lambda x:( (x['Area Weighted Score (ha)'] / base_yr_score) * 100) 
        ).assign(Type='Agricultural Landuse', Year=yr_cal)

    priority_non_ag = (non_ag_dvar_rk * bio_non_ag_priority_rk
        ).sum(['cell']
        ).to_dataframe('Area Weighted Score (ha)'
        ).reset_index(
        ).assign(Relative_Contribution_Percentage = lambda x:( x['Area Weighted Score (ha)'] / base_yr_score * 100)
        ).assign(Type='Non-Agricultural land-use', Year=yr_cal)

    priority_am = (ag_mam_dvar_mrj * bio_am_priority_tmrj
        ).sum(['cell','lm'], skipna=False
        ).to_dataframe('Area Weighted Score (ha)'
        ).reset_index(
        ).assign(Relative_Contribution_Percentage = lambda x:( x['Area Weighted Score (ha)'] / base_yr_score * 100)
        ).dropna(
        ).assign(Type='Agricultural Management', Year=yr_cal)


    # Save the biodiversity scores
    pd.concat([ priority_ag, priority_non_ag, priority_am], axis=0
        ).rename(columns={
            'lu':'Landuse',
            'am':'Agri-Management',
            'Relative_Contribution_Percentage':'Contribution Relative to Base Year Level (%)'}
        ).reset_index(drop=True
        ).to_csv( os.path.join(path, f'biodiversity_overall_priority_scores_{yr_cal}.csv'), index=False)
    


def write_biodiversity_GBF2_scores(data: Data, yr_cal, path):

    # Do nothing if biodiversity limits are off and no need to report
    if settings.BIODIVERSITY_TARGET_GBF_2 == 'off':
        return

    print(f'Writing biodiversity GBF2 scores (PRIORITY) for {yr_cal}')
    
    # Unpack the ag managements and land uses
    am_lu_unpack = [(am, l) for am, lus in data.AG_MAN_LU_DESC.items() for l in lus]

    # Get decision variables for the year
    ag_dvar_mrj = tools.ag_mrj_to_xr(data, data.ag_dvars[yr_cal])
    non_ag_dvar_rk = tools.non_ag_rk_to_xr(data, data.non_ag_dvars[yr_cal])
    am_dvar_jri = tools.am_mrj_to_xr(data, data.ag_man_dvars[yr_cal]).stack(idx=('am', 'lu'))
    am_dvar_jri = am_dvar_jri.sel(idx=am_dvar_jri['idx'].isin(pd.MultiIndex.from_tuples(am_lu_unpack)))

    # Get the priority degraded areas score
    priority_degraded_area_score_r = xr.DataArray(
        data.BIO_PRIORITY_DEGRADED_AREAS_R,
        dims=['cell'],
        coords={'cell':range(data.NCELLS)}
    ).chunk({'cell': min(4096, data.NCELLS)}) # Chunking to save mem use

    # Get the impacts of each ag/non-ag/am to vegetation matrices
    ag_impact_j = xr.DataArray(
        ag_biodiversity.get_ag_biodiversity_contribution(data),
        dims=['lu'],
        coords={'lu':data.AGRICULTURAL_LANDUSES}
    )
    non_ag_impact_k = xr.DataArray(
        list(non_ag_biodiversity.get_non_ag_lu_biodiv_contribution(data).values()),
        dims=['lu'],
        coords={'lu':data.NON_AGRICULTURAL_LANDUSES}
    )
    am_impact_ir = xr.DataArray(
        np.stack([arr for _, v in ag_biodiversity.get_ag_management_biodiversity_contribution(data, yr_cal).items() for arr in v.values()]), 
        dims=['idx', 'cell'], 
        coords={
            'idx': pd.MultiIndex.from_tuples(am_lu_unpack, names=['am', 'lu']),
            'cell': range(data.NCELLS)}
    )

    # Get the total area of the priority degraded areas
    total_priority_degraded_area = data.BIO_PRIORITY_DEGRADED_AREAS_R.sum()

    GBF2_score_ag = (priority_degraded_area_score_r * ag_impact_j * ag_dvar_mrj
        ).sum(['cell','lm']
        ).to_dataframe('Area Weighted Score (ha)'
        ).reset_index(
        ).assign(Relative_Contribution_Percentage = lambda x:((x['Area Weighted Score (ha)'] / total_priority_degraded_area) * 100)
        ).assign(Type='Agricultural Landuse', Year=yr_cal)
    GBF2_score_non_ag = (priority_degraded_area_score_r * non_ag_impact_k * non_ag_dvar_rk
        ).sum(['cell']
        ).to_dataframe('Area Weighted Score (ha)'
        ).reset_index(
        ).assign(Relative_Contribution_Percentage = lambda x:(x['Area Weighted Score (ha)'] / total_priority_degraded_area * 100)
        ).assign(Type='Non-Agricultural land-use', Year=yr_cal)  
    GBF2_score_am = (priority_degraded_area_score_r * am_impact_ir * am_dvar_jri
        ).sum(['cell','lm'], skipna=False
        ).to_dataframe('Area Weighted Score (ha)'
        ).reset_index(allow_duplicates=True
        ).T.drop_duplicates(
        ).T.assign(Relative_Contribution_Percentage = lambda x:(x['Area Weighted Score (ha)'] / total_priority_degraded_area * 100)
        ).assign(Type='Agricultural Management', Year=yr_cal)
        
    # Fill nan to empty dataframes
    if GBF2_score_ag.empty:
        GBF2_score_ag.loc[0] = 0
        GBF2_score_ag = GBF2_score_ag.astype({'Type':str, 'lu':str,'Year':'int'})
        GBF2_score_ag.loc[0, ['Type', 'lu' ,'Year']] = ['Agricultural Landuse', 'Apples', yr_cal]

    if GBF2_score_non_ag.empty:
        GBF2_score_non_ag.loc[0] = 0
        GBF2_score_non_ag = GBF2_score_non_ag.astype({'Type':str, 'lu':str,'Year':'int'})
        GBF2_score_non_ag.loc[0, ['Type', 'lu' ,'Year']] = ['Agricultural Management', 'Apples', yr_cal]

    if GBF2_score_am.empty:
        GBF2_score_am.loc[0] = 0
        GBF2_score_am = GBF2_score_am.astype({'Type':str, 'lu':str,'Year':'int'})
        GBF2_score_am.loc[0, ['Type', 'lu' ,'Year']] = ['Non-Agricultural land-use', 'Environmental Plantings', yr_cal]
        
    # Save to disk  
    pd.concat([
            GBF2_score_ag,
            GBF2_score_non_ag,
            GBF2_score_am], axis=0
        ).assign( Priority_Target=(data.get_GBF2_target_for_yr_cal(yr_cal) / total_priority_degraded_area) * 100,
        ).rename(columns={
            'lu':'Landuse',
            'am':'Agri-Management',
            'Relative_Contribution_Percentage':'Contribution Relative to Pre-1750 Level (%)',
            'Priority_Target':'Priority Target (%)'}
        ).reset_index(drop=True
        ).to_csv(os.path.join(path, f'biodiversity_GBF2_priority_scores_{yr_cal}.csv'), index=False)
    
    
    
def write_biodiversity_GBF3_scores(data: Data, yr_cal: int, path) -> None:
        
    # Do nothing if biodiversity limits are off and no need to report
    if settings.BIODIVERSITY_TARGET_GBF_3 == 'off':
        return
    
    # Unpack the agricultural management land-use
    am_lu_unpack = [(am, l) for am, lus in data.AG_MAN_LU_DESC.items() for l in lus]

    # Get decision variables for the year
    ag_dvar_mrj = tools.ag_mrj_to_xr(data, data.ag_dvars[yr_cal])
    non_ag_dvar_rk = tools.non_ag_rk_to_xr(data, data.non_ag_dvars[yr_cal])
    am_dvar_jri = tools.am_mrj_to_xr(data, data.ag_man_dvars[yr_cal]).stack(idx=('am', 'lu'))
    am_dvar_jri = am_dvar_jri.sel(idx=am_dvar_jri['idx'].isin(pd.MultiIndex.from_tuples(am_lu_unpack)))


    # Get vegetation matrices for the year
    vegetation_score_vr = xr.DataArray(
        ag_biodiversity.get_GBF3_major_vegetation_matrices_vr(data), 
        dims=['group','cell'], 
        coords={'group':list(data.BIO_GBF3_ID2DESC.values()),  'cell':range(data.NCELLS)}
    ).chunk({'cell': min(4096, data.NCELLS), 'group': 1})

    # Get the impacts of each ag/non-ag/am to vegetation matrices
    ag_impact_j = xr.DataArray(
        ag_biodiversity.get_ag_biodiversity_contribution(data),
        dims=['lu'],
        coords={'lu':data.AGRICULTURAL_LANDUSES}
    )
    non_ag_impact_k = xr.DataArray(
        list(non_ag_biodiversity.get_non_ag_lu_biodiv_contribution(data).values()),
        dims=['lu'],
        coords={'lu':data.NON_AGRICULTURAL_LANDUSES}
    )
    am_impact_ir = xr.DataArray(
        np.stack([arr for _, v in ag_biodiversity.get_ag_management_biodiversity_contribution(data, yr_cal).items() for arr in v.values()]), 
        dims=['idx', 'cell'], 
        coords={
            'idx': pd.MultiIndex.from_tuples(am_lu_unpack, names=['am', 'lu']),
            'cell': range(data.NCELLS)}
    )
    
    # Get the base year biodiversity scores
    veg_base_score_score = pd.DataFrame({
            'group': data.BIO_GBF3_ID2DESC.values(), 
            'BASE_OUTSIDE_SCORE': data.BIO_GBF3_BASELINE_SCORE_OUTSIDE_LUTO, 
            'BASE_TOTAL_SCORE': data.BIO_GBF3_BASELINE_SCORE_ALL_AUSTRALIA}
        ).eval('Relative_Contribution_Percentage = BASE_OUTSIDE_SCORE / BASE_TOTAL_SCORE * 100')

    GBF3_score_ag = (vegetation_score_vr * ag_impact_j * ag_dvar_mrj
        ).sum(['cell','lm']
        ).to_dataframe('Area Weighted Score (ha)'
        ).reset_index(
        ).merge(veg_base_score_score
        ).eval('Relative_Contribution_Percentage = `Area Weighted Score (ha)` / BASE_TOTAL_SCORE * 100'
        ).assign(Type='Agricultural Landuse', Year=yr_cal)

    GBF3_score_am = (vegetation_score_vr * am_impact_ir * am_dvar_jri
        ).sum(['cell','lm'], skipna=False
        ).to_dataframe('Area Weighted Score (ha)'
        ).reset_index(allow_duplicates=True
        ).T.drop_duplicates(
        ).T.merge(veg_base_score_score,
        ).astype({'Area Weighted Score (ha)': 'float', 'BASE_TOTAL_SCORE': 'float'}
        ).eval('Relative_Contribution_Percentage = `Area Weighted Score (ha)` / BASE_TOTAL_SCORE * 100'
        ).assign(Type='Agricultural Management', Year=yr_cal)
        
    GBF3_score_non_ag = (vegetation_score_vr * non_ag_impact_k * non_ag_dvar_rk
        ).sum(['cell']
        ).to_dataframe('Area Weighted Score (ha)'
        ).reset_index(
        ).merge(veg_base_score_score,
        ).eval('Relative_Contribution_Percentage = `Area Weighted Score (ha)` / BASE_TOTAL_SCORE * 100'
        ).assign(Type='Non-Agricultural land-use', Year=yr_cal)

    # Concatenate the dataframes, rename the columns, and reset the index, then save to a csv file
    veg_base_score_score = veg_base_score_score.assign(
        Type='Outside LUTO study area', 
        Year=yr_cal, 
        lu='Outside LUTO study area'
    )
    pd.concat([
        GBF3_score_ag, 
        GBF3_score_am, 
        GBF3_score_non_ag,
        veg_base_score_score],axis=0
        ).rename(columns={
            'lu':'Landuse',
            'am':'Agri-Management',
            'group':'Vegetation Group',
            'Relative_Contribution_Percentage':'Contribution Relative to Pre-1750 Level (%)'}
        ).reset_index(drop=True
        ).to_csv(os.path.join(path, f'biodiversity_GBF3_scores_{yr_cal}.csv'), index=False)
        


def write_biodiversity_GBF4_SNES_scores(data: Data, yr_cal: int, path) -> None:
    if not settings.BIODIVERSITY_TARGET_GBF_4_SNES == "on":
        return
    
    print(f"Writing species of national environmental significance scores (GBF4 SNES) for {yr_cal}")
    
    # Unpack the agricultural management land-use
    am_lu_unpack = [(am, l) for am, lus in data.AG_MAN_LU_DESC.items() for l in lus]

    # Get decision variables for the year
    ag_dvar_mrj = tools.ag_mrj_to_xr(data, data.ag_dvars[yr_cal])
    non_ag_dvar_rk = tools.non_ag_rk_to_xr(data, data.non_ag_dvars[yr_cal])
    am_dvar_jri = tools.am_mrj_to_xr(data, data.ag_man_dvars[yr_cal]).stack(idx=('am', 'lu'))
    am_dvar_jri = am_dvar_jri.sel(idx=am_dvar_jri['idx'].isin(pd.MultiIndex.from_tuples(am_lu_unpack)))

    # Get the biodiversity scores for the year
    bio_snes_sr = xr.DataArray(
        ag_biodiversity.get_GBF4_SNES_matrix_sr(data), 
        dims=['species','cell'], 
        coords={'species':data.BIO_GBF4_SNES_SEL_ALL, 'cell':np.arange(data.NCELLS)}
    ).chunk({'cell': min(4096, data.NCELLS), 'species': 1})

    # Apply habitat contribution from ag/am/non-ag land-use to biodiversity scores
    ag_impact_j = xr.DataArray(
        ag_biodiversity.get_ag_biodiversity_contribution(data),
        dims=['lu'],
        coords={'lu':data.AGRICULTURAL_LANDUSES}
    )
    non_ag_impact_k = xr.DataArray(
        list(non_ag_biodiversity.get_non_ag_lu_biodiv_contribution(data).values()),
        dims=['lu'],
        coords={'lu':data.NON_AGRICULTURAL_LANDUSES}
    )
    am_impact_ir = xr.DataArray(
        np.stack([arr for _, v in ag_biodiversity.get_ag_management_biodiversity_contribution(data, yr_cal).items() for arr in v.values()]), 
        dims=['idx', 'cell'], 
        coords={
            'idx': pd.MultiIndex.from_tuples(am_lu_unpack, names=['am', 'lu']),
            'cell': np.arange(data.NCELLS)}
    )

    # Get the base year biodiversity scores
    bio_snes_scores = pd.read_csv(settings.INPUT_DIR + '/BIODIVERSITY_GBF4_TARGET_SNES.csv')
    idx_row = [bio_snes_scores.query('SCIENTIFIC_NAME == @i').index[0] for i in data.BIO_GBF4_SNES_SEL_ALL]
    idx_all_score = [bio_snes_scores.columns.get_loc(f'HABITAT_SIGNIFICANCE_BASELINE_ALL_AUSTRALIA_{col}') for col in data.BIO_GBF4_PRESENCE_SNES_SEL]
    idx_outside_score =  [bio_snes_scores.columns.get_loc(f'HABITAT_SIGNIFICANCE_BASELINE_OUT_LUTO_NATURAL_{col}') for col in data.BIO_GBF4_PRESENCE_SNES_SEL]

    base_yr_score = pd.DataFrame({
            'species': data.BIO_GBF4_SNES_SEL_ALL, 
            'BASE_TOTAL_SCORE': [bio_snes_scores.iloc[row, col] for row, col in zip(idx_row, idx_all_score)],
            'BASE_OUTSIDE_SCORE': [bio_snes_scores.iloc[row, col] for row, col in zip(idx_row, idx_outside_score)],
            'TARGET_INSIDE_SCORE': data.get_GBF4_SNES_target_inside_LUTO_by_year(yr_cal)}
    ).eval('Target_by_Percent = (TARGET_INSIDE_SCORE + BASE_OUTSIDE_SCORE) / BASE_TOTAL_SCORE * 100')

    # Calculate the biodiversity scores
    GBF4_score_ag = (bio_snes_sr * ag_impact_j * ag_dvar_mrj
        ).sum(['cell','lm']
        ).to_dataframe('Area Weighted Score (ha)'
        ).reset_index(
        ).merge(base_yr_score
        ).eval('Relative_Contribution_Percentage = `Area Weighted Score (ha)` / BASE_TOTAL_SCORE * 100'
        ).assign(Type='Agricultural Landuse', Year=yr_cal)
        
    GBF4_score_am = (bio_snes_sr * am_impact_ir * am_dvar_jri
        ).sum(['cell','lm'], skipna=False).to_dataframe('Area Weighted Score (ha)'
        ).reset_index(allow_duplicates=True
        ).T.drop_duplicates(
        ).T.merge(base_yr_score,
        ).astype({'Area Weighted Score (ha)': 'float', 'BASE_TOTAL_SCORE': 'float'}
        ).eval('Relative_Contribution_Percentage = `Area Weighted Score (ha)` / BASE_TOTAL_SCORE * 100'
        ).assign(Type='Agricultural Management', Year=yr_cal)
        
    GBF4_score_non_ag = (bio_snes_sr * non_ag_impact_k * non_ag_dvar_rk
        ).sum(['cell']
        ).to_dataframe('Area Weighted Score (ha)'
        ).reset_index(
        ).merge(base_yr_score,
        ).eval('Relative_Contribution_Percentage = `Area Weighted Score (ha)` / BASE_TOTAL_SCORE * 100'
        ).assign(Type='Non-Agricultural land-use', Year=yr_cal)
        
    
    # Concatenate the dataframes, rename the columns, and reset the index, then save to a csv file
    base_yr_score = base_yr_score.assign(Type='Outside LUTO study area', Year=yr_cal, lu='Outside LUTO study area'
        ).eval('Relative_Contribution_Percentage = BASE_OUTSIDE_SCORE / BASE_TOTAL_SCORE * 100')
    
    pd.concat([
            GBF4_score_ag, 
            GBF4_score_am, 
            GBF4_score_non_ag,
            base_yr_score], axis=0
        ).rename(columns={
            'lu':'Landuse',
            'am':'Agri-Management',
            'Relative_Contribution_Percentage':'Contribution Relative to Pre-1750 Level (%)',
            'Target_by_Percent':'Target by Percent (%)'}).reset_index(drop=True
        ).to_csv(os.path.join(path, f'biodiversity_GBF4_SNES_scores_{yr_cal}.csv'), index=False)
            



def write_biodiversity_GBF4_ECNES_scores(data: Data, yr_cal: int, path) -> None:
    
    if not settings.BIODIVERSITY_TARGET_GBF_4_ECNES == "on":
        return
    
    print(f"Writing ecological communities of national environmental significance scores (GBF4 ECNES) for {yr_cal}")
    
    # Unpack the agricultural management land-use
    am_lu_unpack = [(am, l) for am, lus in data.AG_MAN_LU_DESC.items() for l in lus]

    # Get decision variables for the year
    ag_dvar_mrj = tools.ag_mrj_to_xr(data, data.ag_dvars[yr_cal])
    non_ag_dvar_rk = tools.non_ag_rk_to_xr(data, data.non_ag_dvars[yr_cal])
    am_dvar_jri = tools.am_mrj_to_xr(data, data.ag_man_dvars[yr_cal]).stack(idx=('am', 'lu'))
    am_dvar_jri = am_dvar_jri.sel(idx=am_dvar_jri['idx'].isin(pd.MultiIndex.from_tuples(am_lu_unpack)))

    # Get the biodiversity scores for the year
    bio_ecnes_sr = xr.DataArray(
        ag_biodiversity.get_GBF4_ECNES_matrix_sr(data), 
        dims=['species','cell'], 
        coords={'species':data.BIO_GBF4_ECNES_SEL_ALL, 'cell':np.arange(data.NCELLS)}
    ).chunk({'cell': min(4096, data.NCELLS), 'species': 1})

    # Apply habitat contribution from ag/am/non-ag land-use to biodiversity scores
    ag_impact_j = xr.DataArray(
        ag_biodiversity.get_ag_biodiversity_contribution(data),
        dims=['lu'],
        coords={'lu':data.AGRICULTURAL_LANDUSES}
    )
    non_ag_impact_k = xr.DataArray(
        list(non_ag_biodiversity.get_non_ag_lu_biodiv_contribution(data).values()),
        dims=['lu'],
        coords={'lu': data.NON_AGRICULTURAL_LANDUSES}
    )
    am_impact_ir = xr.DataArray(
        np.stack([arr for _, v in ag_biodiversity.get_ag_management_biodiversity_contribution(data, yr_cal).items() for arr in v.values()]),
        dims=['idx', 'cell'],
        coords={
            'idx': pd.MultiIndex.from_tuples(am_lu_unpack, names=['am', 'lu']),
            'cell': np.arange(data.NCELLS)
        }
    )

    # Get the base year biodiversity scores
    bio_ecnes_scores = pd.read_csv(settings.INPUT_DIR + '/BIODIVERSITY_GBF4_TARGET_ECNES.csv')
    idx_row = [bio_ecnes_scores.query('COMMUNITY == @i').index[0] for i in data.BIO_GBF4_ECNES_SEL_ALL]
    idx_all_score = [bio_ecnes_scores.columns.get_loc(f'HABITAT_SIGNIFICANCE_BASELINE_ALL_AUSTRALIA_{col}') for col in data.BIO_GBF4_PRESENCE_ECNES_SEL]
    idx_outside_score = [bio_ecnes_scores.columns.get_loc(f'HABITAT_SIGNIFICANCE_BASELINE_OUT_LUTO_NATURAL_{col}') for col in data.BIO_GBF4_PRESENCE_ECNES_SEL]

    base_yr_score = pd.DataFrame({
        'species': data.BIO_GBF4_ECNES_SEL_ALL,
        'BASE_TOTAL_SCORE': [bio_ecnes_scores.iloc[row, col] for row, col in zip(idx_row, idx_all_score)],
        'BASE_OUTSIDE_SCORE': [bio_ecnes_scores.iloc[row, col] for row, col in zip(idx_row, idx_outside_score)],
        'TARGET_INSIDE_SCORE': data.get_GBF4_ECNES_target_inside_LUTO_by_year(yr_cal)
    }).eval('Target_by_Percent = (TARGET_INSIDE_SCORE + BASE_OUTSIDE_SCORE) / BASE_TOTAL_SCORE * 100')

    # Calculate the biodiversity scores
    GBF4_score_ag = (bio_ecnes_sr * ag_impact_j * ag_dvar_mrj
        ).sum(['cell', 'lm']
        ).to_dataframe('Area Weighted Score (ha)'
        ).reset_index(
        ).merge(base_yr_score
        ).eval('Relative_Contribution_Percentage = `Area Weighted Score (ha)` / BASE_TOTAL_SCORE * 100'
        ).assign(Type='Agricultural Landuse', Year=yr_cal)

    GBF4_score_am = (bio_ecnes_sr * am_impact_ir * am_dvar_jri
        ).sum(['cell', 'lm'], skipna=False).to_dataframe('Area Weighted Score (ha)'
        ).reset_index(allow_duplicates=True
        ).T.drop_duplicates(
        ).T.merge(base_yr_score,
        ).astype({'Area Weighted Score (ha)': 'float', 'BASE_TOTAL_SCORE': 'float'}
        ).eval('Relative_Contribution_Percentage = `Area Weighted Score (ha)` / BASE_TOTAL_SCORE * 100'
        ).assign(Type='Agricultural Management', Year=yr_cal)

    GBF4_score_non_ag = (bio_ecnes_sr * non_ag_impact_k * non_ag_dvar_rk
        ).sum(['cell']).to_dataframe('Area Weighted Score (ha)').reset_index(
        ).merge(base_yr_score,
        ).eval('Relative_Contribution_Percentage = `Area Weighted Score (ha)` / BASE_TOTAL_SCORE * 100'
        ).assign(Type='Non-Agricultural land-use', Year=yr_cal)

    # Concatenate the dataframes, rename the columns, and reset the index, then save to a csv file
    base_yr_score = base_yr_score.assign(Type='Outside LUTO study area', Year=yr_cal, lu='Outside LUTO study area'
        ).eval('Relative_Contribution_Percentage = BASE_OUTSIDE_SCORE / BASE_TOTAL_SCORE * 100')
    
    pd.concat([
            GBF4_score_ag,
            GBF4_score_am,
            GBF4_score_non_ag,
            base_yr_score], axis=0
        ).rename(columns={
            'lu':'Landuse',
            'am':'Agri-Management',
            'Relative_Contribution_Percentage': 'Contribution Relative to Pre-1750 Level (%)',
            'Target_by_Percent': 'Target by Percent (%)'}
        ).reset_index(drop=True
        ).to_csv(os.path.join(path, f'biodiversity_GBF4_ECNES_scores_{yr_cal}.csv'), index=False)
        
        

def write_biodiversity_GBF8_scores_groups(data: Data, yr_cal, path):
    
    # Do nothing if biodiversity limits are off and no need to report
    if not settings.BIODIVERSITY_TARGET_GBF_8 == 'on':
        return

    print(f'Writing biodiversity GBF8 scores (GROUPS) for {yr_cal}')
    
    # Unpack the agricultural management land-use
    am_lu_unpack = [(am, l) for am, lus in data.AG_MAN_LU_DESC.items() for l in lus]

    # Get decision variables for the year
    ag_dvar_mrj = tools.ag_mrj_to_xr(data, data.ag_dvars[yr_cal])
    non_ag_dvar_rk = tools.non_ag_rk_to_xr(data, data.non_ag_dvars[yr_cal])
    am_dvar_jri = tools.am_mrj_to_xr(data, data.ag_man_dvars[yr_cal]).stack(idx=('am', 'lu'))
    am_dvar_jri = am_dvar_jri.sel(idx=am_dvar_jri['idx'].isin(pd.MultiIndex.from_tuples(am_lu_unpack)))

    # Get biodiversity scores for selected species
    bio_scores_sr = xr.DataArray(
        data.get_GBF8_bio_layers_by_yr(yr_cal, level='group') * data.REAL_AREA[None,:],
        dims=['group','cell'],
        coords={
            'group': data.BIO_GBF8_GROUPS_NAMES,
            'cell': np.arange(data.NCELLS)}
    ).chunk({'cell': min(4096, data.NCELLS), 'group': 1})  # Chunking to save mem use
        
    # Get the habitat contribution for ag/non-ag/am land-use to biodiversity scores
    ag_impact_j = xr.DataArray(
        ag_biodiversity.get_ag_biodiversity_contribution(data),
        dims=['lu'],
        coords={'lu':data.AGRICULTURAL_LANDUSES}
    )
    non_ag_impact_k = xr.DataArray(
        list(non_ag_biodiversity.get_non_ag_lu_biodiv_contribution(data).values()),
        dims=['lu'],
        coords={'lu': data.NON_AGRICULTURAL_LANDUSES}
    )
    am_impact_ir = xr.DataArray(
        np.stack([arr for _, v in ag_biodiversity.get_ag_management_biodiversity_contribution(data, yr_cal).items() for arr in v.values()]),
        dims=['idx', 'cell'],
        coords={
            'idx': pd.MultiIndex.from_tuples(am_lu_unpack, names=['am', 'lu']),
            'cell': np.arange(data.NCELLS)}
    )

    # Get the base year biodiversity scores
    base_yr_score = pd.DataFrame({
            'group': data.BIO_GBF8_GROUPS_NAMES, 
            'BASE_OUTSIDE_SCORE': data.get_GBF8_score_outside_natural_LUTO_by_yr(yr_cal, level='group'),
            'BASE_TOTAL_SCORE': data.BIO_GBF8_BASELINE_SCORE_GROUPS['HABITAT_SUITABILITY_BASELINE_SCORE_ALL_AUSTRALIA']}
        ).eval('Relative_Contribution_Percentage = BASE_OUTSIDE_SCORE / BASE_TOTAL_SCORE * 100')

    # Calculate GBF8 scores for groups
    GBF8_scores_groups_ag = (bio_scores_sr * ag_impact_j * ag_dvar_mrj
        ).sum(['cell', 'lm']
        ).to_dataframe('Area Weighted Score (ha)'
        ).reset_index(
        ).merge(base_yr_score
        ).eval('Relative_Contribution_Percentage = `Area Weighted Score (ha)` / BASE_TOTAL_SCORE * 100'
        ).assign(Type='Agricultural Landuse', Year=yr_cal)
        
    GBF8_scores_groups_am = (am_dvar_jri * bio_scores_sr * am_impact_ir
        ).sum(['cell', 'lm']
        ).to_dataframe('Area Weighted Score (ha)'
        ).reset_index(allow_duplicates=True
        ).T.drop_duplicates(
        ).T.merge(base_yr_score
        ).astype({'Area Weighted Score (ha)': 'float', 'BASE_TOTAL_SCORE': 'float'}
        ).eval('Relative_Contribution_Percentage = `Area Weighted Score (ha)` / BASE_TOTAL_SCORE * 100'
        ).assign(Type='Agricultural Management', Year=yr_cal)
        
    GBF8_scores_groups_non_ag = (non_ag_dvar_rk * bio_scores_sr * non_ag_impact_k
        ).sum(['cell']
        ).to_dataframe('Area Weighted Score (ha)'
        ).reset_index(
        ).merge(base_yr_score
        ).eval('Relative_Contribution_Percentage = `Area Weighted Score (ha)` / BASE_TOTAL_SCORE * 100'
        ).assign(Type='Non-Agricultural land-use', Year=yr_cal)

    # Concatenate the dataframes, rename the columns, and reset the index, then save to a csv file
    base_yr_score = base_yr_score.assign(Type='Outside LUTO study area', Year=yr_cal) 

    pd.concat([
        GBF8_scores_groups_ag, 
        GBF8_scores_groups_am, 
        GBF8_scores_groups_non_ag,
        base_yr_score], axis=0
        ).rename(columns={
            'group': 'Group',
            'lu': 'Landuse',
            'am': 'Agri-Management',
            'Relative_Contribution_Percentage': 'Contribution Relative to Pre-1750 Level (%)'}
        ).reset_index(drop=True
        ).to_csv(os.path.join(path, f'biodiversity_GBF8_groups_scores_{yr_cal}.csv'), index=False)



def write_biodiversity_GBF8_scores_species(data: Data, yr_cal, path):
    # Caculate the biodiversity scores for species, if user selected any species
    if (not settings.BIODIVERSITY_TARGET_GBF_8 == 'on') or (len(data.BIO_GBF8_SEL_SPECIES) == 0):
        return
    
    print(f'Writing biodiversity GBF8 scores (SPECIES) for {yr_cal}')
    
    # Unpack the agricultural management land-use
    am_lu_unpack = [(am, l) for am, lus in data.AG_MAN_LU_DESC.items() for l in lus]

    # Get decision variables for the year
    ag_dvar_mrj = tools.ag_mrj_to_xr(data, data.ag_dvars[yr_cal])
    non_ag_dvar_rk = tools.non_ag_rk_to_xr(data, data.non_ag_dvars[yr_cal])
    am_dvar_jri = tools.am_mrj_to_xr(data, data.ag_man_dvars[yr_cal]).stack(idx=('am', 'lu'))
    am_dvar_jri = am_dvar_jri.sel(idx=am_dvar_jri['idx'].isin(pd.MultiIndex.from_tuples(am_lu_unpack)))

    # Get biodiversity scores for selected species
    bio_scores_sr = xr.DataArray(
        data.get_GBF8_bio_layers_by_yr(yr_cal, level='species') * data.REAL_AREA[None, :],
        dims=['species', 'cell'],
        coords={
            'species': data.BIO_GBF8_SEL_SPECIES,
            'cell': np.arange(data.NCELLS)}
    ).chunk({'cell': min(4096, data.NCELLS), 'species': 1})  # Chunking to save mem use

    # Get the habitat contribution for ag/non-ag/am land-use to biodiversity scores
    ag_impact_j = xr.DataArray(
        ag_biodiversity.get_ag_biodiversity_contribution(data),
        dims=['lu'],
        coords={'lu':data.AGRICULTURAL_LANDUSES}
    )
    non_ag_impact_k = xr.DataArray(
        list(non_ag_biodiversity.get_non_ag_lu_biodiv_contribution(data).values()),
        dims=['lu'],
        coords={'lu': data.NON_AGRICULTURAL_LANDUSES}
    )
    am_impact_ir = xr.DataArray(
        np.stack([arr for _, v in ag_biodiversity.get_ag_management_biodiversity_contribution(data, yr_cal).items() for arr in v.values()]),
        dims=['idx', 'cell'],
        coords={
            'idx': pd.MultiIndex.from_tuples(am_lu_unpack, names=['am', 'lu']),
            'cell': np.arange(data.NCELLS)}
    )

    # Get the base year biodiversity scores
    base_yr_score = pd.DataFrame({
            'species': data.BIO_GBF8_SEL_SPECIES,
            'BASE_OUTSIDE_SCORE': data.get_GBF8_score_outside_natural_LUTO_by_yr(yr_cal),
            'BASE_TOTAL_SCORE': data.BIO_GBF8_BASELINE_SCORE_AND_TARGET_PERCENT_SPECIES['HABITAT_SUITABILITY_BASELINE_SCORE_ALL_AUSTRALIA'],
            'TARGET_INSIDE_SCORE': data.get_GBF8_target_inside_LUTO_by_yr(yr_cal),}
        ).eval('Relative_Contribution_Percentage = BASE_OUTSIDE_SCORE / BASE_TOTAL_SCORE * 100')

    # Calculate GBF8 scores for species
    GBF8_scores_species_ag = (bio_scores_sr * ag_impact_j * ag_dvar_mrj
        ).sum(['cell', 'lm']
        ).to_dataframe('Area Weighted Score (ha)'
        ).reset_index(
        ).merge(base_yr_score
        ).eval('Relative_Contribution_Percentage = `Area Weighted Score (ha)` / BASE_TOTAL_SCORE * 100'
        ).assign(Type='Agricultural Landuse', Year=yr_cal)

    GBF8_scores_species_am = (am_dvar_jri * bio_scores_sr * am_impact_ir
        ).sum(['cell', 'lm']
        ).to_dataframe('Area Weighted Score (ha)'
        ).reset_index(allow_duplicates=True
        ).T.drop_duplicates(
        ).T.merge(base_yr_score
        ).astype({'Area Weighted Score (ha)': 'float', 'BASE_TOTAL_SCORE': 'float'}
        ).eval('Relative_Contribution_Percentage = `Area Weighted Score (ha)` / BASE_TOTAL_SCORE * 100'
        ).assign(Type='Agricultural Management', Year=yr_cal)

    GBF8_scores_species_non_ag = (non_ag_dvar_rk * bio_scores_sr * non_ag_impact_k
        ).sum(['cell']
        ).to_dataframe('Area Weighted Score (ha)'
        ).reset_index(
        ).merge(base_yr_score
        ).eval('Relative_Contribution_Percentage = `Area Weighted Score (ha)` / BASE_TOTAL_SCORE * 100'
        ).assign(Type='Non-Agricultural land-use', Year=yr_cal)

    # Concatenate the dataframes, rename the columns, and reset the index, then save to a csv file
    base_yr_score = base_yr_score.assign(Type='Outside LUTO study area', Year=yr_cal)

    pd.concat([
        GBF8_scores_species_ag,
        GBF8_scores_species_am,
        GBF8_scores_species_non_ag,
        base_yr_score], axis=0
        ).rename(columns={
            'species': 'Species',
            'lu': 'Landuse',
            'am': 'Agri-Management',
            'Relative_Contribution_Percentage': 'Contribution Relative to Pre-1750 Level (%)'}
        ).reset_index(drop=True
        ).to_csv(os.path.join(path, f'biodiversity_GBF8_species_scores_{yr_cal}.csv'), index=False)
        



def write_ghg(data: Data, yr_cal, path):
    """Calculate total GHG emissions from on-land agricultural sector.
        Takes a simulation object, a target calendar year (e.g., 2030),
        and an output path as input."""

    if settings.GHG_EMISSIONS_LIMITS == 'off':
        return

    print(f'Writing GHG outputs for {yr_cal}')

    yr_idx = yr_cal - data.YR_CAL_BASE

    # Get GHG emissions limits used as constraints in model
    ghg_limits = data.GHG_TARGETS[yr_cal]

    # Get GHG emissions from model
    if yr_cal >= data.YR_CAL_BASE + 1:
        ghg_emissions = data.prod_data[yr_cal]['GHG']
    else:
        ghg_emissions = (ag_ghg.get_ghg_matrices(data, yr_idx, aggregate=True) * data.ag_dvars[settings.SIM_YEARS[0]]).sum()

    # Save GHG emissions to file
    df = pd.DataFrame({
        'Variable':['GHG_EMISSIONS_LIMIT_TCO2e','GHG_EMISSIONS_TCO2e'],
        'Emissions (t CO2e)':[ghg_limits, ghg_emissions]
        })
    df['Year'] = yr_cal
    df.to_csv(os.path.join(path, f'GHG_emissions_{yr_cal}.csv'), index=False)
    




def write_ghg_separate(data: Data, yr_cal, path):

    if settings.GHG_EMISSIONS_LIMITS == 'off':
        return

    print(f'Writing GHG emissions_Separate for {yr_cal}')

    # Convert calendar year to year index.
    yr_idx = yr_cal - data.YR_CAL_BASE

    # Get the landuse descriptions for each validate cell (i.e., 0 -> Apples)
    lu_desc_map = {**data.AGLU2DESC,**data.NONAGLU2DESC}
    lu_desc = [lu_desc_map[x] for x in data.lumaps[yr_cal]]

    # -------------------------------------------------------#
    # Get greenhouse gas emissions from agricultural landuse #
    # -------------------------------------------------------#

    # Get the ghg_df
    ag_g_df = ag_ghg.get_ghg_matrices(data, yr_idx, aggregate=False)

    GHG_cols = []
    for col in ag_g_df.columns:
        # Get the index of each column
        s,m,j = [ag_g_df.columns.levels[i].get_loc(col[i]) for i in range(len(col))]
        # Get the GHG emissions
        ghg_col = np.nan_to_num(ag_g_df.loc[slice(None), col])
        # Get the dvar coresponding to the (m,j) dimension
        dvar = data.ag_dvars[yr_cal][m,:,j]
        # Multiply the GHG emissions by the dvar
        ghg_e = (ghg_col * dvar).sum()
        # Create a dataframe with the GHG emissions
        ghg_col = pd.DataFrame([ghg_e], index=pd.MultiIndex.from_tuples([col]))

        GHG_cols.append(ghg_col)

    # Concatenate the GHG emissions
    ghg_df = pd.concat(GHG_cols).reset_index()
    ghg_df.columns = ['Source','Water_supply','Landuse','GHG Emissions (t)']

    # Pivot the dataframe
    ghg_df = ghg_df.pivot(index='Landuse', columns=['Water_supply','Source'], values='GHG Emissions (t)')

    # Rename the columns
    ghg_df.columns = pd.MultiIndex.from_tuples([['Agricultural Landuse'] + list(col) for col in ghg_df.columns])
    column_rename = [(i[0],i[1],i[2].replace('CO2E_KG_HA','TCO2E')) for i in ghg_df.columns]
    column_rename = [(i[0],i[1],i[2].replace('CO2E_KG_HEAD','TCO2E')) for i in column_rename]
    ghg_df.columns = pd.MultiIndex.from_tuples(column_rename)
    ghg_df = ghg_df.fillna(0)

    # Reorganize the df to long format
    ghg_df = ghg_df.melt(ignore_index=False).reset_index()
    ghg_df.columns = ['Land-use','Type','Water_supply','CO2_type','Value (t CO2e)']
    ghg_df['Water_supply'] = ghg_df['Water_supply'].replace({'dry':'Dryland', 'irr':'Irrigated'})

    # Save table to disk
    ghg_df['Year'] = yr_cal
    ghg_df.to_csv(os.path.join(path, f'GHG_emissions_separate_agricultural_landuse_{yr_cal}.csv'), index=False)



    # -----------------------------------------------------------#
    # Get greenhouse gas emissions from non-agricultural landuse #
    # -----------------------------------------------------------#
    
    # Get ghg array
    ag_g_mrj = ag_ghg.get_ghg_matrices(data, yr_idx, aggregate=True)
    
    # Get the non_ag GHG reduction
    non_ag_g_rk = non_ag_ghg.get_ghg_matrix(data, ag_g_mrj, data.lumaps[yr_cal])

    # Multiply with decision variable to get the GHG in yr_cal
    non_ag_g_rk = non_ag_g_rk * data.non_ag_dvars[yr_cal]
    lmmap_mr = np.stack([data.lmmaps[yr_cal] ==0, data.lmmaps[yr_cal] ==1], axis=0)

    # get the non_ag GHG reduction on dry/irr land
    non_ag_g_mrk = np.einsum('rk, mr -> mrk', non_ag_g_rk, lmmap_mr)
    non_ag_g_mk = np.sum(non_ag_g_mrk, axis=1)

    # Convert arr to df
    df = pd.DataFrame(non_ag_g_mk.flatten(), index=pd.MultiIndex.from_product((data.LANDMANS, NON_AG_LAND_USES.keys()))).reset_index()
    df.columns = ['Water_supply', 'Land-use', 'Value (t CO2e)']
    df['Type'] = 'Non-Agricultural land-use'
    df = df.replace({'dry': 'Dryland', 'irr':'Irrigated'})

    # Save table to disk
    df['Year'] = yr_cal
    df.to_csv(os.path.join(path, f'GHG_emissions_separate_no_ag_reduction_{yr_cal}.csv'), index=False)


    # -------------------------------------------------------------------#
    # Get greenhouse gas emissions from landuse transformation penalties #
    # -------------------------------------------------------------------#

    # Retrieve list of simulation years (e.g., [2010, 2050] for snapshot or [2010, 2011, 2012] for timeseries)
    simulated_year_list = sorted(list(data.lumaps.keys()))

    # Get index of yr_cal in simulated_year_list (e.g., if yr_cal is 2050 then yr_idx_sim = 2 if snapshot)
    yr_idx_sim = simulated_year_list.index(yr_cal)

    # Get index of year previous to yr_cal in simulated_year_list (e.g., if yr_cal is 2050 then yr_cal_sim_pre = 2010 if snapshot)
    if yr_cal == data.YR_CAL_BASE:
        pass
    else:
        yr_cal_sim_pre = simulated_year_list[yr_idx_sim - 1]
        ghg_t_dict = ag_ghg.get_ghg_transition_emissions(data, data.lumaps[yr_cal_sim_pre], separate=True)
        transition_types = ghg_t_dict.keys()
        ghg_t = np.stack([ghg_t_dict[tt] for tt in transition_types], axis=0)


        # Get the GHG emissions from lucc-convertion compared to the previous year
        ghg_t_smj = np.einsum('mrj,smrj->smj', data.ag_dvars[yr_cal], ghg_t)

        # Summarize the array as a df
        ghg_t_df = pd.DataFrame(ghg_t_smj.flatten(), index=pd.MultiIndex.from_product((transition_types, data.LANDMANS, data.AGRICULTURAL_LANDUSES))).reset_index()
        ghg_t_df.columns = ['Type','Water_supply', 'Land-use', 'Value (t CO2e)']
        ghg_t_df = ghg_t_df.replace({'dry': 'Dryland', 'irr':'Irrigated'})
        ghg_t_df['Year'] = yr_cal
        
        # Save table to disk
        ghg_t_df.to_csv(os.path.join(path, f'GHG_emissions_separate_transition_penalty_{yr_cal}.csv'), index=False)



    # -------------------------------------------------------------------#
    # Get greenhouse gas emissions from agricultural management          #
    # -------------------------------------------------------------------#

    # Get the ag_man_g_mrj
    ag_man_g_mrj = ag_ghg.get_agricultural_management_ghg_matrices(data, yr_idx)

    am_dfs = []
    for am, am_lus in data.AG_MAN_LU_DESC.items():

        # Get the lucc_code for this the agricultural management in this loop
        am_j = np.array([data.DESC2AGLU[lu] for lu in am_lus])

        # Get the GHG emission from agricultural management, then reshape it to starte with row (r) dimension
        am_ghg_mrj = ag_man_g_mrj[am] * data.ag_man_dvars[yr_cal][am][:, :, am_j]
        am_ghg_mj = np.einsum('mrj -> mj', am_ghg_mrj)

        am_ghg_df = pd.DataFrame(am_ghg_mj.flatten(), index=pd.MultiIndex.from_product([data.LANDMANS, am_lus])).reset_index()
        am_ghg_df.columns = ['Water_supply', 'Land-use', 'Value (t CO2e)']
        am_ghg_df['Type'] = 'Agricultural Management'
        am_ghg_df['Agricultural Management Type'] = am
        am_ghg_df = am_ghg_df.replace({'dry': 'Dryland', 'irr':'Irrigated'})
        am_ghg_df['Year'] = yr_cal

        # Summarize the df by calculating the total value of each column
        am_dfs.append(am_ghg_df)

    # Save table to disk
    am_df = pd.concat(am_dfs, axis=0)
    am_df['Year'] = yr_cal
    am_df.to_csv(os.path.join(path, f'GHG_emissions_separate_agricultural_management_{yr_cal}.csv'), index=False)





def write_ghg_offland_commodity(data: Data, yr_cal, path):
    """Write out offland commodity GHG emissions"""

    if settings.GHG_EMISSIONS_LIMITS == 'off':
        return

    print(f'Writing offland commodity GHG emissions for {yr_cal}')

    # Get the offland commodity data
    offland_ghg = data.OFF_LAND_GHG_EMISSION.query(f'YEAR == {yr_cal}').rename(columns={'YEAR':'Year'})

    # Save to disk
    offland_ghg.to_csv(os.path.join(path, f'GHG_emissions_offland_commodity_{yr_cal}.csv'), index = False)

def write_npy(data: Data, yr_cal, path, yr_cal_sim_pre=None):
    write_rev_cost_npy(data, yr_cal, path, yr_cal_sim_pre)
    write_cost_transition_npy(data, yr_cal, path, yr_cal_sim_pre)
    write_GHG_npy(data, yr_cal, path)
    write_map_npy(data, yr_cal, path)
    write_GBF2_npy(data, yr_cal, path)
    write_bio_score_npy(data, yr_cal, path)
    # write_rev_non_ag_npy(data, yr_cal, path)


def save_map_to_npy(data, product, filename_prefix, yr_cal, path):
    """
    Creates a map array and saves it as a TIFF file. Creates the directory if it does not exist.
    """
    # Check if the path exists, create it if not
    path = os.path.join(path,"data_for_carbon_price")
    os.makedirs(path, exist_ok=True)

    filename = f"{filename_prefix}_{yr_cal}.npy"
    full_path = os.path.join(path, filename)
    full_path = full_path.replace('\\', '/')
    np.save(f"{full_path}", product)
    print(f'Map saved to {filename}')

    # # Create the map array
    # map_arr = create_2d_map(data, product, filler = data.MASK_LU_CODE)
    #
    # # Construct the filename and save the TIFF file
    # filename = f"{filename_prefix}_{yr_cal}.tiff"
    # full_path = os.path.join(path, filename)
    # write_gtiff(map_arr, full_path)
    # print(f"Map saved to {full_path}")


def write_rev_cost_npy(data: Data, yr_cal, path, yr_cal_sim_pre=None):
    yr_idx = yr_cal - data.YR_CAL_BASE

    ##############################################
    ## write agricultural revenue and cost maps to tiff ##
    ##############################################

    # Get the ag_dvar_mrj in the yr_cal
    ag_dvar_mrj = data.ag_dvars[yr_cal]

    # Get agricultural revenue/cost for year in mrjs format. Note the s stands for sources:
    # E.g., Sources for crops only contains ['Revenue'],
    #    but sources for livestock includes ['Meat', 'Wool', 'Live Exports', 'Milk']
    ag_rev_df_rjms = ag_revenue.get_rev_matrices(data, yr_idx, aggregate=False)
    ag_cost_df_rjms = ag_cost.get_cost_matrices(data, yr_idx, aggregate=False)
    ag_rev_df_rjms = ag_rev_df_rjms.reindex(columns=pd.MultiIndex.from_product(ag_rev_df_rjms.columns.levels),
                                            fill_value=0)
    ag_rev_rjms = ag_rev_df_rjms.values.reshape(-1, *ag_rev_df_rjms.columns.levshape)

    ag_cost_df_rjms = ag_cost_df_rjms.reindex(columns=pd.MultiIndex.from_product(ag_cost_df_rjms.columns.levels),
                                              fill_value=0)
    ag_cost_rjms = ag_cost_df_rjms.values.reshape(-1, *ag_cost_df_rjms.columns.levshape)

    # Multiply the ag_dvar_mrj with the ag_rev_mrj to get the ag_rev_jm
    ag_rev_r = np.einsum('mrj,rjms -> r', ag_dvar_mrj, ag_rev_rjms)
    ag_cost_r = np.einsum('mrj,rjms -> r', ag_dvar_mrj, ag_cost_rjms)

    # Use the function to save agricultural revenue and cost maps
    save_map_to_npy(data, ag_rev_r, 'revenue_ag', yr_cal, path)
    save_map_to_npy(data, ag_cost_r, 'cost_ag', yr_cal, path)

    ##############################################
    ## write agricultural management revenue and cost maps to tiff ##
    ##############################################

    # Get the revenue/cost matirces for each agricultural land-use
    ag_rev_mrj = ag_revenue.get_rev_matrices(data, yr_idx)
    ag_cost_mrj = ag_cost.get_cost_matrices(data, yr_idx)

    # Get the revenuecost matrices for each agricultural management
    am_revenue_mat = ag_revenue.get_agricultural_management_revenue_matrices(data, ag_rev_mrj, yr_idx)
    am_cost_mat = ag_cost.get_agricultural_management_cost_matrices(data, ag_cost_mrj, yr_idx)

    # Iterate through agricultural management and non-agricultural land uses
    for am, am_desc in settings.AG_MANAGEMENTS_TO_LAND_USES.items():
        if not settings.AG_MANAGEMENTS[am]:
            continue

        # Get the land use codes for the agricultural management
        am_code = [data.DESC2AGLU[desc] for desc in am_desc]

        # Get the revenue/cost matrix for the agricultural management
        am_rev = np.nan_to_num(am_revenue_mat[am])  # Replace NaNs with 0
        am_cost = np.nan_to_num(am_cost_mat[am])  # Replace NaNs with 0

        # Get the decision variable for each agricultural management
        am_dvar = data.ag_man_dvars[yr_cal][am][:, :, am_code]

        # Multiply the decision variable by revenue matrix
        am_rev_r = np.einsum('mrj,mrj->r', am_dvar, am_rev)
        am_cost_r = np.einsum('mrj,mrj->r', am_dvar, am_cost)

        save_map_to_npy(data, am_rev_r, f'revenue_am_{am}', yr_cal, path)
        save_map_to_npy(data, am_cost_r, f'cost_am_{am}', yr_cal, path)

    ##############################################
    ## write non-agricultural land use revenue and cost maps to tiff ##
    ##############################################

    # Get the non-agricultural decision variables
    non_ag_dvar = data.non_ag_dvars[yr_cal]  # rk

    # Get the non-agricultural revenue/cost matrices
    ag_r_mrj = ag_revenue.get_rev_matrices(data, yr_idx)
    non_ag_rev_mat = non_ag_revenue.get_rev_matrix(data, yr_cal, ag_r_mrj, data.lumaps[yr_cal])  # rk
    ag_c_mrj = ag_cost.get_cost_matrices(data, yr_idx)
    non_ag_cost_mat = non_ag_cost.get_cost_matrix(data, ag_c_mrj, data.lumaps[yr_cal], yr_cal)  # rk

    # Replace nan with 0
    non_ag_rev_mat = np.nan_to_num(non_ag_rev_mat)
    non_ag_cost_mat = np.nan_to_num(non_ag_cost_mat)

    # Assuming non_ag_rev_mat and non_ag_cost_mat have been correctly computed
    for index, non_ag in enumerate(data.NON_AGRICULTURAL_LANDUSES):
        rev_non_ag_r = np.einsum('r,r->r', non_ag_dvar[:, index], non_ag_rev_mat[:, index])
        cost_non_ag_r = np.einsum('r,r->r', non_ag_dvar[:, index], non_ag_cost_mat[:, index])

        save_map_to_npy(data, rev_non_ag_r, f'revenue_non-ag_{non_ag}', yr_cal, path)
        save_map_to_npy(data, cost_non_ag_r, f'cost_non-ag_{non_ag}', yr_cal, path)


def write_cost_transition_npy(data: Data, yr_cal, path, yr_cal_sim_pre=None):
    # Retrieve list of simulation years (e.g., [2010, 2050] for snapshot or [2010, 2011, 2012] for timeseries)
    simulated_year_list = sorted(list(data.lumaps.keys()))
    # Get index of yr_cal in timeseries (e.g., if yr_cal is 2050 then yr_idx = 40)
    yr_idx = yr_cal - data.YR_CAL_BASE

    # Get index of yr_cal in simulated_year_list (e.g., if yr_cal is 2050 then yr_idx_sim = 2 if snapshot)
    yr_idx_sim = simulated_year_list.index(yr_cal)
    # Get index of year previous to yr_cal in simulated_year_list (e.g., if yr_cal is 2050 then yr_cal_sim_pre = 2010 if snapshot)
    yr_cal_sim_pre = simulated_year_list[yr_idx_sim - 1] if yr_cal_sim_pre is None else yr_cal_sim_pre

    # Get the decision variables for agricultural land-use
    ag_dvar = data.ag_dvars[yr_cal]  # (m,r,j)
    # Get the non-agricultural decision variable
    non_ag_dvar = data.non_ag_dvars[yr_cal]  # (r,k)

    # ---------------------------------------------------------------------
    #              Agricultural land-use transition costs
    # ---------------------------------------------------------------------

    # Get the transition cost matrices for agricultural land-use
    # Get the base_year mrj matirx
    base_mrj = tools.lumap2ag_l_mrj(data.lumaps[yr_cal_sim_pre], data.lmmaps[yr_cal_sim_pre])

    # Get the transition cost matrices for agricultural land-use
    if yr_idx == 0:
        ag_transitions_cost_mat = {
            'Establishment cost': np.zeros((data.NLMS, data.NCELLS, data.N_AG_LUS)).astype(np.float32)}
    else:
        # Get the transition cost matrices for agricultural land-use
        ag_transitions_cost_mat = ag_transitions.get_transition_matrices_ag2ag_from_base_year(data, yr_idx, yr_cal_sim_pre, separate=True)

    cost_dfs = []
    # Convert the transition cost matrices to a DataFrame
    for from_lu_desc, from_lu_idx in data.DESC2AGLU.items():
        for from_lm_idx, from_lm in enumerate(data.LANDMANS):
            for cost_type in ag_transitions_cost_mat.keys():

                base_lu_arr = base_mrj[from_lm_idx, :, from_lu_idx]
                if base_lu_arr.sum() == 0: continue

                arr_dvar = ag_dvar[:, base_lu_arr,
                           :]  # Get the decision variable of the from land-use % from water-supply (mr*j)
                arr_trans = ag_transitions_cost_mat[cost_type][:, base_lu_arr,
                            :]  # Get the transition cost matrix of the from land-use % from water-supply (mr*j)
                cost_arr = np.einsum('mrj,mrj->r', arr_dvar,
                                     arr_trans).flatten()  # Calculate the cost array (r flatten)
                # Create a zero array with the same shape as the original base_lu_arr
                full_cost_arr = np.zeros(base_lu_arr.shape[0], dtype=cost_arr.dtype)

                # Fill the positions where base_lu_arr is True with the values from cost_arr
                full_cost_arr[base_lu_arr] = cost_arr

                cost_dfs.append(full_cost_arr)
    summed_array_r = np.sum(cost_dfs, axis=0)
    save_map_to_npy(data, summed_array_r, f'cost_transition_ag2ag', yr_cal, path)


    # ---------------------------------------------------------------------
    #              Agricultural management transition costs
    # ---------------------------------------------------------------------

    # The agricultural management transition cost are all zeros, so skip the calculation here
    # am_cost = ag_transitions.get_agricultural_management_transition_matrices(sim.data)

    # --------------------------------------------------------------------
    #              Non-agricultural land-use transition costs (from ag to non-ag)
    # --------------------------------------------------------------------

    # Get the transition cost matirces for non-agricultural land-use
    if yr_idx == 0:
        non_ag_transitions_cost_mat = {
            k: {'Transition cost': np.zeros(data.NCELLS).astype(np.float32)}
            for k in NON_AG_LAND_USES.keys()
        }
    else:
        non_ag_transitions_cost_mat = non_ag_transitions.get_transition_matrix_ag2nonag(
            data, yr_idx, data.lumaps[yr_cal_sim_pre], data.lmmaps[yr_cal_sim_pre], separate=True
        )

    # Get all land use decision variables
    desc2lu_all = {**data.DESC2AGLU, **data.DESC2NONAGLU}

    cost_dfs = []
    for from_lu in desc2lu_all.keys():
        for from_lm in data.LANDMANS:
            for to_lu in NON_AG_LAND_USES.keys():
                for cost_type in non_ag_transitions_cost_mat[to_lu].keys():

                    lu_idx = data.lumaps[yr_cal_sim_pre] == desc2lu_all[
                        from_lu]  # Get the land-use index of the from land-use (r)
                    lm_idx = data.lmmaps[yr_cal_sim_pre] == data.LANDMANS.index(
                        from_lm)  # Get the land-management index of the from land-management (r)
                    from_lu_idx = lu_idx & lm_idx  # Get the land-use index of the from land-use (r*)

                    arr_dvar = non_ag_dvar[from_lu_idx, data.NON_AGRICULTURAL_LANDUSES.index(
                        to_lu)]  # Get the decision variable of the from land-use (r*)
                    arr_trans = non_ag_transitions_cost_mat[to_lu][cost_type][
                        from_lu_idx]  # Get the transition cost matrix of the unchanged land-use (r)

                    if arr_dvar.size == 0:
                        continue

                    cost_arr = np.einsum('r,r->r', arr_trans, arr_dvar)
                    # Create a zero array with the same shape as from_lu_idx
                    full_cost_arr = np.zeros(from_lu_idx.shape, dtype=cost_arr.dtype)

                    # Fill the positions where from_lu_idx is True with the values from cost_arr
                    full_cost_arr[from_lu_idx] = cost_arr

                    cost_dfs.append(full_cost_arr)

    summed_array_r = np.sum(cost_dfs, axis=0)
    save_map_to_npy(data, summed_array_r, f'cost_transition_ag2non-ag', yr_cal, path)

    # --------------------------------------------------------------------
    #              Non-agricultural land-use transition costs (from non-ag to ag)
    # --------------------------------------------------------------------

    # Get the transition cost matirces for non-agricultural land-use
    if yr_idx == 0:
        non_ag_transitions_cost_mat = {
            k: {'Transition cost': np.zeros((data.NLMS, data.NCELLS, data.N_AG_LUS)).astype(np.float32)}
            for k in NON_AG_LAND_USES.keys()}
    else:
        non_ag_transitions_cost_mat = non_ag_transitions.get_transition_matrix_nonag2ag(data,
                                                                                        yr_idx,
                                                                                        data.lumaps[yr_cal_sim_pre],
                                                                                        data.lmmaps[yr_cal_sim_pre],
                                                                                        separate=True)

    cost_dfs = []
    for non_ag_type in non_ag_transitions_cost_mat:
        for cost_type in non_ag_transitions_cost_mat[non_ag_type]:
            arr = non_ag_transitions_cost_mat[non_ag_type][cost_type]  # Get the transition cost matrix
            arr = np.einsum('mrj,mrj->r', arr, ag_dvar)
            cost_dfs.append(arr)


    summed_array_r = np.sum(cost_dfs, axis=0)
    save_map_to_npy(data, summed_array_r, f'cost_transition_non-ag2ag', yr_cal, path)


def write_GHG_npy(data: Data, yr_cal, path):
    if settings.GHG_EMISSIONS_LIMITS == 'off':
        return
    print(f'Writing GHG outputs for {yr_cal}')

    yr_idx = yr_cal - data.YR_CAL_BASE

    # -------------------------------------------------------#
    # Get greenhouse gas emissions from agricultural landuse #
    # -------------------------------------------------------#
    ag_dvar_mrj = tools.ag_mrj_to_xr(data, data.ag_dvars[yr_cal])  # (m,r,j)
    ag_g_mrj = tools.ag_mrj_to_xr(data, ag_ghg.get_ghg_matrices(data, yr_idx, aggregate=True)).chunk({'cell': min(4096, data.NCELLS)})
    ag_g_r = (ag_dvar_mrj * ag_g_mrj).sum(['lm', 'lu']).values  # (r)
    save_map_to_npy(data, ag_g_r, 'GHG_ag', yr_cal, path)

    # -----------------------------------------------------------#
    # Get greenhouse gas emissions from non-agricultural landuse #
    # -----------------------------------------------------------#
    # Get the non_ag GHG reduction
    non_ag_dvar_rk = tools.non_ag_rk_to_xr(data, data.non_ag_dvars[yr_cal])  # (cell, lu)
    ag_g_mrj = ag_ghg.get_ghg_matrices(data, yr_idx, aggregate=True)
    non_ag_g_rk = tools.non_ag_rk_to_xr(data, non_ag_ghg.get_ghg_matrix(data, ag_g_mrj, data.lumaps[yr_cal])).chunk(
        {'cell': min(4096, data.NCELLS)})
    lmmap_mr = xr.DataArray(
        np.stack([data.lmmaps[yr_cal] == 0, data.lmmaps[yr_cal] == 1], axis=0),
        dims=('lm', 'cell'),
        coords={'lm': data.LANDMANS, 'cell': np.arange(data.NCELLS)}
    )

    non_ag_g_r = (non_ag_g_rk * non_ag_dvar_rk * lmmap_mr).sum(['lm', 'lu']).values  # (cell)
    save_map_to_npy(data, non_ag_g_r, 'GHG_non-ag', yr_cal, path)

    # -------------------------------------------------------------------#
    # Get greenhouse gas emissions from landuse transformation penalties #
    # -------------------------------------------------------------------#

    # Retrieve list of simulation years (e.g., [2010, 2050] for snapshot or [2010, 2011, 2012] for timeseries)
    simulated_year_list = sorted(list(data.lumaps.keys()))

    # Get index of yr_cal in simulated_year_list (e.g., if yr_cal is 2050 then yr_idx_sim = 2 if snapshot)
    yr_idx_sim = simulated_year_list.index(yr_cal)

    # Get index of year previous to yr_cal in simulated_year_list (e.g., if yr_cal is 2050 then yr_cal_sim_pre = 2010 if snapshot)
    if yr_cal == data.YR_CAL_BASE:
        ghg_t_mrj = tools.ag_mrj_to_xr(data, np.zeros(data.ag_dvars[yr_cal].shape, dtype=np.bool_))
    else:
        yr_cal_sim_pre = simulated_year_list[yr_idx_sim - 1]
        ghg_t_mrj = tools.ag_mrj_to_xr(data, ag_ghg.get_ghg_transition_emissions(data, data.lumaps[yr_cal_sim_pre]))

    # Get the GHG emissions from lucc-convertion compared to the previous year
    ghg_t_r = (ag_dvar_mrj * ghg_t_mrj).sum(['lm', 'lu']).values  # (r)
    save_map_to_npy(data, ghg_t_r, 'GHG_transition', yr_cal, path)

    # -------------------------------------------------------------------#
    # Get greenhouse gas emissions from agricultural management          #
    # -------------------------------------------------------------------#
    # Get the ag_man_g_mrj
    ag_man_g_mrj = ag_ghg.get_agricultural_management_ghg_matrices(data, yr_idx)

    am_dfs = []
    for am, am_lus in settings.AG_MANAGEMENTS_TO_LAND_USES.items():
        if not settings.AG_MANAGEMENTS[am]:
            continue
        # Get the lucc_code for this the agricultural management in this loop
        am_j = np.array([data.DESC2AGLU[lu] for lu in am_lus])

        # Get the GHG emission from agricultural management, then reshape it to starte with row (r) dimension
        am_ghg_mrj = ag_man_g_mrj[am] * data.ag_man_dvars[yr_cal][am][:, :, am_j]
        am_ghg_r = np.einsum('mrj -> r', am_ghg_mrj)
        save_map_to_npy(data, am_ghg_r, f'GHG_am_{am}', yr_cal, path)

    # -------------------------------------------------------------------#
    # Get greenhouse gas emissions from off_land          #
    # -------------------------------------------------------------------#


def write_map_npy(data: Data, yr_cal, path):
    # Get the Agricultural Management applied to each pixel
    ag_man_dvar = np.stack([np.einsum('mrj -> r', v) for _, v in data.ag_man_dvars[yr_cal].items()]).T  # (r, am)
    ag_man_dvar_mask = ag_man_dvar.sum(
        1) > 0.01  # Meaning that they have at least 1% of agricultural management applied
    ag_man_dvar = np.argmax(ag_man_dvar, axis=1) + 1  # Start from 1
    ag_man_dvar_argmax = np.where(ag_man_dvar_mask, ag_man_dvar, 0).astype(np.float32)

    # Get the non-agricultural landuse for each pixel
    non_ag_dvar = data.non_ag_dvars[yr_cal]  # (r, k)
    non_ag_dvar_mask = non_ag_dvar.sum(
        1) > 0.01  # Meaning that they have at least 1% of non-agricultural landuse applied
    non_ag_dvar = np.argmax(non_ag_dvar, axis=1) + settings.NON_AGRICULTURAL_LU_BASE_CODE  # Start from 100
    non_ag_dvar_argmax = np.where(non_ag_dvar_mask, non_ag_dvar, 0).astype(np.float32)

    save_map_to_npy(data, ag_man_dvar_argmax, f'am_map', yr_cal, path)
    save_map_to_npy(data, non_ag_dvar_argmax, f'non-ag_map', yr_cal, path)
    save_map_to_npy(data, data.lumaps[yr_cal], f'lu_map', yr_cal, path)
    save_map_to_npy(data, data.lmmaps[yr_cal], f'lm_map', yr_cal, path)

def write_rev_non_ag_npy(data: Data, yr_cal, path):
    print(f'Writing non-agricultural land-use revenue for {yr_cal}')
    yr_idx = yr_cal - data.YR_CAL_BASE
    # non_ag_dvar = tools.non_ag_rk_to_xr(data.non_ag_dvars[yr_cal])  # rk
    #
    # # Get the non-agricultural revenue/cost matrices
    # ag_r_mrj = tools.ag_mrj_to_xr(data, ag_revenue.get_rev_matrices(data, yr_idx))
    # ag_c_mrj = tools.ag_mrj_to_xr(data, ag_cost.get_cost_matrices(data, yr_idx))
    # non_ag_rev_mat = tools.non_ag_rk_to_xr(data, non_ag_revenue.get_rev_matrix(data, yr_cal, ag_r_mrj, data.lumaps[yr_cal]))  # rk
    # non_ag_cost_mat = tools.non_ag_rk_to_xr(data, non_ag_cost.get_cost_matrix(data, ag_c_mrj, data.lumaps[yr_cal], yr_cal))  # rk
    # non_ag_rev_mat = np.nan_to_num(non_ag_rev_mat)
    # non_ag_cost_mat = np.nan_to_num(non_ag_cost_mat)
    #
    # rev_non_ag_r = (non_ag_dvar * non_ag_rev_mat).sum(['lu'])  # (r)
    # cost_non_ag_r = (non_ag_dvar * non_ag_cost_mat).sum(['lu'])  # (r)

    agroforestry_x_r = tools.get_exclusions_agroforestry_base(data, data.lumaps[yr_cal])
    cp_belt_x_r = tools.get_exclusions_carbon_plantings_belt_base(data, data.lumaps[yr_cal])
    ag_r_mrj = ag_revenue.get_rev_matrices(data, yr_idx)

    # rev_sheep_agroforestry-------------------------------------------------------------------------------------------
    sheep_j = tools.get_sheep_code(data)

    # Only use the dryland version of sheep
    sheep_rev = ag_r_mrj[0, :, sheep_j]
    base_agroforestry_rev = non_ag_revenue.get_rev_agroforestry_base(data, yr_cal)

    # Calculate contributions and return the sum
    sheep_agroforestry_contr = base_agroforestry_rev * agroforestry_x_r
    sheep_contr = sheep_rev * (1 - agroforestry_x_r)

    index = 2

    rev_non_ag_non_ag_r = np.einsum('r,r->r', non_ag_dvar[:, index], sheep_agroforestry_contr)
    rev_non_ag_ag_r = np.einsum('r,r->r', non_ag_dvar[:, index], sheep_contr)

    save_map_to_npy(data, rev_non_ag_non_ag_r, f'revenue_non-ag_non-ag_{data.NON_AGRICULTURAL_LANDUSES[index]}', yr_cal, path)
    save_map_to_npy(data, rev_non_ag_ag_r, f'revenue_non-ag_ag_{data.NON_AGRICULTURAL_LANDUSES[index]}', yr_cal, path)


    # rev_beef_agroforestry-------------------------------------------------------------------------------------------
    beef_j = tools.get_beef_code(data)

    # Only use the dryland version of beef
    beef_rev = ag_r_mrj[0, :, beef_j]
    base_agroforestry_rev = non_ag_revenue.get_rev_agroforestry_base(data, yr_cal)

    # Calculate contributions and return the sum
    beef_agroforestry_contr = base_agroforestry_rev * agroforestry_x_r
    beef_contr = beef_rev * (1 - agroforestry_x_r)

    index = 3
    non_ag_dvar = data.non_ag_dvars[yr_cal]  # rk

    rev_non_ag_non_ag_r = np.einsum('r,r->r', non_ag_dvar[:, index], beef_agroforestry_contr)
    rev_non_ag_ag_r = np.einsum('r,r->r', non_ag_dvar[:, index], beef_contr)

    save_map_to_npy(data, rev_non_ag_non_ag_r, f'revenue_non-ag_non-ag_{data.NON_AGRICULTURAL_LANDUSES[index]}', yr_cal,
                    path)
    save_map_to_npy(data, rev_non_ag_ag_r, f'revenue_non-ag_ag_{data.NON_AGRICULTURAL_LANDUSES[index]}', yr_cal, path)

    # rev_sheep_carbon_plantings_belt-------------------------------------------------------------------------------------------
    sheep_j = tools.get_sheep_code(data)

    # Only use the dryland version of sheep
    sheep_rev = ag_r_mrj[0, :, sheep_j]
    base_cp_rev = non_ag_revenue.get_rev_carbon_plantings_belt_base(data, yr_cal)

    # Calculate contributions and return the sum
    sheep_cp_contr = base_cp_rev * cp_belt_x_r
    sheep_contr = sheep_rev * (1 - cp_belt_x_r)

    index = 5
    non_ag_dvar = data.non_ag_dvars[yr_cal]  # rk

    rev_non_ag_non_ag_r = np.einsum('r,r->r', non_ag_dvar[:, index], sheep_cp_contr)
    rev_non_ag_ag_r = np.einsum('r,r->r', non_ag_dvar[:, index], sheep_contr)

    save_map_to_npy(data, rev_non_ag_non_ag_r, f'revenue_non-ag_non-ag_{data.NON_AGRICULTURAL_LANDUSES[index]}', yr_cal,
                    path)
    save_map_to_npy(data, rev_non_ag_ag_r, f'revenue_non-ag_ag_{data.NON_AGRICULTURAL_LANDUSES[index]}', yr_cal, path)

    # rev_beef_carbon_plantings_belt-------------------------------------------------------------------------------------------
    beef_j = tools.get_beef_code(data)

    # Only use the dryland version of beef
    beef_rev = ag_r_mrj[0, :, beef_j]
    base_cp_rev = non_ag_revenue.get_rev_carbon_plantings_belt_base(data, yr_cal)

    # Calculate contributions and return the sum
    beef_cp_contr = base_cp_rev * cp_belt_x_r
    beef_contr = beef_rev * (1 - cp_belt_x_r)

    index = 6
    non_ag_dvar = data.non_ag_dvars[yr_cal]  # rk

    rev_non_ag_non_ag_r = np.einsum('r,r->r', non_ag_dvar[:, index], beef_cp_contr)
    rev_non_ag_ag_r = np.einsum('r,r->r', non_ag_dvar[:, index], beef_contr)

    save_map_to_npy(data, rev_non_ag_non_ag_r, f'revenue_non-ag_non-ag_{data.NON_AGRICULTURAL_LANDUSES[index]}', yr_cal,
                    path)
    save_map_to_npy(data, rev_non_ag_ag_r, f'revenue_non-ag_ag_{data.NON_AGRICULTURAL_LANDUSES[index]}', yr_cal, path)

def write_GBF2_npy(data: Data, yr_cal, path):
    print(f'Writing GBF2 biodiversity outputs for {yr_cal}')
    if settings.BIODIVERSITY_TARGET_GBF_2 == 'off':
        return
    # Unpack the ag managements and land uses
    am_lu_unpack = [
        (am, l)
        for am, lus in settings.AG_MANAGEMENTS_TO_LAND_USES.items()
        if settings.AG_MANAGEMENTS[am]
        for l in lus
    ]

    # Get decision variables for the year
    ag_dvar_mrj = tools.ag_mrj_to_xr(data, data.ag_dvars[yr_cal])
    non_ag_dvar_rk = tools.non_ag_rk_to_xr(data, data.non_ag_dvars[yr_cal])
    am_dvar_jri = tools.am_mrj_to_xr(data, data.ag_man_dvars[yr_cal]).stack(idx=('am', 'lu'))
    am_dvar_jri = am_dvar_jri.sel(idx=am_dvar_jri['idx'].isin(pd.MultiIndex.from_tuples(am_lu_unpack)))

    # Get the priority degrade areas scores
    priority_degraded_area_score_r = xr.DataArray(
        data.BIO_PRIORITY_DEGRADED_AREAS_R,
        dims=['cell'],
        coords={'cell': range(data.NCELLS)}
    ).chunk({'cell': min(4096, data.NCELLS)})  # Chunking to save mem use

    # Get the impacts of each ag/non-ag/am to vegetation matrices
    ag_impact_j = xr.DataArray(
        ag_biodiversity.get_ag_biodiversity_contribution(data),
        dims=['lu'],
        coords={'lu': data.AGRICULTURAL_LANDUSES}
    )
    non_ag_impact_k = xr.DataArray(
        list(non_ag_biodiversity.get_non_ag_lu_biodiv_contribution(data).values()),
        dims=['lu'],
        coords={'lu': data.NON_AGRICULTURAL_LANDUSES}
    )
    am_impact_ir = xr.DataArray(
        np.stack(
            [arr for _, v in ag_biodiversity.get_ag_management_biodiversity_contribution(data, yr_cal).items() for arr
             in v.values()]),
        dims=['idx', 'cell'],
        coords={
            'idx': pd.MultiIndex.from_tuples(am_lu_unpack, names=['am', 'lu']),
            'cell': range(data.NCELLS)}
    )

    bio_ag_r = (priority_degraded_area_score_r * ag_impact_j * ag_dvar_mrj
                     ).sum(['lm', 'lu'])
    bio_am_r = (priority_degraded_area_score_r * am_impact_ir * am_dvar_jri
        ).sum(['lm', 'idx'], skipna=False
        )
    bio_non_ag_r = (priority_degraded_area_score_r * non_ag_impact_k * non_ag_dvar_rk
                         ).sum(['lu'])

    save_map_to_npy(data, bio_ag_r, 'BIO_ag', yr_cal, path)
    save_map_to_npy(data, bio_am_r, 'BIO_am', yr_cal, path)
    save_map_to_npy(data, bio_non_ag_r, 'BIO_non-ag', yr_cal, path)

def write_bio_score_npy(data: Data, yr_cal, path):
    yr_cal_previouse = sorted(data.lumaps.keys())[sorted(data.lumaps.keys()).index(yr_cal) - 1]
    yr_idx = yr_cal - data.YR_CAL_BASE

    # Get the decision variables for the year
    ag_dvar_mrj = tools.ag_mrj_to_xr(data, data.ag_dvars[yr_cal])
    ag_mam_dvar_mrj = tools.am_mrj_to_xr(data, data.ag_man_dvars[yr_cal])
    non_ag_dvar_rk = tools.non_ag_rk_to_xr(data, data.non_ag_dvars[yr_cal])

    # Get the biodiversity scores b_mrj
    bio_ag_priority_mrj = tools.ag_mrj_to_xr(data, ag_biodiversity.get_bio_overall_priority_score_matrices_mrj(data))
    bio_am_priority_tmrj = tools.am_mrj_to_xr(data,
                                              ag_biodiversity.get_agricultural_management_biodiversity_matrices(data,
                                                                                                                bio_ag_priority_mrj.values,
                                                                                                                yr_idx))
    bio_non_ag_priority_rk = tools.non_ag_rk_to_xr(data,
                                                   non_ag_biodiversity.get_breq_matrix(data, bio_ag_priority_mrj.values,
                                                                                       data.lumaps[yr_cal_previouse]))

    # Calculate the biodiversity scores
    base_yr_score = np.einsum('j,mrj->', ag_biodiversity.get_ag_biodiversity_contribution(data), data.AG_L_MRJ)

    priority_ag_r = (ag_dvar_mrj * bio_ag_priority_mrj).sum(['lm','lu'], skipna=False)

    priority_am_r = (ag_mam_dvar_mrj * bio_am_priority_tmrj
                   ).sum(['am','lm','lu'], skipna=False)

    priority_non_ag_r = (non_ag_dvar_rk * bio_non_ag_priority_rk).sum(['lu'], skipna=False)

    save_map_to_npy(data, priority_ag_r, 'PBIO_ag', yr_cal, path)
    save_map_to_npy(data, priority_am_r, 'PBIO_am', yr_cal, path)
    save_map_to_npy(data, priority_non_ag_r, 'PBIO_non-ag', yr_cal, path)


