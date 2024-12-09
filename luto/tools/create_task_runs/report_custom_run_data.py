import os
import json
import numpy as np
import pandas as pd
import plotnine as p9

from luto.tools.create_task_runs.parameters import TASK_ROOT_DIR
from luto.tools.report.data_tools import get_all_files


TASK_ROOT_DIR = "C:/Users/Jinzhu/Desktop/Snapshoot_multiple_scenarios"
grid_search_params = pd.read_csv(f"{TASK_ROOT_DIR}/grid_search_parameters.csv")

grid_paras = set(grid_search_params.columns.tolist()) - set(['run_idx', 'MEM', 'NCPUS', 'MODE'])



# Get the last run directory
report_data = pd.DataFrame()

for _, row in grid_search_params.iterrows():
    
    # Get the last run directory
    json_path = f"{TASK_ROOT_DIR}/Run_{row['run_idx']}/DATA_REPORT/data"
    
    # Get the profit data
    with open(f"{json_path}/economics_0_rev_cost_all_wide.json") as f:
        data = json.load(f)
        
    df = pd.json_normalize(data, 'data', ['name', 'type'])\
           .rename(columns={0: 'year', 1: 'val'})
    df_profit = df.query('name == "Profit"')
    

    # Get the GHG deviation 
    with open(f"{json_path}/GHG_2_individual_emission_Mt.json") as f:
        data = json.load(f)
        
    df = pd.json_normalize(data, 'data', ['name', 'type'])\
           .rename(columns={0: 'year', 1: 'val'})
    df_target = df.query('name == "GHG emissions limit"')
    df_actual = df.query('name == "Net emissions"')
    df_deviation = df_target.merge(df_actual, on='year', suffixes=('_target', '_actual'))
    df_deviation['name'] = 'GHG deviation'
    df_deviation['val'] = df_deviation['val_actual'] - df_deviation['val_target']
    
    # Combine the data
    report_data = pd.concat([
        report_data,
        df_profit[['year','name', 'val']].assign(**row),
        df_deviation[['year','name', 'val']].assign(**row)
    ]).reset_index(drop=True)


report_data.query('BIODIV_GBF_TARGET_2_DICT == "{2010: 0, 2030: 0.3, 2050: 0.3, 2100: 0.3}" and year == 2050 and name == "Profit"')



# Pivot the data
report_data_wide = report_data.pivot(index=['year', 'run_idx', 'SOLVE_ECONOMY_WEIGHT'], columns='name', values='val').reset_index()      
report_data_wide = report_data_wide.query('year != "2010"')


# Plot the data
p9.options.figure_size = (15, 8)
p9.options.dpi = 150

p = (p9.ggplot(report_data_wide, 
               p9.aes(x='GHG deviation', 
                      y='Profit', 
                      color='SOLVE_ECONOMY_WEIGHT')
               ) +
     p9.facet_grid('BIODIV_GBF_TARGET_2_DICT ~ GHG_LIMITS_FIELD', scales='free') +
     p9.geom_point() +
     p9.theme_bw() 
    )





