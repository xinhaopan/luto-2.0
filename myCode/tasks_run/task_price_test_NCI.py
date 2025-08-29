import os
import pandas as pd
from tools.helpers import create_grid_search_template,create_task_runs
import re

grid_search = {
    'TASK_NAME': ['20250830_Price_Task_RES13_NCI'],
    'KEEP_OUTPUTS': [True],  # If False, only keep report HTML
    'QUEUE': ['normalsr'],
    'NUMERIC_FOCUS': [0],
    # ---------Computational settings, which are not relevant to LUTO itself---------
    'MEM': ['12GB'],
    'NCPUS': ['3'],
    'WRITE_THREADS': ['2'],
    'TIME': ['6:00:00'],

    'GHG_EMISSIONS_LIMITS': ['high','off'],
    'BIODIVERSITY_TARGET_GBF_2': ['high','off'],

    # ---------------------------------- Model settings ------------------------------
    'SOLVE_WEIGHT_ALPHA': [1],
    'SOLVE_WEIGHT_BETA': [0.9],
    'OBJECTIVE': ['maxprofit'], # maxprofit
    'WRITE_OUTPUT_GEOTIFFS': [True],
    'RESFACTOR': [13],
    'SIM_YEARS': [[i for i in range(2010,2051,1)]],

    # ----------------------------------- GHG settings --------------------------------

    'GHG_CONSTRAINT_TYPE': ['hard'],
    'CARBON_PRICES_FIELD': ['CONSTANT'],
    'GHG_PERCENT': [20,40,60,80,100],

    # ----------------------------- Biodiversity settings -------------------------------

    'GBF2_CONSTRAINT_TYPE': ['hard'],
    'GBF2_TARGETS_DICT': [{
        'low': {2030: 0, 2050: 0, 2100: 0},
        'medium': {2030: 0.30, 2050: 0.30, 2100: 0.30},
        'high': {2030: 0.30, 2050: 0.50, 2100: 0.50},
    }],
    'BIODIVERSTIY_TARGET_GBF_3': ['off'],
    'BIODIVERSTIY_TARGET_GBF_4_SNES': ['off'],
    'BIODIVERSTIY_TARGET_GBF_4_ECNES': ['off'],
    'BIODIVERSTIY_TARGET_GBF_8': ['off'],
    'GBF2_PRIORITY_DEGRADED_AREAS_PERCENTAGE_CUT': [10,20,30,40,50],

    # ----------------------------------- Water settings --------------------------------
    'WATER_STRESS': [0.6],
    'WATER_LIMITS': ['on'],
    'WATER_CONSTRAINT_TYPE': ['hard'],
    'INCLUDE_WATER_LICENSE_COSTS': [1],
    'WATER_REGION_DEF': ['Drainage Division'],

    # ----------------------------------- Demand settings --------------------------------
    'DEMAND_CONSTRAINT_TYPE': ['soft'],
}
settings_name_dict = {
    'GHG_EMISSIONS_LIMITS': 'GHG',
    'GHG_PERCENT':'PERCENT',
    'BIODIVERSITY_TARGET_GBF_2': 'BIO',
    'GBF2_PRIORITY_DEGRADED_AREAS_PERCENTAGE_CUT':'CUT',
}

task_root_dir = f'../../output/{grid_search['TASK_NAME'][0]}'
grid_search_settings_df = create_grid_search_template(grid_search,settings_name_dict)

def drop_if_percent_not_100(col: str) -> bool:
    # 匹配 GHG_off_PERCENT_ 和 _BIO_ 之间的内容
    m = re.search(r"GHG_off_PERCENT_([^_]+)_BIO_", col)
    if not m:
        return False
    return int(m.group(1)) != 100  # 只保留等于 100

def drop_if_cut_not_50(col: str) -> bool:
    # 匹配 BIO_off_CUT_ 后紧跟的数字（遇到非数字停止）
    m = re.search(r"BIO_off_CUT_([0-9]+)", col)
    if not m:
        return False
    return int(m.group(1)) != 50  # 只保留等于 50

cols_to_drop = [c for c in grid_search_settings_df.columns if drop_if_percent_not_100(c) or drop_if_cut_not_50(c)]

# 删除列
grid_search_settings_df = grid_search_settings_df.drop(columns=cols_to_drop)
grid_search_settings_df.to_csv(os.path.join(task_root_dir, 'grid_search_template.csv'))
# grid_search_settings_df = pd.read_csv(os.path.join(task_root_dir, 'grid_search_template.csv'), index_col=0)
create_task_runs(task_root_dir, grid_search_settings_df, platform="NCI", n_workers=min(len(grid_search_settings_df.columns), 50),use_parallel=True)