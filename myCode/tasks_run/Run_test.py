import os
import numpy as np
import pandas as pd
from tools.helpers import create_grid_search_template,create_task_runs

grid_search = {
    'TASK_NAME': ['Run_task_test'],
    'KEEP_OUTPUTS': [False],  # If False, only keep report HTML
    'QUEUE': ['normalsr'],
    'NUMERIC_FOCUS': [0],
    # ---------Computational settings, which are not relevant to LUTO itself---------
    'MEM': ['40GB'],
    'NCPUS': ['10'],
    'WRITE_THREADS': ['2'],
    'TIME': ['2:00:00'],

    'GBF2_PRIORITY_DEGRADED_AREAS_PERCENTAGE_CUT': [50],
    # ---------------------------------- Model settings ------------------------------
    'SOLVE_WEIGHT_ALPHA': [1],
    'SOLVE_WEIGHT_BETA': [0.9],
    'OBJECTIVE': ['maxprofit'], # maxprofit
    'WRITE_OUTPUT_GEOTIFFS': [True],
    'RESFACTOR': [13],
    'SIM_YEARS': [[i for i in range(2010,2051,10)]],

    # ----------------------------------- GHG settings --------------------------------
    'GHG_EMISSIONS_LIMITS': ['high','off'],
    'GHG_CONSTRAINT_TYPE': ['hard'],
    'CARBON_PRICES_FIELD': ['CONSTANT'],

    # ----------------------------- Biodiversity settings -------------------------------
    'BIODIVERSITY_TARGET_GBF_2': ['high','off'],
    'GBF2_CONSTRAINT_TYPE': ['hard'],
    'GBF2_TARGETS_DICT': [{
        'low': {2030: 0, 2050: 0, 2100: 0},
        'medium': {2030: 0.30, 2050: 0.30, 2100: 0.30},
        'high': {2030: 0.30, 2050: 0.50, 2100: 0.50},
    }],
    'BIODIVERSITY_TARGET_GBF_3': ['off'],
    'BIODIVERSITY_TARGET_GBF_4_SNES': ['off'],
    'BIODIVERSITY_TARGET_GBF_4_ECNES': ['off'],
    'BIODIVERSITY_TARGET_GBF_8': ['off'],

    # ----------------------------------- Water settings --------------------------------
    'WATER_LIMITS': ['on'],
    'WATER_CONSTRAINT_TYPE': ['hard'],
    'INCLUDE_WATER_LICENSE_COSTS': [1],
    'WATER_REGION_DEF': ['Drainage Division'],

    # ----------------------------------- Demand settings --------------------------------
    'DEMAND_CONSTRAINT_TYPE': ['soft'],
}
settings_name_dict = {
    'GHG_EMISSIONS_LIMITS':'GHG',
    'BIODIVERSITY_TARGET_GBF_2':'BIO',
}

task_root_dir = f'../../output/{grid_search['TASK_NAME'][0]}'
grid_search_settings_df = create_grid_search_template(grid_search,settings_name_dict)
# grid_search_settings_df = pd.read_csv(os.path.join(task_root_dir, 'grid_search_template.csv'), index_col=0)
create_task_runs(task_root_dir, grid_search_settings_df, platform="Denethor", n_workers=min(len(grid_search_settings_df.columns), 50),use_parallel=True)


