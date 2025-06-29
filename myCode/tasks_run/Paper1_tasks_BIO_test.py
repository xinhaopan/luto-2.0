import os
import numpy as np
import pandas as pd
from tools.helpers import create_grid_search_template,create_task_runs

grid_search = {
    'TASK_NAME': ['20250625_Paper1_BIO_test_RES3_years'],
    'KEEP_OUTPUTS': [True],  # If False, only keep report HTML
    'QUEUE': ['normalsr'],
    # ---------Computational settings, which are not relevant to LUTO itself---------
    'MEM': ['90GB'],
    'NCPUS': ['6'],
    'WRITE_THREADS': ['3'],
    'TIME': ['50:00:00'],

    'GBF2_PRIORITY_DEGRADED_AREAS_PERCENTAGE_CUT': [40,50,100],
    # ---------------------------------- Model settings ------------------------------
    'SOLVE_WEIGHT_ALPHA': [1],
    'SOLVE_WEIGHT_BETA': [0.9],
    'OBJECTIVE': ['maxprofit'], # maxprofit
    'WRITE_OUTPUT_GEOTIFFS': [False],
    'RESFACTOR': [3],
    'SIM_YEARS': [[i for i in range(2010,2051,1)]],

    # ----------------------------------- GHG settings --------------------------------
    'GHG_EMISSIONS_LIMITS': ['low'],
    'GHG_CONSTRAINT_TYPE': ['hard'],
    'CARBON_PRICES_FIELD': ['CONSTANT'],

    # ----------------------------- Biodiversity settings -------------------------------
    'BIODIVERSITY_TARGET_GBF_2': ['l1','l2','l3','l4'],
    'GBF2_TARGETS_DICT': [{
        'l1': {2030: 0.15,    2050: 0.25,    2100: 0.25},
        'l2': {2030: 0.15,    2050: 0.30,    2100: 0.30},
        'l3': {2030: 0.30,    2050: 0.30,    2100: 0.30},
        'l4': {2030: 0.30,    2050: 0.50,    2100: 0.50},
    }],
    'GBF2_CONSTRAINT_TYPE': ['hard'],

    'BIODIVERSTIY_TARGET_GBF_3': ['off'],
    'BIODIVERSTIY_TARGET_GBF_4_SNES': ['off'],
    'BIODIVERSTIY_TARGET_GBF_4_ECNES': ['off'],
    'BIODIVERSTIY_TARGET_GBF_8': ['off'],

    'HABITAT_CONDITION': ['USER_DEFINED'],

    # ----------------------------------- Water settings --------------------------------
    'WATER_LIMITS': ['on'],
    'WATER_CONSTRAINT_TYPE': ['hard'],
    'INCLUDE_WATER_LICENSE_COSTS': [1],

    # ----------------------------------- Demand settings --------------------------------
    'DEMAND_CONSTRAINT_TYPE': ['soft'],
}
settings_name_dict = {
    'GHG_EMISSIONS_LIMITS':'GHG',
    'BIODIVERSITY_TARGET_GBF_2':'BIO',
    'GBF2_PRIORITY_DEGRADED_AREAS_PERCENTAGE_CUT': 'PRI',
}

task_root_dir = f'../../output/{grid_search['TASK_NAME'][0]}'
grid_search_settings_df = create_grid_search_template(grid_search,settings_name_dict)
create_task_runs(task_root_dir, grid_search_settings_df, platform="HPC", n_workers=min(len(grid_search_settings_df.columns), 50),use_parallel=True)


