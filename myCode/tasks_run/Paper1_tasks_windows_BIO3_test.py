import os
import numpy as np
import pandas as pd
from tools.helpers import create_grid_search_template,create_task_runs

grid_search = {
    'TASK_NAME': ['20250608_Paper1_results_windows_BIO3'],
    'KEEP_OUTPUTS': [True],  # If False, only keep report HTML
    'QUEUE': ['normalsr'],
    'NUMERIC_FOCUS': [2],
    # ---------Computational settings, which are not relevant to LUTO itself---------
    'MEM': ['48GB'],
    'NCPUS': ['12'],
    'TIME': ['15:00:00'],

    # ---------------------------------- Model settings ------------------------------
    'SOLVE_WEIGHT_ALPHA': [1],
    'SOLVE_WEIGHT_BETA': [0.99],
    'OBJECTIVE': ['maxprofit'], # maxprofit
    'WRITE_OUTPUT_GEOTIFFS': [True],
    'RESFACTOR': [5],
    'SIM_YEARS': [[i for i in range(2010,2051,1)]],

    # ----------------------------------- GHG settings --------------------------------
    'GHG_EMISSIONS_LIMITS': ['low','medium','high'],
    'GHG_CONSTRAINT_TYPE': ['hard'],
    'CARBON_PRICES_FIELD': ['CONSTANT'],

    # ----------------------------- Biodiversity settings -------------------------------
    'BIODIVERSTIY_TARGET_GBF_2': ['low','medium2','high3'],
    'GBF2_CONSTRAINT_TYPE': ['hard'],

    'BIODIVERSTIY_TARGET_GBF_3': ['off'],
    'BIODIVERSTIY_TARGET_GBF_4_SNES': ['off'],
    'BIODIVERSTIY_TARGET_GBF_4_ECNES': ['off'],
    'BIODIVERSTIY_TARGET_GBF_8': ['off'],
    'GBF2_PRIORITY_DEGRADED_AREAS_PERCENTAGE_CUT': [100],

    # ----------------------------------- Water settings --------------------------------
    'WATER_LIMITS': ['on'],
    'WATER_CONSTRAINT_TYPE': ['hard'],
    'INCLUDE_WATER_LICENSE_COSTS': [1],

    # ----------------------------------- Demand settings --------------------------------
    'DEMAND_CONSTRAINT_TYPE': ['soft'],

    # # ----------------------------------- Paper 1 settings --------------------------------
    # 'NON_AG_LAND_USES' : [{
    #     'Environmental Plantings': True,
    #     'Riparian Plantings': True,
    #     'Sheep Agroforestry': True,
    #     'Beef Agroforestry': True,
    #     'Carbon Plantings (Block)': True,
    #     'Sheep Carbon Plantings (Belt)': True,
    #     'Beef Carbon Plantings (Belt)': True,
    #     'BECCS': False,
    #     'Destocked - natural land': False,
    # }],
    #
    # 'AG_MANAGEMENTS' : [{
    #     'Asparagopsis taxiformis': True,
    #     'Precision Agriculture': True,
    #     'Ecological Grazing': False,
    #     'Savanna Burning': True,
    #     'AgTech EI': True,
    #     'Biochar': True,
    #     'HIR - Beef': False,
    #     'HIR - Sheep': False,
    # }]
}
settings_name_dict = {
    'GHG_EMISSIONS_LIMITS':'GHG',
    'BIODIVERSTIY_TARGET_GBF_2':'BIO',
}

task_root_dir = f'../../output/{grid_search['TASK_NAME'][0]}'
grid_search_settings_df = create_grid_search_template(grid_search,settings_name_dict)
create_task_runs(task_root_dir, grid_search_settings_df, mode='single', n_workers=min(len(grid_search_settings_df.columns), 50),use_parallel=True)


