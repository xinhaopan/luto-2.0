import os
import numpy as np
import pandas as pd
from tools.helpers import create_grid_search_template

grid_search = {
    # ---------Computational settings, which are not relevant to LUTO itself---------
    'MEM': ['4'],
    'NCPUS': ['1'],
    'TIME': ['1:00:00'],

    # ---------------------------------- Model settings ------------------------------
    'SOLVE_WEIGHT_ALPHA': [0.1],
    'SOLVE_WEIGHT_BETA': [0.9],
    'OBJECTIVE': ['maxprofit'], # maxprofit
    'WRITE_OUTPUT_GEOTIFFS': [True],
    'RESFACTOR': [20],
    'SIM_YERAS': [[i for i in range(2010,2051,1)]],

    # ----------------------------------- GHG settings --------------------------------
    'GHG_EMISSIONS_LIMITS': ['high'], # ['high', 'medium', 'low']
    'GHG_CONSTRAINT_TYPE': ['hard'],
    'CARBON_PRICES_FIELD': ['CONSTANT'],

    # ----------------------------- Biodiversity settings -------------------------------
    'BIODIVERSTIY_TARGET_GBF_2': ['low', 'medium', 'high'],
    'GBF2_CONSTRAINT_TYPE': ['hard'],

    'BIODIVERSTIY_TARGET_GBF_3': ['off'],
    'BIODIVERSTIY_TARGET_GBF_4_SNES': ['off'],
    'BIODIVERSTIY_TARGET_GBF_4_ECNES': ['off'],
    'BIODIVERSTIY_TARGET_GBF_8': ['off'],
    'GBF2_PRIORITY_DEGRADED_AREAS_PERCENTAGE_CUT': [40],

    # ----------------------------------- Water settings --------------------------------
    'WATER_LIMITS': ['on'],
    'WATER_CONSTRAINT_TYPE': ['hard'],
    'INCLUDE_WATER_LICENSE_COSTS': [1],

    # ----------------------------------- Demand settings --------------------------------
    'DEMAND_CONSTRAINT_TYPE': ['soft'],
}
settings_name_dice = {
    'GHG_EMISSIONS_LIMITS':'GHG',
    'BIODIVERSTIY_TARGET_GBF_2':'BIO',
    'RESFACTOR':'RES',
}

output_file = os.path.join("Custom_runs", "setting_0529.csv")
create_grid_search_template(grid_search,output_file,settings_name_dice)

print(f"saved to {output_file}")

