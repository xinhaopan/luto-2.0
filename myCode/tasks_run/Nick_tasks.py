import os
import numpy as np
import pandas as pd
from tools.helpers import create_grid_search_template,create_task_runs

grid_search = {
    'TASK_NAME': ['20251002_Cost_curve_task'],
    'KEEP_OUTPUTS': [False],  # If False, only keep report HTML
    'QUEUE': ['normalsr'],
    'NUMERIC_FOCUS': [2],
    # ---------Computational settings, which are not relevant to LUTO itself---------
    'MEM': ['28GB'],
    'NCPUS': ['4'],
    'WRITE_THREADS': ['2'],
    'TIME': ['6:00:00'],

    'GHG_EMISSIONS_LIMITS': ['off'],
    'BIODIVERSITY_TARGET_GBF_2': ['high', 'off'],
    'GBF2_PRIORITY_DEGRADED_AREAS_PERCENTAGE_CUT': [50,40,30,20,10],
    'CARBON_PRICE_COSTANT': [0,8.92,17.85,26.77,35.69,44.61,53.54,62.46,75.84,89.23,133.84,178.46,233.07,267.69,312.3,356.92],
    # ---------------------------------- Model settings ------------------------------
    'SOLVE_WEIGHT_ALPHA': [1],
    'SOLVE_WEIGHT_BETA': [0.9],
    'OBJECTIVE': ['maxprofit'], # maxprofit
    'WRITE_OUTPUT_GEOTIFFS': [True],
    'RESFACTOR': [5],
    'SIM_YEARS': [[i for i in range(2010,2051,5)]],

    # ----------------------------------- GHG settings --------------------------------
    'GHG_CONSTRAINT_TYPE': ['hard'],
    'CARBON_PRICES_FIELD': ['CONSTANT'],
    'CARBON_EFFECTS_WINDOW': [91],

    # ----------------------------- Biodiversity settings -------------------------------
    'GBF2_CONSTRAINT_TYPE': ['hard'],
    'GBF2_TARGETS_DICT': [{
        'low': {2030: 0, 2050: 0, 2100: 0},
        'medium': {2030: 0.30, 2050: 0.30, 2100: 0.30},
        'high': {2030: 0.30, 2050: 0.50, 2100: 0.50},
    }],
    'BIO_QUALITY_LAYER': ['Suitability'],
    'BIODIVERSITY_TARGET_GBF_3': ['off'],
    'BIODIVERSITY_TARGET_GBF_4_SNES': ['off'],
    'BIODIVERSITY_TARGET_GBF_4_ECNES': ['off'],
    'BIODIVERSITY_TARGET_GBF_8': ['off'],

    # ----------------------------------- Water settings --------------------------------
    'WATER_STRESS': [0.6],
    'WATER_LIMITS': ['on'],
    'WATER_CONSTRAINT_TYPE': ['hard'],
    'INCLUDE_WATER_LICENSE_COSTS': [1],
    'WATER_REGION_DEF': ['Drainage Division'],
    'WATER_CLIMATE_CHANGE_IMPACT': ['on'],
    # ----------------------------------- Demand settings --------------------------------
    'DEMAND_CONSTRAINT_TYPE': ['soft'],
    #----------------------------------- other settings --------------------------------
    'REGIONAL_ADOPTION_CONSTRAINTS': ['off'],

    "AG_MANAGEMENTS": [{
        'Asparagopsis taxiformis': False,
        'Precision Agriculture': False,
        'Ecological Grazing': False,
        'Savanna Burning': True,
        'AgTech EI': False,
        'Biochar': False,
        'HIR - Beef': True,
        'HIR - Sheep': True,
    }],
}

conditional_rules = [
    # 最具体的规则优先
    {
        'conditions': {'BIODIVERSITY_TARGET_GBF_2': ['off']},
        'restrictions': {'GBF2_PRIORITY_DEGRADED_AREAS_PERCENTAGE_CUT': [50]}
    }
]


settings_name_dict = {
    'BIODIVERSITY_TARGET_GBF_2':'GBF2',
    'GBF2_PRIORITY_DEGRADED_AREAS_PERCENTAGE_CUT':'CUT',
    'CARBON_PRICE_COSTANT':'CarbonPrice',
}

task_root_dir = f'../../output/{grid_search['TASK_NAME'][0]}'
# grid_search_settings_df = create_grid_search_template(grid_search,settings_name_dict,conditional_rules=conditional_rules)
# print(grid_search_settings_df.columns)
grid_search_settings_df = pd.read_csv(os.path.join(task_root_dir, 'grid_search_template.csv'), index_col=0)
create_task_runs(task_root_dir, grid_search_settings_df, platform="NCI", n_workers=min(len(grid_search_settings_df.columns), 100),use_parallel=True)