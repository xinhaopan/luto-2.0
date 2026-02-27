import os
import numpy as np
import pandas as pd
from tools.helpers import create_grid_search_template,create_task_runs

grid_search = {
    'TASK_NAME': ['20260226_Paper2_Results_NCI'],
    'KEEP_OUTPUTS': [True],  # If False, only keep report HTML
    'QUEUE': ['normalsr'],
    'NUMERIC_FOCUS': [3],
    # ---------Computational settings, which are not relevant to LUTO itself---------
    'MEM': ['60GB'],
    'NCPUS': ['15'],
    'WRITE_THREADS': ['2'],
    'TIME': ['48:00:00'],

    'GHG_EMISSIONS_LIMITS': ['high', 'low', 'off'],
    'BIODIVERSITY_TARGET_GBF_2': ['high', 'off'],
    'GBF2_PRIORITY_DEGRADED_AREAS_PERCENTAGE_CUT': [50,40,30,20,10],
    # ---------------------------------- Model settings ------------------------------
    'SOLVE_WEIGHT_ALPHA': [1],
    'SOLVE_WEIGHT_BETA': [0.9],
    'OBJECTIVE': ['maxprofit'], # maxprofit
    'WRITE_OUTPUT_GEOTIFFS': [True],
    'RESFACTOR': [5],
    'SIM_YEARS': [[i for i in range(2010,2051,1)]],

    # ----------------------------------- GHG settings --------------------------------
    'GHG_CONSTRAINT_TYPE': ['hard'],
    'CARBON_PRICES_FIELD': ['CONSTANT'],
    'CARBON_EFFECTS_WINDOW': [60],
    'GHG_TARGETS_DICT': [{
        'off': None,
        'low': '1.8C (67%) excl. avoided emis SCOPE1',
        'medium': '1.5C (50%) excl. avoided emis SCOPE1',
        'high': '1.5C (67%) excl. avoided emis SCOPE1',
    }],

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
    'DYNAMIC_PRICE': [False],
    'DEMAND_CONSTRAINT_TYPE': ['soft'],
    #----------------------------------- other settings --------------------------------
    'REGIONAL_ADOPTION_CONSTRAINTS': ['off'],
    'NON_AG_LAND_USES': [{
        'Environmental Plantings': True,
        'Riparian Plantings': True,
        'Sheep Agroforestry': True,
        'Beef Agroforestry': True,
        'Carbon Plantings (Block)':  True,
        'Sheep Carbon Plantings (Belt)':  True,
        'Beef Carbon Plantings (Belt)':  True,
        'BECCS': False,
        'Destocked - natural land': True,
    }]
}

conditional_rules = [
    # 最具体的规则优先
    # {
    #     'conditions': {'GHG_EMISSIONS_LIMITS': ['off'], 'BIODIVERSITY_TARGET_GBF_2': ['off']},
    #     'restrictions': {'GBF2_PRIORITY_DEGRADED_AREAS_PERCENTAGE_CUT': [50]}
    # },
    # {
    #     'conditions': {'GHG_EMISSIONS_LIMITS': ['high'], 'BIODIVERSITY_TARGET_GBF_2': ['off']},
    #     'restrictions': {'GBF2_PRIORITY_DEGRADED_AREAS_PERCENTAGE_CUT': [50]}
    # },
    # {
    #     'conditions': {'GHG_EMISSIONS_LIMITS': ['low'], 'BIODIVERSITY_TARGET_GBF_2': ['off']},
    #     'restrictions': {'GBF2_PRIORITY_DEGRADED_AREAS_PERCENTAGE_CUT': [50]}
    # },
    {
        'conditions': {'GHG_EMISSIONS_LIMITS': ['off'], 'BIODIVERSITY_TARGET_GBF_2': ['high']},
        'restrictions': {}
    }
]


settings_name_dict = {
    'GHG_EMISSIONS_LIMITS':'GHG',
    'BIODIVERSITY_TARGET_GBF_2':'BIO',
    'GBF2_PRIORITY_DEGRADED_AREAS_PERCENTAGE_CUT':'CUT'
}

task_root_dir = f'../../output/{grid_search['TASK_NAME'][0]}'
grid_search_settings_df = create_grid_search_template(grid_search,settings_name_dict,conditional_rules=conditional_rules)
print(grid_search_settings_df.columns)
# grid_search_settings_df = pd.read_csv(os.path.join(task_root_dir, 'grid_search_template.csv'), index_col=0)
create_task_runs(task_root_dir, grid_search_settings_df, platform="NCI", n_workers=min(len(grid_search_settings_df.columns), 50),use_parallel=True)