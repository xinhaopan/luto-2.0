import os
import numpy as np
import pandas as pd
from tools.helpers import create_grid_search_template,create_task_runs

grid_search = {
    'TASK_NAME': ['20260502_paper4_HPC'],
    'KEEP_OUTPUTS': [False],  # If False, only keep ZIP
    'QUEUE': ['normalsr'],
    'NUMERIC_FOCUS': [2], 
    # ---------Computational settings, which are not relevant to LUTO itself---------
    'MEM': ['40GB'],
    'NCPUS': ['10'], 
    'WRITE_THREADS': ['2'],
    'TIME': ['6:00:00'],

    'GHG_EMISSIONS_LIMITS': ['off'],
    'BIODIVERSITY_TARGET_GBF_2': ['off'],
    'GBF2_PRIORITY_DEGRADED_AREAS_PERCENTAGE_CUT': [50],
    'CARBON_PRICES_FIELD': ['CONSTANT'],
    'CARBON_PRICE_COSTANT': [i for i in range(0,361,20)], # [0,8.92,17.85,26.77,35.69,44.61,53.54,62.46,75.84,89.23,133.84,178.46,233.07,267.69,312.3,356.92],
    'BIODIVERSITY_PRICES_FIELD': ['CONSTANT'],
    'BIODIVERSITY_PRICE_CONSTANT': [i for i in range(0,90001,5000)], # [0, 5500, 11000, 16500, 22000, 27500, 33000, 38500,44000, 49500, 55000, 60500, 66000, 71500, 77000, 82500],
    # ---------------------------------- Model settings ------------------------------
    'SOLVE_WEIGHT_ALPHA': [1],
    'SOLVE_WEIGHT_BETA': [0.9],
    'OBJECTIVE': ['maxprofit'], # maxprofit
    'WRITE_OUTPUT_GEOTIFFS': [True],
    'RESFACTOR': [5],
    'SIM_YEARS': [[i for i in range(2010,2051,5)]],

    # ----------------------------------- GHG settings --------------------------------
    'GHG_CONSTRAINT_TYPE': ['hard'],
    'CARBON_EFFECTS_WINDOW': [60],

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
        'Asparagopsis taxiformis': True,
        'Precision Agriculture': True,
        'Ecological Grazing': True,
        'Savanna Burning': True,
        'AgTech EI': True,
        'Biochar': True,
        'HIR - Beef': True,
        'HIR - Sheep': True,
        'Utility Solar PV':False,
        'Onshore Wind':False,
    }],
}

carbon_prices = grid_search['CARBON_PRICE_COSTANT']
bio_prices = grid_search['BIODIVERSITY_PRICE_CONSTANT']

conditional_rules = [
    # When biodiversity price > 0, carbon price must be 0
    {
        'conditions':    {'BIODIVERSITY_PRICE_CONSTANT': [price for price in bio_prices if price > 0]},
        'restrictions':  {'CARBON_PRICE_COSTANT': [0]},
    },
    # When carbon price > 0, biodiversity price must be 0
    {
        'conditions':    {'CARBON_PRICE_COSTANT': [price for price in carbon_prices if price > 0]},
        'restrictions':  {'BIODIVERSITY_PRICE_CONSTANT': [0]},
    },
]


settings_name_dict = {
    'CARBON_PRICE_COSTANT': 'CarbonPrice',
    'BIODIVERSITY_PRICE_CONSTANT': 'BioPrice',
}

task_root_dir = f'../../output/{grid_search['TASK_NAME'][0]}'
grid_search_settings_df = create_grid_search_template(grid_search,settings_name_dict,conditional_rules=conditional_rules)
print(grid_search_settings_df.columns)
# grid_search_settings_df = pd.read_csv(os.path.join(task_root_dir, 'grid_search_template.csv'), index_col=0)
create_task_runs(task_root_dir, grid_search_settings_df, platform="HPC", n_workers=min(len(grid_search_settings_df.columns), 16),use_parallel=True)
