import os
import pandas as pd
from tools.helpers import create_grid_search_template,create_task_runs

grid_search = {
    'TASK_NAME': ['20260529_paper4'],
    'KEEP_OUTPUTS': [False],  # If False, only keep ZIP
    'QUEUE': ['normalsr'],
    'NUMERIC_FOCUS': [2],
    # ---------Computational settings, which are not relevant to LUTO itself---------
    'MEM': ['40GB'],
    'NCPUS': ['10'],
    'WRITE_THREADS': ['2'],
    'TIME': ['12:00:00'],

    'GHG_EMISSIONS_LIMITS': ['off'],
    'BIODIVERSITY_TARGET_GBF_2': ['off'],
    'GBF2_PRIORITY_DEGRADED_AREAS_PERCENTAGE_CUT': [50],
    'CARBON_PRICES_FIELD': ['CONSTANT'],
    # Carbon: 0 to 360, step = 20 (multiple of 10) → 19 points
    'CARBON_PRICE_COSTANT': list(range(0, 361, 20)),
    'BIODIVERSITY_PRICES_FIELD': ['CONSTANT'],
    # Biodiversity: 0 to 22,000, step = 1000 → 23 points
    # Headline P50 = A$22,000/ha (NSW BOS ecosystem-credit median: A$4,000/credit × 5.5 credits/ha)
    'BIODIVERSITY_PRICE_CONSTANT': list(range(0, 22001, 1000)),
    # ---------------------------------- Model settings ------------------------------
    'SOLVE_WEIGHT_ALPHA': [1],
    'SOLVE_WEIGHT_BETA': [0.9],
    'OBJECTIVE': ['maxprofit'], # maxprofit
    'WRITE_OUTPUT_GEOTIFFS': [True],
    'RESFACTOR': [5],
    'SIM_YEARS': [[i for i in range(2010,2026,1)]],

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
    'DYNAMIC_PRICE': [False],
    'DEMAND_CONSTRAINT_TYPE': ['hard'],
    'DEMAND_BOUNDS': [{
        'sheep lexp': [1.0, 1.0],
        'sheep meat': [1.0, 1.0],
        'sheep wool': [0, 1.5],

        # all other commodities
        'apples': [1.0, 1.0],
        'beef lexp': [1.0, 1.0],
        'beef meat': [1.0, 1.0],
        'citrus': [1.0, 1.0],
        'cotton': [1.0, 1.0],
        'dairy': [1.0, 1.0],
        'grapes': [1.0, 1.0],
        'hay': [1.0, 1.0],
        'nuts': [1.0, 1.0],
        'other non-cereal crops': [1.0, 1.0],
        'pears': [1.0, 1.0],
        'plantation fruit': [1.0, 1.0],
        'rice': [1.0, 1.0],
        'stone fruit': [1.0, 1.0],
        'sugar': [1.0, 1.0],
        'summer cereals': [1.0, 1.0],
        'summer legumes': [1.0, 1.0],
        'summer oilseeds': [1.0, 1.0],
        'tropical stone fruit': [1.0, 1.0],
        'vegetables': [1.0, 1.0],
        'winter cereals': [1.0, 1.0],
        'winter legumes': [1.0, 1.0],
        'winter oilseeds': [1.0, 1.0],
    }],
    #----------------------------------- other settings --------------------------------
    'REGIONAL_ADOPTION_CONSTRAINTS': ['off'],

    "AG_MANAGEMENTS": [{
        'Asparagopsis taxiformis': True,
        'Precision Agriculture': True,
        'Ecological Grazing': False,
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
