import sys
from pathlib import Path


TASKS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TASKS_DIR.parents[1]
for path in (TASKS_DIR, REPO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from tools.helpers import create_grid_search_template, create_task_runs


DEMAND_BOUNDS_1_TO_101 = {
    'sheep lexp':             [1.0, 1.01],
    'sheep meat':             [1.0, 1.01],
    'sheep wool':             [1.0, 1.01],
    'apples':                 [1.0, 1.01],
    'beef lexp':              [1.0, 1.01],
    'beef meat':              [1.0, 1.01],
    'citrus':                 [1.0, 1.01],
    'cotton':                 [1.0, 1.01],
    'dairy':                  [1.0, 1.01],
    'grapes':                 [1.0, 1.01],
    'hay':                    [1.0, 1.01],
    'nuts':                   [1.0, 1.01],
    'other non-cereal crops': [1.0, 1.01],
    'pears':                  [1.0, 1.01],
    'plantation fruit':       [1.0, 1.01],
    'rice':                   [1.0, 1.01],
    'stone fruit':            [1.0, 1.01],
    'sugar':                  [1.0, 1.01],
    'summer cereals':         [1.0, 1.01],
    'summer legumes':         [1.0, 1.01],
    'summer oilseeds':        [1.0, 1.01],
    'tropical stone fruit':   [1.0, 1.01],
    'vegetables':             [1.0, 1.01],
    'winter cereals':         [1.0, 1.01],
    'winter legumes':         [1.0, 1.01],
    'winter oilseeds':        [1.0, 1.01],
}


grid_search = {
    'TASK_NAME': ['20260529_test_food_hard_res5_1yr_nci'],
    'KEEP_OUTPUTS': [False],
    'QUEUE': ['normalsr'],
    'NUMERIC_FOCUS': [2],

    # Computational settings
    'MEM': ['40GB'],
    'NCPUS': ['10'],
    'WRITE_THREADS': ['2'],
    'TIME': ['15:00:00'],

    # Price and environmental target settings kept neutral for the food test.
    'GHG_EMISSIONS_LIMITS': ['off'],
    'BIODIVERSITY_TARGET_GBF_2': ['off'],
    'GBF2_PRIORITY_DEGRADED_AREAS_PERCENTAGE_CUT': [50],
    'CARBON_PRICES_FIELD': ['CONSTANT'],
    'CARBON_PRICE_COSTANT': [0],
    'BIODIVERSITY_PRICES_FIELD': ['CONSTANT'],
    'BIODIVERSITY_PRICE_CONSTANT': [0],

    # Model settings
    'SOLVE_WEIGHT_ALPHA': [1],
    'SOLVE_WEIGHT_BETA': [0.9],
    'OBJECTIVE': ['maxprofit'],
    'WRITE_OUTPUT_GEOTIFFS': [True],
    'RESFACTOR': [5],
    'SIM_YEARS': [[i for i in range(2010, 2051, 1)]],

    # GHG settings
    'GHG_CONSTRAINT_TYPE': ['hard'],
    'CARBON_EFFECTS_WINDOW': [60],

    # Biodiversity settings
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

    # Water settings
    'WATER_STRESS': [0.6],
    'WATER_LIMITS': ['on'],
    'WATER_CONSTRAINT_TYPE': ['hard'],
    'INCLUDE_WATER_LICENSE_COSTS': [1],
    'WATER_REGION_DEF': ['Drainage Division'],
    'WATER_CLIMATE_CHANGE_IMPACT': ['on'],

    # Demand / food settings
    'DIET_DOM': ['BAU', 'VGN'],
    'DIET_GLOB': ['BAU', 'VGN'],
    'CONVERGENCE': [2050],
    'IMPORT_TREND': ['Static'],
    'WASTE': [1],
    'FEED_EFFICIENCY': ['BAU'],
    'DYNAMIC_PRICE': [False],
    'DEMAND_CONSTRAINT_TYPE': ['hard'],
    'DEMAND_BOUNDS': [DEMAND_BOUNDS_1_TO_101],

    # Other settings
    'REGIONAL_ADOPTION_CONSTRAINTS': ['off'],
    'AG_MANAGEMENTS': [{
        'Asparagopsis taxiformis': True,
        'Precision Agriculture': True,
        'Ecological Grazing': True,
        'Savanna Burning': True,
        'AgTech EI': True,
        'Biochar': True,
        'HIR - Beef': True,
        'HIR - Sheep': True,
        'Utility Solar PV': False,
        'Onshore Wind': False,
    }],
}


conditional_rules = [
    {
        'conditions': {'DIET_DOM': ['BAU']},
        'restrictions': {'DIET_GLOB': ['BAU']},
    },
    {
        'conditions': {'DIET_DOM': ['VGN']},
        'restrictions': {'DIET_GLOB': ['VGN']},
    },
]


settings_name_dict = {
    'DIET_DOM': 'DietDom',
    'DIET_GLOB': 'DietGlob',
}


task_root_dir = f"../../output/{grid_search['TASK_NAME'][0]}"
grid_search_settings_df = create_grid_search_template(
    grid_search,
    settings_name_dict,
    conditional_rules=conditional_rules,
)
print(grid_search_settings_df.columns)
create_task_runs(
    task_root_dir,
    grid_search_settings_df,
    platform='NCI',
    n_workers=min(len(grid_search_settings_df.columns), 2),
    use_parallel=True,
    max_concurrent_tasks=2,
)
