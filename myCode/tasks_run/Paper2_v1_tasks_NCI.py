import os
import numpy as np
import pandas as pd
from tools.helpers import create_grid_search_template,create_task_runs

# Paper 2 revision runs (One Earth, ONE-EARTH-D-26-01194R1) with the current LUTO2 and cell grouping
# Code   branch paper2_v1 (team repository land-use-trade-offs, jinzhu branch merged), input copied from Jinzhu
# Grid   the April grid of Paper2_tasks_NCI.py (20260414_Paper2_Results_NCI) plus GHG off with BIO high (the NP-only
#        runs, restoration first, added on 2026-11-12), so 3 x 2 x 5 = 30 runs. Run_01 to Run_20 keep the April names,
#        Run_21 to Run_25 are now GHG off BIO high and GHG off BIO off moves to Run_26 to Run_30
# GHG    GHG_NET_ROW puts back the single net row of the first submission (luto/settings.py, luto/solvers/row_builder.py)
#        low  = 1.8C (67%) excl. avoided emis SCOPE1 with the 1.8C demand
#        high = 1.5C (67%) excl. avoided emis SCOPE1 with the 1.5C demand
# SSP370 takes the RCP 6.0 yield layer in the code (luto/settings.py YIELD_RCP), say so when writing the paper
# Settings that no longer exist in the current code are left out
#        NUMERIC_FOCUS, WRITE_THREADS, SOLVE_WEIGHT_ALPHA/BETA, WRITE_OUTPUT_GEOTIFFS, GHG_TARGETS_DICT (old meaning),
#        *_CONSTRAINT_TYPE (rows are hard, demand uses DEMAND_BOUNDS)
# Outputs are written after the solve by a write_only job (python_script.py), since writing at RES1 with
#        cell grouping is untested (Jinzhu, 2026-11-18). Data_RES1.lz4 is saved either way (luto/simulation.py:172)

grid_search = {
    'TASK_NAME': ['20261119_Paper2_v1_NCI'],
    'KEEP_OUTPUTS': [False],  # False zips each run into Run_Archive.zip and deletes the rest (gdata file-count quota)
    'QUEUE': ['normalsr'],
    # ---------Computational settings, which are not relevant to LUTO itself---------
    'MEM': ['500GB'],                     # set from the test run before submitting
    'NCPUS': ['104'],
    'TIME': ['48:00:00'],

    'GHG_EMISSIONS_LIMITS': ['high', 'low', 'off'],
    'GBF2_TARGET': ['high', 'off'],
    'GBF2_PRIORITY_DEGRADED_AREAS_PERCENTAGE_CUT': [50,40,30,20,10],
    # ---------------------------------- Model settings ------------------------------
    'OBJECTIVE': ['maxprofit'],
    'RESFACTOR': [1],
    'SIM_YEARS': [[i for i in range(2010,2051,1)]],
    'WRITE_OUTPUTS': [False],

    # ------------------------- Cell grouping (Jinzhu, 2026-11-18) -------------------------
    'CELL_GROUPING': [True],
    'BIO_ROW_GENERATION': [False],        # raises an error with cell grouping
    'ELASTIC_FAMILIES': [[]],             # raises an error with cell grouping

    # ----------------------------------- GHG settings --------------------------------
    'GHG_NET_ROW': [True],
    'GHG_NET_TARGETS_DICT': [{
        'low':  '1.8C (67%) excl. avoided emis SCOPE1',
        'high': '1.5C (67%) excl. avoided emis SCOPE1',
    }],
    'CARBON_PRICES_FIELD': ['CONSTANT'],
    'CARBON_EFFECTS_WINDOW': [60],

    # ----------------------------- Biodiversity settings -------------------------------
    'GBF2_TARGETS_DICT': [{
        'off': None,
        'low': {2030: 0, 2050: 0, 2100: 0},
        'medium': {2030: 0.30, 2050: 0.30, 2100: 0.30},
        'high': {2030: 0.30, 2050: 0.50, 2100: 0.50},
    }],
    'BIO_QUALITY_LAYER': ['Suitability'],
    'GBF3_NVIS_TARGET': ['off'],
    'GBF4_TARGET_SNES': ['off'],
    'GBF4_TARGET_ECNES': ['off'],
    'GBF8_TARGET': ['off'],

    # ----------------------------------- Water settings --------------------------------
    'WATER_STRESS': [0.6],
    'WATER_LIMITS': ['on'],
    'INCLUDE_WATER_LICENSE_COSTS': [1],
    'WATER_REGION_DEF': ['Drainage Division'],
    'WATER_CLIMATE_CHANGE_IMPACT': ['on'],
    # ----------------------------------- Demand settings --------------------------------
    'DYNAMIC_PRICE': [False],             # the current code defaults to True, the first submission used False
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

conditional_rules = []   # the April grid left out GHG off with BIO high, the revision keeps it (NP-only runs)


settings_name_dict = {
    'GHG_EMISSIONS_LIMITS':'GHG',
    'GBF2_TARGET':'BIO',
    'GBF2_PRIORITY_DEGRADED_AREAS_PERCENTAGE_CUT':'CUT'
}

task_root_dir = f'../../output/{grid_search['TASK_NAME'][0]}'
grid_search_settings_df = create_grid_search_template(grid_search,settings_name_dict,conditional_rules=conditional_rules)
print(grid_search_settings_df.columns)
# grid_search_settings_df = pd.read_csv(os.path.join(task_root_dir, 'grid_search_template.csv'), index_col=0)
create_task_runs(task_root_dir, grid_search_settings_df, platform="NCI", n_workers=min(len(grid_search_settings_df.columns), 50),use_parallel=True)
