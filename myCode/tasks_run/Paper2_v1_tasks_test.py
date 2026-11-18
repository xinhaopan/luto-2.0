import os
import numpy as np
import pandas as pd
from tools.helpers import create_grid_search_template,create_task_runs

# Test of the current LUTO2 with cell grouping (branch paper2_v1, merged from land-use-trade-offs/jinzhu 95c23d5b, 2026-11-18)
# Input: copy of Jinzhu's NCI input /g/data/jk53/jinzhu/LUTO/luto-2.0/input (synced 2026-11-18)
# SSP 370 takes the RCP 6.0 yield layer in the code (luto/settings.py YIELD_RCP), since GAEZ has no RCP 7.0
# RES1 settings follow Jinzhu's notes of 2026-11-18: CELL_GROUPING on, BIO_ROW_GENERATION off, ELASTIC_FAMILIES empty,
# GBF8 off, RETRY_PARAMS and SOLUTION_SHARE_FLOOR left at their defaults, no report (WRITE_OUTPUTS False)

grid_search = {
    'TASK_NAME': ['20261118_Paper2_v1_test'],
    'KEEP_OUTPUTS': [True],  # If False, only keep report HTML
    'QUEUE': ['normalsr'],
    # ---------Computational settings, which are not relevant to LUTO itself---------
    'MEM': ['500GB'],
    'NCPUS': ['104'],
    'TIME': ['48:00:00'],

    # ---------------------------------- Model settings ------------------------------
    'RESFACTOR': [1],
    'SIM_YEARS': [[i for i in range(2010,2051,1)]],
    'WRITE_OUTPUTS': [False],             # writing and the dashboard are untested at RES1 with grouping
    'CELL_GROUPING': [True],
    'BIO_ROW_GENERATION': [False],
    'ELASTIC_FAMILIES': [[]],

    # ----------------------------------- GHG settings --------------------------------
    'GHG_EMISSIONS_LIMITS': ['low'],      # one GHG series (agriculture and LULUCF rows); low or high only sets demand and carbon price

    # ----------------------------- Biodiversity settings -------------------------------
    'GBF2_TARGET': ['high'],
    'GBF8_TARGET': ['off'],
}

conditional_rules = []


settings_name_dict = {
    'RESFACTOR':'RES',
    'GHG_EMISSIONS_LIMITS':'GHG',
    'GBF2_TARGET':'BIO',
}

task_root_dir = f'../../output/{grid_search['TASK_NAME'][0]}'
grid_search_settings_df = create_grid_search_template(grid_search,settings_name_dict,conditional_rules=conditional_rules)
print(grid_search_settings_df.columns)
# grid_search_settings_df = pd.read_csv(os.path.join(task_root_dir, 'grid_search_template.csv'), index_col=0)
create_task_runs(task_root_dir, grid_search_settings_df, platform="NCI", n_workers=min(len(grid_search_settings_df.columns), 50),use_parallel=True)
