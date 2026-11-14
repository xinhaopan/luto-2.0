import os
import numpy as np
import pandas as pd
from tools.helpers import create_grid_search_template,create_task_runs

# Test of the current LUTO2 (branch paper2_v1, from land-use-trade-offs/jinzhu e95e9c0d, 2026-11-11)
# Input: copy of Jinzhu's NCI input /g/data/jk53/jinzhu/LUTO/luto-2.0/input (2026-10-22)
# climate_change_impacts_rcp7p0_* in the input is the RCP6.0 layer under the rcp7p0 name
# Test only: 2020-2050 in 5-year steps. The formal Paper 2 runs are 2010-2050 yearly.

grid_search = {
    'TASK_NAME': ['20261114_Paper2_v1_test'],
    'KEEP_OUTPUTS': [True],  # If False, only keep report HTML
    'QUEUE': ['normalsr'],
    # ---------Computational settings, which are not relevant to LUTO itself---------
    'MEM': ['128GB', '500GB'],
    'NCPUS': ['32', '104'],
    'TIME': ['10:00:00', '48:00:00'],
    'WRITE_REPORT_MAX_MEM_MB': [100 * 1024, 300 * 1024],

    # ---------------------------------- Model settings ------------------------------
    'RESFACTOR': [5, 1],
    'SIM_YEARS': [[i for i in range(2020,2051,5)]],
    'WRITE_OUTPUTS': [True],

    # ----------------------------------- GHG settings --------------------------------
    'GHG_EMISSIONS_LIMITS': ['low'],      # two rows in the current code: agriculture and net LULUCF

    # ----------------------------- Biodiversity settings -------------------------------
    'GBF2_TARGET': ['high'],
}

# RES5 and RES1 each take their own resources
conditional_rules = [
    {
        'conditions': {'RESFACTOR': [5]},
        'restrictions': {'MEM': ['128GB'], 'NCPUS': ['32'], 'TIME': ['10:00:00'], 'WRITE_REPORT_MAX_MEM_MB': [100 * 1024]}
    },
    {
        'conditions': {'RESFACTOR': [1]},
        'restrictions': {'MEM': ['500GB'], 'NCPUS': ['104'], 'TIME': ['48:00:00'], 'WRITE_REPORT_MAX_MEM_MB': [300 * 1024]}
    },
]


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
