import os
import numpy as np
from tools.helpers import create_settings_template, generate_csv,create_grid_search_template

grid_search = {
    ###############################################################
    # Task run settings for submitting the job to the cluster
    ###############################################################
    'MEM': ['300'],
    'NCPUS': [75],
    'TIME': ['20:00:00'],
    'QUEUE': ['normalsr'],

    ###############################################################
    # Working settings for the model run
    ###############################################################
    'MODE': ['snapshot'],                # 'snapshot' or 'timeseries'
    'RESFACTOR': [1],
    'WRITE_THREADS': [10],
    'WRITE_OUTPUT_GEOTIFFS': [True],
    ###############################################################
    # Scenario settings for the model run
    ###############################################################
    'SOLVE_ECONOMY_WEIGHT': [0.25],
    'GHG_CONSTRAINT_TYPE': ['soft'],
    'GHG_LIMITS_FIELD': [
        '1.5C (67%) excl. avoided emis',
        '1.5C (50%) excl. avoided emis',
        '1.8C (67%) excl. avoided emis'
    ],
    'BIODIV_GBF_TARGET_2_DICT': [
        {2010: 0, 2030: 0, 2050: 0, 2100: 0 },
        {2010: 0, 2030: 0.3, 2050: 0.3, 2100: 0.3 },
        {2010: 0, 2030: 0.3, 2050: 0.5, 2100: 0.5 }
    ]
}


template_df = create_settings_template('Custom_runs')
# generate_csv(output_csv="Custom_runs/setting_template_windows_test1.csv")

output_file = os.path.join("Custom_runs", "setting_template_linux_0.csv")
suffix='RESFACTOR'
create_grid_search_template(template_df, grid_search,output_file,suffix)
print(f"saved to {output_file}")

