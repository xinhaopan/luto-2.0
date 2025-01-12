import os
import numpy as np
from tools.helpers import create_settings_template, generate_csv,create_grid_search_template

grid_search = {
    # Computational settings, which are not relevant to LUTO itself
    'MEM': ['252'],
    'CPU_PER_TASK': [43],
    'TIME': ['20:00:00'],
    'MODE': [
        'snapshot',
        # 'timeseries'
    ],
    'WRITE_OUTPUT_GEOTIFFS': [True],
    'RESFACTOR': [5],
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

output_file = os.path.join("Custom_runs", "setting_template_windows_100125.csv")
suffix='RESFACTOR'
create_grid_search_template(template_df, grid_search,output_file,suffix)
print(f"saved to {output_file}")

