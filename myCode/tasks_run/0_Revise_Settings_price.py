import os
import numpy as np
from tools.helpers import create_grid_search_template

grid_search = {
    # Computational settings, which are not relevant to LUTO itself
    'MEM': ['4'],
    'NCPUS': ['1'],
    'TIME': ['01:00:00'],
    'MODE': [
        # 'snapshot',
        'timeseries'
    ],
    'OBJECTIVE': ['maxprofit'],
    'WRITE_OUTPUT_GEOTIFFS': [False],
    'RESFACTOR': [20],
    'GBF2_PRIORITY_CRITICAL_AREA_PERCENTAGE': [20],
    # 'SOLVE_BIODIV_PRIORITY_WEIGHT': [0],
    # GHG settings
    'GHG_EMISSIONS_LIMITS': ['off','on'],
    'GHG_CONSTRAINT_TYPE': ['hard'],
    'GHG_LIMITS_FIELD': [
        # '1.5C (67%) excl. avoided emis',
        '1.5C (50%) excl. avoided emis',
        # '1.8C (67%) excl. avoided emis'
    ],
    'CARBON_PRICES_FIELD': ['c0'],
    # Water settings
    'WATER_CONSTRAINT_TYPE': ['soft'],
    # Biodiversity settings
    'BIODIVERSTIY_TARGET_GBF_2': ['off','on'],
    'GBF2_CONSTRAINT_TYPE': ['soft'],
    'HCAS_PERCENTILE': [50],
    'BIODIV_GBF_TARGET_2_DICT': [
        # {2010: 0, 2030: 0, 2050: 0, 2100: 0 },
        {2010: 0, 2030: 0.3, 2050: 0.3, 2100: 0.3 },
        # {2010: 0, 2030: 0.3, 2050: 0.5, 2100: 0.5 }
    ],

    'BIODIVERSTIY_TARGET_GBF_3': ['off'],
    'BIODIVERSTIY_TARGET_GBF_4': ['off'],
    'INCLUDE_WATER_LICENSE_COSTS ': [1],

}

suffixs = ['GHG_EMISSIONS_LIMITS','BIODIVERSTIY_TARGET_GBF_2','RESFACTOR']

output_file = os.path.join("Custom_runs", "setting_0408_cp.csv")
create_grid_search_template( grid_search,output_file,suffixs)
print(f"saved to {output_file}")

