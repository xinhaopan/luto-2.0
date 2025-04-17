import os
import numpy as np
from tools.helpers import create_grid_search_template

grid_search = {
    # Computational settings, which are not relevant to LUTO itself
    'MEM': ['4'],
    'NCPUS': ['1'],
    'TIME': ['1:00:00'],
    'MODE': [
        # 'snapshot',
        'timeseries'
    ],
    'SOLVE_WEIGHT_ALPHA': [0.8],
    'SOLVE_WEIGHT_BETA': [0.98],
    'OBJECTIVE': ['maxprofit'], # maxprofit
    'WRITE_OUTPUT_GEOTIFFS': [True],
    'RESFACTOR': [5],
    'SIM_YERAS': [[i for i in range(2010,2051,1)]],

    # GHG settings
    'GHG_EMISSIONS_LIMITS' : ['on'],
    'GHG_CONSTRAINT_TYPE': ['hard'],
    'GHG_LIMITS_FIELD': [
        '1.5C (67%) excl. avoided emis SCOPE1',
        '1.5C (50%) excl. avoided emis SCOPE1',
        '1.8C (67%) excl. avoided emis SCOPE1'
    ],
    'CARBON_PRICES_FIELD': ['CONSTANT'],

    # Biodiversity settings
    'BIODIVERSTIY_TARGET_GBF_2': ['on'],
    'GBF2_CONSTRAINT_TYPE': ['hard'],
    'BIODIV_GBF_TARGET_2_DICT': [
        {2010: 0, 2030: 0, 2050: 0, 2100: 0 },
        {2010: 0, 2030: 0.3, 2050: 0.3, 2100: 0.3 },
        {2010: 0, 2030: 0.3, 2050: 0.5, 2100: 0.5 }
    ],
    'BIODIVERSTIY_TARGET_GBF_3': ['off'],
    'BIODIVERSTIY_TARGET_GBF_4_SNES': ['off'],
    'BIODIVERSTIY_TARGET_GBF_4_ECNES': ['off'],
    'BIODIVERSTIY_TARGET_GBF_8': ['off'],
    'GBF2_PRIORITY_DEGRADED_AREAS_PERCENTAGE_CUT': [20],

    # Water settings
    'WATER_LIMITS': ['on'],
    'WATER_CONSTRAINT_TYPE': ['hard'],
    'INCLUDE_WATER_LICENSE_COSTS': [1],

    # Demand settings
    'DEMAND_CONSTRAINT_TYPE': ['soft'],
}

suffixs = ['GHG_LIMITS_FIELD','BIODIV_GBF_TARGET_2_DICT','RESFACTOR']

output_file = os.path.join("Custom_runs", "setting_0416_5.csv")
create_grid_search_template(grid_search,output_file,suffixs)
print(f"saved to {output_file}")

