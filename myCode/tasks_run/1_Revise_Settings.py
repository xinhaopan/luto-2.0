import os.path

from tools.helpers import create_settings_template, generate_csv,create_grid_search_template

create_settings_template('Custom_runs')
# generate_csv(output_csv="Custom_runs/setting_template_windows_test1.csv")

grid_search = {
    # Computational settings, which are not relevant to LUTO itself
    'MEM': ['72'],
    'CPU_PER_TASK': [18],
    'TIME': ['3:30:00'],

    'MODE': [
        # 'snapshot',
        'timeseries'
    ],
    'RESFACTOR': [10],

    'SOLVE_ECONOMY_WEIGHT': [0.25],
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

map_dict = {
    'GHG_LIMITS_FIELD': grid_search['GHG_LIMITS_FIELD'],
    'CARBON_PRICES_FIELD': [
        ghg[:9].replace('(', '').replace(')', '') for ghg in grid_search['GHG_LIMITS_FIELD']
    ]
}

output_file = os.path.join("Custom_runs", "setting_template_windows_0_test.csv")
grid_search_df = create_grid_search_template(grid_search,map_dict,output_file)

