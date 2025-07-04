import os
import numpy as np
import pandas as pd
from tools.helpers import create_task_runs
import time
# time.sleep(60*60*5)
platform = "Denethor"  # 可选值: 'HPC', 'Denethor', 'NCI'
tasks = ['20250705_Paper1_Results']
input_path_dict = {"HPC": "/home/remote/s222552331/LUTO2_XH/LUTO2/input",
                    "Denethor": "N:/LUF-Modelling/LUTO2_XH/LUTO2/input",
                    "NCI": "/g/data/jk53/LUTO_XH/LUTO2/input"}
for task in tasks:
    task_root_dir = f'../../output/{task}'
    grid_search_settings_df = pd.read_csv(os.path.join(task_root_dir, 'grid_search_template.csv'))

    update_values = {
        'WRITE_THREADS': 2,
        'MEM': '150GB',
        'INPUT_DIR': input_path_dict[platform],  # 根据平台选择输入路径
        'RAW_DATA': f"{input_path_dict[platform]}/raw_data",
        'NO_GO_VECTORS': {
            'Winter cereals': f'{input_path_dict[platform]}/no_go_areas/no_go_Winter_cereals.shp',
            'Environmental Plantings': f'{input_path_dict[platform]}/no_go_areas/no_go_Enviornmental_Plantings.shp'
        }
    }
    # 除了 'Name' 列之外的所有列名
    cols_to_update = [col for col in grid_search_settings_df.columns if col != 'Name']

    for name, new_value in update_values.items():
        idxs = grid_search_settings_df.index[grid_search_settings_df['Name'] == name]
        for idx in idxs:
            for col in cols_to_update:
                grid_search_settings_df.at[idx, col] = new_value

    create_task_runs(task_root_dir, grid_search_settings_df, platform, n_workers=min(len(grid_search_settings_df.columns), 50),use_parallel=True,model_name='Write')