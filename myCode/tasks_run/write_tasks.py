import os
import numpy as np
import pandas as pd
from tools.helpers import create_task_runs
import time
# time.sleep(60*60*5)
platform = "NCI"  # 可选值: 'HPC', 'Denethor', 'NCI'
tasks = ['20250908_Paper2_Results_NCI']
input_path_dict = {"HPC": "/home/remote/s222552331/LUTO2_XH/LUTO2/input",
                    "Denethor": "N:/LUF-Modelling/LUTO2_XH/LUTO2/input",
                    "NCI": "/g/data/jk53/LUTO_XH/LUTO2/input"}
for task in tasks:
    task_root_dir = f'../../output/{task}'
    grid_search_settings_df = pd.read_csv(os.path.join(task_root_dir, 'grid_search_template.csv'), index_col=0)

    update_values = {
        'WRITE_THREADS': 2,
        'MEM': '52GB',
        'NCPUS': 13,
        'TIME': '05:00:00',
        'INPUT_DIR': input_path_dict[platform],  # 根据平台选择输入路径
        'RAW_DATA': f"{input_path_dict[platform]}/raw_data",
        'NO_GO_VECTORS': {
            'Winter cereals': f'{input_path_dict[platform]}/no_go_areas/no_go_Winter_cereals.shp',
            'Environmental Plantings': f'{input_path_dict[platform]}/no_go_areas/no_go_Enviornmental_Plantings.shp'
        }
    }

    # 只能使用双层for循环,不能使用广播,否则字典无法写入
    for name, new_value in update_values.items():
        for col in grid_search_settings_df.columns:
            grid_search_settings_df.at[name, col] = new_value

    create_task_runs(task_root_dir, grid_search_settings_df, platform, n_workers=min(len(grid_search_settings_df.columns), 100),use_parallel=True,model_name='Write')
