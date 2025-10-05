import os
import numpy as np
import pandas as pd
from tools.helpers import create_task_runs
import time
# time.sleep(60*60*5)
platform = "HPC"  # 可选值: 'HPC', 'Denethor', 'NCI'
tasks = ['20251003_Paper2_Results']

for task in tasks:
    task_root_dir = f'../../output/{task}'
    grid_search_settings_df = pd.read_csv(os.path.join(task_root_dir, 'grid_search_template.csv'), index_col=0)

    for task in tasks:
        task_root_dir = f'../../output/{task}'
        grid_search_settings_df = pd.read_csv(os.path.join(task_root_dir, 'grid_search_template.csv'), index_col=0)

        update_values = {
            'KEEP_OUTPUTS': False,
        }

    # 只能使用双层for循环,不能使用广播,否则字典无法写入
    for name, new_value in update_values.items():
        for col in grid_search_settings_df.columns:
            grid_search_settings_df.at[name, col] = new_value

    create_task_runs(task_root_dir, grid_search_settings_df, platform, n_workers=min(len(grid_search_settings_df.columns), 4),use_parallel=True,model_name='Zip')
