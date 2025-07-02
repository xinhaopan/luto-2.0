import os
import numpy as np
import pandas as pd
from tools.helpers import create_task_runs
import time
# time.sleep(60*60*5)
tasks = ['20250702_Paper1_Results_res5','20250703_Paper1_Results_res5']

for task in tasks:
    task_root_dir = f'../../output/{task}'
    grid_search_settings_df = pd.read_csv(os.path.join(task_root_dir, 'grid_search_template.csv'))

    # 修改 'WRITE_THREADS' 的值为 2
    mask = grid_search_settings_df['Name'] == 'WRITE_THREADS'
    # 除了 'Name' 列之外的所有列名
    cols_to_update = [col for col in grid_search_settings_df.columns if col != 'Name']
    # 修改该行
    grid_search_settings_df.loc[mask, cols_to_update] = 2

    # 修改 'WRITE_THREADS' 的值为 2
    mask = grid_search_settings_df['Name'] == 'MEM'
    # 除了 'Name' 列之外的所有列名
    cols_to_update = [col for col in grid_search_settings_df.columns if col != 'Name']
    # 修改该行
    grid_search_settings_df.loc[mask, cols_to_update] = '150GB'

    create_task_runs(task_root_dir, grid_search_settings_df, platform="HPC", n_workers=min(len(grid_search_settings_df.columns), 50),use_parallel=True,model_name='Write')