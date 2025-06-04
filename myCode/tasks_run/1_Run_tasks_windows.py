from myCode.tasks_run.tools.helpers import create_task_runs
import os
import pandas as pd

if os.name == 'nt':
    task_root_dir = '../../output/setting_0603'
    grid_search_settings_df = pd.read_csv(f'{task_root_dir}/grid_search_template.csv')
    create_task_runs(task_root_dir, grid_search_settings_df, mode='single', n_workers=min(len(grid_search_settings_df.columns), 100),use_parallel=False)
else:
    print("This script is designed to run only on Windows.")