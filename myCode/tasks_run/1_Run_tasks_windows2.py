from myCode.tasks_run.tools.helpers import create_task_runs
import os

if os.name == 'nt':
    input_file = 'Custom_runs/setting_paper1_0314_20.csv'
    create_task_runs(input_file, use_multithreading=True, num_workers=9)
else:
    print("This script is designed to run only on Windows.")