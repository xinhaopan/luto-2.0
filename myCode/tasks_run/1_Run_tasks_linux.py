from myCode.tasks_run.tools.helpers import create_task_runs
import os

if os.name == 'posix':
    input_file = 'Custom_runs/setting_0409_test9.csv'
    create_task_runs(input_file, use_multithreading=False, num_workers=3)
else:
    print("This script is designed to run only on Windows.")


