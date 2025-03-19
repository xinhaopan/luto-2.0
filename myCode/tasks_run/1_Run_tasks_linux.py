from myCode.tasks_run.tools.helpers import create_task_runs
import os

if os.name == 'posix':
    input_file = 'Custom_runs/setting_0318_linux.csv'
    create_task_runs(input_file, use_multithreading=True, num_workers=-1, delay=0)
else:
    print("This script is designed to run only on Linux.")

