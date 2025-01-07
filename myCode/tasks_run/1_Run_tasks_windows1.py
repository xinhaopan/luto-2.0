from myCode.tasks_run.tools.helpers import create_task_runs
import os

if os.name == 'nt':
    input_file = 'Custom_runs/setting_template_windows_05_1.csv'
    create_task_runs(input_file, use_multithreading=False, num_workers=5)
else:
    print("This script is designed to run only on Windows.")