from myCode.tasks_run.tools.helpers import create_task_runs
import os

if os.name == 'nt':
    input_file = 'Custom_runs/20250430_setting_price_1_5.csv'
    create_task_runs(input_file, use_multithreading=True, num_workers=9)
else:
    print("This script is designed to run only on Windows.")