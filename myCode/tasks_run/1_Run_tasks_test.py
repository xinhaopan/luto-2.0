import os

def main():
    if os.name == 'nt':
        os.chdir('../..')  # 这里改变工作目录
        from myCode.tasks_run.tools.helpers import create_task_runs
        input_file = 'myCode/tasks_run/Custom_runs/settings_template_windows_7backup.csv'
        create_task_runs(input_file, use_multithreading=False, num_workers=5)
    else:
        print("This script is designed to run only on Windows.")

if __name__ == "__main__":
    main()
