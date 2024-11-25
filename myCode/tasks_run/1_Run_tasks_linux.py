import os

def main():
    if os.name == 'posix':
        os.chdir('../..')  # 这里改变工作目录
        from myCode.tasks_run.tools.helpers import create_task_runs
        input_file = 'myCode/tasks_run/Custom_runs/GHG_penalty_test_template_1.csv'
        create_task_runs(input_file, use_multithreading=True, num_workers=8)
    else:
        print("This script is only compatible with Linux systems.")

if __name__ == "__main__":
    main()
