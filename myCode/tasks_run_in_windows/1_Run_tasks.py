import os

def main():
    os.chdir('../..')  # 这里改变工作目录
    from myCode.tasks_run_in_windows.tools.helpers import create_task_runs
    create_task_runs(use_multithreading=True, num_workers=3)

if __name__ == "__main__":
    main()
