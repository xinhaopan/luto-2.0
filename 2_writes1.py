import os
import subprocess
import sys
import shutil
import pandas as pd
from joblib import Parallel, delayed
import re
from pathlib import Path

from myCode.tasks_run.tools.helpers import create_task_runs

def delete_path(path):
    """删除单个文件或文件夹"""
    try:
        if os.path.isfile(path) or os.path.islink(path):
            os.remove(path)
        elif os.path.isdir(path):
            shutil.rmtree(path)
        print(f"Deleted: {path}")
    except Exception as e:
        print(f"Error deleting {path}: {e}")

def delete_folder_multiprocessing(folder_path, num_processes=128):
    """使用多进程删除单个目录内容"""
    if not os.path.exists(folder_path):
        print(f"Path does not exist: {folder_path}")
        return

    # 列出所有文件和子目录
    entries = [os.path.join(folder_path, entry) for entry in os.listdir(folder_path)]

    # 使用多进程删除每个子文件或子目录
    with Pool(processes=num_processes) as pool:
        pool.map(delete_path, entries)

    # 删除空的根文件夹
    os.rmdir(folder_path)
    print(f"Folder '{folder_path}' has been completely deleted.")



def check_from_csv(csv_filename):
    """
    从 CSV 文件读取 file_dir，并检查相应目录下是否存在 'data_with_solution.pkl'。
    将 error 的列保留，删除所有 success 的列，生成新的 CSV 文件，文件名为原文件名后加 '_1'。
    :param csv_filename: CSV 文件的路径
    """
    try:
        # 读取 CSV 文件
        df = pd.read_csv(csv_filename, index_col=0)

        error_dirs = []  # 存储 error 的 file_dir 列名
        file_dirs = [col for col in df.columns if col not in ['Default_run']]

        for file_dir in file_dirs:
            working_directory = os.path.join("output", file_dir)

            # 确保工作目录存在
            time_file_path = os.path.join(working_directory, 'output')
            if not os.path.exists(time_file_path):
                print(f"Time file path does not exist: {time_file_path}")
                error_dirs.append(file_dir)
                continue

            # 获取 time_file_dirs
            time_file_dirs = [name for name in os.listdir(time_file_path) if
                              os.path.isdir(os.path.join(time_file_path, name))]
            if not time_file_dirs:
                print(f"{file_dir} error: without {time_file_path}")
                error_dirs.append(file_dir)
                continue

            has_2010_2050 = False
            for time_file_dir in time_file_dirs:
                if "2010-2050" in time_file_dir:
                    has_2010_2050 = True
                    pkl_path = os.path.join(time_file_path, time_file_dir, 'data_with_solution.pkl')

                    # 检查 PKL 文件是否存在
                    if os.path.exists(pkl_path):
                        print(f"{file_dir} success: with pkl")
                    else:
                        print(f"{file_dir} error: without pkl")
                        error_dirs.append(file_dir)

            if not has_2010_2050:
                print(f"{file_dir} error: without 2010-2050 directory")
                error_dirs.append(file_dir)

        # **创建新的 CSV 文件，仅保留 error 的列**
        if error_dirs:
            new_csv_filename = csv_filename.replace(".csv", "_1.csv")
            df_error = df[error_dirs]  # 仅保留 error 对应的列
            df_error.to_csv(new_csv_filename)
            print(f"错误文件已保存: {new_csv_filename}")
            return new_csv_filename

    except Exception as e:
        print(f"Error processing CSV file '{csv_filename}': {e}")


if __name__ == "__main__":
    run_path = "myCode/tasks_run"
    csv_path = "Custom_runs/setting_paper1_0312_1.csv"
    new_csv_filename = check_from_csv(os.path.join(run_path, csv_path))
    os.chdir(run_path)
    # create_task_runs("/".join(Path(new_csv_filename).parts[-2:]) , use_multithreading=True, num_workers=3, script_name="1_write")
    create_task_runs(csv_path, use_multithreading=True, num_workers=3,script_name="1_write")


