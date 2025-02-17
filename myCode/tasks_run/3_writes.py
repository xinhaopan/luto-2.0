import os
import subprocess
import sys
import shutil
import pandas as pd
from joblib import Parallel, delayed
import re

from tools.helpers import create_task_runs

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
    从 CSV 文件读取 file_dir，并检查相应目录下是否存在 'data_with_solution.pkl'
    :param csv_filename: CSV 文件的路径
    """
    try:
        # 读取 CSV 文件
        df = pd.read_csv(csv_filename, index_col=0)

        file_dirs = [col for col in df.columns if col not in ['Default_run']]

        for file_dir in file_dirs:
            working_directory = os.path.join("../../output", file_dir)

            # 确保工作目录存在
            time_file_path = os.path.join(working_directory, 'output')
            if not os.path.exists(time_file_path):
                print(f"Time file path does not exist: {time_file_path}")
                continue

            # 获取 time_file_dirs
            time_file_dirs = [name for name in os.listdir(time_file_path) if
                              os.path.isdir(os.path.join(time_file_path, name))]
            if not time_file_dirs:
                print(f"{file_dir} without {time_file_path}")
                # delete_folder_multiprocessing(os.path.join(file_dir, time_file_dir))
                continue

            for time_file_dir in time_file_dirs:
                if "2010-2050" in time_file_dir:
                    pkl_path = os.path.join(time_file_path, time_file_dir, 'data_with_solution.pkl')
                    print(f"Checking PKL file at path: {pkl_path}")

                    # 检查 PKL 文件是否存在
                    if not os.path.exists(pkl_path):
                        print(f"PKL file does not exist at path: {pkl_path}")
                else:
                    print(f"{file_dir} without {time_file_dir}")

    except Exception as e:
        print(f"Error processing CSV file '{csv_filename}': {e}")

if __name__ == "__main__":
    csv_path = "Custom_runs/setting_template_windows_0216.csv"
    check_from_csv(csv_path)
    create_task_runs(csv_path, use_multithreading=False, num_workers=5, script_name="1_write")


