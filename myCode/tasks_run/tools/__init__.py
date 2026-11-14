import os
from pathlib import Path
import pandas as pd
import numpy as np
import datetime
import re

def get_path(path_name):
    output_path = f"../../output/{path_name}/output"
    try:
        if os.path.exists(output_path):
            # 获取 output_path 下所有的子目录（文件夹）
            subdirectories = [
                s for s in os.listdir(output_path)
                if os.path.isdir(os.path.join(output_path, s)) and s[0].isdigit()
            ]
            if subdirectories:
                # 返回第一个符合条件的文件夹路径
                subdirectory_path = os.path.join(output_path, subdirectories[0])
                return subdirectory_path
            else:
                raise FileNotFoundError(f"No numeric starting folder found in {output_path}")
        else:
            raise FileNotFoundError(f"The specified output path does not exist: {output_path}")
    except (IndexError, FileNotFoundError) as e:
        print(f"Error occurred while getting path for {path_name}: {e}")
        if os.path.exists(output_path):
            print(f"Current directory content for {output_path}: {os.listdir(output_path)}")
        else:
            print(f"Directory not found: {output_path}")
        return None


def get_folders_in_directory(directory_path):
    """
    获取某目录下所有文件夹的名称。

    Args:
        directory_path (str): 目标目录的路径。

    Returns:
        list: 文件夹名称列表。
    """
    path = Path(directory_path)
    folder_names = [f.name for f in path.iterdir() if f.is_dir()]
    return folder_names

def calculate_total_cost(df):
    def convert_time_to_hours(time_obj):
        if pd.isna(time_obj) or time_obj is None:
            return 0
        if isinstance(time_obj, datetime.time):
            return time_obj.hour + time_obj.minute / 60 + time_obj.second / 3600
        if isinstance(time_obj, str) and ":" in time_obj:
            h, m, s = map(int, time_obj.split(":"))
            return h + m / 60 + s / 3600
        return 0

    # Remove unused columns
    columns_to_remove = {'Default_run'}.intersection(df.columns)
    df = df.drop(columns=columns_to_remove)
    df.columns = df.columns.astype(str)
    df = df.loc[:, ~df.columns.str.startswith('Unnamed')]
    if 'Name' in df.columns:
        df = df.set_index('Name')

    cpu_row = df.loc['NCPUS']
    mem_row = df.loc['MEM']
    time_row = df.loc['TIME']

    # CPU as float Series
    cpu_values = cpu_row.astype(float)

    # MEM: 提取数字部分并转为 float Series
    mem_values = mem_row.str.replace(r'([0-9]*\.?[0-9]+).*', r'\1', regex=True).astype(float)
    # TIME: 转小时
    time_values = time_row.apply(convert_time_to_hours).astype(float)

    # Memory-based CPUs
    memory_cpu_values = np.ceil(mem_values / 4)

    # 保证所有变量都是 Series
    effective_cpus = pd.concat([cpu_values, memory_cpu_values], axis=1).max(axis=1)

    every_cost = 2 * effective_cpus * time_values
    total_cost = every_cost.sum() / 1000
    print(f"Every Job Cost: {every_cost.iloc[0]},Number:{len(every_cost)}")
    print(f"Total Job Cost: {total_cost}k")
    return total_cost
