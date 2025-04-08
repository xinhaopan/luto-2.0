import numpy as np
import pandas as pd
import os
import re


def get_year(path_name):
    # 列出目录中的所有文件
    for file_name in os.listdir(path_name):
        if file_name.startswith("begin_end_compare_"):
            # 使用正则表达式提取年份
            match = re.search(r'(\d{4})_(\d{4})', file_name)
            if match:
                year_start, year_end = map(int, match.groups())
                return list(range(year_start, year_end + 1))
    return []

def get_path(path_name):
    output_path = f"../../../output/{path_name}/output"
    try:
        if os.path.exists(output_path):
            subdirectories = os.listdir(output_path)
            # numeric_starting_subdir = [s for s in subdirectories if s[0].isdigit()][0]
            numeric_starting_subdir = [s for s in subdirectories if "2010-2050" in s][0]
            subdirectory_path = os.path.join(output_path, numeric_starting_subdir)
            return subdirectory_path
        else:
            raise FileNotFoundError(f"The specified output path does not exist: {output_path}")
    except (IndexError, FileNotFoundError) as e:
        print(f"Error occurred while getting path for {path_name}: {e}")
        print(f"Current directory content for {output_path}: {os.listdir(output_path) if os.path.exists(output_path) else 'Directory not found'}")
