import pandas as pd
import os
import re
from tools import get_path
def get_folders(file_path):
    """获取文件夹"""
    df = pd.read_csv(file_path)

    # 获取所有列名
    all_columns = df.columns.tolist()

    # 排除 'Name' 和 'Default_run' 列
    filtered_columns = [col for col in all_columns if col not in ["Name", "Default_run"]]
    return filtered_columns

file_path = "../../tasks_run/Custom_runs/setting_template_windows.csv"
folders = get_folders(file_path)
output_result_file = "deviation_results.xlsx"

for folder in folders:
    # 获取文件夹路径
    folder_path = get_path(folder)
    # 获取所有文件
    files = os.listdir(folder_path)

    # 获取所有文件名
    file_names = [file for file in files if re.match(r"^\d{4

