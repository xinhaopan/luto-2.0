import numpy as np
import pandas as pd
import os
def sum_files(path_dir, prefix):
    """
    加载并求和给定目录中具有指定前缀的 NumPy 数组文件。

    参数:
    path_dir (str): 包含文件的目录。
    prefix (str): 要匹配文件名的前缀。

    返回:
    np.ndarray: 求和后的 NumPy 数组。
    """
    file_names = list_files_with_prefix(path_dir, prefix)
    arr_list = []
    for file_name in file_names:
        arr = np.load(os.path.join(path_dir, file_name))
        arr = np.nan_to_num(arr, nan=0.0)
        arr_list.append(arr)
    return np.sum(arr_list, axis=0)

def list_files_with_prefix(directory, prefix):
    """
    列出给定目录中以指定前缀开头并以.npy结尾的所有文件。

    参数:
    directory (str): 要搜索文件的目录。
    prefix (str): 要匹配文件名的前缀。

    返回:
    list: 以指定前缀开头并以.npy结尾的文件名列表。
    """
    try:
        files = [f for f in os.listdir(directory) if f.startswith(prefix) and f.endswith('.npy')]
        return files
    except FileNotFoundError:
        return f"Error: The directory '{directory}' does not exist."
    except Exception as e:
        return f"An error occurred: {e}"

def process_year_data(year, path, categories, column_keywords):
    """处理每一年的数据"""
    year_data = {'Year': year}
    for category, files in categories.items():
        total_sum = 0
        for file in files:
            file_path = f'{path}/out_{year}/{file}_{year}.csv'
            data = pd.read_csv(file_path)
            # 使用文件名确定关键字
            keyword = column_keywords.get(file, column_keywords['default'])
            # print(file,keyword)
            value_columns = data.filter(like=keyword)
            total_sum += value_columns.sum().sum()
        year_data[category] = total_sum
    return year_data