import numpy as np
import pandas as pd
import os
import re

from tools.desc import *


def apply_operations_on_files(path_dir, files_with_ops):
    """
    读取给定路径下的多个 .npy 文件，并根据指定操作（+ 或 -）进行加减运算。

    :param path_dir: 文件所在的目录
    :param files_with_ops: 包含 (文件名, 操作) 的元组列表，操作为 '+' 或 '-'
    :return: 所有文件内容的加减总和
    """
    total_sum = None

    for file, operation in files_with_ops:
        file_path = os.path.join(path_dir, file)
        if os.path.exists(file_path):
            # 读取 .npy 文件
            data = np.load(file_path)

            # 如果是第一次加载数据，初始化 total_sum
            if total_sum is None:
                total_sum = data if operation == '+' else -data
            else:
                if operation == '+':
                    total_sum += data
                elif operation == '-':
                    total_sum -= data
        else:
            print(f"File not found: {file_path}")

    return total_sum


def apply_sum_on_files_with_prefix(path_dir, file_prefix):
    """
    对路径下所有以指定前缀开头的文件进行加法运算。

    :param path_dir: 文件所在的目录
    :param file_prefix: 文件名前缀（如 'cost_non_ag'）
    :return: 所有以指定前缀开头的文件内容的总和
    """
    # 获取目录下所有以指定前缀开头的文件名
    # 创建一个包含 cost_dict 和 revenue_dict 的字典
    combined_dict = {
        'cost_am': cost_dict.get('cost_am', []),
        'cost_non_ag': cost_dict.get('cost_non_ag', []),
        'revenue_am': revenue_dict.get('revenue_am', []),
        'revenue_non_ag': revenue_dict.get('revenue_non_ag', [])
    }

    # 优化后的逻辑
    files_with_ops = []
    if file_prefix in combined_dict:
        for file in os.listdir(path_dir):
            # 遍历 combined_dict[file_prefix] 中的每一个元素
            for prefix in combined_dict[file_prefix]:
                # 检查文件名是否以 file_prefix + "_" + prefix 开头
                if file.startswith(file_prefix + "_" + prefix):
                    files_with_ops.append((file, '+'))  # 如果匹配则加入列表
    else:
        files_with_ops = [(file, '+') for file in os.listdir(path_dir) if file.startswith(file_prefix)]

    if not files_with_ops:
        print(f"No files found starting with '{file_prefix}'")
        return None

    # 调用 apply_operations_on_files 函数
    total_sum = apply_operations_on_files(path_dir, files_with_ops)

    return total_sum


def process_and_save(path_dir, save_path, prefix, year, rows_nums):
    """
    对指定前缀的文件进行求和，保存结果，并将结果加入 rows_nums。

    :param path_dir: 文件所在的目录
    :param save_path: 保存 .npy 文件的路径
    :param prefix: 文件名前缀（如 'cost_ag', 'cost_non_ag' 等）
    :param year: 当前年份，用于命名保存的文件
    :param rows_nums: 用于存储各项结果的列表
    :return: 更新后的 rows_nums
    """
    print(prefix)
    arr = apply_sum_on_files_with_prefix(path_dir, prefix)
    if arr is None:
        print(f"No valid data found for {prefix} in year {year}")
    rows_nums.append(np.sum(arr) / 1000000)
    if prefix.endswith('_'):
        prefix = prefix[:-1]
    np.save(os.path.join(save_path, f"{prefix}_{year}.npy"), arr)
    return rows_nums


def process_files_with_operations(path_dir, save_path, files_with_ops, file_prefix, year, rows_nums, negate=False):
    """
    处理文件加减操作，将结果保存为 .npy 文件并更新 rows_nums。

    :param path_dir: 文件所在的目录
    :param save_path: 保存 .npy 文件的路径
    :param files_with_ops: 包含 (文件名, 操作符) 的元组列表，操作符为 '+' 或 '-'
    :param file_prefix: 保存文件的前缀
    :param year: 当前年份，用于命名保存的文件
    :param rows_nums: 用于存储各项结果的列表
    :param negate: 是否对结果进行取负操作
    :return: 更新后的 rows_nums
    """
    files_with_ops = [(file + ".npy", op) for file, op in files_with_ops]

    # 处理文件加减操作
    arr = apply_operations_on_files(path_dir, files_with_ops)

    if arr is None:
        print(f"No valid data found for {file_prefix} in year {year}")
        rows_nums.append(0)
        return rows_nums

    # 如果需要取负操作，则取负
    if negate:
        arr = -arr

    # 保存结果到 .npy 文件
    if file_prefix.endswith('_'):
        file_prefix = file_prefix[:-1]
    np.save(os.path.join(save_path, f"{file_prefix}_{year}.npy"), arr)

    # 将结果添加到 rows_nums，按百万单位
    rows_nums.append(np.sum(arr) / 1000000)

    return rows_nums

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
