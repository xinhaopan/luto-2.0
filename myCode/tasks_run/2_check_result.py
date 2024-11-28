import os
import pandas as pd
import shutil
import platform
from multiprocessing import Pool

from tools import get_folders_in_directory, get_path,calculate_total_cost

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

def check_success(folders, base_path="../../output", check_file="DATA_REPORT/REPORT_HTML/pages/water_usage.html"):
    """
    检查文件夹列表中的路径是否存在，验证目标文件是否存在，并分类成功与失败。

    Args:
        folders (list): 要检查的文件夹列表。
        base_path (str): 基础路径，默认 "../../output"。
        check_file (str): 检查的文件相对路径。

    Returns:
        tuple: 成功和失败的文件夹列表 (success_folders, failed_folders)。
    """
    # 初始化成功和失败的文件夹名称列表
    success_folders = []
    success_path = []
    failed_folders = []

    for folder in folders:
        path = os.path.join(base_path, folder, 'output')

        # 检查 path 是否存在
        if not os.path.exists(path):
            failed_folders.append(folder)
            continue  # 跳过当前文件夹，检查下一个

        # 获取子文件夹路径
        file_paths = [name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name))]

        for file_path in file_paths:
            # 构建完整的子文件夹路径
            full_file_path = os.path.join(path, file_path)

            # 检查 file_path 是否为空或无效
            if not os.path.exists(full_file_path):
                failed_folders.append(folder)
                break  # 跳出循环，标记文件夹为失败

            # 构建目标文件路径
            check_path = os.path.join(full_file_path, check_file)

            # 检查目标文件是否存在
            if os.path.exists(check_path):
                success_folders.append(folder)
                success_path.append(full_file_path)
                break  # 找到目标文件后，跳过对该文件夹的后续检查
        else:
            # 如果没有找到目标文件，标记为失败
            failed_folders.append(folder)

    # 输出成功的文件夹
    print("成功的文件夹：")
    for folder in success_folders:
        print(folder)

    # 输出失败的文件夹
    print("\n失败的文件夹：")
    for folder in failed_folders:
        print(folder)

    return success_folders, success_path, failed_folders


def delete_except(folder_path, folder_to_keep, file_to_keep='simulation_log.txt'):
    """
    删除路径下除指定文件夹和文件以外的所有内容，支持跨平台。

    :param folder_path: 要清理的路径
    :param folder_to_keep: 要保留的文件夹名称
    :param file_to_keep: 要保留的文件名称
    """
    print(f"开始清理路径：{folder_path}")
    for item in os.listdir(folder_path):
        item_path = os.path.join(folder_path, item)

        # 如果是要保留的文件夹，跳过
        if os.path.isdir(item_path) and item == folder_to_keep:
            continue

        # 如果是要保留的文件，跳过
        if os.path.isfile(item_path) and item == file_to_keep:
            continue

        # 针对 Windows 系统使用 \\?\ 前缀
        if platform.system() == "Windows":
            item_path = f"\\\\?\\{os.path.abspath(item_path)}"

        # 删除文件或文件夹
        try:
            if os.path.isfile(item_path):
                os.remove(item_path)  # 删除文件
            elif os.path.isdir(item_path):
                delete_folder_multiprocessing(item_path)  # 删除文件夹及其内容
        except Exception as e:
            print(f"Error deleting {item_path}: {e}")

def save_failed_columns(csv_path, failed_cols):
    """
    保存原 CSV 文件中的前两列以及失败的列到新文件。

    Args:
        csv_path (str): 原 CSV 文件路径。
        failed_cols (list): 失败的列名列表。
    """
    # 读取原 CSV 文件
    df = pd.read_csv(csv_path)

    # 提取前两列
    first_two_columns = df.iloc[:, :2]

    # 提取失败的列
    failed_columns_data = df[failed_cols]

    # 合并前两列和失败的列
    combined_df = pd.concat([first_two_columns, failed_columns_data], axis=1)

    # 保存到新文件
    failed_csv_path = csv_path.replace('.csv', '_failed.csv')
    combined_df.to_csv(failed_csv_path, index=False)
    print(f"失败的列数据已保存到文件：{failed_csv_path}")
    calculate_total_cost(combined_df)

def check_csv(csv_path):
    """
    检查 CSV 文件中的列，排除 'Name' 和 'Default_run' 后，验证剩余列对应的文件是否存在。
    将失败的列名保存到一个新文件。

    Args:
        csv_path (str): 输入的 CSV 文件路径。
    """
    # 读取 CSV 文件
    df = pd.read_csv(csv_path)

    # 提取所有列名，并排除 'Name' 和 'Default_run'（即使它们不存在）
    columns_to_check = [col for col in df.columns if col not in ['Name', 'Default_run']]

    _, success_paths, failed_cols = check_success(columns_to_check)

    # 保存失败的列名到新文件
    if failed_cols:
        save_failed_columns(csv_path, failed_cols)
    else:
        print("所有列名对应的文件都存在，无需保存失败的列名。")
    if success_paths:
        for success_path in success_paths:
            normalized_path = os.path.normpath(success_path)
            directory, last_part = os.path.split(normalized_path)
            delete_except(directory, last_part)


csv_path = 'Custom_runs/settings_template_windows_all_failed.csv'
check_csv(csv_path)



