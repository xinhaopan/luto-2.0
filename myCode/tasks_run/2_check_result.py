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

def delete_folder_multiprocessing(folder_path, num_processes=32):
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


def check_success(folders, base_path="../../output",
                  check_file="DATA_REPORT/REPORT_HTML/pages/water_usage.html",
                  pkl_file="data_with_solution.pkl"):
    """
    检查文件夹列表中的路径是否存在，验证目标文件是否存在，并分类成功与失败。

    Args:
        folders (list): 要检查的文件夹列表。
        base_path (str): 基础路径，默认 "../../output"。
        check_file (str): 检查的文件相对路径。
        pkl_file (str): 检查的 .pkl 文件相对路径。

    Returns:
        tuple: 成功的文件夹列表 (success_folders, partial_success_folders, failed_folders)。
    """
    # 初始化成功、部分成功和失败的文件夹列表
    success_full_file_paths = []
    partial_success_folders = []  # .pkl 存在，但目标文件不存在
    failed_folders = []

    for folder in folders:
        path = os.path.join(base_path, folder, 'output')

        # 检查路径是否存在
        if not os.path.exists(path):
            failed_folders.append(folder)
            continue  # 跳过当前文件夹，检查下一个

        # 获取子文件夹路径
        file_paths = [name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name))]

        for file_path in file_paths:
            # 构建完整的子文件夹路径
            full_file_path = os.path.join(path, file_path)

            # 检查子文件夹是否为空或无效
            if not os.path.exists(full_file_path):
                failed_folders.append(folder)
                break  # 跳出循环，标记文件夹为失败

            # 检查目标文件
            check_path = os.path.join(full_file_path, check_file)
            pkl_path = os.path.join(full_file_path, pkl_file)

            if os.path.exists(check_path):
                success_full_file_paths.append(full_file_path)  # 目标文件存在，成功
                break  # 找到目标文件后，跳过对该文件夹的后续检查
            elif os.path.exists(pkl_path):
                partial_success_folders.append(folder)  # pkl 文件存在，部分成功
                break  # 跳过对该文件夹的后续检查
        else:
            # 如果两个文件都不存在，标记为失败
            failed_folders.append(folder)

    # 输出结果
    print("成功的文件夹：")
    for folder in success_full_file_paths:
        print(folder)

    print("\n部分成功的文件夹（仅 pkl 文件存在）：")
    for folder in partial_success_folders:
        print(folder)

    print("\n失败的文件夹：")
    for folder in failed_folders:
        print(folder)

    return success_full_file_paths, partial_success_folders, failed_folders


def delete_except(folder_path, full_name, preserve_log="simulation_log.txt"):
    """
    删除路径下除指定前缀文件夹和文件以外的所有内容，同时保留指定的日志文件。

    :param folder_path: 要清理的路径
    :param full_name: 传入的完整文件夹或文件名称，用于提取前缀。
    :param preserve_log: 始终保留的文件名称，默认为 "simulation_log.txt"。
    """
    # 提取前 17 位作为前缀
    prefix_to_keep = full_name[:17]
    print(f"开始清理路径：{folder_path}")

    for item in os.listdir(folder_path):
        item_path = os.path.join(folder_path, item)

        # 如果是要保留的文件夹或文件，跳过
        if prefix_to_keep in item or item == preserve_log:
            continue

        # 针对 Windows 系统使用 \\?\ 前缀
        if platform.system() == "Windows":
            item_path = f"\\\\?\\{os.path.abspath(item_path)}"

        # 删除文件或文件夹
        try:
            if os.path.isdir(item_path):
                shutil.rmtree(item_path)  # 删除文件夹及其内容
            elif os.path.isfile(item_path):
                os.remove(item_path)  # 删除文件
        except Exception as e:
            print(f"删除失败：{item_path}")

def save_failed_columns(csv_path, write_failed_cols, failed_cols):
    """
    保存原 CSV 文件中的前两列，以及分别保存 write_failed 和 failed 的列到新文件。

    Args:
        csv_path (str): 原 CSV 文件路径。
        write_failed_cols (list): write_failed 的列名列表。
        failed_cols (list): failed 的列名列表。
    """
    # 读取原 CSV 文件
    df = pd.read_csv(csv_path)

    # 提取前两列
    first_two_columns = df.iloc[:, :2]

    # 提取 write_failed 和 failed 的列
    write_failed_data = df[write_failed_cols]
    failed_columns_data = df[failed_cols]

    # 保存 write_failed 数据
    write_failed_df = pd.concat([first_two_columns, write_failed_data], axis=1)
    write_failed_csv_path = csv_path.replace('.csv', '_write_failed.csv')
    write_failed_df.to_csv(write_failed_csv_path, index=False)
    print(f"写失败的列数据已保存到文件：{write_failed_csv_path}")

    # 保存 failed 数据
    failed_df = pd.concat([first_two_columns, failed_columns_data], axis=1)
    failed_csv_path = csv_path.replace('.csv', '_failed.csv')
    failed_df.to_csv(failed_csv_path, index=False)
    print(f"失败的列数据已保存到文件：{failed_csv_path}")

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

    success_full_file_paths, partial_success_folders, failed_folders = check_success(columns_to_check)

    # 保存失败的列名到新文件
    if partial_success_folders or failed_folders:
        save_failed_columns(csv_path, partial_success_folders, failed_folders)
    else:
        print("所有列名对应的文件都存在，无需保存失败的列名。")
    if success_full_file_paths:
        for success_path in success_full_file_paths:
            normalized_path = os.path.normpath(success_path)
            directory, last_part = os.path.split(normalized_path)
            delete_except(directory, last_part)


csv_path = 'Custom_runs/setting_template_windows_4.csv'
check_csv(csv_path)



