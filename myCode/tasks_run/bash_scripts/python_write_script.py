import os
import time
from datetime import datetime
import glob
from luto.tools.write import write_outputs
import luto.settings as settings
import joblib
import shutil
import zipfile
import sys

def print_with_time(message):
    """打印带有时间戳的信息"""
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}")

def get_first_subfolder_name(output_path="output"):
    """
    获取 output 文件夹下第一个符合条件的子文件夹，并找到数据文件。

    参数:
        output_path (str): output 文件夹路径，默认为 "output"。

    返回:
        str or int: 数据文件路径, 如果报告已存在则返回 1, 否则返回 None。
    """
    try:
        # 查找包含 '2010-2050' 的子文件夹
        subfolders = [f for f in os.listdir(output_path) if
                      os.path.isdir(os.path.join(output_path, f)) and '2010-2050' in f]

        if not subfolders:
            print(f"No subfolders found in '{output_path}'.")
            return None

        first_subfolder = subfolders[0]
        html_path = os.path.join(output_path, first_subfolder, "DATA_REPORT", "data", "map_layers", "map_water_yield_NonAg.js")

        # 检查最终报告文件是否已存在，如果存在则无需再处理
        if os.path.exists(html_path):
            print(f"Final report already exists: {html_path}")
            return 1

        # 查找 lz4 格式的数据文件
        pattern = os.path.join(output_path, first_subfolder, "Data_RES*.lz4")
        found_files = glob.glob(pattern)

        if not found_files:
            print(f"No 'Data_RES*.lz4' file found in '{first_subfolder}'.")
            return None

        return found_files[0]
    except FileNotFoundError:
        print(f"Error: Directory '{output_path}' does not exist.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred in get_first_subfolder_name: {e}")
        return None

def is_zip_valid(path):
    """
    快速检查 zip 文件是否可读。
    - 仅检查文件是否存在，以及zip文件头/目录是否可读。
    - 不会进行耗时的完整CRC校验，速度快得多。
    """
    if not os.path.exists(path):
        return False
    try:
        # 尝试打开文件并读取中央目录，这是一个快速操作。
        # 如果文件尾部损坏或不是zip文件，会抛出 BadZipFile 异常。
        with zipfile.ZipFile(path, 'r') as zf:
            # 读取文件名列表是验证中央目录是否可读的快速方法。
            zf.infolist()
    except zipfile.BadZipFile:
        print_with_time(f"File '{path}' is not a valid zip archive or is corrupted.")
        return False
    # 如果能成功打开并读取目录，就认为它是有效的
    return True
# --- 主要逻辑开始 ---

# 定义归档文件路径
archive_path = './Run_Archive.zip'
archive_filename = os.path.basename(archive_path)

# 1. 检查归档文件是否存在且完好
if os.path.exists(archive_path):
    if is_zip_valid(archive_path):
        print_with_time(f"'{archive_path}' exists and is valid. Cleaning up other files.")
        for item in os.listdir('.'):
            if item != archive_filename:
                try:
                    if os.path.isfile(item) or os.path.islink(item):
                        os.unlink(item)
                    elif os.path.isdir(item):
                        shutil.rmtree(item)
                except Exception as e:
                    print_with_time(f"Failed to delete {item}. Reason: {e}")
        print_with_time("Cleanup complete. Exiting.")
        time.sleep(60)
        sys.exit(0)
    else:
        print_with_time(f"'{archive_path}' is corrupted. Deleting it and proceeding to regenerate.")
        try:
            os.remove(archive_path)
        except Exception as e:
            print_with_time(f"Failed to delete corrupted archive '{archive_path}'. Reason: {e}")
            sys.exit(1) # 如果无法删除损坏的归档，则退出以避免问题

# 2. 如果归档文件不存在，执行原始的数据处理流程
print_with_time("Archive not found. Starting data processing workflow.")
pkl_path = get_first_subfolder_name("output")

if not pkl_path:
    print_with_time("Could not find data file to process. Exiting.")
    # 在某些情况下，可能需要等待文件生成，这里可以保留 sleep
    time.sleep(60)
    raise FileNotFoundError("Could not find valid data file path.")

elif pkl_path == 1:
    print_with_time("Data already processed, but archive was not created. This might indicate an incomplete previous run.")
    # 你可以在这里决定是退出还是尝试重新归档
    time.sleep(60)

else:
    # 如果文件存在，则加载并处理数据
    print_with_time(f"Loading data from: {pkl_path}")
    data = joblib.load(pkl_path)

    print_with_time("Writing outputs...")
    write_outputs(data)
    print_with_time("Data processed successfully.")

    if settings.KEEP_OUTPUTS:
        print_with_time("KEEP_OUTPUTS is True. Skipping archiving and cleanup.")
    else:
        # 3. 创建归档并进行清理
        print_with_time("Archiving results...")
        report_dir = f"{data.path}"

        try:
            with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for root, dirs, files in os.walk(report_dir):
                    for file in files:
                        abs_path = os.path.join(root, file)
                        # 确保文件存在再写入
                        if os.path.exists(abs_path):
                            rel_path = os.path.relpath(abs_path, start=report_dir)
                            zipf.write(abs_path, arcname=rel_path)

            print_with_time("Archiving complete. Cleaning up...")
            # 删除除归档外的所有文件
            for item in os.listdir('.'):
                if item != archive_filename:
                    try:
                        if os.path.isfile(item) or os.path.islink(item):
                            os.unlink(item)
                        elif os.path.isdir(item):
                            shutil.rmtree(item)
                    except Exception as e:
                        print(f"Failed to delete {item}. Reason: {e}")
        except Exception as e:
            print_with_time(f"An error occurred during archiving or cleanup: {e}")
