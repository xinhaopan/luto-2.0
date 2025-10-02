import dill
import gzip
import os
import time
from datetime import datetime
import glob
from luto.tools.write import write_outputs
import luto.settings as settings
import traceback
import shutil, zipfile

def print_with_time(message):
    """打印带有时间戳的信息"""
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}")

def get_first_subfolder_name(output_path="output"):
    """
    获取 output 文件夹下第一个子文件夹的路径，并拼接文件名 'data_with_solution.pkl'。

    参数:
        output_path (str): output 文件夹路径，默认为 "output"。

    返回:
        str: 拼接后的 pkl 文件路径。如果没有子文件夹，返回 None。
    """
    try:
        # 获取所有子文件夹
        # subfolders = [f for f in os.listdir(output_path) if os.path.isdir(os.path.join(output_path, f))]
        subfolders = [f for f in os.listdir(output_path) if
                      os.path.isdir(os.path.join(output_path, f)) and '2010-2050' in f]
        html_path = os.path.join(output_path, subfolders[0],"DATA_REPORT","data","map_layers","map_water_yield_NonAg.js")

        pattern = os.path.join(os.path.join(output_path, subfolders[0]), "Data_RES*.gz")
        found_file = glob.glob(pattern)[0]

        if not subfolders:
            print(f"No subfolders found in '{output_path}'.")
            return None
        elif os.path.exists(html_path):
            print(f"{html_path} exists.")
            return 1
        # 返回拼接后的路径
        else:
            # return os.path.join(output_path, subfolders[0], 'data_with_solution.pkl')
            return found_file
    except FileNotFoundError:
        print(f"Error: Directory '{output_path}' does not exist.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None


# 调用函数并加载数据
pkl_path = get_first_subfolder_name("output")
# pkl_path = 'output/2025_03_30__20_17_18_RF15_2010-2050_snapshot/data_with_solution.gz'
print_with_time(f"PKL file path: {pkl_path}")
if not pkl_path or not os.path.exists(pkl_path):
    time.sleep(60)
    raise FileNotFoundError(f"PKL file does not exist at path: {pkl_path}")
elif pkl_path == 1:
    time.sleep(60)
    print_with_time("Data already processed.")
else:
    # 如果文件存在，则加载并处理数据
    print_with_time(f"{pkl_path} Loading...")

    with gzip.open(pkl_path, 'rb') as f:
        data = dill.load(f)

    # Update the timestamp
    # data.timestamp_sim = datetime.now().strftime('%Y_%m_%d__%H_%M_%S')

    print_with_time("Writing outputs...")
    write_outputs(data)
    print_with_time("Data processed successfully.")

if settings.KEEP_OUTPUTS:

    # Save the data object to disk
    pass

else:
    report_dir = f"{data.path}/DATA_REPORT"
    archive_path = './DATA_REPORT.zip'

    # Zip the output directory, and remove the original directory
    with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(report_dir):
            for file in files:
                abs_path = os.path.join(root, file)
                rel_path = os.path.relpath(abs_path, start=report_dir)
                zipf.write(abs_path, arcname=rel_path)

    # —— 只保留 ZIP 和指定的 RES gz 文件 ——
    keep = {
        os.path.basename(archive_path),  # 'DATA_REPORT.zip'
        f"Data_RES{settings.RESFACTOR}.gz",
    }

    for k in keep:
        if k == os.path.basename(archive_path):
            continue  # DATA_REPORT.zip 已经在当前目录，无需复制
        src = os.path.join(data.path, k)
        dst = os.path.join('.', k)
        if os.path.exists(src):
            if not os.path.exists(dst):
                shutil.copy2(src, dst)
        else:
            print(f"[WARN] 源文件不存在: {src}")

    for item in os.listdir('.'):
        if item in keep:
            continue
        try:
            if os.path.isfile(item) or os.path.islink(item):
                os.unlink(item)
            elif os.path.isdir(item):
                shutil.rmtree(item)
        except Exception as e:
            print(f"Failed to delete {item}. Reason: {e}")
