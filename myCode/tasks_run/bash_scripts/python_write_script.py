import os
import time
from datetime import datetime
import glob
from luto.tools.write import write_outputs
import luto.settings as settings
import joblib
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

        pattern = os.path.join(os.path.join(output_path, subfolders[0]), "Data_RES*.lz4")
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

    data = joblib.load(pkl_path)

    print_with_time("Writing outputs...")
    write_outputs(data)
    print_with_time("Data processed successfully.")

if settings.KEEP_OUTPUTS:

    # Save the data object to disk
    pass

else:
    # Set up report directory and archive path
    report_dir = f"{data.path}"
    archive_path = './Run_Archive.zip'

    # Zip the output directory, and remove the original directory
    with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(report_dir):
            for file in files:
                abs_path = os.path.join(root, file)
                rel_path = os.path.relpath(abs_path, start=report_dir)
                zipf.write(abs_path, arcname=rel_path)

    # Remove all files after archiving
    for item in os.listdir('.'):
        if item != 'Run_Archive.zip':
            try:
                if os.path.isfile(item) or os.path.islink(item):
                    os.unlink(item)
                elif os.path.isdir(item):
                    shutil.rmtree(item)
            except Exception as e:
                print(f"Failed to delete {item}. Reason: {e}")
