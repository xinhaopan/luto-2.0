import dill
import gzip
import os
import time
from datetime import datetime
import glob
import shutil
from luto import tools
from luto.data import Data
from luto import settings
from luto.tools.report.create_report_layers import save_report_layer
from luto.tools.report.create_report_data import save_report_data

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


def create_report(data: Data):
    """Create report using dynamic timestamp from read_timestamp."""

    # Generate path using read_timestamp each time this function is called
    current_timestamp = tools.read_timestamp()
    save_dir = f"{settings.OUTPUT_DIR}/{current_timestamp}_RF{settings.RESFACTOR}_{settings.SIM_YEARS[0]}-{settings.SIM_YEARS[-1]}"
    log_path = f"{save_dir}/LUTO_RUN_"

    @tools.LogToFile(log_path, mode='a')
    def _create_report():
        print('Creating report...')
        print(' --| Copying report template...')
        shutil.copytree('luto/tools/report/VUE_modules', f"{data.path}/DATA_REPORT", dirs_exist_ok=True)
        print(' --| Creating chart data...')
        save_report_data(data.path)
        print(' --| Creating map data...')
        save_report_layer(data)
        print(' --| Report created successfully!')

    return _create_report()


# 调用函数并加载数据
# pkl_path = get_first_subfolder_name("output")
pkl_path = 'output/2025_10_03__15_55_58_RF13_2020-2050/Data_RES13.gz'
print_with_time(f"PKL file path: {pkl_path}")
# 如果文件存在，则加载并处理数据
print_with_time(f"{pkl_path} Loading...")

with gzip.open(pkl_path, 'rb') as f:
    data = dill.load(f)

# Update the timestamp
# data.timestamp_sim = datetime.now().strftime('%Y_%m_%d__%H_%M_%S')

print_with_time("Writing report...")
create_report(data)
print_with_time("Report created successfully.")
