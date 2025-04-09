import numpy as np
import sys
import os
import pandas as pd
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

from tools.data_helper import *
from tools.plot_helper import *
from tools.parameters import *


def format_ghg_string(input_string):
    # 提取 GHG_ 后的部分，例如 "1_5C_67"
    match = re.search(r"GHG_(\d+)_(\d+C)_(\d+)", input_string)
    if match:
        # 提取数字部分并格式化
        ghg_value = match.group(1)  # "1"
        temp_value = match.group(2)  # "5C"
        percentage = match.group(3)  # "67"

        # 转换为目标格式
        result = f"{ghg_value}.{temp_value} ({percentage}%) excl. avoided emis"
        return result
    else:
        raise ValueError("Input string does not match the expected format")



plt.rcParams['font.family'] = 'Arial'

# 获取到文件的绝对路径，并将其父目录添加到 sys.path
sys.path.append(os.path.abspath('../../../luto'))

# 导入 settings.py
import settings

font_size = 30
csv_name, value_column_name, filter_column_name = 'area_agricultural_landuse', 'Area (ha)', 'Land-use'
area_dict = get_dict_data(input_files, csv_name, value_column_name, filter_column_name)
area_group_dict = aggregate_by_mapping(area_dict, 'tools/land use group.xlsx', 'desc', 'ag_group')