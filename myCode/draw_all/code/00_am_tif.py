import rasterio
import numpy as np
import os
from tools.parameters import *
from tools.data_helper import *

def shift_tif_values(input_tif, output_tif):
    with rasterio.open(input_tif) as src:
        data = src.read(1)
        profile = src.profile

    # 创建一个副本，避免连锁替换错误
    new_data = data.copy()

    # 使用 np.where 或布尔掩码做精准替换
    new_data[data == 3] = 4
    new_data[data == 4] = 5
    new_data[data == 5] = 6
    new_data[data == 6] = 7
    new_data[data == 7] = 8

    # 写入新 tif 文件
    with rasterio.open(output_tif, 'w', **profile) as dst:
        dst.write(new_data, 1)

for input_name in input_files:
    path = get_path(input_name)
    input_path = os.path.join(path,"out_2050","ammap_2050.tiff")
    output_path = os.path.join(path,"out_2050","ammap_2050_1.tiff")
    shift_tif_values(input_path, output_path)
