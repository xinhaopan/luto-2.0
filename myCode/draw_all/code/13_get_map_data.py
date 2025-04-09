import os
import rasterio
import numpy as np

def process_tif_files(folder_path, threshold=0.01):
    """
    将文件夹中所有 .tif 文件中值小于 threshold 的像元设为 0 并保存。

    参数:
    - folder_path: str, tif 文件所在文件夹路径
    - threshold: float, 小于该值的像元将设为 0
    """
    # 遍历文件夹中的所有 tif 文件
    for filename in os.listdir(folder_path):
        if filename.lower().endswith('.tiff'):
            file_path = os.path.join(folder_path, filename)
            print(f"🔄 正在处理: {filename}")

            # 读取 tif 文件
            with rasterio.open(file_path) as src:
                data = src.read(1)  # 读取第一波段
                profile = src.profile  # 保存元数据

            # 将小于阈值的像元设为 0
            data[data < threshold] = 0

            # 保存为原文件（覆盖）
            with rasterio.open(file_path, 'w', **profile) as dst:
                dst.write(data, 1)

            print(f"✅ 已处理并保存: {filename}")

# 输入文件夹路径
folder_path = r"N:\LUF-Modelling\LUTO2_XH\Map\Data\output"
process_tif_files(folder_path, threshold=0.1)
