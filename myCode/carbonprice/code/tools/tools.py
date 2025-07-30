import numpy as np
import pandas as pd
import os
import re
import rasterio
import tools.config as config


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
    output_path = f"{config.TASK_DIR}/{path_name}/output"
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


def npy_to_map(input_arr, output_tif, proj_file,
               fill_value=np.nan, shift=0,
               dtype=rasterio.float32):
    """
    将一维 .npy 数组铺回到栅格地图中。
    - input_arr: path to .npy (1D array of length = number of valid pixels in proj_file)
    - output_tif: 输出 GeoTIFF 路径
    - proj_file: 用于投影和形状参照的已有 GeoTIFF
    - fill_value: 初始填充值（默认 np.nan）
    - shift: 在写入前对数据统一加的偏移量
    - dtype: 输出栅格的数据类型
    """
    if not input_arr.lower().endswith(".npy"):
        return

    # 1) 读取参考栅格
    with rasterio.open(proj_file) as src:
        mask2D = src.read(1) >= 0
        transform = src.transform
        crs = src.crs
        profile = src.profile.copy()
        shape = src.shape

    # 2) 加载一维数组
    nonzeroes = np.where(mask2D)
    lumap = np.load(input_arr)

    if lumap.ndim != 1:
        raise ValueError(f"{input_arr} 中的数组不是一维的")
    if len(lumap) != len(nonzeroes[0]):
        print(f"Warning: {input_arr} 的长度为 {len(lumap)}, proj_file 中有效像元数量为 {len(nonzeroes[0])}.")
        raise ValueError("lumap 的长度与 proj_file 中的有效像元数量不一致")

    # 3) 构建全图，并赋值
    themap = np.full(shape, fill_value=fill_value, dtype=float)
    themap[nonzeroes] = lumap + shift

    # 4) 把 +/- inf 都变成 np.nan
    themap[~np.isfinite(themap)] = np.nan

    # 5) 更新 profile 并写出
    profile.update({
        'dtype': dtype,
        'count': 1,
        'compress': 'lzw',
        'nodata': fill_value
    })
    with rasterio.open(output_tif, 'w', **profile) as dst:
        dst.write(themap.astype(dtype), 1)

    return output_tif