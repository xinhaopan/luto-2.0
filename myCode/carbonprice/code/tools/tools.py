import numpy as np
import pandas as pd
import os
import re
import rasterio
import xarray as xr
import tools.config as config


def get_year(parent_dir):
    """
    遍历指定目录，找到所有以 "out_" 开头的文件夹，并提取年份。

    Args:
        parent_dir (str): 要搜索的父目录的路径。

    Returns:
        list[int]: 一个包含所有有效年份的整数列表，并按升序排序。
                   如果目录不存在或没有找到匹配的文件夹，则返回一个空列表。
    """
    # 检查父目录是否存在
    if not os.path.isdir(parent_dir):
        print(f"错误: 目录 '{parent_dir}' 不存在或不是一个有效的目录。")
        return []

    years_list = []
    # 遍历父目录下的所有项目（文件和文件夹）
    for item_name in os.listdir(parent_dir):
        # 构造完整的项目路径
        full_path = os.path.join(parent_dir, item_name)

        # 检查该项目是否是一个文件夹，并且其名称是否以 "out_" 开头
        if os.path.isdir(full_path) and item_name.startswith("out_"):
            # 提取 "out_" 后面的部分作为年份字符串
            # "out_" 长度为 4，所以我们从第5个字符开始切片
            year_str = item_name[4:]

            # 尝试将提取的字符串转换为整数，以确保它是一个有效的年份
            # 如果转换失败（例如，文件夹名为 "out_final"），则忽略它
            try:
                year_int = int(year_str)
                years_list.append(year_int)
            except ValueError:
                # 转换失败，说明 "out_" 后面的不是纯数字，静默忽略
                print(f"提示: 忽略无效格式的文件夹 '{item_name}'")
                continue

    # 返回排序后的年份列表，使其更加整洁
    return sorted(years_list)


import xarray as xr
import os


def save2nc(in_xr, save_path: str):
    """
    【最终修正版】
    - 自动处理单变量 Dataset。
    - 健壮地处理分块和非分块的 DataArray，不再产生 chunksizes 错误。
    """
    # 确保保存路径的目录存在
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # --- 1. 统一输入为 DataArray ---
    if isinstance(in_xr, xr.Dataset):
        if len(in_xr.data_vars) == 1:
            # 如果是单变量 Dataset，提取出 DataArray
            var_name = list(in_xr.data_vars)[0]
            da_to_save = in_xr[var_name]
        else:
            raise ValueError(f"输入 Dataset 包含 {len(in_xr.data_vars)} 个变量, 此函数只为单变量设计。")
    elif isinstance(in_xr, xr.DataArray):
        da_to_save = in_xr
        var_name = da_to_save.name or 'data'  # 如果没名字，默认叫 'data'
    else:
        raise TypeError("输入必须是 xarray.DataArray 或 xarray.Dataset 类型。")

    # --- 2. 正确构造编码 (Encoding) ---
    # 这一步至关重要。我们先定义不含 chunksizes 的编码。
    encoding = {var_name: {
        'dtype': 'float32',
        'zlib': True,
        'complevel': 4,
    }}

    # 【关键修复】只有当数据确实是分块的时候，才添加 'chunksizes'
    # hasattr 检查确保 .data.chunks 属性存在
    if hasattr(da_to_save.data, 'chunks') and da_to_save.data.chunks is not None:
        # 如果是分块的，直接使用 xarray 推荐的 .chunks 属性
        encoding[var_name]['chunksizes'] = da_to_save.chunks

    # --- 3. 准备并保存数据 ---
    # 确保 DataArray 有正确的名字
    if da_to_save.name != var_name:
        da_to_save = da_to_save.rename(var_name)

    # 移除所有非维度的坐标，生成更干净的文件
    coords_to_drop = set(da_to_save.coords) - set(da_to_save.dims)
    if coords_to_drop:
        da_to_save = da_to_save.drop_vars(coords_to_drop, errors='ignore')

    da_to_save.astype('float32').to_netcdf(
        path=save_path,
        encoding=encoding,
        compute=True
    )


def get_path(task_name, path_name):
    """
    获取指定路径下的输出子目录。
    - path_name: 任务名称或路径名称
    - 返回: 输出子目录的完整路径
    """
    output_path = f"../../../output/{task_name}/{path_name}/output"
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