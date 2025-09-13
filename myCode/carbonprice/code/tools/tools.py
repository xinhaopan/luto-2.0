import numpy as np
import pandas as pd
import os
import re
import rasterio
import xarray as xr
import os
import tempfile
import shutil
from filelock import FileLock, Timeout
from typing import Union

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


import os
import shutil
import time
import random
import xarray as xr
from filelock import FileLock, Timeout
import tempfile


def save2nc(
        in_xr,
        save_path: str,
        *,
        engine: str = "h5netcdf",
        allow_overwrite: bool = True,
        compress: bool = True,
        lock_timeout: int = 600,
        lock_path: str | None = None,
        compute_before_write: bool = True,
        max_retries: int = 5,  # 新增：最大重试次数
        retry_delay: float = 1.0  # 新增：初始重试延迟（秒）
):
    # 1. 准备数据和路径（与你之前的代码相同）
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    if isinstance(in_xr, xr.Dataset):
        if len(in_xr.data_vars) != 1:
            raise ValueError(f"输入 Dataset 含 {len(in_xr.data_vars)} 个变量，需单变量。")
        var_name = next(iter(in_xr.data_vars))
        da = in_xr[var_name]
    elif isinstance(in_xr, xr.DataArray):
        da = in_xr
        var_name = da.name or "data"
    else:
        raise TypeError("in_xr 必须是 xarray.DataArray 或 单变量 xarray.Dataset")

    if da.name != var_name:
        da = da.rename(var_name)

    coords_to_drop = set(da.coords) - set(da.dims)
    if coords_to_drop:
        da = da.drop_vars(coords_to_drop, errors="ignore")

    if compute_before_write:
        da = da.load()

    # 2. 设置编码和文件锁（与你之前的代码相同）
    enc = {var_name: {"dtype": "float32"}}
    if compress:
        enc[var_name].update({"zlib": True, "complevel": 4})
    if hasattr(da.data, "chunks") and da.data.chunks is not None:
        enc[var_name]["chunksizes"] = da.chunks

    lockfile = lock_path or (save_path + ".lock")
    lock = FileLock(lockfile)
    save_dir = os.path.dirname(save_path) or '.'

    # 3. 核心逻辑：获取锁并执行带重试的写入/移动操作
    # --------------------------------------------------
    try:
        with lock.acquire(timeout=lock_timeout):
            # 锁内二次检查
            if os.path.exists(save_path) and not allow_overwrite:
                raise FileExistsError(f"目标已存在且不允许覆盖：{save_path}")

            # 使用 tempfile 创建唯一的临时文件
            with tempfile.NamedTemporaryFile(dir=save_dir, suffix='.nc', delete=False) as tmp_file:
                temp_path = tmp_file.name

            try:
                # 步骤 A: 写入临时文件
                da.astype("float32").to_netcdf(path=temp_path, engine=engine, encoding=enc)

                # 步骤 B: 带重试的原子移动
                for attempt in range(max_retries):
                    try:
                        shutil.move(temp_path, save_path)
                        # 成功移动，直接跳出重试循环
                        break
                    except PermissionError:
                        if attempt < max_retries - 1:
                            # 指数退避 + 随机抖动
                            sleep_time = retry_delay * (1.5 ** attempt) + random.uniform(0, 0.5)
                            print(f"⚠️ 移动文件时权限被拒绝，将在 {sleep_time:.2f}s 后重试...")
                            time.sleep(sleep_time)
                        else:
                            # 所有重试失败后，抛出原始异常
                            raise

            except Exception as e:
                # 如果在写入或移动过程中发生任何其他错误，确保清理临时文件
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                raise e

    except Timeout:
        raise TimeoutError(f"获取写锁超时（{lock_timeout}s）：{lockfile}")


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


def filter_all_from_dims(ds: Union[xr.Dataset, xr.DataArray]) -> Union[xr.Dataset, xr.DataArray]:
    """
    从 xarray 对象的所有维度中筛选并移除值为 'ALL' 的坐标。

    该函数会遍历对象的所有维度，检查其坐标是否包含字符串 'ALL'。
    如果包含，则只选择不等于 'ALL' 的部分。

    Args:
        ds (Union[xr.Dataset, xr.DataArray]): 需要进行筛选的 xarray 数据集或数据数组。

    Returns:
        Union[xr.Dataset, xr.DataArray]: 一个新的、经过筛选的 xarray 对象。
    """
    # 将输入对象作为筛选的起点
    filtered_ds = ds

    # 遍历所有维度名称
    for dim_name in ds.dims:
        # 安全检查：确保维度有关联的坐标，并且坐标是字符串类型
        if dim_name in ds.coords and ds[dim_name].dtype.kind in ['U', 'S', 'O']:
            # 检查坐标值中是否含有 'ALL'
            if np.isin(ds[dim_name].values, ['ALL']).any():
                # 使用 .sel() 和布尔索引来选择不等于 'ALL' 的部分
                filtered_ds = filtered_ds.sel({dim_name: filtered_ds[dim_name] != 'ALL'})

    return filtered_ds