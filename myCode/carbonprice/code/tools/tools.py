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


import os
import xarray as xr
from filelock import FileLock, Timeout

def save2nc(
    in_xr,
    save_path: str,
    *,
    engine: str = "h5netcdf",     # netCDF 后端；并行更稳
    allow_overwrite: bool = True, # 已存在时是否允许覆盖
    compress: bool = True,        # 是否压缩(zlib)
    lock_timeout: int = 600,      # 获取锁的最长等待秒数
    lock_path: str | None = None, # 自定义锁文件路径；默认 <save_path>.lock
    compute_before_write: bool = True  # 写前是否先 .load()
):
    """
    并发安全保存到 NetCDF：
      - DataArray/Dataset（仅单变量）都支持；
      - 写前可选 .load()，避免写入阶段触发 dask 计算；
      - 用 filelock 对 <save_path> 加文件锁，避免多进程同时写坏文件；
      - 仅在数据确有分块时写入 chunksizes，防止编码报错。
    """

    # 目录
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # 处理 Dataset / DataArray
    if isinstance(in_xr, xr.Dataset):
        if len(in_xr.data_vars) != 1:
            raise ValueError(f"输入 Dataset 含 {len(in_xr.data_vars)} 个变量，需单变量才可写 NetCDF（或请自行拆分）。")
        var_name = next(iter(in_xr.data_vars))
        da = in_xr[var_name]
    elif isinstance(in_xr, xr.DataArray):
        da = in_xr
        var_name = da.name or "data"
    else:
        raise TypeError("in_xr 必须是 xarray.DataArray 或 单变量 xarray.Dataset")

    # 统一变量名
    if da.name != var_name:
        da = da.rename(var_name)

    # 去掉非维度坐标（更干净，避免写坐标冲突）
    coords_to_drop = set(da.coords) - set(da.dims)
    if coords_to_drop:
        da = da.drop_vars(coords_to_drop, errors="ignore")

    # 写前可选计算（强烈建议 True：更稳定）
    if compute_before_write:
        da = da.load()

    # 编码
    enc = {var_name: {"dtype": "float32"}}
    if compress:
        enc[var_name].update({"zlib": True, "complevel": 4})
    # 只有确实 chunked 才写 chunksizes
    if hasattr(da.data, "chunks") and da.data.chunks is not None:
        enc[var_name]["chunksizes"] = da.chunks

    # 已存在文件的策略
    if os.path.exists(save_path) and not allow_overwrite:
        raise FileExistsError(f"目标已存在且不允许覆盖：{save_path}")

    # 文件锁（避免并发写同一目标）
    lockfile = lock_path or (save_path + ".lock")
    lock = FileLock(lockfile)

    try:
        with lock.acquire(timeout=lock_timeout):
            # 再次二次检查（锁内）是否允许覆盖
            if os.path.exists(save_path) and not allow_overwrite:
                raise FileExistsError(f"目标已存在且不允许覆盖：{save_path}")

            # 写入
            da.astype("float32").to_netcdf(
                path=save_path,
                engine=engine,
                encoding=enc,
                compute=True,
            )
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