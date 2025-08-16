import numpy as np
import pandas as pd
import os
import re
import rasterio
import tools.config as config


def get_year(path_name,year_interval=config.period):
    # 列出目录中的所有文件
    for file_name in os.listdir(path_name):
        if file_name.startswith("begin_end_compare_"):
            # 使用正则表达式提取年份
            match = re.search(r'(\d{4})_(\d{4})', file_name)
            if match:
                year_start, year_end = map(int, match.groups())
                return list(range(year_start, year_end + 1, year_interval))
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


import os
import uuid
import time
from contextlib import contextmanager
import numpy as np
import rasterio

@contextmanager
def _file_lock(lock_path, timeout=120, poll=0.1):
    """
    简单跨进程锁：基于 os.mkdir 的原子性创建目录充当锁。
    """
    start = time.time()
    while True:
        try:
            os.mkdir(lock_path)   # 创建成功即获得锁
            break
        except FileExistsError:
            if time.time() - start > timeout:
                raise TimeoutError(f"Acquire lock timeout: {lock_path}")
            time.sleep(poll)
    try:
        yield
    finally:
        # 释放锁
        try:
            os.rmdir(lock_path)
        except FileNotFoundError:
            pass

def _safe_remove(path, retries=5, delay=0.2):
    """
    安全删除文件：处理 Windows 上句柄未释放的情况，带重试。
    """
    for i in range(retries):
        try:
            if os.path.exists(path):
                os.remove(path)
            return
        except PermissionError:
            time.sleep(delay * (i + 1))
    # 兜底：改名为陈旧文件，避免阻塞
    try:
        os.replace(path, path + f".stale.{os.getpid()}")
    except Exception:
        pass

def npy_to_map(input_arr, output_tif, proj_file,
               fill_value=np.nan, shift=0,
               dtype=rasterio.float32):
    """
    将一维 .npy 数组铺回到栅格地图中，并安全写出：
    - 加文件锁，防止并行进程同时读/写同一 output_tif
    - 写入临时文件后原子替换，避免“边删边写”引发的权限问题
    """
    if not isinstance(input_arr, str) or not input_arr.lower().endswith(".npy"):
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
        print(f"Warning: {input_arr} 的长度为 {len(lumap)}, "
              f"proj_file 中有效像元数量为 {len(nonzeroes[0])}.")
        raise ValueError("lumap 的长度与 proj_file 中的有效像元数量不一致")

    # 3) 构建全图，并赋值
    themap = np.full(shape, fill_value=fill_value, dtype=float)
    themap[nonzeroes] = lumap + shift
    themap[~np.isfinite(themap)] = np.nan

    # 4) 更新 profile
    profile.update({
        'dtype': dtype,
        'count': 1,
        'compress': 'lzw',
        'nodata': fill_value,
        'transform': transform,
        'crs': crs
    })

    # 5) 安全写出：锁 + 临时文件 + 原子替换
    out_dir = os.path.dirname(os.path.abspath(output_tif))
    os.makedirs(out_dir, exist_ok=True)
    lock_path = output_tif + ".lock"
    tmp_tif = output_tif + f".tmp.{uuid.uuid4().hex}.tif"

    with _file_lock(lock_path, timeout=120, poll=0.1):
        # 删除旧文件（带重试）
        _safe_remove(output_tif)

        # 写到临时文件
        with rasterio.open(tmp_tif, 'w', **profile) as dst:
            dst.write(themap.astype(dtype), 1)

        # 关闭句柄后原子替换
        os.replace(tmp_tif, output_tif)

    return output_tif
