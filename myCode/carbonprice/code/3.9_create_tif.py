import os
import numpy as np
import xarray as xr
import rasterio
from tools.tools import (
    get_path, get_year, save2nc, filter_all_from_dims, nc_to_tif, get_data_RES_path
)
from joblib import Parallel, delayed
import tools.config as config
import time
import gzip
import dill


def xarrays_to_tifs(env_cat, file_part, base_dir, tif_dir, data):
    """处理一个类别+文件部分，并输出tif"""
    print(f"Processing {env_cat} - {file_part}")

    # 构造输入路径
    if file_part == 'total_cost':
        input_path = f"{base_dir}/{env_cat}/2050/xr_{file_part}_{env_cat}_amortised_2050.nc"
    else:
        input_path = f"{base_dir}/{env_cat}/2050/xr_{file_part}_{env_cat}_2050.nc"

    # 读取和处理
    da = xr.open_dataarray(input_path)
    da = da.sum(dim=[d for d in da.dims if d != 'cell'])
    da = da.where(da >= 1)

    # 输出路径
    out_tif = f"{tif_dir}/{env_cat}/xr_{file_part}_{env_cat}_cell_2050.tif"
    os.makedirs(os.path.dirname(out_tif), exist_ok=True)
    nc_to_tif(data, da, out_tif)

    out_tif = f"{tif_dir}/{env_cat}/xr_{file_part}_{env_cat}_area_2050.tif"
    da = da / data.REAL_AREA
    nc_to_tif(data, da, out_tif)

    return out_tif

def subtract_tifs(a_path, b_path, out_path):
    with rasterio.open(a_path) as A, rasterio.open(b_path) as B:
        # 1) 基本一致性检查
        if (A.width, A.height) != (B.width, B.height) or A.transform != B.transform or A.crs != B.crs:
            raise ValueError("输入影像的大小/transform/CRS 不一致，请先重采样/重投影对齐。")

        # 2) 读为 masked array（会自动将 nodata 屏蔽），再转为含 NaN 的数组
        arr_a = A.read(1, masked=True).filled(np.nan).astype(np.float32)
        arr_b = B.read(1, masked=True).filled(np.nan).astype(np.float32)

        arr_a[arr_a < 0] = np.nan
        arr_b[arr_b < 0] = np.nan

        # 3) 相减
        out = arr_a - arr_b

        # 4) 将 <0 的结果置为 NaN（其他位置原本的 NaN 将自动保留）
        out[out <= 0] = np.nan

        # 5) 写出（Float32 + LZW 压缩；nodata 设为 NaN）
        nodata_value = -9999
        profile = A.profile.copy()
        profile.update(dtype="float32", compress="lzw", nodata=nodata_value)
        out = np.where(np.isnan(out), nodata_value, out)

        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(out, 1)



task_name = config.TASK_NAME
input_files_0 = config.input_files_0
output_all_names = config.carbon_names + config.carbon_bio_names + config.counter_carbon_bio_names
tif_dir = f"../../../output/{config.TASK_NAME}/carbon_price/4_tif"
output_path = f"../../../output/{config.TASK_NAME}/carbon_price/0_base_data"
data_path = get_data_RES_path(f"../../../output/{config.TASK_NAME}/{input_files_0[0]}/output")

njobs = 41

with gzip.open(data_path, 'rb') as f:
    data = dill.load(f)

file_parts = ['total_cost', 'cost_ag', 'cost_agricultural_management', 'cost_non_ag', 'cost_transition_ag2ag_diff',
                  'transition_cost_ag2non_ag_amortised_diff', 'total_carbon', 'total_bio', 'bio_price', 'carbon_price']

tasks = [(env_cat, file_part) for env_cat in output_all_names for file_part in file_parts]

results = Parallel(n_jobs=njobs)(  # 这里你可以改 n_jobs，比如 8 或 -1 用所有CPU
    delayed(xarrays_to_tifs)(env_cat, file_part, output_path, tif_dir, data)
    for env_cat, file_part in tasks
)

tif_path_1 = os.path.join(tif_dir, 'carbon_high', "xr_carbon_price_carbon_high_2050.tif")
tif_path_2 = os.path.join(tif_dir, 'Counterfactual_carbon_high_bio_50',
                          f"xr_carbon_price_Counterfactual_carbon_high_bio_50_2050.tif")
tif_output = os.path.join(tif_dir, 'carbon_high_bio_50', f"xr_carbon_price_carbon_high_bio_50_2050.tif")
subtract_tifs(tif_path_2, tif_path_1, tif_output)