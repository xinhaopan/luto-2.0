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


def xarrays_to_tifs(env_cat, file_part, base_dir, tif_dir, data, remove_negative=True, per_ha=True):
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

    if per_ha:
        da = da / data.REAL_AREA
    if remove_negative:
        da = da.where(da >= 0, np.nan)

    # 输出 cell 版本
    out_tif = f"{tif_dir}/{env_cat}/xr_{file_part}_{env_cat}_2050.tif"
    os.makedirs(os.path.dirname(out_tif), exist_ok=True)
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

def xarrays_to_tifs_by_type(
    env_cat,
    file_part,
    base_dir,
    tif_dir,
    data,
    sum_dim,                # 你要合并的第二个维度（比如 "year"）
    remove_negative=False,
    per_ha=False
):
    """
    按照 'cell' 和 sum_dim 对数据求和，分别输出每个 sum_dim 坐标的 tif 文件，
    并先输出总和版本（对除了cell的所有维度求和）

    Parameters
    ----------
    env_cat: str
    file_part: str
    base_dir: str
    tif_dir: str
    data: object
    sum_dim: str   # 你要分组的维度，比如 'year'
    remove_negative: bool
    per_ha: bool
    """
    print(f"Processing {env_cat} - {file_part} by {sum_dim}")

    # 构造输入路径
    input_path = f"{base_dir}/{env_cat}/2050/xr_{file_part}_2050.nc"

    # 读取和处理
    da = xr.open_dataarray(input_path)

    # ----------- 1. 求总和版本（除了cell的所有维度都求和） -----------
    sum_dims_total = [d for d in da.dims if d != 'cell']
    da_total = da.sum(dim=sum_dims_total)

    if per_ha:
        da_total = da_total / data.REAL_AREA
    if remove_negative:
        da_total = da_total.where(da_total >= 0, np.nan)

    out_total_tif = f"{tif_dir}/{env_cat}/xr_total_{file_part}_{env_cat}_2050.tif"
    os.makedirs(os.path.dirname(out_total_tif), exist_ok=True)
    nc_to_tif(data, da_total, out_total_tif)
    print(f"Saved {out_total_tif}")

    # ----------- 2. 按 sum_dim 输出分组 tif -----------
    if sum_dim not in da.dims:
        raise ValueError(f"{sum_dim} 不在数据的维度 {da.dims} 中！")

    results = [out_total_tif]
    # 遍历新维度的所有坐标
    for coord_val in da[sum_dim].values:
        da_slice = da.sel({sum_dim: coord_val})
        sum_dims = [d for d in da_slice.dims if d != 'cell']
        da_out = da_slice.sum(dim=sum_dims)

        if per_ha:
            da_out = da_out / data.REAL_AREA
        if remove_negative:
            da_out = da_out.where(da_out >= 0, np.nan)

        out_tif = f"{tif_dir}/{env_cat}/xr_{file_part}_{env_cat}_{coord_val}_2050.tif"
        os.makedirs(os.path.dirname(out_tif), exist_ok=True)
        nc_to_tif(data, da_out, out_tif)
        results.append(out_tif)
        print(f"Saved {out_tif}")

    return results


task_name = config.TASK_NAME
input_files_0 = config.input_files_0
# output_all_names = config.carbon_names + config.carbon_bio_names + config.counter_carbon_bio_names
output_all_names = ['carbon_high', 'carbon_high_bio_50', 'Counterfactual_carbon_high_bio_50']
tif_dir = f"../../../output/{config.TASK_NAME}/carbon_price/4_tif"
output_path = f"../../../output/{config.TASK_NAME}/carbon_price/0_base_data"
data_path = get_data_RES_path(f"../../../output/{config.TASK_NAME}/{input_files_0[0]}/output")

njobs = 41

with gzip.open(data_path, 'rb') as f:
    data = dill.load(f)

cost_file_parts = ['total_cost',  'cost_agricultural_management', 'cost_non_ag','transition_cost_ag2non_ag_amortised_diff']
tasks = [(env_cat, file_part) for env_cat in output_all_names for file_part in cost_file_parts]
results = Parallel(n_jobs=njobs)(  # 这里你可以改 n_jobs，比如 8 或 -1 用所有CPU
    delayed(xarrays_to_tifs)(env_cat, file_part, output_path, tif_dir, data)
    for env_cat, file_part in tasks
)

GHG_file_parts = ['total_carbon', 'GHG_ag_management', 'GHG_non_ag']
tasks = [(env_cat, file_part) for env_cat in output_all_names for file_part in GHG_file_parts]
results = Parallel(n_jobs=njobs)(  # 这里你可以改 n_jobs，比如 8 或 -1 用所有CPU
    delayed(xarrays_to_tifs)(env_cat, file_part, output_path, tif_dir, data, remove_negative=False)
    for env_cat, file_part in tasks
)

bio_file_parts = ['total_bio','biodiversity_GBF2_priority_ag_management','biodiversity_GBF2_priority_non_ag']
tasks = [(env_cat, file_part) for env_cat in output_all_names for file_part in bio_file_parts]
results = Parallel(n_jobs=njobs)(  # 这里你可以改 n_jobs，比如 8 或 -1 用所有CPU
    delayed(xarrays_to_tifs)(env_cat, file_part, output_path, tif_dir, data, remove_negative=False, per_ha=False)
    for env_cat, file_part in tasks
)

price_file_parts = ['carbon_price','bio_price']
tasks = [(env_cat, file_part) for env_cat in output_all_names for file_part in price_file_parts]
results = Parallel(n_jobs=njobs)(  # 这里你可以改 n_jobs，比如 8 或 -1 用所有CPU
    delayed(xarrays_to_tifs)(env_cat, file_part, output_path, tif_dir, data, per_ha=False)
    for env_cat, file_part in tasks
)
tif_path_1 = os.path.join(tif_dir, 'carbon_high', "xr_carbon_price_carbon_high_2050.tif")
tif_path_2 = os.path.join(tif_dir, 'Counterfactual_carbon_high_bio_50',
                          f"xr_carbon_price_Counterfactual_carbon_high_bio_50_2050.tif")
tif_output = os.path.join(tif_dir, 'carbon_high_bio_50', f"xr_carbon_price_carbon_high_bio_50_2050.tif")
subtract_tifs(tif_path_2, tif_path_1, tif_output)

input_files = ['Run_01_GHG_high_BIO_high_CUT_50','Run_06_GHG_high_BIO_off_CUT_50','Run_18_GHG_off_BIO_off_CUT_50']
agmgt_file_parts = ['area_agricultural_management']

tasks = [(env_cat, file_part) for env_cat in input_files for file_part in agmgt_file_parts]
results = Parallel(n_jobs=njobs)(  # 这里你可以改 n_jobs，比如 8 或 -1 用所有CPU
    delayed(xarrays_to_tifs_by_type)(
        env_cat, file_part, output_path, tif_dir, data,
        sum_dim='am'
    )
    for env_cat, file_part in tasks
)

nonag = ['area_non_agricultural_landuse']
tasks = [(env_cat, file_part) for env_cat in input_files for file_part in nonag]
results = Parallel(n_jobs=njobs)(  # 这里你可以改 n_jobs，比如 8 或 -1 用所有CPU
    delayed(xarrays_to_tifs_by_type)(
        env_cat, file_part, output_path, tif_dir, data,
        sum_dim='lu'
    )
    for env_cat, file_part in tasks
)