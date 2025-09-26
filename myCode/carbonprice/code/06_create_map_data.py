import os

import tools.config as config
from tools.tools import nc_to_tif
import xarray as xr
import dill
import gzip
from joblib import Parallel, delayed
import rasterio
import numpy as np

def process_one(env_cat, file_part, base_dir, tif_dir, data):
    """处理一个类别+文件部分，并输出tif"""
    print(f"Processing {env_cat} - {file_part}")

    # 构造输入路径
    if file_part == 'total_cost':
        input_path = f"{base_dir}/{env_cat}/2050/xr_{file_part}_{env_cat}_amortised_2050.nc"
    else:
        input_path = f"{base_dir}/{env_cat}/2050/xr_{file_part}_{env_cat}_2050.nc"

    # 输出路径
    out_tif = f"{tif_dir}/{env_cat}/xr_{file_part}_{env_cat}_2050.tif"
    os.makedirs(os.path.dirname(out_tif), exist_ok=True)
    # 读取和处理
    da = xr.open_dataarray(input_path)
    da = da.sum(dim=[d for d in da.dims if d != 'cell'])
    da = da.where(da >= 1)
    if 'cost' in  file_part:
        da = da / 1e6
    # 保存到 GeoTIFF
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

        # 3) 相减
        out = arr_a - arr_b

        # 4) 将 <0 的结果置为 NaN（其他位置原本的 NaN 将自动保留）
        out[out < 0] = np.nan

        # 5) 写出（Float32 + LZW 压缩；nodata 设为 NaN）
        profile = A.profile.copy()
        profile.update(dtype="float32", compress="lzw", nodata=np.nan)

        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(out, 1)

base_dir = f"../../../output/{config.TASK_NAME}/carbon_price/0_base_data"
tif_dir = f"../../../output/{config.TASK_NAME}/carbon_price/4_tif"
data_path = f"../../../output/{config.TASK_NAME}/Run_01_GHG_high_BIO_high_CUT_50/output/2025_09_22__04_48_18_RF5_2010-2050/Data_RES5.gz"



# env_categorys =  ['carbon_high', 'carbon_high_bio_50','carbon_low',
#         'carbon_low_bio_10', 'carbon_low_bio_20', 'carbon_low_bio_30', 'carbon_low_bio_40', 'carbon_low_bio_50',
#         'carbon_high_bio_10', 'carbon_high_bio_20', 'carbon_high_bio_30', 'carbon_high_bio_40']

env_categorys =  ['carbon_high_bio_50','carbon_high','Counterfactual_carbon_high_bio_50']

file_parts = ['total_cost','cost_ag', 'cost_agricultural_management', 'cost_non_ag', 'cost_transition_ag2ag_diff',
                 'transition_cost_ag2non_ag_amortised_diff','total_carbon','total_bio','bio_price','carbon_price']

tasks = [(env_cat, file_part) for env_cat in env_categorys for file_part in file_parts]

with gzip.open(data_path, 'rb') as f:
    data = dill.load(f)
results = Parallel(n_jobs=8)(   # 这里你可以改 n_jobs，比如 8 或 -1 用所有CPU
    delayed(process_one)(env_cat, file_part, base_dir, tif_dir, data)
    for env_cat, file_part in tasks
)

tif_path_1 = os.path.join(tif_dir, env_categorys[1],f"xr_carbon_price_{env_categorys[1]}_2050.tif")
tif_path_2 = os.path.join(tif_dir, env_categorys[2],f"xr_carbon_price_{env_categorys[2]}_2050.tif")
tif_output =  os.path.join(tif_dir, env_categorys[0],f"xr_carbon_price_{env_categorys[0]}_2050.tif")
subtract_tifs(tif_path_2,tif_path_1,tif_output)