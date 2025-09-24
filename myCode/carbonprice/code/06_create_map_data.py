import os

import tools.config as config
from tools.tools import nc_to_tif
import xarray as xr
import dill
import gzip
from joblib import Parallel, delayed

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

base_dir = f"../../../output/{config.TASK_NAME}/carbon_price/0_base_data"
tif_dir = f"../../../output/{config.TASK_NAME}/carbon_price/4_tif"
data_path = f"../../../output/{config.TASK_NAME}/Run_01_GHG_high_BIO_high_CUT_50/output/2025_09_22__04_48_18_RF5_2010-2050/Data_RES5.gz"

with gzip.open(data_path, 'rb') as f:
    data = dill.load(f)

# env_categorys =  ['carbon_high', 'carbon_high_bio_50','carbon_low',
#         'carbon_low_bio_10', 'carbon_low_bio_20', 'carbon_low_bio_30', 'carbon_low_bio_40', 'carbon_low_bio_50',
#         'carbon_high_bio_10', 'carbon_high_bio_20', 'carbon_high_bio_30', 'carbon_high_bio_40']

env_categorys =  ['carbon_high_bio_50','carbon_high','Counterfactual_carbon_high_bio_50']

file_parts = ['total_cost','cost_ag', 'cost_agricultural_management', 'cost_non_ag', 'cost_transition_ag2ag_diff',
                 'transition_cost_ag2non_ag_amortised_diff','total_carbon','total_bio','bio_price','carbon_price']

tasks = [(env_cat, file_part) for env_cat in env_categorys for file_part in file_parts]

results = Parallel(n_jobs=8)(   # 这里你可以改 n_jobs，比如 8 或 -1 用所有CPU
    delayed(process_one)(env_cat, file_part, base_dir, tif_dir, data)
    for env_cat, file_part in tasks
)