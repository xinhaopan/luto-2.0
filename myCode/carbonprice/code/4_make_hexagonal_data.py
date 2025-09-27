import tools.config as config
from tools.tools import get_path

import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from joblib import Parallel, delayed
from rasterio.features import geometry_mask
import numpy as np
import geopandas as gpd
import pandas as pd
import rasterio
from rasterio.features import rasterize

def process_env_cat(env_cat, shp_name, file_parts, tif_dir):
    for file_part in file_parts:
        tif_env_dir = os.path.join(tif_dir, env_cat)
        input_tif_name = f'xr_{file_part}_{env_cat}_2050.tif'
        out_shp = os.path.join(tif_env_dir, f'{shp_name}',f'{shp_name}_{file_part}_{env_cat}_2050.shp')
        os.makedirs(os.path.dirname(out_shp), exist_ok=True)
        zonal_stats_rasterized(tif_env_dir, input_tif_name, shp_path, out_shp)

import os
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.features import rasterize

def zonal_stats_rasterized(input_tif_dir, input_tif_name, shp_path, out_shp,
                           extra_nodata_vals=(-9999.0,), drop_allnan=True):
    # 1) 读 shp 与 tif
    gdf = gpd.read_file(shp_path)
    input_tif = os.path.join(input_tif_dir, input_tif_name)

    with rasterio.open(input_tif) as src:
        img_m = src.read(1, masked=True)   # MaskedArray（若 nodata 未设置，mask 可能无效）
        transform = src.transform
        shape = (src.height, src.width)
        if gdf.crs is not None and src.crs is not None and gdf.crs != src.crs:
            gdf = gdf.to_crs(src.crs)

    n_shapes = len(gdf)

    # 2) 将掩膜与哨兵值统一转为 NaN
    arr = img_m.filled(np.nan).astype('float64', copy=False)
    for nd in (extra_nodata_vals or ()):
        arr[np.isclose(arr, nd)] = np.nan

    # 3) 栅格化矢量：像元值=1..n_shapes；0 为背景
    shapes = ((geom, i + 1) for i, geom in enumerate(gdf.geometry))
    id_arr = rasterize(shapes, out_shape=shape, transform=transform, fill=0, dtype="int32")

    # 4) 只统计有效像元（区域内 且 非 NaN）
    valid_mask = (id_arr > 0) & np.isfinite(arr)
    if not np.any(valid_mask):
        if drop_allnan:
            print("⚠️ 所有多边形均无有效像元，未输出。")
            return
        # 不删除则写出全 NaN 的结果
        gdf["sum"] = np.nan
        gdf["mean"] = np.nan
        gdf.to_file(out_shp)
        print(f"✅ Saved {out_shp} (all NaN)")
        return

    vals = arr[valid_mask]
    ids  = id_arr[valid_mask]

    # 5) 分组聚合
    sum_per_id = np.bincount(ids, weights=vals, minlength=n_shapes + 1)
    cnt_per_id = np.bincount(ids, minlength=n_shapes + 1)

    sum_stat = sum_per_id[1:]
    cnt_stat = cnt_per_id[1:]

    mean_stat = np.full_like(sum_stat, np.nan, dtype="float64")
    np.divide(sum_stat, cnt_stat, out=mean_stat, where=cnt_stat > 0)

    if 'total_carbon' in input_tif_name:
        sum_stat = sum_stat / 1e6
        mean_stat = mean_stat / 1e6

    # 6) 赋值到 gdf
    gdf["sum"]  = sum_stat
    gdf["mean"] = mean_stat
    gdf["count"] = cnt_stat  # 方便筛选

    # 7) （新增）删除全 NaN（即 count==0）的要素
    if drop_allnan:
        before = len(gdf)
        gdf = gdf[gdf["count"] > 0].copy()
        removed = before - len(gdf)
        print(f"🧹 移除了 {removed} 个全 NaN 的多边形。")

        if gdf.empty:
            print("⚠️ 过滤后无要素，未输出。")
            return

    # 可选：不想保留 count 字段就注释掉下一行
    # gdf = gdf.drop(columns=["count"])

    # 8) 输出
    gdf.to_file(out_shp)
    print(f"✅ Saved {out_shp}（共 {len(gdf)} 个要素）")


tif_dir = f"../../../output/{config.TASK_NAME}/carbon_price/4_tif"
carbon_names = config.carbon_names
carbon_bio_names = config.carbon_bio_names
counter_carbon_bio_names = config.counter_carbon_bio_names
output_all_names = carbon_names + carbon_bio_names + counter_carbon_bio_names
file_parts = ['total_cost', 'cost_ag', 'cost_agricultural_management', 'cost_non_ag', 'cost_transition_ag2ag_diff',
                  'transition_cost_ag2non_ag_amortised_diff', 'total_carbon', 'total_bio', 'bio_price', 'carbon_price']

# shp_names = ['H_1kkm2','H_2kkm2','H_5kkm2','H_100km2']
shp_names = ['H_5kkm2']

for shp_name in shp_names:
    shp_path = f"../Map/{shp_name}.shp"
    Parallel(n_jobs=len(output_all_names))(
        delayed(process_env_cat)(env_cat, shp_name, file_parts, tif_dir)
        for env_cat in output_all_names
    )
# for shp_name in shp_names:
#     shp_path = f"../Map/{shp_name}.shp"
#     for env_cat in output_all_names:
#         for file_part in file_parts:
#             tif_env_dir = os.path.join(tif_dir, env_cat)
#             input_tif_name = f'xr_{file_part}_{env_cat}_2050.tif'
#             out_shp = os.path.join(tif_env_dir, f'{shp_name}',f'{shp_name}_{file_part}_{env_cat}_2050.shp')
#             os.makedirs(os.path.dirname(out_shp), exist_ok=True)
#             zonal_stats_rasterized(tif_env_dir, input_tif_name, shp_path, out_shp)