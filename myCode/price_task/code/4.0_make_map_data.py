import os
import pandas as pd
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
import tools.config as config


def clip_tif_values_with_mask(
    src_path,
    dst_path,
    vmin=None,
    vmax=None,
    mask_path=None,          # 可选：掩膜 tif 路径
    mask_threshold=1,
    compress="deflate",
    cost=False
):
    """
    读取 src_path，执行：
      1) 如果提供 mask_path，则 mask < mask_threshold 的位置置为 NoData
      2) 对有效像元做区间裁剪到 [vmin, vmax]
      3) 可选将结果除以 1e6（cost=True）
    最后保留原始空间参考，写到 dst_path。

    - mask_path=None 时，不使用掩膜
    - 自动重投影掩膜到源栅格
    - 支持多波段
    """

    if vmin is None and vmax is None and mask_path is None:
        raise ValueError("至少需要提供 vmin/vmax 中的一个，或提供 mask_path。")

    # 打开源数据
    with rasterio.open(src_path) as src:
        profile = src.profile.copy()
        data = src.read()
        src_nodata = src.nodata

        # 源有效像元布尔掩膜
        if src_nodata is not None:
            valid_mask_src = (data != src_nodata)
        else:
            valid_mask_src = np.ones_like(data, dtype=bool)
            src_nodata = -9999
            profile.update(nodata=src_nodata)

        # 如果提供 mask_path
        if mask_path is not None and os.path.exists(mask_path):
            with rasterio.open(mask_path) as msk_ds:
                mask_src = msk_ds.read(1)
                mask_reproj = np.empty((src.height, src.width), dtype=np.float32)
                reproject(
                    source=mask_src,
                    destination=mask_reproj,
                    src_transform=msk_ds.transform,
                    src_crs=msk_ds.crs,
                    dst_transform=src.transform,
                    dst_crs=src.crs,
                    resampling=Resampling.nearest
                )
            # 扩展到多波段形状，并结合源有效性
            valid_mask = valid_mask_src & (mask_reproj >= mask_threshold)[None, :, :]
        else:
            # 不使用 mask
            valid_mask = valid_mask_src

        # 数据副本
        out = data.astype(np.float64, copy=True)

        # 无效位置置为 nodata
        out[~valid_mask] = src_nodata

        # 裁剪区间
        if vmin is not None:
            sel = valid_mask & (out < vmin)
            out[sel] = vmin
        if vmax is not None:
            sel = valid_mask & (out > vmax)
            out[sel] = vmax

        # cost 转换
        if cost:
            out = (out / 1e6).astype(np.float32, copy=False)
            profile.update(dtype="float32")
        else:
            out = out.astype(data.dtype, copy=False)

        # 保证无效位置是 nodata
        out[~valid_mask] = src_nodata

        # 压缩
        if compress is not None:
            profile.update(compress=compress)

        # 写出
        with rasterio.open(dst_path, "w", **profile) as dst:
            dst.write(out)

def analyze_masked_arrays_by_path(years, arr_path, arr1_name, arr2_name, percentiles=[99, 98, 97, 96, 95]):
    stats = []
    arr_dict = {}

    for year in years:
        arr1_path = os.path.join(arr_path, f"{arr1_name}_{year}.npy")
        arr2_path = os.path.join(arr_path, f"{arr2_name}_{year}.npy")

        arr1 = np.load(arr1_path)
        arr2 = np.load(arr2_path)

        mask = arr2 > 1
        masked_arr1 = np.where(mask, arr1, np.nan)
        arr_dict[year] = masked_arr1

        year_stats = {"Year": year}
        for p in percentiles:
            year_stats[f"{p}%"] = np.nanpercentile(masked_arr1, p)

        stats.append(year_stats)

    return pd.DataFrame(stats).set_index("Year"), arr_dict



arr_path = f"{config.TASK_DIR}/carbon_price/data"
percentile = list(range(90, 100))

df_ghg, _ = analyze_masked_arrays_by_path([2050], arr_path, "ghg", "ghg", percentile)
max_ghg = float(np.nanmedian(df_ghg.loc[2050].values))
src_path = f"{config.TASK_DIR}/carbon_price/data/ghg_2050.tif"
dst_path = f"{config.TASK_DIR}/carbon_price/map_data/ghg_2050.tif"
clip_tif_values_with_mask(src_path, dst_path, vmin=1, vmax=max_ghg,mask_path=f"{arr_path}/ghg_2050.tif", mask_threshold=1)

df_bio, _ = analyze_masked_arrays_by_path([2050], arr_path, "bio", "bio", percentile)
max_bio = float(np.nanmedian(df_bio.loc[2050].values))
src_path = f"{config.TASK_DIR}/carbon_price/data/bio_2050.tif"
dst_path = f"{config.TASK_DIR}/carbon_price/map_data/bio_2050.tif"
clip_tif_values_with_mask(src_path, dst_path, vmin=1, vmax=max_bio, mask_path=f"{arr_path}/bio_2050.tif", mask_threshold=1)

df_cp, cp_arrs = analyze_masked_arrays_by_path([2050], arr_path, "carbon_price", "ghg", percentile)
max_cp = float(np.nanmedian(df_cp.loc[2050].values))
src_path = f"{config.TASK_DIR}/carbon_price/data/carbon_price_2050.tif"
dst_path = f"{config.TASK_DIR}/carbon_price/map_data/carbon_price_2050.tif"
clip_tif_values_with_mask(src_path, dst_path, vmin=1, vmax=max_cp, mask_path=f"{arr_path}/ghg_2050.tif", mask_threshold=1)

df_bio, bp_arrs = analyze_masked_arrays_by_path([2050], arr_path, "bio_price", "bio", percentile)
max_bp = float(np.nanmedian(df_bio.loc[2050].values))
src_path = f"{config.TASK_DIR}/carbon_price/data/bio_price_2050.tif"
dst_path = f"{config.TASK_DIR}/carbon_price/map_data/bio_price_2050.tif"
clip_tif_values_with_mask(src_path, dst_path, vmin=1, vmax=max_bp, mask_path=f"{arr_path}/bio_2050.tif", mask_threshold=1)

df_ghg_cost, ghg_arrs = analyze_masked_arrays_by_path([2050], arr_path, "carbon_cost", "ghg", percentile)
max_ghg_cost = float(np.nanmedian(df_ghg_cost.loc[2050].values))
src_path = f"{config.TASK_DIR}/carbon_price/data/carbon_cost_2050.tif"
dst_path = f"{config.TASK_DIR}/carbon_price/map_data/carbon_cost_2050.tif"
clip_tif_values_with_mask(src_path, dst_path, vmin=1, vmax=max_ghg_cost,mask_path=f"{arr_path}/ghg_2050.tif", mask_threshold=1)

df_bio_cost, bio_arrs = analyze_masked_arrays_by_path([2050], arr_path, "bio_cost", "bio", percentile)
max_bio_cost = float(np.nanmedian(df_bio_cost.loc[2050].values))
src_path = f"{config.TASK_DIR}/carbon_price/data/bio_cost_2050.tif"
dst_path = f"{config.TASK_DIR}/carbon_price/map_data/bio_cost_2050.tif"
clip_tif_values_with_mask(src_path, dst_path, vmin=1, vmax=max_bio_cost, mask_path=f"{arr_path}/bio_2050.tif", mask_threshold=1)

