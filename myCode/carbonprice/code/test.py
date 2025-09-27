import os
import numpy as np
import rasterio
import tools.config as config


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

tif_dir = f"../../../output/{config.TASK_NAME}/carbon_price/4_tif"
tif_path_1 = os.path.join(tif_dir, 'carbon_high', "xr_carbon_price_carbon_high_2050.tif")
tif_path_2 = os.path.join(tif_dir, 'Counterfactual_carbon_high_bio_50',
                          f"xr_carbon_price_Counterfactual_carbon_high_bio_50_2050.tif")
tif_output = os.path.join(tif_dir, 'carbon_high_bio_50', f"xr_carbon_price_carbon_high_bio_50_2050.tif")
subtract_tifs(tif_path_2, tif_path_1, tif_output)