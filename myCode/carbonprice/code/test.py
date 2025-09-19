import numpy as np, rasterio
from tools.tools import nc_to_tif
import xarray as xr
import gzip, dill
import tools.config as config
import os

data_path = f"../../../output/{config.TASK_NAME}/Run_01_GHG_high_BIO_high_CUT_50/output/2025_09_08__09_33_05_RF5_2010-2050/Data_RES5.gz"

with gzip.open(data_path, 'rb') as f:
    data = dill.load(f)

input_path = r"N:\LUF-Modelling\LUTO2_XH\LUTO2\output\20250908_Paper2_Results_NCI\carbon_price\0_base_data\carbon_high_bio_50\2050\xr_total_cost_carbon_high_bio_50_amortised_2050.nc"

# 输出路径
out_tif = r"N:\LUF-Modelling\LUTO2_XH\LUTO2\output\20250908_Paper2_Results_NCI\carbon_price\4_tif\xr_total_cost_carbon_high_bio_50_2050.tif"

# 读取和处理
da = xr.open_dataarray(input_path)
da = da.sum(dim=[d for d in da.dims if d != 'cell'])
da = da.where(da >= 1)
print(np.min(da))
da = da / 1e6
if "cell" not in da.dims or len(da.dims) != 1:
        raise ValueError(f"维度是 {da.dims}，只支持一维 'cell'。")
nodata_value = -9999.0
arr1d = da.values.astype(np.float32)
arr1d = np.where(np.isfinite(arr1d), arr1d, nodata_value)

# 铺回 2D（保持你的逻辑）
full_res_raw = (arr1d.size == data.LUMAP_NO_RESFACTOR.size)
if full_res_raw:
    geo_meta = data.GEO_META_FULLRES.copy()
    arr_2d = np.full(data.NLUM_MASK.shape, nodata_value, dtype=np.float32)
    np.place(arr_2d, data.NLUM_MASK, arr1d)
else:
    geo_meta = data.GEO_META.copy()
    arr_2d = data.LUMAP_2D_RESFACTORED.copy().astype(np.float32)
    np.place(arr_2d,
             (arr_2d != data.MASK_LU_CODE) & (arr_2d != data.NODATA),
             arr1d)

# arr_2d[arr_2d == data.MASK_LU_CODE] = nodata_value
# 生成有效性掩膜：True=有效，False=无效
valid_mask = np.isfinite(arr_2d)  # 先按是否为有限数
# 把 NaN/inf 替换为 nodata 值
arr_2d = np.where(valid_mask, arr_2d, nodata_value).astype(np.float32)
arr_2d[arr_2d == data.MASK_LU_CODE] = nodata_value

meta = geo_meta.copy()
meta.update(
    count=1,
    dtype="float32",
    nodata=nodata_value,
    compress="deflate",   # 可选：压缩
    predictor=3,          # 浮点预测器
    tiled=True,           # 平铺
    blockxsize=256,       # 块大小（可按需）
    blockysize=256,
)
tif_path = out_tif
os.makedirs(os.path.dirname(tif_path), exist_ok=True)
with rasterio.open(tif_path, "w", **meta) as dst:
    dst.write(arr_2d, 1)
    # 写 GDAL 内部掩膜（0=无效, 255=有效）
    dst.write_mask((valid_mask.astype(np.uint8) * 255))

print(f"✅ 已保存: {tif_path}")

with rasterio.open(out_tif) as src:
    bounds = src.bounds
    arr = src.read(1, masked=True)
    extent = (bounds.left, bounds.right, bounds.bottom, bounds.top)
    raster_crs = src.crs
    nodata = src.nodata
    if nodata is not None:
        arr = np.ma.masked_equal(arr, nodata)  # -9999 -> mask=True

np.nanmin(arr), np.min(arr)
