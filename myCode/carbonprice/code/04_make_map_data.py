import xarray as xr
import tools.config as config
import rasterio
import numpy as np

def create_map_data(base_dir, env_category, env_name, mask=None, q=None):
    # 打开数据
    xr_data = xr.open_dataset(f"{base_dir}/{env_category}/2050/{env_name}.nc")['data']
    da_result = xr_data.sum(dim=[d for d in xr_data.dims if d != "cell"], skipna=True)

    # 应用分位数裁剪
    if q is not None:
        q_value = da_result.quantile(q, dim="cell", skipna=True).item()
        da_result = da_result.where(da_result <= q_value, q_value)

    # 应用 mask
    if mask is not None:
        xr_mask_datas = xr.open_dataset(f"{base_dir}/Results/xr_{mask}_masks_scenario_cell.nc")
        xr_mask_data = xr_mask_datas['mask'].sel(env_category=env_category, year=2050)
        da_result = da_result.where(xr_mask_data == 1)   # 注意这里用 da_result，而不是 xr_data['data']

    da_result = da_result.where(da_result > 1)
    return da_result




def xr_to_geotif(input_xr, output_dir, output_name,
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

    # 1) 读取参考栅格
    proj_file = f"{output_dir}/ammap_2050.tiff"
    with rasterio.open(proj_file) as src:
        mask2D = src.read(1) >= 0
        transform = src.transform
        crs = src.crs
        profile = src.profile.copy()
        shape = src.shape

    # 2) 加载一维数组
    nonzeroes = np.where(mask2D)
    input_arr = input_xr.to_numpy()

    if input_arr.ndim != 1:
        raise ValueError(f"{input_arr} 中的数组不是一维的")
    if len(input_arr) != len(nonzeroes[0]):
        print(f"Warning: {input_arr} 的长度为 {len(input_arr)}, proj_file 中有效像元数量为 {len(nonzeroes[0])}.")
        raise ValueError("lumap 的长度与 proj_file 中的有效像元数量不一致")

    # 3) 构建全图，并赋值
    themap = np.full(shape, fill_value=fill_value, dtype=float)
    themap[nonzeroes] = input_arr + shift

    # 4) 把 +/- inf 都变成 np.nan
    themap[~np.isfinite(themap)] = np.nan

    # 5) 更新 profile 并写出
    profile.update({
        'dtype': dtype,
        'count': 1,
        'compress': 'lzw',
        'nodata': fill_value
    })
    output_tif = f"{output_dir}/{output_name}.tif"
    with rasterio.open(output_tif, 'w', **profile) as dst:
        dst.write(themap.astype(dtype), 1)

    return output_tif

base_dir = f"../../../output/{config.TASK_NAME}/carbon_price/0_base_data"
output_tif_dir = f"{base_dir}/map_data"

env_category = 'carbon_100'
mask = 'carbon'

env_name = f"xr_total_cost_{env_category}_amortised_2050"
xr_carbon_cost_map = create_map_data(base_dir, env_category, env_name,mask=mask) / 1e6
print(xr_carbon_cost_map.max(skipna=True).item(),xr_carbon_cost_map.min(skipna=True).item())
xr_to_geotif(xr_carbon_cost_map, output_tif_dir, env_name)

env_name = f"xr_total_{env_category}_2050"
xr_carbon_cost_map = create_map_data(base_dir, env_category, env_name,mask=mask)
print(xr_carbon_cost_map.max(skipna=True).item(),xr_carbon_cost_map.min(skipna=True).item())
output_tif_dir = f"{base_dir}/map_data"
xr_to_geotif(xr_carbon_cost_map, output_tif_dir, env_name)

env_names = [ 'xr_cost_ag','xr_cost_agricultural_management','xr_cost_non_ag','xr_cost_transition_ag2ag_diff','xr_transition_cost_ag2non_ag_amortised_diff']
for env_name in env_names:
    env_name_full = f"{env_name}_{env_category}_2050"
    mask = 'carbon'
    xr_carbon_cost_map = create_map_data(base_dir, env_category,env_name_full, mask) / 1e6
    print(env_name, xr_carbon_cost_map.max(skipna=True).item(),xr_carbon_cost_map.min(skipna=True).item())
    output_tif_dir = f"{base_dir}/map_data"
    xr_to_geotif(xr_carbon_cost_map, output_tif_dir, env_name_full)


env_category = 'carbon_100_bio_50'
env_name = f"xr_total_cost_{env_category}_amortised_2050"
mask = 'bio'
xr_carbon_cost_map = create_map_data(base_dir, env_category, env_name,mask=mask) / 1e6
print(xr_carbon_cost_map.max(skipna=True).item(),xr_carbon_cost_map.min(skipna=True).item())
output_tif_dir = f"{base_dir}/map_data"
xr_to_geotif(xr_carbon_cost_map, output_tif_dir, env_name)

env_name = f"xr_total_{env_category}_2050"
xr_carbon_cost_map = create_map_data(base_dir, env_category, env_name,mask=mask)
print(xr_carbon_cost_map.max(skipna=True).item(),xr_carbon_cost_map.min(skipna=True).item())
output_tif_dir = f"{base_dir}/map_data"
xr_to_geotif(xr_carbon_cost_map, output_tif_dir, env_name)

env_names = [ 'xr_cost_ag','xr_cost_agricultural_management','xr_cost_non_ag','xr_cost_transition_ag2ag_diff','xr_transition_cost_ag2non_ag_amortised_diff']
for env_name in env_names:
    env_name_full = f"{env_name}_{env_category}_2050"
    mask = 'bio'
    xr_carbon_cost_map = create_map_data(base_dir, env_category,env_name_full, mask) / 1e6
    print(xr_carbon_cost_map.max(skipna=True).item(),xr_carbon_cost_map.min(skipna=True).item())
    output_tif_dir = f"{base_dir}/map_data"
    xr_to_geotif(xr_carbon_cost_map, output_tif_dir, env_name_full)

env_category = 'carbon_100'
env_name = "xr_carbon_price_2050"
xr_carbon_price = create_map_data(base_dir, env_category, env_name, q=0.97, mask='carbon')
print(xr_carbon_price.max(skipna=True).item(),xr_carbon_price.min(skipna=True).item())
xr_to_geotif(xr_carbon_price, output_tif_dir, env_name)

env_category = 'carbon_100_bio_50'
env_name = "xr_bio_price_2050"
xr_bio_price = create_map_data(base_dir, env_category, env_name, q=0.92, mask='bio')
print(xr_bio_price.max(skipna=True).item(),xr_bio_price.min(skipna=True).item())
xr_to_geotif(xr_bio_price, output_tif_dir, env_name)