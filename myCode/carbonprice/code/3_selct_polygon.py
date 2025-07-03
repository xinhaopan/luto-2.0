import tools.config as config
from tools.tools import get_path

import rasterio
import numpy as np
import geopandas as gpd
from rasterio import features
import pandas as pd


def reclassify_and_calculate_proportion(input_tif, shp_path):
    """
    重分类 TIF 并按 shp 分区计算值为 1 的像元比例（包含 nodata）。

    输入:
    - input_tif: str，TIF 文件路径
    - shp_path: str，Shapefile 文件路径（必须包含 'id' 字段）

    输出:
    - pd.DataFrame，列为 ['id', 'proportion_of_1']
    """
    # 读取并重分类 tif：-1 → 0，其余为 1
    with rasterio.open(input_tif) as src:
        arr = src.read(1)
        transform = src.transform
        arr = np.where(arr == -1, 0, 1)

    # 读取 shapefile
    gdf = gpd.read_file(shp_path)
    results = []

    # 遍历每个多边形，按掩膜统计 1 的比例
    for _, row in gdf.iterrows():
        geom = row.geometry
        mask = features.geometry_mask([geom], transform=transform,
                                      invert=True, out_shape=arr.shape)

        zone_values = arr[mask]
        total = zone_values.size
        count_1 = np.count_nonzero(zone_values == 1)

        proportion = count_1 / total if total > 0 else np.nan
        results.append({'id': row['id'], 'proportion_of_1': proportion})

    return arr, pd.DataFrame(results).set_index(gdf.index)


def zonal_stats_masked(input_tif, study_arr, shp, stats='mean', column_name=None):
    """
    对shp分区，统计input_tif在study_arr==1部分的像元，返回df，index为shp的index。

    参数:
    - input_tif: str，tif文件路径
    - study_arr: numpy.ndarray，与tif同尺寸，值为1的区域参与统计
    - shp: str 或 GeoDataFrame，shapefile路径或GeoDataFrame
    - stats: 'mean' 或 'sum'
    - column_name: str，输出结果列名（可选）

    返回:
    - pd.DataFrame，index与shp一致，列为指定统计值
    """
    if isinstance(shp, str):
        gdf = gpd.read_file(shp)
    else:
        gdf = shp

    with rasterio.open(input_tif) as src:
        img = src.read(1)
        transform = src.transform

    assert study_arr.shape == img.shape, "study_arr尺寸必须和tif一致"

    result = []
    for idx, row in gdf.iterrows():
        mask = rasterio.features.geometry_mask([row.geometry], img.shape, transform, invert=True)
        zone_mask = mask & (study_arr == 1)
        vals = img[zone_mask]

        if vals.size == 0:
            stat_val = np.nan
        else:
            if stats == 'mean':
                stat_val = float(np.nanmean(vals))
            elif stats == 'sum':
                stat_val = float(np.nansum(vals))
            else:
                raise ValueError("stats参数只支持'mean'或'sum'")

        result.append(stat_val)

    col_name = column_name if column_name else f'{stats}_val'
    df = pd.DataFrame(result, index=gdf.index, columns=[col_name])
    return df


# —— 使用示例 ——
if __name__ == "__main__":
    input_tif = get_path(f"{config.TASK_DIR}/{config.INPUT_FILES[0]}/out_2010/lmmap_2010.tiff")
    shp_path = "../Map/H_1wkm2.shp"
    study_arr,df_proportion = reclassify_and_calculate_proportion(input_tif, shp_path)

    arr_path = f"{config.TASK_DIR}/carbon_price/data"
    raster_files = ["carbon_cost", "bio_cost", "ghg", "bio"]
    for raster_file in raster_files:
        raster_path = f"{arr_path}/{raster_file}_2050.tif"
        df_stat = zonal_stats_masked(raster_path, study_arr, shp_path, stats='sum', column_name=raster_file)
        df_proportion = df_proportion.join(df_stat)

