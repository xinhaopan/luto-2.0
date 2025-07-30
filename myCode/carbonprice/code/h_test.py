import tools.config as config
from tools.tools import get_path
import os
import numpy as np
import pandas as pd
import geopandas as gpd
from joblib import Parallel, delayed
from rasterio.features import rasterize, geometry_mask
import rasterio


def reclassify_and_calculate_proportion(input_tif, shp_path):
    """
    向量化版：
    1) 重分类（-1→0，其它→1，保留 nodata）
    2) 一次性 rasterize 把每个面烧成一个整数 ID 图层
    3) 用 bincount 统计每个 ID 区域的总像元数和值==1 的像元数
    4) 计算比例并返回 DataFrame
    """
    # 1) 读入栅格并重分类
    with rasterio.open(input_tif) as src:
        arr = src.read(1)
        nodata = src.nodata
        transform = src.transform
        shape = src.shape

    # 重分类：-1→0，其它→1，nodata 保持
    arr2 = np.where(arr == -1, 0, 1)
    if nodata is not None:
        arr2[arr == nodata] = nodata

    # 2) 读入矢量，并给每个面编号 1..N
    gdf = gpd.read_file(shp_path)
    n_shapes = len(gdf)

    shapes = ((geom, i + 1) for i, geom in enumerate(gdf.geometry))
    id_raster = rasterize(
        shapes,
        out_shape=shape,
        transform=transform,
        fill=0,
        dtype='int32'
    )

    # 3) 统计每个面内的像元总数 & 值==1 的像元数
    ids_all = id_raster[id_raster > 0]
    total_count = np.bincount(ids_all, minlength=n_shapes + 1)

    # 只统计 arr2==1 的像元
    mask_one = (arr2 == 1) & (id_raster > 0)
    ids_one = id_raster[mask_one]
    one_count = np.bincount(ids_one, minlength=n_shapes + 1)

    # 4) 计算比例
    total = total_count.astype(float)
    with np.errstate(divide='ignore', invalid='ignore'):
        proportion = one_count / total
    # 丢掉背景 ID=0，留下 1..N 部分
    prop = proportion[1:]

    # 5) 结果 DataFrame
    df = pd.DataFrame({'proportion_of_1': prop}, index=gdf.index)

    return arr2, df


def zonal_stats_vectorized(input_tif, study_arr, shp,
                           stats='mean', column_name=None):
    """
    向量化版 Zonal Stats，输出行数始终和 shp 面数量一致。
    """
    # 1) 读取面和栅格
    if isinstance(shp, str):
        gdf = gpd.read_file(shp)
    else:
        gdf = shp.copy()
    n_shapes = len(gdf)

    with rasterio.open(input_tif) as src:
        img = src.read(1)
        transform = src.transform
        shape = src.shape

    assert study_arr.shape == img.shape, "study_arr 和 input_tif 必须同尺寸"

    # 2) 一次性 rasterize：给每个面编号 1..n_shapes
    shapes = ((geom, i + 1) for i, geom in enumerate(gdf.geometry))
    id_raster = rasterize(
        shapes,
        out_shape=shape,
        transform=transform,
        fill=0,
        dtype='int32'
    )

    # 3) 掩膜出要统计的像元：study_arr==1 且 属于某个面
    mask = (study_arr == 1) & (id_raster > 0)
    vals = img[mask]
    ids = id_raster[mask]

    # 4) 用 bincount，长度设为 n_shapes+1，确保从 1..n_shapes 全都有值
    sum_per_id = np.bincount(ids, weights=vals, minlength=n_shapes + 1)
    count_per_id = np.bincount(ids, minlength=n_shapes + 1)

    # 5) 取出 1..n_shapes 部分并计算
    if stats == 'sum':
        stat = sum_per_id[1:]
    elif stats == 'mean':
        with np.errstate(divide='ignore', invalid='ignore'):
            stat = sum_per_id[1:] / count_per_id[1:]
    else:
        raise ValueError("stats 参数只能是 'sum' 或 'mean'")

    # 6) 构造 DataFrame
    col = column_name or f"{stats}_val"
    df = pd.DataFrame({col: stat}, index=gdf.index)

    return df


def process_single_year(year, input_tif, shp_path, arr_path, unit_div=1e6):
    """
    处理单个年份的数据，计算碳价和生物多样性价格的原始数据

    参数:
    - year: 年份
    - input_tif: 土地管理图路径
    - shp_path: 六边形矢量数据路径
    - arr_path: 栅格数据目录
    - unit_div: 单位换算除数，默认1e6

    返回:
    - dict: 包含年份和价格数据的字典
    """
    print(f"Processing year {year}...")

    try:
        # 获得比例与分区统计
        study_arr, df_proportion = reclassify_and_calculate_proportion(input_tif, shp_path)

        # 需要处理的栅格字段
        raster_files = ['carbon_cost', 'bio_cost', 'ghg', 'bio']

        # 依次统计各个栅格文件
        for raster_file in raster_files:
            raster_path = os.path.join(arr_path, f"{raster_file}_{year}.tif")
            if not os.path.exists(raster_path):
                print(f"Warning: {raster_path} not found, skipping...")
                continue

            df_stat = zonal_stats_vectorized(
                raster_path, study_arr, shp_path,
                stats='sum', column_name=raster_file
            )
            df_stat.index = df_stat.index.astype(df_proportion.index.dtype)
            df_proportion = df_proportion.join(df_stat)

        # 检查是否有必要的列
        required_cols = ['carbon_cost', 'bio_cost', 'ghg', 'bio']
        missing_cols = [col for col in required_cols if col not in df_proportion.columns]
        if missing_cols:
            print(f"Warning: Missing columns for year {year}: {missing_cols}")
            return {
                'year': year,
                'carbon_price_data': None,
                'bio_price_data': None,
                'status': 'missing_data'
            }

        # 创建掩膜（过滤异常值）
        mask_cc = df_proportion["carbon_cost"].between(-1e10, 1, inclusive="both")
        mask_bc = df_proportion["bio_cost"].between(-1e10, 1, inclusive="both")
        mask_c = df_proportion["ghg"].between(-1e10, 1, inclusive="both")
        mask_b = df_proportion["bio"].between(-1e10, 1, inclusive="both")

        # 单位换算
        df_proportion[raster_files] /= unit_div

        # 计算Shadow Price
        with np.errstate(divide='ignore', invalid='ignore'):
            carbon_price = df_proportion["carbon_cost"] / df_proportion["ghg"]
            bio_price = df_proportion["bio_cost"] / df_proportion["bio"]

        # 应用掩膜过滤异常值
        mask_cp = carbon_price.between(-1e10, 1, inclusive="both")
        mask_bp = bio_price.between(-1e10, 1, inclusive="both")

        # 将异常值设为NaN
        carbon_price.loc[mask_cp | mask_c] = np.nan
        bio_price.loc[mask_bp | mask_b] = np.nan

        return {
            'year': year,
            'carbon_price_data': carbon_price.values,
            'bio_price_data': bio_price.values,
            'valid_carbon_count': (~carbon_price.isna()).sum(),
            'valid_bio_count': (~bio_price.isna()).sum(),
            'status': 'success'
        }

    except Exception as e:
        print(f"Error processing year {year}: {str(e)}")
        return {
            'year': year,
            'carbon_price_data': None,
            'bio_price_data': None,
            'status': f'error: {str(e)}'
        }


def calculate_quantile_prices_all_years(years, input_tif, shp_path, arr_path,
                                        quantiles=None, output_path=None,
                                        n_jobs=-1, unit_div=1e6):
    """
    并行计算所有年份的碳价和生物多样性价格的多个分位数值

    参数:
    - years: 年份列表
    - input_tif: 土地管理图路径
    - shp_path: 六边形矢量数据路径
    - arr_path: 栅格数据目录
    - quantiles: 分位数列表，默认[90,91,92,...,100]
    - output_path: 输出Excel文件路径（可选）
    - n_jobs: 并行进程数，默认-1（使用所有核心）
    - unit_div: 单位换算除数，默认1e6

    返回:
    - dict: 键为分位数，值为对应的DataFrame结果
    """
    # 设置默认分位数
    if quantiles is None:
        quantiles = list(range(90, 101))  # [90, 91, 92, ..., 100]

    print(f"开始并行计算 {len(years)} 个年份的价格数据...")
    print(f"分位数范围: {min(quantiles)}% - {max(quantiles)}%")
    print(f"使用 {n_jobs} 个进程进行并行计算")

    # 并行处理所有年份，获取原始价格数据
    results = Parallel(n_jobs=n_jobs)(
        delayed(process_single_year)(year, input_tif, shp_path, arr_path, unit_div)
        for year in years
    )

    # 整理原始数据
    year_data = {}
    for result in results:
        if result['status'] == 'success' and result['carbon_price_data'] is not None:
            year_data[result['year']] = {
                'carbon_price': result['carbon_price_data'],
                'bio_price': result['bio_price_data'],
                'valid_carbon_count': result['valid_carbon_count'],
                'valid_bio_count': result['valid_bio_count']
            }

    print(f"成功处理的年份数: {len(year_data)}")
    print(f"失败的年份数: {len(years) - len(year_data)}")

    # 为每个分位数计算结果
    quantile_results = {}

    for quantile in quantiles:
        print(f"计算 {quantile}% 分位数...")

        # 为当前分位数创建结果表格
        quantile_data = []

        for year in sorted(year_data.keys()):
            data = year_data[year]

            # 计算碳价分位数
            carbon_prices = data['carbon_price']
            if not np.all(np.isnan(carbon_prices)):
                if quantile == 100:
                    carbon_quantile = np.nanmax(carbon_prices)
                else:
                    valid_carbon = carbon_prices[~np.isnan(carbon_prices)]
                    if len(valid_carbon) > 0:
                        carbon_quantile = np.percentile(valid_carbon, quantile)
                    else:
                        carbon_quantile = np.nan
            else:
                carbon_quantile = np.nan

            # 计算生物多样性价格分位数
            bio_prices = data['bio_price']
            if not np.all(np.isnan(bio_prices)):
                if quantile == 100:
                    bio_quantile = np.nanmax(bio_prices)
                else:
                    valid_bio = bio_prices[~np.isnan(bio_prices)]
                    if len(valid_bio) > 0:
                        bio_quantile = np.percentile(valid_bio, quantile)
                    else:
                        bio_quantile = np.nan
            else:
                bio_quantile = np.nan

            quantile_data.append({
                'Year': year,
                'Carbon Price (AU$ tCO2e-1)': carbon_quantile,
                'Biodiversity Price (AU$ ha-1)': bio_quantile,
                'Valid Carbon Count': data['valid_carbon_count'],
                'Valid Bio Count': data['valid_bio_count']
            })

        # 创建当前分位数的DataFrame
        df_quantile = pd.DataFrame(quantile_data)
        quantile_results[quantile] = df_quantile

        # 输出当前分位数的统计信息
        if not df_quantile.empty:
            carbon_max = df_quantile['Carbon Price (AU$ tCO2e-1)'].max()
            bio_max = df_quantile['Biodiversity Price (AU$ ha-1)'].max()
            print(f"  {quantile}%分位数 - 碳价最大值: {carbon_max:.2f}, 生物价格最大值: {bio_max:.2f}")

    # 保存到Excel文件，每个分位数一个sheet
    if output_path:
        print(f"\n保存结果到: {output_path}")
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            for quantile in quantiles:
                sheet_name = f'P{quantile}' if quantile < 100 else 'Max'
                quantile_results[quantile].to_excel(
                    writer,
                    sheet_name=sheet_name,
                    index=False
                )
        print(f"成功保存 {len(quantiles)} 个分位数的结果")

    return quantile_results


def main():
    """
    主函数：设置参数并执行计算
    """
    # 配置参数
    input_tif = f"{get_path(config.INPUT_FILES[0])}/out_2010/lmmap_2010.tiff"
    shp_name = 'H_1kkm2'  # 或者 'H_1wkm2'
    shp_path = f"../Map/{shp_name}.shp"
    arr_path = f"{config.TASK_DIR}/carbon_price/data"

    # 定义要计算的年份范围
    years = list(range(2011, 2051))  # 2030-2050年，根据实际情况调整

    # 设置分位数列表 (90-100)
    quantiles = list(range(90, 101))  # [90, 91, 92, 93, ..., 100]

    # 输出路径
    output_path = f"{config.TASK_DIR}/carbon_price/results/quantile_prices_all_years_{shp_name}.xlsx"

    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # 执行计算
    quantile_results = calculate_quantile_prices_all_years(
        years=years,
        input_tif=input_tif,
        shp_path=shp_path,
        arr_path=arr_path,
        quantiles=quantiles,
        output_path=output_path,
        n_jobs=-1,  # 使用所有可用核心
        unit_div=1e6
    )

    # 显示结果摘要
    print(f"\n计算完成! 共生成 {len(quantile_results)} 个分位数的结果")
    print("\n各分位数结果预览:")
    for quantile in [90, 95, 100]:  # 显示几个关键分位数
        if quantile in quantile_results:
            df = quantile_results[quantile]
            print(f"\n{quantile}% 分位数 (前5行):")
            print(df.head())

    return quantile_results


if __name__ == "__main__":
    results = main()