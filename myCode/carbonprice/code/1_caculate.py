from joblib import Parallel, delayed
import time
from tools.tools import get_path, get_year
import shutil
import os
import xarray as xr
import numpy_financial as npf
import numpy as np
import pandas as pd
import os

import tools.config as config


def create_xarray(years, base_path, env_category, env_name, mask=None,
                  engine="h5netcdf",
                  cell_dim="cell", cell_chunk="auto",
                  year_chunk=1, parallel=False):
    """
    以 year 维度拼接多个年度 NetCDF，懒加载+分块，避免过多文件句柄。
    """
    file_paths = [
        os.path.join(base_path, str(env_category), str(y), f"xr_{env_name}_{y}.nc")
        for y in years
    ]
    missing = [p for p in file_paths if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(f"以下文件未找到:\n" + "\n".join(missing))

    # 从文件名提取实际年份，确保坐标与文件顺序一致
    valid_years = [int(os.path.basename(p).split("_")[-1].split(".")[0]) for p in file_paths]

    ds = xr.open_mfdataset(
        file_paths,
        engine=engine,
        combine="nested",  # 明确“按给定顺序拼接”
        concat_dim="year",  # 新增 year 维度
        parallel=parallel,  # 一般 False 更稳，避免句柄并发
        chunks={cell_dim: cell_chunk, "year": year_chunk}  # year=1，cell 分块
    ).assign_coords(year=valid_years)

    if mask is not None:
        ds = ds.where(mask, other=0)  # 使用掩码，非掩码区域设为 0

    return ds



def create_summary(env_category, years, base_path, colnames):
    """
    计算指定 env_category 的 GHG benefits, Carbon cost, Average Carbon price，
    并返回一个带自定义列名的 DataFrame。

    参数
    ----
    env_category : str
        环境类别，例如 "carbon_high"
    years : list 或 array
        年份序列
    base_path : str
        数据路径
    colnames : list of str
        列名列表，长度必须是 3，依次对应 [GHG benefits, Carbon cost, Average Carbon price]

    返回
    ----
    df : pandas.DataFrame
        索引为 year，列名由 colnames 指定
    """
    # 加载数据
    xr_carbon_cost_a = create_xarray(years, base_path, env_category,
                                     f"total_cost_{env_category}_amortised")
    xr_carbon = create_xarray(years, base_path, env_category,
                              f"total_{env_category}")

    # 计算指标
    xr_carbon_price_ave_a = (xr_carbon_cost_a.sum(dim="cell") /
                             xr_carbon.sum(dim="cell"))
    da_price = xr_carbon_price_ave_a["data"].sortby("year")
    da_benefits = (xr_carbon.sum(dim="cell")["data"] / 1e6).sortby("year")
    da_cost = (xr_carbon_cost_a.sum(dim="cell")["data"] / 1e6).sortby("year")

    # 检查列名长度
    if len(colnames) != 3:
        raise ValueError("colnames 必须是长度为 3 的列表，顺序为 [GHG benefits, Carbon cost, Average Carbon price]")

    # 组装 DataFrame
    df = pd.DataFrame({
        "year": da_price["year"].values,
        colnames[0]: da_benefits.values,
        colnames[1]: da_cost.values,
        colnames[2]: da_price.values
    }).set_index("year")

    excel_path = base_path.replace("0_base_data", "1_excel")
    output_path = os.path.join(excel_path, f"2_{env_category}_cost_series.xlsx")
    df.to_excel(output_path)
    print(f"已保存: {output_path}")
    return df


task_name = config.TASK_NAME
base_path = f"../../../output/{task_name}/carbon_price/0_base_data"
excel_path = f"../../../output/{task_name}/carbon_price/1_excel"
figure_path = f"../../../output/{task_name}/carbon_price/2_figure"
os.makedirs(excel_path, exist_ok=True)
os.makedirs(figure_path, exist_ok=True)

years = list(range(2011, 2051))

colnames = ["GHG benefits (Mt CO2e)", "Carbon cost (M AUD$)", "Average Carbon price (AUD$/t CO2e)"]
create_summary("carbon_high", years, base_path, colnames)
create_summary("carbon_low", years, base_path, colnames)

bio_names = [
        'carbon_low_bio_10', 'carbon_low_bio_20', 'carbon_low_bio_30', 'carbon_low_bio_40', 'carbon_low_bio_50',
        'carbon_high_bio_10', 'carbon_high_bio_20', 'carbon_high_bio_30', 'carbon_high_bio_40', 'carbon_high_bio_50'
    ]
colnames = ["Biodiversity benefits (Mt CO2e)", "Biodiversity cost (M AUD$)", "Average Biodiversity price (AUD$/t CO2e)"]
for bio_name in bio_names:
    create_summary(bio_name, years, base_path, colnames)


