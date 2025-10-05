# --- Standard library ---
import os
import math
import time
import gzip
import threading
from datetime import datetime
from typing import Sequence, Optional, Union, Dict, Tuple

# --- Third-party ---
import numpy as np
import numpy_financial as npf
import pandas as pd
import xarray as xr
import dill
from joblib import Parallel, delayed
import geopandas as gpd
import rasterio
from rasterio.features import rasterize
import io
import zipfile
from threading import Lock


# --- Local packages ---
from tools.tools import (
    get_path, get_year, save2nc, filter_all_from_dims, nc_to_tif, get_data_RES
)
from tools.helper_data import (
    summarize_to_type, summarize_to_category, build_profit_and_cost_nc,create_processed_xarray,build_sol_profit_and_cost_nc,
    make_prices_nc, summarize_netcdf_to_excel, create_profit_for_cost, create_summary, make_sol_prices_nc
)
from tools import LogToFile, log_memory_usage
import tools.config as config


def tprint(*args, **kwargs):
    """
    打印时自动加上时间戳 (YYYY-MM-DD HH:MM:SS)
    """
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}]", *args, **kwargs)
    return

def get_main_data_variable_name(ds: xr.Dataset) -> str:
    """自动从 xarray.Dataset 中获取唯一的数据变量名。"""
    data_vars_list = list(ds.data_vars)
    if len(data_vars_list) == 1:
        return data_vars_list[0]
    elif len(data_vars_list) == 0:
        raise ValueError("错误：数据集中不包含任何数据变量。")
    else:
        raise ValueError(f"错误：数据集中包含多个数据变量: {data_vars_list}。")

def sum_dims_if_exist(
    nc_path: str,
    vars: Optional[Sequence[str]] = None,   # 指定只处理哪些变量；None=处理全部
    dims = ['lm',"source","Type","GHG_source","Cost type","From water-supply","To water-supply"],
    engine: Optional[str] = "h5netcdf",           # 例如 "h5netcdf" 或 "netcdf4"
    chunks="auto",                           # 大文件建议保留懒加载
    keep_attrs: bool = True,
    finalize: str = "compute",                  # "lazy" | "persist" | "compute"
):
    """
    打开 NetCDF 文件，对给定的维度（如果该变量里存在）执行 sum 归约。
    返回 xarray.Dataset（默认懒计算）。

    参数
    ----
    nc_path : str
        NetCDF 文件路径
    dims : str | list[str]
        想要求和的维度名集合；仅当维度存在于变量中时才会被求和
    vars : list[str] | None
        仅处理这些变量；None 表示处理所有 data_vars
    engine : str | None
        xarray 后端引擎（如 "h5netcdf"）
    chunks : "auto" | dict | None
        dask 分块设置
    keep_attrs : bool
        归约时是否保留 attrs
    finalize : "lazy" | "persist" | "compute"
        返回前是否触发计算：
        - "lazy"：不计算（默认）
        - "persist"：把结果持久在内存（适合反复用）
        - "compute"：直接计算成 numpy-backed

    返回
    ----
    xr.Dataset
    """
    if isinstance(dims, str):
        dims = [dims]

    ds = xr.open_dataset(nc_path, engine=engine, chunks=chunks)

    def _reduce(da: xr.DataArray) -> xr.DataArray:
        present = [d for d in dims if d in da.dims]
        return da.sum(dim=present, keep_attrs=keep_attrs, skipna=True) if present else da

    if vars is None:
        out = ds.map(_reduce)  # 对所有变量应用
    else:
        missing = [v for v in vars if v not in ds.data_vars]
        if missing:
            raise KeyError(f"变量不存在: {missing}")
        out = ds.copy()
        for v in vars:
            out[v] = _reduce(ds[v])

    if finalize == "compute":
        res = out.compute()
        ds.close()
        return res
    if finalize == "persist":
        res = out.persist()
        ds.close()
        return res
    return out

def amortize_costs(data_path_name, amortize_file, years, njobs=0, rate=0.07, horizon=60):
    """
    【最终修复版 - 逐年输出】计算成本均摊，并为每一年生成一个累计成本文件。
    1. 使用 Dask 构建完整的计算图，计算出所有年份的累计摊销成本。
    2. 在保存阶段，通过循环和切片，为每一年单独触发计算并保存一个文件。
    """
    tprint(f"开始计算 '{data_path_name}' 的摊销成本... (逐年输出模式)")
    # --- 1. 数据加载与预处理 (与之前版本完全相同) ---
    file_paths = [os.path.join(data_path_name, f'{year}', f'{amortize_file}_{year}.nc') for year in years]
    existing_files = [p for p in file_paths if os.path.exists(p)]
    if not existing_files: raise FileNotFoundError(
        f"在路径 {data_path_name} 下找不到任何与 '{amortize_file}' 相关的文件。")
    valid_years = sorted([int(path.split('_')[-1].split('.')[0]) for path in existing_files])

    all_costs_ds = xr.open_mfdataset(
        existing_files,
        engine="h5netcdf",  # 推荐后端
        combine="nested",
        concat_dim="year",
        parallel=False,  # 关键：避免句柄并发问题
        chunks={ "cell":'auto', "year": -1}  # year 整块、cell 分块
    ).assign_coords(year=valid_years)

    cost_variable_name = get_main_data_variable_name(all_costs_ds)
    pv_values_all_years = all_costs_ds[cost_variable_name]

    annual_payments = xr.apply_ufunc(
        lambda x: -1 * npf.pmt(rate, horizon, pv=x.astype(np.float64), fv=0, when='begin'),
        pv_values_all_years,
        dask="parallelized",
        output_dtypes=[np.float32],
    ).astype('float32')

    all_years = annual_payments.year.values  # e.g., np.arange(2010, 2051)
    base_shape = annual_payments.sel(year=all_years[0]).drop_vars('year').shape
    n_years = len(all_years)

    # 初始化 numpy array，用于累加所有影响
    amortized_matrix = np.zeros((n_years,) + base_shape, dtype=np.float32)
    for source_year in all_years:
        tprint(f"  - 处理{data_path_name}起始年份 {source_year} ...")
        payment = annual_payments.sel(year=source_year).drop_vars('year').values
        payment = np.nan_to_num(payment, nan=0.0)
        for offset in range(horizon):
            affect_year = source_year + offset
            if affect_year in all_years:
                affect_idx = affect_year - all_years[0]
                amortized_matrix[affect_idx] += payment
    # 构建 xarray.DataArray，添加坐标信息
    coords = {k: v for k, v in annual_payments.coords.items() if k != 'year'}
    coords['year'] = all_years
    dims = ('year',) + tuple(d for d in annual_payments.dims if d != 'year')
    amortized_by_affect_year = xr.DataArray(
        data=amortized_matrix,
        dims=dims,
        coords=coords,
        name='data',
    )
    tprint("start compute...")
    amortized_by_affect_year.compute()
    tprint("compute done.")

    # 关闭句柄
    all_costs_ds.close()

    # === 保存函数 ===
    # 保存各年份输出
    if njobs and njobs > 0:
        def _save_one_year(y: int):
            try:
                out_dir = os.path.join(data_path_name, f"{y}")
                os.makedirs(out_dir, exist_ok=True)
                out_path = os.path.join(out_dir, f"{amortize_file}_amortised_{y}.nc")
                tprint(f"  - [thread] 保存年份 {y} -> {out_path}")

                da_y = amortized_by_affect_year.sel(year=y)
                ds_y = xr.Dataset({'data': da_y})
                save2nc(ds_y, out_path)
                return f"✅ 年份 {y} 已保存"
            except Exception as e:
                return f"❌ 年份 {y} 失败: {e}"

        results = Parallel(n_jobs=njobs, backend="threading")(
            delayed(_save_one_year)(y) for y in all_years
        )
        for msg in results:
            tprint(msg)

    else:
        for y in all_years:
            out_dir = os.path.join(data_path_name, f"{y}")
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, f"{amortize_file}_amortised_{y}.nc")
            tprint(f"  - 保存年份 {y} -> {out_path}")
            da_y = amortized_by_affect_year.sel(year=y)
            ds_y = xr.Dataset({'data': da_y})
            save2nc(ds_y, out_path)
    return

# --- 辅助函数：专门用于计算单个文件对的差异，以便并行化 ---
def calculate_and_save_single_diff(diff_file, year, data_path_name):
    """
    计算并保存单个文件对的差异。
    这个函数将被并行调用。
    """
    # 1. 构造上一年度和当前年度的文件路径
    src_file_0 = os.path.join(data_path_name, str(year),  f"{diff_file}_{year}.nc")
    src_file_1 = os.path.join(data_path_name, f'{year - 1}',  f"{diff_file}_{year-1}.nc")
    tprint(f"Calculating diff for {src_file_0} between years {year-1} and {year}...")

    # 4. 构造目标路径并保存
    variable_name = diff_file.replace('.nc', '')
    dst_filename = f"{variable_name}_diff_{year}.nc"
    dst_file = os.path.join(data_path_name, str(year), dst_filename)
    # 2. 打开这对文件
    with xr.open_dataset(src_file_0) as ds_0, xr.open_dataset(src_file_1) as ds_1:
        # 3. 计算差异
        ds_res = ds_0 - ds_1

    save2nc(ds_res, dst_file)

    return f"  - Success: Calculated and saved diff for {dst_filename}"



def copy_single_file(
        origin_path_name: str,
        target_path_name: str,
        var_prefix: str,  # 例如 "xr_cost_ag"
        year: int,
        dims_to_sum=('lm', 'source', 'Type', 'GHG_source', 'Cost type', 'From water-supply', 'To water-supply'),
        engine: str = "h5netcdf",
        chunks="auto",
        allow_missing_2010: bool = True,
) -> str:
    """
    【静默版】复制并处理单个 NetCDF 文件，移除了所有日志记录，适用于并行环境。
    """
    # 1. 构建文件路径
    year_path = os.path.join(origin_path_name, f"out_{year}")
    target_year_path = os.path.join(target_path_name, str(year))
    os.makedirs(target_year_path, exist_ok=True)

    src_file = os.path.join(year_path, f"{var_prefix}_{year}.nc")
    dst_file = os.path.join(target_year_path, f"{var_prefix}_{year}.nc")

    tprint(f"Copying: {os.path.basename(src_file)} to {dst_file}")

    # 2. 检查源文件是否存在
    if not os.path.exists(src_file):
        if allow_missing_2010 and year == 2010:
            tprint( f"Skipped: {os.path.basename(src_file)} (missing but allowed for year 2010)")
            return

    def _reduce_one(da: xr.DataArray) -> xr.DataArray:
        """对 DataArray 中存在的维度进行求和"""
        if not np.issubdtype(da.dtype, np.number):
            return da

        present_dims = [d for d in dims_to_sum if d in da.dims]

        if present_dims:
            return da.sum(dim=present_dims, keep_attrs=True, skipna=True)
        return da

    with xr.open_dataset(src_file, engine=engine, chunks=chunks) as ds:
        ds = filter_all_from_dims(ds)
        ds_filled = ds.fillna(0)
        out = ds_filled.map(_reduce_one).load()
        save2nc(out, dst_file)
    return f"✅ Copied: {os.path.basename(src_file)} to {dst_file}"


def extract_files_from_zip(
        zip_path: str,
        copy_files: list,
        years: list,
        allow_missing_2010: bool = True
) -> Dict[Tuple[str, int], bytes]:
    """
    一次性从 zip 提取所有需要的文件到内存

    返回: {(var_prefix, year): file_bytes} 字典
    """
    files_dict = {}

    tprint(f"Extracting all files from {os.path.basename(zip_path)}...")
    with zipfile.ZipFile(zip_path, 'r') as zf:
        all_names = set(zf.namelist())

        for var_prefix in copy_files:
            for year in years:
                src_file_in_zip = f"out_{year}/{var_prefix}_{year}.nc"

                if src_file_in_zip in all_names:
                    files_dict[(var_prefix, year)] = zf.read(src_file_in_zip)
                    tprint(f"  Extracted: {src_file_in_zip}")
                elif allow_missing_2010 and year == 2010:
                    tprint(f"  Skipped: {src_file_in_zip} (missing but allowed)")
                    files_dict[(var_prefix, year)] = None  # 标记为跳过
                else:
                    tprint(f"  Warning: {src_file_in_zip} not found")
                    files_dict[(var_prefix, year)] = None

    tprint(f"Extraction complete: {sum(1 for v in files_dict.values() if v is not None)} files loaded")
    return files_dict


def process_single_file_from_memory(
        file_bytes: bytes,
        var_prefix: str,
        year: int,
        target_path_name: str,
        dims_to_sum=('source',),
        engine: str = "h5netcdf",
        chunks="auto",
) -> str:
    """
    处理已经在内存中的文件（无需锁，可并行）
    """
    if file_bytes is None:
        return f"⏭️ Skipped: {var_prefix}_{year}.nc"

    target_year_path = os.path.join(target_path_name, str(year))
    os.makedirs(target_year_path, exist_ok=True)
    dst_file = os.path.join(target_year_path, f"{var_prefix}_{year}.nc")

    tprint(f"Processing: {var_prefix}_{year}.nc")

    def _reduce_one(da: xr.DataArray) -> xr.DataArray:
        if not np.issubdtype(da.dtype, np.number):
            return da
        present_dims = [d for d in dims_to_sum if d in da.dims]
        if present_dims:
            return da.sum(dim=present_dims, keep_attrs=True, skipna=True)
        return da

    with io.BytesIO(file_bytes) as bio:
        with xr.open_dataset(bio, engine=engine, chunks=chunks) as ds:
            ds = filter_all_from_dims(ds)
            ds_filled = ds.fillna(0)
            out = ds_filled.map(_reduce_one).load()

    save2nc(out, dst_file)
    return f"✅ Processed: {var_prefix}_{year}.nc"

# ==============================================================================
# STAGE 1: 计算利润 (Profit = Revenue - Cost)
# ==============================================================================
def calculate_profit_for_run(year, out_path, run_name, cost_basename, revenue_basename):
    """
    为单个情景(Run)和单个类别计算利润。
    """
    tprint(f"{out_path}/{run_name}/{year}: 计算利润...")
    # 构建输入文件路径
    cost_file = os.path.join(out_path, run_name, str(year), f'{cost_basename}_{year}.nc')
    revenue_file = os.path.join(out_path, run_name, str(year), f'{revenue_basename}_{year}.nc')

    # 使用 with 语句确保文件正确关闭
    with xr.open_dataset(cost_file,chunks='auto') as ds_cost, \
            xr.open_dataset(revenue_file,chunks='auto') as ds_revenue:
        # 1. 应用您自定义的过滤器
        ds_revenue_processed = filter_all_from_dims(ds_revenue)
        ds_cost_processed = filter_all_from_dims(ds_cost)

        # 2. 填充 NaN 值
        ds_revenue_filled = ds_revenue_processed.fillna(0)
        ds_cost_filled = ds_cost_processed.fillna(0)

        # --- 【关键修正】 检查 'source' 维度是否存在，如果存在则进行聚合 ---

        # 处理 Revenue 数据集
        # ds.dims 是一个包含所有维度名称的类元组对象
        if 'source' in ds_revenue_filled.dims:
            total_revenue = ds_revenue_filled.sum(dim='source')
        else:
            total_revenue = ds_revenue_filled

        # 处理 Cost 数据集
        if 'source' in ds_cost_filled.dims:
            total_cost = ds_cost_filled.sum(dim='source')
        else:
            total_cost = ds_cost_filled

        profit = total_revenue - total_cost
        profit_out_path = os.path.join(out_path, run_name, str(year))
        os.makedirs(profit_out_path, exist_ok=True)

        # 为了区分，我们给文件名加上 profit 前缀
        profit_filename = f'xr_profit_{cost_basename.replace("xr_cost_", "")}_{year}.nc'
        save2nc(profit, os.path.join(profit_out_path, profit_filename))

        return f"✅ Profit: Calculated for {os.path.basename(out_path)}/{profit_filename}"




# ==============================================================================


def calculate_policy_cost(year, output_path, run_all_names, cost_category, policy_type, cost_names):
    """
    基于利润差计算政策成本 (Carbon 或 Bio)。【优化版】
    """
    tprint(f"Calculating policy cost for {policy_type}/{cost_category} in year {year}...")
    profit_file_basename = f'xr_profit_{cost_category}_{year}.nc'

    if policy_type == 'carbon':
        input_all_names_dif = [run_all_names[1], run_all_names[0]]
        caculate_diff_two_scenarios(input_all_names_dif, cost_names, output_path, year, profit_file_basename, f"xr_cost_{cost_category}")

    elif policy_type == 'bio':
        input_all_names_dif = [run_all_names[2], run_all_names[1]]
        caculate_diff_two_scenarios(input_all_names_dif, cost_names, output_path, year, profit_file_basename, f"xr_cost_{cost_category}")

    elif policy_type == 'counter':
        input_all_names_dif = [run_all_names[2], run_all_names[0]]
        caculate_diff_two_scenarios(input_all_names_dif, cost_names, output_path, year, profit_file_basename, f"xr_cost_{cost_category}")

    tprint(f"✅ All {policy_type} policy cost calculations complete for year {year}.")
    return


def calculate_transition_cost_diff(year, output_path, run_all_names, tran_cost_file, policy_type, cost_names):
    """
    计算转型成本文件的差值 (Run1-Run0 或 Run2-Run1)。
    【优化版】: 使用 .persist() 避免在循环中重复读取文件，提高性能并增强并行稳定性。
    """
    # tprint(f"Calculating transition cost diff for {tran_cost_file} {policy_type} in year {year}...")

    tran_file_basename = f"{tran_cost_file}_{year}.nc"

    if policy_type == 'carbon':
        input_all_names_dif = [run_all_names[0], run_all_names[1]]
        caculate_diff_two_scenarios(input_all_names_dif, cost_names, output_path, year, tran_file_basename,f"{tran_cost_file}_diff")

    elif policy_type == "bio":
        input_all_names_dif = [run_all_names[1], run_all_names[2]]
        caculate_diff_two_scenarios(input_all_names_dif, cost_names, output_path, year, tran_file_basename,f"{tran_cost_file}_diff")

    elif policy_type == "counter":
        input_all_names_dif = [run_all_names[0], run_all_names[2]]
        caculate_diff_two_scenarios(input_all_names_dif, cost_names, output_path, year, tran_file_basename,f"{tran_cost_file}_diff")
    else:
        raise ValueError(f"Invalid policy_type '{policy_type}'. Use 'carbon' or 'bio'.")

    tprint(f"✅ All  {tran_cost_file} {policy_type} cost diff calculations complete for year {year}.")
    return


def caculate_diff_two_scenarios(input_all_names, output_names, output_path, year, env_file_basename,output_part_name):

    for i, (run_name_0, run_name_1) in enumerate(zip(input_all_names[0], input_all_names[1])):
        output_subdir = output_names[i]
        run0_path = os.path.join(output_path, run_name_0, str(year), env_file_basename)
        run1_path = os.path.join(output_path, run_name_1, str(year), env_file_basename)

        # 现在，ds_A 直接从内存中读取，ds_B 从磁盘读取
        with xr.open_dataset(run0_path, chunks='auto') as ds_0, xr.open_dataset(run1_path, chunks='auto') as ds_1:
            ds_0 = filter_all_from_dims(ds_0)
            ds_1 = filter_all_from_dims(ds_1)
            env_diff = ds_1 - ds_0

        # 保存结果
        output_dir = os.path.join(output_path, output_subdir, str(year))
        os.makedirs(output_dir, exist_ok=True)
        output_filename = f"{output_part_name}_{output_subdir}_{year}.nc"
        save2nc(env_diff, os.path.join(output_dir, output_filename))
        tprint(f"  - Saved: {output_filename}")

def calculate_env_diff(year, output_path, input_all_names, env_file, policy_type, output_names):
    env_file_basename = f"{env_file}_{year}.nc"
    if policy_type == "carbon":
        input_all_names_dif = [input_all_names[1], input_all_names[0]]
        caculate_diff_two_scenarios(input_all_names_dif, output_names, output_path, year, env_file_basename, env_file)

    elif policy_type == "bio":
        input_all_names_dif = [input_all_names[1], input_all_names[2]]
        caculate_diff_two_scenarios(input_all_names_dif, output_names, output_path, year, env_file_basename, env_file)

    elif policy_type == "counter":
        input_all_names_dif = [input_all_names[0], input_all_names[2]]
        caculate_diff_two_scenarios(input_all_names_dif, output_names, output_path, year, env_file_basename, env_file)

    else:
        raise ValueError(f"Invalid policy_type '{policy_type}'. Use 'carbon' or 'bio'.")
    return

def aggregate_and_save_cost(year, output_path, cost_names):
    """
    【最终版】聚合单个年份的成本文件，使用一个精确的文件列表。
    """

    base_names = [
        'xr_cost_ag',
        'xr_cost_agricultural_management',
        'xr_cost_non_ag',
        'xr_cost_transition_ag2ag_diff',
    ]
    # 注意：你的输入名带有 _diff，这里兼容并据此判断 am_type
    add_variants = [
        'xr_transition_cost_ag2non_ag_amortised_diff',
        'xr_transition_cost_ag2non_ag_diff',
    ]
    for i in range(len(cost_names)):
        file_dir = os.path.join(output_path, f'{cost_names[i]}', str(year))

        for add_name in add_variants:
            data_type_names_all = base_names + [add_name]

            # 1) 先生成全路径并逐一校验存在性；缺哪个立即报错
            full_paths = [
                os.path.join(file_dir, f'{basename}_{cost_names[i]}_{year}.nc')
                for basename in data_type_names_all
            ]

            # 2) 初始化累加器
            total_sum_ds = None

            # 3) 逐个文件读取 -> 预检查 -> 求和 -> 累加

            # 5) 保存：根据是否包含 'amortised' 判定 am_type
            am_type = 'amortised' if 'amortised' in add_name else 'original'
            final_path = os.path.join(file_dir, f'xr_total_cost_{cost_names[i]}_{am_type}_{year}.nc')

            for file_path in full_paths:
                tprint(f"Aggregated total cost file: {file_path}")
                with xr.open_dataset(file_path,chunks='auto') as ds:
                    ds = filter_all_from_dims(ds)
                    # 将除 'cell' 外的维度全部求和
                    sum_dims = [d for d in ds.dims if d != 'cell']
                    summed_single_ds = ds.sum(dim=sum_dims) if sum_dims else ds

                    if total_sum_ds is None:
                        total_sum_ds = summed_single_ds
                    else:
                        total_sum_ds = total_sum_ds + summed_single_ds
                    save2nc(total_sum_ds, final_path)

            tprint(f"Saved aggregated total cost to {final_path}")
    return

def aggregate_and_save_cost_sol(year, output_path, cost_names):
    """
    【最终版】聚合单个年份的成本文件，使用一个精确的文件列表。
    """

    base_names = [
        'xr_cost_agricultural_management',
        'xr_cost_non_ag',
        'xr_transition_cost_ag2non_ag_amortised_diff',
    ]
    # 注意：你的输入名带有 _diff，这里兼容并据此判断 am_type

    for i in range(len(cost_names)):
        file_dir = os.path.join(output_path, f'{cost_names[i]}', str(year))

        # 1) 先生成全路径并逐一校验存在性；缺哪个立即报错
        full_paths = [
            os.path.join(file_dir, f'{basename}_{cost_names[i]}_{year}.nc')
            for basename in base_names
        ]

        # 2) 初始化累加器
        total_sum_ds = None

        # 3) 逐个文件读取 -> 预检查 -> 求和 -> 累加

        # 5) 保存：根据是否包含 'amortised' 判定 am_type
        final_path = os.path.join(file_dir, f'xr_total_sol_cost_{cost_names[i]}_{year}.nc')

        for file_path in full_paths:
            tprint(f"Aggregated total cost file: {file_path}")
            with xr.open_dataset(file_path,chunks='auto') as ds:
                ds = filter_all_from_dims(ds)
                # 将除 'cell' 外的维度全部求和
                sum_dims = [d for d in ds.dims if d != 'cell']
                summed_single_ds = ds.sum(dim=sum_dims) if sum_dims else ds

                if total_sum_ds is None:
                    total_sum_ds = summed_single_ds
                else:
                    total_sum_ds = total_sum_ds + summed_single_ds
        save2nc(total_sum_ds, final_path)

        tprint(f"Saved aggregated total cost to {final_path}")
    return

def aggregate_and_save_summary(year, output_path, data_type_names, input_files_names, type):
    # 1. 【关键修改】根据传入的列表构建完整的文件路径
    for i in range(len(input_files_names)):
        tprint(f"Aggregating summary for {input_files_names[i]} in year {year}...")
        input_files_name = input_files_names[i]
        file_dir = os.path.join(output_path, f'{input_files_name}', str(year))

        final_dir = os.path.join(output_path, input_files_name, str(year))
        os.makedirs(final_dir, exist_ok=True)

        # 2. 初始化累加器
        total_sum_ds = None

        # 3. 循环处理每一个文件
        for basename in data_type_names:
            file_path = os.path.join(file_dir, f'{basename}_{input_files_name}_{year}.nc')
            with xr.open_dataset(file_path,chunks='auto') as ds:
                filtered_ds = filter_all_from_dims(ds)
                summed_single_ds = filtered_ds.sum(dim=[d for d in filtered_ds.dims if d != 'cell'])
                if total_sum_ds is None:
                    total_sum_ds = summed_single_ds
                else:
                    total_sum_ds += summed_single_ds

        # 5. 保存
        final_path = os.path.join(final_dir, f'xr_total_{type}_{input_files_name}_{year}.nc')
        save2nc(total_sum_ds, final_path)
    return


def calculate_cell_price(input_file, year, base_dir,type,chunks='auto'):
    tprint(f"Processing price {input_file} for year {year}...")

    output_path = os.path.join(base_dir, input_file, str(year), f"xr_{type}_price_{input_file}_{year}.nc")
    cost_path = os.path.join(base_dir, input_file, str(year), f"xr_total_cost_{input_file}_amortised_{year}.nc")
    env_path = os.path.join(base_dir, input_file, str(year), f"xr_total_{type}_{input_file}_{year}.nc")

    with xr.open_dataarray(cost_path, chunks=chunks) as cost_da, xr.open_dataarray(env_path, chunks=chunks) as env_da:
        mask_da = (cost_da >= 1) & (env_da >= 1)
        price_da = cost_da / env_da
        price_da = price_da.where(mask_da, np.nan)
        save2nc(price_da, output_path)

def calculate_price_sol(input_file, year, base_dir,type,chunks='auto'):
    tprint(f"Processing price {input_file} for year {year}...")

    output_path = os.path.join(base_dir, input_file, str(year), f"xr_{type}_price_{input_file}_{year}.nc")
    cost_path = os.path.join(base_dir, input_file, str(year), f"xr_total_sol_cost_{input_file}_{year}.nc")
    env_path = os.path.join(base_dir, input_file, str(year), f"xr_total_{type}_{input_file}_{year}.nc")

    with xr.open_dataarray(cost_path, chunks=chunks) as cost_da, xr.open_dataarray(env_path, chunks=chunks) as env_da:
        mask_da = (cost_da >= 1) & (env_da >= 1)
        price_da = cost_da / env_da
        price_da = price_da.where(mask_da, np.nan)
        save2nc(price_da, output_path)


def xarrays_to_tifs(env_cat, file_part, base_dir, tif_dir, data, remove_negative=True, per_ha=True):
    """处理一个类别+文件部分，并输出tif"""
    print(f"Processing {env_cat} - {file_part}")

    # 构造输入路径
    if file_part == 'total_cost':
        input_path = f"{base_dir}/{env_cat}/2050/xr_{file_part}_{env_cat}_amortised_2050.nc"
    else:
        input_path = f"{base_dir}/{env_cat}/2050/xr_{file_part}_{env_cat}_2050.nc"

    # 读取和处理
    da = xr.open_dataarray(input_path)
    da = da.sum(dim=[d for d in da.dims if d != 'cell'])

    if per_ha:
        da = da / data.REAL_AREA
    if remove_negative:
        da = da.where(da >= 0, np.nan)

    # 输出 cell 版本
    out_tif = f"{tif_dir}/{env_cat}/xr_{file_part}_{env_cat}_2050.tif"
    os.makedirs(os.path.dirname(out_tif), exist_ok=True)
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

def plus_tifs(base_dir, env_cat, cost_names, outpath_part, remove_negative=True):
    cost_arrs = []
    for fname_part in cost_names:
        fname = f"{base_dir}/{env_cat}/xr_{fname_part}_{env_cat}_2050.tif"
        with rasterio.open(fname) as src:
            arr = src.read(1, masked=True).filled(np.nan).astype(np.float32)
            cost_arrs.append(arr)
            if len(cost_arrs) == 1:
                cost_profile = src.profile.copy()

    # 堆叠到三维
    cost_stack = np.stack(cost_arrs, axis=0)  # shape: (n_files, height, width)
    # 计算每个像元是否全为有效值
    all_valid = ~np.any(np.isnan(cost_stack), axis=0)  # True表示所有层都非nan
    # 求和
    cost_sum = np.nansum(cost_stack, axis=0)
    # 把非全有效的地方设为 np.nan
    cost_sum[~all_valid] = np.nan

    # nodata处理
    nodata_value = -9999
    cost_sum[np.isnan(cost_sum)] = nodata_value
    if remove_negative:
        cost_sum[cost_sum < 1] = nodata_value

    # 更新 profile
    profile = cost_profile.copy()
    profile.update(dtype="float32", compress="lzw", nodata=nodata_value)

    # 写出
    out_path = f"{base_dir}/{env_cat}/xr_{outpath_part}_{env_cat}_2050.tif"
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(cost_sum, 1)

def divide_tifs(base_dir,env_cat, cost_names, benefit_names, outpath_part):
    """
    用 cost_names 的所有 tif 求和（去掉小于1的），
    用 benefit_names 的所有 tif 求和（去掉小于1的），
    然后相除，输出结果为 tif。
    """
    # 读取并累加成本影像
    cost_sum = None
    for fname_part in cost_names:
        fname = f"{base_dir}/{env_cat}/xr_{fname_part}_{env_cat}_2050.tif"
        with rasterio.open(fname) as src:
            arr = src.read(1, masked=True).filled(np.nan).astype(np.float32)
            arr[arr < 1] = np.nan
            if cost_sum is None:
                cost_sum = arr
                cost_profile = src.profile.copy()
            else:
                cost_sum = np.nansum(np.stack([cost_sum, arr]), axis=0)

    # 读取并累加效益影像
    benefit_sum = None
    for fname_part in benefit_names:
        fname = f"{base_dir}/{env_cat}/xr_{fname_part}_{env_cat}_2050.tif"
        with rasterio.open(fname) as src:
            arr = src.read(1, masked=True).filled(np.nan).astype(np.float32)
            arr[arr < 1] = np.nan
            if benefit_sum is None:
                benefit_sum = arr
                benefit_profile = src.profile.copy()
            else:
                benefit_sum = np.nansum(np.stack([benefit_sum, arr]), axis=0)

    # 输入一致性检查
    if cost_sum.shape != benefit_sum.shape:
        raise ValueError("成本和效益影像的大小不一致！")
    if cost_profile['transform'] != benefit_profile['transform'] or cost_profile['crs'] != benefit_profile['crs']:
        raise ValueError("成本和效益影像的 transform 或 CRS 不一致！")

    # 相除
    out = cost_sum / benefit_sum

    # 只保留有效值，其他位置设为 nodata
    nodata_value = -9999
    out[np.isnan(out)] = nodata_value
    out[benefit_sum < 1] = nodata_value  # 避免除以小于1的效益
    out[cost_sum < 1] = nodata_value     # 避免成本小于1

    # 更新 profile
    profile = cost_profile.copy()
    profile.update(dtype="float32", compress="lzw", nodata=nodata_value)

    # 写出
    out_path = f"{base_dir}/{env_cat}/xr_{outpath_part}_{env_cat}_2050.tif"
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(out, 1)

def xarrays_to_tifs_by_type(
    env_cat,
    file_part,
    base_dir,
    tif_dir,
    data,
    sum_dim,                # 你要合并的第二个维度（比如 "year"）
    remove_negative=False,
    per_ha=True
):
    """
    按照 'cell' 和 sum_dim 对数据求和，分别输出每个 sum_dim 坐标的 tif 文件，
    并先输出总和版本（对除了cell的所有维度求和）

    Parameters
    ----------
    env_cat: str
    file_part: str
    base_dir: str
    tif_dir: str
    data: object
    sum_dim: str   # 你要分组的维度，比如 'year'
    remove_negative: bool
    per_ha: bool
    """
    print(f"Processing {env_cat} - {file_part} by {sum_dim}")

    # 构造输入路径
    input_path = f"{base_dir}/{env_cat}/2050/xr_{file_part}_2050.nc"

    # 读取和处理
    da = xr.open_dataarray(input_path)

    # ----------- 1. 求总和版本（除了cell的所有维度都求和） -----------
    sum_dims_total = [d for d in da.dims if d != 'cell']
    da_total = da.sum(dim=sum_dims_total)

    if per_ha:
        da_total = da_total / data.REAL_AREA
    if remove_negative:
        da_total = da_total.where(da_total >= 0, np.nan)

    out_total_tif = f"{tif_dir}/{env_cat}/xr_total_{file_part}_{env_cat}_2050.tif"
    os.makedirs(os.path.dirname(out_total_tif), exist_ok=True)
    nc_to_tif(data, da_total, out_total_tif)
    print(f"Saved {out_total_tif}")

    # ----------- 2. 按 sum_dim 输出分组 tif -----------
    if sum_dim not in da.dims:
        raise ValueError(f"{sum_dim} 不在数据的维度 {da.dims} 中！")

    results = [out_total_tif]
    # 遍历新维度的所有坐标
    for coord_val in da[sum_dim].values:
        da_slice = da.sel({sum_dim: coord_val})
        sum_dims = [d for d in da_slice.dims if d != 'cell']
        da_out = da_slice.sum(dim=sum_dims)

        if per_ha:
            da_out = da_out / data.REAL_AREA
        if remove_negative:
            da_out = da_out.where(da_out >= 0, np.nan)

        out_tif = f"{tif_dir}/{env_cat}/xr_{file_part}_{env_cat}_{coord_val}_2050.tif"
        os.makedirs(os.path.dirname(out_tif), exist_ok=True)
        nc_to_tif(data, da_out, out_tif)
        results.append(out_tif)
        print(f"Saved {out_tif}")

    return results


def create_shp(env_cat, shp_name, file_parts, tif_dir):
    for file_part in file_parts:
        tif_env_dir = os.path.join(tif_dir, env_cat)
        input_tif_name = f'xr_{file_part}_{env_cat}_2050.tif'
        out_shp = os.path.join(tif_env_dir, f'{shp_name}', f'{shp_name}_{file_part}_{env_cat}_2050.shp')
        os.makedirs(os.path.dirname(out_shp), exist_ok=True)
        shp_path = f"../Map/{shp_name}.shp"
        zonal_stats_rasterized(tif_env_dir, input_tif_name, shp_path, out_shp)

def zonal_stats_rasterized(input_tif_dir, input_tif_name, shp_path, out_shp,
                           extra_nodata_vals=(-9999.0,), drop_allnan=True):
    # 1) 读 shp 与 tif
    gdf = gpd.read_file(shp_path)
    input_tif = os.path.join(input_tif_dir, input_tif_name)

    with rasterio.open(input_tif) as src:
        img_m = src.read(1, masked=True)  # MaskedArray（若 nodata 未设置，mask 可能无效）
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
    ids = id_arr[valid_mask]

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
    gdf["sum"] = sum_stat
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

def main(task_dir, njobs):
    # ============================================================================
    output_path = f'{task_dir}/carbon_price/0_base_data'
    os.makedirs(output_path, exist_ok=True)
    tprint(f"任务目录: {task_dir}")

    area_files = ['xr_area_agricultural_landuse', 'xr_area_agricultural_management','xr_area_non_agricultural_landuse']
    cost_files = ['xr_cost_ag', 'xr_cost_agricultural_management', 'xr_cost_non_ag', 'xr_cost_transition_ag2ag',
                  'xr_transition_cost_ag2non_ag']
    revenue_files = ['xr_revenue_ag', 'xr_revenue_agricultural_management', 'xr_revenue_non_ag']
    carbon_files = ['xr_GHG_ag', 'xr_GHG_ag_management', 'xr_GHG_non_ag', 'xr_transition_GHG']
    bio_files = ['xr_biodiversity_GBF2_priority_ag', 'xr_biodiversity_GBF2_priority_ag_management',
                 'xr_biodiversity_GBF2_priority_non_ag']
    carbon_sol_files = ['xr_GHG_ag', 'xr_GHG_ag_management', 'xr_GHG_non_ag']
    bio_sol_files = ['xr_biodiversity_GBF2_priority_ag','xr_biodiversity_GBF2_priority_ag_management', 'xr_biodiversity_GBF2_priority_non_ag']

    amortize_files = ['xr_transition_cost_ag2non_ag']
    economic_files = config.economic_files
    env_files = carbon_files + bio_files

    economic_sol_files = ['xr_cost_agricultural_management', 'xr_cost_non_ag', 'xr_transition_cost_ag2non_ag_amortised', 'xr_revenue_ag','xr_revenue_agricultural_management', 'xr_revenue_non_ag']

    input_files_0 = config.input_files_0
    input_files_1 = config.input_files_1
    input_files_2 = config.input_files_2
    input_files = input_files_0 + input_files_1 + input_files_2
    input_all_names = [input_files_0, input_files_1, input_files_2]

    carbon_names = config.carbon_names
    carbon_bio_names = config.carbon_bio_names
    counter_carbon_bio_names = config.counter_carbon_bio_names
    output_all_names = carbon_names + carbon_bio_names + counter_carbon_bio_names


    years = [i for i in range(2010, 2051)]

    # ============================================================================
    # 第一批：数据预处理阶段 (摊销成本计算 + 文件复制/差异计算)
    # ============================================================================
    start_time = time.time()

    tprint("=" * 80)

    # --- 第一批任务 (拆分为两个独立的组) ---
    # ----------------------------------------------------------------------------
    # ===========================================================================
    # --- 阶段 1: 文件处理 ---
#     tprint("\n--- 文件copy ---")
#
#     for input_file in list(dict.fromkeys(input_files)):
#         origin_path_name = os.path.join(task_dir, input_file,'Run_Archive.zip')
#         target_path_name = os.path.join(output_path, input_file)
#         tprint(f"  -> 正在copy: {origin_path_name}")
#         copy_files = cost_files + revenue_files + carbon_files + bio_files + area_files
#         # 直接调用函数，而不是用 delayed 包装
#
#         # --- 1. 并行化文件复制 (逻辑不变) ---
#         if copy_files:
#             # 步骤1: 一次性提取所有文件到内存（串行，但只执行一次）
#             files_in_memory = extract_files_from_zip(
#                 origin_path_name,
#                 copy_files,
#                 years,
#                 allow_missing_2010=True
#             )
#
#             # 步骤2: 并行处理所有文件（无锁，充分并行）
#             if njobs == 0:
#                 # 串行处理
#                 for (var_prefix, year), file_bytes in files_in_memory.items():
#                     process_single_file_from_memory(
#                         file_bytes, var_prefix, year, target_path_name, dims_to_sum=('source',)
#                     )
#             else:
#                 # 并行处理
#                 Parallel(n_jobs=njobs)(
#                     delayed(process_single_file_from_memory)(
#                         file_bytes, var_prefix, year, target_path_name, dims_to_sum=('source',)
#                     )
#                     for (var_prefix, year), file_bytes in files_in_memory.items()
#                 )
#
#     tprint(f"✅ 文件copy任务完成!")
#
#     if njobs == 0:
#         for i in range(len(list(dict.fromkeys(input_files)))):
#             data_path_name = os.path.join(output_path, list(dict.fromkeys(input_files))[i])
#             amortize_costs(data_path_name, amortize_files[0], years, njobs=njobs)
#     else:
#         Parallel(n_jobs=7, backend="loky")(
#             delayed(amortize_costs)(
#                 os.path.join(output_path, run_name),  # data_path_name
#                 amortize_files[0],  # 你的第二个参数
#                 years,
#                 njobs=math.ceil(njobs/7)  # 传给内部的并行参数（若有）
#             )
#             for run_name in list(list(dict.fromkeys(input_files)))
#         )
#     tprint("摊销成本计算 完成!")
#     #
#     ##--- 阶段 2: carbon & bio计算 ---
#     if njobs == 0:
#         for env_file in env_files:
#             for year in years[1:]:
#                 calculate_env_diff(year, output_path, input_all_names, env_file, 'carbon', carbon_names)
#                 calculate_env_diff(year, output_path, input_all_names, env_file, 'bio', carbon_bio_names)
#                 calculate_env_diff(year, output_path, input_all_names, env_file, 'counter', counter_carbon_bio_names)
#     else:
#         for env_file in env_files:
#             Parallel(n_jobs=njobs)(
#                 delayed(calculate_env_diff)(year, output_path, input_all_names, env_file, 'carbon', carbon_names)
#                 for year in years[1:]
#             )
#             Parallel(n_jobs=njobs)(
#                 delayed(calculate_env_diff)(year, output_path, input_all_names, env_file, 'bio', carbon_bio_names)
#                 for year in years[1:]
#             )
#             Parallel(n_jobs=njobs)(
#                 delayed(calculate_env_diff)(year, output_path, input_all_names, env_file, 'counter', counter_carbon_bio_names)
#                 for year in years[1:]
#             )
#
#     tprint("\n--- 阶段 2: 汇总carbon & bio计算 ---")
#     if njobs == 0:
#         for year in years[1:]:
#             # 直接调用
#             aggregate_and_save_summary(year, output_path, carbon_files, output_all_names,'carbon')
#             aggregate_and_save_summary(year, output_path, bio_files, output_all_names,'bio')
#     else:
#         Parallel(n_jobs=njobs)(
#             delayed(aggregate_and_save_summary)(year, output_path, carbon_files, output_all_names,'carbon')
#             for year in years[1:]
#         )
#         Parallel(n_jobs=njobs)(
#             delayed(aggregate_and_save_summary)(year, output_path, bio_files, output_all_names,'bio')
#             for year in years[1:]
#         )
#
#     if njobs == 0:
#         for year in years[1:]:
#             # 直接调用
#             aggregate_and_save_summary(year, output_path, carbon_sol_files, output_all_names,'sol_carbon')
#             aggregate_and_save_summary(year, output_path, bio_sol_files, output_all_names,'sol_bio')
#     else:
#         Parallel(n_jobs=njobs)(
#             delayed(aggregate_and_save_summary)(year, output_path, carbon_sol_files, output_all_names,'sol_carbon')
#             for year in years[1:]
#         )
#         Parallel(n_jobs=njobs)(
#             delayed(aggregate_and_save_summary)(year, output_path, bio_sol_files, output_all_names,'sol_bio')
#             for year in years[1:]
#         )
#
#     tprint(f"✅ 第2批任务汇总carbon & bio完成! ")
#
#     # --- 阶段 3: 利润计算 ---
#     tprint("\n--- 阶段 3: 利润计算 ---")
#     profit_categories = zip(cost_files, revenue_files)
#     for cost_base, rev_base in profit_categories:
#         if njobs == 0:
#             for run_names in input_all_names:
#                 for run_name in run_names:
#                     for year in years:
#                         # 直接调用
#                         calculate_profit_for_run(year, output_path, run_name, cost_base, rev_base)
#         else:
#             for run_names in input_all_names:
#                 for run_name in run_names:
#                     Parallel(n_jobs=njobs)(
#                         delayed(calculate_profit_for_run)(year, output_path, run_name, cost_base, rev_base)
#                         for year in years
#                     )
#     tprint(f"✅ 第3批任务完成!")
#
#     ##--- 阶段 4: 政策成本计算 ---
#     tprint("\n--- 阶段 4: 政策成本计算 ---")
#     category_costs = ['ag', 'agricultural_management', 'non_ag']
#     for category in category_costs:
#         if njobs == 0:
#             for year in years[1:]:
#                 # 直接调用
#                 calculate_policy_cost(year, output_path, input_all_names, category, 'carbon',carbon_names)
#                 calculate_policy_cost(year, output_path, input_all_names, category, 'bio', carbon_bio_names)
#                 calculate_policy_cost(year, output_path, input_all_names, category, 'counter', counter_carbon_bio_names)
#         else:
#             Parallel(n_jobs=njobs)(
#                 delayed(calculate_policy_cost)(year, output_path, input_all_names, category, 'carbon', carbon_names)
#                 for year in years[1:]
#             )
#             Parallel(n_jobs=njobs)(
#                 delayed(calculate_policy_cost)(year, output_path, input_all_names, category, 'bio', carbon_bio_names)
#                 for year in years[1:]
#             )
#             Parallel(n_jobs=njobs)(
#                 delayed(calculate_policy_cost)(year, output_path, input_all_names, category, 'counter', counter_carbon_bio_names)
#                 for year in years[1:]
#             )
#     tprint(f"✅ 第4批任务完成! ")
#
#     ##--- 阶段 5: 转型成本差值计算 (仅独立部分) ---
#     tprint("\n--- 阶段 5: 转型成本差值计算 ---")
#     independent_tran_files = ['xr_cost_transition_ag2ag', 'xr_transition_cost_ag2non_ag',
#                               'xr_transition_cost_ag2non_ag_amortised']
#     for tran_file in independent_tran_files:
#         tprint(f"Processing transition cost file: {tran_file}...")
#         if njobs == 0:
#             for year in years[1:]:
#                 # 直接调用
#                 calculate_transition_cost_diff(year, output_path, input_all_names, tran_file, 'carbon', carbon_names)
#                 calculate_transition_cost_diff(year, output_path, input_all_names, tran_file, 'bio', carbon_bio_names)
#                 calculate_transition_cost_diff(year, output_path, input_all_names, tran_file, 'counter', counter_carbon_bio_names)
#         else:
#             Parallel(n_jobs=math.ceil(njobs/2))(
#                 delayed(calculate_transition_cost_diff)(year, output_path, input_all_names, tran_file, 'carbon', carbon_names)
#                 for year in years[1:]
#             )
#             Parallel(n_jobs=math.ceil(njobs/2))(
#                 delayed(calculate_transition_cost_diff)(year, output_path, input_all_names, tran_file, 'bio', carbon_bio_names)
#                 for year in years[1:]
#             )
#             Parallel(n_jobs=math.ceil(njobs/2))(
#                 delayed(calculate_transition_cost_diff)(year, output_path, input_all_names, tran_file, 'counter', counter_carbon_bio_names)
#                 for year in years[1:]
#             )
#     tprint(f"✅ 第5批 转型成本差值计算 任务完成! ")
#
#     # --- 阶段 6: 成本聚合 ---
#     tprint("\n--- 阶段 6: 成本聚合 ---")
#
#     if njobs == 0:
#         for year in years[1:]:
#             # 直接调用
#             aggregate_and_save_cost(year, output_path,carbon_names)
#             aggregate_and_save_cost(year, output_path,carbon_bio_names)
#             aggregate_and_save_cost(year, output_path,counter_carbon_bio_names)
#     else:
#         Parallel(n_jobs=njobs)(
#             delayed(aggregate_and_save_cost)(year, output_path, carbon_names)
#             for year in years[1:]
#         )
#         Parallel(n_jobs=njobs)(
#             delayed(aggregate_and_save_cost)(year, output_path, carbon_bio_names)
#             for year in years[1:]
#         )
#         Parallel(n_jobs=njobs)(
#             delayed(aggregate_and_save_cost)(year, output_path, counter_carbon_bio_names)
#             for year in years[1:]
#         )
#
#     if njobs == 0:
#         for year in years[1:]:
#             # 直接调用
#             aggregate_and_save_cost_sol(year, output_path,carbon_names)
#             aggregate_and_save_cost_sol(year, output_path,carbon_bio_names)
#             aggregate_and_save_cost_sol(year, output_path,counter_carbon_bio_names)
#     else:
#         Parallel(n_jobs=njobs)(
#             delayed(aggregate_and_save_cost_sol)(year, output_path, carbon_names)
#             for year in years[1:]
#         )
#         Parallel(n_jobs=njobs)(
#             delayed(aggregate_and_save_cost_sol)(year, output_path, carbon_bio_names)
#             for year in years[1:]
#         )
#         Parallel(n_jobs=njobs)(
#             delayed(aggregate_and_save_cost_sol)(year, output_path, counter_carbon_bio_names)
#             for year in years[1:]
#         )
#
#     tprint(f"✅ 第6批 (最终聚合) 任务完成! ")
#
#     # --- 阶段 7: 价格计算 ---
#     tprint("\n--- 阶段 7: 价格计算 ---")
#
#     if njobs == 0:
#         for input_file in output_all_names:
#             for year in years[1:]:
#                 calculate_cell_price(input_file, year, output_path,'carbon')
#                 calculate_cell_price(input_file, year, output_path,'bio')
#     else:
#         for input_file in output_all_names:
#             Parallel(n_jobs=njobs)(
#                 delayed(calculate_cell_price)(input_file, year, output_path,'carbon')
#                 for year in years[1:]
#             )
#             Parallel(n_jobs=njobs)(
#                 delayed(calculate_cell_price)(input_file, year, output_path,'bio')
#                 for year in years[1:]
#             )
#
#     tprint(f"✅ 第7批 价格计算 任务完成! ")
#    # ==========================================================================
#
# # ============================================================================
#     excel_path = f"../../../output/{config.TASK_NAME}/carbon_price/1_excel"
#     os.makedirs(excel_path, exist_ok=True)
#
#     for input_file in list(dict.fromkeys(input_files)):
#         print(f"carbon: {input_file}")
#         df = summarize_netcdf_to_excel(input_file, years[1:], carbon_files, njobs, 'carbon')
#     for input_file in list(dict.fromkeys(input_files)):
#         print(f"biodiversity: {input_file}")
#         df = summarize_netcdf_to_excel(input_file, years[1:], bio_files, njobs, 'biodiversity')
#     for input_file in list(dict.fromkeys(input_files)):
#         print(f"economic: {input_file}")
#         df = summarize_netcdf_to_excel(input_file, years[1:], economic_files, np.ceil(njobs/2), 'economic')
# #
# #     # ---------------------------------------make excel 1_cost---------------------------------------
#     profit_0_list = []
#     for input_file in input_files_0:
#         profit_0_list.append(create_profit_for_cost(excel_path, input_file))
#     profit_1_list = []
#     for input_file in input_files_1:
#         profit_1_list.append(create_profit_for_cost(excel_path, input_file))
#     profit_2_list = []
#     for input_file in input_files_2:
#         profit_2_list.append(create_profit_for_cost(excel_path, input_file))
#
#     bio_nums = int(len(input_files_2) / len(input_files_1))
#     for i in range(len(input_files_1)):
#         df = profit_0_list[i] - profit_1_list[i]
#         df.columns = df.columns.str.replace('profit', '')
#         df['Total'] = df.sum(axis=1)
#         df.to_excel(os.path.join(excel_path, f'1_Cost_{carbon_names[i]}.xlsx'))
#
#         df = profit_1_list[i] - profit_2_list[i]
#         df.columns = df.columns.str.replace('profit', '')
#         df['Total'] = df.sum(axis=1)
#         df.to_excel(os.path.join(excel_path, f'1_Cost_{carbon_bio_names[i]}.xlsx'))
#
#         df = profit_0_list[i] - profit_2_list[i]
#         df.columns = df.columns.str.replace('profit', '')
#         df['Total'] = df.sum(axis=1)
#         df.to_excel(os.path.join(excel_path, f'1_Cost_{counter_carbon_bio_names[i]}.xlsx'))
#
#
#     # -----------------------------------make excel 1_processed carbon/bio---------------------------------------
#     for input_file in dict.fromkeys(input_files):
#         df = pd.read_excel(os.path.join(excel_path, f'0_Origin_carbon_{input_file}.xlsx'), index_col=0)
#         df.columns = df.columns.str.replace(' GHG', '')
#         new_rows_list = []
#
#         # 从第二行开始循环 (索引 i 从 1 到 df 的末尾)
#         for i in range(1, len(df)):
#             # 取出当前行并取负
#             new_row = df.iloc[i].copy()
#             new_row = new_row * -1
#
#             # 关键步骤：新行的第一列 = (原值取负) + (原df中上一行第一列的值)
#             new_row.iloc[0] = -df.iloc[i, 0] + df.iloc[i - 1, 0]
#
#             # 将计算出的新行（这是一个 Series）添加到列表中
#             new_rows_list.append(new_row)
#
#         # 使用收集到的行列表一次性创建新的 DataFrame
#         # 这样做比在循环中反复 concat 更高效
#         new_df = pd.DataFrame(new_rows_list)
#
#         # 将新 DataFrame 的索引设置为与原数据对应（从 1 开始）
#         new_df.index = df.index[1:]
#         new_df['Total'] = new_df.sum(axis=1)
#         new_df.to_excel(os.path.join(excel_path, f'1_Processed_carbon_{input_file}.xlsx'))
#
#     for input_file in dict.fromkeys(input_files):
#         df = pd.read_excel(os.path.join(excel_path, f'0_Origin_biodiversity_{input_file}.xlsx'), index_col=0)
#         df.columns = df.columns.str.replace(' biodiversity', '')
#         new_rows_list = []
#
#         # 从第二行开始循环 (索引 i 从 1 到 df 的末尾)
#         for i in range(1, len(df)):
#             # 取出当前行并取负
#             new_row = df.iloc[i].copy()
#
#             new_row.iloc[0] = df.iloc[i, 0] - df.iloc[i - 1, 0]
#
#             # 将计算出的新行（这是一个 Series）添加到列表中
#             new_rows_list.append(new_row)
#
#         # 使用收集到的行列表一次性创建新的 DataFrame
#         # 这样做比在循环中反复 concat 更高效
#         new_df = pd.DataFrame(new_rows_list)
#
#         # 将新 DataFrame 的索引设置为与原数据对应（从 1 开始）
#         new_df.index = df.index[1:]
#         new_df['Total'] = new_df.sum(axis=1)
#         new_df.to_excel(os.path.join(excel_path, f'1_Processed_bio_{input_file}.xlsx'))
# #
# #
# #     # -----------------------------------make excel 2_cost & carbon/bio & average price---------------------------------------
#     colnames = ["Change in GHG benefits (Mt CO2e)", "Carbon cost (M AUD$)", "Average Carbon price (AUD$/t CO2e)"]
#     if njobs == 0:
#         for carbon_name in carbon_names:
#             create_summary(carbon_name, years[1:], output_path,'carbon', colnames)
#         for carbon_bio_name in carbon_bio_names:
#             create_summary(carbon_bio_name, years[1:], output_path,'carbon', colnames)
#         for counter_carbon_bio_name in counter_carbon_bio_names:
#             create_summary(counter_carbon_bio_name, years[1:], output_path,'carbon', colnames)
#     else:
#         Parallel(n_jobs=njobs)(
#             delayed(create_summary)(carbon_name, years[1:], output_path,'carbon', colnames)
#             for carbon_name in carbon_names
#         )
#         Parallel(n_jobs=njobs)(
#             delayed(create_summary)(carbon_bio_name, years[1:], output_path,'carbon', colnames)
#             for carbon_bio_name in carbon_bio_names
#         )
#         Parallel(n_jobs=njobs)(
#             delayed(create_summary)(counter_carbon_bio_name, years[1:], output_path,'carbon', colnames)
#             for counter_carbon_bio_name in counter_carbon_bio_names
#         )
#
#     colnames = ["Change in biodiversity benefits (Mt CO2e)", "Biodiversity cost (M AUD$)",
#                 "Average Biodiversity price (AUD$/t CO2e)"]
#     if njobs == 0:
#         for bio_name in carbon_bio_names:
#             create_summary(bio_name, years[1:], output_path,'bio', colnames)
#         for counter_carbon_bio_name in counter_carbon_bio_names:
#             create_summary(counter_carbon_bio_name, years[1:], output_path,'bio', colnames)
#     else:
#         Parallel(n_jobs=njobs)(
#             delayed(create_summary)(bio_name, years[1:], output_path,'bio', colnames)
#             for bio_name in carbon_bio_names
#         )
#         Parallel(n_jobs=njobs)(
#             delayed(create_summary)(counter_carbon_bio_name, years[1:], output_path,'bio', colnames)
#             for counter_carbon_bio_name in counter_carbon_bio_names
#         )
#
#     summarize_to_category(output_all_names, years[1:], carbon_files, 'xr_total_carbon', n_jobs=41)
#     summarize_to_category(output_all_names, years[1:], bio_files, 'xr_total_bio', n_jobs=41)
#
#     summarize_to_category(list(dict.fromkeys(input_files)), years[1:], carbon_files, 'xr_total_carbon_original', n_jobs=41,scenario_name=False)
#     summarize_to_category(list(dict.fromkeys(input_files)), years[1:], bio_files, 'xr_total_bio_original', n_jobs=41,scenario_name=False)
#
#     profit_da = summarize_to_category(list(dict.fromkeys(input_files)), years[1:], economic_files, 'xr_cost_for_profit', n_jobs=41,scenario_name=False)
#     build_profit_and_cost_nc(profit_da, list(dict.fromkeys(input_files_0)), input_files_1, input_files_2, carbon_names, carbon_bio_names,
#                              counter_carbon_bio_names)
#
#     make_prices_nc(output_all_names)
#
#     summarize_to_category(output_all_names, years[1:], carbon_sol_files, 'xr_total_sol_carbon', n_jobs=41)
#     summarize_to_category(output_all_names, years[1:], bio_sol_files, 'xr_total_sol_bio', n_jobs=41)
#
#     profit_sol_da = summarize_to_category(list(dict.fromkeys(input_files)), years[1:], economic_sol_files, 'xr_sol_cost_for_profit', n_jobs=41,scenario_name=False)
#     build_sol_profit_and_cost_nc(profit_sol_da, list(dict.fromkeys(input_files_0)), input_files_1, input_files_2, carbon_names, carbon_bio_names,
#                                  counter_carbon_bio_names)
#
#     make_sol_prices_nc(output_all_names)
#
#     files = ['xr_cost_agricultural_management', 'xr_cost_non_ag', 'xr_transition_cost_ag2non_ag_amortised_diff',
#              'xr_GHG_ag_management', 'xr_GHG_non_ag', 'xr_biodiversity_GBF2_priority_ag_management',
#              'xr_biodiversity_GBF2_priority_non_ag']
#     dim_names = ['am', 'lu', 'To land-use', 'am', 'lu', 'am', 'lu']
#
#     for file, dim_name in zip(files, dim_names):
#         summarize_to_type(
#             scenarios=output_all_names,
#             years=years[1:],
#             file=file,
#             keep_dim=dim_name,
#             output_file=f'{file}',
#             var_name='data',
#             scale=1e6,
#             n_jobs=njobs,
#             dtype='float32',
#         )
#
#     files = ['xr_area_agricultural_management','xr_area_non_agricultural_landuse',
#              'xr_biodiversity_GBF2_priority_ag_management','xr_biodiversity_GBF2_priority_non_ag',
#              'xr_GHG_ag_management','xr_GHG_non_ag',
#              'xr_cost_agricultural_management', 'xr_cost_non_ag', 'xr_transition_cost_ag2non_ag_amortised']
#     dim_names = ['am','lu','am','lu','am','lu','am', 'lu', 'To land-use']
#
#     for file, dim_name in zip(files, dim_names):
#         summarize_to_type(
#             scenarios=list(dict.fromkeys(input_files)),
#             years=years[1:],
#             file=file,
#             keep_dim=dim_name,
#             output_file=f'{file}',
#             var_name='data',
#             scale=1e6,
#             n_jobs=njobs,
#             dtype='float32',
#             scenario_name=False
#         )

    tif_dir = f"../../../output/{config.TASK_NAME}/carbon_price/4_tif"
    output_path = f"../../../output/{config.TASK_NAME}/carbon_price/0_base_data"
    data = get_data_RES(f"../../../output/{config.TASK_NAME}/{input_files_0[0]}")

    cost_file_parts = ['total_cost', 'cost_agricultural_management', 'cost_non_ag',
                       'transition_cost_ag2non_ag_amortised_diff']
    tasks = [(env_cat, file_part) for env_cat in output_all_names for file_part in cost_file_parts]
    results = Parallel(n_jobs=njobs)(  # 这里你可以改 n_jobs，比如 8 或 -1 用所有CPU
        delayed(xarrays_to_tifs)(env_cat, file_part, output_path, tif_dir, data)
        for env_cat, file_part in tasks
    )

    GHG_file_parts = ['total_carbon', 'GHG_ag_management', 'GHG_non_ag']
    tasks = [(env_cat, file_part) for env_cat in output_all_names for file_part in GHG_file_parts]
    results = Parallel(n_jobs=njobs)(  # 这里你可以改 n_jobs，比如 8 或 -1 用所有CPU
        delayed(xarrays_to_tifs)(env_cat, file_part, output_path, tif_dir, data, remove_negative=False)
        for env_cat, file_part in tasks
    )

    bio_file_parts = ['total_bio', 'biodiversity_GBF2_priority_ag_management', 'biodiversity_GBF2_priority_non_ag']
    tasks = [(env_cat, file_part) for env_cat in output_all_names for file_part in bio_file_parts]
    results = Parallel(n_jobs=njobs)(  # 这里你可以改 n_jobs，比如 8 或 -1 用所有CPU
        delayed(xarrays_to_tifs)(env_cat, file_part, output_path, tif_dir, data, remove_negative=False, per_ha=False)
        for env_cat, file_part in tasks
    )

    agmgt_file_parts = ['area_agricultural_management']

    tasks = [(env_cat, file_part) for env_cat in input_files for file_part in agmgt_file_parts]
    results = Parallel(n_jobs=njobs)(  # 这里你可以改 n_jobs，比如 8 或 -1 用所有CPU
        delayed(xarrays_to_tifs_by_type)(
            env_cat, file_part, output_path, tif_dir, data,
            sum_dim='am'
        )
        for env_cat, file_part in tasks
    )

    nonag = ['area_non_agricultural_landuse']
    tasks = [(env_cat, file_part) for env_cat in input_files for file_part in nonag]
    results = Parallel(n_jobs=njobs)(  # 这里你可以改 n_jobs，比如 8 或 -1 用所有CPU
        delayed(xarrays_to_tifs_by_type)(
            env_cat, file_part, output_path, tif_dir, data,
            sum_dim='lu'
        )
        for env_cat, file_part in tasks
    )

    solution_cost_parts = ['cost_agricultural_management', 'cost_non_ag', 'transition_cost_ag2non_ag_amortised_diff']
    results = Parallel(n_jobs=njobs)(
        delayed(plus_tifs)(tif_dir, env_cat, solution_cost_parts, "total_sol_cost")
        for env_cat in output_all_names
    )

    solution_ghg_benefit_parts = ['GHG_ag_management', 'GHG_non_ag']
    results = Parallel(n_jobs=njobs)(
        delayed(plus_tifs)(tif_dir, env_cat, solution_ghg_benefit_parts, "total_sol_ghg_benefit", remove_negative=False)
        for env_cat in output_all_names
    )
    results = Parallel(n_jobs=njobs)(
        delayed(divide_tifs)(tif_dir, env_cat, solution_cost_parts, solution_ghg_benefit_parts, "carbon_sol_price")
        for env_cat in output_all_names
    )

    solution_bio_benefit_parts = ['biodiversity_GBF2_priority_ag_management', 'biodiversity_GBF2_priority_non_ag']
    results = Parallel(n_jobs=njobs)(
        delayed(plus_tifs)(tif_dir, env_cat, solution_bio_benefit_parts, "total_sol_bio_benefit", remove_negative=False)
        for env_cat in output_all_names
    )
    results = Parallel(n_jobs=njobs)(
        delayed(divide_tifs)(tif_dir, env_cat, solution_cost_parts, solution_bio_benefit_parts, "bio_sol_price")
        for env_cat in output_all_names
    )

    tif_path_1 = os.path.join(tif_dir, 'carbon_high', "xr_carbon_price_carbon_high_2050.tif")
    tif_path_2 = os.path.join(tif_dir, 'Counterfactual_carbon_high_bio_50',
                              f"xr_ghg_sol_price_Counterfactual_carbon_high_bio_50_2050.tif")
    tif_output = os.path.join(tif_dir, 'carbon_high_bio_50', f"xr_ghg_sol_price_carbon_high_bio_50_2050.tif")
    subtract_tifs(tif_path_2, tif_path_1, tif_output)
    #
    # # --- 阶段 8: shp计算 ---
    # tprint("\n--- 阶段 8: shp计算 ---")
    # shp_names = ['H_1kkm2', 'H_2kkm2', 'H_5kkm2', 'H_100km2']
    #
    # for shp_name in shp_names:
    #     if njobs == 0:
    #         for env_cat in output_all_names:
    #             create_shp(env_cat, shp_name, file_parts, tif_dir)
    #     else:
    #         Parallel(n_jobs=njobs)(
    #             delayed(create_shp)(env_cat, shp_name, file_parts, tif_dir)
    #             for env_cat in output_all_names
    #         )

    # --- 总结 ---
    end_time = time.time()
    total_time = end_time - start_time
    tprint("\n" + "=" * 80)
    tprint("所有任务已按顺序执行完毕")
    tprint(f"总执行时间: {total_time / 60 / 60:.2f} 小时 )")
    tprint("=" * 80)
    return

def run(task_dir, njobs):
    save_dir = os.path.join(task_dir, 'carbon_price')
    log_path = os.path.join(save_dir,'log_0_preprocess')
    @LogToFile(log_path)
    def _run():
        # Start recording memory usage
        stop_event = threading.Event()
        memory_thread = threading.Thread(target=log_memory_usage, args=(save_dir, 'a', 1, stop_event))
        memory_thread.start()

        try:
            print('\n')
            main(task_dir, njobs)
        except Exception as e:
            print(f"An error occurred during the simulation: {e}")
            raise e
        finally:
            # Ensure the memory logging thread is stopped
            stop_event.set()
            memory_thread.join()

    return _run()

if __name__ == "__main__":
    task_name = config.TASK_NAME
    njobs = math.ceil(41/1)
    task_dir = f'../../../output/{task_name}'

    run(task_dir, njobs)