from joblib import Parallel, delayed
import time
from tools.tools import get_path, get_year,save2nc
import shutil
import os
import xarray as xr
import numpy_financial as npf
import numpy as np
import glob
import traceback
import dask
import tempfile



import tools.config as config



def get_main_data_variable_name(ds: xr.Dataset) -> str:
    """自动从 xarray.Dataset 中获取唯一的数据变量名。"""
    data_vars_list = list(ds.data_vars)
    if len(data_vars_list) == 1:
        return data_vars_list[0]
    elif len(data_vars_list) == 0:
        raise ValueError("错误：数据集中不包含任何数据变量。")
    else:
        raise ValueError(f"错误：数据集中包含多个数据变量: {data_vars_list}。")


def amortize_costs(origin_path_name, target_path_name, amortize_file, years, njobs=0, rate=0.07, horizon=30):
    """
    【最终修复版 - 逐年输出】计算成本均摊，并为每一年生成一个累计成本文件。
    1. 使用 Dask 构建完整的计算图，计算出所有年份的累计摊销成本。
    2. 在保存阶段，通过循环和切片，为每一年单独触发计算并保存一个文件。
    """
    print(f"开始计算 '{amortize_file}' 的摊销成本... (逐年输出模式)")
    # --- 1. 数据加载与预处理 (与之前版本完全相同) ---
    file_paths = [os.path.join(origin_path_name, f'out_{year}', f'{amortize_file}_{year}.nc') for year in years]
    existing_files = [p for p in file_paths if os.path.exists(p)]
    if not existing_files: raise FileNotFoundError(
        f"在路径 {origin_path_name} 下找不到任何与 '{amortize_file}' 相关的文件。")
    valid_years = sorted([int(path.split('_')[-1].split('.')[0]) for path in existing_files])

    all_costs_ds = xr.open_mfdataset(
        existing_files,
        engine="h5netcdf",  # 推荐后端
        combine="nested",
        concat_dim="year",
        parallel=False,  # 关键：避免句柄并发问题
        # chunks={ "cell", "year": -1}  # year 整块、cell 分块
    ).assign_coords(year=valid_years)

    cost_variable_name = get_main_data_variable_name(all_costs_ds)
    pv_values_all_years = all_costs_ds[cost_variable_name]

    annual_payments = xr.apply_ufunc(
        lambda x: -1 * npf.pmt(rate, horizon, pv=x.astype(np.float64), fv=0, when='begin'),
        pv_values_all_years,
        dask="parallelized",
        output_dtypes=[np.float32],
    ).astype('float32')

    all_years = pv_values_all_years.year.values
    amortized_by_affect_year = {y: [] for y in all_years}

    # 计算每个 source_year 对哪些 affect_year 有影响
    for source_year in all_years:
        # 获取该年的年金数据，移除 year 坐标
        payment = annual_payments.sel(year=source_year).drop_vars('year')

        # 计算该 payment 影响到的年份区间
        for affect_year in all_years:
            if source_year <= affect_year <= source_year + horizon - 1:
                amortized_by_affect_year[affect_year].append(payment)

    # 关闭句柄
    all_costs_ds.close()
    print("✅ 所有源文件句柄已关闭。")

    # === 保存函数 ===
    def _save_one_year(y: int):
        try:
            if not amortized_by_affect_year[y]:
                return f"⚠️ 年份 {y} 无摊销内容，跳过"

            summed = xr.concat(amortized_by_affect_year[y], dim="temp_concat_dim").sum(dim="temp_concat_dim")
            summed = xr.DataArray(
                data=summed.values,  # 获取实际数据值
                coords=summed.coords,
                dims=summed.dims,
                name=summed.name
            )

            out_dir = os.path.join(target_path_name, f"{y}")
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, f"{amortize_file}_amortised_{y}.nc")
            print(f"  - 保存年份 {y} -> {out_path}")
            save2nc(summed, out_path)
            return f"✅ 年份 {y} 保存完成"
        except Exception as e:
            return f"❌ 年份 {y} 失败: {e}"

    # === 并行或顺序保存 ===
    if njobs and njobs > 0:
        results = Parallel(n_jobs=njobs, backend='threading')(
            delayed(_save_one_year)(y) for y in all_years
        )
        for msg in results:
            print(msg)
    else:
        for y in all_years:
            print(_save_one_year(y))
    print(f"\n✅ 任务完成: '{amortize_file}' 的所有年份摊销成本已成功计算并逐年保存。")


if __name__ == "__main__":
    # ============================================================================
    task_name = '20250823_Paper2_Results'
    njobs = 11
    task_dir =  f'../../../output/{task_name}'
    input_files = config.INPUT_FILES
    path_name_0 = get_path(task_name, input_files[0])
    path_name_1 = get_path(task_name, input_files[1])
    path_name_2 = get_path(task_name, input_files[2])
    years = get_year(path_name_0)
    run_names = [input_files[0], input_files[1], input_files[2]]
    run_paths = [path_name_0, path_name_1, path_name_2]
    amortize_files = ['xr_transition_cost_ag2non_ag']

    output_path = f'{task_dir}/carbon_price/0_base_data'
    # --- 第一批任务 (拆分为两个独立的组) ---
    # for i in range(3):
    for i in [2]:
        origin_path_name = get_path(task_name, input_files[i])
        target_path_name = os.path.join(output_path, input_files[i])
        amortize_costs(origin_path_name, target_path_name, amortize_files[0], years, njobs)