import xarray as xr
import os
import numpy_financial as npf
import time
from tools.tools import get_path, get_year,save2nc
import tools.config as config
import numpy as np

def get_main_data_variable_name(ds: xr.Dataset) -> str:
    """自动从 xarray.Dataset 中获取唯一的数据变量名。"""
    data_vars_list = list(ds.data_vars)
    if len(data_vars_list) == 1:
        return data_vars_list[0]
    elif len(data_vars_list) == 0:
        raise ValueError("错误：数据集中不包含任何数据变量。")
    else:
        raise ValueError(f"错误：数据集中包含多个数据变量: {data_vars_list}。")

task_name = '20250823_Paper2_Results_RES13'
njobs = 7
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
target_path_name = os.path.join(output_path, input_files[0])

rate=0.07
horizon=30

output_path = f'{task_dir}/carbon_price/0_base_data'

amortize_file = amortize_files[0]
origin_path_name = get_path(task_name, input_files[0])
file_paths = [os.path.join(origin_path_name, f'out_{year}', f'{amortize_file}_{year}.nc') for year in years]


existing_files = [p for p in file_paths if os.path.exists(p)]
if not existing_files: raise FileNotFoundError(
    f"在路径 {origin_path_name} 下找不到任何与 '{amortize_file}' 相关的文件。")
valid_years = sorted([int(path.split('_')[-1].split('.')[0]) for path in existing_files])

all_costs_ds = xr.open_mfdataset(
    existing_files,
    engine="h5netcdf",            # 推荐后端
    combine="nested",
    concat_dim="year",
    parallel=False,               # 关键：避免句柄并发问题
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

# 新结构：每年一个独立的 amortized cost 图层（提前准备好空容器）
amortized_by_year = {int(y): None for y in pv_values_all_years.year.values}
all_years = pv_values_all_years.year.values

print("  - 步骤2: 为每个成本年份分摊年金到对应年份...")

for cost_year in all_years:
    # 年金值（不含年份坐标）
    payment_from_this_year = annual_payments.sel(year=cost_year)
    start_year = cost_year
    end_year = cost_year + horizon - 1

    # 这笔年金分摊到的年份（在 all_years 范围内）
    affected_years = [y for y in all_years if start_year <= y <= end_year]

    for y in affected_years:
        # 创建一个只有当前年份 y 的 xarray，用于填入年金值
        y_da = xr.zeros_like(pv_values_all_years.sel(year=[y]), dtype=np.float32)
        y_da.loc[dict(year=[y])] = payment_from_this_year
        # print(f"Year {y} after assignment: min={y_da.min().values}, max={y_da.max().values}")
        if amortized_by_year[cost_year] is None:
            amortized_by_year[cost_year] = y_da
        else:
            amortized_by_year[cost_year] = amortized_by_year[cost_year] + y_da  # dask 会合并任务图

# 合并所有年份的结果（list → concat）
print("  - 步骤3: 拼接所有年份的摊销结果...")

amortized_list = [amortized_by_year[y] for y in all_years if amortized_by_year[y] is not None]
total_amortized_costs = xr.concat(amortized_list, dim='year')

# 确保年份顺序一致
total_amortized_costs = total_amortized_costs.sortby('year')
total_amortized_costs.name = 'data'
print("starting to compute...")
total_amortized_costs = total_amortized_costs.chunk({'year': 1, 'cell': 1024}).compute()
print("✅ 所有年份的摊销计算已完成。")

# 关闭句柄
all_costs_ds.close()
print("✅ 所有源文件句柄已关闭。")