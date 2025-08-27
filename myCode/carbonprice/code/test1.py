import xarray as xr
import os
import numpy_financial as npf
import time
from tools.tools import get_path, get_year, save2nc
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
task_dir = f'../../../output/{task_name}'
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

rate = 0.07
horizon = 30

output_path = f'{task_dir}/carbon_price/0_base_data'

amortize_file = amortize_files[0]
origin_path_name = get_path(task_name, input_files[0])
file_paths = [os.path.join(origin_path_name, f'out_{year}', f'{amortize_file}_{year}.nc') for year in years]

print(f"正在处理文件: {amortize_file}，路径: {origin_path_name}")
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

all_years = annual_payments.year.values  # e.g., np.arange(2010, 2051)
base_shape = annual_payments.sel(year=all_years[0]).drop_vars('year').shape
n_years = len(all_years)
# 初始化 numpy array，用于累加所有影响
amortized_matrix = np.zeros((n_years,) + base_shape, dtype=np.float32)


for source_year in all_years:
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
    name='data'
)

print("start compute...")
amortized_by_affect_year.compute()
print("compute done.")
# 关闭句柄
all_costs_ds.close()
print("✅ 所有源文件句柄已关闭。")
for y in all_years:
    out_dir = os.path.join(target_path_name, f"{y}")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{amortize_file}_amortised_{y}.nc")
    print(f"  - 保存年份 {y} -> {out_path}")
    da_y = amortized_by_affect_year.sel(year=y)
    ds_y = xr.Dataset({'data': da_y})
    save2nc(ds_y, out_path)