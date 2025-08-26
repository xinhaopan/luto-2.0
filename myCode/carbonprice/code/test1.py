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
annual_payments.name = 'annual_payment'
annual_payments = annual_payments.chunk({'year': 1})

# 2. 准备一个列表来收集所有独立的“摊销计划”
payment_schedules = []
all_years = pv_values_all_years.year.values

print("  - 步骤2: 为每个成本年份创建独立的摊销计划...")
for cost_year in all_years:
    # 提取这一年产生的年金值
    payment_from_this_year = annual_payments.sel(year=cost_year)

    # 确定该年金影响的年份范围
    start_year = cost_year
    end_year = cost_year + horizon - 1
    affected_years = [y for y in all_years if start_year <= y <= end_year]

    if not affected_years:
        continue

    # 创建一个全尺寸的零数组作为当前计划的模板
    # 使用 .data 可以避免继承不必要的坐标
    schedule_template = xr.zeros_like(pv_values_all_years.sel(year=affected_years), dtype=np.float32)

    # 将年金值填充到受影响的年份
    # .loc 在这里是安全的，因为它作用于一个临时的、非Dask累加的变量上
    schedule_template.loc[dict(year=affected_years)] = payment_from_this_year.drop_vars(
        'year')  # drop_vars避免坐标冲突

    # 将这个构建好的、独立的摊销计划（一个Dask数组）添加到列表中
    payment_schedules.append(schedule_template)

# 3. 对列表中的所有摊销计划求和
#    这是整个计算的核心，Dask会在这里构建一个高效的、宽阔的求和图
print("  - 步骤3: 对所有摊销计划进行求和以构建最终计算图...")
if not payment_schedules:
    # 如果列表为空，返回一个零数组
    total_amortized_costs = xr.zeros_like(pv_values_all_years, dtype=np.float32)
else:
    # sum() 是将列表中的 xarray 对象相加的最直接方式
    total_amortized_costs = sum(payment_schedules)

# 2. 【关键修复 - 第二步】
# 手动关闭所有源文件，彻底释放文件句柄
all_costs_ds.close()
print("✅ 所有源文件句柄已关闭。")

sorted_years = sorted(total_amortized_costs.year.values.tolist())
for y in sorted_years:
    out_dir = os.path.join(target_path_name, f"{valid_years[y]}")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{amortize_file}_amortised_{valid_years[y]}.nc")
    print(f"  - 正在处理年份 {y} -> {out_path}")

    # 明确使用 .sel 来选择该年的 amortized cost
    da_y = total_amortized_costs.sel(year=y)

    save2nc(da_y, out_path)


