from joblib import Parallel, delayed
import time
from tools.tools import get_path, get_year
import shutil
import os
import xarray as xr
import numpy_financial as npf
import numpy as np
import glob

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



import traceback


def amortize_costs(origin_path_name, target_path_name, amortize_file, years, check_files=False, rate=0.07, horizon=30):
    """
    【最终决定版】计算成本均摊。
    1. 修正了 `npf.pmt` 导致 xarray 维度信息丢失的问题。
    2. 使用 lambda 包装 `np.einsum` 调用，彻底解决 apply_ufunc 的参数混淆问题，保证在任何旧版本 xarray 中都能正确执行。
    """
    print(f"开始计算 '{amortize_file}' 的摊销成本...")

    # --- 1. 数据加载与预处理 ---
    # ... (这部分代码无需修改，保持原样) ...
    file_paths = [os.path.join(origin_path_name, f'out_{year}', f'{amortize_file}_{year}.nc') for year in years]
    try:
        # ... (此处省略了与之前版本相同的、完整的加载逻辑) ...
        # 为了简洁，此处使用快速加载逻辑作为代表
        print("模式：快速加载已开启（无深度文件检查）。")
        existing_files = [p for p in file_paths if os.path.exists(p)]
        if not existing_files: raise FileNotFoundError(
            f"在路径 {origin_path_name} 下找不到任何与 '{amortize_file}' 相关的文件。")
        valid_years = sorted([int(path.split('_')[-1].split('.')[0]) for path in existing_files])
        all_costs_ds = xr.open_mfdataset(
            existing_files, combine='nested', concat_dim='year', parallel=True,
        ).assign_coords(year=valid_years)
    except Exception as e:
        print(f"❌ 数据加载失败: {e}")
        return

    # --- 2. 核心计算逻辑 (最终决定性方案) ---
    print("\n数据加载完毕，开始核心计算...")
    print(f"数据维度信息: {all_costs_ds.dims}")

    try:
        cost_variable_name = get_main_data_variable_name(all_costs_ds)
        pv_values_all_years = all_costs_ds[cost_variable_name]

        print(f"原始数据 ({cost_variable_name}) 形状: {pv_values_all_years.shape}")

        # 计算年度支付额，并立即将其重新包装为 xarray.DataArray 以保留维度信息
        annual_payments_np = -1 * npf.pmt(rate, horizon, pv=pv_values_all_years.data, fv=0, when='begin')
        annual_payments = xr.DataArray(
            annual_payments_np,
            dims=pv_values_all_years.dims,
            coords=pv_values_all_years.coords
        )
        print(f"恢复后的年度支付额 (annual_payments) 形状: {annual_payments.shape}")

        # 创建影响矩阵
        n_years = len(valid_years)
        influence_matrix = np.zeros((n_years, n_years))
        for i in range(n_years):
            influence_matrix[i, i:min(i + horizon, n_years)] = 1
        influence_da = xr.DataArray(
            influence_matrix,
            dims=['year', 'target_year'],
            coords={'year': valid_years, 'target_year': valid_years}
        )

        # ✅✅✅ 【最终决定性修复】 ✅✅✅
        # 使用 lambda 函数包装 np.einsum 调用。
        # 这样 apply_ufunc 看到的就是一个清晰的、只接收两个参数(x, y)的函数。
        # 它不再会对参数数量感到困惑。
        total_amortized_costs = xr.apply_ufunc(
            lambda x, y: np.einsum('...y,yz->...z', x, y),  # 包装后的函数
            annual_payments,  # 第1个数据参数 (对应 x)
            influence_da,  # 第2个数据参数 (对应 y)
            input_core_dims=[['year'], ['year', 'target_year']],
            output_core_dims=[['target_year']],
            keep_attrs=True
        )

        total_amortized_costs = total_amortized_costs.rename({'target_year': 'year'})
        print(f"最终结果 (total_amortized_costs) 形状: {total_amortized_costs.shape}")

    except Exception as e:
        print(f"❌ 在核心计算环节发生错误: {e}")
        traceback.print_exc()
        return

    # --- 3. 并行保存结果 ---
    # ... (这部分代码无需修改，保持原样) ...
    print("\n计算完成，正在并行保存结果...")
    result_ds = total_amortized_costs.to_dataset(name=f"data")

    def save_year_slice(year_to_save):
        try:
            target_year_path = os.path.join(target_path_name, str(year_to_save))
            os.makedirs(target_year_path, exist_ok=True)
            dst_file = os.path.join(target_year_path, f'{amortize_file}_amortised_{year_to_save}.nc')
            result_ds.sel(year=year_to_save).copy(deep=True).to_netcdf(dst_file)
            return f"  - ✅ 已保存: {os.path.basename(dst_file)}"
        except Exception as e:
            return f"  - ❌ 保存失败 (Year {year_to_save}): {e}"

    save_results = Parallel(n_jobs=-1)(delayed(save_year_slice)(year) for year in valid_years)
    for res in save_results:
        print(res)

    print(f"\n✅ 任务完成: '{amortize_file}' 的所有年份摊销成本已成功计算并保存。")


# --- 辅助函数：专门用于计算单个文件对的差异，以便并行化 ---
def calculate_and_save_single_diff(diff_file, year, year_path, year_path_0, target_year_path, files_without_year):
    """
    计算并保存单个文件对的差异。
    这个函数将被并行调用。
    """
    try:
        # 1. 构造上一年度和当前年度的文件路径
        if diff_file in files_without_year:
            src_file_0 = os.path.join(year_path_0, diff_file)
            src_file_1 = os.path.join(year_path, diff_file)
        else:
            src_file_0 = os.path.join(year_path_0, f"{diff_file}_{year - 1}.nc")
            src_file_1 = os.path.join(year_path, f"{diff_file}_{year}.nc")

        # 2. 打开这对文件
        with xr.open_dataset(src_file_0) as ds_0, xr.open_dataset(src_file_1) as ds_1:
            # 3. 计算差异
            ds_res = ds_1 - ds_0

            # 4. 构造目标路径并保存
            variable_name = diff_file.replace('.nc', '')
            dst_filename = f"{variable_name}_diff_{year}.nc"
            if diff_file in files_without_year:
                dst_filename = f"{variable_name}_diff_{year}.nc"
            dst_file = os.path.join(target_year_path, dst_filename)

            ds_res.to_netcdf(dst_file)
            return f"  - Success: Calculated and saved diff for {dst_filename}"

    except FileNotFoundError:
        return f"  - ❌ Error (File Not Found): Could not calculate diff for '{target_year_path} {diff_file}'. One of the source files is missing."
    except Exception as e:
        # 明确指出是哪个文件对出了问题
        return f"  - ❌ Error (Calculation Failed): Could not process '{diff_file}'. Reason: {e}"


def copy_single_file(src_file, dst_file):
    """健壮地复制单个文件，如果源文件不存在则跳过并警告。"""
    try:
        os.makedirs(os.path.dirname(dst_file), exist_ok=True)
        shutil.copy(src_file, dst_file)
        return f"Copied {os.path.basename(src_file)}"
    except FileNotFoundError:
        return f"Warning: Source file not found, skipping copy: {src_file}"
    except Exception as e:
        return f"Error copying {src_file}: {e}"


# --- 辅助函数：专门用于计算单个文件对的差异，以便并行化 ---

def process_single_year(year, years, origin_path_name, target_path_name, copy_files, diff_files):
    """【最终正确版】并行处理文件复制和差异计算。"""
    year_path = os.path.join(origin_path_name, f'out_{year}')
    target_year_path = os.path.join(target_path_name, str(year))
    os.makedirs(target_year_path, exist_ok=True)

    # --- 1. 并行化文件复制 (逻辑不变) ---
    copy_tasks = [
        delayed(copy_single_file)(
            os.path.join(year_path, f'{f}_{year}.nc'),
            os.path.join(target_year_path, f'{f}_{year}.nc')
        )
        for f in copy_files if not (year == 2010 and f == 'xr_transition_GHG')
    ]
    if copy_tasks:
        print(f"\nYear {year} Copy Tasks:")
        copy_results = Parallel(n_jobs=-1)(copy_tasks)
        for res in copy_results:
            print(res)

    # --- 2. 【正确地】并行化差异计算 ---
    if year > years[0] and diff_files:

        # --- 1. 前置条件检查 ---
        # 使用 if/else 结构替代 continue。
        # 只有当 "跳过" 条件不满足时，才执行主要的逻辑。
        if input_files[0] in target_year_path:
            print(f"信息 (Year {year}): 检测到 '{input_files[0]}' 路径，按规则跳过所有差异计算。")
        else:
            # --- 主要逻辑块开始 ---
            print(f"\nYear {year}: 开始准备差异计算任务...")

            # --- 2. 确定当前年份需要处理的文件列表 ---
            # 创建一个副本进行操作，以避免修改原始的 diff_files 集合
            diff_files_for_this_run = diff_files.copy()

            # 根据特例规则，从副本中移除特定文件
            if input_files[1] in target_year_path:
                diff_files_for_this_run.remove('xr_biodiversity_GBF2_priority_ag')
                print(f"  - 规则应用：从任务列表移除了 'xr_biodiversity_GBF2_priority_ag' (由于路径匹配 '{input_files[1]}')。")

            # --- 3. 检查是否有任务需要执行 ---
            # 再次使用 if/else 结构替代 continue。
            # 只有当任务列表不为空时，才执行创建和运行的步骤。
            if not diff_files_for_this_run:
                print(f"  - 信息：经过筛选后，没有需要计算差异的文件。")
            else:
                # --- 任务创建与执行块开始 ---
                year_path_0 = os.path.join(origin_path_name, f'out_{year - 1}')
                files_without_year = {'xr_GHG_ag.nc'}

                print(f"  - 正在为 {len(diff_files_for_this_run)} 个文件创建并行任务...")

                # 为筛选后的文件列表创建独立的计算任务
                diff_tasks = [
                    delayed(calculate_and_save_single_diff)(
                        diff_file, year, year_path, year_path_0, target_year_path, files_without_year
                    )
                    for diff_file in diff_files_for_this_run
                ]

                # 并行执行所有差异计算任务
                diff_results = Parallel(n_jobs=-1)(diff_tasks)

                # 打印所有任务的结果
                print(f"Year {year}: 所有差异计算任务已完成。结果如下：")
                for res in diff_results:
                    print(res)

    return f"Year {year} processed."


# ==============================================================================
# STAGE 1: 计算利润 (Profit = Revenue - Cost)
# ==============================================================================
def calculate_profit_for_run(year, run_path,run_name, cost_basename, revenue_basename):
    """
    为单个情景(Run)和单个类别计算利润。
    """
    try:
        # 构建输入文件路径
        cost_file = os.path.join(run_path, run_name, str(year), f'{cost_basename}_{year}.nc')
        revenue_file = os.path.join(run_path, run_name, str(year), f'{revenue_basename}_{year}.nc')

        # 使用 with 语句确保文件正确关闭
        with xr.open_dataset(cost_file) as ds_cost, \
                xr.open_dataset(revenue_file) as ds_revenue:

            if cost_basename == 'xr_cost_ag':
                profit = ds_cost['data'].sum(dim='source') - ds_revenue['data'].sum(dim='source')
            # 计算利润
            else:
                profit = ds_revenue - ds_cost

            # 构建输出路径并保存
            # 我们将利润文件保存在一个专门的 'profit' 子目录中，以保持整洁
            profit_run_path = os.path.join(run_path, run_name, str(year))
            os.makedirs(profit_run_path, exist_ok=True)

            # 为了区分，我们给文件名加上 profit 前缀
            profit_filename = f'xr_profit_{cost_basename.replace("xr_cost_", "")}_{year}.nc'
            profit.to_netcdf(os.path.join(profit_run_path, profit_filename))

            return f"✅ Profit: Calculated for {os.path.basename(run_path)}/{profit_filename}"

    except FileNotFoundError as e:
        return f"❌ Profit Error (File Not Found) for year {year}: {e.filename}"
    except Exception as e:
        return f"❌ Profit Error (Calculation Failed) for year {year}: {e}"


# ==============================================================================
# STAGE 2: 基于利润差计算 Carbon 和 Bio 成本
# ==============================================================================
def calculate_policy_cost(year, output_path, run_names, cost_category, policy_type):
    """
    基于利润差计算政策成本 (Carbon 或 Bio)。
    policy_type: 'carbon' 或 'bio'
    """
    try:
        if policy_type == 'carbon':
            # Carbon Cost = Profit_Run0 - Profit_Run1
            run_A_name, run_B_name = run_names[0], run_names[1]
            output_dir = os.path.join(output_path, 'carbon_cost', str(year))
        elif policy_type == 'bio':
            # Bio Cost = Profit_Run1 - Profit_Run2
            run_A_name, run_B_name = run_names[1], run_names[2]
            output_dir = os.path.join(output_path, 'bio_cost', str(year))
        else:
            return f"❌ Policy Error: Invalid policy_type '{policy_type}'"

        # 构建利润文件路径 (由 Stage 1 生成)
        profit_file_basename = f'xr_profit_{cost_category}_{year}.nc'
        profit_file_A = os.path.join(output_path, run_A_name, str(year), profit_file_basename)
        profit_file_B = os.path.join(output_path, run_B_name, str(year), profit_file_basename)

        with xr.open_dataset(profit_file_A) as ds_profit_A, \
                xr.open_dataset(profit_file_B) as ds_profit_B:

            policy_cost = ds_profit_A - ds_profit_B

            os.makedirs(output_dir, exist_ok=True)
            output_filename = f'xr_cost_{cost_category}_{policy_type}_{year}.nc'
            policy_cost.to_netcdf(os.path.join(output_dir, output_filename))

            return f"✅ Policy Cost: Calculated {policy_type}/{cost_category} for year {year}"

    except FileNotFoundError as e:
        return f"❌ Policy Cost Error (File Not Found) for {policy_type}/{cost_category} year {year}: {e.filename}"
    except Exception as e:
        return f"❌ Policy Cost Error (Calculation Failed) for {policy_type}/{cost_category} year {year}: {e}"


# ==============================================================================
# STAGE 3: 计算转型成本 (Transition Cost) 的差值
# ==============================================================================
def calculate_transition_cost_diff(year, output_path, run_names, tran_cost_file, policy_type):
    """
    【全新逻辑】计算转型成本文件的差值，并区分 Carbon 和 Bio。
    - Carbon: Run1 - Run0
    - Bio: Run2 - Run1
    """
    try:
        # 根据 policy_type 选择正确的路径对和输出目录
        if policy_type == 'carbon':
            path_A = os.path.join(output_path, run_names[0], str(year), f'{tran_cost_file}_{year}.nc')  # Run0
            path_B = os.path.join(output_path, run_names[1], str(year), f'{tran_cost_file}_{year}.nc')  # Run1
            output_subdir = 'carbon_cost'
        elif policy_type == 'bio':
            path_A = os.path.join(output_path, run_names[1], str(year), f'{tran_cost_file}_{year}.nc')  # Run1
            path_B = os.path.join(output_path, run_names[2], str(year), f'{tran_cost_file}_{year}.nc')  # Run2
            output_subdir = 'bio_cost'
        else:
            return f"❌ TranCost Diff Error: Invalid policy_type '{policy_type}'"

        # 使用 with 语句确保文件正确关闭
        with xr.open_dataset(path_B) as ds_B, xr.open_dataset(path_A) as ds_A:
            # 执行减法 (B - A)
            tran_cost_diff = ds_B - ds_A

        # 构建输出路径并保存
        output_dir = os.path.join(output_path, output_subdir, str(year))
        os.makedirs(output_dir, exist_ok=True)
        # 在文件名中也加入 policy_type 以便区分
        output_filename = f'{tran_cost_file}_diff_{policy_type}_{year}.nc'
        tran_cost_diff.to_netcdf(os.path.join(output_dir, output_filename))

        return f"✅ TranCost Diff: Calculated {policy_type} for {tran_cost_file} year {year}"

    except FileNotFoundError as e:
        return f"❌ TranCost Diff Error (File Not Found) for {policy_type}/{tran_cost_file} year {year}: {e.filename}"
    except Exception as e:
        return f"❌ TranCost Diff Error (Calculation Failed) for {policy_type}/{tran_cost_file} year {year}: {e}"


def aggregate_and_save_cost(year, output_path,cost_type):
    """
    【最终版】聚合单个年份的成本文件，使用一个精确的文件列表。
    """
    # 1. 【关键修改】根据传入的列表构建完整的文件路径
    data_type_names = ['xr_cost_ag', 'xr_cost_agricultural_management', 'xr_cost_non_ag', 'xr_cost_transition_ag2ag_diff']
    data_type_names_adds = ['xr_transition_cost_ag2non_ag_amortised_diff','xr_transition_cost_ag2non_ag_diff']
    for data_type_names_add in data_type_names_adds:
        data_type_names_all = data_type_names + [f"{data_type_names_add}"]
        file_dir = os.path.join(output_path, f'{cost_type}_cost', str(year))
        # `files_to_process` 包含的是 'fileA', 'fileB' 这样的基础名

        full_file_paths = [
            os.path.join(file_dir, f'{basename}_{cost_type}_{year}.nc')
            for basename in data_type_names_all
        ]

        # 2. 初始化累加器
        total_sum_ds = None

        # 3. 循环处理每一个文件
        for file_path in full_file_paths:
            try:
                with xr.open_dataset(file_path) as ds:
                    summed_single_ds = ds.sum(dim=[d for d in ds.dims if d != 'cell'])
                    if total_sum_ds is None:
                        total_sum_ds = summed_single_ds
                    else:
                        total_sum_ds += summed_single_ds
            except FileNotFoundError:
                # 如果列表中的某个文件不存在，打印警告并跳过
                print(f"  - Warning: File not found and will be skipped: {file_path}")
                continue

        # 5. 保存
        if data_type_names_add == 'xr_transition_cost_ag2non_ag_amortised':
            am_type = 'amortised'
        elif data_type_names_add == 'xr_transition_cost_ag2non_ag':
            am_type = 'original'
        final_path = os.path.join(file_dir, f'xr_total_cost_{cost_type}_{am_type}_{year}.nc')

        total_sum_ds.to_netcdf(final_path)


def aggregate_and_save_summary(year, output_path, data_type,data_type_names, input_files_name):
    """
    【最终版】聚合单个年份的成本文件，使用一个精确的文件列表。
    """
    try:
        # 1. 【关键修改】根据传入的列表构建完整的文件路径
        file_dir = os.path.join(output_path, f'{input_files_name}', str(year))
        # `files_to_process` 包含的是 'fileA', 'fileB' 这样的基础名

        full_file_paths = [
            os.path.join(file_dir, f'{basename}_{year}.nc')
            for basename in data_type_names
        ]

        if not full_file_paths:
            return f"⚠️ Stage 2 Warning: An empty list of '{data_type}' files was provided for year {year}. Skipping."

        # 2. 初始化累加器
        total_sum_ds = None

        # 3. 循环处理每一个文件
        for file_path in full_file_paths:
            try:
                with xr.open_dataset(file_path) as ds:
                    summed_single_ds = ds.sum(dim=[d for d in ds.dims if d != 'cell'])
                    if total_sum_ds is None:
                        total_sum_ds = summed_single_ds
                    else:
                        total_sum_ds += summed_single_ds
            except FileNotFoundError:
                # 如果列表中的某个文件不存在，打印警告并跳过
                print(f"  - Warning: File not found and will be skipped: {file_path}")
                continue

        # 检查是否处理了任何文件
        if total_sum_ds is None:
            return f"❌ Stage 2 Error: None of the specified '{data_type}' files could be found or processed for year {year}."

        # 4. 后续处理
        if data_type == 'carbon':
            total_sum_ds = -total_sum_ds

        # 5. 保存
        final_dir = os.path.join(output_path, data_type, str(year))
        os.makedirs(final_dir, exist_ok=True)
        final_path = os.path.join(final_dir, f'{data_type}_{year}.nc')

        total_sum_ds.to_netcdf(final_path)

        return f"✅ Stage 2: Aggregated {len(full_file_paths)} '{data_type}' files for year {year} and saved to {final_path}."

    except Exception as e:
        return f"❌ Stage 2 Error (Aggregation Failed) for '{data_type}' year {year}: {e}"

def process_all_years_serially(years, origin_path_name, target_path_name, copy_files, diff_files):
    """
    辅助函数：按顺序处理所有年份。
    这个函数本身是串行的，因为它内部的年份之间有依赖关系。
    """
    print("信息：开始串行处理所有年份 (process_all_years_serially)...")
    try:
        for year in years:
            # 您原来的 process_single_year 调用保持不变
            process_single_year(year, years, origin_path_name, target_path_name, copy_files, diff_files)
        print("✅ 成功：所有年份已串行处理完毕。")
        return "process_all_years_serially: OK" # 返回成功状态
    except Exception as e:
        print(f"❌ 错误：在串行处理年份时发生错误: {e}")
        import traceback
        traceback.print_exc()
        return f"process_all_years_serially: FAILED with {e}" # 返回失败状态


# ============================================================================
print(config.TASK_DIR)
input_files = config.INPUT_FILES
path_name_0 = get_path(input_files[0])
path_name_1 = get_path(input_files[1])
path_name_2 = get_path(input_files[2])
years = get_year(path_name_0)
run_names = [input_files[0], input_files[1], input_files[2]]
run_paths = [path_name_0, path_name_1, path_name_2]

output_path = f'{config.TASK_DIR}/carbon_price/0_base_data'
os.makedirs(output_path, exist_ok=True)

copy_files = ['xr_biodiversity_GBF2_priority_ag_management','xr_biodiversity_GBF2_priority_non_ag','xr_cost_ag','xr_cost_agricultural_management','xr_cost_non_ag','xr_cost_transition_ag2ag','xr_GHG_ag_management','xr_GHG_non_ag','xr_revenue_ag','xr_revenue_agricultural_management','xr_revenue_non_ag','xr_transition_cost_ag2non_ag','xr_transition_GHG']
amortize_files = ['xr_transition_cost_ag2non_ag']
diff_files = ['xr_biodiversity_GBF2_priority_ag','xr_GHG_ag.nc']
cost_files = ['xr_cost_ag', 'xr_cost_agricultural_management', 'xr_cost_non_ag']
revenue_files = ['xr_revenue_ag', 'xr_revenue_agricultural_management', 'xr_revenue_non_ag']

# ============================================================================
# 第一批：数据预处理阶段 (摊销成本计算 + 文件复制/差异计算)
# ============================================================================
print("\n--- 第一批：数据预处理阶段 ---")
print("这一批包含摊销成本计算和基础文件处理，它们之间可以并行执行")

# ----------------------------------------------------------------------------
# 步骤 1: 准备所有任务清单 (先不执行，只定义)
# ----------------------------------------------------------------------------

# --- 第一批任务 (拆分为两个独立的组) ---
amortization_tasks = []
for i in range(3):
    origin_path_name = get_path(input_files[i])
    target_path_name = os.path.join(output_path, input_files[i])
    amortization_tasks.append(
        delayed(amortize_costs)(origin_path_name, target_path_name, amortize_files[0], years)
    )
print(f"第一批摊销成本计算任务数量: {len(amortization_tasks)}")
file_processing_tasks = []
for i in range(3):
    origin_path_name = get_path(input_files[i])
    target_path_name = os.path.join(output_path, input_files[i])
    file_processing_tasks.append(
        delayed(process_all_years_serially)(years, origin_path_name, target_path_name, copy_files, diff_files)
    )
print(f"第一批文件处理任务数量: {len(file_processing_tasks)}")
# ============================================================================
# 第二批：独立任务 (汇总计算 - 不依赖其他计算结果)
# ============================================================================
print("\n--- 第二批：独立汇总计算 ---")

batch_2_tasks = []

# STAGE 5: 汇总计算任务（不依赖摊销成本和利润计算）
print("添加汇总计算任务...")
for year in years:
    batch_2_tasks.append(
        delayed(aggregate_and_save_summary)(year, output_path, 'carbon',
                                          ['xr_GHG_ag_diff', 'xr_GHG_ag_management','xr_GHG_non_ag','xr_transition_GHG'],
                                          input_files[1])
    )
    batch_2_tasks.append(
        delayed(aggregate_and_save_summary)(year, output_path, 'bio',
                                          ['xr_biodiversity_GBF2_priority_ag_diff', 'xr_biodiversity_GBF2_priority_ag_management','xr_biodiversity_GBF2_priority_non_ag'],
                                          input_files[2])
    )

print(f"第二批任务数量: {len(batch_2_tasks)}")


# ============================================================================
# 第三批：利润计算 (STAGE 1)
# ============================================================================
print("\n--- 第三批：利润计算 ---")
print("计算所有情景的利润，为后续政策成本计算提供基础")

batch_3_tasks = []
profit_categories = zip(cost_files, revenue_files)
for cost_base, rev_base in profit_categories:
    for i, run_name in enumerate(run_names):
        for year in years:
            batch_3_tasks.append(
                delayed(calculate_profit_for_run)(year, output_path, run_name, cost_base, rev_base)
            )

print(f"第三批任务数量: {len(batch_3_tasks)}")

# ============================================================================
# 第四批：政策成本计算 (STAGE 2) - 依赖第三批的利润结果
# ============================================================================
print("\n--- 第四批：政策成本计算 ---")
print("基于利润差异计算政策成本 (Carbon & Bio)")

batch_4_tasks = []
policy_cost_categories = [f.replace('xr_cost_', '') for f in cost_files]
for category in policy_cost_categories:
    for year in years:
        batch_4_tasks.append(
            delayed(calculate_policy_cost)(year, output_path, run_names, category, 'carbon')
        )
        batch_4_tasks.append(
            delayed(calculate_policy_cost)(year, output_path, run_names, category, 'bio')
        )

print(f"第四批任务数量: {len(batch_4_tasks)}")

# ============================================================================
# 第五批：转型成本差值计算 (部分依赖摊销，部分独立)
# ============================================================================
print("\n--- 第五批：转型成本差值计算 ---")
print("计算转型成本文件的差值 - 部分任务需要等待摊销完成")

# 分离独立任务和依赖任务
batch_5a_tasks = []  # 不依赖摊销的任务
batch_5b_tasks = []  # 依赖摊销的任务

independent_tran_files = ['xr_cost_transition_ag2ag', 'xr_transition_cost_ag2non_ag']
dependent_tran_files = ['xr_transition_cost_ag2non_ag_amortised']

# 独立任务：可以与第四批并行执行
for tran_file in independent_tran_files:
    for year in years:
        batch_5a_tasks.append(
            delayed(calculate_transition_cost_diff)(year, output_path, run_names, tran_file, 'carbon')
        )
        batch_5a_tasks.append(
            delayed(calculate_transition_cost_diff)(year, output_path, run_names, tran_file, 'bio')
        )

print(f"第五批A (独立任务) 数量: {len(batch_5a_tasks)}")

# 依赖任务：必须等待摊销完成
for tran_file in dependent_tran_files:
    for year in years:
        batch_5b_tasks.append(
            delayed(calculate_transition_cost_diff)(year, output_path, run_names, tran_file, 'carbon')
        )
        batch_5b_tasks.append(
            delayed(calculate_transition_cost_diff)(year, output_path, run_names, tran_file, 'bio')
        )

print(f"第五批B (依赖摊销) 任务数量: {len(batch_5b_tasks)}")


# ============================================================================
# 第六批：成本聚合 (依赖第五批所有转型成本差值计算完成)
# ============================================================================
print("\n--- 第六批：成本聚合阶段 ---")
print("这一批需要等待:")
print("  1. 第一批的摊销成本计算完成")
print("  2. 第五批的所有转型成本差值计算完成")
print("现在开始最终的成本聚合...")

batch_6_tasks = []
for year in years:
    batch_6_tasks.append(
        delayed(aggregate_and_save_cost)(year, output_path, 'carbon')
    )
    batch_6_tasks.append(
        delayed(aggregate_and_save_cost)(year, output_path, 'bio')
    )

print(f"第六批任务数量: {len(batch_6_tasks)}")

# ============================================================================
# 总结
# ============================================================================
start_time = time.time()
print("=" * 80)
print("开始终极优化的并行处理管道 (基于DAG依赖分析)")
print("=" * 80)

# 使用 with 语句创建一个持久的并行“工人池”
with Parallel(n_jobs=-1) as parallel:
    # --- 阶段 1: 并行启动所有没有依赖的基础任务 ---
    print("\n--- 阶段 1: 启动所有基础任务 ---")

    # 提交【摊销任务】到后台执行，并获得一个“未来”对象
    amortization_future = parallel(amortization_tasks)
    print(f"✅ {len(amortization_tasks)}个【摊销任务】已提交到后台...")

    # 提交【文件处理任务】到后台执行，并获得另一个“未来”对象
    file_processing_future = parallel(file_processing_tasks)
    print(f"✅ {len(file_processing_tasks)}个【文件处理任务】已提交到后台...")

    # --- 阶段 2: 等待文件处理完成，然后启动所有仅依赖它的任务 ---
    print("\n--- 阶段 2: 等待文件处理，并启动后续独立任务 ---")
    print("⏳ [等待点] 正在等待文件处理任务完成...")

    # 访问 file_processing_future 会导致程序在此阻塞，直到任务完成
    file_processing_results = file_processing_future
    print("✅ 文件处理任务完成!")

    # 将所有仅依赖“文件处理”的后续任务（第2、3、5A批）合并成一个大列表
    # 这样 joblib 可以在它们之间进行最大程度的并行优化
    independent_tasks_group = batch_2_tasks + batch_3_tasks + batch_5a_tasks
    print(f"🚀 启动 {len(independent_tasks_group)} 个后续独立任务 (第2, 3, 5A批)...")
    independent_tasks_future = parallel(independent_tasks_group)

    # --- 阶段 3: 并行处理两个独立的依赖分支 ---
    print("\n--- 阶段 3: 处理并行的依赖分支 ---")

    # 分支 A: 处理第四批（它依赖第三批，即 independent_tasks_group）
    print("⏳ [等待点] 等待第3批(利润计算)等任务完成...")
    independent_tasks_results = independent_tasks_future  # 等待第2,3,5A批完成
    print("✅ 第2, 3, 5A批任务完成!")

    print(f"🚀 启动第4批 ({len(batch_4_tasks)}个政策成本任务)...")
    batch_4_future = parallel(batch_4_tasks)

    # 分支 B: 处理第五批B（它依赖第一批的摊销任务）
    # 注意：这个等待可以和上面的任务并行发生
    print("⏳ [等待点] 等待第1批(摊销)任务完成 (可能早已在后台完成)...")
    amortization_results = amortization_future  # 等待摊销任务完成
    print("✅ 摊销任务完成!")

    print(f"🚀 启动第5批B ({len(batch_5b_tasks)}个依赖摊销的任务)...")
    batch_5b_future = parallel(batch_5b_tasks)

    # --- 阶段 4: 最终聚合，等待所有分支完成 ---
    print("\n--- 阶段 4: 最终聚合 ---")
    print("⏳ [最终等待点] 等待所有计算完成...")

    # 等待第四批和第五批B全部完成
    batch_4_results = batch_4_future
    batch_5b_results = batch_5b_future
    print("✅ 所有依赖计算均已完成!")

    print(f"🚀 启动最终的第6批 ({len(batch_6_tasks)}个成本聚合任务)...")
    final_results = parallel(batch_6_tasks)
    print("✅ 最终聚合任务完成!")

# ============================================================================
# 总结
# ============================================================================
end_time = time.time()
total_time = end_time - start_time

print("\n" + "=" * 80)
print("终极优化管道执行完毕")
print("=" * 80)
print(f"总执行时间: {total_time:.2f} 秒")
print("\n✅ 所有任务已根据精确的依赖关系高效执行完成!")