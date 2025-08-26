from joblib import Parallel, delayed
import time
from tools.tools import get_path, get_year, save2nc
import shutil
import os
import xarray as xr
import numpy_financial as npf
import numpy as np
import glob
import traceback

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
    # 提前准备每个来源年份的摊销图层（包含 affects_year 维度）
    template = pv_values_all_years.sel(year=all_years[0]).drop_vars('year')

    total_amortized_costs = xr.concat(
        [
            xr.concat(
                [xr.zeros_like(template, dtype=np.float32) for _ in all_years],
                dim=xr.DataArray(all_years, dims='affects_year', name='affects_year')
            )
            for _ in all_years
        ],
        dim=xr.DataArray(all_years, dims='source_year', name='source_year')
    )

    # 逐个 source_year 计算影响年份并填值
    for cost_year in all_years:
        payment = annual_payments.sel(year=cost_year).drop_vars('year')

        for affect_year in all_years:
            if cost_year <= affect_year <= cost_year + horizon - 1:
                total_amortized_costs.loc[dict(source_year=cost_year, affects_year=affect_year)] = payment

    # 汇总所有 source_year → 每个 affects_year 的摊销值
    # 得到结果维度为 (affects_year, cell, ...)
    total_amortized_costs = total_amortized_costs.sum(dim='source_year')
    total_amortized_costs.name = 'data'

    print("开始计算总摊销成本数据集...")
    total_amortized_costs = total_amortized_costs.chunk({'affects_year': 1, 'cell': 'auto'}).compute()
    print("✅ 总摊销成本数据集计算完成。")

    # 关闭句柄
    all_costs_ds.close()
    print("✅ 所有源文件句柄已关闭。")

    if njobs and njobs > 0:
        # 线程并行：共享内存，避免把 total_amortized_costs 复制到多个进程
        def _save_one_year(y: int):
            try:
                out_dir = os.path.join(target_path_name, f"{y}")
                os.makedirs(out_dir, exist_ok=True)
                out_path = os.path.join(out_dir, f"{amortize_file}_amortised_{y}.nc")
                print(f"  - [thread] 保存年份 {y} -> {out_path}")
                # 直接使用闭包中的 total_amortized_costs（已 .load() 在内存）
                da_y = total_amortized_costs.sel(affects_year=y).expand_dims(year=[y])
                save2nc(da_y, out_path)  # 这里只做 I/O，不再触发大计算
                return f"✅ 年份 {y} 已保存"
            except Exception as e:
                return f"❌ 年份 {y} 失败: {e}"

        results = Parallel(n_jobs=njobs, backend="threading")(
            delayed(_save_one_year)(y) for y in all_years
        )
        for msg in results:
            print(msg)

    else:
        # 顺序写盘（最稳，最省内存）
        for y in all_years:
            out_dir = os.path.join(target_path_name, f"{y}")
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, f"{amortize_file}_amortised_{y}.nc")
            da_y = total_amortized_costs.sel(affects_year=y).expand_dims(year=[y])
            save2nc(da_y, out_path)

    print(f"\n✅ 任务完成: '{amortize_file}' 的所有年份摊销成本已成功计算并逐年保存。")


# --- 辅助函数：专门用于计算单个文件对的差异，以便并行化 ---
def calculate_and_save_single_diff(diff_file, year, year_path, year_path_0, target_year_path):
    """
    计算并保存单个文件对的差异。
    这个函数将被并行调用。
    """
    try:
        # 1. 构造上一年度和当前年度的文件路径
        src_file_0 = os.path.join(year_path_0, f"{diff_file}_{year - 1}.nc")
        src_file_1 = os.path.join(year_path, f"{diff_file}_{year}.nc")

        # 2. 打开这对文件
        with xr.open_dataset(src_file_0) as ds_0, xr.open_dataset(src_file_1) as ds_1:
            # 3. 计算差异
            ds_res = ds_1 - ds_0

            # 4. 构造目标路径并保存
            variable_name = diff_file.replace('.nc', '')
            dst_filename = f"{variable_name}_diff_{year}.nc"
            dst_file = os.path.join(target_year_path, dst_filename)
            save2nc(ds_res, dst_file)
            # ds_res.to_netcdf(dst_file)
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
    for f in copy_files:
        copy_single_file(
            os.path.join(year_path, f'{f}_{year}.nc'),
            os.path.join(target_year_path, f'{f}_{year}.nc')
        )


    # --- 2. 【正确地】并行化差异计算 ---
    if year > years[0] and diff_files:

        # --- 1. 前置条件检查 ---
        # 使用 if/else 结构替代 continue。
        # 只有当 "跳过" 条件不满足时，才执行主要的逻辑。
        if input_files[0] in target_year_path:
            print(f"信息 (Year {year}): 检测到 '{input_files[0]}' 路径，按规则跳过所有差异计算。")
        else:
            # --- 主要逻辑块开始 ---
            print(f"\n {year_path}: 开始准备差异计算任务...")
            year_path_0 = os.path.join(origin_path_name, f'out_{year - 1}')
            for diff_file in diff_files:
                calculate_and_save_single_diff(diff_file, year, year_path, year_path_0, target_year_path)


            # 打印所有任务的结果
            print(f"Year {year}: 所有差异计算任务已完成。")
    return f"Year {year} processed."


# ==============================================================================
# STAGE 1: 计算利润 (Profit = Revenue - Cost)
# ==============================================================================
def calculate_profit_for_run(year, run_path, run_name, cost_basename, revenue_basename):
    """
    为单个情景(Run)和单个类别计算利润。
    """
    print(f"{run_path}/{run_name}/{year}: 计算利润...")
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
            save2nc(profit, os.path.join(profit_run_path, profit_filename))

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
    print(f"{output_path}: 计算政策成本 {policy_type}/{cost_category}...")
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
            save2nc(policy_cost, os.path.join(output_dir, output_filename))
            # policy_cost.to_netcdf(os.path.join(output_dir, output_filename))

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
    print(f"{output_path}: 计算转型成本差值 {policy_type}/{tran_cost_file}...")
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
        save2nc(tran_cost_diff, os.path.join(output_dir, output_filename))
        # tran_cost_diff.to_netcdf(os.path.join(output_dir, output_filename))

        return f"✅ TranCost Diff: Calculated {policy_type} for {tran_cost_file} year {year}"

    except FileNotFoundError as e:
        return f"❌ TranCost Diff Error (File Not Found) for {policy_type}/{tran_cost_file} year {year}: {e.filename}"
    except Exception as e:
        return f"❌ TranCost Diff Error (Calculation Failed) for {policy_type}/{tran_cost_file} year {year}: {e}"


def aggregate_and_save_cost(year, output_path, cost_type):
    """
    【最终版】聚合单个年份的成本文件，使用一个精确的文件列表。
    """
    print(f"Starting aggregation all cost for '{cost_type}' files for year {year}...")
    # 1. 【关键修改】根据传入的列表构建完整的文件路径
    data_type_names = ['xr_cost_ag', 'xr_cost_agricultural_management', 'xr_cost_non_ag',
                       'xr_cost_transition_ag2ag_diff']
    data_type_names_adds = ['xr_transition_cost_ag2non_ag_amortised_diff', 'xr_transition_cost_ag2non_ag_diff']
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
                # 正常打开和处理文件
                with xr.open_dataset(file_path) as ds:

                    # 预检查：如果文件本身就没有任何数据变量，则立即失败。
                    if not ds.data_vars:
                        raise ValueError("文件不包含任何数据变量 (is empty or contains only coordinates)。")

                    summed_single_ds = ds.sum(dim=[d for d in ds.dims if d != 'cell'])

                    if total_sum_ds is None:
                        # 这是第一个有效文件，将其设为基础
                        total_sum_ds = summed_single_ds
                    else:
                        # 对后续文件进行原地相加
                        total_sum_ds += summed_single_ds

            except FileNotFoundError:
                # 对于文件不存在的情况，我们只打印警告并继续，因为这是一个预料之中的IO问题。
                print(f"  - 警告: 文件未找到，将被跳过: {file_path}")
                continue

            except ValueError as e:
                # ✅ 核心修改：捕获 ValueError
                # 这是最可能发生数据不一致错误的地方。
                # 我们构造一个新的、信息更丰富的错误消息，然后重新抛出，使程序崩溃。

                error_message = (
                    f"\n"
                    f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n"
                    f"!! 数据处理失败，脚本已终止。\n"
                    f"!!\n"
                    f"!! 问题文件 (Problematic File): {file_path}\n"
                    f"!!\n"
                    f"!! 错误原因: 该文件的数据变量与之前的文件不匹配，导致无法相加。\n"
                    f"!!           (常见原因：此文件为空，或数据变量名称不同)。\n"
                    f"!!\n"
                    f"!! 原始错误回溯: {e}\n"
                    f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n"
                )

                # 重新抛出异常，并附带我们构造的清晰信息。
                # `from e` 会保留原始的异常堆栈，便于深度调试。
                raise ValueError(error_message) from e

            except Exception as e:
                # 捕获其他所有意外错误
                error_message = (
                    f"\n"
                    f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n"
                    f"!! 处理文件时发生未知错误，脚本已终止。\n"
                    f"!!\n"
                    f"!! 问题文件 (Problematic File): {file_path}\n"
                    f"!!\n"
                    f"!! 原始错误类型: {type(e).__name__}\n"
                    f"!! 原始错误回溯: {e}\n"
                    f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n"
                )

        # 5. 保存
        if data_type_names_add == 'xr_transition_cost_ag2non_ag_amortised':
            am_type = 'amortised'
        elif data_type_names_add == 'xr_transition_cost_ag2non_ag':
            am_type = 'original'
        final_path = os.path.join(file_dir, f'xr_total_cost_{cost_type}_{am_type}_{year}.nc')
        save2nc(total_sum_ds, final_path)
        # total_sum_ds.to_netcdf(final_path)


def aggregate_and_save_summary(year, output_path, data_type, data_type_names, input_files_name):
    """
    【最终版】聚合单个年份的成本文件，使用一个精确的文件列表。
    """
    print(f"Starting aggregation for '{data_type}' files for year {year}...")
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
        final_path = os.path.join(final_dir, f'xr_total_{data_type}_{year}.nc')
        save2nc(total_sum_ds, final_path)
        # total_sum_ds.to_netcdf(final_path)

        return f"✅ Stage 2: Aggregated {len(full_file_paths)} '{data_type}' files for year {year} and saved to {final_path}."

    except Exception as e:
        return f"❌ Stage 2 Error (Aggregation Failed) for '{data_type}' year {year}: {e}"


def process_all_years_serially(years, origin_path_name, target_path_name, copy_files, diff_files,n_jobs=0):
    """
    辅助函数：按顺序处理所有年份。
    这个函数本身是串行的，因为它内部的年份之间有依赖关系。
    """
    print("信息：开始copy and diff 任务...")
    try:
        if n_jobs == 0:
            for year in years:
                # 您原来的 process_single_year 调用保持不变
                process_single_year(year, years, origin_path_name, target_path_name, copy_files, diff_files)
        else:
            Parallel(n_jobs=n_jobs)(
                delayed(process_single_year)(
                    year, years, origin_path_name, target_path_name, copy_files, diff_files
                ) for year in years
            )
        return "process_all_years_serially: OK"  # 返回成功状态
    except Exception as e:
        print(f"❌ 错误：在串行处理年份时发生错误: {e}")
        import traceback
        traceback.print_exc()
        return f"process_all_years_serially: FAILED with {e}"  # 返回失败状态


if __name__ == "__main__":
    # ============================================================================
    task_name = '20250823_Paper2_Results1'
    njobs = 41
    task_dir = f'../../../output/{task_name}'
    input_files = config.INPUT_FILES
    path_name_0 = get_path(task_name, input_files[0])
    path_name_1 = get_path(task_name, input_files[1])
    path_name_2 = get_path(task_name, input_files[2])
    years = get_year(path_name_0)
    run_names = [input_files[0], input_files[1], input_files[2]]
    run_paths = [path_name_0, path_name_1, path_name_2]

    output_path = f'{task_dir}/carbon_price/0_base_data'
    os.makedirs(output_path, exist_ok=True)

    cost_files = ['xr_cost_ag', 'xr_cost_agricultural_management', 'xr_cost_non_ag', 'xr_cost_transition_ag2ag',
                  'xr_transition_cost_ag2non_ag']
    revenue_files = ['xr_revenue_ag', 'xr_revenue_agricultural_management', 'xr_revenue_non_ag']
    carbon_files = ['xr_GHG_ag', 'xr_GHG_ag_management', 'xr_GHG_non_ag', 'xr_transition_GHG']
    bio_files = ['xr_biodiversity_GBF2_priority_ag', 'xr_biodiversity_GBF2_priority_ag_management',
                 'xr_biodiversity_GBF2_priority_non_ag']
    amortize_files = ['xr_transition_cost_ag2non_ag']

    carbon_files_diff = ['xr_GHG_ag_diff', 'xr_GHG_ag_management', 'xr_GHG_non_ag', 'xr_transition_GHG']
    bio_files_diff = ['xr_biodiversity_GBF2_priority_ag_diff', 'xr_biodiversity_GBF2_priority_ag_management',
                      'xr_biodiversity_GBF2_priority_non_ag']
    # ============================================================================
    # 第一批：数据预处理阶段 (摊销成本计算 + 文件复制/差异计算)
    # ============================================================================
    start_time = time.time()

    print("=" * 80)

    # --- 第一批任务 (拆分为两个独立的组) ---
    for i in range(3):
        origin_path_name = get_path(task_name, input_files[i])
        target_path_name = os.path.join(output_path, input_files[i])
        amortize_costs(origin_path_name, target_path_name, amortize_files[0], years, njobs=njobs)
    print("✅ 第一批任务 (摊销成本计算) 完成!")
    # ----------------------------------------------------------------------------
    # ===========================================================================
    # --- 阶段 1: 文件处理 ---
    print("\n--- 阶段 1: 文件处理 ---")
    # 直接定义并执行任务，不再使用 delayed
    file_processing_tasks_count = 0
    for i in range(3):
        origin_path_name = get_path(task_name, input_files[i])
        target_path_name = os.path.join(output_path, input_files[i])
        if i == 0:
            copy_files = cost_files + revenue_files
            diff_files = []
        elif i == 1:
            copy_files = cost_files + revenue_files + carbon_files
            diff_files = ['xr_GHG_ag']
        elif i == 2:
            copy_files = cost_files + revenue_files + carbon_files + bio_files
            diff_files = ['xr_biodiversity_GBF2_priority_ag', 'xr_GHG_ag']
        else:
            copy_files = []

        print(f"  -> 正在处理: {input_files[i]}")
        # 直接调用函数，而不是用 delayed 包装
        process_all_years_serially(years, origin_path_name, target_path_name, copy_files, diff_files, n_jobs=njobs)
    print(f"✅ 文件处理任务完成!")

    # --- 阶段 2: 独立汇总计算 ---
    print("\n--- 阶段 2: 独立汇总计算 ---")
    if njobs == 0:
        for year in years[1:]:
            # 直接调用
            aggregate_and_save_summary(year, output_path, 'carbon', carbon_files_diff, input_files[1])
            aggregate_and_save_summary(year, output_path, 'bio', bio_files_diff, input_files[2])
    else:
        # --- 正确的代码 ---
        Parallel(n_jobs=njobs)(
            delayed(aggregate_and_save_summary)(year, output_path, 'carbon', carbon_files, input_files[1])
            for year in years[1:]
        )
        Parallel(n_jobs=njobs)(
            delayed(aggregate_and_save_summary)(year, output_path, 'bio', bio_files, input_files[2])
            for year in years[1:]
        )


    print(f"✅ 第2批任务完成! ")

    # --- 阶段 3: 利润计算 ---
    print("\n--- 阶段 3: 利润计算 ---")
    profit_categories = zip(cost_files, revenue_files)
    for cost_base, rev_base in profit_categories:
        if njobs == 0:
            for cost_base, rev_base in profit_categories:
                for i, run_name in enumerate(run_names):
                    for year in years:
                        # 直接调用
                        calculate_profit_for_run(year, output_path, run_name, cost_base, rev_base)
        else:
            for i, run_name in enumerate(run_names):
                Parallel(n_jobs=njobs)(
                    delayed(calculate_profit_for_run)(year, output_path, run_name, cost_base, rev_base)
                    for year in years
                )
    print(f"✅ 第3批任务完成!")

    # --- 阶段 4: 政策成本计算 ---
    print("\n--- 阶段 4: 政策成本计算 ---")
    batch_4_tasks_count = 0
    policy_cost_categories = [f.replace('xr_cost_', '') for f in cost_files]
    for category in policy_cost_categories:
        if njobs == 0:
            for year in years[1:]:
                # 直接调用
                calculate_policy_cost(year, output_path, run_names, category, 'carbon')
                calculate_policy_cost(year, output_path, run_names, category, 'bio')
            else:
                Parallel(n_jobs=njobs)(
                    delayed(calculate_policy_cost)(year, output_path, run_names, category, 'carbon')
                    for year in years[1:]
                )
                Parallel(n_jobs=njobs)(
                    delayed(calculate_policy_cost)(year, output_path, run_names, category, 'bio')
                    for year in years[1:]
                )
    print(f"✅ 第4批任务完成! ")

    # --- 阶段 5: 转型成本差值计算 (仅独立部分) ---
    print("\n--- 阶段 5: 转型成本差值计算 ---")
    batch_5a_tasks_count = 0
    independent_tran_files = ['xr_cost_transition_ag2ag', 'xr_transition_cost_ag2non_ag',
                              'xr_transition_cost_ag2non_ag_amortised']
    for tran_file in independent_tran_files:
        if njobs == 0:
            for year in years[1:]:
                # 直接调用
                calculate_transition_cost_diff(year, output_path, run_names, tran_file, 'carbon')
                calculate_transition_cost_diff(year, output_path, run_names, tran_file, 'bio')
        else:
            Parallel(n_jobs=njobs)(
                delayed(calculate_transition_cost_diff)(year, output_path, run_names, tran_file, 'carbon')
                for year in years[1:]
            )
            Parallel(n_jobs=njobs)(
                delayed(calculate_transition_cost_diff)(year, output_path, run_names, tran_file, 'bio')
                for year in years[1:]
            )
    print(f"✅ 第5a批 (独立部分) 任务完成! (共 {batch_5a_tasks_count} 个)")

    # --- 阶段 6: 成本聚合 ---
    print("\n--- 阶段 6: 成本聚合 ---")
    batch_6_tasks_count = 0
    if njobs == 0:
        for year in years[1:]:
            # 直接调用
            aggregate_and_save_cost(year, output_path, 'carbon')
            aggregate_and_save_cost(year, output_path, 'bio')
        else:
            Parallel(n_jobs=njobs)(
                delayed(aggregate_and_save_cost)(year, output_path, 'carbon')
                for year in years[1:]
            )
            Parallel(n_jobs=njobs)(
                delayed(aggregate_and_save_cost)(year, output_path, 'bio')
                for year in years[1:]
            )

    print(f"✅ 第6批 (最终聚合) 任务完成! ")

    # --- 总结 ---
    end_time = time.time()
    total_time = end_time - start_time
    print("\n" + "=" * 80)
    print("所有任务已按顺序执行完毕")
    print(f"总执行时间: {total_time / 60 / 60:.2f} 小时 )")
    print("=" * 80)