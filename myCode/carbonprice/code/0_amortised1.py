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


def amortize_costs(origin_path_name, target_path_name, amortize_file, years, njobs=-1, rate=0.07, horizon=30,
                   temp_path=None):
    """
    【最终稳健版 v5】
    - 使用 xr.apply_ufunc 替换 xr.map_blocks 来解决 "AttributeError: 'numpy.ndarray' object has no attribute 'sizes'" 的问题。
    - apply_ufunc 对维度和元数据的处理更为严格和稳健，能避免在并行计算中丢失xarray对象的类型信息。
    """
    is_parallel = njobs != 0
    joblib_n_jobs = njobs if is_parallel else 1
    if is_parallel:
        dask_num_workers = njobs if njobs > 0 else None
        dask_scheduler = 'threads'
        parallel_mode_desc = f"并行模式 ({njobs} 个核心)"
    else:
        dask_num_workers = 1
        dask_scheduler = 'single-threaded'
        parallel_mode_desc = "串行模式 (njobs=0)"

    print(f"开始计算 '{amortize_file}' 的摊销成本... [{parallel_mode_desc}]")

    if temp_path:
        os.makedirs(temp_path, exist_ok=True)
        temp_dir = tempfile.mkdtemp(dir=temp_path, prefix="amortize_chunked_")
        print(f"将在指定路径下创建临时目录: {temp_dir}")
    else:
        temp_dir = tempfile.mkdtemp(prefix="amortize_chunked_")
        print(f"将在系统默认临时路径下创建目录: {temp_dir}")

    temp_zarr_path = os.path.join(temp_dir, "computed_results.zarr")

    try:
        # --- 步骤1: 构建输入数据的Dask计算图，只对'cell'分块 ---
        print("\n步骤1: 构建输入数据计算图，只对'cell'分块...")
        file_paths = [os.path.join(origin_path_name, f'out_{year}', f'{amortize_file}_{year}.nc') for year in years]
        existing_files = [p for p in file_paths if os.path.exists(p)]
        if not existing_files: raise FileNotFoundError("找不到任何相关文件。")

        lazy_data_arrays = []
        for file_path in existing_files:
            with xr.open_dataset(file_path, chunks={'cell': 'auto'}) as ds:
                lazy_data_arrays.append(ds[get_main_data_variable_name(ds)])

        pv_values_all_years_graph = xr.concat(lazy_data_arrays, dim='year', join='outer')
        valid_years = sorted([int(path.split('_')[-1].split('.')[0]) for path in existing_files])
        pv_values_all_years_graph = pv_values_all_years_graph.assign_coords(year=valid_years)
        print("✅ 输入数据计算图构建完成。")

        # --- 步骤2: 【核心修正】使用 xr.apply_ufunc 构建最终结果的计算图 ---
        print("\n步骤2: 使用xr.apply_ufunc构建最终结果的计算图...")

        # 准备用于计算的两个Dask数组
        # 1. 输入数据，确保'year'不分块，'cell'分块
        input_data_chunked = pv_values_all_years_graph.chunk({'year': -1, 'cell': 'auto'})

        # 2. 影响矩阵，作为一个xarray.DataArray
        n_years = len(valid_years)
        influence_matrix_np = np.zeros((n_years, n_years))
        for i in range(n_years):
            influence_matrix_np[i, i:min(i + horizon, n_years)] = 1
        influence_da = xr.DataArray(
            influence_matrix_np, dims=['year', 'target_year'],
            coords={'year': valid_years, 'target_year': valid_years}
        )

        # 定义计算函数
        def calculate_amortization_ufunc(pv_array, influence_matrix_array):
            annual_payments = -1 * npf.pmt(rate, horizon, pv=pv_array, fv=0, when='begin')
            # einsum在numpy数组上操作
            return np.einsum('...y,yz->...z', annual_payments, influence_matrix_array)

        # 调用 apply_ufunc
        total_amortized_costs_graph = xr.apply_ufunc(
            calculate_amortization_ufunc,
            input_data_chunked,
            influence_da,
            input_core_dims=[['year'], ['year', 'target_year']],  # 定义输入的核心维度
            output_core_dims=[['target_year']],  # 定义输出的核心维度
            dask="parallelized",  # 启用Dask并行计算
            output_dtypes=[input_data_chunked.dtype]  # 指定输出类型
        ).rename({'target_year': 'year'})  # 将新维度重命名回 'year'

        print("✅ 最终结果的计算图已通过apply_ufunc构建。")

        # --- 步骤3: 执行计算并流式写入临时Zarr存储 ---
        print(f"\n步骤3: 开始执行分块计算并流式写入临时磁盘存储 [{parallel_mode_desc}]...")
        final_ds_graph = total_amortized_costs_graph.to_dataset(name='transition_cost_amortised')

        with dask.config.set(scheduler=dask_scheduler, num_workers=dask_num_workers):
            final_ds_graph.to_zarr(
                temp_zarr_path,
                compute=True,
                mode='w',
                consolidated=True  # 推荐添加，可以提高后续读取性能
            )

        print("✅ 所有块计算完成，完整结果已保存在临时存储中。")

        # --- 步骤4: 从临时存储中快速、并行地分发保存 ---
        print(f"\n步骤4: 从临时存储中快速、并行地读取并保存为最终的NetCDF文件 [{parallel_mode_desc}]...")
        computed_ds = xr.open_zarr(temp_zarr_path)

        def save_from_zarr(year_to_save, base_path, file_prefix, save_function):
            yearly_data = computed_ds.sel(year=year_to_save)
            output_folder = os.path.join(base_path, f'{year_to_save}')
            os.makedirs(output_folder, exist_ok=True)
            target_file_path = os.path.join(output_folder, f'{file_prefix}_amortised_{year_to_save}.nc')
            save_function(yearly_data, target_file_path)
            return f"✅ 年份 {year_to_save} 已成功保存。"

        tasks_generator = (
            delayed(save_from_zarr)(year, target_path_name, amortize_file, save2nc)
            for year in valid_years
        )
        results = Parallel(n_jobs=joblib_n_jobs)(tasks_generator)
        for res in results: print(res)

    finally:
        # --- 步骤5: 清理临时文件 ---
        print(f"\n步骤5: 清理临时目录 {temp_dir}...")
        shutil.rmtree(temp_dir)
        print("✅ 清理完成。")

    print(f"\n✅ 任务完成。")


if __name__ == "__main__":
    # ============================================================================
    task_name = '20250823_Paper2_Results1'
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
    # --- 第一批任务 (拆分为两个独立的组) ---
    # for i in range(3):
    for i in [2]:
        origin_path_name = get_path(task_name, input_files[i])
        target_path_name = os.path.join(output_path, input_files[i])
        amortize_costs(origin_path_name, target_path_name, amortize_files[0], years, njobs)