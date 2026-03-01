import os
import math
from contextlib import suppress
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np
import pandas as pd
import xarray as xr
import cf_xarray as cfxr
from joblib import Parallel, delayed

from tools.tools import get_path, get_year, save2nc, filter_all_from_dims
import tools.config as config


def summarize_to_type(
        scenarios: List[str],
        years: List[int],
        file: str,
        keep_dim: str,
        output_file: str,
        var_name: str = "data",
        scale: float = 1e6,
        n_jobs: int = None,
        dtype: str = "float32",
        scenario_name: bool = True,
        chunks: Optional[Dict[str, int]] = 'auto',
) -> xr.DataArray:
    """
    每次只处理一个 scenario，scenario 内用 dask 并发处理所有 year。处理完后关闭再处理下一个 scenario，最后拼接。
    """
    print(f"Summarizing to type from {file}, keep_dim={keep_dim}...")
    base_dir = f'../../../output/{config.TASK_NAME}/carbon_price'

    # --- 1) 找一个示例文件确定 type 坐标 ---
    if scenario_name:
        sample_path = os.path.join(base_dir, '0_base_data', scenarios[-1],
                                   f"{years[-1]}/{file}_{scenarios[-1]}_{years[-1]}.nc")
    else:
        sample_path = os.path.join(base_dir, '0_base_data', scenarios[-1],
                                   f"{years[-1]}/{file}_{years[-1]}.nc")
    if not os.path.exists(sample_path):
        raise FileNotFoundError(f"未找到{sample_path}，无法确定 type 坐标。")

    with xr.open_dataarray(sample_path, chunks=chunks) as da0:
        da0 = filter_all_from_dims(da0)
        if keep_dim not in da0.dims:
            raise ValueError(f"示例文件中不包含 keep_dim='{keep_dim}'，实际维度为 {da0.dims}")

    # --- 2) 逐 scenario 处理 ---

    def process_single_file(s, y):
        if scenario_name:
            path = os.path.join(base_dir, '0_base_data', s, str(y), f"{file}_{s}_{y}.nc")
        else:
            path = os.path.join(base_dir, '0_base_data', s, str(y), f"{file}_{y}.nc")
        if not os.path.exists(path):
            return None
        try:
            with xr.open_dataarray(path, chunks=chunks) as da:
                da = filter_all_from_dims(da)
                sum_dims = [d for d in da.dims if d != keep_dim]
                da_processed = da.sum(dim=sum_dims, keep_attrs=False) / scale
                if keep_dim != "type":
                    da_processed = da_processed.rename({keep_dim: "type"})
                da_processed = da_processed.expand_dims({"Year": [y], "scenario": [s]})
                da_processed.name = var_name
                da_processed = da_processed.load()  # 读入内存，关闭文件句柄
            return da_processed
        except Exception as e:
            print(f"Error opening {path}: {e}")
            return None

    if n_jobs is None or n_jobs <= 0:
        import dask
        per_scenario = []
        for s in scenarios:
            print(f"Processing scenario {s}...")
            # 为当前 scenario 构造所有 year 的 delayed 任务
            delayed_tasks = [process_single_file(s, y) for y in years]
            # dask 并发处理当前 scenario 的所有 year 文件
            year_results = dask.compute(*delayed_tasks)
            # 过滤有效结果
            valid_year_results = [r for r in year_results if r is not None]
            if len(valid_year_results) == 0:
                continue

            da_year = xr.concat(valid_year_results, dim="Year", join='outer', fill_value=0)
            per_scenario.append(da_year)  # 读入内存，关闭文件句柄

    else:
        def process_scenario(s, n_jobs=n_jobs):
            year_results = Parallel(n_jobs=n_jobs, backend="loky")(
                delayed(process_single_file)(s, y) for y in years
            )
            valid_year_results = [r for r in year_results if r is not None]
            return xr.concat(valid_year_results, dim="Year", join='outer', fill_value=0) if valid_year_results else None

        per_scenario = [process_scenario(s,n_jobs) for s in scenarios]
        per_scenario = [r for r in per_scenario if r is not None]

    if len(per_scenario) == 0:
        raise RuntimeError("所有场景均为空，无法生成结果。")

    # --- 3) 按 scenario 拼接 ---
    print("Concatenating scenarios...")
    out_da = xr.concat(per_scenario, dim="scenario", join='outer', fill_value=0)
    out_da = out_da.transpose("scenario", "Year", "type")
    out_da.name = var_name
    out_da = out_da.astype(dtype, copy=False)

    # --- 4) 写 NetCDF ---
    output_dir = os.path.join(base_dir, '1_draw_data')
    os.makedirs(output_dir, exist_ok=True)
    if scenario_name:
        output_path = os.path.join(output_dir, f'{output_file}.nc')
    else:
        output_path = os.path.join(output_dir, f'{output_file}_original.nc')

    print("Saving to NetCDF...")
    save2nc(out_da, output_path)
    print(f"✅ Saved to {output_path}")

    return out_da

# def summarize_to_type(
#         scenarios,
#         years,
#         file,
#         keep_dim,
#         output_file,
#         var_name="data",
#         scale=1e6,
#         dtype="float32",
#         chunks='auto',
# ):
#     print(f"Summarizing to type from {file}, keep_dim={keep_dim}...")
#     base_dir = f'../../../output/{config.TASK_NAME}/carbon_price'
#
#     # 1) 收集文件
#     file_paths, file_scenarios, file_years = [], [], []
#     for s in scenarios:
#         for y in years:
#             p = os.path.join(base_dir, '0_base_data', s, str(y), f"{file}_{s}_{y}.nc")
#             if os.path.exists(p):
#                 file_paths.append(p)
#                 file_scenarios.append(s)
#                 file_years.append(y)
#     if not file_paths:
#         raise RuntimeError("未找到任何有效文件")
#
#     print(f"Found {len(file_paths)} files, opening with mfdataset (chunks={chunks})...")
#
#     # 2) 先探测目标变量
#     probe = xr.open_dataset(file_paths[0], engine="netcdf4", decode_cf=False, mask_and_scale=False)
#     try:
#         numeric_vars = [k for k, v in probe.data_vars.items() if np.issubdtype(v.dtype, np.number)]
#         if not numeric_vars:
#             raise RuntimeError("数据集中没有数值变量可用于汇总")
#         target_var = numeric_vars[0] if var_name == "data" else var_name
#         if target_var not in probe.data_vars:
#             target_var = numeric_vars[0]
#
#         if keep_dim not in probe[target_var].dims:
#             raise ValueError(f"变量 '{target_var}' 中不包含 keep_dim='{keep_dim}'，实际维度: {probe[target_var].dims}")
#         if keep_dim in probe.coords:
#             type_coord = probe.coords[keep_dim].values
#         else:
#             type_coord = np.arange(probe[target_var].sizes[keep_dim])
#     finally:
#         with suppress(Exception):
#             probe.close()
#
#     # 3) 只读目标变量
#     def _preprocess(ds):
#         return ds[[target_var]]
#
#     ds = xr.open_mfdataset(
#         file_paths,
#         engine="netcdf4",
#         combine='nested',
#         concat_dim='file_idx',
#         preprocess=_preprocess,
#         decode_cf=False,
#         mask_and_scale=False,
#         chunks=chunks,
#         parallel=True,
#     )
#
#     try:
#         da = ds[target_var]
#         print(f"Loaded data shape: {da.shape}, dims: {da.dims}")
#
#         da = filter_all_from_dims(da)
#
#         sum_dims = [d for d in da.dims if d not in [keep_dim, 'file_idx']]
#         print(f"Summing over dimensions: {sum_dims}, keeping: [{keep_dim}, 'file_idx']")
#         da_summed = (da if not sum_dims else da.sum(dim=sum_dims, keep_attrs=False)) / scale
#
#         if keep_dim != "type":
#             da_summed = da_summed.rename({keep_dim: "type"})
#         try:
#             da_summed = da_summed.sel(type=type_coord)
#         except Exception:
#             da_summed = da_summed.assign_coords(type=("type", type_coord))
#
#         da_summed = da_summed.assign_coords(
#             scenario=('file_idx', file_scenarios),
#             Year=('file_idx', file_years)
#         )
#         da_indexed = da_summed.set_index(file_idx=['scenario', 'Year'])
#         da_final = da_indexed.unstack('file_idx')
#
#         da_final = da_final.transpose("scenario", "Year", "type")
#         da_final.name = ("data" if var_name == "data" else var_name)
#         da_final = da_final.astype(dtype, copy=False)
#
#         print(f"Final data shape: {da_final.shape}")
#
#         output_dir = os.path.join(base_dir, '1_draw_data')
#         os.makedirs(output_dir, exist_ok=True)
#         output_path = os.path.join(output_dir, f'{output_file}.nc')
#         print("Starting computation and saving...")
#         save2nc(da_final, output_path)
#         print(f"✅ Saved to {output_path}")
#
#     finally:
#         with suppress(Exception):
#             ds.close()
#
#     return da_final


def summarize_to_category(
        scenarios: List[str],
        years: List[int],
        files: List[str],
        output_file: str,
        n_jobs: int = 41,
        var_name: str = "data",
        scale: float = 1e6,
        dtype: str = "float32",
        scenario_name = True,
        chunks: Dict[str, int] | None = None,
) -> xr.DataArray:
    """
    同时处理多个 input_files，返回一个 (scenario, Year, type) 的 DataArray 并保存为 NetCDF。

    - scenario: input_files
    - Year: 指定的年份
    - type: files (用 KEY_TO_COLUMN_MAP 替换后的名称)
    """

    base_dir = f'../../../output/{config.TASK_NAME}/carbon_price'
    years_sorted = sorted(years)
    type_names = [config.KEY_TO_COLUMN_MAP.get(f, f) for f in files]

    def _sum_single(scenario: str, year: int, file: str) -> float | None:
        input_path = os.path.join(base_dir,'0_base_data', scenario)
        if scenario_name:
            nc_path = os.path.join(input_path, f'{year}', f'{file}_{scenario}_{year}.nc')
        else:
            nc_path = os.path.join(input_path, f'{year}', f'{file}_{year}.nc')
        if not os.path.exists(nc_path):
            raise FileNotFoundError(f"未找到文件: {nc_path}")
        with xr.open_dataarray(nc_path, engine="h5netcdf") as da:
            filtered = filter_all_from_dims(da).load()
            return filtered.sum().item()

    # 并行计算：三重循环 scenario × year × type
    results_flat = Parallel(n_jobs=n_jobs)(
        delayed(_sum_single)(scenario, year, file)
        for scenario in scenarios
        for year in years_sorted
        for file in files
    )

    # reshape -> (scenario, Year, type)
    values = np.array(results_flat, dtype="float64").reshape(
        len(scenarios), len(years_sorted), len(files)
    )

    if scale:
        values = values / scale

    da = xr.DataArray(
        values.astype(dtype),
        coords={
            "scenario": scenarios,
            "Year": years_sorted,
            "type": type_names,
        },
        dims=("scenario", "Year", "type"),
        name=var_name,
    )

    type_vals = da.coords["type"].values
    new_type_vals = []
    for v in type_vals:
        v = v.replace(" GHG", "")
        v = v.replace(" biodiversity", "")
        new_type_vals.append(v)
    da = da.assign_coords(type=("type", new_type_vals))

    if chunks:
        da = da.chunk(chunks)

    output_dir = os.path.join(
        base_dir,
        '1_draw_data')
    out_nc = os.path.join(
        base_dir,
        '1_draw_data',
        f'{output_file}.nc'
    )
    os.makedirs(os.path.dirname(output_dir), exist_ok=True)
    save2nc(da, out_nc)

    print(f"✅ 保存 NetCDF: {out_nc} | shape={da.shape} dims={da.dims}")
    return da


def build_profit_and_cost_nc(
    economic_da: xr.DataArray,
    input_files_0,              # baseline 列表（用第 0 个）
    input_files_1,              # 碳情景列表
    input_files_2,              # 碳+生物多样性 列表
    carbon_names,               # 与 input_files_1 对齐的输出名
    carbon_bio_names,           # 与 input_files_2 对齐的输出名
    counter_carbon_bio_names,   # 与 input_files_2 对齐的输出名
    add_total=False,            # 是否把 Total 作为额外的 type 写入
):
    """
    economic_da 维度必须是 (scenario, Year, type) 且 type 名包含：
      'Ag revenue','Ag cost','AgMgt revenue','AgMgt cost',
      'Non-ag revenue','Non-ag cost','Transition(ag→ag) cost',
      'Transition(ag→non-ag) amortised cost'
    """
    out_dir = f'../../../output/{config.TASK_NAME}/carbon_price/1_draw_data'
    os.makedirs(out_dir, exist_ok=True)

    # ---------- 1) 计算 profit(scenario, Year, type) ----------
    required_types = [
        "Ag revenue", "Ag cost",
        "AgMgt revenue", "AgMgt cost",
        "Non-ag revenue", "Non-ag cost",
        "Transition(ag→ag) cost",
        "Transition(ag→non-ag) amortised cost",
    ]
    type_names = set(economic_da.coords["type"].astype(str).values)
    missing = [t for t in required_types if t not in type_names]
    if missing:
        raise ValueError(f"economic_da 缺少必要 type: {missing}")

    def pick(name: str) -> xr.DataArray:
        # 选中一个类型 -> 去掉原来的 type 维度，避免后续按坐标对齐
        da = economic_da.sel(type=name).squeeze(drop=True)
        # 现在 da 的维度里已经没有 type 了（只剩 scenario/year 等）
        return da

    profit_list = [
        (pick("Ag revenue") - pick("Ag cost")).expand_dims(type=["Ag profit"]),
        (pick("AgMgt revenue") - pick("AgMgt cost")).expand_dims(type=["AgMgt profit"]),
        (pick("Non-ag revenue") - pick("Non-ag cost")).expand_dims(type=["Non-ag profit"]),
        (-pick("Transition(ag→ag) cost")).expand_dims(type=["Transition(ag→ag) profit"]),
        (-pick("Transition(ag→non-ag) amortised cost")).expand_dims(type=["Transition(ag→non-ag) amortised profit"]),
    ]

    profit_da = xr.concat(profit_list, dim="type", join='outer', fill_value=0)
    profit_da.name = "data"

    # 如果需要固定维度顺序（比如 scenario, year, type）：
    want = [d for d in ["scenario", "Year", "type"] if d in profit_da.dims]
    profit_da = profit_da.transpose(*want)
    profit_da.name = "data"

    # 存 profit（修正：encoding 的 key 要用当前 name）
    profit_nc = os.path.join(out_dir, "xr_profit.nc")
    save2nc(profit_da, profit_nc)

    # ---------- 2) 构造三个差值 DataArray ----------
    baseline = input_files_0[0]

    # A) carbon：baseline - input_files_1[i]  ->  (policy, Year, type)
    diffs = []
    for scen in input_files_1:
        diffs.append(profit_da.sel(scenario=baseline) - profit_da.sel(scenario=scen))
    carbon_cost = xr.concat(diffs, dim="policy", join='outer', fill_value=0).assign_coords(policy=list(carbon_names))

    # B) carbon_bio：input_files_1[i] - input_files_2[idx]
    if len(input_files_2) % len(input_files_1) != 0:
        raise ValueError("len(input_files_2) 必须是 len(input_files_1) 的整数倍，用于分组匹配。")
    bio_nums = len(input_files_2) // len(input_files_1)

    diffs = []
    for k, scen2 in enumerate(input_files_2):
        i = k // bio_nums
        scen1 = input_files_1[i]
        diffs.append(profit_da.sel(scenario=scen1) - profit_da.sel(scenario=scen2))
    carbon_bio_cost = xr.concat(diffs, dim="policy", join='outer', fill_value=0).assign_coords(policy=list(carbon_bio_names))

    # C) counter：input_files_2 的前 bio_nums 个与 baseline
    diffs = []
    for i in range(len(counter_carbon_bio_names)):
        scen2 = input_files_2[i]
        diffs.append(profit_da.sel(scenario=baseline) - profit_da.sel(scenario=scen2))
    counter_cost = xr.concat(diffs, dim="policy", join='outer', fill_value=0).assign_coords(policy=list(counter_carbon_bio_names))

    # 可选把 Total 作为额外的 type 追加
    def append_total(da, add_total=add_total):
        if not add_total:
            return da
        total = da.sum(dim="type")
        total = total.expand_dims({"type": ["Total"]})
        da2 = xr.concat([da, total], dim="type", join='outer', fill_value=0)
        return da2

    carbon_cost = append_total(carbon_cost)
    carbon_bio_cost = append_total(carbon_bio_cost)
    counter_cost = append_total(counter_cost)

    # ---------- 3) 在“情景维度”上合并三个输出 ----------
    # 关键点：
    # - 先把三个 DataArray 的 'policy' 维重命名为 'scenario'
    # - 再各自添加一个辅助坐标 'category'，用于区分来源
    # - 最后在 'scenario' 维上 concat 成一个总的 DataArray
    def tag_and_rename(da, category_name: str) -> xr.DataArray:
        da2 = da.rename(policy="scenario")
        da2 = da2.assign_coords(
            category=("scenario", [category_name] * da2.sizes["scenario"])
        )
        return da2

    carbon_cost_sc   = tag_and_rename(carbon_cost,   "carbon")
    carbon_bio_sc    = tag_and_rename(carbon_bio_cost, "carbon_bio")
    counter_cost_sc  = tag_and_rename(counter_cost,  "counter_carbon_bio")

    # 合并：(scenario, Year, type)
    all_cost = xr.concat([carbon_cost_sc, carbon_bio_sc, counter_cost_sc], dim="scenario", join='outer', fill_value=0)
    type_vals = all_cost.coords["type"].values
    new_type_vals = [v.replace(" profit", "") for v in type_vals]
    all_cost = all_cost.assign_coords(type=("type", new_type_vals))
    all_cost.name = "data"

    # 存一个总的 nc
    all_nc = os.path.join(out_dir, "xr_total_cost.nc")
    save2nc(all_cost, all_nc)

    return {
        "profit": profit_da,
        "cost_carbon": carbon_cost,
        "cost_carbon_bio": carbon_bio_cost,
        "cost_counter_carbon_bio": counter_cost,
        "cost_all": all_cost,  # 合并后的总结果 (scenario, Year, type) + 辅助坐标 category
    }

def build_sol_profit_and_cost_nc(
    economic_da: xr.DataArray,
    input_files_0,              # baseline 列表（用第 0 个）
    input_files_1,              # 碳情景列表
    input_files_2,              # 碳+生物多样性 列表
    carbon_names,               # 与 input_files_1 对齐的输出名
    carbon_bio_names,           # 与 input_files_2 对齐的输出名
    counter_carbon_bio_names,   # 与 input_files_2 对齐的输出名
    add_total=False,            # 是否把 Total 作为额外的 type 写入
):
    """
    economic_da 维度必须是 (scenario, Year, type) 且 type 名包含：
      'Ag revenue','Ag cost','AgMgt revenue','AgMgt cost',
      'Non-ag revenue','Non-ag cost','Transition(ag→ag) cost',
      'Transition(ag→non-ag) amortised cost'
    """
    out_dir = f'../../../output/{config.TASK_NAME}/carbon_price/1_draw_data'
    os.makedirs(out_dir, exist_ok=True)

    # ---------- 1) 计算 profit(scenario, Year, type) ----------
    required_types = [
        # "AgMgt revenue", "AgMgt cost",
        "Non-ag revenue", "Non-ag cost",
        "Transition(ag→non-ag) amortised cost",
    ]
    type_names = set(economic_da.coords["type"].astype(str).values)
    missing = [t for t in required_types if t not in type_names]
    if missing:
        raise ValueError(f"economic_da 缺少必要 type: {missing}")

    def pick(name: str) -> xr.DataArray:
        # 选中一个类型 -> 去掉原来的 type 维度，避免后续按坐标对齐
        da = economic_da.sel(type=name).squeeze(drop=True)
        # 现在 da 的维度里已经没有 type 了（只剩 scenario/year 等）
        return da

    profit_list = [
        # (pick("AgMgt revenue") - pick("AgMgt cost")).expand_dims(type=["AgMgt profit"]),
        (pick("Non-ag revenue") - pick("Non-ag cost")).expand_dims(type=["Non-ag profit"]),
        (-pick("Transition(ag→non-ag) amortised cost")).expand_dims(type=["Transition(ag→non-ag) amortised profit"]),
    ]

    profit_da = xr.concat(profit_list, dim="type", join='outer', fill_value=0)
    profit_da.name = "data"

    # 如果需要固定维度顺序（比如 scenario, year, type）：
    want = [d for d in ["scenario", "Year", "type"] if d in profit_da.dims]
    profit_da = profit_da.transpose(*want)
    profit_da.name = "data"

    # 存 profit（修正：encoding 的 key 要用当前 name）
    profit_nc = os.path.join(out_dir, "xr_sol_profit.nc")
    save2nc(profit_da, profit_nc)

    # ---------- 2) 构造三个差值 DataArray ----------
    baseline = input_files_0[0]

    # A) carbon：baseline - input_files_1[i]  ->  (policy, Year, type)
    diffs = []
    for scen in input_files_1:
        diffs.append(profit_da.sel(scenario=baseline) - profit_da.sel(scenario=scen))
    carbon_cost = xr.concat(diffs, dim="policy", join='outer', fill_value=0).assign_coords(policy=list(carbon_names))

    # B) carbon_bio：input_files_1[i] - input_files_2[idx]
    if len(input_files_2) % len(input_files_1) != 0:
        raise ValueError("len(input_files_2) 必须是 len(input_files_1) 的整数倍，用于分组匹配。")
    bio_nums = len(input_files_2) // len(input_files_1)

    diffs = []
    for k, scen2 in enumerate(input_files_2):
        i = k // bio_nums
        scen1 = input_files_1[i]
        diffs.append(profit_da.sel(scenario=scen1) - profit_da.sel(scenario=scen2))
    carbon_bio_cost = xr.concat(diffs, dim="policy", join='outer', fill_value=0).assign_coords(policy=list(carbon_bio_names))

    # C) counter：input_files_2 的前 bio_nums 个与 baseline
    diffs = []
    for i in range(len(counter_carbon_bio_names)):
        scen2 = input_files_2[i]
        diffs.append(profit_da.sel(scenario=baseline) - profit_da.sel(scenario=scen2))
    counter_cost = xr.concat(diffs, dim="policy", join='outer', fill_value=0).assign_coords(policy=list(counter_carbon_bio_names))

    # 可选把 Total 作为额外的 type 追加
    def append_total(da, add_total=add_total):
        if not add_total:
            return da
        total = da.sum(dim="type")
        total = total.expand_dims({"type": ["Total"]})
        da2 = xr.concat([da, total], dim="type", join='outer', fill_value=0)
        return da2

    carbon_cost = append_total(carbon_cost)
    carbon_bio_cost = append_total(carbon_bio_cost)
    counter_cost = append_total(counter_cost)

    # ---------- 3) 在“情景维度”上合并三个输出 ----------
    # 关键点：
    # - 先把三个 DataArray 的 'policy' 维重命名为 'scenario'
    # - 再各自添加一个辅助坐标 'category'，用于区分来源
    # - 最后在 'scenario' 维上 concat 成一个总的 DataArray
    def tag_and_rename(da, category_name: str) -> xr.DataArray:
        da2 = da.rename(policy="scenario")
        da2 = da2.assign_coords(
            category=("scenario", [category_name] * da2.sizes["scenario"])
        )
        return da2

    carbon_cost_sc   = tag_and_rename(carbon_cost,   "carbon")
    carbon_bio_sc    = tag_and_rename(carbon_bio_cost, "carbon_bio")
    counter_cost_sc  = tag_and_rename(counter_cost,  "counter_carbon_bio")

    # 合并：(scenario, Year, type)
    all_cost = xr.concat([carbon_cost_sc, carbon_bio_sc, counter_cost_sc], dim="scenario", join='outer', fill_value=0)
    type_vals = all_cost.coords["type"].values
    new_type_vals = [v.replace(" profit", "") for v in type_vals]
    all_cost = all_cost.assign_coords(type=("type", new_type_vals))
    all_cost.name = "data"

    # 存一个总的 nc
    all_nc = os.path.join(out_dir, "xr_total_sol_cost.nc")
    save2nc(all_cost, all_nc)

    return {
        "profit": profit_da,
        "cost_carbon": carbon_cost,
        "cost_carbon_bio": carbon_bio_cost,
        "cost_counter_carbon_bio": counter_cost,
        "cost_all": all_cost,  # 合并后的总结果 (scenario, Year, type) + 辅助坐标 category
    }

def _make_prices_nc_impl(
    output_names,
    cost_file: str,
    carbon_file: str,
    bio_file: str,
    carbon_out: str,
    bio_out: str,
):
    """
    Shared implementation for make_prices_nc / make_sol_prices_nc.

    Reads three pre-computed (scenario, Year, ...) DataArrays, selects the
    requested scenarios, sums away any extra dimensions, then computes and
    saves carbon_price = cost / carbon and bio_price = cost / bio.
    """
    out_dir = f'../../../output/{config.TASK_NAME}/carbon_price/1_draw_data'

    def _load_and_collapse(fname: str) -> xr.DataArray:
        """Load file, select output_names scenarios, collapse extra dims."""
        with xr.open_dataarray(os.path.join(out_dir, fname), engine="h5netcdf") as da:
            da = da.sel(scenario=output_names).load()
        sum_dims = [d for d in da.dims if d not in {'scenario', 'Year'}]
        if sum_dims:
            da = da.sum(dim=sum_dims, skipna=True)
        if 'Year' in da.dims:
            da = da.transpose('scenario', 'Year')
        return da

    cost_all   = _load_and_collapse(cost_file)
    carbon_all = _load_and_collapse(carbon_file)
    bio_all    = _load_and_collapse(bio_file)

    carbon_price = cost_all / carbon_all
    bio_price    = cost_all / bio_all
    carbon_price.name = 'data'
    bio_price.name    = 'data'

    os.makedirs(out_dir, exist_ok=True)
    save2nc(carbon_price, os.path.join(out_dir, carbon_out))
    save2nc(bio_price,    os.path.join(out_dir, bio_out))

    return carbon_price, bio_price


def make_prices_nc(output_names):
    """
    Compute carbon_price and bio_price from xr_total_cost / carbon / bio NetCDFs.
    Selects the requested scenarios, collapses extra dims, saves results.
    """
    return _make_prices_nc_impl(
        output_names,
        cost_file   = 'xr_total_cost.nc',
        carbon_file = 'xr_total_carbon.nc',
        bio_file    = 'xr_total_bio.nc',
        carbon_out  = 'xr_carbon_price.nc',
        bio_out     = 'xr_bio_price.nc',
    )


def make_sol_prices_nc(output_names):
    """
    Compute carbon_price and bio_price from xr_total_sol_cost / carbon / bio NetCDFs.
    Selects the requested scenarios, collapses extra dims, saves results.
    """
    return _make_prices_nc_impl(
        output_names,
        cost_file   = 'xr_total_sol_cost.nc',
        carbon_file = 'xr_total_sol_carbon.nc',
        bio_file    = 'xr_total_sol_bio.nc',
        carbon_out  = 'xr_carbon_sol_price.nc',
        bio_out     = 'xr_bio_sol_price.nc',
    )

def summarize_netcdf_to_excel(
        input_file: str,
        years: List[int],
        files: List[str],
        n_jobs: int = 41,
        excel_type: str = "all",
) -> None:
    """
    从指定的目录结构中读取单变量NetCDF文件，计算其总和，并将结果保存到Excel。

    Args:
        input_path (str): 包含 'output_{year}' 子目录的基础路径。
        years (List[int]): 需要处理的年份列表。
        files (List[str]): 文件的基础名称列表 (不含年份和扩展名)。
        output_excel_path (str, optional): 输出的Excel文件名。
                                            默认为 "economic_summary.xlsx"。
    """
    input_dir = f'../../../output/{config.TASK_NAME}/carbon_price/0_base_data'
    input_path = os.path.join(input_dir, input_file)
    # print(f"Start {input_path}...")

    def _process_single_year(
            year: int,
            files: List[str],
            input_path: str,
            task_name: str,
    ) -> Dict[str, Any]:
        """
        Processes all specified files for a single year.
        This function is designed to be called in parallel by joblib.

        Returns:
            A dictionary containing the results for the given year (e.g., {'Year': 2025, 'file1': 123.4, ...}).
        """
        # print(f"Processing year: {year}")
        year_data = {'Year': year}

        for file in files:
            total_sum = None
            # Build the full file path based on the file name
            file_path = os.path.join(
                input_path,
                f'{year}',
                f'{file}_{year}.nc'
            )

            # Check for file existence before trying to open
            if os.path.exists(file_path):
                with xr.open_dataarray(file_path, engine="h5netcdf", chunks='auto') as da:
                    filtered_da = filter_all_from_dims(da).load()
                    total_sum = filtered_da.sum().item()
            else:
                print(f"  - WARNING: File '{file_path}' for year {year} does not exist.")

            # Add the result to the dictionary for the current year
            year_data[file] = total_sum

        return year_data

    all_data = Parallel(n_jobs=n_jobs)(
        delayed(_process_single_year)(year, files, input_path, config.TASK_NAME)
        for year in sorted(years)
    )
    # 将结果列表转换为 pandas DataFrame
    results_df = pd.DataFrame(all_data)

    # 将 'Year' 列设为索引
    results_df = results_df.set_index('Year')
    results_df = results_df / 1e6
    results_df = results_df.rename(columns=config.KEY_TO_COLUMN_MAP)
    output_excel_path = os.path.join(
        f'../../../output/{config.TASK_NAME}',
        'carbon_price',
        '1_excel',
        f'0_Origin_{excel_type}_{input_file}.xlsx')
    results_df.to_excel(output_excel_path)
    return results_df

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
        concat_dim="Year",  # 新增 year 维度
        parallel=parallel,  # 一般 False 更稳，避免句柄并发
        chunks={cell_dim: cell_chunk, "Year": year_chunk}  # year=1，cell 分块
    ).assign_coords(year=valid_years)

    if mask is not None:
        ds = ds.where(mask, other=0)  # 使用掩码，非掩码区域设为 0

    return ds

def create_processed_xarray(years, base_path, env_category, env_name, mask=None,
                  engine="h5netcdf",
                  cell_dim="cell", cell_chunk="auto",
                  year_chunk=1, parallel=False):
    """
    以 year 维度拼接多个年度 NetCDF，懒加载+分块，避免过多文件句柄。
    """
    file_paths = [
        os.path.join(base_path, str(env_category), str(y), f"xr_{env_name}_{env_category}_{y}.nc")
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
        concat_dim="Year",  # 新增 year 维度
        parallel=parallel,  # 一般 False 更稳，避免句柄并发
        chunks={cell_dim: cell_chunk, "Year": year_chunk}  # year=1，cell 分块
    ).assign_coords(year=valid_years)

    if mask is not None:
        ds = ds.where(mask, other=0)  # 使用掩码，非掩码区域设为 0

    return ds

def create_summary(env_category, years, base_path,env_type, colnames):
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
                              f"total_{env_type}_{env_category}")

    # 计算指标
    xr_carbon_price_ave_a = (xr_carbon_cost_a.sum(dim="cell") /
                             xr_carbon.sum(dim="cell"))
    da_price = xr_carbon_price_ave_a["data"].sortby("Year")
    da_benefits = (xr_carbon.sum(dim="cell")["data"] / 1e6).sortby("Year")
    da_cost = (xr_carbon_cost_a.sum(dim="cell")["data"] / 1e6).sortby("Year")

    # 检查列名长度
    if len(colnames) != 3:
        raise ValueError("colnames 必须是长度为 3 的列表，顺序为 [GHG benefits, Carbon cost, Average Carbon price]")

    # 组装 DataFrame
    df = pd.DataFrame({
        "Year": years,
        colnames[0]: da_benefits.values,
        colnames[1]: da_cost.values,
        colnames[2]: da_price.values
    }).set_index("Year")

    excel_path = base_path.replace("0_base_data", "1_excel")
    output_path = os.path.join(excel_path, f"2_{env_type}_{env_category}_cost_series.xlsx")
    df.to_excel(output_path)
    print(f"已保存: {output_path}")
    return df

def create_profit_for_cost(excel_dir,input_file: str) -> pd.DataFrame:
    excel_path = os.path.join(excel_dir, f'0_Origin_economic_{input_file}.xlsx')
    original_df = pd.read_excel(excel_path, index_col=0)
    profit_df = pd.DataFrame()

    # 规则 1: Ag profit = Ag revenue - Ag cost
    profit_df['Ag profit'] = original_df['Ag revenue'] - original_df['Ag cost']

    # 规则 2: AgMgt profit = AgMgt revenue - AgMgt cost
    profit_df['AgMgt profit'] = original_df['AgMgt revenue'] - original_df['AgMgt cost']

    # 规则 3: Non-ag profit = Non-ag revenue - Non-ag cost
    profit_df['Non-ag profit'] = original_df['Non-ag revenue'] - original_df['Non-ag cost']

    # 规则 4: Transition(ag→ag) profit = 0 - Transition(ag→ag) cost
    profit_df['Transition(ag→ag) profit'] = 0 - original_df['Transition(ag→ag) cost']

    # 规则 5: Transition(ag→non-ag) amortised profit = 0 - Transition(ag→non-ag) amortised cost
    profit_df['Transition(ag→non-ag) amortised profit'] = 0 - original_df['Transition(ag→non-ag) amortised cost']
    return profit_df