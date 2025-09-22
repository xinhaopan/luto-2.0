import os
from typing import List, Dict, Any, Optional
import numpy as np
import xarray as xr

import os
import xarray as xr
from joblib import Parallel, delayed
import pandas as pd
from typing import List, Dict, Any
from contextlib import suppress
from pathlib import Path
import math


from tools.tools import get_path, get_year, save2nc, filter_all_from_dims
from tools.tools import get_path,get_year,filter_all_from_dims
import tools.config as config


def summarize_to_type(
        scenarios,
        years,
        file,
        keep_dim,
        output_file,
        var_name="data",
        scale=1e6,
        dtype="float32",
        chunks='auto',
):
    print(f"Summarizing to type from {file}, keep_dim={keep_dim}...")
    base_dir = f'../../../output/{config.TASK_NAME}/carbon_price'

    # 1) 收集文件
    file_paths, file_scenarios, file_years = [], [], []
    for s in scenarios:
        for y in years:
            p = os.path.join(base_dir, '0_base_data', s, str(y), f"{file}_{s}_{y}.nc")
            if os.path.exists(p):
                file_paths.append(p)
                file_scenarios.append(s)
                file_years.append(y)
    if not file_paths:
        raise RuntimeError("未找到任何有效文件")

    print(f"Found {len(file_paths)} files, opening with mfdataset (chunks={chunks})...")

    # 2) 先探测目标变量
    probe = xr.open_dataset(file_paths[0], engine="netcdf4", decode_cf=False, mask_and_scale=False)
    try:
        numeric_vars = [k for k, v in probe.data_vars.items() if np.issubdtype(v.dtype, np.number)]
        if not numeric_vars:
            raise RuntimeError("数据集中没有数值变量可用于汇总")
        target_var = numeric_vars[0] if var_name == "data" else var_name
        if target_var not in probe.data_vars:
            target_var = numeric_vars[0]

        if keep_dim not in probe[target_var].dims:
            raise ValueError(f"变量 '{target_var}' 中不包含 keep_dim='{keep_dim}'，实际维度: {probe[target_var].dims}")
        if keep_dim in probe.coords:
            type_coord = probe.coords[keep_dim].values
        else:
            type_coord = np.arange(probe[target_var].sizes[keep_dim])
    finally:
        with suppress(Exception):
            probe.close()

    # 3) 只读目标变量
    def _preprocess(ds):
        return ds[[target_var]]

    ds = xr.open_mfdataset(
        file_paths,
        engine="netcdf4",
        combine='nested',
        concat_dim='file_idx',
        preprocess=_preprocess,
        decode_cf=False,
        mask_and_scale=False,
        chunks=chunks,
        parallel=True,
    )

    try:
        da = ds[target_var]
        print(f"Loaded data shape: {da.shape}, dims: {da.dims}")

        da = filter_all_from_dims(da)

        sum_dims = [d for d in da.dims if d not in [keep_dim, 'file_idx']]
        print(f"Summing over dimensions: {sum_dims}, keeping: [{keep_dim}, 'file_idx']")
        da_summed = (da if not sum_dims else da.sum(dim=sum_dims, keep_attrs=False)) / scale

        if keep_dim != "type":
            da_summed = da_summed.rename({keep_dim: "type"})
        try:
            da_summed = da_summed.sel(type=type_coord)
        except Exception:
            da_summed = da_summed.assign_coords(type=("type", type_coord))

        da_summed = da_summed.assign_coords(
            scenario=('file_idx', file_scenarios),
            Year=('file_idx', file_years)
        )
        da_indexed = da_summed.set_index(file_idx=['scenario', 'Year'])
        da_final = da_indexed.unstack('file_idx')

        da_final = da_final.transpose("scenario", "Year", "type")
        da_final.name = ("data" if var_name == "data" else var_name)
        da_final = da_final.astype(dtype, copy=False)

        print(f"Final data shape: {da_final.shape}")

        output_dir = os.path.join(base_dir, '1_draw_data')
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f'{output_file}.nc')
        print("Starting computation and saving...")
        save2nc(da_final, output_path)
        print(f"✅ Saved to {output_path}")

    finally:
        with suppress(Exception):
            ds.close()

    return da_final


def summarize_to_category(
        scenarios: List[str],
        years: List[int],
        files: List[str],
        output_file: str,
        n_jobs: int = 41,
        var_name: str = "data",
        scale: float = 1e6,
        dtype: str = "float32",
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
        nc_path = os.path.join(input_path, f'{year}', f'{file}_{year}.nc')
        if not os.path.exists(nc_path):
            return np.nan
        with xr.open_dataarray(nc_path) as da:
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
      'Ag revenue','Ag cost','Agmgt revenue','Agmgt cost',
      'Non-ag revenue','Non-ag cost','Transition(ag→ag) cost',
      'Transition(ag→non-ag) amortised cost'
    """
    out_dir = f'../../../output/{config.TASK_NAME}/carbon_price/1_draw_data'
    os.makedirs(out_dir, exist_ok=True)

    # ---------- 1) 计算 profit(scenario, Year, type) ----------
    required_types = [
        "Ag revenue", "Ag cost",
        "Agmgt revenue", "Agmgt cost",
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
        (pick("Agmgt revenue") - pick("Agmgt cost")).expand_dims(type=["Agmgt profit"]),
        (pick("Non-ag revenue") - pick("Non-ag cost")).expand_dims(type=["Non-ag profit"]),
        (-pick("Transition(ag→ag) cost")).expand_dims(type=["Transition(ag→ag) profit"]),
        (-pick("Transition(ag→non-ag) amortised cost")).expand_dims(type=["Transition(ag→non-ag) amortised profit"]),
    ]

    profit_da = xr.concat(profit_list, dim="type")
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
    carbon_cost = xr.concat(diffs, dim="policy").assign_coords(policy=list(carbon_names))

    # B) carbon_bio：input_files_1[i] - input_files_2[idx]
    if len(input_files_2) % len(input_files_1) != 0:
        raise ValueError("len(input_files_2) 必须是 len(input_files_1) 的整数倍，用于分组匹配。")
    bio_nums = len(input_files_2) // len(input_files_1)

    diffs = []
    for k, scen2 in enumerate(input_files_2):
        i = k // bio_nums
        scen1 = input_files_1[i]
        diffs.append(profit_da.sel(scenario=scen1) - profit_da.sel(scenario=scen2))
    carbon_bio_cost = xr.concat(diffs, dim="policy").assign_coords(policy=list(carbon_bio_names))

    # C) counter：input_files_2 的前 bio_nums 个与 baseline
    diffs = []
    for i in range(len(counter_carbon_bio_names)):
        scen2 = input_files_2[i]
        diffs.append(profit_da.sel(scenario=scen2) - profit_da.sel(scenario=baseline))
    counter_cost = xr.concat(diffs, dim="policy").assign_coords(policy=list(counter_carbon_bio_names))

    # 可选把 Total 作为额外的 type 追加
    def append_total(da, add_total=add_total):
        if not add_total:
            return da
        total = da.sum(dim="type")
        total = total.expand_dims({"type": ["Total"]})
        da2 = xr.concat([da, total], dim="type")
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
    all_cost = xr.concat([carbon_cost_sc, carbon_bio_sc, counter_cost_sc], dim="scenario")
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



def make_prices_nc(output_names):
    """
    从 xr_total_cost.nc / xr_total_carbon.nc / xr_total_bio.nc 读取数据，
    对每个情景只保留 (scenario, Year)，其余维度（若存在）全部按和聚合；
    然后计算：
        carbon_price = total_cost / total_carbon
        bio_price    = total_cost / total_bio
    合并所有情景 -> 两个 DataArray：(scenario, Year)，并保存到 out_dir。
    """
    out_dir = f'../../../output/{config.TASK_NAME}/carbon_price/1_draw_data'
    base = f'../../../output/{config.TASK_NAME}/carbon_price/1_draw_data'
    cost_da = xr.open_dataarray(os.path.join(base, 'xr_total_cost.nc'))
    carbon_da = xr.open_dataarray(os.path.join(base, 'xr_total_carbon.nc'))
    bio_da = xr.open_dataarray(os.path.join(base, 'xr_total_bio.nc'))

    def _collapse_to_year(da_all: xr.DataArray, scen_name: str) -> xr.DataArray:
        """
        选中单个情景，除 Year 外的所有维度求和，返回维度仅 (scenario, Year) 的 DataArray。
        """
        da = da_all.sel(scenario=scen_name)

        # 需要保留的维度：Year（和 scenario 这个坐标）
        keep = {'Year'}
        # 允许有/无 Year 这一坐标（某些数据可能命名不同，必要时在外面改名）
        sum_dims = [d for d in da.dims if d not in keep]

        if sum_dims:
            da = da.sum(dim=sum_dims, skipna=True)

        # 确保有 scenario 维度（如果被掉了就补回来）
        if 'scenario' not in da.dims:
            da = da.expand_dims({'scenario': [scen_name]})
        else:
            # 把单场景的坐标设置为该 scen_name（防止保留了原坐标数组）
            da = da.assign_coords(scenario=[scen_name])

        # 维度顺序统一成 (scenario, Year)（若 Year 不存在，这里会报错，需保证 Year 维存在）
        if 'Year' in da.dims:
            da = da.transpose('scenario', 'Year')
        return da

    # 聚合每个情景
    cost_list, car_list, bio_list = [], [], []
    for scen in output_names:
        cost_list.append(_collapse_to_year(cost_da, scen))
        car_list.append(_collapse_to_year(carbon_da, scen))
        bio_list.append(_collapse_to_year(bio_da, scen))

    # 合并所有情景
    cost_all = xr.concat(cost_list, dim='scenario')
    carbon_all = xr.concat(car_list, dim='scenario')
    bio_all = xr.concat(bio_list, dim='scenario')

    # 安全除法（分母<=0 或 NaN 时返回 NaN）
    def _safe_div(num: xr.DataArray, den: xr.DataArray) -> xr.DataArray:
        return xr.where((den > 0) & np.isfinite(den), num / den, np.nan)

    carbon_price = _safe_div(cost_all, carbon_all)
    bio_price = _safe_div(cost_all, bio_all)

    carbon_price.name = 'carbon_price'
    bio_price.name = 'biodiversity_price'

    # 保存
    os.makedirs(out_dir, exist_ok=True)
    save2nc(carbon_price, os.path.join(out_dir, 'xr_carbon_price.nc'))
    save2nc(bio_price, os.path.join(out_dir, 'xr_bio_price.nc'))

    return carbon_price, bio_price

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
                with xr.open_dataarray(file_path) as da:
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
    da_price = xr_carbon_price_ave_a["data"].sortby("Year")
    da_benefits = (xr_carbon.sum(dim="cell")["data"] / 1e6).sortby("Year")
    da_cost = (xr_carbon_cost_a.sum(dim="cell")["data"] / 1e6).sortby("Year")

    # 检查列名长度
    if len(colnames) != 3:
        raise ValueError("colnames 必须是长度为 3 的列表，顺序为 [GHG benefits, Carbon cost, Average Carbon price]")

    # 组装 DataFrame
    df = pd.DataFrame({
        "Year": da_price["Year"].values,
        colnames[0]: da_benefits.values,
        colnames[1]: da_cost.values,
        colnames[2]: da_price.values
    }).set_index("Year")

    excel_path = base_path.replace("0_base_data", "1_excel")
    output_path = os.path.join(excel_path, f"2_{env_category}_cost_series.xlsx")
    df.to_excel(output_path)
    print(f"已保存: {output_path}")
    return df

def create_profit_for_cost(excel_dir,input_file: str) -> pd.DataFrame:
    excel_path = os.path.join(excel_dir, f'0_Origin_economic_{input_file}.xlsx')
    original_df = pd.read_excel(excel_path, index_col=0)
    profit_df = pd.DataFrame()

    # 规则 1: Ag profit = Ag revenue - Ag cost
    profit_df['Ag profit'] = original_df['Ag revenue'] - original_df['Ag cost']

    # 规则 2: Agmgt profit = Agmgt revenue - Agmgt cost
    profit_df['Agmgt profit'] = original_df['Agmgt revenue'] - original_df['Agmgt cost']

    # 规则 3: Non-ag profit = Non-ag revenue - Non-ag cost
    profit_df['Non-ag profit'] = original_df['Non-ag revenue'] - original_df['Non-ag cost']

    # 规则 4: Transition(ag→ag) profit = 0 - Transition(ag→ag) cost
    profit_df['Transition(ag→ag) profit'] = 0 - original_df['Transition(ag→ag) cost']

    # 规则 5: Transition(ag→non-ag) amortised profit = 0 - Transition(ag→non-ag) amortised cost
    profit_df['Transition(ag→non-ag) amortised profit'] = 0 - original_df['Transition(ag→non-ag) amortised cost']
    return profit_df