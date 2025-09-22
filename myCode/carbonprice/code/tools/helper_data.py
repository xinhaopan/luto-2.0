import os
from typing import List, Dict, Any, Optional
import numpy as np
import xarray as xr

import os
import xarray as xr
from joblib import Parallel, delayed
import pandas as pd
from typing import List, Dict, Any
from pathlib import Path
import math


from tools.tools import get_path, get_year, save2nc, filter_all_from_dims
from tools.tools import get_path,get_year,filter_all_from_dims
import tools.config as config

def summarize_to_type(
        scenarios: List[str],
        years: List[int],
        file: str,
        keep_dim: str,                     # 需要保留为 type 维度的那个维（例如 'am'）
        output_file: str,
        var_name: str = "data",
        scale: float = 1e6,
        dtype: str = "float32",
        chunks: Optional[Dict[str, int]] = None,  # 传入则使用 dask 分块（并行）
) -> xr.DataArray:
    """
    汇总 (scenario, year) 下的 {file}_{year}.nc：
    - 对除 keep_dim 外的所有维度求和，保留 keep_dim（作为最终的 'type' 维）
    - 除以 scale（例如 1e6）
    - 组合成 DataArray，维度为 (scenario, year, type)
    - 写出 NetCDF 并返回该 DataArray（dask 延迟计算，到 to_netcdf 时触发）
    """

    base_dir = f'../../../output/{config.TASK_NAME}/carbon_price/0_base_data'

    # ---------- 1) 先找一个示例文件，确定 type 维度的坐标 ----------
    sample_da = None
    sample_path = None
    for s in scenarios:
        for y in years:
            p = os.path.join(base_dir, s, f"{years[-1]}/{file}_{y}.nc")
            if os.path.exists(p):
                sample_path = p
                break
        if sample_path:
            break

    if sample_path is None:
        raise FileNotFoundError("未找到任何可用的输入文件，无法确定 type 坐标。")

    with xr.open_dataarray(sample_path, chunks=chunks) as da0:
        da0 = filter_all_from_dims(da0)
        if keep_dim not in da0.dims:
            raise ValueError(f"示例文件中不包含 keep_dim='{keep_dim}'，实际维度为 {da0.dims}")
        type_coord = da0.coords[keep_dim].values

    # ---------- 2) 逐个 (scenario, year) 构造 dask-backed 的部分并拼接 ----------
    per_scenario = []
    for s in scenarios:
        per_year = []
        for y in years:
            path = os.path.join(base_dir, s, str(y), f"{file}_{y}.nc")
            if not os.path.exists(path):
                raise ValueError(f"文件不存在: {path}")

            da = xr.open_dataarray(path, chunks=chunks)  # 不加 with，延迟直到计算
            da = filter_all_from_dims(da)

            # 求和保留 keep_dim -> 只剩 keep_dim 这个维度
            sum_dims = [d for d in da.dims if d != keep_dim]
            da_am = da.sum(dim=sum_dims, keep_attrs=False) / scale

            # 统一重命名 keep_dim -> 'type'
            if keep_dim != "type":
                da_am = da_am.rename({keep_dim: "type"})
            # 强制对齐到 sample 的 type 顺序（防止不同文件顺序不同）
            da_am = da_am.sel(type=type_coord)

            # 添加 year 维
            da_am = da_am.expand_dims({"year": [y]})
            da_am.name = var_name
            per_year.append(da_am)

        # 按 year 拼
        if len(per_year) == 0:
            continue
        da_year = xr.concat(per_year, dim="year")
        # 添加 scenario 维
        da_year = da_year.expand_dims({"scenario": [s]})
        per_scenario.append(da_year)

    if len(per_scenario) == 0:
        raise RuntimeError("所有场景均为空，无法生成结果。")

    # 按 scenario 拼
    out_da = xr.concat(per_scenario, dim="scenario")
    # 确保维度顺序
    out_da = out_da.transpose("scenario", "year", "type")

    # 设定坐标类型、名称
    out_da.name = var_name
    out_da = out_da.astype(dtype, copy=False)

    # ---------- 3) 写 NetCDF（此处会触发 dask 计算） ----------
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    save2nc(out_da, output_file)

    print(f"✅ Saved to {output_file}")

    return out_da

# from your_module import filter_all_from_dims

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

    base_dir = f'../../../output/{config.TASK_NAME}/carbon_price/0_base_data'
    years_sorted = sorted(years)
    type_names = [config.KEY_TO_COLUMN_MAP.get(f, f) for f in files]

    def _sum_single(scenario: str, year: int, file: str) -> float | None:
        input_path = os.path.join(base_dir, scenario)
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
        f'../../../output/{config.TASK_NAME}',
        'carbon_price',
        '1_draw_data')
    out_nc = os.path.join(
        f'{output_dir}',
        'carbon_price',
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
    # 修正：这里原来少了 f
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

    def g(name):  # 便捷选择
        return economic_da.sel(type=name)

    profit_vars = {
        "Ag profit": g("Ag revenue") - g("Ag cost"),
        "Agmgt profit": g("Agmgt revenue") - g("Agmgt cost"),
        "Non-ag profit": g("Non-ag revenue") - g("Non-ag cost"),
        "Transition(ag→ag) profit": - g("Transition(ag→ag) cost"),
        "Transition(ag→non-ag) amortised profit": - g("Transition(ag→non-ag) amortised cost"),
    }
    profit_da = xr.concat(list(profit_vars.values()), dim="type").assign_coords(
        type=list(profit_vars.keys())
    )
    profit_da.name = "data"

    # 存 profit（修正：encoding 的 key 要用当前 name）
    profit_nc = os.path.join(out_dir, "profit.nc")
    profit_da.astype("float32").to_netcdf(
        profit_nc,
        encoding={"data": {"dtype": "float32", "zlib": True, "complevel": 4}}
    )

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
    for i in range(bio_nums):
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
        concat_dim="year",  # 新增 year 维度
        parallel=parallel,  # 一般 False 更稳，避免句柄并发
        chunks={cell_dim: cell_chunk, "year": year_chunk}  # year=1，cell 分块
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
    da_price = xr_carbon_price_ave_a["data"].sortby("year")
    da_benefits = (xr_carbon.sum(dim="cell")["data"] / 1e6).sortby("year")
    da_cost = (xr_carbon_cost_a.sum(dim="cell")["data"] / 1e6).sortby("year")

    # 检查列名长度
    if len(colnames) != 3:
        raise ValueError("colnames 必须是长度为 3 的列表，顺序为 [GHG benefits, Carbon cost, Average Carbon price]")

    # 组装 DataFrame
    df = pd.DataFrame({
        "year": da_price["year"].values,
        colnames[0]: da_benefits.values,
        colnames[1]: da_cost.values,
        colnames[2]: da_price.values
    }).set_index("year")

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