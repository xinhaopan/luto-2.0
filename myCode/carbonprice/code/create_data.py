import os
import xarray as xr
from joblib import Parallel, delayed
import numpy as np

from tools.tools import save2nc
import tools.config as config


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


def _process_single_file(year, cat, base_path, suffix, chunks):
    """
    (辅助函数) 处理单个文件：打开、求和、返回结果。
    这是一个独立的“工作单元”，可以被并行调用。
    """
    path = os.path.join(base_path, suffix, str(year), f"{cat}_{suffix}_{year}.nc")

    # 使用 with 语句确保文件被正确关闭
    with xr.open_dataset(path, chunks=chunks) as ds:
        da = ds['data']
        # 对所有维度求和得到一个标量 DataArray
        scalar_sum = da.sum(dim=list(da.dims))
        return (cat, year, scalar_sum)


def load_sum_series(
        base_path,
        years,
        cost_categories,
        suffix: str = "carbon_100.nc",
        chunks="auto",
        finalize: str = "compute",  # "compute" | "persist" | "lazy"
        n_jobs: int = -1,  # 并行任务数，-1 表示使用所有可用的核心
):
    """
    并行版本：
    打开 base_path/<year>/ 下每个 cost_category 对应的 NetCDF，
    对主变量在所有维度上求和，得到逐年的标量序列。
    """
    # 1. 创建所有要处理的任务列表
    tasks = [
        (year, cat) for cat in cost_categories for year in years
    ]

    # 2. 使用 joblib 并行执行 _process_single_file 函数
    #    对于 I/O 密集型任务（如读文件），'threading' 后端通常更高效且能避免 HDF 错误
    print(f"开始使用 {n_jobs} 个并行工作单元处理 {len(tasks)} 个文件...")
    results = Parallel(n_jobs=n_jobs, backend="threading")(
        delayed(_process_single_file)(year, cat, base_path, suffix, chunks)
        for year, cat in tasks
    )
    print("所有并行任务已完成。")

    # 3. 收集和重组结果
    per_cat_series = {cat: [] for cat in cost_categories}
    per_cat_years = {cat: [] for cat in cost_categories}

    # 过滤掉失败的任务 (返回 None 的) 并将结果按类别分组
    for res in results:
        if res is not None:
            cat, year, scalar_sum = res
            per_cat_series[cat].append(scalar_sum)
            per_cat_years[cat].append(year)

    # 4. 为每个类别创建 DataArray
    final_cat_das = {}
    for cat in cost_categories:
        if per_cat_series[cat]:  # 确保有数据
            # 把每年的标量拼成 (year) 的 DataArray
            cat_da = xr.concat(per_cat_series[cat], dim="year")
            cat_da = cat_da.assign_coords(year=np.array(per_cat_years[cat], dtype=int))
            final_cat_das[cat] = cat_da

    ds_year = xr.Dataset(final_cat_das)
    ds_year = ds_year.sortby("year").transpose("year", ...)

    if finalize == "compute":
        print("正在计算最终结果 (ds.compute())...")
        ds_year = ds_year.compute()
    elif finalize == "persist":
        print("正在持久化结果到内存 (ds.persist())...")
        ds_year = ds_year.persist()

    da_year = ds_year.to_array("cost_category").transpose("year", "cost_category").rename("data")

    return da_year

task_name = config.TASK_NAME
base_path = f"../../../output/{task_name}/carbon_price/0_base_data"

years = list(range(2011, 2051))

env_names = config.carbon_names + config.carbon_bio_names
cost_category = ['xr_cost_ag', 'xr_cost_agricultural_management', 'xr_cost_non_ag','xr_cost_transition_ag2ag_diff','xr_transition_cost_ag2non_ag_amortised_diff']
for env_name in env_names:
    for cost_category in env_names:
        xr_carbon_cost_category = load_sum_series(base_path,years,cost_category,suffix)

cost_category = ['xr_cost_ag', 'xr_cost_agricultural_management', 'xr_cost_non_ag','xr_cost_transition_ag2ag_diff','xr_transition_cost_ag2non_ag_amortised_diff']
suffix = 'carbon_high_bio_50'
xr_bio_cost_category = load_sum_series(base_path,years,cost_category,suffix)
save2nc(xr_carbon_cost_category,os.path.join(base_path, 'Results', "xr_carbon_cost_category.nc"))
save2nc(xr_bio_cost_category,os.path.join(base_path, 'Results', "xr_bio_cost_category.nc"))

# xr_carbon_cost_tn = create_xarray(years, base_path,'carbon_high','transition_cost_ag2non_ag_amortised_diff_carbon_high')
# xr_carbon_cost_tn = xr_carbon_cost_tn.sum(dim=['From land-use','cell'])
# save2nc(xr_carbon_cost_tn,os.path.join(base_path, 'Results', "xr_carbon_cost_tn.nc"))
#
# xr_carbon_cost_tn = create_xarray(years, base_path,'carbon_high','transition_cost_ag2non_ag_amortised_diff_carbon_high')
# xr_carbon_cost_tn = xr_carbon_cost_tn.sum(dim=['From land-use','cell'])
# save2nc(xr_carbon_cost_tn,os.path.join(base_path, 'Results', "xr_carbon_cost_tn.nc"))
#
# xr_carbon_cost_tn = create_xarray(years, base_path,'carbon_high_bio_50','transition_cost_ag2non_ag_amortised_diff_carbon_high_bio_50')
# xr_carbon_cost_tn = xr_carbon_cost_tn.sum(dim=['From land-use','cell'])
# save2nc(xr_carbon_cost_tn,os.path.join(base_path, 'Results', "xr_bio_cost_tn.nc"))

