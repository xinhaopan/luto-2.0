import xarray as xr
import matplotlib.pyplot as plt
import os
import xarray as xr

import os
import xarray as xr
from joblib import Parallel, delayed

def create_mask(years, base_path,env_category, chunks="auto"):
    masks = []
    for y in years:
        data_tmpl = f"{base_path}/{env_category}/{y}/xr_total_{env_category}_{y}.nc"
        data_xr  = xr.open_dataarray(data_tmpl.format(year=y), chunks=chunks)
        m = data_xr >= 1
        masks.append(m.expand_dims(year=[y]))
    mask = xr.concat(masks, dim="year")
    mask.name = "mask"
    mask.attrs["description"] = "True if both env and cost >= 1, else False"
    return mask


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


def sum_except_year(obj: xr.Dataset | xr.DataArray) -> xr.Dataset | xr.DataArray:
    """
    对 xarray 对象在除 'year' 以外的所有维度上求和。
    支持 DataArray 和 Dataset。
    """
    if isinstance(obj, xr.DataArray):
        return obj.sum(dim=set(obj.dims) - {"year"})
    elif isinstance(obj, xr.Dataset):
        return obj.map(lambda v: v.sum(dim=set(v.dims) - {"year"}))
    else:
        raise TypeError("输入必须是 xarray.DataArray 或 xarray.Dataset")


def process_file(file_path, year_idx, cat):
    """处理单个文件：加载、应用掩码、求和、添加 year 维度"""
    print(f"Processing {file_path}")
    ds_one = xr.open_dataset(file_path, chunks='auto')
    main_var = next(iter(ds_one.data_vars))
    da = ds_one[main_var]
    s = da.sum(dim=list(da.dims))
    return xr.Dataset({cat: xr.DataArray([s], dims=["year"], coords={"year": [year_idx]})})

def create_cost_year_series(
    years, base_path, env_category,
    year_chunk=-1, parallel=True,
    cost_categories=("cost_ag", "cost_agricultural_management", "cost_non_ag",
                     "cost_transition_ag2ag_diff", "transition_cost_ag2non_ag_amortised_diff"),
    finalize="compute"   # "lazy" | "persist" | "compute"
):
    """
    对每个 cost_category:
      - 逐文件打开，预处理时把所有非 year 维度求和成标量
      - 沿 year 维拼接出 1D 序列
    合并所有 cost_category，并追加 TOTAL。

    返回：
      ds_year: Dataset，变量为各 cost_category（1D: year）
      da_year: DataArray，维度为 (cost_category, year)，含 'TOTAL'
    """

    # 处理 mask：仅允许 DataArray/单变量 Dataset；转换为 bool

    per_cat_ds = []   # 每个元素都是 1 个变量（变量名=cat）的 Dataset，维度为 year


    for cat in cost_categories:
        # 生成文件路径
        paths = [os.path.join(base_path, str(env_category), str(y), f"xr_{cat}_{env_category}_{y}.nc")
                 for y in years]
        exist = [p for p in paths if os.path.exists(p)]
        if not exist:
            raise FileNotFoundError(f"未找到 {cat} 的 NetCDF 文件。")

        # 保证文件和年份一一对应
        valid_years = [int(os.path.basename(p).split("_")[-1].split(".")[0]) for p in exist]

        # 并行或串行处理文件
        if parallel:
            ds_list = Parallel(n_jobs=-1, backend="loky")(
                delayed(process_file)(file_path, valid_years[idx], cat)
                for idx, file_path in enumerate(exist)
            )
        else:
            ds_list = [process_file(file_path, valid_years[idx], cat)
                       for idx, file_path in enumerate(exist)]

        # 沿 year 维度拼接
        ds_cat = xr.concat(ds_list, dim="year").assign_coords(year=valid_years)
        if year_chunk > 0 and finalize != "compute":
            ds_cat = ds_cat.chunk({"year": year_chunk})
        per_cat_ds.append(ds_cat)

    # 合并所有 cost_category
    ds_year = xr.merge(per_cat_ds, compat="override", join="exact")

    # to_array 得到 (cost_category, year)
    da_year = ds_year.to_array(dim="cost_category")  # shape: (cost_category, year)

    # 若 year 为坐标不是维度，需保证 year 是维度
    if "year" not in da_year.dims:
        da_year = da_year.expand_dims(year=da_year.coords["year"])

    # 强制排序保证 (cost_category, year)
    da_year = da_year.transpose("cost_category", "year")

    # 追加 total（每年所有 cost_category 求和）
    da_total = da_year.sum(dim="cost_category", skipna=True)
    da_total = da_total.expand_dims(cost_category=["TOTAL"])
    da_year = xr.concat([da_year, da_total], dim="cost_category")

    # 是否立刻计算
    if finalize == "compute":
        return da_year.compute()
    if finalize == "persist":
        return da_year.persist()
    return da_year

task_name = '20250831_Price_Task_RES13'
base_path = f"../../../output/{task_name}/carbon_price/0_base_data"
excel_path = f"../../../output/{task_name}/carbon_price/1_excel"
os.makedirs(excel_path, exist_ok=True)

env_category = "bio_50"
years = list(range(2011, 2051))
xr_mask = create_mask(years, base_path,env_category)

env_category = "bio_50"
xr_bio_cost_series = create_cost_year_series(years, base_path, env_category)
bio_cost = xr_bio_cost_series.T.to_pandas() / 1e6
bio_cost.index.name = "Year"
bio_cost.to_excel(f"{excel_path}/{env_category}_sector_cost.xlsx")

env_category = "carbon_100"
