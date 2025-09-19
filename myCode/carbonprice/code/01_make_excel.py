import os
import xarray as xr
from joblib import Parallel, delayed
import pandas as pd
from typing import List, Dict, Any
from pathlib import Path
import math

from tools.tools import get_path,get_year,filter_all_from_dims
import tools.config as config


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

economic_files = config.economic_files
carbon_files = config.carbon_files
bio_files = config.bio_files

input_files_0 = config.input_files_0
input_files_1 = config.input_files_1
input_files_2 = config.input_files_2
input_files = config.input_files

carbon_names = ['carbon_low', 'carbon_high']
carbon_bio_names = [
        'carbon_low_bio_10', 'carbon_low_bio_20', 'carbon_low_bio_30', 'carbon_low_bio_40', 'carbon_low_bio_50',
        'carbon_high_bio_10', 'carbon_high_bio_20', 'carbon_high_bio_30', 'carbon_high_bio_40', 'carbon_high_bio_50'
    ]

base_path = f"../../../output/{config.TASK_NAME}/carbon_price/0_base_data"
excel_path = f"../../../output/{config.TASK_NAME}/carbon_price/1_excel"
os.makedirs(excel_path, exist_ok=True)

# ---------------------------------------make excel 0_origin economic/carbon/bio---------------------------------------
njobs = math.ceil(41/2)
task_dir = f'../../../output/{config.TASK_NAME}/carbon_price/0_base_data'
years = list(range(2011,2051,1))
# for input_file in input_files_1:
#     print(f"carbon: {input_file}")
#     df = summarize_netcdf_to_excel(input_file, years, carbon_files,njobs,'carbon')
# for input_file in input_files_2:
#     print(f"biodiversity: {input_file}")
#     df = summarize_netcdf_to_excel(input_file, years, bio_files,njobs,'biodiversity')
# for input_file in input_files:
#     print(f"economic: {input_file}")
#     df = summarize_netcdf_to_excel(input_file, years, economic_files,njobs,'economic')

# ---------------------------------------make excel 1_cost---------------------------------------
profit_0_list = []
for input_file in input_files_0:
    # 在实际使用中，取消下面的注释
    profit_0_list.append(create_profit_for_cost(excel_path, input_file))
profit_1_list = []
for input_file in input_files_1:
    # 在实际使用中，取消下面的注释
    profit_1_list.append(create_profit_for_cost(excel_path, input_file))
profit_2_list = []
for input_file in input_files_2:
    # 在实际使用中，取消下面的注释
    profit_2_list.append(create_profit_for_cost(excel_path, input_file))

bio_nums = int(len(input_files_2)/len(input_files_1))
for i in range(len(input_files_1)):
    df = profit_0_list[0] - profit_1_list[i]
    df.columns = df.columns.str.replace('profit', '')
    df['Total'] = df.sum(axis=1)
    df.to_excel(os.path.join(excel_path, f'1_Cost_{carbon_names[i]}.xlsx'))
for i in range(len(input_files_1)):
    for j in range(bio_nums):
        idx = i * bio_nums + j
        df = profit_1_list[i] - profit_2_list[idx]
        df.columns = df.columns.str.replace('profit', '')
        df['Total'] = df.sum(axis=1)
        df.to_excel(os.path.join(excel_path, f'1_Cost_{carbon_bio_names[idx]}.xlsx'))


# -----------------------------------make excel 1_processed carbon/bio---------------------------------------
for input_file in input_files_1:
    df = pd.read_excel(os.path.join(excel_path, f'0_Origin_carbon_{input_file}.xlsx'), index_col=0)
    df.columns = df.columns.str.replace(' GHG', '')
    new_rows_list = []

    # 从第二行开始循环 (索引 i 从 1 到 df 的末尾)
    for i in range(1, len(df)):
        # 取出当前行并取负
        new_row = df.iloc[i].copy()
        new_row = new_row * -1

        # 关键步骤：新行的第一列 = (原值取负) + (原df中上一行第一列的值)
        new_row.iloc[0] = -df.iloc[i, 0] + df.iloc[i - 1, 0]

        # 将计算出的新行（这是一个 Series）添加到列表中
        new_rows_list.append(new_row)

    # 使用收集到的行列表一次性创建新的 DataFrame
    # 这样做比在循环中反复 concat 更高效
    new_df = pd.DataFrame(new_rows_list)

    # 将新 DataFrame 的索引设置为与原数据对应（从 1 开始）
    new_df.index = df.index[1:]
    new_df['Total'] = new_df.sum(axis=1)
    new_df.to_excel(os.path.join(excel_path, f'1_Processed_carbon_{input_file}.xlsx'))

for input_file in input_files_2:
    df = pd.read_excel(os.path.join(excel_path, f'0_Origin_biodiversity_{input_file}.xlsx'), index_col=0)
    df.columns = df.columns.str.replace(' biodiversity', '')
    new_rows_list = []

    # 从第二行开始循环 (索引 i 从 1 到 df 的末尾)
    for i in range(1, len(df)):
        # 取出当前行并取负
        new_row = df.iloc[i].copy()

        new_row.iloc[0] = df.iloc[i, 0] - df.iloc[i - 1, 0]

        # 将计算出的新行（这是一个 Series）添加到列表中
        new_rows_list.append(new_row)

    # 使用收集到的行列表一次性创建新的 DataFrame
    # 这样做比在循环中反复 concat 更高效
    new_df = pd.DataFrame(new_rows_list)

    # 将新 DataFrame 的索引设置为与原数据对应（从 1 开始）
    new_df.index = df.index[1:]
    new_df['Total'] = new_df.sum(axis=1)
    new_df.to_excel(os.path.join(excel_path, f'1_Processed_bio_{input_file}.xlsx'))

# -----------------------------------make excel 2_cost & carbon/bio & average price---------------------------------------
# colnames = ["GHG benefits (Mt CO2e)", "Carbon cost (M AUD$)", "Average Carbon price (AUD$/t CO2e)"]
# for carbon_name in carbon_names:
#     create_summary(carbon_name, years, base_path, colnames)
#
# colnames = ["Biodiversity benefits (Mt CO2e)", "Biodiversity cost (M AUD$)", "Average Biodiversity price (AUD$/t CO2e)"]
# for bio_name in carbon_bio_names:
#     create_summary(bio_name, years, base_path, colnames)
