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
    print(f"Start {input_path}...")

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
        print(f"Processing year: {year}")
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

economic_files = config.economic_files
carbon_files = config.carbon_files
bio_files = config.bio_files

input_files_0 = config.input_files_0
input_files_1 = config.input_files_1
input_files_2 = config.input_files_2
input_files = config.input_files

njobs = math.ceil(41 / 4)
task_dir = f'../../../output/{config.TASK_NAME}/carbon_price/0_base_data'
years = list(range(2011,2051,1))
for input_file in input_files_1:
    df = summarize_netcdf_to_excel(input_file, years, carbon_files,njobs,'carbon')
for input_file in input_files_2:
    df = summarize_netcdf_to_excel(input_file, years, bio_files,njobs,'biodiversity')
for input_file in input_files:
    df = summarize_netcdf_to_excel(input_file, years, economic_files,njobs,'economic')
