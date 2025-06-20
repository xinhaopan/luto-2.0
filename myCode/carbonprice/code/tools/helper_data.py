import numpy as np
import numpy_financial as npf
import os
import pandas as pd
import re
from joblib import Parallel, delayed
from tqdm import *

from .tools import get_path, get_year, npy_to_map
import tools.config as config


def apply_operations_on_files(path_dir, files_with_ops):
    """
    读取给定路径下的多个 .npy 文件，并根据指定操作（+ 或 -）进行加减运算。

    :param path_dir: 文件所在的目录
    :param files_with_ops: 包含 (文件名, 操作) 的元组列表，操作为 '+' 或 '-'
    :return: 所有文件内容的加减总和
    """
    total_sum = None

    for file, operation in files_with_ops:
        file_path = os.path.join(path_dir, file)
        if "restored" not in file_path:
            if os.path.exists(file_path):
                # 读取 .npy 文件
                data = np.load(file_path)
                data = np.nan_to_num(data, nan=0.0)
                # 如果是第一次加载数据，初始化 total_sum
                if total_sum is None:
                    total_sum = data if operation == '+' else -data
                else:
                    if operation == '+':
                        total_sum += data
                    elif operation == '-':
                        total_sum -= data

            else:
                print(f"File not found: {file_path}")
                # 如果文件不存在，将值补充为 0
                if total_sum is None:
                    total_sum = np.zeros(1) if operation == '+' else -np.zeros(1)
                else:
                    if operation == '+':
                        total_sum += np.zeros(1)
                    elif operation == '-':
                        total_sum -= np.zeros(1)

    return total_sum


def apply_sum_on_files_with_prefix(path_dir, file_prefix):
    """
    对路径下所有以指定前缀开头的文件进行加法运算。

    :param path_dir: 文件所在的目录
    :param file_prefix: 文件名前缀（如 'cost_non_ag'）
    :return: 所有以指定前缀开头的文件内容的总和
    """
    # 获取目录下所有以指定前缀开头的文件名
    # 创建一个包含 cost_dict 和 revenue_dict 的字典
    combined_dict = {
        'cost_am': config.COST_DICT.get('cost_am', []),
        'cost_non_ag': config.COST_DICT.get('cost_non_ag', []),
        'revenue_am': config.REVENUE_DICT.get('revenue_am', []),
        'revenue_non_ag': config.REVENUE_DICT.get('revenue_non_ag', [])
    }

    # 优化后的逻辑
    files_with_ops = []
    if file_prefix in combined_dict:
        for file in os.listdir(path_dir):
            # 遍历 combined_dict[file_prefix] 中的每一个元素
            for prefix in combined_dict[file_prefix]:
                # 检查文件名是否以 file_prefix + "_" + prefix 开头
                if file.startswith(file_prefix + "_" + prefix):
                    files_with_ops.append((file, '+'))  # 如果匹配则加入列表
    else:
        files_with_ops = [(file, '+') for file in os.listdir(path_dir) if file.startswith(file_prefix)]

    if not files_with_ops:
        print(f"No files found starting with '{file_prefix}'")
        return None

    # 调用 apply_operations_on_files 函数
    total_sum = apply_operations_on_files(path_dir, files_with_ops)

    return total_sum


def process_and_save(path_dir, save_path, prefix, year, rows_nums):
    """
    对指定前缀的文件进行求和，保存结果，并将结果加入 rows_nums。

    :param path_dir: 文件所在的目录
    :param save_path: 保存 .npy 文件的路径
    :param prefix: 文件名前缀（如 'cost_ag', 'cost_non_ag' 等）
    :param year: 当前年份，用于命名保存的文件
    :param rows_nums: 用于存储各项结果的列表
    :return: 更新后的 rows_nums
    """
    arr = apply_sum_on_files_with_prefix(path_dir, prefix)
    if arr is None:
        print(f"No valid data found for {prefix} in year {year}")
        arr = np.zeros(1)

    rows_nums.append(np.sum(arr) / 1000000)
    if prefix.endswith('_'):
        prefix = prefix[:-1]
    np.save(os.path.join(save_path, f"{prefix}_{year}.npy"), arr)
    return rows_nums


def process_files_with_operations(path_dir, save_path, files_with_ops, file_prefix, year, row_data, negate=False):
    """
    处理文件加减操作，将结果保存为 .npy 文件并更新 row_data 字典。

    :param path_dir: 文件所在的目录
    :param save_path: 保存 .npy 文件的路径
    :param files_with_ops: 包含 (文件名, 操作符) 的元组列表，操作符为 '+' 或 '-'
    :param file_prefix: 保存文件的前缀
    :param year: 当前年份，用于命名保存的文件
    :param row_data: 用于存储各项结果的字典
    :param negate: 是否对结果进行取负操作
    :return: 更新后的 row_data
    """
    files_with_ops = [(file + ".npy", op) for file, op in files_with_ops]

    # 处理文件加减操作
    arr = apply_operations_on_files(path_dir, files_with_ops)

    if arr is None:
        print(f"No valid data found for {file_prefix} in year {year}")
        row_data[file_prefix] = 0
        return row_data

    # 如果需要取负操作，则取负
    if negate:
        arr = -arr

    # 保存结果到 .npy 文件
    if file_prefix.endswith('_'):
        file_prefix = file_prefix[:-1]
    np.save(os.path.join(save_path, f"{file_prefix}_{year}.npy"), arr)

    # 将结果添加到 row_data 字典，按百万单位
    arr = np.nan_to_num(arr, nan=0.0)
    row_data[file_prefix] = np.sum(arr) / 1000000

    return row_data

def list_files_with_prefix(directory, prefix):
    """
    列出给定目录中以指定前缀开头并以.npy结尾的所有文件。

    参数:
    directory (str): 要搜索文件的目录。
    prefix (str): 要匹配文件名的前缀。

    返回:
    list: 以指定前缀开头并以.npy结尾的文件名列表。
    """
    try:
        files = [f for f in os.listdir(directory) if f.startswith(prefix) and f.endswith('.npy')]
        return files
    except FileNotFoundError:
        return f"Error: The directory '{directory}' does not exist."
    except Exception as e:
        return f"An error occurred: {e}"

def process_year_data(year, path, categories, column_keywords):
    """处理每一年的数据"""
    year_data = {'Year': year}
    for category, files in categories.items():
        total_sum = 0
        for file in files:
            file_path = f'{path}/out_{year}/{file}_{year}.csv'
            data = pd.read_csv(file_path)
            # 使用文件名确定关键字
            keyword = column_keywords.get(file, column_keywords['default'])
            # print(file,keyword)
            value_columns = data.filter(like=keyword)
            total_sum += value_columns.sum().sum()
        year_data[category] = total_sum
    return year_data

def amortize_costs(input_file, file_name = "cost_transition_ag2non_ag", rate=0.07, horizon=30):
    """
    计算成本均摊并保存新的成本数组。

    参数：
    path_name: str - 文件路径
    file_name: str - 文件名
    rate: float - 利率，默认为0.07
    horizon: int - 时间跨度，默认为30年
    """
    # 初始化一个空字典来存储所有年份的成本数据
    print("Start to calculate cost amortize.")
    costs = {}
    path_name = get_path(input_file)
    years = get_year(path_name)
    start_year = min(years)
    end_year = max(years)

    for year in trange(years[0], years[-1] + 1):
        full_file_name = os.path.join(path_name, f"out_{year}/data_for_carbon_price/{file_name}_{year}.npy")
        costs[year] = np.load(full_file_name)

    # 初始化新的成本数组字典
    new_costs = {year: np.zeros_like(costs[start_year]) for year in years}

    # 计算每个年份的均摊成本，并加到相应年份上
    for year in years:
        pv_values = costs[year]
        annual_payment = -1 * npf.pmt(rate, horizon, pv=pv_values, fv=0, when='begin')
        for offset in range(horizon):
            target_year = year + offset
            if target_year <= end_year:
                new_costs[target_year] += annual_payment

    # 保存新的成本数组到新的.npy文件
    for year in years:
        file_name_with_amortised = file_name.replace("cost", "cost_amortised", 1)
        output_file_name = os.path.join(path_name, f"out_{year}/data_for_carbon_price/{file_name_with_amortised}_{year}.npy")
        np.save(output_file_name, new_costs[year])


def calculate_baseline_costs(input_file, use_parallel=False,output=True):
    print("Start jobs to calculate cost_revenue.")
    path_name = get_path(input_file)
    years=get_year(path_name)
    # 初始化结果列表

    columns_name = ["Year", "cost_ag(M$)", "cost_am(M$)", "cost_non_ag(M$)", "cost_transition_ag2ag(M$)","cost_transition_ag2non_ag(M$)",
                    "cost_amortised_transition_ag2non_ag(M$)","revenue_ag(M$)","revenue_am(M$)","revenue_non_ag(M$)",
                    "GHG_ag(MtCOe2)", "GHG_am(MtCOe2)", "GHG_non-ag(MtCOe2)", "GHG_transition(MtCOe2)",
                    "BIO_ag(M ha)", "BIO_am(M ha)", "BIO_non_ag(M ha)"]

    save_path = os.path.join(path_name, "data_for_carbon_price")
    os.makedirs(save_path, exist_ok=True)

    prefixes = columns_name[1:]

    # 生成所有 (year, prefix) 任务
    tasks = [
        (year, re.match(r"([^\(]+)", prefix).group(1).strip())
        for year in range(years[0], years[-1] + 1)
        for prefix in prefixes
    ]

    # 并行执行 process_and_save，并收集结果
    if use_parallel:
        # 并行执行 process_and_save，并收集结果
        results = Parallel(n_jobs=config.N_JOBS)(
            delayed(process_and_save)(
                os.path.join(path_name, f"out_{year}/data_for_carbon_price"),
                save_path,
                prefix,
                year,
                [year]  # 每个任务初始化自己的 rows_nums
            )
            for year, prefix in tasks
        )
    else:
        # 顺序执行 process_and_save，并使用 tqdm 显示进度条
        results = []
        for year, prefix in tqdm(tasks, desc="Processing tasks"):
            result = process_and_save(
                os.path.join(path_name, f"out_{year}/data_for_carbon_price"),
                save_path,
                prefix,
                year,
                [year]  # 每个任务初始化自己的 rows_nums
            )
            results.append(result)

    # 转换为 DataFrame
    df_results = pd.DataFrame(results).groupby(0)[1].agg(list).reset_index().apply(lambda x: [x[0]] + list(x[1]), axis=1).apply(
        pd.Series).set_axis(columns_name, axis=1)

    if output:
        output_dir = f"{config.TASK_DIR}/carbon_price/excel"
        os.makedirs(output_dir, exist_ok=True)
        output_excel_path = os.path.join(output_dir, f"01_origin_{input_file}.xlsx")
        df_results.to_excel(output_excel_path, index=False)
    return df_results


def compute_unit_prices(input_file, output=True):
    path_name = get_path(input_file)
    years = get_year(path_name)
    path_dir = os.path.join(path_name, "data_for_carbon_price")
    save_path = os.path.join(path_name, "data_for_carbon_price")
    print("Start to calculate unit price.")

    # 定义固定的字段名称，确保顺序和一致性
    columns_name = ["Year"] + config.COST_COLUMN + ["All cost(M$)", "GHG Abatement(MtCOe2)", "BIO(Mha)"]
    results = []

    for year in trange(years[1], years[-1] + 1):
        row_data = {"Year": year}

        # 使用 process_files_with_operations 更新 row_data 字典
        row_data = process_files_with_operations(path_dir, save_path,
                                                 [("revenue_ag_" + str(year), '+'), ("cost_ag_" + str(year), '-'),
                                                  ("revenue_ag_" + str(year - 1), '-'),
                                                  ("cost_ag_" + str(year - 1), '+')],
                                                 "opportunity_cost", year, row_data)

        row_data = process_files_with_operations(path_dir, save_path,
                                                 [("revenue_am_" + str(year), '-'), ("cost_am_" + str(year), '+')],
                                                 "am_net_cost", year, row_data)

        row_data = process_files_with_operations(path_dir, save_path,
                                                 [("revenue_non_ag_" + str(year), '-'),
                                                  ("cost_non_ag_" + str(year), '+')],
                                                 "non_ag_net_cost", year, row_data)

        row_data = process_files_with_operations(path_dir, save_path,
                                                 [("cost_amortised_transition_ag2non_ag_" + str(year), '+'),("cost_transition_ag2ag_" + str(year), '+')],
                                                 "transition_cost", year, row_data)


        row_data = process_files_with_operations(path_dir, save_path,
                                                 [("opportunity_cost_" + str(year), '+'), ("transition_cost_" + str(year), '+'),
                                                  ("am_net_cost_" + str(year), '+'),("non_ag_net_cost_" + str(year), '+')],
                                                 "cost", year, row_data)

        row_data = process_files_with_operations(path_dir, save_path,
                                                 [("GHG_ag_" + str(year), '+'), ("GHG_ag_" + str(year - 1), '-'),
                                                  ("GHG_am_" + str(year), '+'), ("GHG_non-ag_" + str(year), '+'),
                                                  ("GHG_transition_" + str(year), '+')],
                                                 "ghg", year, row_data, negate=True)

        row_data = process_files_with_operations(path_dir, save_path,
                                                 [("BIO_ag_" + str(year), '+'),("BIO_ag_" + str(year - 1), '-'),
                                                  ("BIO_am_" + str(year), '+'), ("BIO_non_ag_" + str(year), '+')],
                                                 "bio", year, row_data, negate=False)

        # 将字典结果追加到列表
        results.append(row_data)

    # 创建 DataFrame
    df_results = pd.DataFrame(results)
    
    # 映射键到列名，确保顺序
    df_results.rename(columns=config.KEY_TO_COLUMN_MAP, inplace=True)
    df_results = df_results[columns_name]  # 按照 columns_name 排序
    output_dir = f"{config.TASK_DIR}/carbon_price/excel"
    os.makedirs(output_dir, exist_ok=True)
    output_excel_path = os.path.join(output_dir,f"02_process_{input_file}.xlsx")
    df_results.to_excel(output_excel_path, index=False)
    return df_results





def calculate_year_data(year, ghg_bio_path, ghg_path):
    """
    计算单年的生物多样性和碳排放数据，并返回结果。
    """
    try:
        # 加载数据
        cost_all = np.load(os.path.join(ghg_bio_path, f"cost_{year}.npy"))
        cost_ghg = np.load(os.path.join(ghg_path, f"cost_{year}.npy"))
        cost_bio = cost_all - cost_ghg

        ghg_arr = np.load(os.path.join(ghg_path, f"ghg_{year}.npy"))
        bio_arr = np.load(os.path.join(ghg_bio_path, f"bio_{year}.npy"))

        # 计算价格
        price_bio = cost_bio / bio_arr
        price_carbon = cost_ghg / ghg_arr

        # 保存每年的中间数据
        save_data(f"../data/bio_cost_{year}.npy", cost_bio)
        save_data(f"../data/carbon_cost_{year}.npy", cost_ghg)
        save_data(f"../data/bio_price_{year}.npy", price_bio)
        save_data(f"../data/carbon_price_{year}.npy", price_carbon)
        save_data(f"../data/bio_{year}.npy", bio_arr)
        save_data(f"../data/ghg_{year}.npy", ghg_arr)

        # 返回计算结果
        return {
            "Year": year,
            "Carbon cost(M$)": cost_ghg.sum() / 1e6,
            "Biodiversity cost(M$)": cost_bio.sum() / 1e6,
            "GHG abatement(MtCO2e)": ghg_arr.sum() / 1e6,
            "Biodiversity score(Mha)": bio_arr.sum() / 1e6,
            "Carbon price($/tCO2e)": cost_ghg.sum() / ghg_arr.sum(),
            "Biodiversity price($/ha)": cost_bio.sum() / bio_arr.sum()
        }

    except Exception as e:
        print(f"Error processing year {year}: {e}")
        return None


def calculate_bio_price(input_files, enable_parallel=True, n_jobs=-1):
    """
    计算生物多样性价格 (Biodiversity price)，支持并行处理。

    参数:
    - input_files: 输入文件列表。
    - enable_parallel: 是否启用并行处理。
    - n_jobs: 并行处理的工作线程数（默认为使用所有可用线程）。
    """
    print("Start to calculate biodiversity price.")

    # 获取路径和年份范围
    path_name = get_path(input_files[0])
    years = get_year(path_name)
    ghg_bio_path = os.path.join(get_path(input_files[0]), "data_for_carbon_price")
    ghg_path = os.path.join(get_path(input_files[1]), "data_for_carbon_price")

    # 初始化结果 DataFrame
    df_columns = [
        "Year", "Carbon cost(M$)", "Biodiversity cost(M$)",
        "GHG abatement(MtCO2e)", "Biodiversity score(Mha)",
        "Carbon price($/tCO2e)", "Biodiversity price($/ha)"
    ]

    # 使用并行或串行计算年度数据
    if enable_parallel:
        print("Running in parallel mode...")
        results = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(calculate_year_data)(year, ghg_bio_path, ghg_path)
            for year in tqdm(range(years[1], years[-1] + 1), desc="Processing years")
        )
    else:
        print("Running in sequential mode...")
        results = [
            calculate_year_data(year, ghg_bio_path, ghg_path)
            for year in tqdm(range(years[1], years[-1] + 1), desc="Processing years")
        ]

    # 过滤掉返回值为 None 的结果
    results = [result for result in results if result is not None]

    # 转换为 DataFrame
    df = pd.DataFrame(results, columns=df_columns)

    # 保存结果到 Excel 文件
    output_dir = f"{config.TASK_DIR}/carbon_price/excel"
    output_path =  f"{output_dir}/03_price.xlsx"
    try:
        df.to_excel(output_path, index=False)
        print(f"Results saved to {output_path}")
    except Exception as e:
        print(f"Failed to save results: {e}")
    return df

def save_data(file_path, data):
    """
    保存中间数据，增加异常处理。
    """
    try:
        np.save(file_path, data)
        output_tif = file_path.replace(".npy", ".tif")
        proj_file = os.path.join(get_path(input_files[1]), "out_2050", "ammap_2050.tiff")
        npy_to_map(file_path, output_tif, proj_file)
    except Exception as e:
        print(f"Failed to save file {file_path}: {e}")

def calculate_shadow_price(input_file,percentile_num=95,mask_use=True, output=True):
    print("Start to calculate Uniform carbon price.")
    path_name = get_path(input_file)
    years = get_year(path_name)
    results = []

    for year in trange(years[1], years[-1] + 1):
        # 加载每年的 .npy 文件
        payment_arr = np.load(os.path.join(path_name, "data_for_carbon_price", f"cost_{year}.npy"))

        # GHG
        ghg_arr = np.load(os.path.join(path_name, "data_for_carbon_price", f"ghg_{year}.npy"))
        cp_arr = np.load(os.path.join(path_name, "data_for_carbon_price", f"cp_{year}.npy"))

        # 使用 mask 过滤掉 ghg_arr < 1的值
        if mask_use:
            mask = ghg_arr >= 1
            cp_arr1 = cp_arr[mask]

        # 计算所需的值
        carbon_price_uniform = np.percentile(cp_arr1, percentile_num)
        total_emissions_abatement = np.sum(ghg_arr)
        total_ghg_uniform = carbon_price_uniform * total_emissions_abatement
        total_ghg_discriminatory = np.sum(cp_arr * ghg_arr)
        carbon_price_discriminatory_avg = total_ghg_discriminatory / total_emissions_abatement if total_emissions_abatement != 0 else 0

        # BIO
        bio_arr = np.load(os.path.join(path_name, "data_for_carbon_price", f"bio_{year}.npy"))
        bp_arr = np.load(os.path.join(path_name, "data_for_carbon_price", f"bp_{year}.npy"))

        if mask_use:
            mask = bio_arr >= 1
            bp_arr1 = bp_arr[mask]
            bio_price_uniform = np.percentile(bp_arr1, percentile_num)
        else:
            bio_price_uniform = np.percentile(bp_arr, percentile_num)
        total_bio = np.sum(bio_arr)
        total_bio_uniform = bio_price_uniform * total_bio
        total_bio_discriminatory = np.sum(bp_arr * bio_arr)
        bio_price_discriminatory_avg = total_bio_discriminatory / total_bio if total_bio != 0 else 0


            # 将计算结果添加到列表中
        results.append({
            "Year": year,
            "Total emissions abatement (MtCO2e)": total_emissions_abatement / 1000000,
            "Total cost for ghg Uniform Payment (M$)": total_ghg_uniform / 1000000,
            "Total cost for ghg Discriminatory Payment (M$)": total_ghg_discriminatory / 1000000,
            "Carbon price for Uniform Payment ($/tCO2e)": carbon_price_uniform,
            "Carbon Price for Discriminatory Payment average ($/tCO2e)": carbon_price_discriminatory_avg,

            "Total bio (Mha)": total_bio / 1000000,
            "Total cost for bio Uniform Payment (M$)": total_bio_uniform / 1000000,
            "Total cost for bio Discriminatory Payment (M$)": total_bio_discriminatory / 1000000,
            "BIO price for Uniform Payment ($/tCO2e)": bio_price_uniform,
            "BIO Price for Discriminatory Payment average ($/tCO2e)": bio_price_discriminatory_avg
        })

    # 创建DataFrame
    df_results = pd.DataFrame(results)

    # 保存结果到CSV文件
    if output:
        output_excel_path = os.path.join(f"../output/03_{input_file}_shadow_price.xlsx")
        df_results.to_excel(output_excel_path, index=False)
    return df_results

def calculate_carbonprice_compare(path_name,percentiles = [100, 99, 98, 97, 96, 95, 94, 93, 92, 91, 90, 89, 88, 87, 86, 85, 84, 83, 82, 81, 80]):
    print("Start to calculate carbon price.")
    path_name = "output/" + path_name
    # 定义分位数列表
    # percentiles = [100, 99, 98, 97, 96, 95, 94, 93, 92, 91, 90, 89, 88, 87, 86, 85, 84, 83, 82, 81, 80]

    # 初始化表格字典
    folder_name = next((f for f in os.listdir(path_name) if
                        os.path.isdir(os.path.join(path_name, f)) and f.startswith("begin_end_compare_")), None)
    years = [int(year) for year in re.findall(r'\d{4}', folder_name)]

    carbon_price_with_mask = {year: [] for year in range(years[0] + 1, years[1] + 1)}
    ghg_max_cp_with_mask = {year: [] for year in range(years[0] + 1, years[1] + 1)}
    cost_max_cp_with_mask = {year: [] for year in range(years[0] + 1, years[1] + 1)}

    carbon_price_without_mask = {year: [] for year in range(years[0] + 1, years[1] + 1)}
    ghg_max_cp_without_mask = {year: [] for year in range(years[0] + 1, years[1] + 1)}
    cost_max_cp_without_mask = {year: [] for year in range(years[0] + 1, years[1] + 1)}

    # 循环每个年份
    for year in trange(years[0] + 1, years[1] + 1):
        # 加载每年的 .npy 文件
        payment_arr = np.load(os.path.join(path_name, "data_for_carbon_price", f"cost_{year}.npy"))
        ghg_arr = np.load(os.path.join(path_name, "data_for_carbon_price", f"ghg_{year}.npy"))
        cp_arr = np.load(os.path.join(path_name, "data_for_carbon_price", f"cp_{year}.npy"))

        # 不使用 mask 直接使用原始数组
        filtered_cp_arr_without_mask = cp_arr
        filtered_ghg_arr_without_mask = ghg_arr
        filtered_payment_arr_without_mask = payment_arr

        # 使用 mask 过滤掉 ghg_arr < 1的值
        mask = ghg_arr >= 1
        filtered_cp_arr_with_mask = cp_arr[mask]
        filtered_ghg_arr_with_mask = ghg_arr[mask]
        filtered_payment_arr_with_mask = payment_arr[mask]

        # 循环每个分位数
        for percentile_num in percentiles:
            # 计算使用 mask 的值
            carbon_price_uniform_with_mask = np.percentile(filtered_cp_arr_with_mask, percentile_num)
            # total_emissions_abatement_with_mask = np.sum(filtered_ghg_arr_with_mask)
            # total_cost_uniform_with_mask = carbon_price_uniform_with_mask * total_emissions_abatement_with_mask

            # 找到分位数所在位置对应的索引
            percentile_value_with_mask = np.percentile(filtered_cp_arr_with_mask, percentile_num)
            max_index_with_mask = np.unravel_index(np.argmax(filtered_cp_arr_with_mask >= percentile_value_with_mask),
                                                   filtered_cp_arr_with_mask.shape)

            # 获取对应位置的 ghg_arr 和 payment_arr 的值
            ghg_value_at_max_cp_with_mask = filtered_ghg_arr_with_mask[max_index_with_mask]
            cost_value_at_max_cp_with_mask = filtered_payment_arr_with_mask[max_index_with_mask]

            carbon_price_with_mask[year].append(carbon_price_uniform_with_mask)
            ghg_max_cp_with_mask[year].append(ghg_value_at_max_cp_with_mask)
            cost_max_cp_with_mask[year].append(cost_value_at_max_cp_with_mask)

            # 计算不使用 mask 的值
            carbon_price_uniform_without_mask = np.percentile(filtered_cp_arr_without_mask, percentile_num)
            # total_emissions_abatement_without_mask = np.sum(filtered_ghg_arr_without_mask)
            # total_cost_uniform_without_mask = carbon_price_uniform_without_mask * total_emissions_abatement_without_mask

            # 找到分位数所在位置对应的索引
            percentile_value_without_mask = np.percentile(filtered_cp_arr_without_mask, percentile_num)
            max_index_without_mask = np.unravel_index(
                np.argmax(filtered_cp_arr_without_mask >= percentile_value_without_mask),
                filtered_cp_arr_without_mask.shape)

            # 获取对应位置的 ghg_arr 和 payment_arr 的值
            ghg_value_at_max_cp_without_mask = filtered_ghg_arr_without_mask[max_index_without_mask]
            cost_value_at_max_cp_without_mask = filtered_payment_arr_without_mask[max_index_without_mask]

            carbon_price_without_mask[year].append(carbon_price_uniform_without_mask)
            ghg_max_cp_without_mask[year].append(ghg_value_at_max_cp_without_mask)
            cost_max_cp_without_mask[year].append(cost_value_at_max_cp_without_mask)

    # 创建DataFrame
    df_carbon_price_with_mask = pd.DataFrame(carbon_price_with_mask).T
    df_ghg_max_cp_with_mask = pd.DataFrame(ghg_max_cp_with_mask).T
    df_cost_max_cp_with_mask = pd.DataFrame(cost_max_cp_with_mask).T

    df_carbon_price_without_mask = pd.DataFrame(carbon_price_without_mask).T
    df_ghg_max_cp_without_mask = pd.DataFrame(ghg_max_cp_without_mask).T
    df_cost_max_cp_without_mask = pd.DataFrame(cost_max_cp_without_mask).T

    # 设置列名为百分位数
    df_carbon_price_with_mask.columns = percentiles
    df_ghg_max_cp_with_mask.columns = percentiles
    df_cost_max_cp_with_mask.columns = percentiles

    df_carbon_price_without_mask.columns = percentiles
    df_ghg_max_cp_without_mask.columns = percentiles
    df_cost_max_cp_without_mask.columns = percentiles

    # 保存结果到Excel文件
    output_excel_path = os.path.join(r"output\Carbon_Price", path_name.split("/")[1] + "_carbon_price_comparison1.xlsx")
    with pd.ExcelWriter(output_excel_path) as writer:
        df_carbon_price_with_mask.to_excel(writer, sheet_name='Carbon Price with Mask', index_label='Year')
        df_ghg_max_cp_with_mask.to_excel(writer, sheet_name='Ghg Max Cp with Mask', index_label='Year')
        df_cost_max_cp_with_mask.to_excel(writer, sheet_name='Cost Max Cp with Mask', index_label='Year')

        df_carbon_price_without_mask.to_excel(writer, sheet_name='Carbon Price without Mask', index_label='Year')
        df_ghg_max_cp_without_mask.to_excel(writer, sheet_name='Ghg Max Cp without Mask', index_label='Year')
        df_cost_max_cp_without_mask.to_excel(writer, sheet_name='Cost Max Cp without Mask', index_label='Year')

    print(f"Results saved to {output_excel_path}")

import numpy as np
import rasterio
import os

def compute_and_save_class_metrics(output_dir='../Hexagon'):
    def read_tif(filename):
        with rasterio.open(filename) as src:
            arr = src.read(1)
            nodata_value = src.nodata
            nan_mask = (arr == nodata_value) if nodata_value is not None else np.isnan(arr)
            return arr, src.profile, nan_mask

    # 读取数据
    class_arr, profile, nan_mask = read_tif(os.path.join(output_dir, "Hexagonal.tif"))
    ghg, _, _  = read_tif(os.path.join("../data", "ghg_2050.tif"))
    carbon_cost, _, _ = read_tif(os.path.join("../data", "carbon_cost_2050.tif"))
    bio, _, _ = read_tif(os.path.join("../data", "bio_2050.tif"))
    bio_cost, _, _ = read_tif(os.path.join("../data", "bio_cost_2050.tif"))

    n_rows, n_cols = class_arr.shape

    # 初始化输出数组
    total_carbon_cost_raster = np.zeros_like(class_arr, dtype=np.float32)
    total_bio_cost_raster = np.zeros_like(class_arr, dtype=np.float32)
    carbon_price_raster = np.zeros_like(class_arr, dtype=np.float32)
    bio_price_raster = np.zeros_like(class_arr, dtype=np.float32)

    unique_classes = np.unique(class_arr)
    unique_classes = unique_classes[unique_classes != 0]

    for cls in unique_classes:
        # 找到这一类所有坐标位置
        indices = np.argwhere(class_arr == cls)

        # 有效索引：确保不越界 & bio 和 bio_cost 不是 nan
        valid_rows = []
        valid_cols = []

        for row, col in indices:
            if row < bio.shape[0] and col < bio.shape[1]:
                b = bio[row, col]
                bc = bio_cost[row, col]
                if not np.isnan(b) and not np.isnan(bc):
                    valid_rows.append(row)
                    valid_cols.append(col)

        if not valid_rows:
            continue  # 无有效数据，跳过

        valid_rows = np.array(valid_rows)
        valid_cols = np.array(valid_cols)

        # 提取有效值
        ghg_vals = ghg[valid_rows, valid_cols]
        bio_vals = bio[valid_rows, valid_cols]
        carbon_cost_vals = carbon_cost[valid_rows, valid_cols]
        bio_cost_vals = bio_cost[valid_rows, valid_cols]

        # 计算总量与价格
        ghg_sum = np.sum(ghg_vals)
        bio_sum = np.sum(bio_vals)
        carbon_cost_sum = np.sum(carbon_cost_vals)
        bio_cost_sum = np.sum(bio_cost_vals)

        carbon_price = carbon_cost_sum / ghg_sum if ghg_sum != 0 else 0
        bio_price = bio_cost_sum / bio_sum if bio_sum != 0 else 0

        # 所有属于该类的像元赋相同值
        class_mask = (class_arr == cls)
        total_carbon_cost_raster[class_mask] = carbon_cost_sum
        total_bio_cost_raster[class_mask] = bio_cost_sum
        carbon_price_raster[class_mask] = carbon_price
        bio_price_raster[class_mask] = bio_price

    # 将 class_arr 中为 NaN 的位置，设为 NaN
    total_carbon_cost_raster[nan_mask] = np.nan
    total_bio_cost_raster[nan_mask] = np.nan
    carbon_price_raster[nan_mask] = np.nan
    bio_price_raster[nan_mask] = np.nan

    def write_tif(filename, array, profile):
        profile.update(dtype=rasterio.float32, count=1, nodata=np.nan)
        with rasterio.open(filename, 'w', **profile) as dst:
            dst.write(array, 1)

    write_tif(os.path.join(output_dir, "total_carbon_cost.tif"), total_carbon_cost_raster, profile)
    write_tif(os.path.join(output_dir, "total_bio_cost.tif"), total_bio_cost_raster, profile)
    write_tif(os.path.join(output_dir, "carbon_price.tif"), carbon_price_raster, profile)
    write_tif(os.path.join(output_dir, "bio_price.tif"), bio_price_raster, profile)

    print("✅ All output files saved to", output_dir)


import numpy as np
import rasterio
import geopandas as gpd
from rasterio.features import rasterize
import os


def compute_class_metrics_from_shapefile(shapefile_path, ref_raster_path, output_dir):
    """
    从 shapefile 中计算每个区域的 bio, bio_cost, carbon_price, ghg 等指标，并生成对应的 TIF 文件。

    参数:
        shapefile_path (str): 输入的 shapefile 文件路径。
        ref_raster_path (str): 参考栅格路径，用于确定输出 TIF 文件的形状和投影。
        output_dir (str): 输出 TIF 文件保存目录。

    输出:
        生成的 4 个 TIF 文件，分别为 total_carbon_cost.tif、total_bio_cost.tif、carbon_price.tif 和 bio_price.tif。
    """

    # 1. 读取参考栅格（用于确定 shape、transform、crs 等信息）
    with rasterio.open(ref_raster_path) as ref:
        ref_shape = ref.shape
        ref_transform = ref.transform
        ref_crs = ref.crs
        ref_profile = ref.profile

    # 2. 读取生物和碳数据
    def read_tif(path):
        with rasterio.open(path) as src:
            arr = src.read(1)
            return arr

    ghg = read_tif("../data/ghg_2050.tif")
    carbon_cost = read_tif("../data/carbon_cost_2050.tif")
    bio = read_tif("../data/bio_2050.tif")
    bio_cost = read_tif("../data/bio_cost_2050.tif")

    # 3. 读取 shapefile
    gdf = gpd.read_file(shapefile_path)

    # 栅格化：将每个 polygon 的 id 字段 rasterize 到栅格中
    print("Rasterizing shapefile...")
    shapes = [(geom, fid) for geom, fid in zip(gdf.geometry, gdf['id'])]
    class_arr = rasterize(
        shapes=shapes,
        out_shape=ref_shape,
        transform=ref_transform,
        fill=0,
        dtype=np.int32
    )

    # 4. 初始化输出栅格
    total_carbon_cost_raster = np.zeros(ref_shape, dtype=np.float32)
    total_bio_cost_raster = np.zeros(ref_shape, dtype=np.float32)
    carbon_price_raster = np.zeros(ref_shape, dtype=np.float32)
    bio_price_raster = np.zeros(ref_shape, dtype=np.float32)

    # 获取唯一的区域 ID
    unique_ids = np.unique(class_arr)
    unique_ids = unique_ids[unique_ids != 0]  # 去掉背景值

    # 5. 遍历每个区域 ID，计算指标并赋值
    print("Processing unique IDs...")
    for uid in unique_ids:
        mask = (class_arr == uid)

        # 去除 bio 或 bio_cost 是 nan 的像元
        valid_mask = mask & ~np.isnan(bio) & ~np.isnan(bio_cost)

        if not np.any(valid_mask):
            continue

        # 获取有效区域的值
        ghg_vals = ghg[valid_mask]
        bio_vals = bio[valid_mask]
        carbon_cost_vals = carbon_cost[valid_mask]
        bio_cost_vals = bio_cost[valid_mask]

        # 计算总量和价格
        ghg_sum = np.sum(ghg_vals)
        bio_sum = np.sum(bio_vals)
        carbon_cost_sum = np.sum(carbon_cost_vals)
        bio_cost_sum = np.sum(bio_cost_vals)

        carbon_price = carbon_cost_sum / ghg_sum if ghg_sum != 0 else 0
        bio_price = bio_cost_sum / bio_sum if bio_sum != 0 else 0

        # 将计算结果写入对应栅格
        total_carbon_cost_raster[mask] = carbon_cost_sum
        total_bio_cost_raster[mask] = bio_cost_sum
        carbon_price_raster[mask] = carbon_price
        bio_price_raster[mask] = bio_price

    # 6. 写出结果到 TIF 文件
    def write_tif(filename, array):
        profile = ref_profile.copy()
        profile.update(dtype=rasterio.float32, count=1, nodata=np.nan)
        with rasterio.open(os.path.join(output_dir, filename), 'w', **profile) as dst:
            dst.write(array, 1)

    print("Writing output TIF files...")
    write_tif("total_carbon_cost.tif", total_carbon_cost_raster)
    write_tif("total_bio_cost.tif", total_bio_cost_raster)
    write_tif("carbon_price.tif", carbon_price_raster)
    write_tif("bio_price.tif", bio_price_raster)

    print("Processing completed. TIF files saved to:", output_dir)

import geopandas as gpd
from rasterstats import zonal_stats
import numpy as np
import os

def compute_zonal_stats_to_shp(
    shapefile_path,
    raster_paths,  # dict: {"ghg": ..., "carbon_cost": ..., ...}
    output_shp_path,
    id_field='id'
):
    gdf = gpd.read_file(shapefile_path)
    assert id_field in gdf.columns, f"Shapefile must contain '{id_field}' field"

    stats = {}
    for name, path in raster_paths.items():
        zs = zonal_stats(shapefile_path, path, stats=["sum"], nodata=np.nan)
        stats[name + "_sum"] = [s["sum"] if s["sum"] is not None else 0 for s in zs]

    for k, v in stats.items():
        gdf[k] = v

    # 计算价格字段
    gdf["carbon_price"] = gdf["carbon_cost_sum"] / gdf["ghg_sum"]
    gdf["bio_price"] = gdf["bio_cost_sum"] / gdf["bio_sum"]

    # 替换 inf 和 nan 为 0（可选）
    gdf = gdf.replace([np.inf, -np.inf], np.nan)
    gdf = gdf.fillna(0)

    # 保存为新的 shapefile
    gdf.to_file(output_shp_path)
    print(f"✅ Zonal stats written to {output_shp_path}")

import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_bounds
import numpy as np

def rasterize_fields_from_shp(
    shapefile_path,
    output_dir,
    fields,         # list of fields to rasterize
    resolution=1000 # 输出分辨率（单位：米）
):
    os.makedirs(output_dir, exist_ok=True)
    gdf = gpd.read_file(shapefile_path)

    minx, miny, maxx, maxy = gdf.total_bounds
    width = int(np.ceil((maxx - minx) / resolution))
    height = int(np.ceil((maxy - miny) / resolution))
    transform = from_bounds(minx, miny, maxx, maxy, width, height)

    profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": 1,
        "dtype": "float32",
        "crs": gdf.crs,
        "transform": transform,
        "nodata": np.nan
    }

    for field in fields:
        print(f"▶ Rasterizing field: {field}")
        shapes = [(geom, val) for geom, val in zip(gdf.geometry, gdf[field])]
        out_arr = rasterize(
            shapes=shapes,
            out_shape=(height, width),
            transform=transform,
            fill=np.nan,
            dtype='float32'
        )
        out_path = os.path.join(output_dir, f"{field}.tif")
        with rasterio.open(out_path, 'w', **profile) as dst:
            dst.write(out_arr, 1)

    print("✅ Raster files saved to", output_dir)
