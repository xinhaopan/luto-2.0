import numpy as np
import numpy_financial as npf
import os
import pandas as pd
import re
from joblib import Parallel, delayed
from tqdm import *

from .tools import get_path, get_year
from .config import n_jobs,cost_dict,revenue_dict


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
        if os.path.exists(file_path):
            try:
                # 读取 .npy 文件
                data = np.load(file_path)

                # 如果是第一次加载数据，初始化 total_sum
                if total_sum is None:
                    total_sum = data if operation == '+' else -data
                else:
                    if operation == '+':
                        total_sum += data
                    elif operation == '-':
                        total_sum -= data
            except Exception as e:
                print(f"Error loading file {file_path}: {e}")
                # 如果读取失败，将值补充为 0
                if total_sum is None:
                    total_sum = np.zeros(1) if operation == '+' else -np.zeros(1)
                else:
                    if operation == '+':
                        total_sum += np.zeros(1)
                    elif operation == '-':
                        total_sum -= np.zeros(1)
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
        'cost_am': cost_dict.get('cost_am', []),
        'cost_non_ag': cost_dict.get('cost_non_ag', []),
        'revenue_am': revenue_dict.get('revenue_am', []),
        'revenue_non_ag': revenue_dict.get('revenue_non_ag', [])
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


def process_files_with_operations(path_dir, save_path, files_with_ops, file_prefix, year, rows_nums, negate=False):
    """
    处理文件加减操作，将结果保存为 .npy 文件并更新 rows_nums。

    :param path_dir: 文件所在的目录
    :param save_path: 保存 .npy 文件的路径
    :param files_with_ops: 包含 (文件名, 操作符) 的元组列表，操作符为 '+' 或 '-'
    :param file_prefix: 保存文件的前缀
    :param year: 当前年份，用于命名保存的文件
    :param rows_nums: 用于存储各项结果的列表
    :param negate: 是否对结果进行取负操作
    :return: 更新后的 rows_nums
    """
    files_with_ops = [(file + ".npy", op) for file, op in files_with_ops]

    # 处理文件加减操作
    arr = apply_operations_on_files(path_dir, files_with_ops)

    if arr is None:
        print(f"No valid data found for {file_prefix} in year {year}")
        rows_nums.append(0)
        return rows_nums

    # 如果需要取负操作，则取负
    if negate:
        arr = -arr

    # 保存结果到 .npy 文件
    if file_prefix.endswith('_'):
        file_prefix = file_prefix[:-1]
    np.save(os.path.join(save_path, f"{file_prefix}_{year}.npy"), arr)

    # 将结果添加到 rows_nums，按百万单位
    arr = np.nan_to_num(arr, nan=0.0)
    rows_nums.append(np.sum(arr) / 1000000)

    return rows_nums

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

    columns_name = ["Year", "cost_ag(M$)", "cost_am(M$)", "cost_non_ag(M$)", "cost_transition_ag2ag(M$)",
                    "cost_amortised_transition_ag2non_ag(M$)","revenue_ag(M$)","revenue_am(M$)","revenue_non_ag(M$)",
                    "GHG_ag(MtCOe2)", "GHG_am(MtCOe2)", "GHG_non-ag(MtCOe2)", "GHG_transition(MtCOe2)",
                    "BIO_ag(M ha)", "BIO_am(M ha)", "BIO_non_ag(M ha)"]

    save_path = os.path.join(path_name, "data_for_carbon_price")
    os.makedirs(save_path, exist_ok=True)

    results = []
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
        results = Parallel(n_jobs=n_jobs)(
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
        output_excel_path = os.path.join(f"../output/01_{input_file}_summary.xlsx")
        df_results.to_excel(output_excel_path, index=False)
    return df_results


def compute_unit_prices(input_file, output=True):
    path_name = get_path(input_file)
    years = get_year(path_name)
    path_dir = os.path.join(path_name, "data_for_carbon_price")
    save_path = os.path.join(path_name, "data_for_carbon_price")
    print("Start to calculate Discriminatory carbon price.")
    columns_name = ["Year", "Opportunity cost(M$)","Transition cost(M$)","AM net cost(M$)","Non_AG net cost(M$)","All cost(M$)","GHG Abatement(MtCOe2)","BIO(Mha)","carbon price($/tCOe2)","biodiversity price($/ha)"]
    results = []
    for year in trange(years[1], years[-1] + 1):
        rows_nums = [year]

        rows_nums = process_files_with_operations(path_dir, save_path,
                                                  [("revenue_ag_" + str(year), '+'), ("cost_ag_" + str(year), '-'),
                                                   ("revenue_ag_" + str(year - 1), '-'),
                                                   ("cost_ag_" + str(year - 1), '+')],
                                                  "opportunity_cost", year, rows_nums)

        rows_nums = process_files_with_operations(path_dir, save_path,
                                                  [("cost_transition_ag2ag_" + str(year), '+'),
                                                   ("cost_amortised_transition_ag2non_ag_" + str(year), '+')],
                                                  "transition_cost", year, rows_nums)

        rows_nums = process_files_with_operations(path_dir, save_path,
                                                  [("revenue_am_" + str(year), '-'), ("cost_am_" + str(year), '+')],
                                                  "am_profit", year, rows_nums)

        rows_nums = process_files_with_operations(path_dir, save_path,
                                                  [("revenue_non_ag_" + str(year), '-'),
                                                   ("cost_non_ag_" + str(year), '+')],
                                                  "non_ag_profit", year, rows_nums)

        rows_nums = process_files_with_operations(path_dir, save_path,
                                                  [("opportunity_cost_" + str(year), '+'),("transition_cost_" + str(year), '+'),
                                                   ("am_profit_" + str(year), '+'),("non_ag_profit_" + str(year), '+')],
                                                  "cost", year, rows_nums)
        cost_arr = np.load(os.path.join(path_dir, f"cost_{year}.npy"))

        rows_nums = process_files_with_operations(path_dir, save_path,
                                                  [("GHG_ag_" + str(year), '+'), ("GHG_ag_" + str(year - 1), '-'),
                                                   ("GHG_am_" + str(year), '+'), ("GHG_non-ag_" + str(year), '+'),
                                                   ("GHG_transition_" + str(year), '+')],
                                                  "ghg", year, rows_nums, negate=True)
        ghg_arr = np.load(os.path.join(path_dir, f"ghg_{year}.npy"))
        rows_nums = process_files_with_operations(path_dir, save_path,
                                                  [("BIO_ag_" + str(year), '-'),
                                                   ("BIO_am_" + str(year), '-'), ("BIO_non_ag_" + str(year), '-')],
                                                  "bio", year, rows_nums, negate=True)
        bio_arr = np.load(os.path.join(path_dir, f"bio_{year}.npy"))

        with np.errstate(divide='ignore', invalid='ignore'):
            cp_arr = np.divide(cost_arr, ghg_arr)
            cp_arr[np.isinf(cp_arr)] = 0
            cp_arr = np.nan_to_num(cp_arr, nan=0.0)

            bp_arr = np.divide(cost_arr, bio_arr)
            bp_arr[np.isinf(bp_arr)] = 0
            bp_arr = np.nan_to_num(bp_arr, nan=0.0)

        rows_nums.append(np.sum(cp_arr * ghg_arr) / np.sum(ghg_arr))
        rows_nums.append(np.sum(bp_arr * bio_arr) / np.sum(bio_arr))

        np.save(os.path.join(save_path, f"cp_{year}.npy"), cp_arr)
        np.save(os.path.join(save_path, f"bp_{year}.npy"), bp_arr)

        results.append(rows_nums)

    df_results = pd.DataFrame(results, columns=columns_name)
    if output:
        output_excel_path = os.path.join(f"../output/02_{input_file}_price.xlsx")
        df_results.to_excel(output_excel_path, index=False)
    return df_results


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

