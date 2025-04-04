import numpy as np
import numpy_financial as npf
import os
import matplotlib.pyplot as plt
import pandas as pd
import re
from joblib import Parallel, delayed
from tqdm import *

from tools.helpers import *
from tools.desc import *

def amortize_costs(input_file, file_name = "cost_transition_ag2non_ag", rate=0.07, horizon=30):
    path_name = get_path(input_file)
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


def calculate_cost_revenue(input_file, use_parallel=False,output=True):
    print("Start jobs to calculate cost_revenue.")
    path_name = get_path(input_file)
    years=get_year(path_name)
    # 初始化结果列表

    columns_name = ["Year", "cost_ag(M$)", "cost_am(M$)", "cost_non_ag(M$)", "cost_transition_ag2ag(M$)",
                    "cost_transition_ag2non_ag(M$)","revenue_ag(M$)","revenue_am(M$)","revenue_non_ag(M$)",
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


def calculate_price_cells(input_file, output=True):
    path_name = get_path(input_file)
    years = get_year(path_name)
    path_dir = os.path.join(path_name, "data_for_carbon_price")
    save_path = os.path.join(path_name, "data_for_carbon_price")
    print("Start to calculate Discriminatory carbon price.")
    columns_name = ["Year", "Opportunity cost(M$)","Transition cost(M$)","AM profit(M$)","Non_AG profit(M$)","All cost(M$)","GHG Abatement(MtCOe2)","BIO(Mha)","carbon price($/tCOe2)","biodiversity price($/ha)"]
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
                                                   ("cost_transition_ag2non_ag_" + str(year), '+')],
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

def caculate_shadow_price(input_file,percentile_num=95,mask_use=True, output=True):
    print("Start to calculate Uniform carbon price.")
    path_name = get_path(input_file)
    years = get_year(path_name)
    results = []

    for year in trange(years[1], years[-1] + 1):
        # 加载每年的 .npy 文件
        payment_arr = np.load(os.path.join(path_name, "data_for_carbon_price", f"cost_{year}.npy"))
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

        try:
            bio_arr = np.load(os.path.join(path_name, "data_for_carbon_price", f"bio_{year}.npy"))
            bp_arr = np.load(os.path.join(path_name, "data_for_carbon_price", f"bp_{year}.npy"))
        except Exception as e:
            print(f"Error loading file for year {year}: {e}")
            bio_arr = np.zeros_like(payment_arr)  # 示例形状，请根据实际数据调整
            bp_arr = np.zeros_like(payment_arr)  # 示例形状，请根据实际数据调整

        if mask_use:
            mask = bio_arr >= 1
            if mask.shape == bio_arr.shape and bp_arr.shape == bio_arr.shape:
                bp_arr1 = bp_arr[mask]
                bio_price_uniform = np.percentile(bp_arr1, percentile_num)
                total_bio = np.sum(bio_arr)
                total_bio_uniform = bio_price_uniform * total_bio
                total_bio_discriminatory = np.sum(bp_arr * bio_arr)
                bio_price_discriminatory_avg = total_bio_discriminatory / total_bio if total_bio != 0 else 0


            else:
                print(
                    f"Dimension mismatch: mask shape {mask.shape}, bio_arr shape {bio_arr.shape}, bp_arr shape {bp_arr.shape}")
                bio_price_uniform = 0
                total_bio = 0
                total_bio_uniform = 0
                total_bio_discriminatory = 0
                bio_price_discriminatory_avg = 0

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

def caculate_carbonprice_compare(path_name,percentiles = [100, 99, 98, 97, 96, 95, 94, 93, 92, 91, 90, 89, 88, 87, 86, 85, 84, 83, 82, 81, 80]):
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

def draw_histogram(path_name,year=2050, mask_use=True):
    print("Start to draw histogram.")
    path_name = get_path(input_file)
    # 加载 2050 年的 .npy 文件
    payment_arr_2050 = np.load(os.path.join(path_name, "data_for_carbon_price", f"cost_{year}.npy"))
    ghg_arr_2050 = np.load(os.path.join(path_name, "data_for_carbon_price", f"ghg_{year}.npy"))
    cp_arr_2050 = np.load(os.path.join(path_name, "data_for_carbon_price", f"cp_{year}.npy"))

    # 使用 mask 过滤掉 ghg_arr 小于 1 的值
    output_name =  f"histogram_{year}_nomask.png"
    if mask_use:
        mask = ghg_arr_2050 >= 1
        filtered_cp_arr_with_mask = cp_arr_2050[mask]
        filtered_ghg_arr_with_mask = ghg_arr_2050[mask]
        filtered_payment_arr_with_mask = payment_arr_2050[mask]
        output_name = f"histogram_{year}_mask.png"

    # 绘制直方图
    plt.figure(figsize=(18, 6))

    plt.subplot(1, 3, 1)
    plt.hist(filtered_cp_arr_with_mask, bins=50)
    plt.yscale('log')
    plt.title(f'Carbon Price for Uniform Payment ($/tCO2e) - {year}')
    plt.xlabel('Price ($/tCO2e)')
    plt.ylabel('Frequency')

    plt.subplot(1, 3, 2)
    plt.hist(filtered_ghg_arr_with_mask, bins=50)
    plt.yscale('log')
    plt.title(f'GHG with mask - {year}')
    plt.xlabel('GHG')
    plt.ylabel('Frequency')

    plt.subplot(1, 3, 3)
    plt.hist(filtered_payment_arr_with_mask, bins=50)
    plt.yscale('log')
    plt.title(f'Cost with mask - {year}')
    plt.xlabel('Cost')
    plt.ylabel('Frequency')

    plt.tight_layout()
    plt.savefig(os.path.join(r"output\Carbon_Price", path_name.split("/")[1] + "_" + f"histogram_{year}.png"))
    plt.show()


def summarize_from_csv(path_name):
    print("Start to summarize from csv.")
    path_name = "output/" + path_name
    # 初始化路径和列名
    categories = {
        'Cost_ag($)': ['cost_agricultural_commodity'],
        'Cost_am($)': ['cost_agricultural_management'],
        'Cost_non_ag($)': ['cost_non_ag'],
        'Cost_transition_ag2ag($)': ['cost_transition_ag2ag'],
        'Cost_transition_ag2non_ag($)': ['cost_transition_ag2non_ag'],
        'Cost_transition_non_ag2ag($)': ['cost_transition_non_ag2_ag'],
        'Revenue_ag($)': ['revenue_agricultural_commodity'],
        'Revenue_am($)': ['revenue_agricultural_management'],
        'Revenue_non_ag($)': ['revenue_non_ag'],
        'GHG_ag(tCO2e)': ['GHG_emissions_separate_agricultural_landuse'],
        'GHG_am(tCO2e)': ['GHG_emissions_separate_agricultural_management'],
        'GHG_non_ag(tCO2e)': ['GHG_emissions_separate_no_ag_reduction'],
        'GHG_off_land(tCO2e)': ['GHG_emissions_offland_commodity'],
        'GHG_transition(tCO2e)': ['GHG_emissions_separate_transition_penalty']
    }

    # 定义列名关键字的映射字典
    column_keywords = {
        'cost_transition_ag2ag': 'Cost ($)',
        'cost_transition_ag2non_ag': 'Cost ($)',
        'cost_transition_non_ag2_ag': 'Cost ($)',
        'GHG_emissions_offland_commodity': 'Total GHG Emissions',
        'default': 'Value'
    }

    # 创建DataFrame并保存到CSV文件
    # 处理所有年份的数据
    folder_name = next((f for f in os.listdir(path_name) if
                        os.path.isdir(os.path.join(path_name, f)) and f.startswith("begin_end_compare_")), None)
    years = [int(year) for year in re.findall(r'\d{4}', folder_name)]
    results = [process_year_data(year, path_name, categories, column_keywords) for year in range(years[0], years[1] + 1)]
    df_result = pd.DataFrame(results)

    df_result.to_excel("output/Carbon_Price/" + path_name.split("/")[1] + "_" + 'calculated_carbon_price_fromcsv.xlsx', index=False)

    # 用于计算的字典，包含四种计算情况
    calculations = {
        'Carbon_price1($/tCO2e)': {
            'cost_columns': ['Cost_ag($)', 'Cost_am($)', 'Cost_non_ag($)', 'Cost_transition_ag2ag($)',
                             'Cost_transition_ag2non_ag($)', 'Cost_transition_non_ag2ag($)'],
            'ghg_columns': ['GHG_ag(tCO2e)', 'GHG_am(tCO2e)', 'GHG_non_ag(tCO2e)', 'GHG_off_land(tCO2e)',
                            'GHG_transition(tCO2e)'],
            'revenue_columns': []
        },
        'Carbon_price2($/tCO2e)': {
            'cost_columns': ['Cost_ag($)', 'Cost_am($)', 'Cost_non_ag($)', 'Cost_transition_ag2ag($)',
                             'Cost_transition_ag2non_ag($)', 'Cost_transition_non_ag2ag($)'],
            'ghg_columns': ['GHG_ag(tCO2e)', 'GHG_am(tCO2e)', 'GHG_non_ag(tCO2e)', 'GHG_off_land(tCO2e)',
                            'GHG_transition(tCO2e)'],
            'revenue_columns': ['Revenue_ag($)', 'Revenue_am($)']
        },
        'Carbon_price3($/tCO2e)': {
            'cost_columns': ['Cost_am($)', 'Cost_non_ag($)', 'Cost_transition_ag2non_ag($)',
                             'Cost_transition_non_ag2ag($)'],
            'ghg_columns': ['GHG_am(tCO2e)', 'GHG_non_ag(tCO2e)', 'GHG_transition(tCO2e)'],
            'revenue_columns': []
        },
        'Carbon_price4($/tCO2e)': {
            'cost_columns': ['Cost_am($)', 'Cost_non_ag($)', 'Cost_transition_ag2non_ag($)',
                             'Cost_transition_non_ag2ag($)'],
            'ghg_columns': ['GHG_am(tCO2e)', 'GHG_non_ag(tCO2e)', 'GHG_transition(tCO2e)'],
            'revenue_columns': ['Revenue_am($)']
        }
    }

    with pd.ExcelWriter("output/Carbon_Price/" + path_name.split("/")[1] + "_" + 'calculated_carbon_price_fromcsv.xlsx', engine='openpyxl', mode='a',
                        if_sheet_exists='new') as writer:
        start_sheet = len(writer.sheets) + 1  # 获取现有工作表数量，加1以从下一个空白工作表开始
        for i, (key, value) in enumerate(calculations.items(), start=start_sheet):
            # 计算成本总和，温室气体总和和收入总和
            df_result1 = df_result.copy()
            df_result1['Cost_all(M$)'] = df_result1[value['cost_columns']].sum(axis=1) / 1e6
            df_result1['GHG_all(MtCO2e)'] = - df_result1[value['ghg_columns']].sum(axis=1) / 1e6

            if value['revenue_columns']:
                df_result1['Revenue_all(M$)'] = df_result1[value['revenue_columns']].sum(axis=1) / 1e6
                df_result1[key] = (df_result1['Cost_all(M$)'] - df_result1['Revenue_all(M$)']) / df_result1[
                    'GHG_all(MtCO2e)']
            else:
                df_result1[key] = df_result1['Cost_all(M$)'] / df_result1['GHG_all(MtCO2e)']

            # 选择要写入的列
            columns_to_write = ['Year', 'Cost_all(M$)', 'GHG_all(MtCO2e)', key]
            if 'Revenue_all(M$)' in df_result1.columns:
                columns_to_write.insert(2, 'Revenue_all(M$)')

            # 写入Excel的对应工作表
            df_result1.to_excel(writer, sheet_name=key[:-9], index=False, columns=columns_to_write)

def sanitize_filename(filename):
    return filename.replace('/', '_').replace('\\', '_').replace(':', '_')


def draw_figure(input_file):
    # 读取表格数据
    file_path = f'../output/03_{input_file}_shadow_price.xlsx'  # 替换为你的文件路径
    df = pd.read_excel(file_path)

    # 设置 Year 列为索引
    df.set_index('Year', inplace=True)

    # 遍历每一列并绘制点线图
    for column in df.columns:
        # 计算均值和标准差
        mean = df[column].mean()
        std = df[column].std()

        # 将超出均值±3倍标准差的值替换为NaN
        df[column] = df[column].where((df[column] <= mean + 3*std) & (df[column] >= mean - 3*std), np.nan)

        plt.figure()
        plt.plot(df.index, df[column], marker='o', linestyle='-', label=column, color='b')

        # 检查是否有NaN值
        has_nan = df[column].isna().any()

        if has_nan:
            # 绘制带断点的图
            plt.plot(df.index, df[column].interpolate(), linestyle='--', color='r')  # 插值显示断点

        plt.xlabel('Year')
        plt.ylabel(column)
        plt.title(f'{input_file} {column}')
        plt.legend()
        plt.grid(True)

        sanitized_column = sanitize_filename(column)
        plt.savefig(f'../Figure/{input_file}_{sanitized_column}.png')  # 保存图像为文件
        plt.close()  # 关闭当前图像，避免内存问题
        # plt.show()
# def draw_figure(input_file):
#     # 读取表格数据
#     file_path = f'../output/03_{input_file}_shadow_price.xlsx'  # 替换为你的文件路径
#     df = pd.read_excel(file_path)
#
#     # 设置 Year 列为索引
#     df.set_index('Year', inplace=True)
#
#     # 遍历每一列并绘制点线图
#     for column in df.columns:
#         # 计算均值和标准差
#         mean = df[column].mean()
#         std = df[column].std()
#
#         # 将超出均值±3倍标准差的值标记为异常值
#         is_outlier = (df[column] > mean + 3*std) | (df[column] < mean - 3*std)
#
#         # 获取非异常值的最大最小值
#         non_outlier_max = df[column][~is_outlier].max()
#         non_outlier_min = df[column][~is_outlier].min()
#
#         # 设置断点范围（略低于断点的值，略高于去掉断点后的最大值）
#         break_start = mean + 3 * std - 0.1 * std
#         break_end = non_outlier_max + 0.1 * std
#
#         # 创建带有断点的双轴图
#         fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10, 6), gridspec_kw={'height_ratios': [3, 1]})
#
#         # 绘制正常值
#         ax1.plot(df.index, df[column], marker='o', linestyle='-', label=column, color='b')
#         ax2.plot(df.index, df[column], marker='o', linestyle='-', label=column, color='b')
#
#         # 隐藏异常值
#         ax1.plot(df.index[is_outlier], df[column][is_outlier], 'o', color='white')
#         ax2.plot(df.index[~is_outlier], df[column][~is_outlier], 'o', color='white')
#
#         # 设置轴范围
#         ax1.set_ylim(break_start, df[column].max() + std)
#         ax2.set_ylim(df[column].min() - std, break_end)
#
#         # 添加断点标记
#         d = .015  # 断点大小
#         kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False)
#         ax1.plot((-d, +d), (-d, +d), **kwargs)        # top-left diagonal
#         ax1.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal
#
#         kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
#         ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
#         ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal
#
#         plt.xlabel('Year')
#         plt.ylabel(column)
#         plt.title(f'{input_file} {column}')
#         plt.legend()
#         plt.grid(True)
#
#         sanitized_column = sanitize_filename(column)
#         plt.savefig(f'../Figure/{input_file}_{sanitized_column}.png')  # 保存图像为文件
#         plt.close()  # 关闭当前图像，避免内存问题
#

def calculate_price(input_file,use_parallel=False):
    # amortize_costs(path_name)
    calculate_cost_revenue(input_file,use_parallel)
    calculate_price_cells(input_file)
    caculate_shadow_price(input_file, percentile_num=97, mask_use=True)
    # caculate_carbonprice_compare(input_file)
    # draw_histogram(input_file)
    # summarize_from_csv(path_name)