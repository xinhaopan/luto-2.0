import numpy as np
import numpy_financial as npf
import os
import matplotlib.pyplot as plt
import pandas as pd

from tqdm import *


from myCode.carbonprice.helpers import *

def amortize_costs(path_name, file_name = "cost_transition_ag2non_ag", rate=0.07, horizon=30):
    path_name = "output/" + path_name
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

    # 读取所有年份的文件
    for year in trange(2010, 2051):
        full_file_name = os.path.join(path_name, f"out_{year}/data_for_carbon_price/{file_name}_{year}.npy")
        costs[year] = np.load(full_file_name)

    years = list(costs.keys())
    start_year = min(years)
    end_year = max(years)

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
        output_file_name = os.path.join(path_name, f"out_{year}/data_for_carbon_price/{file_name}_amortised_{year}.npy")
        np.save(output_file_name, new_costs[year])


# path_name = "output/2024_06_06__15_17_00_soft_mincost_RF3_P1e5_2010-2050_timeseries_-269Mt"
# file_name = "cost_transition_ag2non_ag"
# rate = 0.07  # 假设的利率
# horizon = 30  # 时间跨度30年

# amortize_costs(path_name, file_name, rate, horizon)


def calculate_payment(path_name):
    print("Start to calculate payment.")
    path_name = "output/" + path_name

    # 初始化结果列表
    results = []
    columns_name = ["Year", "cost_ag(M$)", "cost_am(M$)", "cost_non_ag(M$)", "cost_transition_ag2ag(M$)",
                    "cost_transition_ag2non_ag_amortised(M$)", "revenue_ag(M$)", "revenue_am(M$)", "revenue_non_ag(M$)",
                    "GHG Emission target(MtCOe2)", "Total cost for Discriminatory Payment(M$)", "GHG Abatement(MCOe2)",
                    "Carbon Price for Discriminatory Payment average($/tCO2e)"]

    # 初始化字典来存储每年的利润
    profits = {}

    for year in trange(2010, 2051):
        rows_nums = [year]
        path_dir = os.path.join(path_name, f"out_{year}/data_for_carbon_price")
        prefixes = columns_name[1:9]

        for prefix in prefixes:
            arr = sum_files(path_dir, prefix[:-4])
            rows_nums.append(np.sum(arr) / 1000000)

        ghg_arr = sum_files(path_dir, "GHG_")
        rows_nums.append(np.sum(ghg_arr) / 1000000)

        # 读取成本和收入数组
        cost_arr = np.load(os.path.join(path_dir, f"cost_ag_{year}.npy"))
        cost_arr += sum_files(path_dir, "cost_non_ag_")
        cost_arr += sum_files(path_dir, "cost_am_")

        rev_arr = np.load(os.path.join(path_dir, f"revenue_ag_{year}.npy"))
        rev_arr += sum_files(path_dir, "revenue_non_ag_")
        rev_arr += sum_files(path_dir, "revenue_am_")

        profit_arr = rev_arr - cost_arr
        profits[year] = profit_arr  # 存储当年的利润

        if year > 2010:
            # 直接从字典中获取前一年的利润
            profit_arr_last = profits[year - 1]

            tra_arr = np.load(os.path.join(path_dir, f"cost_transition_ag2ag_{year}.npy")) + np.load(
                os.path.join(path_dir, f"cost_transition_ag2non_ag_amortised_{year}.npy"))
            payment_arr = tra_arr + (profit_arr - profit_arr_last)
            rows_nums.append(np.sum(payment_arr) / 1000000)

            ghg_arr -= sum_files(os.path.join(path_name, f"out_{year - 1}/data_for_carbon_price"), "GHG_ag_")
            ghg_arr = -ghg_arr
            rows_nums.append(np.sum(ghg_arr) / 1000000)

            with np.errstate(divide='ignore', invalid='ignore'):
                cp_arr = np.divide(payment_arr, ghg_arr)
                cp_arr[np.isinf(cp_arr)] = 0
                cp_arr = np.nan_to_num(cp_arr, nan=0.0)
            rows_nums.append(np.sum(cp_arr * ghg_arr) / np.sum(ghg_arr))

            save_path = os.path.join(path_name, "data_for_carbon_price")
            os.makedirs(save_path, exist_ok=True)
            np.save(os.path.join(save_path, f"payment_{year}.npy"), payment_arr)
            np.save(os.path.join(save_path, f"ghg_{year}.npy"), ghg_arr)
            np.save(os.path.join(save_path, f"cp_{year}.npy"), cp_arr)

        results.append(rows_nums)

    df_results = pd.DataFrame(results, columns=columns_name)
    df_results.to_excel(os.path.join(path_name, "cost_revenue_summary1.xlsx"), index=False)

def caculate_carbonprice(path_name,percentile_num=95,mask_use=True):
    print("Start to calculate carbon price.")
    path_name = "output/" + path_name
    # 初始化列表以存储每年的数据
    results = []
    for year in trange(2011, 2051):
        # 加载每年的 .npy 文件
        payment_arr = np.load(os.path.join(path_name, "data_for_carbon_price", f"payment_{year}.npy"))
        ghg_arr = np.load(os.path.join(path_name, "data_for_carbon_price", f"ghg_{year}.npy"))
        cp_arr = np.load(os.path.join(path_name, "data_for_carbon_price", f"cp_{year}.npy"))

        output_file = f"carbon_price_analysis_{percentile_num}_nomask1.csv"
        # 使用 mask 过滤掉 ghg_arr < 1的值
        if mask_use:
            mask = ghg_arr >= 1
            cp_arr1 = cp_arr[mask]
            output_file =  f"carbon_price_analysis_{percentile_num}_mask1.csv"

        # 计算所需的值
        carbon_price_uniform = np.percentile(cp_arr1, percentile_num)
        total_emissions_abatement = np.sum(ghg_arr)
        total_cost_uniform = carbon_price_uniform * total_emissions_abatement
        total_cost_discriminatory = np.sum(cp_arr * ghg_arr)
        carbon_price_discriminatory_avg = total_cost_discriminatory / total_emissions_abatement if total_emissions_abatement != 0 else 0

        # 将计算结果添加到列表中
        results.append({
            "Year": year,
            "Total emissions abatement (MtCO2e)": total_emissions_abatement / 1000000,
            "Total cost for Uniform Payment (M$)": total_cost_uniform / 1000000,
            "Total cost for Discriminatory Payment (M$)": total_cost_discriminatory / 1000000,
            "Carbon price for Uniform Payment ($/tCO2e)": carbon_price_uniform,
            "Carbon Price for Discriminatory Payment average ($/tCO2e)": carbon_price_discriminatory_avg
        })

    # 创建DataFrame
    df_results = pd.DataFrame(results)

    # 保存结果到CSV文件
    output_csv_path = os.path.join(path_name, output_file)
    df_results.to_csv(output_csv_path, index=False)

def caculate_carbonprice_compare(path_name,percentiles = [100, 99, 98, 97, 96, 95, 94, 93, 92, 91, 90, 89, 88, 87, 86, 85, 84, 83, 82, 81, 80]):
    print("Start to calculate carbon price.")
    path_name = "output/" + path_name
    # 定义分位数列表
    # percentiles = [100, 99, 98, 97, 96, 95, 94, 93, 92, 91, 90, 89, 88, 87, 86, 85, 84, 83, 82, 81, 80]

    # 初始化表格字典
    carbon_price_with_mask = {year: [] for year in range(2011, 2051)}
    ghg_max_cp_with_mask = {year: [] for year in range(2011, 2051)}
    cost_max_cp_with_mask = {year: [] for year in range(2011, 2051)}

    carbon_price_without_mask = {year: [] for year in range(2011, 2051)}
    ghg_max_cp_without_mask = {year: [] for year in range(2011, 2051)}
    cost_max_cp_without_mask = {year: [] for year in range(2011, 2051)}

    # 循环每个年份
    for year in trange(2011, 2051):
        # 加载每年的 .npy 文件
        payment_arr = np.load(os.path.join(path_name, "data_for_carbon_price", f"payment_{year}.npy"))
        ghg_arr = np.load(os.path.join(path_name, "data_for_carbon_price", f"ghg_{year}.npy"))
        cp_arr = np.load(os.path.join(path_name, "data_for_carbon_price", f"cp_{year}.npy"))

        # 不使用 mask 直接使用原始数组
        filtered_cp_arr_without_mask = cp_arr
        filtered_ghg_arr_without_mask = ghg_arr
        filtered_payment_arr_without_mask = payment_arr

        # 使用 mask 过滤掉 ghg_arr 在 -1 到 1 范围内的值
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
    output_excel_path = os.path.join(path_name, "carbon_price_comparison1.xlsx")
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
    path_name = "output/" + path_name
    # 加载 2050 年的 .npy 文件
    payment_arr_2050 = np.load(os.path.join(path_name, "data_for_carbon_price", f"payment_{year}.npy"))
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
    plt.savefig(os.path.join(path_name, f"histogram_{year}.png"))
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
    results = [process_year_data(year, path_name, categories, column_keywords) for year in range(2010, 2051)]
    df_result = pd.DataFrame(results)

    df_result.to_excel(f'{path_name}/calculated_carbon_price_fromcsv.xlsx', index=False)

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

    with pd.ExcelWriter(f'{path_name}/calculated_carbon_price_fromcsv.xlsx', engine='openpyxl', mode='a',
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

def carbon_price(path_name):
    # amortize_costs(path_name)
    calculate_payment(path_name)
    caculate_carbonprice(path_name)
    caculate_carbonprice_compare(path_name)
    draw_histogram(path_name)
    summarize_from_csv(path_name)