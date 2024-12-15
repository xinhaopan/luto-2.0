import pandas as pd
import os
import re
from tools import get_path


def get_folders_containing_string(path, string):
    """筛选出路径下包含特定字符串的文件夹"""
    return [f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f)) and string in f]

def get_folders(file_path):
    """获取文件夹"""
    df = pd.read_csv(file_path)

    # 获取所有列名
    all_columns = df.columns.tolist()

    # 排除 'Name' 和 'Default_run' 列
    filtered_columns = [col for col in all_columns if col not in ["Name", "Default_run"]]
    return filtered_columns

def calculate_deviation(path, folder):
    """
    计算并保存 GHG 和 Demand 偏差数据
    """
    # 用列表收集每年的数据
    all_data = []

    for year in range(2011, 2051):
        # 处理 GHG 数据
        df_ghg = pd.read_csv(os.path.join(path, f"out_{year}", f"GHG_emissions_{year}.csv"), index_col=0)
        ghg_limit = df_ghg.loc["GHG_EMISSIONS_LIMIT_TCO2e", "Emissions (t CO2e)"]
        ghg_actual = df_ghg.loc["GHG_EMISSIONS_TCO2e", "Emissions (t CO2e)"]
        ghg_difference = abs(ghg_actual - ghg_limit)
        ghg_deviation_ratio = ghg_difference / ghg_limit * 100

        # 处理 Demand 数据
        df_demand = pd.read_csv(os.path.join(path, f"out_{year}", f"quantity_comparison_{year}.csv"))
        demand_base = df_demand["Prod_base_year (tonnes, KL)"].sum()
        demand_target = df_demand["Prod_targ_year (tonnes, KL)"].sum()
        demand_difference = demand_target - demand_base
        demand_deviation_ratio = demand_difference / demand_base * 100

        # 处理 Cost 和 Revenue 数据
        total_cost = sum(pd.read_csv(os.path.join(path, f"out_{year}", file))[(
            "Value ($)" if "Value ($)" in pd.read_csv(os.path.join(path, f"out_{year}", file)).columns else "Cost ($)"
        )].sum() for file in os.listdir(os.path.join(path, f"out_{year}")) if file.startswith("cost_"))

        total_revenue = sum(pd.read_csv(os.path.join(path, f"out_{year}", file))["Value ($)"].sum()
                            for file in os.listdir(os.path.join(path, f"out_{year}")) if file.startswith("revenue_"))

        # 计算 Profit
        profit = total_revenue - total_cost

        # 添加当前年的数据到列表
        all_data.append({
            "Year": year,
            "GHG Difference": ghg_difference,
            "GHG Deviation Ratio (%)": ghg_deviation_ratio,
            "Demand Difference": demand_difference,
            "Demand Deviation Ratio (%)": demand_deviation_ratio,
            "Demand_target": demand_target,
            "Profit": profit
        })

    # 用收集的数据一次性创建 DataFrame
    df_devation = pd.DataFrame(all_data)

    return df_devation



def extract_log_coeff_data(path, folder):
    """
    从日志中提取年份、GHG 偏差、Demand 偏差、Economic Objective 和 Objective Value 数据
    """
    # 定义结果 DataFrame
    df_result = pd.DataFrame(columns=["Year", "GHG Deviation", "Demand Deviation", "Economic Objective", "Objective Value"])

    # 定义正则表达式
    year_pattern = re.compile(
        r"(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) - Running for year (?P<year>\d{4})"
    )
    deviation_pattern = re.compile(
        r"(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) - GHG deviation Coeff: (?P<ghg_deviation>[\d\.e\-]+); "
        r"Demand deviation Coeff: (?P<demand_deviation>[\d\.e\-]+);\s+Economic objective Coeff: (?P<economic_objective>[\d\.e\-]+)"
    )
    objective_pattern = re.compile(
        r"(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) - Objective value: (?P<objective_value>[\d\.e\-]+)"
    )

    # 提取日志文件的路径
    timestamp_pattern = re.compile(r"\d{4}_\d{2}_\d{2}__\d{2}_\d{2}_\d{2}")
    match = timestamp_pattern.search(path)
    if not match:
        raise ValueError(f"无法从路径 {path} 提取时间戳。")
    log_timestamp = match.group()
    txt_file_path = os.path.join(path, f"run_{log_timestamp}_stdout.log")

    # 读取日志文件内容
    if not os.path.exists(txt_file_path):
        raise FileNotFoundError(f"日志文件不存在: {txt_file_path}")

    with open(txt_file_path, "r") as file:
        lines = file.readlines()

    # 用于临时存储年份和匹配的值
    current_year = None
    for line in lines:
        # 匹配年份
        match_year = year_pattern.search(line)
        if match_year:
            current_year = int(match_year.group("year"))
            continue

        # 匹配 GHG、Demand 和 Economic Objective
        match_deviation = deviation_pattern.search(line)
        if match_deviation and current_year is not None:
            ghg_deviation = float(match_deviation.group("ghg_deviation"))
            demand_deviation = float(match_deviation.group("demand_deviation")) / 26
            economic_objective = float(match_deviation.group("economic_objective")) / 917199
            timestamp = match_deviation.group("timestamp")

            # 添加到 DataFrame
            df_result = pd.concat([df_result, pd.DataFrame([{
                "Year": current_year,
                "GHG Deviation": ghg_deviation,
                "Demand Deviation": demand_deviation,
                "Economic Objective": economic_objective,
                "Objective Value": None  # 占位符
            }])], ignore_index=True)
            continue

        # 匹配 Objective Value
        match_objective = objective_pattern.search(line)
        if match_objective:
            objective_value = float(match_objective.group("objective_value"))
            timestamp = match_objective.group("timestamp")

            # 更新最后一行的 Objective Value
            if not df_result.empty:
                df_result.iloc[-1, df_result.columns.get_loc("Objective Value")] = objective_value

    return df_result

def extract_log_data(path, folder):
    """
    从日志中提取年份、GHG 偏差、Demand 偏差、Economic Objective 和 Objective Value 数据
    """
    # 定义结果 DataFrame
    df_result = pd.DataFrame(columns=["Year", "GHG Deviation Value", "Demand Deviation Value", "Economic Objective Value", "Objective Value"])

    # 定义正则表达式
    year_pattern = re.compile(
        r"(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) - Running for year (?P<year>\d{4})"
    )
    deviation_pattern = re.compile(
        r"(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) - GHG deviation value: (?P<ghg_deviation>[\d.e\-]+); "
        r"Demand deviation value: (?P<demand_deviation>[\d.e\-]+); Economic objective value: (?P<economic_objective>[\d.e\-]+)"
    )
    objective_pattern = re.compile(
        r"(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) - Objective value: (?P<objective_value>[\d\.e\-]+)"
    )

    # 提取日志文件的路径
    timestamp_pattern = re.compile(r"\d{4}_\d{2}_\d{2}__\d{2}_\d{2}_\d{2}")
    match = timestamp_pattern.search(path)
    if not match:
        raise ValueError(f"无法从路径 {path} 提取时间戳。")
    log_timestamp = match.group()
    txt_file_path = os.path.join(path, f"run_{log_timestamp}_stdout.log")

    # 读取日志文件内容
    if not os.path.exists(txt_file_path):
        raise FileNotFoundError(f"日志文件不存在: {txt_file_path}")

    with open(txt_file_path, "r") as file:
        lines = file.readlines()

    # 用于临时存储年份和匹配的值
    current_year = None
    for line in lines:
        # 匹配年份
        match_year = year_pattern.search(line)
        if match_year:
            current_year = int(match_year.group("year"))
            continue

        # 匹配 GHG、Demand 和 Economic Objective
        match_deviation = deviation_pattern.search(line)
        if match_deviation and current_year is not None:
            ghg_deviation = float(match_deviation.group("ghg_deviation"))
            demand_deviation = float(match_deviation.group("demand_deviation")) / 26
            economic_objective = float(match_deviation.group("economic_objective")) / 917199
            timestamp = match_deviation.group("timestamp")

            # 添加到 DataFrame
            df_result = pd.concat([df_result, pd.DataFrame([{
                "Year": current_year,
                "GHG Deviation Value": ghg_deviation,
                "Demand Deviation Value": demand_deviation,
                "Economic Objective Value": economic_objective,
                "Objective Value": None  # 占位符
            }])], ignore_index=True)
            continue

        # 匹配 Objective Value
        match_objective = objective_pattern.search(line)
        if match_objective:
            objective_value = float(match_objective.group("objective_value"))
            timestamp = match_objective.group("timestamp")

            # 更新最后一行的 Objective Value
            if not df_result.empty:
                df_result.iloc[-1, df_result.columns.get_loc("Objective Value")] = objective_value

    return df_result


def match_files_in_folder(keywords, folder_path="../../../output"):
    """
    根据指定的关键字匹配文件夹内所有合适的文件名，返回匹配的文件名列表。

    :param folder_path: 文件夹路径
    :param keywords: 字符串或字符串列表，用于匹配文件名
    :return: 匹配的文件名列表
    """
    # 确保 keywords 是列表
    if isinstance(keywords, str):
        keywords = [keywords]

    matched_files = []

    # 遍历文件夹中的所有文件
    for root, dirs, _ in os.walk(folder_path):
        for dir_name in dirs:
            # 如果文件夹名称包含任何一个关键字，加入列表
            if any(keyword in dir_name for keyword in keywords):
                matched_files.append(dir_name)

    return matched_files

file_path = "../../tasks_run/Custom_runs/setting_template_windows_9.csv"
# folders = get_folders(file_path)
folders = match_files_in_folder("20241205_10_w10_GHG_1_8C_67_BIO_0_2")
output_log_coeff_file = "../Result/output_log_coeff.xlsx"
output_log_file = "../Result/output_log.xlsx"
output_result_file = "../Result/output_result12.xlsx"

# 写入 df_devation 到 output_result_file
with pd.ExcelWriter(output_result_file, engine="openpyxl") as result_writer:
    for folder in folders:
        print(f"Processing folder for deviations: {folder}")
        path = get_path(folder)  # 获取路径
        df_devation = calculate_deviation(path, folder)
        sheet_name = '_'.join(f"{folder}_devation".split('_')[2:])[:31]  # 确保 sheet_name 不超过 31 个字符
        df_devation.to_excel(result_writer, sheet_name=sheet_name, index=False)
    print(f"Deviation results saved to {output_result_file}")

with pd.ExcelWriter(output_log_file, engine="openpyxl") as log_writer:
    for folder in folders:
        print(f"Processing folder for logs: {folder}")
        path = get_path(folder)  # 获取路径
        df_result = extract_log_data(path, folder)
        sheet_name = '_'.join(f"{folder}_value".split('_')[2:])[:31]  # 确保 sheet_name 不超过 31 个字符
        df_result.to_excel(log_writer, sheet_name=sheet_name, index=False)
    print(f"Log results saved to {output_log_file}")

# 写入 df_result 到 output_log_file
with pd.ExcelWriter(output_log_coeff_file, engine="openpyxl") as log_writer:
    for folder in folders:
        print(f"Processing folder for logs: {folder}")
        path = get_path(folder)  # 获取路径
        df_result = extract_log_coeff_data(path, folder)
        sheet_name = '_'.join(f"{folder}_coeff".split('_')[2:])[:31]  # 确保 sheet_name 不超过 31 个字符
        df_result.to_excel(log_writer, sheet_name=sheet_name, index=False)
    print(f"Log results saved to {output_log_coeff_file}")


