import pandas as pd
import os
import re
from tools import get_path


def get_folders_containing_string(path, string):
    """筛选出路径下包含特定字符串的文件夹"""
    return [f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f)) and string in f]

base_path = "../../../output"
string = "20241203_test_2_GHG_1_8C_67_BIO_0"
filtered_folders = get_folders_containing_string(base_path, string)

for folder in filtered_folders:
    path = get_path(folder)
    df_devation = pd.DataFrame(columns=[
        "Year",
        "GHG Absolute Difference",
        "GHG Deviation Ratio",
        "Demand Absolute Difference",
        "Demand Deviation Ratio",
        "Profit"
    ])

    for year in range(2011, 2051):
        # 处理 GHG 数据
        df_ghg = pd.read_csv(os.path.join(path, f"out_{year}", f"GHG_emissions_{year}.csv"), index_col=0)
        ghg_limit = df_ghg.loc["GHG_EMISSIONS_LIMIT_TCO2e", "Emissions (t CO2e)"]
        ghg_actual = df_ghg.loc["GHG_EMISSIONS_TCO2e", "Emissions (t CO2e)"]
        ghg_difference = ghg_actual - ghg_limit
        ghg_absolute_difference = abs(ghg_difference)
        ghg_deviation_ratio = ghg_difference / ghg_limit

        # 处理 Demand 数据
        df_demand = pd.read_csv(os.path.join(path, f"out_{year}", f"quantity_comparison_{year}.csv"))
        demand_base = df_demand["Prod_base_year (tonnes, KL)"].sum()
        demand_target = df_demand["Prod_targ_year (tonnes, KL)"].sum()
        demand_difference = demand_target - demand_base
        demand_absolute_difference = abs(demand_difference)
        demand_deviation_ratio = demand_difference / demand_base

        # 处理 Cost 数据
        cost_files = [
            os.path.join(path, f"out_{year}", file)
            for file in os.listdir(os.path.join(path, f"out_{year}"))
            if file.startswith(f"cost_") and file.endswith(f"_{year}.csv")
        ]
        total_cost = 0
        for cost_file in cost_files:
            df_cost = pd.read_csv(cost_file)
            total_cost += df_cost["Value ($)"].sum() if "Value ($)" in df_cost.columns else df_cost["Cost ($)"].sum()

        # 处理 Revenue 数据
        revenue_files = [
            os.path.join(path, f"out_{year}", file)
            for file in os.listdir(os.path.join(path, f"out_{year}"))
            if file.startswith(f"revenue_") and file.endswith(f"_{year}.csv")
        ]
        total_revenue = 0
        for revenue_file in revenue_files:
            df_revenue = pd.read_csv(revenue_file)
            total_revenue += df_revenue["Value ($)"].sum()

        # 计算 Profit
        profit = total_revenue - total_cost

        # 将当前年的结果存储为 DataFrame 行
        current_data = pd.DataFrame([{
            "Year": year,
            "GHG Absolute Difference": ghg_absolute_difference,
            "GHG Deviation Ratio": ghg_deviation_ratio,
            "Demand Absolute Difference": demand_absolute_difference,
            "Demand Deviation Ratio": demand_deviation_ratio,
            "Profit": profit
        }])

        # 合并结果
        df_devation = pd.concat([df_devation, current_data], ignore_index=True)

    df_devation.to_excel(f"../{folder}_devation_from_result.xlsx", index=False)

    # 定义存储结果的 DataFrame
    df_result = pd.DataFrame(columns=["Year", "Demand Deviation", "GHG Deviation", "Economic Objective"])

    # 定义正则表达式
    # 第一个部分匹配 Demand deviation 和 economic objective
    demand_pattern = re.compile(
        r"(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) - Demand deviation: (?P<demand_deviation>[\d\.e\-]+); "
        r"GHG deviation: (?P<ghg_deviation>[\d\.e\-]+),\s+economic objective: (?P<economic_objective>[\d\.e\-]+)"
    )
    # 第二个部分匹配年份
    year_pattern = re.compile(r"Processing for (?P<year>\d{4}) completed in (?P<time_taken>\d+) seconds")
    pattern = re.compile(r"\d{4}_\d{2}_\d{2}__\d{2}_\d{2}_\d{2}")
    match = pattern.search(path).group()
    # 读取日志文件
    txt_file_path = f"{path}/run_{match}_stdout.log"

    with open(txt_file_path, "r") as file:
        lines = file.readlines()

    # 将多行内容组合处理
    content = "\n".join(lines)

    # 匹配所有符合 demand_pattern 的内容
    demand_matches = demand_pattern.findall(content)

    # 匹配所有符合 year_pattern 的内容
    year_matches = year_pattern.findall(content)

    # 确保两种匹配数量一致
    if len(demand_matches) != len(year_matches):
        raise ValueError("Demand deviation and year information count mismatch.")

    # 遍历匹配结果并添加到 DataFrame
    for i in range(len(demand_matches)):
        timestamp, demand_deviation, ghg_deviation, economic_objective = demand_matches[i]
        year, _ = year_matches[i]

        # 添加到 DataFrame
        df_result = pd.concat([df_result, pd.DataFrame([{
            "Year": int(year),
            "Demand Deviation": float(demand_deviation),
            "GHG Deviation": float(ghg_deviation),
            "Economic Objective": float(economic_objective)
        }])], ignore_index=True)

    # 保存结果到 CSV 文件
    df_result.to_excel(f"../{folder}_devation_from_log.xlsx", index=False)