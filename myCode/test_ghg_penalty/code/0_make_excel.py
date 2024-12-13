import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tools import get_path

def get_folders(file_path):
    """获取文件夹"""
    df = pd.read_csv(file_path)
    all_columns = df.columns.tolist()
    filtered_columns = [col for col in all_columns if col not in ["Name", "Default_run"]]
    return filtered_columns

results = pd.DataFrame(columns=["Folder", "Weight", "GHG Deviation %", "Demand Difference", "Profit"])
custom_runs = ["setting_template_windows", "setting_template_windows_1", "setting_template_windows_2"]

# 指定需要绘图的分组
selected_groups = ["0", "1", "2", "3", "4", "5"]
output_file = "../Result/ghg_weight_results_1.xlsx"

# 构建筛选关键词
keywords = [f"_{keyword}_w" for keyword in selected_groups]

for custom_run in custom_runs:
    file_path = f"../../tasks_run/Custom_runs/{custom_run}.csv"
    folders = get_folders(file_path)

    # 筛选文件夹
    folders = [folder for folder in folders if any(keyword in folder for keyword in keywords)]

    for folder in folders:
        # 获取文件夹路径
        folder_path = get_path(folder)

        # 处理 GHG 数据
        folder_dir = os.path.join(folder_path, 'out_2050', 'GHG_emissions_2050.csv')
        df_ghg = pd.read_csv(folder_dir, index_col=0)
        ghg_limit = df_ghg.loc["GHG_EMISSIONS_LIMIT_TCO2e", "Emissions (t CO2e)"]
        ghg_actual = df_ghg.loc["GHG_EMISSIONS_TCO2e", "Emissions (t CO2e)"]
        ghg_difference = ghg_actual - ghg_limit
        ghg_deviation_ratio = abs(ghg_difference / ghg_limit) * 100

        # 处理 Demand 数据
        quantity_file = os.path.join(folder_path, 'out_2050', 'quantity_comparison_2050.csv')
        df_demand = pd.read_csv(quantity_file)
        demand_base = df_demand["Prod_base_year (tonnes, KL)"].sum()
        demand_target = df_demand["Prod_targ_year (tonnes, KL)"].sum()
        demand_difference = demand_target - demand_base
        demand_deviation_ratio = abs(demand_difference / demand_target) * 100

        # 处理 Profit 数据
        cost_files = [
            os.path.join(folder_path, 'out_2050', file)
            for file in os.listdir(os.path.join(folder_path, 'out_2050'))
            if file.startswith("cost_")
        ]
        total_cost = sum(pd.read_csv(file)[
            "Value ($)" if "Value ($)" in pd.read_csv(file).columns else "Cost ($)"
        ].sum() for file in cost_files)

        revenue_files = [
            os.path.join(folder_path, 'out_2050', file)
            for file in os.listdir(os.path.join(folder_path, 'out_2050'))
            if file.startswith("revenue_")
        ]
        total_revenue = sum(pd.read_csv(file)["Value ($)"].sum() for file in revenue_files)

        # 计算 Profit
        profit = total_revenue - total_cost

        # 处理 Weight 数据
        txt_path = os.path.join(folder_path, "model_run_settings.txt")
        pattern = re.compile(r"SOLVE_WEIGHT_DEVIATIONS:(\d+(\.\d+)?)")
        with open(txt_path, "r") as file:
            content = file.read()
        match = pattern.search(content)
        if match:
            solve_weight = float(match.group(1))
        else:
            solve_weight = None

        # 添加结果到 DataFrame
        results = pd.concat([results, pd.DataFrame([{
            "Folder": folder,
            "Weight": solve_weight,
            "GHG Deviation %": ghg_deviation_ratio,
            "Demand deviation ratio %": demand_deviation_ratio,
            "Profit": profit
        }])], ignore_index=True)

# 保存到 Excel
results.to_excel(output_file, index=False)

# 提取 `_0_` 分组信息
results["Group"] = results["Folder"].str.extract(r'_(\d+)_')[0]

# 筛选数据
filtered_results = results[results["Group"].isin(selected_groups)]

# 计算全局最小值和最大值
y_ranges = {
    "GHG Deviation %": [results["GHG Deviation %"].min(), results["GHG Deviation %"].max()],
    "Demand deviation ratio %": [results["Demand deviation ratio %"].min(), results["Demand deviation ratio %"].max()],
    "Profit": [results["Profit"].min(), results["Profit"].max()]
}

# 绘制点线图
output_path = "../Figure"
os.makedirs(output_path, exist_ok=True)

# 按分组绘制图表
for group, group_df in filtered_results.groupby("Group"):
    # GHG Deviation %
    plt.figure()
    plt.plot(group_df["Weight"], group_df["GHG Deviation %"], marker='o', label=f"Group {group}")
    plt.xlabel("Weight")
    plt.xticks(np.arange(0.8, 1.01, 0.02))
    plt.ylabel("GHG Deviation %")
    plt.ylim(y_ranges["GHG Deviation %"])  # 设置统一的Y轴范围
    plt.title(f"GHG Deviation % for Group {group}")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_path, f"ghg_deviation_group_{group}.png"), dpi=300)
    plt.close()

    # Demand deviation ratio %
    plt.figure()
    plt.plot(group_df["Weight"], group_df["Demand deviation ratio %"], marker='o', label=f"Group {group}")
    plt.xlabel("Weight")
    plt.xticks(np.arange(0.8, 1.01, 0.02))
    plt.ylabel("Demand deviation ratio %")
    plt.ylim(y_ranges["Demand deviation ratio %"])  # 设置统一的Y轴范围
    plt.title(f"Demand Deviation Ratio % for Group {group}")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_path, f"demand_deviation_group_{group}.png"), dpi=300)
    plt.close()

    # Profit
    plt.figure()
    plt.plot(group_df["Weight"], group_df["Profit"], marker='o', label=f"Group {group}")
    plt.xlabel("Weight")
    plt.ylabel("Profit")
    plt.ylim(9870000000,1.34000e+11)  # 设置统一的Y轴范围
    plt.xticks(np.arange(0.8, 1.01, 0.02))
    plt.title(f"Profit for Group {group}")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_path, f"profit_group_{group}.png"), dpi=300)
    plt.close()

print(f"所有图表已生成并保存到 {output_path}.")

