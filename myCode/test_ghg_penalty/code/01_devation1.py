import pandas as pd
import os
import re
from tools import get_path
from joblib import Parallel, delayed

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)



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
        ghg_difference = ghg_actual - ghg_limit
        ghg_deviation_ratio = ghg_difference / ghg_limit * 100

        # 处理 Demand 数据
        df_demand = pd.read_csv(os.path.join(path, f"out_{year}", f"quantity_comparison_{year}.csv"))
        demand_base = df_demand["Prod_base_year (tonnes, KL)"].sum()
        demand_target = df_demand["Prod_targ_year (tonnes, KL)"].sum()
        demand_difference = (demand_target - demand_base)
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

def process_deviation_task(folder):
    path = get_path(folder)  # 获取路径
    df_devation = calculate_deviation(path, folder)
    sheet_name = '_'.join(f"{folder}_deviation".split('_')[1:])[:31]  # 确保 sheet_name 不超过 31 个字符
    return sheet_name, df_devation

def process_log_task(folder):
    path = get_path(folder)  # 获取路径
    df_result = extract_log_data(path, folder)
    sheet_name = '_'.join(f"{folder}_value".split('_')[2:])[:31]  # 确保 sheet_name 不超过 31 个字符
    return sheet_name, df_result

def process_log_coeff_task(folder):
    path = get_path(folder)  # 获取路径
    df_result = extract_log_coeff_data(path, folder)
    sheet_name = '_'.join(f"{folder}_coeff".split('_')[2:])[:31]  # 确保 sheet_name 不超过 31 个字符
    return sheet_name, df_result


def process_tasks(use_parallel=True):
    """
    处理任务，并通过 `use_parallel` 参数控制是否启用并行。
    """
    # 处理 deviation 任务
    if use_parallel:
        deviation_results = Parallel(n_jobs=-1)(delayed(process_deviation_task)(folder) for folder in folders)
    else:
        deviation_results = [process_deviation_task(folder) for folder in folders]

    with pd.ExcelWriter(output_result_file, engine="openpyxl") as result_writer:
        for sheet_name, df_devation in deviation_results:
            df_devation.to_excel(result_writer, sheet_name=sheet_name, index=False)
    print(f"Deviation results saved to {output_result_file}")

    # 处理 log 任务
    if use_parallel:
        log_results = Parallel(n_jobs=-1)(delayed(process_log_task)(folder) for folder in folders)
    else:
        log_results = [process_log_task(folder) for folder in folders]

    with pd.ExcelWriter(output_log_file, engine="openpyxl") as log_writer:
        for sheet_name, df_result in log_results:
            df_result.to_excel(log_writer, sheet_name=sheet_name, index=False)
    print(f"Log results saved to {output_log_file}")

    # 处理 log coefficients 任务
    if use_parallel:
        log_coeff_results = Parallel(n_jobs=-1)(delayed(process_log_coeff_task)(folder) for folder in folders)
    else:
        log_coeff_results = [process_log_coeff_task(folder) for folder in folders]

    with pd.ExcelWriter(output_log_coeff_file, engine="openpyxl") as log_writer:
        for sheet_name, df_result in log_coeff_results:
            df_result.to_excel(log_writer, sheet_name=sheet_name, index=False)
    print(f"Log coefficients saved to {output_log_coeff_file}")

import os
import pandas as pd
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

def process_all_sheets(tasks, use_parallel=True):
    """
    处理所有Sheet的绘图任务。
    """
    if use_parallel:
        # 使用并行处理
        Parallel(n_jobs=-1)(
            delayed(process_sheet)(sheet_name, df, y_ranges, output_path)
            for sheet_name, df, y_ranges, output_path in tasks
        )
    else:
        # 顺序处理
        for sheet_name, df, y_ranges, output_path in tasks:
            process_sheet(sheet_name, df, y_ranges, output_path)
def process_sheet(sheet_name, df, y_ranges, output_path):
    """
    处理单个Sheet的绘图任务。
    """
    x = df['Year']

    # 绘制双Y轴图表
    plot_dual_axis(
        x,
        df['GHG Difference'],
        df['GHG Deviation Ratio (%)'],
        'GHG Difference',
        'GHG Deviation Ratio',
        'GHG Difference',
        'GHG Deviation Ratio (%)',
        y_ranges['GHG Difference'],
        y_ranges['GHG Deviation Ratio (%)'],
        f"GHG Differences - {sheet_name}",
        os.path.join(output_path, f"{sheet_name}_ghg.png")
    )

    plot_dual_axis(
        x,
        df['Demand Difference'],
        df['Demand Deviation Ratio (%)'],
        'Demand Difference',
        'Demand Deviation Ratio',
        'Demand Difference',
        'Demand Deviation Ratio (%)',
        y_ranges['Demand Difference'],
        y_ranges['Demand Deviation Ratio (%)'],
        f"Demand Differences - {sheet_name}",
        os.path.join(output_path, f"{sheet_name}_demand.png")
    )

    # 绘制单Y轴图表
    plot_single_axis(
        x,
        df['Profit'],
        'Profit',
        'Profit',
        y_ranges['Profit'],
        f"Profit - {sheet_name}",
        os.path.join(output_path, f"{sheet_name}_profit.png")
    )
    print(f"Sheet {sheet_name} 图表已生成。")

def calculate_global_min_max(excel_data, columns):
    """
    计算所有Sheet中指定列的全局最小值和最大值。
    """
    global_min_max = {col: [float('inf'), float('-inf')] for col in columns}
    for sheet_name in excel_data.sheet_names:
        df = excel_data.parse(sheet_name)
        for column in columns:
            if column in df.columns:
                global_min_max[column][0] = min(global_min_max[column][0], df[column].min())
                global_min_max[column][1] = max(global_min_max[column][1], df[column].max())
    return global_min_max

def plot_dual_axis(x, y1, y2, label1, label2, ylabel1, ylabel2, y1_range, y2_range, title, output_file):
    """
    绘制双Y轴点线图。
    """
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()

    ax1.plot(x, y1, 'g.-', label=label1)
    ax2.plot(x, y2, 'b.-', label=label2)

    ax1.set_xlabel('Year')
    ax1.set_ylabel(ylabel1, color='g')
    ax2.set_ylabel(ylabel2, color='b')
    ax1.set_ylim(y1_range)
    ax2.set_ylim(y2_range)

    plt.title(title)
    fig.legend(bbox_to_anchor=(0.45, 0.25))
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()

def plot_single_axis(x, y, label, ylabel, y_range, title, output_file):
    """
    绘制单Y轴点线图。
    """
    plt.figure()
    plt.plot(x, y, 'm.-', label=label)
    plt.xlabel('Year')
    plt.ylabel(ylabel)
    plt.ylim(y_range)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()

def get_local_min_max(df, columns):
    """
    获取当前Sheet中指定列的最小值和最大值。
    """
    local_min_max = {}
    for col in columns:
        if col in df.columns:
            local_min_max[col] = [df[col].min(), df[col].max()]
    return local_min_max

def process_all_sheets(tasks, use_parallel=True):
    """
    处理所有Sheet的绘图任务。
    """
    if use_parallel:
        # 使用并行处理
        Parallel(n_jobs=-1)(
            delayed(process_sheet)(sheet_name, df, y_ranges, output_path)
            for sheet_name, df, y_ranges, output_path in tasks
        )
    else:
        # 顺序处理
        for sheet_name, df, y_ranges, output_path in tasks:
            process_sheet(sheet_name, df, y_ranges, output_path)


if __name__ == "__main__":
    # file_path = "../../tasks_run/Custom_runs/setting_template_windows_1.csv"
    # folders = get_folders(file_path)
    i = 4
    folders = match_files_in_folder(f"20241206_{i}_w90_GHG_1_8C_67_BIO_0")
    output_log_coeff_file = f"../Result/output_log_coeff_{i}.xlsx"
    output_log_file = f"../Result/output_log_{i}.xlsx"
    output_result_file = f"../Result/output_result_{i}.xlsx"
    use_parallel = False  # 设置是否启用并行
    process_tasks(use_parallel)

    file_path = f'../result/output_result_{i}.xlsx'
    output_path = '../Figure'
    excel_data = pd.ExcelFile(file_path)

    # 设置是否并行执行
    use_parallel = False  # 修改为 False 即可关闭并行

    # 指定需要处理的列
    columns = ['GHG Difference', 'GHG Deviation Ratio (%)',
               'Demand Difference', 'Demand Deviation Ratio (%)', 'Profit']

    # 是否启用统一的Y轴范围
    use_global_y_range = True
    if use_global_y_range:
        global_min_max = calculate_global_min_max(excel_data, columns)
    else:
        global_min_max = None

    # 构建任务列表
    tasks = []
    for sheet_name in excel_data.sheet_names:
        df = excel_data.parse(sheet_name)

        # 确保所需列存在
        required_columns = ['Year'] + columns
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"Sheet {sheet_name} 缺少以下列：{missing_columns}，跳过...")
            continue

        # 获取当前Sheet的Y轴范围
        local_min_max = get_local_min_max(df, columns) if not use_global_y_range else None
        y_ranges = global_min_max if use_global_y_range else local_min_max

        # 添加任务
        tasks.append((sheet_name, df, y_ranges, output_path))

    process_all_sheets(tasks, use_parallel)

    print("所有图表已生成并保存。")

