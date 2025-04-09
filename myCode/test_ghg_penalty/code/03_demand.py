import pandas as pd
import os
import matplotlib.pyplot as plt
from tools import get_path

from joblib import Parallel, delayed
import pandas as pd
import os

def process_folder(folder, result_file):
    """
    处理单个文件夹，计算偏差并写入结果文件。

    参数:
    - folder (str): 文件夹名称。
    - result_file (str): 输出 Excel 文件路径。
    """
    print(f"Processing folder: {folder}")
    path = get_path(folder)  # 获取路径
    df_deviation = calculate_deviation(path, folder)  # 计算偏差
    sheet_name = '_'.join(f"{folder}_demand".split('_')[2:])[:31]  # 生成工作表名称
    return sheet_name, df_deviation

def write_to_excel(output_result_file, results):
    """
    将计算结果写入 Excel 文件。

    参数:
    - output_result_file (str): 输出 Excel 文件路径。
    - results (list): 每个文件夹的计算结果（工作表名称和 DataFrame 的列表）。
    """
    with pd.ExcelWriter(output_result_file, engine="openpyxl") as writer:
        for sheet_name, df in results:
            df.to_excel(writer, sheet_name=sheet_name, index=False)

def parallel_process_and_write(folders, output_result_file, n_jobs=4):
    """
    并行处理多个文件夹并写入结果到 Excel 文件。

    参数:
    - folders (list): 要处理的文件夹列表。
    - output_result_file (str): 输出 Excel 文件路径。
    - n_jobs (int): 并行任务数量。
    """
    # 并行计算
    results = Parallel(n_jobs=n_jobs)(
        delayed(process_folder)(folder, output_result_file) for folder in folders
    )
    # 写入结果
    write_to_excel(output_result_file, results)

def get_folders(file_path):
    """Get folders"""
    df = pd.read_csv(file_path)
    all_columns = df.columns.tolist()
    filtered_columns = [col for col in all_columns if col not in ["Name", "Default_run"]]
    return filtered_columns

def calculate_deviation(path, folder):
    """Calculate and save GHG and Demand deviation data"""
    all_data = []
    for year in range(2011, 2051):
        df_demand = pd.read_csv(os.path.join(path, f"out_{year}", f"quantity_comparison_{year}.csv"))
        demand_difference = abs(df_demand["Abs_diff (tonnes, KL)"])
        demand_commodity = df_demand['Commodity']
        demand_difference_ratio = abs(100-df_demand['Prop_diff (%)'])
        df_year = pd.DataFrame({
            "Year": [year] * len(demand_commodity),
            "Commodity": demand_commodity,
            "Demand_Difference": demand_difference,
            "Demand_Difference_Ratio": demand_difference_ratio
        })
        all_data.append(df_year)

    result_df = pd.concat(all_data, ignore_index=True)
    return result_df

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

def plot_cumulative_bar_chart(input_excel_file, output_folder, column_name):
    """
    根据指定列绘制每年堆叠的柱状图并保存为 PNG 文件。

    参数:
    - input_excel_file (str): 输入 Excel 文件路径。
    - output_folder (str): 图表保存文件夹路径。
    - column_name (str): 用于绘图的列名。
    """
    os.makedirs(output_folder, exist_ok=True)  # 确保输出文件夹存在

    # 读取 Excel 文件
    data = pd.ExcelFile(input_excel_file)

    for sheet_name in data.sheet_names:
        # 读取工作表
        df = data.parse(sheet_name)

        # 验证是否包含所需列
        required_columns = {'Year', 'Commodity', column_name}
        if not required_columns.issubset(df.columns):
            print(f"Sheet {sheet_name} does not have the required columns {required_columns}. Skipping.")
            continue

        # 按年份和商品分组，计算指定列的总和（同一年内堆叠）
        pivot_data = df.pivot_table(index='Year', columns='Commodity', values=column_name, aggfunc='sum', fill_value=0)

        # 创建堆叠柱状图（只堆叠同一年的 Commodity）
        ax = pivot_data.plot(kind='bar', stacked=True, figsize=(10, 6), colormap='tab20')

        # 设置图表标题和轴标签
        plt.title(f"{column_name} by Year ({sheet_name})")
        plt.xlabel("Year")
        plt.ylabel(column_name)
        plt.xticks(rotation=45, ha='right')

        # 调整图例位置到右侧
        ax.legend(title='Commodity', loc='center left', bbox_to_anchor=(1.0, 0.5))

        # 调整布局
        plt.tight_layout()

        # 保存图表
        chart_image_path = os.path.join(output_folder, f"{sheet_name}_{column_name}_chart.png")
        plt.savefig(chart_image_path, bbox_inches='tight')  # 确保图例不被截断
        plt.close()

        print(f"Chart for {column_name} in sheet {sheet_name} saved to {chart_image_path}")


file_path = "../../tasks_run/Custom_runs/setting_template_windows_10.csv"
folders = get_folders(file_path)

# i=0
# folders = match_files_in_folder(f"20241206_{i}_w90_GHG_1_8C_67_BIO_0")
output_result_file = f"../Result/output_demand.xlsx"
parallel_process_and_write(folders, output_result_file, n_jobs=4)

output_folder = "../Figure"

# 绘制 Demand_Difference 的图表
plot_cumulative_bar_chart(output_result_file, output_folder, "Demand_Difference")
# 绘制 Demand_Difference_Ratio 的图表
plot_cumulative_bar_chart(output_result_file, output_folder, "Demand_Difference_Ratio")