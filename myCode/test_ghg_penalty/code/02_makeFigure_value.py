import os
import pandas as pd
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

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
    fig.legend(loc="upper right", bbox_to_anchor=(0.85, 0.85))
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

# 读取Excel文件
file_path = '../result/output_log_coeff.xlsx'
output_path = '../Figure'
excel_data = pd.ExcelFile(file_path)

# 指定需要处理的列
columns = ['GHG Deviation']

# 计算全局最小值和最大值
use_global_y_range = False  # 设置是否启用统一的Y轴范围
global_min_max = calculate_global_min_max(excel_data, columns) if use_global_y_range else None

# 遍历所有Sheet并并行绘制图表
tasks = []
for sheet_name in excel_data.sheet_names:
    df = excel_data.parse(sheet_name)

    # 确保数据列存在
    required_columns = ['Year'] + columns
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        print(f"Sheet {sheet_name} 缺少以下列：{missing_columns}，跳过...")
        continue

    # 动态计算当前Sheet的最小值和最大值
    local_min_max = get_local_min_max(df, columns) if not use_global_y_range else None

    # 获取Y轴范围
    y_ranges = global_min_max if use_global_y_range else local_min_max

    # 添加任务
    tasks.append((sheet_name, df, y_ranges, output_path))

# 并行处理任务
Parallel(n_jobs=-1)(
    delayed(process_sheet)(sheet_name, df, y_ranges, output_path) for sheet_name, df, y_ranges, output_path in tasks
)

print("所有图表已生成并保存。")
