import os
import pandas as pd
from tools import get_path


def get_folders_containing_string(path, string):
    """筛选出路径下包含特定字符串的文件夹"""
    return [f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f)) and string in f]


def check_ghg_settings(file_path):
    """
    检查 model_run_settings.txt 文件是否包含 GHG_CONSTRAINT_TYPE: soft，
    如果包含，则返回是否存在 GHG_PENALTY。
    """
    # 读取文件内容
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # 筛选包含关键字的行
    constraint_lines = [line.strip() for line in lines if "GHG_CONSTRAINT_TYPE:" in line]
    penalty_lines = [line.strip() for line in lines if "GHG_PENALTY:" in line]

    # 判断是否有 GHG_CONSTRAINT_TYPE: soft
    has_ghg_constraint = any("soft" in line for line in constraint_lines)

    # 如果包含 GHG_CONSTRAINT_TYPE: soft，则返回是否有 GHG_PENALTY
    if has_ghg_constraint:
        return len(penalty_lines) > 0
    else:
        return False  # 不包含 GHG_CONSTRAINT_TYPE: soft，直接返回 False



def get_ghg_penalty(file_path):
    """从 model_run_settings.txt 获取 GHG_PENALTY 的值"""
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # 筛选包含 GHG_PENALTY 的行并提取值
    penalty_lines = [line.strip() for line in lines if "GHG_PENALTY:" in line]
    if penalty_lines:
        return float(penalty_lines[0].split(":")[1].strip())
    else:
        raise ValueError(f"'GHG_PENALTY' not found in file: {file_path}")


def get_ghg_emissions(csv_file):
    """从 GHG_emissions_2050.csv 获取 GHG_EMISSIONS_TCO2e 的值"""
    df = pd.read_csv(csv_file)

    # 筛选出 Variable 为 GHG_EMISSIONS_TCO2e 的行
    row = df[df["Variable"] == "GHG_EMISSIONS_TCO2e"]
    if row.empty:
        raise ValueError(f"'GHG_EMISSIONS_TCO2e' not found in file: {csv_file}")
    return row["Emissions (t CO2e)"].values[0] / 1e6


def process_folders(base_path, folders):
    """处理文件夹，提取 GHG_PENALTY 和 GHG_EMISSIONS_TCO2e 的值"""
    results = []
    for folder in folders:
        folder_path = get_path(folder)  # 获取路径

        # 条件检查
        if (
            folder_path  # 路径有效
            and os.path.exists(os.path.join(folder_path, "model_run_settings.txt"))  # settings 文件存在
            and os.path.exists(os.path.join(folder_path, "out_2050/GHG_emissions_2050.csv"))  # emissions 文件存在
            and check_ghg_settings(os.path.join(folder_path, "model_run_settings.txt"))  # settings 文件满足条件
        ):
            # 提取文件路径
            settings_file = os.path.join(folder_path, "model_run_settings.txt")
            emissions_file = os.path.join(folder_path, "out_2050/GHG_emissions_2050.csv")

            # 提取 GHG_PENALTY 和 GHG_EMISSIONS_TCO2e 的值
            ghg_penalty = get_ghg_penalty(settings_file)
            ghg_emissions = get_ghg_emissions(emissions_file)

            # 检查值是否有效
            if ghg_penalty is not None and ghg_emissions is not None:
                results.append({
                    "folder": folder,
                    "GHG_PENALTY": ghg_penalty,
                    "GHG_EMISSIONS_TCO2e": ghg_emissions
                })
            else:
                print(f"Error: Invalid data in folder '{folder}'")
        else:
            # 条件检查失败时输出调试信息
            if not folder_path:
                print(f"Folder skipped: get_path returned None for folder '{folder}'")
            elif not os.path.exists(os.path.join(folder_path, "model_run_settings.txt")):
                print(f"File missing: 'model_run_settings.txt' not found in folder '{folder}'")
            elif not os.path.exists(os.path.join(folder_path, "out_2050/GHG_emissions_2050.csv")):
                print(f"File missing: 'GHG_emissions_2050.csv' not found in folder '{folder}'")
            elif not check_ghg_settings(os.path.join(folder_path, "model_run_settings.txt")):
                print(f"File check failed: 'model_run_settings.txt' in folder '{folder}' does not meet conditions")

    return pd.DataFrame(results)

def save_results_to_excel(results, output_file):
    """将结果保存到 Excel 文件"""
    df = pd.DataFrame(results)
    df.to_excel(output_file, index=False, sheet_name="Results")


# 主程序
if __name__ == "__main__":
    output_file = "../output.xlsx"
    # 配置路径和筛选条件
    base_path = "../../../output"
    strings = [
        "GHG_1_8C_67_BIO_5", "GHG_1_5C_50_BIO_5", "GHG_1_5C_67_BIO_5",
        "GHG_1_8C_67_BIO_3", "GHG_1_5C_50_BIO_3", "GHG_1_5C_67_BIO_3"
    ]

    # 打开 ExcelWriter
    with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
        for string in strings:
            # 获取筛选后的文件夹
            filtered_folders = get_folders_containing_string(base_path, string)

            # 处理文件夹并提取数据
            results = process_folders(base_path, filtered_folders)

            # 排序结果
            df = results.sort_values(by=results.columns[1], ascending=True)

            # 动态命名 sheet 名称
            sheet_name = string[:31]  # Excel 限制 sheet 名长度最多 31 个字符

            # 将当前 DataFrame 写入 Excel
            df.to_excel(writer, sheet_name=sheet_name, index=False)

            print(f"DataFrames have been saved to {sheet_name}.")

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

# 读取 Excel 文件
output_file = "../output.xlsx"
sheets = pd.ExcelFile(output_file).sheet_names  # 获取所有 sheet 名称

# 创建一个 2x3 的图形布局
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()  # 展平 axes 数组以便迭代

# 遍历每个 sheet 并绘制图形
for i, sheet in enumerate(sheets):
    if i >= len(axes):  # 如果超过 6 个 sheet，跳过
        break

    # 读取当前 sheet 数据
    df = pd.read_excel(output_file, sheet_name=sheet)

    # 绘制图形
    ax = axes[i]
    ax.plot(df['GHG_PENALTY'], df['GHG_EMISSIONS_TCO2e'], label=sheet, marker='o')

    # 设置图表属性
    ax.set_xlim(0, 0.02)  # 横坐标范围
    ax.xaxis.set_major_locator(MultipleLocator(0.005))  # 设置横轴刻度间隔为 0.005
    ax.set_ylim(-400, 0)  # 纵坐标范围
    ax.set_title(sheet)  # 标题为 sheet 名
    ax.set_xlabel('GHG_PENALTY')
    ax.set_ylabel('GHG_EMISSIONS_TCO2e')
    ax.legend()

# 调整布局
plt.tight_layout()
plt.show()
