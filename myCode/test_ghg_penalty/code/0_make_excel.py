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
            # else:
            #     print(f"Error: Invalid data in folder '{folder}'")
        # else:
        #     # 条件检查失败时输出调试信息
        #     if not folder_path:
        #         print(f"Folder skipped: get_path returned None for folder '{folder}'")
        #     elif not os.path.exists(os.path.join(folder_path, "model_run_settings.txt")):
        #         print(f"File missing: 'model_run_settings.txt' not found in folder '{folder}'")
        #     elif not os.path.exists(os.path.join(folder_path, "out_2050/GHG_emissions_2050.csv")):
        #         print(f"File missing: 'GHG_emissions_2050.csv' not found in folder '{folder}'")
        #     elif not check_ghg_settings(os.path.join(folder_path, "model_run_settings.txt")):
        #         print(f"File check failed: 'model_run_settings.txt' in folder '{folder}' does not meet conditions")

    return results

def save_results_to_excel(results, output_file):
    """将结果保存到 Excel 文件"""
    df = pd.DataFrame(results)
    df.to_excel(output_file, index=False, sheet_name="Results")


# 主程序
if __name__ == "__main__":
    # 配置路径和筛选条件
    base_path = "../../../output"
    strings = ["1_8C_67", "1_5C_50", "1_5C_67"]
    for string in strings:
        output_file = f"../results_{string}.xlsx"

        # 获取筛选后的文件夹
        filtered_folders = get_folders_containing_string(base_path, string)

        # 处理文件夹并提取数据
        results = process_folders(base_path, filtered_folders)

        # 保存结果到 Excel
        save_results_to_excel(results, output_file)
        print(f"Results have been saved to {output_file}")

