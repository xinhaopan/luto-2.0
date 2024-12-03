import pandas as pd
import os
from tools.helpers import create_grid_search_template, create_task_runs, create_settings_template
from tools import calculate_total_cost

import numpy as np
import datetime


def recommend_resources(df):
    """
    Recommend suitable CPU based on MEM or suitable MEM based on CPU.

    Args:
        df (pd.DataFrame): Input DataFrame with rows containing CPU_PER_TASK and MEM.

    Returns:
        None: Prints the recommendations for each task.
    """

    # Ensure columns are strings
    df.columns = df.columns.astype(str)

    # Remove unnamed columns if present
    df = df.loc[:, ~df.columns.str.startswith('Unnamed')]

    # Extract relevant rows for CPU and MEM
    cpu_row = df[df['Name'] == 'CPU_PER_TASK']
    mem_row = df[df['Name'] == 'MEM']

    # Extract values as Series
    cpu_values = cpu_row.iloc[0, 1:].astype(float)  # CPU per task
    mem_values = mem_row.iloc[0, 1:].astype(float)  # Memory in GB

    # Recommend CPU based on MEM
    recommended_cpus = np.ceil(mem_values / 4)  # 每核 4GB 内存，向上取整

    # Recommend MEM based on CPU
    recommended_mem = cpu_values * 4  # 每核 4GB

    # Print the recommendations
    for task, cpu, mem, rec_cpu, rec_mem in zip(cpu_values.index, cpu_values, mem_values, recommended_cpus, recommended_mem):
        print(f"Task {task}:")
        print(f"  - Current CPU: {cpu}, Recommended MEM: {rec_mem} GB")
        print(f"  - Current MEM: {mem} GB, Recommended MEM for CPU {rec_cpu} GB")
        break

output = "setting_template_windows"  # 输出文件名
current_time = datetime.datetime.now().strftime("%Y%m%d")
# Create a template for the custom settings, and then create the custom settings
# create_settings_template('Custom_runs')

GHG_Name = {
            "1.8C (67%) excl. avoided emis": "GHG_1_8C_67",
            "1.5C (50%) excl. avoided emis": "GHG_1_5C_50",
            "1.5C (67%) excl. avoided emis": "GHG_1_5C_67"
            }
BIO_Name = {
    "{2010: 0, 2030: 0, 2050: 0, 2100: 0}": "BIO_0",
    "{2010: 0, 2030: 0.3, 2050: 0.3, 2100: 0.3}": "BIO_3",
    "{2010: 0, 2030: 0.3, 2050: 0.5, 2100: 0.5}": "BIO_5"
    }

# 读取数据
df = pd.read_csv("Custom_runs/settings_template.csv")  # 原始数据
df_revise = pd.read_excel("Custom_runs/Revise_settings_template.xlsx",sheet_name="using")  # 修订数据
df_revise.columns = df_revise.columns.astype(str)
df_revise = df_revise.loc[:, ~df_revise.columns.str.startswith('Unnamed')]

# 创建一个新的DataFrame，第一列复制df的第一列
new_df = pd.DataFrame(df.iloc[:, :2])  # 前两列为df的前两列
new_df.columns = df.columns[:2]  # 保持列名一致

# 添加df_revise的列名，初始化为Default_run的值
for col_name in df_revise.columns[1:]:  # 跳过df_revise的第一列
    new_df[col_name] = df["Default_run"]

# 用df_revise每行每列的值，替换new_df相应的值
for idx, row in df_revise.iterrows():
    match_value = row[df_revise.columns[0]]  # 第一列的值作为匹配条件
    matching_condition = new_df.iloc[:, 0] == match_value  # 匹配new_df的第一列
    if matching_condition.any():
        for col_name in df_revise.columns[1:]:  # 遍历df_revise的其他列
            new_df.loc[matching_condition, col_name] = row[col_name]  # 替换对应列的值

# 提取 GHG_LIMITS_FIELD 和 BIODIV_GBF_TARGET_2_DICT 两行
ghg_limits_field = new_df.iloc[new_df[new_df.iloc[:, 0] == "GHG_LIMITS_FIELD"].index[0]]
biodiv_gbf_target_2_dict = new_df.iloc[new_df[new_df.iloc[:, 0] == "BIODIV_GBF_TARGET_2_DICT"].index[0]]
ghg_penalty_field = new_df.iloc[new_df[new_df.iloc[:, 0] == "GHG_PENALTY"].index[0]]

# 遍历列名，根据字典映射生成新列名
new_column_names = []
# 确保第一列用作定位
name_column = df_revise.columns[0]  # 第一列的列名（如 'Name'）

# 检查 Name1 是否存在于第一列中
if "Name1" not in df_revise[name_column].values:
    raise ValueError(f"Name1 不存在于列 {name_column} 中，请检查数据！")

# 获取 Name1 行
name1_row = df_revise[df_revise[name_column] == "Name1"].iloc[0]  # 获取 Name1 行数据

# 初始化新列名列表
new_column_names = []

for col in new_df.columns[2:]:  # 跳过前两列
    ghg_value = GHG_Name.get(ghg_limits_field[col], "Unknown_GHG")  # 映射 GHG 值
    bio_value = BIO_Name.get(biodiv_gbf_target_2_dict[col], "Unknown_BIO")  # 映射 BIO 值

    # 从 Name1 行中获取对应列的值
    name1_value = name1_row.get(col, "")  # 获取列对应的值，如果不存在则返回空字符串

    # 确保 Name1 行中有值
    if pd.isna(name1_value) or name1_value == "":
        raise ValueError(f"Name1 行在列 {col} 中没有有效值，请检查数据！")

    # 根据条件拼接新列名
    col = str(col).split('.')[0]
    new_name = f"{current_time}_{name1_value}_{col}_{ghg_value}_{bio_value}".replace(".", "_")
    new_column_names.append(new_name)

new_df.columns = new_df.columns[:2].tolist() + new_column_names  # 更新列名
# 检查文件是否存在
output_path = f"Custom_runs/{output}.csv"
if os.path.exists(output_path):
    print(f"文件 '{output_path}' 已经存在，未覆盖保存。")
else:
    # 保存结果
    new_df.to_csv(output_path, index=False)
    print(f"新的DataFrame已生成，并保存为 '{output_path}'")

total_cost = calculate_total_cost(df_revise)
print(f"Job Cost: {total_cost}k")
recommend_resources(df_revise)

