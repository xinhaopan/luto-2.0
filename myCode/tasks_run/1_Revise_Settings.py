import pandas as pd
import os

output = "settings_template"  # 输出文件名
GHG_Name = {
            "1.8C (67%) excl. avoided emis": "1_8C_67",
            "1.5C (50%) excl. avoided emis": "1_5C_50",
            "1.5C (67%) excl. avoided emis": "1_5C_67"
            }
BIO_Name = {
    "{2010: 0, 2030: 0, 2050: 0, 2100: 0}": "BIO_0",
    "{2010: 0, 2030: 0.3, 2050: 0.3, 2100: 0.3}": "BIO_3",
    "{2010: 0, 2030: 0.3, 2050: 0.5, 2100: 0.5}": "BIO_5"
    }

# 读取数据
df = pd.read_csv("Custom_runs/settings_template.csv")  # 原始数据
df_revise = pd.read_excel("Custom_runs/Revise_settings_template.xlsx")  # 修订数据

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

# 遍历列名，根据字典映射生成新列名
new_column_names = []
for col in new_df.columns[1:]:  # 跳过第一列（通常为ID或索引列）
    ghg_value = GHG_Name.get(ghg_limits_field[col], "Unknown_GHG")  # 映射 GHG 值
    bio_value = BIO_Name.get(biodiv_gbf_target_2_dict[col], "Unknown_BIO")  # 映射 BIO 值
    new_name = f"{col}_GHG_{ghg_value}_BIO_{bio_value}".replace(".", "_")  # 拼接新列名并替换点号
    new_column_names.append(new_name)

# 检查文件是否存在
output_path = f"Custom_runs/{output}.csv"
if os.path.exists(output_path):
    print(f"文件 '{output_path}' 已经存在，未覆盖保存。")
else:
    # 保存结果
    new_df.to_csv(output_path, index=False)
    print(f"新的DataFrame已生成，并保存为 '{output_path}'")
