import os
import re
import json
import pandas as pd
from tools.data_helper import get_path  # 确保此模块已定义 get_path 函数

# 指定目标目录路径
directory = '../../../output'  # 替换为你的目标目录路径

# 获取目录下的所有文件名
file_names = os.listdir(directory)

# 匹配符合特定命名格式的文件夹
pattern = re.compile(r'20250404_Run_\d+_GHG_1_8C_67_BIO_5')
filtered_file_names = [file for file in file_names if pattern.search(file)]

# 构建 DataFrame 并初始化附加列
df = pd.DataFrame(filtered_file_names, columns=['FileName'])
df['Weight'] = df['FileName'].apply(lambda x: x.split('_')[-1])
df['Transition emission'] = None
df['GHG oversheet'] = None

# 定义要合计的列名
cols_to_sum = [
    'Livestock natural to modified',
    # 'Livestock natural to unallocated natural',
    # 'Unallocated natural to livestock natural',
    'Unallocated natural to modified'
]

# 遍历所有匹配文件，读取 JSON，计算指标并写入 df
for file_name in filtered_file_names:
    try:
        # 获取 JSON 路径
        path = os.path.join(get_path(file_name), 'DATA_REPORT', 'data', 'GHG_2_individual_emission_Mt.json')

        # 读取 JSON 数据
        with open(path, "r") as f:
            data = json.load(f)

        # 构建 DataFrame
        data_dict = {}
        for item in data:
            category = item["name"]
            for year, value in item["data"]:
                if year not in data_dict:
                    data_dict[year] = {}
                data_dict[year][category] = value

        df_temp = pd.DataFrame.from_dict(data_dict, orient="index").sort_index()
        df_temp.index.name = "Year"

        # 计算指标
        total_2050 = df_temp.loc[2050, cols_to_sum].sum()
        oversheet = df_temp.loc[2050, 'Net emissions'] - df_temp.loc[2050, 'GHG emissions limit']

        # 写入 df 对应行
        row_index = df[df['FileName'] == file_name].index[0]
        df.at[row_index, 'Transition emission'] = total_2050
        df.at[row_index, 'GHG oversheet'] = oversheet

    except Exception as e:
        print(f"⚠️ 处理 {file_name} 时出错: {e}")

# 查看最终结果（可选）
print(df)

# 如果需要保存
# df.to_excel("summary_emissions.xlsx", index=False)
df = pd.read_excel("summary_emissions.xlsx")
import matplotlib.pyplot as plt
import pandas as pd

# 确保使用的是包含 'Transition emission' 的 df
df["Transition emission"] = pd.to_numeric(df["Transition emission"], errors='coerce')
df["GHG oversheet"] = pd.to_numeric(df["GHG oversheet"], errors='coerce')
df["Weight"] = pd.to_numeric(df["Weight"], errors='coerce')

# 按 Weight 排序
df_sorted = df.sort_values(by="Weight")

# 图 1：Transition emission
plt.figure(figsize=(8, 5))
plt.plot(df_sorted["Weight"], df_sorted["Transition emission"], marker='o')
plt.xlabel("Weight")
plt.ylabel("Transition emission (Mt CO2e)")
plt.title("Transition Emission vs Weight")
plt.ylim(bottom=0)
plt.grid(True)
plt.tight_layout()
plt.show()

# 图 2：GHG oversheet
plt.figure(figsize=(8, 5))
plt.plot(df_sorted["Weight"], df_sorted["GHG oversheet"], marker='o', color='orange')
plt.xlabel("Weight")
plt.ylabel("GHG oversheet (Mt CO2e)")
plt.ylim(-0.08,0)
plt.title("GHG Oversheet vs Weight")
plt.grid(True)
plt.tight_layout()
plt.show()