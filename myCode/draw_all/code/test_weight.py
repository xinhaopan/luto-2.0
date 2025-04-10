import os
import re
import json
import pandas as pd
from tools.data_helper import get_path  # 确保此模块已定义 get_path 函数

def extract_weight(file_name):
    # 匹配并捕获最后的数字段
    print(file_name)
    match = re.search(r'(\d+_)*(\d+)$', file_name)
    if match:
        # 提取最后一部分，并将下划线替换为小数点
        weight = match.group(0).replace('_', '.')
        return float(weight) if '.' in weight else int(weight)
    return None


# 指定目标目录路径
directory = '../../../output'  # 替换为你的目标目录路径

# 获取目录下的所有文件名
file_names = os.listdir(directory)

# 匹配符合特定命名格式的文件夹
pattern = re.compile(r'^BIODIV_Run_')
filtered_file_names = [file for file in file_names if pattern.search(file)]

# 构建 DataFrame 并初始化附加列
df = pd.DataFrame(filtered_file_names, columns=['FileName'])
df['Weight'] = None
df['Area Weighted Score (Mha)'] = None
df['Profit (billion AU$)'] = None

# 定义要合计的列名
cols_to_sum = [
    'Area Weighted Score (ha)',
]
Weight = 'BIODIV_WEIGHT'

# 遍历所有匹配文件，读取 JSON，计算指标并写入 df
for file_name in filtered_file_names:
    try:
        # bio
        path = os.path.join(get_path(file_name), 'out_2050', 'biodiversity_overall_priority_scores_2050.csv')
        df_temp = pd.read_csv(path)
        column_sum = df_temp[cols_to_sum].sum()
        row_index = df[df['FileName'] == file_name].index[0]
        df.at[row_index, 'Area Weighted Score (Mkm2)'] = column_sum.values[0]/1e6 # 转换为 Mha
        # profit
        profit_path = os.path.join(get_path(file_name), 'DATA_REPORT', 'data','economics_0_rev_cost_all_wide.json')
        with open(profit_path , "r") as f:
            data = json.load(f)

        # 遍历 JSON 数据，找到 "Profit" 的对象
        for item in data:
            if item.get("name") == "Profit":
                # 遍历数据列表，查找年份为2050的项
                for year, value in item.get("data", []):
                    if int(year) == 2050:
                        df.at[row_index, 'Profit (billion AU$)'] = value
                        break
                break

        # weight
        weight_path = os.path.join(directory,file_name, 'luto','settings.py')
        with open(weight_path, 'r') as file:
            # 遍历文件逐行读取
            for line in file:
                # 匹配 "BIODIV_WEIGHT=数字" 的行
                match = re.search(fr'{Weight}\s*=\s*([\d.]+)', line.split('#')[0].strip())
                print(match)
                if match:
                    # 提取并返回权重值，转换为 float
                    df.at[row_index, 'Weight'] = float(match.group(1))
                    break

    except Exception as e:
        print(f"⚠️ 处理 {file_name} 时出错: {e}")

# 查看最终结果（可选）
df = df.sort_values(by="Weight", ascending=True)
print(df)

# 如果需要保存
df.to_excel("summary_bio.xlsx", index=False)
df = pd.read_excel("summary_bio.xlsx")
import matplotlib.pyplot as plt
import pandas as pd

# 确保使用的是包含 'Transition emission' 的 df
df['Area Weighted Score (Mkm2)'] = pd.to_numeric(df['Area Weighted Score (Mkm2)'], errors='coerce')
df["Weight"] = pd.to_numeric(df["Weight"], errors='coerce')

# 按 Weight 排序
df_sorted = df.sort_values(by="Weight")

xmin,x_max = 0,100
plt.figure(figsize=(8, 5))
plt.plot(df_sorted["Weight"], df_sorted['Area Weighted Score (Mkm2)'], marker='o')
plt.xlabel("Weight")
plt.ylabel('Area Weighted Score (Mkm2)')
plt.title('Area Weighted Score (Mkm2)')
# plt.ylim(bottom=0)
plt.xlim(xmin,x_max)
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
plt.plot(df_sorted["Weight"], df_sorted['Profit (billion AU$)'], marker='o')
plt.xlabel("Weight")
plt.ylabel('Profit (billion AU$)')
plt.title('Profit (billion AU$)')
# plt.ylim(bottom=0)
plt.xlim(xmin,x_max)
plt.grid(True)
plt.tight_layout()
plt.show()
