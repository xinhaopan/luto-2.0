import os
import shutil
import pandas as pd

# 路径定义
summary_file = '../result/summary.xlsx'
output_dir   = '../output'
target_dir   = '../suc_output'

# 确保目标目录存在
os.makedirs(target_dir, exist_ok=True)

# 读取 No Issues sheet
df_good = pd.read_excel(summary_file, sheet_name='No Issues')

# 遍历 path_name
for path_name in df_good['path_name']:
    for ext in ['.xlsx', '.png']:
        src = os.path.join(output_dir, f'{path_name}{ext}')
        dst = os.path.join(target_dir, f'{path_name}{ext}')
        if os.path.exists(src):
            shutil.copy2(src, dst)
        else:
            print(f"⚠️ 文件不存在，跳过: {src}")

print(f"✅ 所有成功路径的文件已复制到 {target_dir}")

import os
import pandas as pd

output_dir = '../output'
results = []

# 遍历所有 Excel 文件
for path_name in df_good['path_name']:

    bio_weight_str = path_name[-3:].replace('_', '.')
    try:
        bio_weight = float(bio_weight_str)
    except ValueError:
        print(f"⚠️ 无法解析 BIO_WEIGHT: {path_name}")
        continue

    df = pd.read_excel(os.path.join(output_dir, f'{path_name}.xlsx'))
    df2050 = df[df['Year'] == 2050]

    try:
        econ_val = df2050[df2050['Indicator'] == 'Economy Total Value (Billion AUD)']['Value'].values[0]
        bio_val = df2050[df2050['Indicator'] == 'Biodiversity Total Priority Score (M)']['Value'].values[0]
        results.append({
            'BIO_WEIGHT': bio_weight,
            'Economy Total Value (Billion AUD)': econ_val,
            'Biodiversity Total Priority Score (M)': bio_val
        })
    except IndexError:
        print(f"⚠️ 缺少数据: {path_name}")
        continue

# 创建并保存 DataFrame
df_result = pd.DataFrame(results)
df_result = df_result.sort_values(by='BIO_WEIGHT')  # 按 BIO_WEIGHT 升序排序（可选）

# 保存结果
output_path = '../result/econ_vs_bio.xlsx'
df_result.to_excel(output_path, index=False)

print(f"✅ 结果已保存到 {output_path}")

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
df = pd.read_excel('../result/econ_vs_bio.xlsx')
# 设置风格
sns.set(style="whitegrid", font="Arial", font_scale=1)

# 创建画布和主坐标轴
fig, ax1 = plt.subplots(figsize=(8, 5))

# 画第一个指标：Economy
color1 = "tab:blue"
ax1.set_xlabel("BIO_WEIGHT")
ax1.set_ylabel("Economy Value (Billion AUD)", color=color1)
ax1.plot(df['BIO_WEIGHT'], df['Economy Total Value (Billion AUD)'], color=color1, marker='o', label="Economy")
ax1.tick_params(axis='y', labelcolor=color1)

# 创建共享 x 轴但独立 y 轴的次坐标轴
ax2 = ax1.twinx()
color2 = "tab:green"
ax2.set_ylabel("Biodiversity Score (M)", color=color2)
ax2.plot(df['BIO_WEIGHT'], df['Biodiversity Total Priority Score (M)'], color=color2, marker='s', label="Biodiversity")
ax2.tick_params(axis='y', labelcolor=color2)

# 合并图例
lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
ax2.legend(lines_1 + lines_2, labels_1 + labels_2, loc='lower right', fontsize=9)

# 标题 & 图例
fig.suptitle("Economy vs Biodiversity by BIO_WEIGHT", fontsize=12, fontweight='bold')
fig.tight_layout()
plt.show()
plt.savefig('../result/econ_vs_bio.png', dpi=300, bbox_inches='tight')


