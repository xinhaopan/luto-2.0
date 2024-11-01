import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams

from helper import *



# 1. 读取 GHG_targets.xlsx 文件
font_num = 12

# 读取 Excel 文件
df = pd.read_excel('../../input/GHG_targets.xlsx', index_col=0)

# 选择 2010 到 2050 年的数据
df_filtered = df.loc[2010:2050,
              ['1.5C (67%) excl. avoided emis', '1.5C (50%) excl. avoided emis', '1.8C (67%) excl. avoided emis']]

# 将所有数据单位转换为 million (除以 1,000,000)
df_filtered = df_filtered / 1e6

# 2. 原始数据处理（来自之前生成的 results）
INPUT_NAMEs = ['ON_MAXPROFIT_GHG_15C_67_R5', 'ON_MAXPROFIT_GHG_15C_50_R5', 'ON_MAXPROFIT_GHG_18C_67_R5']
column_names = ['1.5C (67%)', '1.5C (50%)', '1.8C (67%)']

# 初始化主 DataFrame，包含年份列
results = pd.DataFrame({'Year': range(2010, 2051)})

# 循环每个 INPUT_NAME 来处理并添加到 results 中
for INPUT_NAME, column_name in zip(INPUT_NAMEs, column_names):
    base_path = get_path(INPUT_NAME)

    # 初始化临时 DataFrame，用于存储每个 INPUT_NAME 的结果
    temp_results = pd.DataFrame(columns=['Year', column_name])

    # 存储所有要合并的行
    temp_rows = []

    # 遍历2010到2050年的文件
    for year in range(2010, 2051):
        # 构建每个CSV文件的路径
        file_path = os.path.join(base_path, f'out_{year}', f'GHG_emissions_{year}.csv')

        # 读取CSV文件
        df = pd.read_csv(file_path)

        # 提取 "Variable" 行和 "Emissions (t CO2e)" 列的值
        emissions_value = df.loc[df['Variable'] == 'GHG_EMISSIONS_TCO2e', 'Emissions (t CO2e)'].values[0] / 1e6

        # 创建字典并添加到temp_rows列表
        temp_rows.append({'Year': year, column_name: emissions_value})

    # 将所有行转成DataFrame并使用concat进行合并
    temp_results = pd.concat([temp_results, pd.DataFrame(temp_rows)], ignore_index=True)

    # 将当前 temp_results 按年份合并到 results DataFrame 中
    results = pd.merge(results, temp_results, on='Year', how='left')
results.set_index('Year', inplace=True)

# 3. 合并两组数据到一个表
combined_df = pd.concat([results, df_filtered], axis=1)

# 设置字体为 Arial
rcParams['font.family'] = 'Arial'
rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 假设已经有 `combined_df` 数据
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))  # 创建1行2列的子图

colors = ['#E74C3C', '#3498DB', '#2ECC71']  # 使用红色、蓝色、绿色
font_size = 16
# 图1：先绘制 Target 数据
ax1.plot(combined_df.index, combined_df['1.5C (67%) excl. avoided emis'], linestyle='--',marker='o', color=colors[0], linewidth=2.5, label='1.5°C (67%) (Target)')
ax1.plot(combined_df.index, combined_df['1.5C (50%) excl. avoided emis'], linestyle='--',marker='s', color=colors[1], linewidth=2.5, label='1.5°C (50%) (Target)')
ax1.plot(combined_df.index, combined_df['1.8C (67%) excl. avoided emis'], linestyle='--',marker='^', color=colors[2], linewidth=2.5, label='1.8°C (67%) (Target)')

ax1.text(0.02, 0.95, '(a)', transform=ax1.transAxes, fontsize=font_size, verticalalignment='top')

# 保持您指定的 X 和 Y 轴属性
ax1.set_xticks(range(2010, 2051))
ax1.set_xticklabels(range(2010, 2051), rotation=90, fontsize=font_size)
ax1.set_xlim(2010, 2050)
ax1.set_ylim(-300, 100)
ax1.set_yticks(range(-300, 101, 100))
ax1.set_yticklabels(range(-300, 101, 100), fontsize=font_size)
ax1.tick_params(axis='y', direction='in')
ax1.tick_params(axis='x', direction='in')
ax1.set_xlabel('Year', fontsize=font_size)
ax1.set_ylabel('Greenhouse gas (Million tons CO2e)', fontsize=font_size)

# 添加图例和标题
ax1.legend(loc='upper center', bbox_to_anchor=(0.25, 0.25), ncol=1, frameon=False, fontsize=font_size)

# 图2：再绘制 Simulation 数据
ax2.plot(combined_df.index, combined_df['1.5C (67%)'], marker='o', color=colors[0], linewidth=2.5, label='1.5°C (67%) (Simulation)')
ax2.plot(combined_df.index, combined_df['1.5C (50%)'], marker='s', color=colors[1], linewidth=2.5, label='1.5°C (50%) (Simulation)')
ax2.plot(combined_df.index, combined_df['1.8C (67%)'], marker='^', color=colors[2], linewidth=2.5, label='1.8°C (67%) (Simulation)')

ax2.text(0.02, 0.95, '(b)', transform=ax2.transAxes, fontsize=font_size, verticalalignment='top')

# 保持您指定的 X 和 Y 轴属性
ax2.set_xticks(range(2010, 2051))
ax2.set_xticklabels(range(2010, 2051), rotation=90, fontsize=font_size)
ax2.set_xlim(2010, 2050)
ax2.set_ylim(-300, 100)
ax2.set_yticks(range(-300, 101, 100))
ax2.set_yticklabels(range(-300, 101, 100), fontsize=font_size)
ax2.tick_params(axis='y', direction='in')
ax2.tick_params(axis='x', direction='in')
ax2.set_xlabel('Year', fontsize=font_size)
ax2.set_ylabel('Greenhouse gas (Million tons CO2e)', fontsize=font_size)

# 添加图例和标题
ax2.legend(loc='upper center', bbox_to_anchor=(0.27, 0.25), ncol=1, frameon=False, fontsize=font_size)

# 调整子图之间的间距
plt.tight_layout()

# 保存图表
plt.savefig('03_GHG.png', dpi=300)

# 显示图表
plt.show()
