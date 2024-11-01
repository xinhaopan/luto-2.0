'''
tagrget and results
'''

from helper import *
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


plt.rcParams['font.family'] = 'Arial'
def merge_dicts_to_df_dict(ag_dict, am_dict, off_land_dict, non_ag_dict, transition_dict):
    # 创建一个新的字典用于存储合并后的 DataFrame
    merged_dict = {}

    # 遍历每个字典中的 key
    for input_name in ag_dict:
        # 初始化一个空的 DataFrame，并以年份为索引
        merged_df = pd.DataFrame(index=ag_dict[input_name].index)

        # 将各字典对应的列合并到 DataFrame
        merged_df['Agricultural landuse'] = ag_dict[input_name]['Total']
        merged_df['Agricultural management'] = am_dict[input_name]['Total']
        merged_df['Off-land'] = off_land_dict[input_name]['Total']
        merged_df['Non-agricultural landuse'] = non_ag_dict[input_name]['Total']
        merged_df['Transition'] = transition_dict[input_name]['Total']

        # 计算 Net emissions 列为五个列的求和
        merged_df['Net emissions'] = merged_df.sum(axis=1)

        # 将合并后的 DataFrame 存储在新的字典中，保持原来的 key
        merged_dict[input_name] = merged_df

    return merged_dict

def plot_stacked_bar_and_line(merged_dict, input_name, font_size=10):
    merged_df = merged_dict[input_name]
    # 确保索引是 int 类型
    merged_df.index = merged_df.index.astype(int)

    # 设置图形大小
    fig, ax = plt.subplots(figsize=(10, 6))

    # 提取需要堆积的列
    categories = ['Agricultural landuse', 'Agricultural management', 'Off-land', 'Non-agricultural landuse',
                  'Transition']

    # 准备数据
    years = merged_df.index
    data = np.array([merged_df[category] for category in categories])

    # 设置颜色列表，使用指定的五个颜色
    color_list = ['#F9C0B7', '#FCD071', '#B4A7D6', '#85C6BE', '#D2E0FB']

    # 分开处理正数和负数
    bars_pos = []  # 用于存储正数的柱状图对象
    bars_neg = []  # 用于存储负数的柱状图对象

    # 绘制正数的堆积柱状图
    pos_data = np.maximum(data, 0)  # 将负数设置为0
    bars_pos.append(ax.bar(years, pos_data[0], label=categories[0], color=color_list[0], width=0.8))
    for i in range(1, len(categories)):
        bars_pos.append(
            ax.bar(years, pos_data[i], bottom=pos_data[:i].sum(axis=0), label=categories[i], color=color_list[i],
                   width=0.8))

    # 绘制负数的堆积柱状图
    neg_data = np.minimum(data, 0)  # 将正数设置为0
    bars_neg.append(ax.bar(years, neg_data[0], label=categories[0], color=color_list[0], width=0.8))
    for i in range(1, len(categories)):
        bars_neg.append(
            ax.bar(years, neg_data[i], bottom=neg_data[:i].sum(axis=0), label=categories[i], color=color_list[i],
                   width=0.8))

    # 绘制 Net emissions 红色点线图
    line, = ax.plot(years, merged_df['Net emissions'], color='black', marker='o', linewidth=2, label='Net emissions')

    # 设置刻度朝内
    ax.tick_params(axis='both', which='both', direction='in', labelsize=font_size)

    # 设置 x 轴刻度为年份，并确保标签旋转
    ax.set_xlim(2010 - 0.5, 2050 + 0.5)
    ax.set_xticks(years)
    ax.set_xticklabels(years, rotation=90, fontsize=font_size)

    # 设置 y 轴范围和刻度，修改为你数据的合适范围
    ax.set_ylim(-600, 101)
    ax.set_yticks(range(-600, 101, 100))  # 每隔 100 一格
    ax.set_ylabel('GHG Emissions (Mt CO2e)', fontsize=font_size)

    # 隐藏主图中的图例
    ax.legend().remove()

    # 移除网格
    ax.grid(False)

    # 显示主图
    plt.tight_layout()
    plt.savefig(f"03_GHG_{input_name}.png", bbox_inches='tight')  # 保存图片
    plt.show()

    # 创建单独的图例
    fig_legend = plt.figure(figsize=(8, 1))
    handles = [bar[0] for bar in bars_pos] + [line]  # 使用正数柱状图和线条的句柄
    labels = categories + ['Net emissions']
    # 创建没有边框的图例
    legend = fig_legend.legend(handles, labels, loc='center', ncol=len(labels), fontsize=font_size, frameon=False)

    # 保存图例为单独的文件
    fig_legend.savefig("03_GHG_legend.png", bbox_inches='tight')

input_files = ['ON_MAXPROFIT_GHG_15C_67_R10', 'ON_MAXPROFIT_GHG_15C_50_R10', 'ON_MAXPROFIT_GHG_18C_67_R10']
ag_dict = get_value_sum(input_files, 'GHG_emissions_separate_agricultural_landuse',  '*', 'Value (t CO2e)')
am_dict = get_value_sum(input_files, 'GHG_emissions_separate_agricultural_management',  '*', 'Value (t CO2e)')
off_land_dict = get_value_sum(input_files, 'GHG_emissions_offland_commodity',  '*', 'Total GHG Emissions (tCO2e)')
non_ag_dict = get_value_sum(input_files, 'GHG_emissions_separate_no_ag_reduction',  '*', 'Value (t CO2e)')
transition_dict = get_value_sum(input_files, 'GHG_emissions_separate_transition_penalty',  '*', 'Value (t CO2e)')

# 调用函数，得到合并后的字典
merged_dict = merge_dicts_to_df_dict(ag_dict, am_dict, off_land_dict, non_ag_dict, transition_dict)

for input_name in input_files:
    plot_stacked_bar_and_line(merged_dict, input_name, font_size=20)

# 读取 Excel 文件
df = pd.read_excel('../../input/GHG_targets.xlsx', index_col=0)

# 选择 2010 到 2050 年的数据
df_filtered = df.loc[2010:2050,
              ['1.5C (67%) excl. avoided emis', '1.5C (50%) excl. avoided emis', '1.8C (67%) excl. avoided emis']]

# 将所有数据单位转换为 million (除以 1,000,000)
df_filtered = df_filtered / 1e6

# 创建图形和轴对象
fig, ax = plt.subplots(figsize=(10, 6))

# 颜色列表：红色、蓝色、绿色
colors = ['#E74C3C', '#3498DB', '#2ECC71']
font_size = 20

# 绘制三条线，每条线对应不同的目标数据
ax.plot(df_filtered.index, df_filtered['1.5C (67%) excl. avoided emis'], marker='o', color=colors[0], linewidth=2.5, label='1.5°C (67%) ')
ax.plot(df_filtered.index, df_filtered['1.5C (50%) excl. avoided emis'], marker='s', color=colors[1], linewidth=2.5, label='1.5°C (50%) ')
ax.plot(df_filtered.index, df_filtered['1.8C (67%) excl. avoided emis'], marker='^', color=colors[2], linewidth=2.5, label='1.8°C (67%) ')

# 设置刻度朝内
ax.tick_params(axis='both', which='both', direction='in', labelsize=font_size)

# 设置 x 轴刻度为年份，旋转 90 度
ax.set_xlim(2010, 2050)
ax.set_xticks(df_filtered.index)  # 假设 df_filtered 的索引是年份
ax.set_xticklabels(df_filtered.index, rotation=90, fontsize=font_size)
ax.set_xlabel('Year', fontsize=font_size)

# 设置 y 轴范围和间隔，标签名称
ax.set_ylim(-600, 100)
ax.set_yticks(range(-600, 101, 100))
ax.set_ylabel('GHG Emissions (Mt CO2e)', fontsize=font_size)

# 设置图例，并去掉框
ax.legend(fontsize=font_size, frameon=False)

# 显示图形
plt.tight_layout()
plt.savefig("03_GHG_targets.png", bbox_inches='tight')  # 保存图片
plt.show()

# 读取四个保存好的图像
img1 = Image.open("GHG_targets.png")
img2 = Image.open(f"{input_files[0]}_GHG.png")
img3 = Image.open(f"{input_files[1]}_GHG.png")
img4 = Image.open(f"{input_files[2]}_GHG.png")

# 读取单独的图例
legend_img = Image.open("GHG_legend.png")

# 获取图像的宽度和高度，用于调整图例的宽度
width, height = img1.size

# 创建一个新的空白图像用于拼接 (两行两列加上图例)
total_height = 2 * height + legend_img.size[1]
combined_img = Image.new('RGB', (2 * width, total_height), (255, 255, 255))

# 将四个图像粘贴到新的空白图像中
combined_img.paste(img1, (0, 0))
combined_img.paste(img2, (width, 0))
combined_img.paste(img3, (0, height))
combined_img.paste(img4, (width, height))

# 调整图例宽度，使其与拼接后的图像宽度一致
legend_img_resized = legend_img.resize((2 * width, legend_img.size[1]))

# 将图例粘贴到拼接图像的下方
combined_img.paste(legend_img_resized, (0, 2 * height))

# 在图像上添加 (a), (b), (c), (d)
draw = ImageDraw.Draw(combined_img)
font = ImageFont.truetype("arial.ttf", size=40)

# 设置标注的位置和内容
annotations = ['(a)', '(b)', '(c)', '(d)']
x_positions = 120
y_positions = 50
positions = [(x_positions, y_positions), (width + x_positions, y_positions), (x_positions, height + y_positions), (width + x_positions, height + y_positions)]

# 在相应的位置添加标注
for pos, annotation in zip(positions, annotations):
    draw.text(pos, annotation, font=font, fill=(0, 0, 0))

# 保存拼接好的图像
# combined_img.save("03_GHG.png")

# 显示最终拼接好的图像
# combined_img.show()





