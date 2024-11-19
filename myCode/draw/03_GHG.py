from tools.helper import *
from tools.parameters import *
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import sys
from scipy.interpolate import interp1d

# 获取到文件的绝对路径，并将其父目录添加到 sys.path
sys.path.append(os.path.abspath('../../luto'))

# 导入 settings.py
import settings
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
        # merged_df['Off-land'] = off_land_dict[input_name]['Total']
        merged_df['Non-agricultural landuse'] = non_ag_dict[input_name]['Total']
        merged_df['Transition'] = transition_dict[input_name]['Total']

        # 计算 Net emissions 列为五个列的求和
        merged_df['Net emissions'] = merged_df.sum(axis=1)

        # 将合并后的 DataFrame 存储在新的字典中，保持原来的 key
        merged_dict[input_name] = merged_df

    return merged_dict


def plot_stacked_bar_and_line(ax, merged_dict, input_name, categories, color_list, point_data='Net emissions',
                              font_size=10, x_range=(2010, 2050), y_range=(-600, 100), x_ticks=None, y_ticks=None,
                              x_label='Year', y_label='GHG Emissions (Mt CO2e)', show_legend=False):
    merged_df = merged_dict[input_name]
    merged_df.index = merged_df.index.astype(int)

    # 准备数据
    years = merged_df.index
    data = np.array([merged_df[category] for category in categories])

    # 绘制正数的堆积柱状图
    pos_data = np.maximum(data, 0)  # 将负数设置为0
    bar_list = []
    bar_list.append(ax.bar(years, pos_data[0], label=categories[0], color=color_list[0], width=0.8))
    for i in range(1, len(categories)):
        bar_list.append(
            ax.bar(years, pos_data[i], bottom=pos_data[:i].sum(axis=0), label=categories[i], color=color_list[i],
                   width=0.8))

    # 绘制负数的堆积柱状图
    neg_data = np.minimum(data, 0)  # 将正数设置为0
    for i in range(len(categories)):
        bar_list.append(
            ax.bar(years, neg_data[i], bottom=neg_data[:i].sum(axis=0), label=categories[i], color=color_list[i],
                   width=0.8))

    # 绘制 Net emissions 的红色点线图
    line = ax.plot(years, merged_df[point_data], color='black', marker='o', linewidth=1.5, label=point_data, markersize=3)

    # 设置刻度朝内
    ax.tick_params(axis='both', which='both', direction='in', labelsize=font_size)

    # 设置 x 轴刻度和间距
    ax.set_xlim(x_range[0] - 0.5, x_range[1] + 0.5)
    if x_ticks is not None:
        ax.set_xticks(np.arange(x_range[0], x_range[1] + 1, x_ticks))
    ax.set_xticklabels(ax.get_xticks(), fontsize=font_size)

    # 设置 y 轴范围和刻度
    ax.set_ylim(y_range[0], y_range[1])
    if y_ticks is not None:
        ax.set_yticks(np.arange(y_range[0], y_range[1] + 1, y_ticks))

    # 设置自定义 x 轴和 y 轴标签
    ax.set_xlabel(x_label, fontsize=font_size)
    ax.set_ylabel(y_label, fontsize=font_size)

    # 返回柱状图和线图的句柄
    return bar_list, line

ag_dict = get_value_sum(input_files, 'GHG_emissions_separate_agricultural_landuse',  '*', 'Value (t CO2e)')
am_dict = get_value_sum(input_files, 'GHG_emissions_separate_agricultural_management',  '*', 'Value (t CO2e)')
off_land_dict = get_value_sum(input_files, 'GHG_emissions_offland_commodity',  '*', 'Total GHG Emissions (tCO2e)')
non_ag_dict = get_value_sum(input_files, 'GHG_emissions_separate_no_ag_reduction',  '*', 'Value (t CO2e)')
transition_dict = get_value_sum(input_files, 'GHG_emissions_separate_transition_penalty',  '*', 'Value (t CO2e)')

# 调用函数，得到合并后的字典
ghg_dict = merge_dicts_to_df_dict(ag_dict, am_dict, off_land_dict, non_ag_dict, transition_dict)

# categories = ['Agricultural landuse', 'Agricultural management', 'Off-land', 'Non-agricultural landuse', 'Transition']
# color_list = ['#F9C0B7', '#FCD071', '#B4A7D6', '#85C6BE', '#D2E0FB']
categories = ['Agricultural landuse', 'Agricultural management',  'Non-agricultural landuse', 'Transition']
color_list = ['#F9C0B7', '#FCD071', '#85C6BE', '#D2E0FB']
output_png = '03_GHG.png'
# 调用绘图函数，动态指定图的数量和布局，控制图例显示
plot_Combination_figures(ghg_dict, output_png='03_GHG.png', input_names=input_files, plot_func=plot_stacked_bar_and_line,
                         categories=categories, color_list=color_list,
                         point_data='Net emissions',
                         n_rows=3, n_cols=3, font_size=12,
                         x_range=(2010, 2050), y_range=(-600, 100),
                         x_ticks=10, y_ticks=100, x_label='Year', y_label='GHG Emissions (Mt CO2e)',
                         legend_position=(0.5, -0.03), show_legend='last', label_positions=(0.01, 0.95))  # 标签位置