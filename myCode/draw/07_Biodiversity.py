import numpy as np
import sys
import os
import pandas as pd
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from tools.helper import *
from tools.parameters import *

plt.rcParams['font.family'] = 'Arial'

# 获取到文件的绝对路径，并将其父目录添加到 sys.path
sys.path.append(os.path.abspath('../../luto'))

# 导入 settings.py
import settings

def plot_stacked_bar(ax, merged_dict, input_name, categories, color_list, font_size=10,
                     x_range=(2010, 2050), y_range=(-600, 100), x_ticks=None, y_ticks=None,
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

    # 返回柱状图句柄
    return bar_list

csv_name, filter_column_name, value_column_name = 'biodiversity_separate', 'Landuse type', 'Biodiversity score'
bio_dict = get_value_sum(input_files, csv_name, filter_column_name, value_column_name)
bio_target_dict = get_unique_value(input_files,'biodiversity', 'Biodiversity score limit', 'Score')
common_keys = set(bio_dict.keys()) & set(bio_target_dict.keys())
merged_dict = {}

for key in common_keys:
    merged_df = pd.concat([bio_dict[key], bio_target_dict[key]], axis=1)
    merged_df.index.name = 'Year'
    merged_df.columns = ['Agricultural landuse', 'Non-agricultural landuse',  'Agricultural management','Biodiversity Target']
    # 将合并后的 DataFrame 存入字典
    merged_dict[key] = merged_df

categories = ['Agricultural landuse', 'Agricultural management',  'Non-agricultural landuse']
color_list = ['#F9C0B7', '#FCD071', '#85C6BE']
output_png = '04_Biodiversity.png'
# 调用绘图函数，动态指定图的数量和布局，控制图例显示
plot_Combination_figures(merged_dict, output_png=output_png, input_names=input_files, plot_func=plot_stacked_bar,
                         categories=categories, color_list=color_list,
                         n_rows=3, n_cols=3, font_size=11,
                         x_range=(2010, 2050), y_range=(0, 120),
                         x_ticks=10, y_ticks=40, x_label='Year', y_label='Quality-weighted Area (Million ha)',
                         legend_position=(0.5, -0.03), show_legend='last', label_positions=(0.01, 0.95))  # 标签位置