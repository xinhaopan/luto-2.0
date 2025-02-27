import numpy as np
import sys
import os
import pandas as pd
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import math

from tools.data_helper import *
from tools.plot_helper import get_colors,save_legend_as_image,save_figure
from tools.parameters import *

plt.rcParams['font.family'] = 'Arial'

# 获取到文件的绝对路径，并将其父目录添加到 sys.path
sys.path.append(os.path.abspath('../../../luto'))

def plot_Combination_figures(merged_dict, output_png, input_names, plot_func, legend_colors,point_dict=None,point_colors=None,
                            n_rows=3, n_cols=3, font_size=10, x_range=(2010, 2050), y_range=(-600, 100),
                             x_ticks=5, y_ticks=100, legend_position=(0.5, -0.03), show_legend='last', legend_n_rows=1):
    total_plots = len(input_names)
    fig_width = 12
    fig_height = 8
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height))

    # 扁平化 axs 以支持单个子图或多图
    axs = axs.flat if total_plots > 1 else [axs]
    all_handles = []
    all_labels = []

    for i, ax in enumerate(axs):
        if i < total_plots:
            # 标记左侧和底部的子图
            is_left_col = (i % n_cols == 0)
            is_bottom_row = (i >= (n_rows - 1) * n_cols)

            # 生成 y 轴刻度
            yticks = np.arange(y_range[0], y_range[1] + y_ticks, y_ticks)
            ax.set_yticks(yticks)  # 更新刻度
            ax.set_ylim(y_range[0], y_range[1])

            # 显示所有图的水平网格线
            ax.grid(True, axis='y', linestyle='--')
            # 仅隐藏刻度标签，保留刻度和网格线
            # 仅隐藏刻度和刻度标签，保留水平网格线
            if not is_left_col:
                # 如果不是左侧列的子图
                ax.tick_params(axis='y', which='both', left=False, labelleft=False)  # 隐藏 y 轴的刻度和标签
                ax.spines['left'].set_visible(False)  # 隐藏左边框
            else:
                # 如果是左侧列的子图
                ax.spines['left'].set_visible(True)  # 显示左边框
                # ax.spines['left'].set_linewidth(axis_linewidth)
                ax.yaxis.set_ticks_position('left')  # y 轴刻度在左侧
                # ax.set_ylim(y_range[0], y_range[1])  # 设置 y 轴范围

                # # 如果是底部行的子图，移除最底部的刻度值
                # if is_bottom_row:
                #     yticks = yticks[1:]  # 从第二个刻度值开始
                #
                # ax.set_yticks(yticks)  # 更新刻度
                ax.tick_params(axis='y', labelsize=font_size, pad=5)  # 设置刻度标签字体大小

            if is_bottom_row:
                ax.spines['bottom'].set_visible(True)  # 显示 x 轴边框
                # ax.spines['bottom'].set_linewidth(axis_linewidth)
                ax.xaxis.set_ticks_position('bottom')
                ax.tick_params(axis='x', labelsize=font_size, length=8)
                ax.set_xlim(x_range[0] - X_OFFSET, x_range[1] + X_OFFSET)
                ax.set_xticks(np.arange(x_range[0], x_range[1] + 1, x_ticks))
                ax.set_xticklabels(ax.get_xticks(), rotation=0, fontsize=font_size)
            else:
                ax.spines['bottom'].set_visible(False)  # 隐藏 x 轴边框
                ax.xaxis.set_ticks([])  # 隐藏 x 轴刻度

            # 隐藏所有图的顶部和右侧边框
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            if point_dict is not None:
                bar_list, lines = plot_func(ax, merged_dict=merged_dict, input_name=input_names[i], legend_colors=legend_colors,point_dict=point_dict,point_colors=point_colors,
                                        font_size=font_size, x_range=x_range, y_range=y_range,
                                       x_ticks=x_ticks, y_ticks=y_ticks)
            else:
                bar_list = plot_func(ax, merged_dict=merged_dict, input_name=input_names[i],  legend_colors=legend_colors,
                              font_size=font_size, x_range=x_range, y_range=y_range, x_ticks=x_ticks, y_ticks=y_ticks)

            # 只在最后一个图获取图例句柄和标签
            if i == total_plots - 1:
                handles, labels_legend = ax.get_legend_handles_labels()
                if point_dict is not None:
                    all_handles.extend(handles[:len(legend_colors)+len(point_colors)])
                    all_labels.extend(labels_legend[:len(legend_colors)+len(point_colors)])
                else:
                    all_handles.extend(handles[:len(legend_colors)])
                    all_labels.extend(labels_legend[:len(legend_colors)])

    ncol = math.ceil(len(all_labels) / legend_n_rows)
    legend_file = f"{output_png}" + "_legend.svg"
    save_legend_as_image(all_handles, all_labels, legend_file, ncol, font_size=10)
    # 调整布局
    plt.tight_layout()
    save_figure(fig, output_png)
    plt.show()

def plot_stacked_bar_and_line(ax, merged_dict, input_name, legend_colors, point_dict=None, point_colors=None,
                              font_size=10, x_range=(2010, 2050), y_range=(-600, 100),
                              x_ticks=None, y_ticks=None, show_legend=False):
    """
    绘制堆积柱状图和多条点线图。

    Parameters:
        ax: matplotlib Axes 对象。
        merged_dict (dict): 包含堆积柱状图数据的字典。
        input_name (str): 数据键名。
        legend_colors (dict): 包含类别和颜色的字典。
        point_dict (dict): 包含点线图数据的字典。
        point_colors (list): 每条点线图的颜色列表。
        font_size (int): 字体大小。
        x_range (tuple): X 轴范围。
        y_range (tuple): Y 轴范围。
        x_ticks (list): X 轴刻度。
        y_ticks (list): Y 轴刻度。
        show_legend (bool): 是否显示图例。

    Returns:
        bar_list, line_list: 堆积柱状图和点线图的绘图对象。
    """
    merged_df = merged_dict[input_name]
    merged_df.index = merged_df.index.astype(int)

    if point_dict is not None:
        point_df = point_dict[input_name]
        point_df.index = point_df.index.astype(int)

    # 从 legend_colors 中获取 categories 和 color_list
    categories = list(legend_colors.keys())
    color_list = list(legend_colors.values())

    # 准备数据
    years = merged_df.index
    data = np.array([merged_df[category] for category in categories])

    # 绘制正数的堆积柱状图
    pos_data = np.maximum(data, 0)
    bar_list = []
    bar_list.append(ax.bar(years, pos_data[0], label=categories[0], color=color_list[0], width=COLUMN_WIDTH))
    for i in range(1, len(categories)):
        bar_list.append(
            ax.bar(years, pos_data[i], bottom=pos_data[:i].sum(axis=0), label=categories[i], color=color_list[i],
                   width=COLUMN_WIDTH,zorder=2))

    # 绘制负数的堆积柱状图
    neg_data = np.minimum(data, 0)
    for i in range(len(categories)):
        bar_list.append(
            ax.bar(years, neg_data[i], bottom=neg_data[:i].sum(axis=0), label=categories[i], color=color_list[i],
                   width=COLUMN_WIDTH))

    # 绘制点线图
    line_list = []
    if point_dict is not None:
        for idx, column in enumerate(point_df.columns):
            color = point_colors[idx] if point_colors and idx < len(point_colors) else 'black'  # 指定颜色或默认黑色
            line = ax.plot(years, point_df[column], marker='o', linewidth=1.5, label=column, markersize=3, color=color,zorder=3)
            line_list.append(line)

    # Set x-axis limits and ticks
    ax.set_xlim(x_range[0] - X_OFFSET, x_range[1] + X_OFFSET)
    ax.tick_params(length=6)

    # Set y-axis limits and ticks
    ax.set_ylim(y_range[0], y_range[1])
    ax.tick_params(axis='both', direction='in')

    # 设置图例
    if show_legend:
        ax.legend(fontsize=font_size)

    return bar_list, line_list

# 导入 settings.py
import settings

X_OFFSET = 5
font_size = 16

Objective_demand_dict = get_dict_sum_data(input_files, "quantity_comparison", 'Demand (tonnes, KL)', 'Objectives')
Product_demand_dict = get_dict_sum_data(input_files, "quantity_comparison", 'Prod_targ_year (tonnes, KL)', 'Production')
point_dict = concatenate_dicts_by_year([Objective_demand_dict, Product_demand_dict])
point_colors = ['red','black']

csv_name, value_column_name, filter_column_name = 'quantity_comparison', 'Prod_targ_year (tonnes, KL)', 'Commodity'
demand_dict = get_dict_data(input_files, csv_name, value_column_name, filter_column_name)
demand_ag_group_dict,legend_colors = get_colors(demand_dict, 'tools/land use colors.xlsx', sheet_name='food')
output_png = '../output/08_S2_food.png'
plot_Combination_figures(demand_ag_group_dict, output_png, input_files, plot_stacked_bar_and_line, legend_colors,point_dict=point_dict,point_colors=point_colors,
                            n_rows=3, n_cols=3, font_size=font_size, x_range=(2015, 2050), y_range=(0, 200),
                             x_ticks=10, y_ticks=50,
                             legend_position=(0.5, -0.25), show_legend='last', legend_n_rows=7)

