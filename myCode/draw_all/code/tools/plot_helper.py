import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import re

from tools.parameters import COLUMN_WIDTH, X_OFFSET


def get_colors(merged_dict, mapping_file, sheet_name=None):
    """
    根据映射文件处理 merged_dict 中的每个表，忽略 `-`，并返回 data_dict 和 legend_colors。

    参数:
    merged_dict (dict): 包含多个 DataFrame 的字典。
    mapping_file (str): 映射文件路径，包含 desc 和 color 列。
    sheet_name (str, 可选): 如果指定，读取映射文件中的指定 sheet。

    返回:
    tuple: data_dict（处理后的 DataFrame 字典）和 legend_colors（图例名称与颜色的对应关系）。
    """
    # 读取映射文件
    if sheet_name:
        mapping_df = pd.read_excel(mapping_file, sheet_name=sheet_name)
    else:
        mapping_df = pd.read_csv(mapping_file)

    # 处理 merged_dict 中的每个 DataFrame，获取处理后的 DataFrame 字典和 legend_colors
    data_dict = {}
    legend_colors = {}
    for key, df in merged_dict.items():
        processed_df, color_dict = process_single_df(df, mapping_df)
        data_dict[key] = processed_df
        legend_colors.update(color_dict)  # 合并每个表的颜色映射

    return data_dict, legend_colors

def process_single_df(df, mapping_df):
    """
    对单个 DataFrame 根据映射文件进行处理，返回重命名和排序后的 DataFrame 及 legend_colors。

    参数:
    df (pd.DataFrame): 需要处理的 DataFrame。
    mapping_df (pd.DataFrame): 包含 desc 和 color 列的映射文件 DataFrame。

    返回:
    tuple: 处理后的 DataFrame 和 legend_colors（仅包含匹配的列）。
    """
    # 创建映射字典，忽略 desc 中的 `-`
    mapping_df['desc_processed'] = mapping_df['desc'].str.replace('-', '').str.lower()
    column_mapping = {row['desc_processed']: row['desc'] for _, row in mapping_df.iterrows()}
    color_mapping = {row['desc']: row['color'] for _, row in mapping_df.iterrows()}

    # 获取并处理 DataFrame 列名，忽略 `-`
    original_columns = list(df.columns)
    processed_columns = [re.sub(r'-', '', col.lower()) for col in original_columns]
    renamed_columns = [column_mapping.get(col, original_columns[i]) for i, col in enumerate(processed_columns)]

    # 按映射文件的顺序排列匹配的列，未匹配的列保持原始位置
    matched_categories = [col for col in renamed_columns if col in column_mapping.values()]
    unmatched_categories = [col for col in renamed_columns if col not in column_mapping.values()]
    categories = matched_categories + unmatched_categories

    # 筛选 legend_colors 只包含匹配的列
    legend_colors = {column_mapping[col]: color_mapping[column_mapping[col]]
                     for col in processed_columns if col in column_mapping}

    # 重命名和排序 DataFrame
    renamed_df = df.rename(columns={orig_col: column_mapping.get(re.sub(r'-', '', orig_col.lower()), orig_col)
                                    for orig_col in df.columns})
    processed_df = renamed_df.reindex(columns=categories, fill_value=0)
    if 'Year' in processed_df.columns:
        processed_df = processed_df.set_index('Year')

    return processed_df, legend_colors



def plot_Combination_figures(merged_dict, output_png, input_names, plot_func, legend_colors,
                             point_data=None, n_rows=3, n_cols=3, font_size=10, x_range=(2010, 2050), y_range=(-600, 100),
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
            ax.grid(True, axis='y', linestyle='--', linewidth=0.5)
            # 仅隐藏刻度标签，保留刻度和网格线
            # 仅隐藏刻度和刻度标签，保留水平网格线
            if not is_left_col:
                # 如果不是左侧列的子图
                ax.tick_params(axis='y', which='both', left=False, labelleft=False)  # 隐藏 y 轴的刻度和标签
                ax.spines['left'].set_visible(False)  # 隐藏左边框
            else:
                # 如果是左侧列的子图
                ax.spines['left'].set_visible(True)  # 显示左边框
                ax.yaxis.set_ticks_position('left')  # y 轴刻度在左侧
                # ax.set_ylim(y_range[0], y_range[1])  # 设置 y 轴范围

                # 如果是底部行的子图，移除最底部的刻度值
                if is_bottom_row:
                    yticks = yticks[1:]  # 从第二个刻度值开始

                ax.set_yticks(yticks)  # 更新刻度
                ax.tick_params(axis='y', labelsize=font_size)  # 设置刻度标签字体大小

            if is_bottom_row:
                ax.spines['bottom'].set_visible(True)  # 显示 x 轴边框
                ax.xaxis.set_ticks_position('bottom')
                ax.tick_params(axis='x', labelsize=font_size)
                ax.set_xlim(x_range[0] - X_OFFSET, x_range[1] + X_OFFSET)
                ax.set_xticks(np.arange(x_range[0], x_range[1] + 1, x_ticks))
                ax.set_xticklabels(ax.get_xticks(), rotation=0, fontsize=font_size)
            else:
                ax.spines['bottom'].set_visible(False)  # 隐藏 x 轴边框
                ax.xaxis.set_ticks([])  # 隐藏 x 轴刻度

            # 隐藏所有图的顶部和右侧边框
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            if point_data is not None:
                bar_list, line = plot_func(ax, merged_dict=merged_dict, input_name=input_names[i], legend_colors=legend_colors,
                                       point_data=point_data, font_size=font_size, x_range=x_range, y_range=y_range,
                                       x_ticks=x_ticks, y_ticks=y_ticks)
            else:
                bar_list = plot_func(ax, merged_dict=merged_dict, input_name=input_names[i],  legend_colors=legend_colors,
                              font_size=font_size, x_range=x_range, y_range=y_range, x_ticks=x_ticks, y_ticks=y_ticks)

            # 只在最后一个图获取图例句柄和标签
            if i == total_plots - 1:
                handles, labels_legend = ax.get_legend_handles_labels()
                if point_data is not None:
                    all_handles.extend(handles[:len(legend_colors)+1])
                    all_labels.extend(labels_legend[:len(legend_colors)+1])
                else:
                    all_handles.extend(handles[:len(legend_colors)])
                    all_labels.extend(labels_legend[:len(legend_colors)])

    ncol = math.ceil(len(legend_colors) / legend_n_rows)
    output_file = output_png.replace(".png", "_legend.png")
    save_legend_as_image(all_handles, all_labels, output_file, ncol, font_size=10)
    # 调整布局
    plt.tight_layout()
    plt.savefig(output_png, bbox_inches='tight', dpi=300, transparent=True)
    plt.show()

def save_legend_as_image(handles, labels, output_file, ncol=3, legend_position=(0.5, -0.03), font_size=10):
    # 创建单独的图，用于保存图例
    fig, ax = plt.subplots(figsize=(6, 1))  # 设置图例的大小
    ax.axis('off')  # 关闭坐标轴

    # 添加图例
    legend = fig.legend(handles, labels, loc='center', ncol=ncol, fontsize=font_size,
                        handlelength=0.5, handleheight=0.4, handletextpad=0.2,
                        labelspacing=0.4, columnspacing=0.5, frameon=False)

    # 保存图例为单独的文件
    plt.savefig(output_file, bbox_inches='tight', pad_inches=0, dpi=300, transparent=True)
    plt.close(fig)  # 关闭图，避免显示在主图上


def plot_stacked_bar_and_line(ax, merged_dict, input_name, legend_colors, point_data='Net emissions',
                              font_size=10, x_range=(2010, 2050), y_range=(-600, 100), x_ticks=None, y_ticks=None,
                              show_legend=False):
    merged_df = merged_dict[input_name]
    merged_df.index = merged_df.index.astype(int)

    # 从 legend_colors 中获取 categories 和 color_list
    categories = list(legend_colors.keys())
    color_list = list(legend_colors.values())

    # 准备数据
    years = merged_df.index
    data = np.array([merged_df[category] for category in categories])

    # 绘制正数的堆积柱状图
    pos_data = np.maximum(data, 0)
    bar_list = []
    bar_list.append(ax.bar(years, pos_data[0], label=categories[0], color=color_list[0], width=COLUMN_WIDTH ))
    for i in range(1, len(categories)):
        bar_list.append(
            ax.bar(years, pos_data[i], bottom=pos_data[:i].sum(axis=0), label=categories[i], color=color_list[i],
                   width=COLUMN_WIDTH))

    # 绘制负数的堆积柱状图
    neg_data = np.minimum(data, 0)
    for i in range(len(categories)):
        bar_list.append(
            ax.bar(years, neg_data[i], bottom=neg_data[:i].sum(axis=0), label=categories[i], color=color_list[i],
                   width=COLUMN_WIDTH))

    # 绘制 Net emissions 的点线图
    line = ax.plot(years, merged_df[point_data], color='red', marker='o', linewidth=1.5, label=point_data, markersize=3)

    # 设置 x 和 y 轴范围
    ax.set_ylim(y_range[0], y_range[1]+1)
    ax.tick_params(axis='both', direction='in')
    return bar_list, line

def plot_stacked_bar(ax, merged_dict, input_name, legend_colors, font_size=10,
                     x_range=(2010, 2050), y_range=(-600, 100), x_ticks=None, y_ticks=None,
                     show_legend=False):
    merged_df = merged_dict[input_name]
    merged_df.index = merged_df.index.astype(int)

    # Extract categories and color_list from legend_colors
    categories = list(legend_colors.keys())
    color_list = list(legend_colors.values())

    # Prepare data
    years = merged_df.index
    data = np.array([merged_df[category] for category in categories])

    # Plot positive stacked bar
    pos_data = np.maximum(data, 0)
    bar_list = []
    bar_list.append(ax.bar(years, pos_data[0], label=categories[0], color=color_list[0], width=COLUMN_WIDTH ))
    for i in range(1, len(categories)):
        bar_list.append(
            ax.bar(years, pos_data[i], bottom=pos_data[:i].sum(axis=0), label=categories[i], color=color_list[i],
                   width=COLUMN_WIDTH))

    # Plot negative stacked bar
    neg_data = np.minimum(data, 0)
    for i in range(len(categories)):
        bar_list.append(
            ax.bar(years, neg_data[i], bottom=neg_data[:i].sum(axis=0), label=categories[i], color=color_list[i],
                   width=COLUMN_WIDTH))

    # Set tick direction inwards
    ax.tick_params(axis='both', which='both', direction='in', labelsize=font_size)

    # Set x-axis limits and ticks
    ax.set_xlim(x_range[0] - X_OFFSET, x_range[1] + X_OFFSET)

    # Set y-axis limits and ticks
    ax.set_ylim(y_range[0], y_range[1])
    # if y_ticks is not None:
    #     ax.set_yticks(np.arange(y_range[0], y_range[1] + 1, y_ticks))

    return bar_list

def calculate_y_axis_range(data_dict, multiplier=10, divisible_by=3):
    """
    计算累积柱状图的 Y 轴范围和刻度
    :param data_dict: 包含多个 DataFrame 的字典，每行代表一个柱状图的柱子
    :param multiplier: 范围是该值的倍数（默认 10）
    :param divisible_by: 范围除以该值后得到间隔（默认 3）
    :return: Y 轴范围 (最小值, 最大值) 和刻度间隔
    """
    min_value, max_value = 0, 0

    # 遍历字典中的每个 DataFrame 和行
    for df in data_dict.values():
        row_sums = df.sum(axis=1)  # 每行累积总和
        min_value = min(min_value, row_sums.min())  # 累积总和的最小值
        max_value = max(max_value, row_sums.max())  # 累积总和的最大值

    # 将最小值向下取整到 multiplier 的倍数
    y_min = np.floor(min_value / multiplier) * multiplier

    # 将最大值向上取整到 multiplier 的倍数
    y_max = np.ceil(max_value / multiplier) * multiplier

    # 确保 (y_max - y_min) 可以整除 divisible_by
    while (y_max - y_min) % divisible_by != 0:
        y_max += multiplier

    interval = (y_max - y_min) // divisible_by

    return (y_min, y_max), interval
