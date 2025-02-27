import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import re
import cairosvg
from lxml import etree

import matplotlib
import matplotlib.pyplot as plt

plt.rcParams['svg.fonttype'] = 'none'  # 让字体保持文本格式，而不是转换为路径
plt.rcParams['text.usetex'] = False  # 确保不使用 LaTeX 渲染
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['pdf.fonttype'] = 42  # 保证 AI 里文字可编辑


from tools.parameters import COLUMN_WIDTH, X_OFFSET, axis_linewidth


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

# def process_single_df(df, mapping_df):
#     """
#     对单个 DataFrame 根据映射文件进行处理，返回重命名和排序后的 DataFrame 及 legend_colors。
#
#     参数:
#     df (pd.DataFrame): 需要处理的 DataFrame。
#     mapping_df (pd.DataFrame): 包含 desc 和 color 列的映射文件 DataFrame。
#
#     返回:
#     tuple: 处理后的 DataFrame 和 legend_colors（仅包含匹配的列）。
#     """
#     # 创建映射字典，忽略 desc 中的 `-`
#     mapping_df['desc_processed'] = mapping_df['desc'].str.replace('-', '').str.lower()
#     column_mapping = {row['desc_processed']: row['desc'] for _, row in mapping_df.iterrows()}
#     color_mapping = {row['desc']: row['color'] for _, row in mapping_df.iterrows()}
#
#     # 获取并处理 DataFrame 列名，忽略 `-`
#     original_columns = list(df.columns)
#     processed_columns = [re.sub(r'-', '', col.lower()) for col in original_columns]
#     renamed_columns = [column_mapping.get(col, original_columns[i]) for i, col in enumerate(processed_columns)]
#
#     # 按映射文件的顺序排列匹配的列，未匹配的列保持原始位置
#     matched_categories = [col for col in renamed_columns if col in column_mapping.values()]
#     unmatched_categories = [col for col in renamed_columns if col not in column_mapping.values()]
#     categories = matched_categories + unmatched_categories
#
#     # 筛选 legend_colors 只包含匹配的列
#     legend_colors = {column_mapping[col]: color_mapping[column_mapping[col]]
#                      for col in processed_columns if col in column_mapping}
#
#     # 重命名和排序 DataFrame
#     renamed_df = df.rename(columns={orig_col: column_mapping.get(re.sub(r'-', '', orig_col.lower()), orig_col)
#                                     for orig_col in df.columns})
#     processed_df = renamed_df.reindex(columns=categories, fill_value=0)
#     if 'Year' in processed_df.columns:
#         processed_df = processed_df.set_index('Year')
#
#     return processed_df, legend_colors



def process_single_df(df, mapping_df):
    """
    根据映射文件过滤和处理 DataFrame，仅保留能匹配上的列名。

    参数:
    df (pd.DataFrame): 需要处理的 DataFrame。
    mapping_df (pd.DataFrame): 包含 desc 和 color 列的映射文件 DataFrame。

    返回:
    tuple: 处理后的 DataFrame 和 legend_colors（仅包含匹配的列）。
    """
    # 创建映射字典，忽略 desc 中的 `-` 和大小写
    mapping_df['desc_processed'] = mapping_df['desc'].str.replace('-', '', regex=True).str.lower()
    column_mapping = {row['desc_processed']: row['desc'] for _, row in mapping_df.iterrows()}
    color_mapping = {row['desc']: row['color'] for _, row in mapping_df.iterrows()}

    # 获取并处理 DataFrame 列名，忽略 `-` 和大小写
    original_columns = list(df.columns)
    processed_columns = [re.sub(r'-', '', col).lower() for col in original_columns]

    # 匹配列名并过滤掉无法匹配的列
    matched_indices = [i for i, col in enumerate(processed_columns) if col in column_mapping]
    matched_original_columns = [original_columns[i] for i in matched_indices]
    matched_renamed_columns = [column_mapping[processed_columns[i]] for i in matched_indices]

    # 检查长度是否一致
    if len(matched_indices) != len(matched_renamed_columns):
        raise ValueError("Mismatch between matched indices and renamed columns.")

    # 筛选 legend_colors 只包含匹配的列
    legend_colors = {column_mapping[processed_columns[i]]: color_mapping[column_mapping[processed_columns[i]]]
                     for i in matched_indices}

    # 重命名和筛选 DataFrame
    filtered_df = df[matched_original_columns].rename(
        columns={matched_original_columns[i]: matched_renamed_columns[i] for i in range(len(matched_indices))}
    )
    if 'Year' in filtered_df.columns:
        filtered_df = filtered_df.set_index('Year')

    # 检查是否有 'desc_new' 列
    if 'desc_new' in mapping_df.columns:
        # 创建映射字典 {旧列名: 新列名}
        rename_dict = dict(zip(mapping_df['desc'], mapping_df['desc_new']))
        # 仅重命名 filtered_df 中存在的列
        filtered_df = filtered_df.rename(
            columns={col: rename_dict[col] for col in filtered_df.columns if col in rename_dict})
        legend_colors = {rename_dict.get(key, key): value for key, value in legend_colors.items()}

    return filtered_df, legend_colors


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
            ax.grid(True, axis='y', linestyle='--', linewidth=1.5)
            # 仅隐藏刻度标签，保留刻度和网格线
            # 仅隐藏刻度和刻度标签，保留水平网格线
            if not is_left_col:
                # 如果不是左侧列的子图
                ax.tick_params(axis='y', which='both', left=False, labelleft=False)  # 隐藏 y 轴的刻度和标签
                ax.spines['left'].set_visible(False)  # 隐藏左边框
            else:
                # 如果是左侧列的子图
                ax.spines['left'].set_visible(True)  # 显示左边框
                ax.spines['left'].set_linewidth(axis_linewidth)
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
                ax.spines['bottom'].set_linewidth(axis_linewidth)
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



def save_figure(fig, output_prefix):
    """
    生成以下三种文件：
    1. `output.svg` - 完整的 SVG（包含所有内容）。
    2. `output_no_text.pdf` - 无文字的 PDF（仅保留图案）。
    3. `output_no_plot.svg` - 仅保留边框、坐标轴、文字（删除折线、散点、柱状图）。

    参数:
    - fig: Matplotlib Figure 对象
    - output_prefix: 文件保存的前缀（不包含扩展名）
    """
    svg_path = f"{output_prefix}.svg"
    pdf_path = f"{output_prefix}_no_text.pdf"
    svg_no_plot_path = f"{output_prefix}_no_plot.svg"

    # Step 1: 保存完整的 SVG（带文字和图案）
    fig.savefig(svg_path, bbox_inches='tight', dpi=300, transparent=True, format='svg')

    # Step 2: 解析 SVG
    with open(svg_path, "r", encoding="utf-8") as f:
        svg_content = f.read()

    parser = etree.XMLParser(remove_blank_text=True)
    tree = etree.fromstring(svg_content.encode("utf-8"), parser)

    # Step 3: 生成无文字的 PDF
    tree_no_text = etree.fromstring(svg_content.encode("utf-8"), parser)

    # 删除所有 <text> 元素（生成无文字版本）
    for text_element in tree_no_text.xpath("//svg:text", namespaces={"svg": "http://www.w3.org/2000/svg"}):
        text_element.getparent().remove(text_element)

    # 直接转换无文字的 SVG 数据为 PDF
    svg_no_text = etree.tostring(tree_no_text, encoding="utf-8")
    cairosvg.svg2pdf(bytestring=svg_no_text, write_to=pdf_path, dpi=300)

    # Step 4: 生成仅保留坐标轴、边框、文字的 SVG（去掉折线、散点、柱状图）
    tree_no_plot = etree.fromstring(svg_content.encode("utf-8"), parser)

    # 仅删除绘图数据（path, polygon, circle）
    for element in tree_no_plot.xpath("//svg:path | //svg:polygon | //svg:circle",
                                      namespaces={"svg": "http://www.w3.org/2000/svg"}):
        element.getparent().remove(element)

    # 保存仅有坐标轴和文字的 SVG
    with open(svg_no_plot_path, "wb") as f:
        f.write(etree.tostring(tree_no_plot, pretty_print=True, encoding="utf-8"))

    # 显示原始图（带文字）
    plt.show()



def save_legend_as_image(handles, labels, output_file, ncol=3, legend_position=(0.5, -0.03), font_size=10,format='svg'):
    # 创建单独的图，用于保存图例
    fig, ax = plt.subplots(figsize=(6, 1))  # 设置图例的大小
    ax.axis('off')  # 关闭坐标轴

    # 添加图例
    legend = fig.legend(handles, labels, loc='center', ncol=ncol, fontsize=font_size,
                        handlelength=0.5, handleheight=0.4, handletextpad=0.2,
                        labelspacing=0.4, columnspacing=0.5, frameon=False)

    # 保存图例为单独的文件
    plt.savefig(output_file, bbox_inches='tight', pad_inches=0, dpi=300, transparent=True, format=format)
    plt.show()
    plt.close(fig)  # 关闭图，避免显示在主图上


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

    # Set y-axis limits and ticks
    ax.set_ylim(y_range[0], y_range[1])
    ax.tick_params(axis='both', direction='in')

    # 设置图例
    if show_legend:
        ax.legend(fontsize=font_size)

    return bar_list, line_list



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


def plot_line_chart(ax, merged_dict, input_name, legend_colors, font_size=10,
                    x_range=(2010, 2050), y_range=(-600, 100), x_ticks=None, y_ticks=None,
                    show_legend=False):
    """
    绘制点线图，每列数据对应一条线。

    Parameters:
        ax: matplotlib.axes.Axes, 绘图的坐标轴。
        merged_dict: dict, 包含多个 DataFrame 的字典。
        input_name: str, 要绘制的 DataFrame 的键名。
        legend_colors: dict, 每条线的颜色，键为列名，值为颜色。
        font_size: int, 字体大小。
        x_range: tuple, x 轴范围。
        y_range: tuple, y 轴范围。
        x_ticks: list, 自定义的 x 轴刻度。
        y_ticks: list, 自定义的 y 轴刻度。
        show_legend: bool, 是否显示图例。
    """
    merged_df = merged_dict[input_name]
    merged_df.index = merged_df.index.astype(int)

    # Extract categories and color_list from legend_colors
    categories = list(legend_colors.keys())
    color_list = list(legend_colors.values())

    # Prepare data
    years = merged_df.index

    # Plot each line
    for i, category in enumerate(categories):
        ax.plot(years, merged_df[category], label=category, color=color_list[i], marker='o')

    # Set tick direction inwards
    ax.tick_params(axis='both', which='both', direction='in', labelsize=font_size)

    # Set x-axis limits and ticks
    ax.set_xlim(x_range[0], x_range[1])

    # Set y-axis limits and ticks
    ax.set_ylim(y_range[0], y_range[1])

    # Optionally show legend
    if show_legend:
        ax.legend(fontsize=font_size)

    return ax


# def calculate_y_axis_range(data_dict, multiplier=10, divisible_by=3):
#     """
#     计算累积柱状图的 Y 轴范围和刻度
#     :param data_dict: 包含多个 DataFrame 的字典，每行代表一个柱状图的柱子
#     :param multiplier: 范围是该值的倍数（默认 10）
#     :param divisible_by: 范围除以该值后得到间隔（默认 3）
#     :return: Y 轴范围 (最小值, 最大值) 和刻度间隔
#     """
#
#     # 初始化值
#     row_positive_max = float('-inf')  # 每行正数的累积最大值
#     row_negative_min = float('inf')  # 每行负数的累积最小值
#     has_negative = False  # 是否存在负数
#     global_min_value = float('inf')  # 初始化全局最小值
#     global_max_value = float('-inf')  # 初始化全局最大值
#
#     for df in data_dict.values():
#         # 遍历每一行，逐行计算
#         for index, row in df.iterrows():
#             row_cumsum = row.cumsum()  # 计算行的累积和
#
#             # 计算该行的正数累积最大值
#             row_positive_sum = row_cumsum[row_cumsum > 0].max() if (row_cumsum > 0).any() else 0
#             row_positive_max = max(row_positive_max, row_positive_sum)
#
#             # 计算该行的负数累积最小值
#             if row.min() < 0:
#                 has_negative = True
#                 row_negative_sum = row_cumsum[row_cumsum < 0].min()
#                 row_negative_min = min(row_negative_min, row_negative_sum)
#
#             # 更新全局最小值和最大值
#             global_min_value = min(global_min_value, row.min())
#             global_max_value = max(global_max_value, row.max())
#
#     # 如果没有负数，将累积负数最小值设为 0
#     if not has_negative:
#         row_negative_min = 0
#
#     # 最终比较累积值和全局值，得到最终结果
#     max_value = max(global_max_value, row_positive_max)
#     min_value = min(global_min_value, row_negative_min)
#
#     # 如果没有负数，将最小值设置为 0
#     if not has_negative:
#         cumulative_negative_min = 0
#         min_value = 0
#
#     # 将最小值向下取整到 multiplier 的倍数
#     y_min = np.floor(min_value / multiplier) * multiplier
#
#     # 将最大值向上取整到 multiplier 的倍数
#     y_max = np.ceil(max_value / multiplier) * multiplier
#
#     # 确保 (y_max - y_min) 可以整除 divisible_by
#     while (y_max - y_min) % divisible_by != 0:
#         y_max += multiplier
#
#     interval = (y_max - y_min) // divisible_by
#
#     return (y_min, y_max), interval



def get_max_min(df):
    # 计算正数累积和
    pos_cumsum = df.where(df > 0, 0).cumsum(axis=1)
    pos_max = pos_cumsum.max().max()

    # 计算负数累积和
    neg_cumsum = df.where(df < 0, 0).cumsum(axis=1)
    neg_min = neg_cumsum.min().min()

    # 计算单个值的最值
    value_min = df.min().min()
    value_max = df.max().max()

    # 计算最终最大值（正数部分）
    max_value = max(pos_max, value_max)

    # 计算最终最小值（负数部分）
    min_value = min(neg_min, value_min)

    return max_value, min_value

def calculate_y_axis_range(data_dict, multiplier=10, divisible_by=3, use_multithreading=False):
    max_value, min_value = (
        max(get_max_min(df)[0] for df in data_dict.values()),
        min(get_max_min(df)[1] for df in data_dict.values())
    )

    # 如果 multiplier 小于 1，放大数据和 multiplier
    scale_factor = 1
    if multiplier < 1:
        scale_factor = int(1 / multiplier) * 10
        max_value *= scale_factor
        min_value *= scale_factor
        multiplier = 1

    y_min = np.floor(min_value / multiplier) * multiplier
    y_max = np.ceil(max_value / multiplier) * multiplier

    while (y_max - y_min) % divisible_by != 0:
        y_max += multiplier

    interval = (y_max - y_min) // divisible_by

    # 缩小结果回原比例
    if scale_factor != 1:
        y_min /= scale_factor
        y_max /= scale_factor
        interval /= scale_factor

    return (y_min, y_max), interval


