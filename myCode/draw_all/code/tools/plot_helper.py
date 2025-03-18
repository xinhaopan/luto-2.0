import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import re
import cairosvg
from lxml import etree
from joblib import Parallel, delayed

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
                ax.tick_params(axis='y', labelsize=font_size, pad=5, direction='out')  # 设置刻度标签字体大小

            if is_bottom_row:
                ax.spines['bottom'].set_visible(True)  # 显示 x 轴边框
                ax.spines['bottom'].set_linewidth(axis_linewidth)
                ax.xaxis.set_ticks_position('bottom')
                ax.tick_params(axis='x', labelsize=font_size, direction='out')
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
    # parser = etree.XMLParser(huge_tree=True)
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
    ax.tick_params(axis='both', direction='out')

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
    ax.tick_params(axis='both', which='both', direction='out', labelsize=font_size)

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
    ax.tick_params(axis='both', which='both', direction='out', labelsize=font_size)

    # Set x-axis limits and ticks
    ax.set_xlim(x_range[0], x_range[1])

    # Set y-axis limits and ticks
    ax.set_ylim(y_range[0], y_range[1])

    # Optionally show legend
    if show_legend:
        ax.legend(fontsize=font_size)

    return ax


# def get_max_min(df):
#     # 计算正数累积和
#     pos_cumsum = df.where(df > 0, 0).cumsum(axis=1)
#     pos_max = pos_cumsum.max().max()
#
#     # 计算负数累积和
#     neg_cumsum = df.where(df < 0, 0).cumsum(axis=1)
#     neg_min = neg_cumsum.min().min()
#
#     # 计算单个值的最值
#     value_min = df.min().min()
#     value_max = df.max().max()
#
#     # 计算最终最大值（正数部分）
#     max_value = max(pos_max, value_max)
#
#     # 计算最终最小值（负数部分）
#     min_value = min(neg_min, value_min)
#
#     return max_value, min_value
#
# def calculate_y_axis_range(data_dict, multiplier=10, divisible_by=3, use_multithreading=False):
#     max_value, min_value = (
#         max(get_max_min(df)[0] for df in data_dict.values()),
#         min(get_max_min(df)[1] for df in data_dict.values())
#     )
#
#     # 如果 multiplier 小于 1，放大数据和 multiplier
#     scale_factor = 1
#     if multiplier < 1:
#         scale_factor = int(1 / multiplier) * 10
#         max_value *= scale_factor
#         min_value *= scale_factor
#         multiplier = 1
#
#     y_min = np.floor(min_value / multiplier) * multiplier
#     y_max = np.ceil(max_value / multiplier) * multiplier
#
#     while (y_max - y_min) % divisible_by != 0:
#         y_max += multiplier
#
#     interval = (y_max - y_min) // divisible_by
#
#     # 缩小结果回原比例
#     if scale_factor != 1:
#         y_min /= scale_factor
#         y_max /= scale_factor
#         interval /= scale_factor
#
#     return (y_min, y_max), interval




def get_max_min(df):
    """
    高效计算 df 的最大最小值：
    - 避免逐行 `cumsum(axis=1)`，直接计算正数最大累积和 & 负数最小累积和
    - 直接转换为 NumPy 数组，加速运算
    """
    arr = df.to_numpy()  # 直接转换为 NumPy 数组，加快计算速度

    # 计算每行正数累积和的最大值（优化版）
    pos_cumsum = np.where(arr > 0, arr, 0).cumsum(axis=1)
    pos_max = np.max(pos_cumsum, axis=1)  # 仅计算每行最大值
    overall_pos_max = np.max(pos_max)  # 取整个 df 的最大正数累积值

    # 计算每行负数累积和的最小值（优化版）
    neg_cumsum = np.where(arr < 0, arr, 0).cumsum(axis=1)
    neg_min = np.min(neg_cumsum, axis=1)  # 仅计算每行最小值
    overall_neg_min = np.min(neg_min)  # 取整个 df 的最小负数累积值

    # 计算单个值的最大最小值
    overall_min = np.min(arr)
    overall_max = np.max(arr)

    return max(overall_pos_max, overall_max), min(overall_neg_min, overall_min)

# def calculate_y_axis_range(data_dict, multiplier=10, divisible_by=3, n_jobs=-1):
#     """
#     并行计算所有 DataFrame 的 (max, min) 值
#     """
#     results = Parallel(n_jobs=n_jobs, prefer="threads")(delayed(get_max_min)(df) for df in data_dict.values())
#
#     # 获取所有 DataFrame 的全局最大最小值
#     max_value = max(res[0] for res in results)
#     min_value = min(res[1] for res in results)
#
#     # 如果 multiplier 小于 1，放大数据和 multiplier
#     scale_factor = 1
#     if multiplier < 1:
#         scale_factor = int(1 / multiplier) * 10
#         max_value *= scale_factor
#         min_value *= scale_factor
#         multiplier = 1
#
#     y_min = np.floor(min_value / multiplier) * multiplier
#     y_max = np.ceil(max_value / multiplier) * multiplier
#
#     while (y_max - y_min) % divisible_by != 0:
#         y_max += multiplier
#
#     interval = (y_max - y_min) // divisible_by
#
#     # 缩小结果回原比例
#     if scale_factor != 1:
#         y_min /= scale_factor
#         y_max /= scale_factor
#         interval /= scale_factor
#
#     return (y_min, y_max), interval


import numpy as np
import pandas as pd
from joblib import Parallel, delayed

import numpy as np


def calculate_y_axis_range(data_dict, multiplier=10):
    """
    计算 y 轴范围和间隔：
    - 确保0是刻度中的一个。
    - 当数据范围绝对值≥100时，确保间隔和刻度标签的个位数为0。
    - 刻度数量在4到6之间。

    参数:
        min_value: 数据最小值
        max_value: 数据最大值
        multiplier: 调整时的间隔基数（默认为10）

    返回:
        ((y_min, y_max), interval): y 轴范围和间隔
    """
    # 并行计算所有 DataFrame 的最大值和最小值
    results = Parallel(n_jobs=-1, prefer="threads")(delayed(get_max_min)(df) for df in data_dict.values())
    # 获取全局最大值和最小值
    max_value = max(res[0] for res in results)
    min_value = min(res[1] for res in results)
    adjust_to_ten = min_value <= -100 or max_value >= 100

    # 确保包含0的原始范围
    original_min, original_max = min_value, max_value
    if min_value > 0:
        min_value = 0
    elif max_value < 0:
        max_value = 0
    data_range = max_value - min_value

    best_candidate = None
    min_expansion = float('inf')

    # 尝试不同刻度数寻找最优解
    for n in [4, 5, 6]:
        if data_range == 0:  # 全零特例处理
            y_min, y_max = (-multiplier, multiplier) if adjust_to_ten else (-1, 1)
            interval = multiplier if adjust_to_ten else 1
            return ((y_min, y_max), interval)

        # 计算初始间隔
        if adjust_to_ten:
            interval = multiplier * max(1, np.ceil(data_range / (multiplier * (n - 1))))
        else:
            interval = max(1, np.ceil(data_range / (n - 1)))
            interval = int(interval)

        # 计算范围边界
        y_min = np.floor(min_value / interval) * interval
        y_max = np.ceil(max_value / interval) * interval

        # 确保包含0
        if y_min > 0:
            y_min = 0
        if y_max < 0:
            y_max = 0

        # 验证刻度数
        tick_count = int((y_max - y_min) / interval) + 1
        if not 4 <= tick_count <= 6:
            continue

        # 计算范围扩展量
        expansion = (y_max - original_max) + (original_min - y_min)
        if expansion < min_expansion:
            min_expansion = expansion
            best_candidate = (y_min, y_max, interval)

    # 回退逻辑（找不到合适解时）
    if not best_candidate:
        if adjust_to_ten:
            y_min = np.floor(original_min / multiplier) * multiplier
            y_max = np.ceil(original_max / multiplier) * multiplier
            interval = multiplier
        else:
            y_min = np.floor(original_min)
            y_max = np.ceil(original_max)
            interval = max(1, (y_max - y_min) // 5)
        # 确保包含0
        if y_min > 0:
            y_min = 0
        if y_max < 0:
            y_max = 0
        best_candidate = (y_min, y_max, interval)

    y_min, y_max, interval = best_candidate

    # 百位数以上强制对齐10的倍数
    if adjust_to_ten:
        y_min = np.floor(y_min / multiplier) * multiplier
        y_max = np.ceil(y_max / multiplier) * multiplier
        interval = multiplier * np.ceil(interval / multiplier)
        # 重新计算刻度数并调整间隔
        tick_count = int((y_max - y_min) / interval) + 1
        if tick_count < 4:
            interval = (y_max - y_min) / 3
        elif tick_count > 6:
            interval = (y_max - y_min) / 5

    return ((y_min, y_max), interval)
