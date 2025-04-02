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
                             x_ticks=5, y_ticks=[-200,0,200], legend_position=(0.5, -0.03), show_legend='last', legend_n_rows=1):
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
            ax.set_yticks(y_ticks)  # 更新刻度
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

    range_value = max_value - min_value
    ideal_interval = range_value / (desired_ticks - 1)

    e = math.floor(math.log10(ideal_interval))
    nice_numbers = [1, 2, 5, 10]
    nice_numbers_scaled = [n * (10 ** e) for n in nice_numbers]
    interval = min(nice_numbers_scaled, key=lambda x: abs(x - ideal_interval))

    if min_value < 0 and min_value >= -1:
        min_tick = 0
        max_tick = math.ceil(max_value / interval) * interval
        ticks = list(np.arange(min_tick, max_tick + interval, interval))
        print("Space left below 0 for small negative values like -1.")
    else:
        min_tick = math.floor(min_value / interval) * interval
        max_tick = math.ceil(max_value / interval) * interval
        ticks = list(np.arange(min_tick, max_tick + interval, interval))
        if 0 not in ticks and min_value < 0 < max_value:
            ticks.append(0)
            ticks.sort()

    return interval, ticks

# def get_y_axis_ticks(min_value, max_value, desired_ticks=4):
#     range_value = max_value - min_value
#     if range_value <= 0:
#         return 0, 1, [0, 0.5, 1]
#
#     ideal_interval = range_value / (desired_ticks - 1)
#     e = math.floor(math.log10(ideal_interval))
#     nice_numbers = [1, 2, 5, 10]
#     nice_numbers_scaled = [n * (10 ** e) for n in nice_numbers]
#     interval = min(nice_numbers_scaled, key=lambda x: abs(x - ideal_interval))
#
#     # 统一逻辑：从 min_value 向下取整（不是硬设为 0）
#     min_tick = math.floor(min_value / interval) * interval
#     max_tick = math.ceil(max_value / interval) * interval
#
#     ticks = list(np.arange(min_tick, max_tick + interval, interval))
#
#     # 可选：强制包含 0
#     if 0 not in ticks and min_value < 0 < max_value:
#         ticks.append(0)
#         ticks = sorted(set(ticks))
#
#     # 统一逻辑：从 min_value 向下取整（不是硬设为 0）
#     if abs(min_value) < interval:
#         min_v = math.floor(min_value)
#     else:
#         min_v = math.floor(min_value / interval) * interval
#     max_v = math.ceil(max_value / interval) * interval
#
#     return min_v, max_v, ticks

def get_y_axis_ticks(min_value, max_value, desired_ticks=4):
    """
    生成Y轴刻度，根据数据范围智能调整刻度间隔和范围。
    优化版本，提高运行速度，并处理0-100特殊情况。
    """
    # 1. 快速处理特殊情况
    if min_value > 0 and max_value > 0:
        min_value = 0
    elif min_value < 0 and max_value < 0:
        max_value = 0

    range_value = max_value - min_value
    if range_value <= 0:
        return 0, 1, np.array([0, 0.5, 1])  # 使用numpy数组替代列表

    # 2. 一次性计算间隔
    ideal_interval = range_value / (desired_ticks - 1)
    e = math.floor(math.log10(ideal_interval))
    base = 10 ** e

    # 使用查表法替代计算
    if ideal_interval / base <= 1:
        interval = base
    elif ideal_interval / base <= 2:
        interval = 2 * base
    elif ideal_interval / base <= 5:
        interval = 5 * base
    else:
        interval = 10 * base

    # 3. 整合计算，减少中间变量
    min_tick = math.floor(min_value / interval) * interval
    max_tick = math.ceil(max_value / interval) * interval

    # 4. 使用numpy直接生成数组，避免Python列表操作
    tick_count = int((max_tick - min_tick) / interval) + 1
    ticks = np.linspace(min_tick, max_tick, tick_count)

    # 5. 高效处理0的插入
    if min_value < 0 < max_value and 0 not in ticks:
        # numpy的searchsorted比Python的排序更高效
        zero_idx = np.searchsorted(ticks, 0)
        ticks = np.insert(ticks, zero_idx, 0)

    # 6. 预计算共享变量，避免重复计算
    close_threshold = 0.3 * interval

    # 7. 简化逻辑，减少条件分支
    max_v = max_tick
    min_v = min_tick

    # 处理刻度和范围调整（仅当有足够刻度且最值不是0时）
    if len(ticks) >= 2:
        # 处理最大值
        if ticks[-1] != 0 and (max_value - ticks[-2]) < close_threshold and (ticks[-1] - max_value) > close_threshold:
            ticks = ticks[:-1]  # 移除最后一个刻度
            max_v = max_value + 0.1 * interval

        # 处理最小值
        if ticks[0] != 0 and (ticks[1] - min_value) < close_threshold and (min_value - ticks[0]) > close_threshold:
            ticks = ticks[1:]  # 移除第一个刻度
            min_v = min_value - 0.1 * interval
        elif abs(min_value) < interval:
            min_v = math.floor(min_value)

    # 8. 特殊情况：当刻度范围是0到100时，使用规则的25间隔
    if (abs(ticks[0]) < 1e-10 and abs(ticks[-1] - 100) < 1e-10) or (min_tick == 0 and max_tick == 100):
        ticks = np.array([0, 25, 50, 75, 100])
        min_v = 0
        max_v = 100

    return min_v, max_v, ticks.tolist()  # 根据需要转回列表


def calculate_y_axis_range(data_dict, desired_ticks=4,use_parallel=True, n_jobs=-1):
    """
    并行计算所有 DataFrame 的 (max, min) 值，并计算 y 轴范围和间隔
    """
    # 并行计算每个 DataFrame 的最大最小值
    if use_parallel:
        results = Parallel(n_jobs=n_jobs, prefer="threads")(delayed(get_max_min)(df) for df in data_dict.values())
    else:
        results = [get_max_min(df) for df in data_dict.values()]

    # 获取所有 DataFrame 的全局最大最小值
    max_value = max(res[0] for res in results)
    min_value = min(res[1] for res in results)

    range_value = max_value - min_value
    scale_factor = 1

    if range_value > -1 and range_value < 1:
        scale_exponent = int(abs(math.floor(math.log10(range_value))))
        scale_factor = 10 ** scale_exponent
        max_value *= scale_factor
        min_value *= scale_factor

    min_v, max_v, ticks = get_y_axis_ticks(min_value, max_value, desired_ticks=desired_ticks)

    # 缩小结果回原比例
    if scale_factor != 1:
        min_v /= scale_factor
        max_v /= scale_factor
        ticks = [t / scale_factor for t in ticks]

    if ticks==[0,20,40,60,80,100]:
        ticks=[0,25,50,75,100]
    return ((min_v, max_v), ticks)


def plot_land_use_polar(input_file,output_file=None, result_file="../output/12_land_use_movement_all.xlsx", yticks=None, fontsize=30,x_offset=1.3):
    """
    绘制土地利用变化的极坐标图。

    参数：
      input_file: 用于生成输出文件名的字符串（例如 "mydata"）。
      result_file: 包含土地利用变化数据的 Excel 文件名，默认 "../output/12_land_use_movement_all.xlsx"。
                   Excel 文件中地类是第一列，列为 MultiIndex(年份,指标)。
      yticks: y 轴（径向轴）的刻度列表。如果为 None，则根据数据自动计算。
    """
    # 读取 Excel 文件，地类是第一列
    df = pd.read_excel(result_file, sheet_name=input_file, header=[0, 1])

    # 确保列是多级索引
    if not isinstance(df.columns, pd.MultiIndex):
        raise ValueError("Excel文件的列必须是多级索引，第一级为年份，第二级为指标")

    # 假设地类列是第一列，获取其列名
    land_use_col = df.columns[0]
    # 排除指定的地类
    excluded_land_uses = [
        "Unallocated - modified land",
        "Unallocated - natural land"
    ]
    # 提取地类名称列表
    names = df[land_use_col].unique().tolist()
    names = [land_use for land_use in names if land_use not in excluded_land_uses]

    # 自动提取年份（列索引第一级，从第二列开始）
    years = list(df.columns.levels[0])  # 使用levels获取第一级唯一值

    color_df = pd.read_excel('tools/land use colors.xlsx', sheet_name='ag')
    color_mapping = dict(zip(color_df['desc'], color_df['color']))

    # 更新绘图字体设置
    plt.rcParams.update({'font.size': 10, 'font.family': 'Arial'})

    # 创建极坐标图
    fig, ax = plt.subplots(figsize=(8, 6), subplot_kw={'projection': 'polar'})

    # 绘制每个 land_use 的移动轨迹
    for land_use in names:
        # 筛选当前地类的数据
        land_use_data = df[df[land_use_col] == land_use]

        # 提取距离和角度数据
        distances = []
        angles_deg = []

        for year in years:
            try:
                distance = land_use_data.loc[:, (year, 'Area×Distance')].values[0]
                angle = land_use_data.loc[:, (year, 'Angle (degrees)')].values[0]

                distances.append(distance)
                angles_deg.append(angle)
            except (KeyError, IndexError):
                # 如果某年份数据缺失，跳过
                continue

        if not distances:  # 如果没有数据，跳过该地类
            continue

        angles_rad = np.radians(angles_deg)
        color = color_mapping.get(land_use, '#C8C8C8')
        # 设置线条的粗细和标记点的大小
        ax.plot(angles_rad, distances, marker='o', label=land_use, color=color, linewidth=5, markersize=10)

    # 设置极坐标的角度刻度与标签
    angles_deg_fixed = np.arange(0, 360, 45)
    labels = ['North', 'Northeast', 'East', 'Southeast',
              'South', 'Southwest', 'West', 'Northwest']
    offset_angles = [45, 90, 135, 225, 270, 315]
    # 对于 offset_angles 的位置先不显示标签，后续用 ax.text 标注
    masked_labels = [label if angle not in offset_angles else "" for angle, label in zip(angles_deg_fixed, labels)]

    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.set_xticks(np.radians(angles_deg_fixed))
    ax.set_xticklabels(masked_labels, fontname='Arial', fontsize=fontsize)

    # 计算所有距离的最大值，用于设置y轴刻度
    all_distances = []
    for land_use in names:
        land_use_data = df[df[land_use_col] == land_use]
        for year in years:
            try:
                distance = land_use_data.loc[:, (year, 'Area×Distance')].values[0]
                if not pd.isna(distance):
                    all_distances.append(distance)
            except (KeyError, IndexError):
                continue

    # 如果未提供 yticks，则根据所有距离值自动计算
    if yticks is None:
        if all_distances:
            max_distance = max(all_distances)
            min_v, max_v, yticks = get_y_axis_ticks(0, max_distance)
        else:
            min_v, max_v, yticks = 0, 1, [0, 0.25, 0.5, 0.75, 1]
    else:
        min_v = yticks[0]
        max_v = yticks[-1]

    ax.set_yticks(yticks)
    ax.set_ylim(min_v, max_v)
    # 如果你还想更改字体，可以使用：
    labels_y = ax.get_yticklabels()
    ax.set_yticklabels(labels_y, fontname='Arial', fontsize=fontsize)

    # 在指定角度处添加完整标签
    # 定义不同方向的偏移乘数
    cardinal_offset = 1.0  # 控制四个主方向（东南西北）的偏移
    east_west_offset = 0.92  # 控制东西方向的额外偏移

    # 在指定角度处添加完整标签
    for angle_deg in offset_angles:
        angle_rad = np.radians(angle_deg)
        # 查找对应的标签
        label = labels[angles_deg_fixed.tolist().index(angle_deg)]

        # 设置基础偏移量
        current_offset = x_offset * cardinal_offset

        # 为东西方向设置额外偏移
        if angle_deg == 90 or angle_deg == 270:  # 东西方向
            current_offset = x_offset * east_west_offset

        # 添加标签
        ax.text(angle_rad,
                ax.get_rmax() * current_offset,
                label,
                ha='center',
                va='center',
                fontsize=fontsize,
                fontname='Arial')

        # 调试信息，确保正确的偏移量
        # print(f'angle_deg: {angle_deg}, current_offset: {current_offset}, label: {label}')

    ax.set_rlabel_position(180)  # 将径向标签移动至180度方向
    # 设置网格线的样式和粗细
    ax.grid(True, linestyle='--', linewidth=2, alpha=0.99)

    # 设置径向网格线的粗细
    ax.xaxis.set_tick_params(width=3)
    ax.yaxis.set_tick_params(width=3)
    # 添加图例
    # ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.0))

    # 保存图像，构造输出文件名
    if output_file is None:
        output_file = f"../output/12_{input_file}_polar"
    else:
        output_file = f"../output/{output_file}"
    fig.savefig(f'{output_file}.pdf', dpi=300, bbox_inches='tight')
    save_figure(fig, output_file)

    plt.show()
