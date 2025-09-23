import xarray as xr
import pandas as pd
import os
import tools.config as config
import statsmodels.api as sm
import pandas as pd
import seaborn as sns
import matplotlib.patches as mpatches
import matplotlib as mpl
from matplotlib.lines import Line2D
import xarray as xr
import matplotlib.gridspec as gridspec

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator
from matplotlib.patches import Patch
import re
from typing import List, Optional, Dict

def xarray_to_dict(file_path: str, scale: float = None, add_total: bool = False,negative=False, year_threshold: int = 2020) -> dict:
    """
    将 (scenario, Year, type) 的 xarray.DataArray 转换为 dict。
    key: scenario坐标
    value: DataFrame，index=Year, columns=type
    可选 scale 参数：如果有，则所有值除以 scale
    可选 add_total 参数：如果为 True，则新增一列'Total'为每行之和
    可选 year_threshold 参数：如果指定，只保留 Year >= year_threshold 的数据
    """
    data_array = xr.open_dataarray(file_path)
    result = {}
    arr = data_array
    if scale is not None:
        arr = arr / scale

    if negative:
        arr = -arr

    # 遍历每个 scenario
    for scenario in arr.coords['scenario'].values:
        da_s = arr.sel(scenario=scenario)
        years = da_s.coords['Year'].values
        # 根据 year_threshold 进行筛选
        if year_threshold is not None:
            mask = years >= year_threshold
            years = years[mask]
            da_s = da_s.sel(Year=years)
        df = pd.DataFrame(
            da_s.values,
            index=da_s.coords['Year'].values,
            columns=da_s.coords['type'].values
        )
        if add_total:
            df['Total'] = df.sum(axis=1)
        result[scenario] = df
    return result

def stacked_area_pos_neg(
        ax, df, colors=None, alpha=0.60,
        title_name='', ylabel='',
        add_line=True, n_col=1,dividing_line=0.5, show_legend=False, bbox_to_anchor=(0.5, -0.25)
):
    """
    绘制可分正负值的堆叠面积图，并在区域间加白色细线。
    """
    # 数据分离
    df_stack = df.iloc[:, :-1]
    total_col_name = df.columns[-1]
    total_col_data = df.iloc[:, -1]

    # 颜色管理
    # if colors is None:
    #     colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    # if len(colors) < df_stack.shape[1]:
    #     num_stack_cols = df_stack.shape[1]
    #     colors = (colors * (num_stack_cols // len(colors) + 1))[:num_stack_cols]

    # 颜色管理
    default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    if colors is None:
        colors = default_colors
    elif isinstance(colors, dict):
        # dict模式：用列名查色，查不到用默认色
        colors = [colors.get(col, default_colors[idx % len(default_colors)]) for idx, col in
                       enumerate(df_stack.columns)]

    cum_pos = np.zeros(len(df_stack))
    cum_neg = np.zeros(len(df_stack))
    for idx, col in enumerate(df_stack.columns):
        y = df_stack[col].values
        pos = np.clip(y, 0, None)
        neg = np.clip(y, None, 0)
        colr = colors[idx]
        # 面积填充
        ax.fill_between(df_stack.index, cum_pos, cum_pos + pos,
                        facecolor=colr, alpha=alpha, linewidth=0, label=col)
        ax.fill_between(df_stack.index, cum_neg, cum_neg + neg,
                        facecolor=colr, alpha=alpha, linewidth=0)
        # 分割线
        if dividing_line:
            ax.plot(df_stack.index, cum_pos + pos, color='white', linewidth=dividing_line, zorder=10)
            ax.plot(df_stack.index, cum_neg + neg, color='white', linewidth=dividing_line, zorder=10)
            # 下边界线（不包括第一个区域）
            if idx > 0:
                ax.plot(df_stack.index, cum_pos, color='white', linewidth=dividing_line, zorder=10)
                ax.plot(df_stack.index, cum_neg, color='white', linewidth=dividing_line, zorder=10)
            cum_pos += pos
            cum_neg += neg

    # 可选总和线
    if add_line:
        ax.plot(
            df.index, total_col_data,
            linestyle='-', marker='o',
            color='black', linewidth=2,
            markersize=5, markeredgewidth=1,
            markerfacecolor='black', markeredgecolor='black',
            label=total_col_name,zorder=20
        )

    # 轴和外观
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:,.0f}'))
    ax.set_xlim(df.index.min(), df.index.max())
    ax.set_title(title_name, pad=6)
    ax.set_ylabel(ylabel)
    ax.set_xlabel('')
    ax.tick_params(direction='out')
    for spine in ax.spines.values():
        spine.set_color('black')
        spine.set_linewidth(1.2)

    # 图例处理
    handles, labels = ax.get_legend_handles_labels()
    unique = {}
    for h, l in zip(handles, labels):
        if l not in unique:
            unique[l] = h
    if add_line and total_col_name in unique:
        total_handle = unique.pop(total_col_name)
        final_labels = list(unique.keys()) + [total_col_name]
        final_handles = list(unique.values()) + [total_handle]
    else:
        final_labels = list(unique.keys())
        final_handles = list(unique.values())

    if show_legend:
        ax.legend(
            handles=final_handles, labels=final_labels,
            loc='upper left', bbox_to_anchor=bbox_to_anchor,
            frameon=False, ncol=n_col,
            handlelength=1.0, handleheight=1.0,
            handletextpad=0.4, labelspacing=0.3
        )

    return ax


def set_plot_style(font_size=12, font_family='Arial'):
    sns.set_theme(style="darkgrid")

    # 2. 设置常规文本字体 (例如 Arial)
    #    这会影响标题、坐标轴标签等非 mathtext 元素
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial']

    # 3. 将可能被修改过的 mathtext 设置恢复到 Matplotlib 的默认状态。
    #    这是一个安全的赋值操作，而不是不稳定的 pop 操作。
    #    默认的 mathtext 字体是 'dejavusans'，默认模式是 'it' (斜体)。
    mpl.rcParams['mathtext.rm'] = 'dejavusans'
    mpl.rcParams['mathtext.default'] = 'it'

    # 4. 现在，覆盖 mathtext 的字体集，让它使用与 Arial 外观相似的 'stixsans'。
    #    这是实现字体统一外观的关键步骤，并且非常稳定。
    plt.rcParams['mathtext.fontset'] = 'stixsans'

    # 5. 应用您其他的自定义样式
    plt.rcParams.update({
        "xtick.bottom": True, "ytick.left": True,
        "xtick.top": False, "ytick.right": False,
        "xtick.direction": "out", "ytick.direction": "out",
        "xtick.major.size": 4, "ytick.major.size": 4,
        "xtick.major.width": 1.2, "ytick.major.width": 1.2,
        "font.size": 24,
    })

    mpl.rcParams.update({
        'font.size': font_size, 'font.family': font_family, 'axes.titlesize': font_size,
        'axes.labelsize': font_size, 'xtick.labelsize': font_size, 'ytick.labelsize': font_size,
        'legend.fontsize': font_size, 'figure.titlesize': font_size
    })


def draw_legend(ax, bbox_to_anchor=(0.85, 0.69), ncol=2, column_spacing=1.0):
    """
    绘制调整过布局的图例。
    - bbox_to_anchor 控制整体位置 (x=0.85 使其左移)。
    - ncol=2 设置为两列。
    - column_spacing=1.0 减小列间距。
    """
    fig = ax.get_figure()
    handles, labels = ax.get_legend_handles_labels()

    # # --- 新增：交换第5项和第6项 ---
    # handles[4], handles[5] = handles[5], handles[4]
    # # 交换 labels 列表中的第5个和第6个元素
    # labels[4], labels[5] = labels[5], labels[4]
    # # --- 交换结束 ---

    ghost_handle = Patch(alpha=0)
    ghost_label = ""

    # 3. 将占位项添加到列表末尾
    for _ in range(3):
        handles.append(ghost_handle)
        labels.append(ghost_label)

    if ax.get_legend() is not None:
        ax.get_legend().remove()

    # ... (创建 new_handles 的代码保持不变) ...
    new_handles = []
    for h in handles:
        if isinstance(h, Patch):
            new_handles.append(
                Patch(facecolor=h.get_facecolor(), edgecolor=h.get_edgecolor(), linewidth=h.get_linewidth()))
        elif isinstance(h, Line2D):
            new_handles.append(Line2D([0], [0], color=h.get_color(), linestyle=h.get_linestyle(),
                                      linewidth=h.get_linewidth(), marker=h.get_marker(),
                                      markersize=h.get_markersize(), markerfacecolor=h.get_markerfacecolor(),
                                      markeredgecolor=h.get_markeredgecolor()))
        else:
            new_handles.append(h)

    # 关键修改：应用新的布局参数
    fig.legend(handles=new_handles, labels=labels, loc='upper left',
               bbox_to_anchor=bbox_to_anchor,
               ncol=ncol,
               frameon=False,
               handlelength=1.0,
               handleheight=1.0,
               handletextpad=0.4,
               labelspacing=0.3,
               columnspacing=column_spacing  # <--- 控制列间距
               )

def get_global_ylim(dict_df: Dict[str, pd.DataFrame], offset=0.05) -> tuple:
    """Calculate global y-limits (supporting positive and negative stacks). Exclude 'Total' column if present."""
    global_max = float("-inf")
    global_min = float("inf")

    for df in dict_df.values():
        # 如果有 'Total' 列，先排除
        df_proc = df.drop(columns=['Total']) if 'Total' in df.columns else df

        arr = np.nan_to_num(df_proc.to_numpy(dtype=float), nan=0.0)

        # 正向累积最大值
        pos_cumsum = np.where(arr > 0, arr, 0).cumsum(axis=1)
        pos_max = np.max(pos_cumsum) if pos_cumsum.size else 0.0

        # 负向累积最小值
        neg_cumsum = np.where(arr < 0, arr, 0).cumsum(axis=1)
        neg_min = np.min(neg_cumsum) if neg_cumsum.size else 0.0

        # 单个值极值
        data_max = np.max(arr) if arr.size else 0.0
        data_min = np.min(arr) if arr.size else 0.0

        row_max = max(pos_max, data_max)
        row_min = min(neg_min, data_min)

        if row_max > global_max:
            global_max = row_max
        if row_min < global_min:
            global_min = row_min

    # 保证 0 在线内
    if global_min > 0:
        global_min = 0
    if global_max < 0:
        global_max = 0

    # 上下各加 5% padding
    span = global_max - global_min
    if span == 0:
        span = 1.0
    return (global_min - offset * span, global_max + offset * span)

from matplotlib.patches import Patch
from matplotlib.lines import Line2D

def draw_legend(ax, bbox_to_anchor=(0.85, 0.69), ncol=2, column_spacing=1.0, ghost_legend_num=None):
    """
    绘制调整过布局的图例。
    - bbox_to_anchor 控制整体位置 (x=0.85 使其左移)。
    - ncol=2 设置为两列。
    - column_spacing=1.0 控制列间距。
    - ghost_legend_num: 若设定则添加透明占位。
    - 图例顺序：先线(Line2D)后块(Patch)
    """
    fig = ax.get_figure()
    handles, labels = ax.get_legend_handles_labels()

    # ghost legend
    ghost_handle = Patch(alpha=0)
    ghost_label = ""
    if ghost_legend_num:
        for _ in range(ghost_legend_num):
            handles.append(ghost_handle)
            labels.append(ghost_label)
        if ax.get_legend() is not None:
            ax.get_legend().remove()

    # 分类排序：先线再块再其它
    line_handles = []
    line_labels = []
    patch_handles = []
    patch_labels = []
    other_handles = []
    other_labels = []

    for h, l in zip(handles, labels):
        if isinstance(h, Line2D):
            line_handles.append(h)
            line_labels.append(l)
        elif isinstance(h, Patch):
            patch_handles.append(h)
            patch_labels.append(l)
        else:
            other_handles.append(h)
            other_labels.append(l)

    # 复制属性（如你原有的 new_handles 代码）
    def clone_handle(handle):
        if isinstance(handle, Patch):
            return Patch(facecolor=handle.get_facecolor(), edgecolor=handle.get_edgecolor(),
                         linewidth=handle.get_linewidth())
        elif isinstance(handle, Line2D):
            return Line2D([0], [0], color=handle.get_color(), linestyle=handle.get_linestyle(),
                          linewidth=handle.get_linewidth(), marker=handle.get_marker(),
                          markersize=handle.get_markersize(), markerfacecolor=handle.get_markerfacecolor(),
                          markeredgecolor=handle.get_markeredgecolor())
        return handle

    new_handles = [clone_handle(h) for h in (line_handles + patch_handles + other_handles)]
    new_labels = line_labels + patch_labels + other_labels

    # 绘制图例
    fig.legend(handles=new_handles, labels=new_labels, loc='upper left',
               bbox_to_anchor=bbox_to_anchor,
               ncol=ncol,
               frameon=False,
               handlelength=1.0,
               handleheight=1.0,
               handletextpad=0.4,
               labelspacing=0.3,
               columnspacing=column_spacing
               )

def plot_12_layout(
    all_dfs: dict,
    title_map: dict,
    colors: list,
    output_path: str,
    summary_ylim: tuple,
    bbox_to_anchor,
    stacked_area_pos_neg=stacked_area_pos_neg,
    draw_legend=draw_legend,
    ylabel: str = 'Net economic returns (Billion AU$)',
    dividing_line=0.5,
    ncol=2,
    column_spacing=-7,
    ghost_legend_num=3,
    figsize=(22, 12)
):
    """
    绘制复杂布局的经济收益图
    all_dfs: dict，key为title_names，value为df
    title_map: dict，key为title_names，value为中文/显示名
    colors: list，每个区域的颜色
    output_dir: 输出目录
    summary_ylim: (ymin, ymax) Y轴范围元组
    bbox_to_anchor: 图例位置参数
    stacked_area_pos_neg: 区域图绘制函数(ax, df, colors, title_name, ylabel, show_legend)
    draw_legend: 图例绘制函数(ax, bbox_to_anchor, ncol, column_spacing)
    ylabel: Y轴标签
    """
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(3, 5, figure=fig, hspace=0.3, wspace=0.3)
    axes = []
    axes.append(fig.add_subplot(gs[0, 0]))
    axes.append(fig.add_subplot(gs[0, 1], sharey=axes[0]))
    axes.append(fig.add_subplot(gs[0, 2], sharey=axes[0]))
    legend_ax = fig.add_subplot(gs[0, 3:])
    legend_ax.axis('off')

    for i in range(5):
        axes.append(fig.add_subplot(gs[1, i], sharey=axes[0]))
    for i in range(5):
        axes.append(fig.add_subplot(gs[2, i], sharey=axes[0]))

    legend_handles, legend_labels = None, None

    title_names = list(all_dfs.keys())
    for i, (ax, title_key) in enumerate(zip(axes, title_names)):
        df = all_dfs[title_key]
        title = title_map.get(title_key, title_key)

        stacked_area_pos_neg(
            ax, df, colors=colors,
            title_name=title,
            ylabel='' if i != 3 else ylabel,
            show_legend=False,
            dividing_line=dividing_line
        )

        x_data = df.index
        start_tick, middle_tick, end_tick = x_data.min(), x_data[len(x_data) // 2], x_data.max()
        tick_positions = [start_tick, middle_tick, end_tick]

        ax.set_xticks(tick_positions)
        if i in [8, 9, 10, 11, 12]:
            ax.tick_params(axis='x')
            x_labels = ax.get_xticklabels()
            if len(x_labels) >= 3:
                x_labels[0].set_horizontalalignment('left')
                x_labels[-1].set_horizontalalignment('right')
        else:
            ax.tick_params(axis='x', labelbottom=False)

        if i not in [0, 3, 8]:
            ax.tick_params(axis='y', labelleft=False)

        if i == 0:
            legend_handles, legend_labels = ax.get_legend_handles_labels()

    axes[0].set_ylim(*summary_ylim)
    legend_ax.set_position(bbox_to_anchor)

    if legend_handles:
        draw_legend(axes[0], bbox_to_anchor=bbox_to_anchor, ncol=ncol, column_spacing=column_spacing,ghost_legend_num=ghost_legend_num)

    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))
    output_path = os.path.join(output_path)
    fig.savefig(output_path, dpi=300)
    fig.show()
    print(f"✅ Saved to {output_path}")

def plot_13_layout(
    all_carbon: list,
    all_bio: list,
    title_carbon_names: list,
    title_bio_names: list,
    colors: list,
    output_path: str,
    carbon_ylim: tuple,
    bio_ylim: tuple,
    bbox_to_anchor,
    stacked_area_pos_neg,
    draw_legend,
    ylabel: str = 'Cost (Billion AU$)',
    dividing_line=0.5,
    ncol=2,
    column_spacing=-7,
    figsize=(20, 12)
):
    """
    绘制复杂布局的碳和生物成本图（3x5子图分布，前两为碳，后十为生物）。
    all_carbon: list，长度为2，每个为df
    all_bio: list，长度为10，每个为df
    title_carbon_names: list，碳图标题
    title_bio_names: list，生物图标题
    colors: list，堆叠区颜色
    output_path: 输出图文件路径
    carbon_ylim: (ymin, ymax)，碳图Y轴范围
    bio_ylim: (ymin, ymax)，生物图Y轴范围
    bbox_to_anchor: 图例位置参数
    stacked_area_pos_neg: 堆叠面积函数
    draw_legend: 图例绘制函数
    ylabel: 全局Y轴标签
    dividing_line: 区域分割线线宽
    ncol: 图例列数
    column_spacing: 图例列间距
    """
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(3, 5, figure=fig, hspace=0.5, wspace=0.2)

    # --- Carbon 图（第一行前两个） ---
    ax_carbon_list = []
    for i in range(2):
        ax = fig.add_subplot(gs[0, i])
        stacked_area_pos_neg(
            ax, all_carbon[i], colors=colors,
            title_name=title_carbon_names[i],
            ylabel='', show_legend=False,
            dividing_line=dividing_line
        )
        ax.set_ylim(*carbon_ylim)
        x_data = all_carbon[i].index
        start_tick, middle_tick, end_tick = x_data.min(), x_data[len(x_data) // 2], x_data.max()
        tick_positions = [start_tick, middle_tick, end_tick]
        ax.set_xticks(tick_positions)
        ax.tick_params(axis='x', labelbottom=False)
        ax_carbon_list.append(ax)
    # 控制共享Y轴和刻度
    ax_carbon_list[0].sharey(ax_carbon_list[1])
    ax_carbon_list[1].tick_params(axis='y', labelleft=False)
    ax_carbon_list[0].set_ylim(*carbon_ylim)
    ax_carbon_list[0].yaxis.set_major_locator(MaxNLocator(4))

    # --- 图例区域（第一行后三个合并） ---
    legend_ax = fig.add_subplot(gs[0, 2:])
    legend_ax.axis('off')

    # --- Bio 图（后两行） ---
    ax_bio_list = []
    shared_bio_ax = None
    for i in range(len(all_bio)):
        row = i // 5 + 1
        col = i % 5
        if shared_bio_ax is None:
            ax = fig.add_subplot(gs[row, col])
            shared_bio_ax = ax
        else:
            ax = fig.add_subplot(gs[row, col], sharey=shared_bio_ax)
        stacked_area_pos_neg(
            ax, all_bio[i], colors=colors,
            title_name=title_bio_names[i],
            ylabel='', show_legend=False,
            dividing_line=dividing_line
        )
        # 控制Y轴刻度：只在最左边一列显示
        if col != 0:
            ax.tick_params(axis='y', labelleft=False)
        # 控制X轴刻度：只在最下面一行显示
        x_data = all_bio[i].index
        start_tick, middle_tick, end_tick = x_data.min(), x_data[len(x_data) // 2], x_data.max()
        tick_positions = [start_tick, middle_tick, end_tick]
        ax.set_xticks(tick_positions)
        if row == 2:
            ax.tick_params(axis='x')
            x_labels = ax.get_xticklabels()
            if len(x_labels) >= 3:
                x_labels[0].set_horizontalalignment('left')
                x_labels[-1].set_horizontalalignment('right')
        else:
            ax.tick_params(axis='x', labelbottom=False)
        ax_bio_list.append(ax)
    # Y轴范围与刻度
    shared_bio_ax.set_ylim(*bio_ylim)
    locator = MaxNLocator(nbins=5, prune=None, min_n_ticks=5)
    shared_bio_ax.yaxis.set_major_locator(locator)

    # --- 全局Y轴标题 ---
    fig.supylabel(ylabel, x=0.08, fontsize=24)

    # --- 图例 ---
    draw_legend(ax_carbon_list[0], bbox_to_anchor=bbox_to_anchor, ncol=ncol, column_spacing=column_spacing)

    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))
    fig.savefig(output_path, dpi=300)
    fig.show()
    plt.close(fig)
    print(f"✅ Saved to {output_path}")

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
    # for key, df in merged_dict.items():
    #     processed_df, color_dict = process_single_df(df, mapping_df)
    #     data_dict[key] = processed_df
    #     legend_colors.update(color_dict)  # 合并每个表的颜色映射

    for key, df in merged_dict.items():
        # 如果有 Total 列，就先取出来
        total_col = None
        if 'Total' in df.columns:
            total_col = df['Total']
            df = df.drop(columns=['Total'])
        processed_df, legend_colors = process_single_df(df, mapping_df)
        # 合并 Total 列到最后
        if total_col is not None:
            processed_df['Total'] = total_col
            # 强制 Total 为最后一列
            cols = [c for c in processed_df.columns if c != 'Total'] + ['Total']
            processed_df = processed_df[cols]
        data_dict[key] = processed_df

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
    # 先处理 mapping_df 和 df 的列名
    mapping_df['desc_processed'] = mapping_df['desc'].str.replace(r'[-\s]', '', regex=True).str.lower()
    column_mapping = {row['desc_processed']: row['desc'] for _, row in mapping_df.iterrows()}
    color_mapping = {row['desc']: row['color'] for _, row in mapping_df.iterrows()}

    original_columns = list(df.columns)
    processed_columns = [re.sub(r'[-\s]', '', col).lower() for col in original_columns]
    col_map = dict(zip(processed_columns, original_columns))  # processed列名 -> 原始列名

    # **按 mapping_df 顺序筛选和重命名**
    matched_processed = []
    matched_original = []
    matched_renamed = []

    for desc_proc in mapping_df['desc_processed']:
        if desc_proc in col_map:
            matched_processed.append(desc_proc)
            matched_original.append(col_map[desc_proc])
            matched_renamed.append(column_mapping[desc_proc])

    if not matched_original:
        raise ValueError("没有任何列名能够和映射文件匹配，请检查输入的 DataFrame 和映射文件。")

    legend_colors = {column_mapping[desc_proc]: color_mapping[column_mapping[desc_proc]] for desc_proc in
                     matched_processed}

    # 保证顺序与 mapping_df 一致
    filtered_df = df[matched_original].rename(
        columns={matched_original[i]: matched_renamed[i] for i in range(len(matched_original))}
    )
    if 'Year' in filtered_df.columns:
        filtered_df = filtered_df.set_index('Year')

    # 如果有 desc_new
    if 'desc_new' in mapping_df.columns:
        rename_dict = dict(zip(mapping_df['desc'], mapping_df['desc_new']))
        filtered_df = filtered_df.rename(
            columns={col: rename_dict[col] for col in filtered_df.columns if col in rename_dict})
        legend_colors = {rename_dict.get(key, key): value for key, value in legend_colors.items()}

    return filtered_df, legend_colors

def plot_22_layout(
    all_dfs: dict,
    title_map: dict,
    colors: list,
    output_path: str,
    summary_ylim: tuple,
    bbox_to_anchor,
    stacked_area_pos_neg=stacked_area_pos_neg,
    draw_legend=draw_legend,
    ylabel: str = 'Cost (Billion AU$)',
    dividing_line=0.5,
    ncol=2,
    column_spacing=1,
    figsize=(22, 20)
):
    """
    绘制5x5布局的22张图。all_dfs为dict，key为图类型+序号（如'carbon_0'），value为df；
    title_map为dict，key同上，value为标题。
    第一行前两列画图，第3-5列合并为图例。其余行每行5张。
    """
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(5, 5, figure=fig, hspace=0.5, wspace=0.2)

    # 画图例（第一行第3~5列合并）
    legend_ax = fig.add_subplot(gs[0, 2:])
    legend_ax.axis('off')

    keys_ordered = list(all_dfs.keys())
    axes_list = []
    shared_ax = None

    for plot_idx, key in enumerate(keys_ordered):
        # 行列计算
        if plot_idx < 2:
            row, col = 0, plot_idx
        else:
            # plot_idx从2开始，对应row=1+(plot_idx-2)//5，col=(plot_idx-2)%5
            row = 1 + (plot_idx - 2) // 5
            col = (plot_idx - 2) % 5

        df = all_dfs.get(key)
        title = title_map.get(key, key)

        if shared_ax is None:
            ax = fig.add_subplot(gs[row, col])
            shared_ax = ax
        else:
            ax = fig.add_subplot(gs[row, col], sharey=shared_ax)
        stacked_area_pos_neg(
            ax, df, colors=colors,
            title_name=title,
            ylabel='', show_legend=False,
            dividing_line=dividing_line
        )
        if col != 0:
            ax.tick_params(axis='y', labelleft=False)
        ax.set_ylim(*summary_ylim)
        x_data = df.index
        start_tick, middle_tick, end_tick = x_data.min(), x_data[len(x_data) // 2], x_data.max()
        tick_positions = [start_tick, middle_tick, end_tick]
        ax.set_xticks(tick_positions)
        if row == 4:
            ax.tick_params(axis='x')
            x_labels = ax.get_xticklabels()
            if len(x_labels) >= 3:
                x_labels[0].set_horizontalalignment('left')
                x_labels[-1].set_horizontalalignment('right')
        else:
            ax.tick_params(axis='x', labelbottom=False)
        axes_list.append(ax)

    # --- 全局Y轴标题 ---
    first_ax = axes_list[0]  # 第一个 subplot
    pos = first_ax.get_position(fig)  # Bbox, figure坐标
    x = pos.x0 - 0.05  # 比左侧再左一点，0.02可以自己试
    x = max(x, 0)  # 防止越界到负数

    fig.text(x, 0.5, ylabel, va='center', rotation='vertical')

    # --- 图例 ---
    draw_legend(axes_list[0], bbox_to_anchor=bbox_to_anchor, ncol=ncol, column_spacing=column_spacing)

    # --- 保存 ---
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))
    fig.savefig(output_path, dpi=300)
    fig.show()
    plt.close(fig)
    print(f"✅ Saved to {output_path}")