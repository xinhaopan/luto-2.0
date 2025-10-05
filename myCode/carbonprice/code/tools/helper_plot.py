import os
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import xarray as xr

from matplotlib import gridspec
import matplotlib.lines as mlines
import matplotlib.patches as Patch
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from pygam import LinearGAM, s

import re
from typing import List, Optional, Dict

import tools.config as config

def xarray_to_dict(file_path: str, scale: float = None, total_name: str =None,negative=False, year_threshold: int = 2020) -> dict:
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

        if total_name:
            df[total_name] = df.sum(axis=1)
        result[scenario] = df
    return result

def xarray_to_long_df(file_path: str, scale: float = None, negative=False, year_threshold: int = 2020) -> pd.DataFrame:
    """
    将 (scenario, Year, type) 的 xarray.DataArray 转换为长格式 DataFrame。
    每行包含: scenario, Year, type, value
    scale: 所有值除以 scale
    negative: 是否取负值
    year_threshold: 只保留 Year >= year_threshold 的数据
    """
    data_array = xr.open_dataarray(file_path)
    arr = data_array
    if scale is not None:
        arr = arr / scale
    if negative:
        arr = -arr

    # 筛选年份
    if year_threshold is not None:
        years = arr.coords['Year'].values
        mask = years >= year_threshold
        years = years[mask]
        arr = arr.sel(Year=years)

    # 转为 DataFrame（长格式）
    df_long = arr.to_dataframe().reset_index()
    # 如果 value 列不是叫 'value'，统一改名
    if arr.name and arr.name in df_long.columns:
        df_long = df_long.rename(columns={arr.name: "value"})
    return df_long[['scenario', 'Year', 'type', 'value']]

def stacked_area_pos_neg(
        ax, df, colors=None, alpha=0.60,
        title_name='', ylabel='',y_ticks_all=None,
        total_name=None, n_col=1,dividing_line=0.5, show_legend=False, bbox_to_anchor=(0.5, -0.25), y_labelpad=6
):
    """
    绘制可分正负值的堆叠面积图，并在区域间加白色细线。
    """
    # 数据分离
    if total_name:
        df_stack = df.drop(columns=[total_name])
        total_col_data = df[total_name]
        ax.plot(
            df.index, total_col_data,
            linestyle='-', color='#404040', linewidth=3,  # thick dark grey
            label=total_name, zorder=20
        )
    else:
        df_stack = df

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


    # 轴和外观
    if y_ticks_all is not None:
        ax.set_ylim(y_ticks_all[0], y_ticks_all[1])
        ax.set_yticks(y_ticks_all[2])
    else:
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    if abs(y_ticks_all[2][-1]) > 1000:
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:,.0f}'))
    elif abs(y_ticks_all[2][1]) < 1 and y_ticks_all[2][1] != 0:
        ax.yaxis.set_major_formatter(
            FuncFormatter(lambda x, _: f'{int(x)}' if abs(x) < 1e-10 else f'{x:.1f}')
        )
    ax.set_xlim(df.index.min(), df.index.max())
    ax.set_title(title_name, pad=6)
    ax.set_ylabel(ylabel,labelpad=y_labelpad)
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
    if total_name:
        total_handle = unique.pop(total_name)
        final_labels = list(unique.keys()) + [total_name]
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

        "font.size": font_size,
        'font.family': font_family,
        'axes.titlesize': font_size * 1.2,
        'axes.labelsize': font_size,
        'xtick.labelsize': font_size,
        'ytick.labelsize': font_size,
        'legend.fontsize': font_size,
        'figure.titlesize': font_size * 1.2
    })

    mpl.rcParams.update({
        'font.size': font_size, 'font.family': font_family, 'axes.titlesize': font_size*1.2,
        'axes.labelsize': font_size, 'xtick.labelsize': font_size, 'ytick.labelsize': font_size,
        'legend.fontsize': font_size, 'figure.titlesize': font_size*1.2
    })


# def draw_legend(ax, bbox_to_anchor=(0.85, 0.69), ncol=2, column_spacing=1.0):
#     """
#     绘制调整过布局的图例。
#     - bbox_to_anchor 控制整体位置 (x=0.85 使其左移)。
#     - ncol=2 设置为两列。
#     - column_spacing=1.0 减小列间距。
#     """
#     fig = ax.get_figure()
#     handles, labels = ax.get_legend_handles_labels()
#
#     # # --- 新增：交换第5项和第6项 ---
#     # handles[4], handles[5] = handles[5], handles[4]
#     # # 交换 labels 列表中的第5个和第6个元素
#     # labels[4], labels[5] = labels[5], labels[4]
#     # # --- 交换结束 ---
#
#     ghost_handle = Patch(alpha=0)
#     ghost_label = ""
#
#     # 3. 将占位项添加到列表末尾
#     for _ in range(3):
#         handles.append(ghost_handle)
#         labels.append(ghost_label)
#
#     if ax.get_legend() is not None:
#         ax.get_legend().remove()
#
#     # ... (创建 new_handles 的代码保持不变) ...
#     new_handles = []
#     for h in handles:
#         if isinstance(h, Patch):
#             new_handles.append(
#                 Patch(facecolor=h.get_facecolor(), edgecolor=h.get_edgecolor(), linewidth=h.get_linewidth()))
#         elif isinstance(h, Line2D):
#             new_handles.append(Line2D([0], [0], color=h.get_color(), linestyle=h.get_linestyle(),
#                                       linewidth=h.get_linewidth(), marker=h.get_marker(),
#                                       markersize=h.get_markersize(), markerfacecolor=h.get_markerfacecolor(),
#                                       markeredgecolor=h.get_markeredgecolor()))
#         else:
#             new_handles.append(h)
#
#     # 关键修改：应用新的布局参数
#     fig.legend(handles=new_handles, labels=labels, loc='upper left',
#                bbox_to_anchor=bbox_to_anchor,
#                ncol=ncol,
#                frameon=False,
#                handlelength=1.0,
#                handleheight=1.0,
#                handletextpad=0.4,
#                labelspacing=0.3,
#                columnspacing=column_spacing  # <--- 控制列间距
#                )

def get_global_ylim(dict_df: Dict[str, pd.DataFrame], offset=0.01) -> tuple:
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

    # ghost legend
    ghost_handle = Patch(alpha=0)
    ghost_label = ""
    if ghost_legend_num:
        for _ in range(ghost_legend_num):
            handles.append(ghost_handle)
            labels.append(ghost_label)
        if ax.get_legend() is not None:
            ax.get_legend().remove()

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
               columnspacing=column_spacing,
               )

def plot_13_layout(
    all_dfs: dict,
    title_map: dict,
    colors: list,
    output_path: str,
    summary_ylim: tuple,
    desired_ticks: int = 5,
    total_name=None,
    bbox_to_anchor=[0.58, 0.82, 0.4, 0.1],
    stacked_area_pos_neg=stacked_area_pos_neg,
    draw_legend=draw_legend,
    ylabel: str = 'Net economic returns (Billion AU$)',
    dividing_line=0.5,
    ncol=2,
    column_spacing=1,
    ghost_legend_num=3,
    figsize=(22, 14),
    post_process=None,
    y_labelpad=6
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
    all_dfs = filter_and_rename_dict_keys(all_dfs, title_map)
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(3, 5, figure=fig, hspace=0.15, wspace=0.15)
    fig.subplots_adjust(left=0.06, right=0.97, top=0.95, bottom=0.05)
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
    y_ticks_all = get_y_axis_ticks(summary_ylim[0], summary_ylim[1],
                                                               desired_ticks=desired_ticks)
    for i, (ax, title_key) in enumerate(zip(axes, title_names)):
        df = all_dfs[title_key]
        df = df[df.index >= config.START_YEAR]
        title = get_partial_match_title(title_key,title_map)

        stacked_area_pos_neg(
            ax, df, colors=colors,
            title_name=title,
            ylabel='' if i != 3 else ylabel,
            y_ticks_all=y_ticks_all,
            total_name=total_name,
            show_legend=False,
            dividing_line=dividing_line,
            y_labelpad=y_labelpad
        )

        x_data = df.index
        x_min, x_max = x_data.min(), x_data.max()
        tick_positions = list(range(int(x_min), int(x_max) + 1, 5))
        tick_positions = [year for year in tick_positions if year in x_data]
        # start_tick, middle_tick, end_tick = x_data.min(), x_data[len(x_data) // 2], x_data.max()
        # tick_positions = [start_tick, middle_tick, end_tick]
        ax.set_xticks(tick_positions)
        ax.tick_params(axis='x', labelrotation=45)
        # 底行显示 x 轴标签，其余行隐藏
        if i in [8, 9, 10, 11, 12]:
            ax.tick_params(axis='x')
        else:
            ax.tick_params(axis='x', labelbottom=False)

        # 左列保留 y 轴标签，其余列隐藏
        if i not in [0, 3, 8]:
            ax.tick_params(axis='y', labelleft=False)

        ax.figure.canvas.draw()

        # 仅对 9,10,11,12：隐藏 x 轴“最左边”的刻度标签（保留刻度线）
        if i in [9, 10, 11, 12]:
            xtlabs = ax.get_xticklabels()
            if xtlabs:
                xtlabs[0].set_visible(False)  # 隐藏第一个（最左）

        # 仅对 8：隐藏 y 轴“最下面”的刻度标签（保留刻度线）
        # if i == 8:
        #     ytlabs = ax.get_yticklabels()
        #     if ytlabs:
        #         ytlabs[0].set_visible(False)  # 隐藏第一个（最下）


        if i == 0:
            legend_handles, legend_labels = ax.get_legend_handles_labels()

    if post_process:
        axes = post_process(axes)


    legend_ax.set_position(bbox_to_anchor)

    if legend_handles:
        draw_legend(axes[-1], bbox_to_anchor=bbox_to_anchor, ncol=ncol, column_spacing=column_spacing,ghost_legend_num=ghost_legend_num)

    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))
    output_path = os.path.join(output_path)
    fig.savefig(output_path, dpi=300)
    fig.show()
    print(f"✅ Saved to {output_path}")


def plot_25_layout(
        all_dfs: dict,
        title_map: dict,
        colors: list,
        output_path: str,
        summary_ylim: tuple,
        desired_ticks: int = 5,
        total_name=None,
        bbox_to_anchor=[0.2, -0.05, 0.6, 0.05],  # 调整图例到底部
        stacked_area_pos_neg=stacked_area_pos_neg,
        draw_legend=draw_legend,
        ylabel: str = 'Net economic returns (Billion AU$)',
        dividing_line=0.5,
        ncol=1,  # 图例列数增加
        column_spacing=1,
        ghost_legend_num=3,
        figsize=(25, 30),  # 增加图形尺寸
        post_process=None,
        y_labelpad=6
):
    """
    绘制5x5布局的经济收益图（25张子图）
    all_dfs: dict，key为title_names，value为df
    title_map: dict，key为title_names，value为中文/显示名
    colors: list，每个区域的颜色
    output_path: 输出路径
    summary_ylim: (ymin, ymax) Y轴范围元组
    bbox_to_anchor: 图例位置参数
    stacked_area_pos_neg: 区域图绘制函数(ax, df, colors, title_name, ylabel, show_legend)
    draw_legend: 图例绘制函数(ax, bbox_to_anchor, ncol, column_spacing)
    ylabel: Y轴标签
    """
    all_dfs = filter_and_rename_dict_keys(all_dfs, title_map)
    fig = plt.figure(figsize=figsize)
    # 5行5列的网格布局
    gs = gridspec.GridSpec(5, 5, figure=fig, hspace=0.15, wspace=0.15)
    fig.subplots_adjust(left=0.06, right=0.97, top=0.97, bottom=0.05)  # 底部留出空间给图例

    axes = []
    # 创建25个子图，所有子图共享y轴
    for row in range(5):
        for col in range(5):
            if row == 0 and col == 0:
                axes.append(fig.add_subplot(gs[row, col]))
            else:
                axes.append(fig.add_subplot(gs[row, col], sharey=axes[0]))

    legend_handles, legend_labels = None, None

    title_names = list(all_dfs.keys())
    y_ticks_all = get_y_axis_ticks(summary_ylim[0], summary_ylim[1],
                                   desired_ticks=desired_ticks)

    for i, (ax, title_key) in enumerate(zip(axes, title_names)):
        df = all_dfs[title_key]
        df = df[df.index >= config.START_YEAR]
        title = get_partial_match_title(title_key, title_map)

        # 确定当前子图的行列位置
        row = i // 5
        col = i % 5

        stacked_area_pos_neg(
            ax, df, colors=colors,
            title_name=title,
            ylabel = ylabel if (row == 2 and col == 0) else '',
            y_ticks_all=y_ticks_all,
            total_name=total_name,
            show_legend=False,
            dividing_line=dividing_line,
            y_labelpad=y_labelpad
        )

        x_data = df.index
        x_min, x_max = x_data.min(), x_data.max()
        tick_positions = list(range(int(x_min), int(x_max) + 1, 5))
        tick_positions = [year for year in tick_positions if year in x_data]

        ax.set_xticks(tick_positions)
        ax.tick_params(axis='x', labelrotation=45)

        # 只有最后一行（第5行，索引20-24）显示x轴标签
        if row == 4:
            ax.tick_params(axis='x')
        else:
            ax.tick_params(axis='x', labelbottom=False)

        # 只有第一列（col==0）保留y轴标签
        if col != 0:
            ax.tick_params(axis='y', labelleft=False)

        ax.figure.canvas.draw()

        # 对最后一行除第一列外的子图：隐藏x轴最左边的刻度标签
        if row == 4 and col != 0:
            xtlabs = ax.get_xticklabels()
            if xtlabs:
                xtlabs[0].set_visible(False)

        # 保存第一个子图的图例句柄和标签
        if i == 0:
            legend_handles, legend_labels = ax.get_legend_handles_labels()

    if post_process:
        axes = post_process(axes)

    # 在底部绘制图例
    if legend_handles:
        draw_legend(axes[-1], bbox_to_anchor=bbox_to_anchor, ncol=ncol,
                    column_spacing=column_spacing, ghost_legend_num=ghost_legend_num)

    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))
    output_path = os.path.join(output_path)
    fig.savefig(output_path, dpi=300)
    fig.show()
    print(f"✅ Saved to {output_path}")

def get_colors(merged_dict, mapping_file, sheet_name=None, total_name='Total'):
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
        if total_name in df.columns:
            total_col = df[total_name]
            df = df.drop(columns=[total_name])
        processed_df, legend_colors = process_single_df(df, mapping_df)
        # 合并 Total 列到最后
        if total_col is not None:
            processed_df[total_name] = total_col
            # 强制 Total 为最后一列
            cols = [c for c in processed_df.columns if c != total_name] + [total_name]
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

def get_partial_match_title(title_key, title_map):
    """
    如果 title_map 的某个 key 是 title_key 的一部分，则返回对应的 value；
    否则返回 title_key 本身。
    """
    for k, v in title_map.items():
        if k in title_key:
            return v
    return title_key

def plot_22_layout(
    all_dfs: dict,
    title_map: dict,
    colors: list,
    output_path: str,
    summary_ylim: tuple,
    desired_ticks: int = 5,
    total_name=None,
    bbox_to_anchor=(0.44, 0.95),
    stacked_area_pos_neg=stacked_area_pos_neg,
    draw_legend=draw_legend,
    ylabel: str = 'Cost (Billion AU$)',
    dividing_line=0.5,
    ncol=1,
    column_spacing=1,
    figsize=(22, 24)
):
    """
    绘制5x5布局的22张图。all_dfs为dict，key为图类型+序号（如'carbon_0'），value为df；
    title_map为dict，key同上，value为标题。
    第一行前两列画图，第3-5列合并为图例。其余行每行5张。
    """
    all_dfs = filter_and_rename_dict_keys(all_dfs, title_map, strict_match=True)
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(5, 5, figure=fig,
                           left=0.06, right=0.98, top=0.95, bottom=0.05,
                           wspace=0.15, hspace=0.14)

    axes = np.empty((5, 5), dtype=object)
    share_col = [None] * 5  # 每列共享 y 轴

    y_ticks_all = get_y_axis_ticks(summary_ylim[0], summary_ylim[1],desired_ticks)
    items = list(all_dfs.items())
    data_idx = 0
    for r in range(5):
        for c in range(5):
            # 跳过第一行后三个，用来放图例
            if r == 0 and c >= 2:
                continue

            # 防止越界（也方便你调试数量是否匹配 22）
            if data_idx >= len(items):
                break  # 或者 raise / continue，看你的需求

            title_key, df = items[data_idx]
            df = df[df.index >= config.START_YEAR]
            data_idx += 1

            title = get_partial_match_title(title_key, title_map)

            # 建轴（同列 sharey）
            ax = fig.add_subplot(gs[r, c], sharey=share_col[c])
            axes[r, c] = ax
            if share_col[c] is None:
                share_col[c] = ax

            # ===== 绘图逻辑 =====
            stacked_area_pos_neg(
                ax, df, colors=colors,
                title_name=title,
                y_ticks_all=y_ticks_all,
                total_name=total_name,
                ylabel=(ylabel if (c == 0 and r == 2) else ''),
                show_legend=False,
                dividing_line=dividing_line
            )

            x_data = df.index
            x_min, x_max = x_data.min(), x_data.max()
            tick_positions = list(range(int(x_min), int(x_max) + 1, 5))
            tick_positions = [year for year in tick_positions if year in x_data]
            ax.set_xticks(tick_positions)
            # start_tick, middle_tick, end_tick = x_data.min(), x_data[len(x_data) // 2], x_data.max()
            # tick_positions = [start_tick, middle_tick, end_tick]
            # ax.set_xticks(tick_positions)
            ax.tick_params(axis='x', labelrotation=45)
            # 设置 x 轴刻度
            if r == 4:
                ax.tick_params(axis='x', labelbottom=True)
            else:
                ax.tick_params(axis='x', labelbottom=False)

            # 左列保留 y 轴标签，其余列隐藏
            if c == 0:
                ax.tick_params(axis='y', labelleft=True)
            else:
                ax.tick_params(axis='y', labelleft=False)
            ax.figure.canvas.draw()

            #
            if r==4 and  c > 0:
                xtlabs = ax.get_xticklabels()
                if xtlabs:
                    xtlabs[0].set_visible(False)  # 隐藏第一个（最左）

            # # 仅对 8：隐藏 y 轴“最下面”的刻度标签（保留刻度线）
            # if r == 4 and c == 0:
            #     ytlabs = ax.get_yticklabels()
            #     if ytlabs:
            #         ytlabs[0].set_visible(False)  # 隐藏第一个（最下）

    # 在第一行右侧三格创建一个“空轴”做图例面板（跨 3 列）
    legend_ax = fig.add_subplot(gs[0, 2:5])
    legend_ax.axis('off')


    # # --- 全局Y轴标题 ---
    # first_ax = axes[0,0]  # 第一个 subplot
    # pos = first_ax.get_position(fig)  # Bbox, figure坐标
    # x = pos.x0 - 0.12  # 比左侧再左一点，0.02可以自己试
    # x = max(x, 0)  # 防止越界到负数
    #
    # fig.text(x, 0.5, ylabel, va='center', rotation='vertical')

    # --- 图例 ---
    draw_legend(axes[0,0], bbox_to_anchor=bbox_to_anchor, ncol=ncol, column_spacing=column_spacing)
    fig.subplots_adjust(left=0.05, right=0.98, top=0.97, bottom=0.03)
    # --- 保存 ---
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))
    fig.savefig(output_path, dpi=300)
    fig.show()
    plt.close(fig)
    print(f"✅ Saved to {output_path}")

def get_y_axis_ticks(min_value, max_value, desired_ticks=5):
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
        return (0, 1, [0,0.5,1])

    # 2. 一次性计算间隔
    ideal_interval = range_value / (desired_ticks - 1)
    # 根据理想间隔选择“nice”间隔
    e = math.floor(math.log10(ideal_interval))  # 计算数量级
    base = 10 ** e
    normalized_interval = ideal_interval / base

    # 定义“nice”间隔选项
    nice_intervals = [1, 2, 5, 10]
    # 选择最接近理想间隔的“nice”值
    interval = min(nice_intervals, key=lambda x: abs(x - normalized_interval)) * base

    # 3. 整合计算，减少中间变量
    min_tick = math.floor(min_value / interval) * interval
    max_tick = math.ceil(max_value / interval) * interval

    # 4. 使用numpy直接生成数组，避免Python列表操作
    tick_count = int((max_tick - min_tick) / interval) + 1
    ticks = np.linspace(min_tick, max_tick, tick_count)

    if len(ticks) > desired_ticks+1:
        # 例如 9→想压到 5：scale = ceil((9-1)/(5-1)) = 2
        scale = math.ceil((len(ticks) - 1) / (desired_ticks - 1))
        interval *= scale
        # 重新对齐边界并重算 ticks
        min_tick = math.floor(min_value / interval) * interval
        max_tick = math.ceil(max_value / interval) * interval
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
        # min_v = 0
        # max_v = 100

    return (min_v, max_v, ticks.tolist())

def draw_12_price(
    df,
    title_map,
    color,
    output_path,
    desired_ticks=5,
    ylabel_carbon="Shadow carbon price under net-zero targets\n(AU\$ tCO$_2$e$^{-1}$ yr$^{-1}$)",
    ylabel_bio="Biodiversity cost\n(AU\$ contribution-weighted area ha$^{-1}$ yr$^{-1}$)",
    figsize=(24, 16),
    ci=95,
):
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(3, 5, figure=fig, hspace=0.15, wspace=0.15)
    fig.subplots_adjust(left=0.06, right=0.97, top=0.95, bottom=0.05)

    # ------ x轴刻度（所有数据统一） ------
    x_data = df.index
    x_min, x_max = x_data.min(), x_data.max()
    # x_middle = x_data[int(len(x_data) // 2)]
    # tick_positions = [x_min, x_middle, x_max]


    tick_positions = list(range(int(x_min), int(x_max) + 1, 5))
    tick_positions = [year for year in tick_positions if year in x_data]

    # ------ Carbon图（第一行前两个） ------
    carbon_y = np.concatenate([df.iloc[:, i].values for i in range(2)])
    y_carbon_all = get_y_axis_ticks(0, np.nanmax(carbon_y), desired_ticks=desired_ticks)
    ax_carbon_list = []
    for i in range(2):
        ax = fig.add_subplot(gs[0, i])
        df_input = df.iloc[:, i].to_frame()
        draw_fit_line_ax(
            ax, df_input, color=color, title_name=get_partial_match_title(df.columns[i], title_map),ci=ci
        )

        ax.set_ylim(y_carbon_all[0],y_carbon_all[1])
        ax.set_yticks(y_carbon_all[2])
        ax.set_xticks(tick_positions)
        ax.tick_params(axis='x', labelbottom=False)
        if i != 0:
            ax.tick_params(axis='y', labelleft=False)
        ax_carbon_list.append(ax)

    # ------ 图例区 ------
    legend_ax = fig.add_subplot(gs[0, 2:])
    legend_ax.axis('off')

    # ------ Bio图（后两行） ------
    bio_y = np.concatenate([df.iloc[:, i + 2].values for i in range(10)])
    y_bio_all = get_y_axis_ticks(0,np.nanmax(bio_y),desired_ticks=desired_ticks-2)
    # bio_ylim = (y_bio_all[0], y_bio_all[1])
    ax_bio_list = []
    for i in range(10):
        row, col = i // 5 + 1, i % 5
        ax = fig.add_subplot(gs[row, col])
        df_input = df.iloc[:, i + 2].to_frame()
        draw_fit_line_ax(
            ax, df_input, color=color, title_name=get_partial_match_title(df.columns[i + 2], title_map),ci=ci)
        ax.set_ylim(y_bio_all[0], y_bio_all[1])
        ax.set_yticks(y_bio_all[2])
        ax.set_xticks(tick_positions)
        ax.tick_params(axis='x')

        if col != 0:
            ax.tick_params(axis='y', labelleft=False)
        if row != 2:
            ax.tick_params(axis='x', labelbottom=False)


        # if row == 2 and col == 0:
        #     ytlabs = ax.get_yticklabels()
        #     if ytlabs:
        #         ytlabs[0].set_visible(False)
        # if row == 2 and col > 0:
        #     xtlabs = ax.get_xticklabels()
        #     if xtlabs:
        #         xtlabs[0].set_visible(False)


        ax_bio_list.append(ax)

    # ------ Y轴标签 ------
    ax_carbon_list[0].set_ylabel(ylabel_carbon)
    x0, y0 = ax_carbon_list[0].yaxis.get_label().get_position()
    ax_carbon_list[0].yaxis.set_label_coords(-0.19, y0)

    ax_bio_list[0].set_ylabel(ylabel_bio)
    ax_bio_list[0].yaxis.set_label_coords(-0.19, -0.02)

    # ------ 图例 ------
    if draw_legend:
        line_handle = mlines.Line2D([], [], color=color, linewidth=2, label="Quadratic fit")
        if ci is not None and ci > 0:
            shade_handle = Patch(color=color, alpha=0.25, label="95% CI")
            leg = fig.legend(
                handles=[line_handle, shade_handle],
                loc='upper center',  # 位置类型：可选 'upper center', 'lower right', etc.
                bbox_to_anchor=(0.55, 0.85),  # (x, y) 坐标，单位是整个图的比例（0~1）
                ncol=1,  # 两列
                frameon=False  # 不显示边框
            )
        else:
            leg = fig.legend(
                handles=[line_handle],
                loc='upper center',  # 位置类型：可选 'upper center', 'lower right', etc.
                bbox_to_anchor=(0.55, 0.85),  # (x, y) 坐标，单位是整个图的比例（0~1）
                ncol=1,  # 两列
                frameon=False  # 不显示边框
            )

        leg.get_frame().set_facecolor('none')
        leg.get_frame().set_edgecolor('none')

    plt.savefig(output_path, dpi=300)
    plt.show()
    plt.close(fig)

# def gamplot(
#     df, x, y, ax,
#     color=None,
#     scatter_kws=None,
#     line_kws=None,
#     ci=95,                 # 与 seaborn 一样用“百分数”，例如 95
#     n_splines=25,          # 样条个数：越大越弯
#     spline_order=3,        # 样条阶数：3 表示三次样条
#     lam='auto'             # 平滑惩罚：'auto' 用网格搜索；或传入 float
# ):
#     scatter_kws = scatter_kws or {}
#     line_kws = line_kws or {}
#
#     X = df[[x]].to_numpy()   # shape (n, 1)
#     yv = df[y].to_numpy()
#
#     # 1) 先画散点（相当于 regplot 的 scatter_kws）
#     ax.scatter(X.ravel(), yv, color=color, **scatter_kws)
#
#     # 2) 拟合 GAM（线性高斯情形）
#     gam = LinearGAM(s(0, n_splines=n_splines, spline_order=spline_order))
#     if lam == 'auto':
#         # 简单网格搜索平滑系数
#         lam_grid = np.logspace(-3, 3, 7)
#         gam.gridsearch(X, yv, lam=lam_grid, progress=False)
#     else:
#         gam.lam = lam
#         gam.fit(X, yv)
#
#     # 3) 生成平滑曲线和置信区间
#     xx = np.linspace(X.min(), X.max(), 400)[:, None]
#     yy = gam.predict(xx)
#
#     ax.plot(xx.ravel(), yy, color=color, **line_kws)
#
#     if ci is not None and ci > 0:
#         width = ci / 100.0
#         lo, hi = gam.prediction_intervals(xx, width=width).T
#         ax.fill_between(xx.ravel(), lo, hi, alpha=0.2, color=color, linewidth=0)
#
#     return gam

def gamplot(
    df, x, y, ax,
    color=None,
    scatter_kws=None,
    line_kws=None,
    ci=95,
    n_splines=25,
    spline_order=3,
    lam='auto'
):
    scatter_kws = scatter_kws or {}
    line_kws = line_kws or {}

    X = df[[x]].to_numpy()   # shape (n, 1)
    yv = df[y].to_numpy()

    # 1) 散点
    ax.scatter(X.ravel(), yv, color=color, **scatter_kws)

    # 2) GAM 拟合（线性高斯）
    gam = LinearGAM(s(0, n_splines=n_splines, spline_order=spline_order))
    if lam == 'auto':
        lam_grid = np.logspace(-3, 3, 7)
        gam.gridsearch(X, yv, lam=lam_grid, progress=False)
    else:
        gam.lam = lam
        gam.fit(X, yv)

    # 3) 曲线 + 置信区间
    xx = np.linspace(X.min(), X.max(), 400)[:, None]
    yy = gam.predict(xx)
    ax.plot(xx.ravel(), yy, color=color, **line_kws)

    if ci is not None and ci > 0:
        width = ci / 100.0
        lo, hi = gam.prediction_intervals(xx, width=width).T
        ax.fill_between(xx.ravel(), lo, hi, alpha=0.2, color=color, linewidth=0)

    # === NEW: 计算并打印整体 p 值 ===
    p = None
    stats = getattr(gam, "statistics_", None)
    # 某些版本提供项级 p 值（通常 index 1 是第一个平滑项）
    if isinstance(stats, dict) and "p_values" in stats:
        try:
            pv = np.atleast_1d(stats["p_values"])
            p = float(pv[1]) if pv.size > 1 else float(pv[0])
        except Exception:
            p = None
    # 否则做空模型 vs GAM 的近似 F 检验
    if p is None:
        yhat = gam.predict(X)
        rss1 = np.sum((yv - yhat)**2)
        rss0 = np.sum((yv - yv.mean())**2)
        n = len(yv)
        edof_per = stats.get("edof_per_term", None) if isinstance(stats, dict) else None
        edof = stats.get("edof", None) if isinstance(stats, dict) else None
        if edof_per is not None:
            df1 = float(np.sum(edof_per))
        elif edof is not None:
            df1 = float(edof)
        else:
            df1 = 1.0 + getattr(gam.terms[1], "n_coeffs", 1)
        df0 = 1.0
        df_num = max(1.0, df1 - df0)
        df_den = max(1.0, n - df1)
        denom = rss1 / df_den
        if denom > 0:
            F = ((rss0 - rss1) / df_num) / denom
            if np.isfinite(F):
                try:
                    from scipy.stats import f
                    p = float(f.sf(F, df_num, df_den))
                except Exception:
                    p = None
    print(f"[GAM] overall p = {p:.3g}" if p is not None else "[GAM] overall p = None")

    return gam
def draw_fit_line_ax(ax, df, color='black', title_name='', order=2, ci=95):
    """在指定ax上画拟合线，y轴自动"""
    scatter_kws = dict(s=18)
    line_kws = dict(linewidth=2)
    if isinstance(df, pd.Series):
        x = df.index.values
        y = df.values
        y_colname = df.name or "value"
    else:
        x = df.index.values
        y_colname = df.columns[0]
        y = df.iloc[:, 0].values
    df_plot = pd.DataFrame({"x": x, "y": y})
    # sns.regplot(
    #     data=df_plot,
    #     x="x",
    #     y="y",
    #     order=order,
    #     ci=ci,
    #     scatter_kws=scatter_kws,
    #     line_kws=line_kws,
    #     color=color,
    #     ax=ax
    # )
    gam = gamplot(
        df_plot, x="x", y="y", ax=ax,
        color=color,
        scatter_kws=scatter_kws,
        line_kws=line_kws,
        ci= ci,  # 与 seaborn 一致；None 表示不画 CI
        n_splines=8,
        spline_order=3,
        lam='auto'
    )
    ax.set_title(title_name, pad=6)
    ax.set_xlabel('')
    ax.set_ylabel('')
    for spine in ax.spines.values():
        spine.set_color('black')
        spine.set_linewidth(1.2)

# def draw_10_price(
#     df2,
#     title_map,
#     color,
#     output_path,
#     desired_ticks=4,
#     ylabel=r"Carbon price for GHG and biodiversity (AU\$ tCO$_2$e$^{-1}$ yr$^{-1}$)",
#     figsize=(24, 10),
#     legend_label_line="Nature-positive targets",
#     legend_label_shade="95% CI",
#     legend_loc="best",
#     legend_on_first_ax=True,
#     ylabel_pos=(-0.3, -0.2),  # (x, y) in axes coords
#     top_space_ratio=0.20,     # y 轴顶部额外空间比例
#     ci=95,
# ):
#     """
#     画两行共 10 张子图（df2 的 10 列），每张做拟合并统一坐标与格式。
#     依赖外部函数：draw_fit_line_ax(ax, df_input, color, title_name)
#     """
#     fig = plt.figure(figsize=figsize)
#     gs = gridspec.GridSpec(2, 5, figure=fig, hspace=0.15, wspace=0.15)
#     fig.subplots_adjust(left=0.06, right=0.97, top=0.95, bottom=0.05)
#
#     # ---- 统一 x 轴刻度 ----
#     x_data = df2.index
#     x_min, x_max = x_data.min(), x_data.max()
#     # x_middle = x_data[int(len(x_data) // 2)]
#     # tick_positions = [x_min, x_middle, x_max]
#
#     tick_positions = list(range(int(x_min), int(x_max) + 1, 5))
#     tick_positions = [year for year in tick_positions if year in x_data]
#
#     # ---- 统一 y 轴范围（根据全部10列）----
#     bio_y = np.concatenate([df2.iloc[:, i].values for i in range(10)])
#     bio_ymax = np.nanmax(bio_y)
#     y0, y1 = 0.0, float(bio_ymax) * (1.0 + top_space_ratio)
#
#     # 也可用你自己的 get_y_axis_ticks：
#     # y_min, y_max, ticks = get_y_axis_ticks(0, np.nanmax(bio_y), desired_ticks=desired_ticks)
#
#     def _int_fmt(x, pos):
#         return f"{int(x)}"
#     int_formatter = FuncFormatter(_int_fmt)
#
#     axes = []
#     for i in range(10):
#         row, col = divmod(i, 5)
#         ax = fig.add_subplot(gs[row, col])
#         df_input = df2.iloc[:, i].to_frame()
#
#         # 你的拟合画线函数（外部提供）
#         draw_fit_line_ax(
#             ax,
#             df_input,
#             color=color,
#             title_name=get_partial_match_title(df2.columns[i], title_map),
#             ci=ci
#         )
#
#         # y 轴范围与刻度
#         ax.set_ylim(y0, y1)
#         ax.yaxis.set_major_locator(MaxNLocator(nbins=desired_ticks, integer=True))
#         ax.yaxis.set_major_formatter(int_formatter)
#
#         # x 轴刻度
#         ax.set_xticks(tick_positions)
#         ax.tick_params(axis='x')
#
#         # 非首列不显示 y 轴刻度文本
#         if col != 0:
#             ax.tick_params(axis='y', labelleft=False)
#         if row != 1:
#             ax.tick_params(axis='x', labelbottom=False)
#
#         # if row == 1 and col == 0:
#         #     ytlabs = ax.get_yticklabels()
#         #     if ytlabs:
#         #         ytlabs[0].set_visible(False)
#         # if row == 1 and col > 0:
#         #     xtlabs = ax.get_xticklabels()
#         #     if xtlabs:
#         #         xtlabs[0].set_visible(False)
#
#         axes.append(ax)
#
#     # y 轴标签放在第一个子图上，并上移
#     axes[0].set_ylabel(ylabel)
#     axes[0].yaxis.set_label_coords(-0.19, -0.03)
#
#     # legend：默认只在第一个子图里放，避免重复
#     if legend_on_first_ax:
#         line_handle = mlines.Line2D([], [], color=color, linewidth=2, label=legend_label_line)
#         if ci is not None and ci > 0:
#             shade_handle = Patch(color=color, alpha=0.25, label=legend_label_shade)
#             axes[0].legend(handles=[line_handle, shade_handle], loc=legend_loc, frameon=False)
#         else:
#             axes[0].legend(handles=[line_handle], loc=legend_loc, frameon=False)
#
#     plt.savefig(output_path, dpi=300)
#     plt.show()
#     return fig, axes

# def draw_22_price(
#     df,
#     title_map,
#     output_path,
#     desired_ticks=5,
#     y_label="Shadow carbon price",
#     figsize=(36, 40),
#     ci=95,
# ):
#     fig = plt.figure(figsize=figsize)
#     gs = gridspec.GridSpec(5, 5, figure=fig, hspace=0.15, wspace=0.15)
#     fig.subplots_adjust(left=0.06, right=0.97, top=0.95, bottom=0.05)
#
#     # ------ x轴刻度（所有数据统一） ------
#     x_data = df.index
#     x_min, x_max = x_data.min(), x_data.max()
#     # x_middle = x_data[int(len(x_data) // 2)]
#     # tick_positions = [x_min, x_middle, x_max]
#
#
#     tick_positions = list(range(int(x_min), int(x_max) + 1, 5))
#     tick_positions = [year for year in tick_positions if year in x_data]
#
#     # ------ Carbon图（第一行前两个） ------
#     carbon_y = np.concatenate([df.iloc[:, i].values for i in range(2)])
#     y_carbon_all = get_y_axis_ticks(0, np.nanmax(carbon_y), desired_ticks=desired_ticks)
#     ax_list = []
#     for i in range(2):
#         ax = fig.add_subplot(gs[0, i])
#         df_input = df.iloc[:, i].to_frame()
#         draw_fit_line_ax(
#             ax, df_input, color='black', title_name=get_partial_match_title(df.columns[i],title_map),ci=ci
#         )
#
#         ax.set_ylim(y_carbon_all[0],y_carbon_all[1])
#         ax.set_yticks(y_carbon_all[2])
#         ax.set_xticks(tick_positions)
#         ax.tick_params(axis='x', labelbottom=False)
#         # if i != 0:
#         #     ax.tick_params(axis='y', labelleft=False)
#         ax_list.append(ax)
#
#     # ------ carbon price图（第2,3,4行） ------
#     # 拼接两段需要画的所有数据
#     bio_y = np.concatenate([
#         # 第2、3行
#         np.concatenate([df.iloc[:, i + 2].values for i in range(10)]),
#         # 第3、4行
#         np.concatenate([df.iloc[:, i + 7].values for i in range(10)])
#     ])
#     # 使用全量数据获取 y 轴范围和刻度
#     y_bio_all = get_y_axis_ticks(0, np.nanmax(bio_y), desired_ticks=desired_ticks - 2)
#
#     # ----- 画第2、3行 -----
#     for i in range(2, 12):
#         row, col = (i - 2) // 5 + 1, (i - 2) % 5
#         ax = fig.add_subplot(gs[row, col])
#         df_input = df.iloc[:, i].to_frame()
#         draw_fit_line_ax(
#             ax, df_input, color='orange', title_name=get_partial_match_title(df.columns[i],title_map), ci=ci)
#         ax.set_ylim(y_bio_all[0], y_bio_all[1])
#         ax.set_yticks(y_bio_all[2])
#         ax.set_xticks(tick_positions)
#         ax.tick_params(axis='x', labelbottom=False)
#         ax_list.append(ax)
#
#     # ----- 画第3、4行 -----
#     for i in range(12, 22):
#         row, col = (i - 2) // 5 + 1, (i - 2) % 5
#         ax = fig.add_subplot(gs[row, col])
#         df_input = df.iloc[:, i].to_frame()
#         draw_fit_line_ax(
#             ax, df_input, color='purple', title_name=get_partial_match_title(df.columns[i],title_map), ci=ci)
#         ax.set_ylim(y_bio_all[0], y_bio_all[1])  # 用统一的 y_bio_all
#         ax.set_yticks(y_bio_all[2])
#         ax.set_xticks(tick_positions)
#
#         if row == 4:
#             ax.tick_params(axis='x', labelbottom=True)
#         else:
#             ax.tick_params(axis='x', labelbottom=False)
#
#         ax_list.append(ax)
#
#     ax_list[0].set_ylabel(y_label)
#     ax_list[0].yaxis.set_label_coords(-0.19, -1.8)
#
#
#     # ------ 图例 ------
#     if draw_legend:
#         # 定义legend句柄
#         handle_carbon = mlines.Line2D([], [], color='black', linewidth=2, label="Net-zero targets")
#         handle_bio12 = mlines.Line2D([], [], color='orange', linewidth=2, label="Net-zero targets and nature-positive targets")
#         handle_bio10 = mlines.Line2D([], [], color='purple', linewidth=2, label="Nature-positive targets")
#
#         shade_carbon = Patch(color='black', alpha=0.25, label="95% CI")
#         shade_bio12 = Patch(color='orange', alpha=0.25, label="95% CI")
#         shade_bio10 = Patch(color='purple', alpha=0.25, label="95% CI")
#
#         handles = [handle_carbon, shade_carbon, handle_bio12, shade_bio12, handle_bio10, shade_bio10]
#
#         fig.legend(
#             handles=handles,
#             loc='upper center',  # 可选: 'lower center', 'upper left', etc.
#             bbox_to_anchor=(0.6, 0.95),  # (x, y)，1.04可以让图例在图像上方
#             ncol=1,  # 每行3个图例（你有3组）
#             frameon=False
#         )
#
#     plt.savefig(output_path, dpi=300)
#     plt.show()
#     plt.close(fig)

def filter_and_rename_dict_keys(data_dict, title_map, strict_match=False):
    """
    只保留 data_dict 的 key 满足 title_map 某个 key 的项，
    并将 key改成 title_map 的 value（格式化名字）。
    strict_match=True 时，必须完全匹配；否则为包含（模糊）匹配。
    匹配不到时直接 raise KeyError。
    """
    new_dict = {}
    matched_keys = set()
    for orig_key, display_name in title_map.items():
        found = False
        for k in data_dict:
            if (strict_match and k == orig_key) or (not strict_match and orig_key in k):
                new_dict[display_name] = data_dict[k]
                matched_keys.add(orig_key)
                found = True
                break
        if not found:
            raise KeyError(f"No match found for key: {orig_key}")
    return new_dict


def draw_22_price(
        df_long,
        title_map,
        output_path,
        start_year,
        desired_ticks=5,
        y_label="Shadow carbon price",
        figsize=(36, 40),
        ci=95,
):
    """
    绘制22个价格趋势图（5行5列布局，前3个空位）

    参数:
    df_long: 长格式DataFrame，包含列 ['scenario', 'Year', 'data']
    title_map: dict，key为scenario名称，value为显示标题
    output_path: 输出路径
    start_year: 起始年份，只使用大于此年份的数据
    desired_ticks: y轴刻度数量
    y_label: y轴标签
    figsize: 图形尺寸
    ci: 置信区间
    """
    # 筛选年份
    df_filtered = df_long[df_long['Year'] > start_year].copy()

    # 按照title_map的顺序获取scenario列表
    scenario_list = list(title_map.keys())

    # 将长格式转换为宽格式，保持scenario顺序
    df_pivot = df_filtered.pivot(index='Year', columns='scenario', values='data')
    df = df_pivot[scenario_list]  # 按照title_map顺序重新排列列

    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(5, 5, figure=fig, hspace=0.15, wspace=0.15)
    fig.subplots_adjust(left=0.06, right=0.97, top=0.95, bottom=0.05)

    # ------ x轴刻度（所有数据统一） ------
    x_data = df.index
    x_min, x_max = x_data.min(), x_data.max()
    tick_positions = list(range(int(x_min), int(x_max) + 1, 5))
    tick_positions = [year for year in tick_positions if year in x_data]

    # ------ Carbon图（第一行前两个） ------
    carbon_y = np.concatenate([df.iloc[:, i].values for i in range(2)])
    y_carbon_all = get_y_axis_ticks(0, np.nanmax(carbon_y), desired_ticks=desired_ticks)

    ax_list = []
    for i in range(2):
        ax = fig.add_subplot(gs[0, i])
        df_input = df.iloc[:, i].to_frame()
        scenario_name = df.columns[i]
        display_title = title_map.get(scenario_name, scenario_name)

        draw_fit_line_ax(
            ax, df_input, color='black', title_name=display_title, ci=ci
        )
        ax.set_ylim(y_carbon_all[0], y_carbon_all[1])
        ax.set_yticks(y_carbon_all[2])
        ax.set_xticks(tick_positions)
        ax.tick_params(axis='x', labelbottom=False)
        ax_list.append(ax)

    # ------ carbon price图（第2,3,4行） ------
    # 拼接两段需要画的所有数据
    bio_y = np.concatenate([
        # 第2、3行
        np.concatenate([df.iloc[:, i + 2].values for i in range(10)]),
        # 第3、4行
        np.concatenate([df.iloc[:, i + 7].values for i in range(10)])
    ])
    # 使用全量数据获取 y 轴范围和刻度
    y_bio_all = get_y_axis_ticks(0, np.nanmax(bio_y), desired_ticks=desired_ticks - 2)

    # ----- 画第2、3行 -----
    for i in range(2, 12):
        row, col = (i - 2) // 5 + 1, (i - 2) % 5
        ax = fig.add_subplot(gs[row, col])
        df_input = df.iloc[:, i].to_frame()
        scenario_name = df.columns[i]
        display_title = title_map.get(scenario_name, scenario_name)

        draw_fit_line_ax(
            ax, df_input, color='orange', title_name=display_title, ci=ci)
        ax.set_ylim(y_bio_all[0], y_bio_all[1])
        ax.set_yticks(y_bio_all[2])
        ax.set_xticks(tick_positions)
        ax.tick_params(axis='x', labelbottom=False)
        ax_list.append(ax)

    # ----- 画第3、4行 -----
    for i in range(12, 22):
        row, col = (i - 2) // 5 + 1, (i - 2) % 5
        ax = fig.add_subplot(gs[row, col])
        df_input = df.iloc[:, i].to_frame()
        scenario_name = df.columns[i]
        display_title = title_map.get(scenario_name, scenario_name)

        draw_fit_line_ax(
            ax, df_input, color='purple', title_name=display_title, ci=ci)
        ax.set_ylim(y_bio_all[0], y_bio_all[1])  # 用统一的 y_bio_all
        ax.set_yticks(y_bio_all[2])
        ax.set_xticks(tick_positions)
        if row == 4:
            ax.tick_params(axis='x', labelbottom=True)
        else:
            ax.tick_params(axis='x', labelbottom=False)
        ax_list.append(ax)

    ax_list[0].set_ylabel(y_label)
    ax_list[0].yaxis.set_label_coords(-0.19, -1.8)

    # 保存图形
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))
    fig.savefig(output_path, dpi=300)
    fig.show()
    print(f"✅ Saved to {output_path}")


def draw_10_price(
        df_long,
        title_map,
        color,
        output_path,
        start_year,
        desired_ticks=4,
        ylabel=r"Carbon price for GHG and biodiversity (AU\$ tCO$_2$e$^{-1}$ yr$^{-1}$)",
        figsize=(24, 10),
        legend_label_line="Nature-positive targets",
        legend_label_shade="95% CI",
        legend_loc="best",
        legend_on_first_ax=True,
        ylabel_pos=(-0.3, -0.2),  # (x, y) in axes coords
        top_space_ratio=0.20,  # y 轴顶部额外空间比例
        ci=95,
):
    """
    画两行共 10 张子图，每张做拟合并统一坐标与格式。

    参数:
    df_long: 长格式DataFrame，包含列 ['scenario', 'Year', 'data']
    title_map: dict，key为scenario名称，value为显示标题
    color: 线条颜色
    output_path: 输出路径
    start_year: 起始年份，只使用大于此年份的数据
    desired_ticks: y轴刻度数量
    ylabel: y轴标签
    figsize: 图形尺寸
    legend_label_line: 图例线条标签
    legend_label_shade: 图例阴影标签
    legend_loc: 图例位置
    legend_on_first_ax: 是否只在第一个子图显示图例
    ylabel_pos: y轴标签位置
    top_space_ratio: y轴顶部额外空间比例
    ci: 置信区间
    """
    # 筛选年份
    df_filtered = df_long[df_long['Year'] > start_year].copy()

    # 按照title_map的顺序获取scenario列表
    scenario_list = list(title_map.keys())

    # 将长格式转换为宽格式，保持scenario顺序
    df_pivot = df_filtered.pivot(index='Year', columns='scenario', values='data')
    df2 = df_pivot[scenario_list]  # 按照title_map顺序重新排列列

    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(2, 5, figure=fig, hspace=0.15, wspace=0.15)
    fig.subplots_adjust(left=0.06, right=0.97, top=0.95, bottom=0.05)

    # ---- 统一 x 轴刻度 ----
    x_data = df2.index
    x_min, x_max = x_data.min(), x_data.max()
    tick_positions = list(range(int(x_min), int(x_max) + 1, 5))
    tick_positions = [year for year in tick_positions if year in x_data]

    # ---- 统一 y 轴范围（根据全部10列）----
    bio_y = np.concatenate([df2.iloc[:, i].values for i in range(10)])
    bio_ymax = np.nanmax(bio_y)
    y0, y1 = 0.0, float(bio_ymax) * (1.0 + top_space_ratio)

    def _int_fmt(x, pos):
        return f"{int(x)}"

    int_formatter = FuncFormatter(_int_fmt)

    axes = []
    for i in range(10):
        row, col = divmod(i, 5)
        ax = fig.add_subplot(gs[row, col])
        df_input = df2.iloc[:, i].to_frame()
        scenario_name = df2.columns[i]
        display_title = title_map.get(scenario_name, scenario_name)

        # 你的拟合画线函数（外部提供）
        draw_fit_line_ax(
            ax,
            df_input,
            color=color,
            title_name=display_title,
            ci=ci
        )

        # y 轴范围与刻度
        ax.set_ylim(y0, y1)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=desired_ticks, integer=True))
        ax.yaxis.set_major_formatter(int_formatter)

        # x 轴刻度
        ax.set_xticks(tick_positions)
        ax.tick_params(axis='x')

        # 非首列不显示 y 轴刻度文本
        if col != 0:
            ax.tick_params(axis='y', labelleft=False)
        if row != 1:
            ax.tick_params(axis='x', labelbottom=False)

        axes.append(ax)

    # y 轴标签放在第一个子图上，并上移
    axes[0].set_ylabel(ylabel)
    axes[0].yaxis.set_label_coords(-0.19, -0.03)

    # legend：默认只在第一个子图里放，避免重复
    if legend_on_first_ax:
        line_handle = mlines.Line2D([], [], color=color, linewidth=2, label=legend_label_line)
        if ci is not None and ci > 0:
            shade_handle = Patch(color=color, alpha=0.25, label=legend_label_shade)
            axes[0].legend(handles=[line_handle, shade_handle], loc=legend_loc, frameon=False)
        else:
            axes[0].legend(handles=[line_handle], loc=legend_loc, frameon=False)

    # 保存图形
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))
    plt.savefig(output_path, dpi=300)
    plt.show()
    print(f"✅ Saved to {output_path}")

    return fig, axes