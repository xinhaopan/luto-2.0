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
from typing import List, Optional

# 1. 使用 Seaborn 设置基础网格和颜色主题
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


# Function definitions (unchanged from your original code)
def stacked_area_pos_neg(
        ax, df, colors=None, alpha=0.60,
        title_name='', ylabel='',
        add_line=True, n_col=1, show_legend=False, bbox_to_anchor=(0.5, -0.25)
):
    """
    绘制一个可以同时显示正值和负值的堆叠面积图。

    此函数假定：
    - 如果 add_line=True，则输入 DataFrame `df` 的最后一列是预先计算好的总和/合计。
    - 除了最后一列之外的所有列都是用于堆叠的子项。

    Args:
        ax: Matplotlib的Axes对象，用于在其上绘图。
        df (pd.DataFrame): 包含绘图数据的DataFrame。
        colors (list, optional): 用于堆叠区域的颜色列表。
        alpha (float, optional): 填充颜色的透明度。
        title_name (str, optional): 图表标题。
        ylabel (str, optional): Y轴标签。
        add_line (bool, optional): 是否绘制代表总和的线条（使用df的最后一列）。
        n_col (int, optional): 图例的列数。
        show_legend (bool, optional): 是否显示图例。
        bbox_to_anchor (tuple, optional): 控制图例位置的参数。
    """
    # ---- 1. 数据分离：分离堆叠项和总和项 ----
    df_stack = df.iloc[:, :-1]  # 所有列，除了最后一列
    total_col_name = df.columns[-1]
    total_col_data = df.iloc[:, -1]

    # ---- 2. 可选总和线 (使用预计算的最后一列) ----
    if add_line:
        ax.plot(
            df.index, total_col_data,  # 使用最后一列的数据
            linestyle='-', marker='o',
            color='black', linewidth=2,
            markersize=5, markeredgewidth=1,
            markerfacecolor='black', markeredgecolor='black',
            label=total_col_name  # 使用最后一列的名称作为标签
        )

    # ---- 3. 颜色管理 ----
    if colors is None:
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    if len(colors) < df_stack.shape[1]:
        # 根据堆叠项的数量来确定颜色
        num_stack_cols = df_stack.shape[1]
        colors = (colors * (num_stack_cols // len(colors) + 1))[:num_stack_cols]

    # ---- 4. 堆叠面积图 (仅使用 df_stack) ----
    cum_pos = np.zeros(len(df_stack))
    cum_neg = np.zeros(len(df_stack))
    # **关键修改**：只遍历 df_stack 的列
    for idx, col in enumerate(df_stack.columns):
        y = df_stack[col].values
        pos = np.clip(y, 0, None)
        neg = np.clip(y, None, 0)
        colr = colors[idx]
        ax.fill_between(df_stack.index, cum_pos, cum_pos + pos,
                        facecolor=colr, alpha=alpha, linewidth=0, label=col)
        cum_pos += pos
        ax.fill_between(df_stack.index, cum_neg, cum_neg + neg,
                        facecolor=colr, alpha=alpha, linewidth=0)
        cum_neg += neg

    # ---- 5. 轴和外观 ----
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

    # ---- 6. 图例处理 ----
    handles, labels = ax.get_legend_handles_labels()
    unique = {}
    for h, l in zip(handles, labels):
        if l not in unique:
            unique[l] = h

    # 确保总和线在图例的最后
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


def calculate_total_profit(df: pd.DataFrame) -> pd.DataFrame:
    """
    计算总利润并清理 DataFrame，同时将成本项表示为负数。

    此函数执行以下操作：
    1. 定义收入列和成本列。
    2. **将所有成本列的数值乘以 -1，使其变为负数。**
    3. 计算总利润 (此时等于所有收入列和成本列的总和)。
    4. 从 DataFrame 中移除不再需要的 'Transition(ag→non-ag) cost' 列。

    Args:
        df (pd.DataFrame): 包含所需收入和成本列的输入 DataFrame。

    Returns:
        pd.DataFrame: 一个新的 DataFrame，其中包含了 'Total Profit' 列，
                      所有成本列的值都为负数，并移除了指定的原始成本列。
    """
    # 创建一个副本以避免修改原始 DataFrame
    df_processed = df.copy()

    # 定义用于计算的列
    revenue_cols = ['Ag revenue', 'Agmgt revenue', 'Non-ag revenue']
    cost_cols = [
        'Ag cost', 'Agmgt cost', 'Non-ag cost',
        'Transition(ag→ag) cost', 'Transition(ag→non-ag) amortised cost'
    ]
    col_to_drop = 'Transition(ag→non-ag) cost'

    # --- 安全检查：确保所有需要的列都存在 ---
    # 在进行任何操作前先检查，确保数据完整性
    all_required_cols = revenue_cols + cost_cols + [col_to_drop]
    missing_cols = [col for col in all_required_cols if col not in df_processed.columns]
    if missing_cols:
        # 使用 f-string 提高可读性
        raise ValueError(f"DataFrame 中缺少以下必需的列: {missing_cols}")

    # --- 核心修改：将所有成本列的数值变为负数 ---
    df_processed[cost_cols] = df_processed[cost_cols] * -1

    # 定义用于最终加总的列（现在包括收入和已变负的成本）
    profit_component_cols = revenue_cols + cost_cols

    # 1. 计算总利润
    # 因为成本已为负，所以直接求和即可
    df_processed['Net economic returns'] = df_processed[profit_component_cols].sum(axis=1)

    # 2. 移除不再需要的原始列
    # drop 方法本身会返回一个新的 DataFrame，我们直接赋值回去
    df_processed = df_processed.drop(columns=[col_to_drop])

    return df_processed


def plot_lines_with_markers(
        df: pd.DataFrame,
        columns_to_plot: List[str],
        colors: List[str],
        legend_labels: Optional[List[str]] = None,  # <-- 1. 添加新参数
        xlabel: Optional[str] = None,
        ylabel: str = "Values",
        output_path: Optional[str] = None
):
    """
    从 DataFrame 中选取指定列，用指定颜色和自定义图例绘制点线图。

    Args:
        df (pd.DataFrame): 包含数据的源 DataFrame。图表的 x 轴将使用此 DataFrame 的索引。
        columns_to_plot (List[str]): 一个包含要绘制的列名的列表。
        colors (List[str]): 一个包含颜色的列表，其顺序与 columns_to_plot 对应。
        legend_labels (Optional[List[str]], optional):
            一个包含图例标签的列表。如果提供，将覆盖默认的列名标签。
            其长度必须与 columns_to_plot 一致。默认为 None。
        xlabel (Optional[str], optional): x 轴的标签。如果为 None，则会尝试使用索引的名称。
                                          默认为 None。
        ylabel (str, optional): y 轴的标签。默认为 "Values"。
        output_path (Optional[str], optional): 图片文件的保存路径 (例如 'my_plot.png')。
                                              如果为 None，则只显示图表而不保存。默认为 None。
    """
    # --- 1. 输入验证 ---
    missing_cols = [col for col in columns_to_plot if col not in df.columns]
    if missing_cols:
        raise ValueError(f"DataFrame 中缺少以下列: {missing_cols}")

    if len(columns_to_plot) != len(colors):
        raise ValueError(f"列的数量 ({len(columns_to_plot)}) 与颜色的数量 ({len(colors)}) 不匹配。")

    # --- 2. 验证新参数 ---
    if legend_labels and len(columns_to_plot) != len(legend_labels):
        raise ValueError(f"列的数量 ({len(columns_to_plot)}) 与图例标签的数量 ({len(legend_labels)}) 不匹配。")

    # --- 3. 设置图表风格和画布 ---
    fig, ax = plt.subplots(figsize=(14, 8))

    # --- 4. 循环绘图 ---
    for i, column in enumerate(columns_to_plot):
        # --- 3. 动态设置标签 ---
        label_for_legend = legend_labels[i] if legend_labels else column

        ax.plot(
            df.index,
            df[column],
            marker='o',
            linestyle='-',
            color=colors[i],
            label=label_for_legend  # <-- 使用我们定义的标签
        )

    # --- 5. 美化图表 ---
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('black')
        spine.set_linewidth(1.2)

    ax.set_xlim(df.index.min(), df.index.max())

    # 设置坐标轴标签
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=12)
    elif df.index.name:
        ax.set_xlabel(df.index.name, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)

    # 设置图例和 Y 轴格式
    ax.legend(frameon=False, loc='best')
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:,.0f}'))

    # --- 6. 保存或显示图表 ---
    if output_path:
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        plt.savefig(output_path, dpi=300)
        print(f"图表已保存至: {output_path}")

    plt.show()


# Main script
set_plot_style(font_size=24, font_family='Arial')

input_dir = f'../../../output/{config.TASK_NAME}/carbon_price/1_excel'
output_dir = f"../../../output/{config.TASK_NAME}/carbon_price/3_Paper_figure"

input_files = config.input_files

input_files_0 = config.input_files_0
input_files_1 = config.input_files_1
input_files_2 = config.input_files_2

processed_dfs = []
for input_file in input_files:
    df = pd.read_excel(f"{input_dir}/0_Origin_economic_{input_file}.xlsx", sheet_name='Sheet1', index_col=0)
    df_processed = calculate_total_profit(df)['Net economic returns']
    processed_dfs.append(df_processed)
final_df = pd.concat(processed_dfs, axis=1)
final_df.columns = input_files

plot_lines_with_markers(final_df,input_files_0+input_files_1,['red','#9b6016','#f7a800'],['Counterfactual','$\mathrm{GHG}_{\mathrm{high}}$','$\mathrm{GHG}_{\mathrm{low}}$'])
plot_lines_with_markers(final_df,[input_files_1[0]]+input_files_2[:5],['#9b6016','#007858', '#1A996D', '#40B57F', '#74C69D', '#A8D8B9'],['$\mathrm{GHG}_{\mathrm{high}}$','$\mathrm{GHG}_{\mathrm{high}}$,$\mathrm{Bio}_{\mathrm{50}}$','$\mathrm{GHG}_{\mathrm{high}}$,$\mathrm{Bio}_{\mathrm{40}}$','$\mathrm{GHG}_{\mathrm{high}}$,$\mathrm{Bio}_{\mathrm{30}}$','$\mathrm{GHG}_{\mathrm{high}}$,$\mathrm{Bio}_{\mathrm{20}}$','$\mathrm{GHG}_{\mathrm{high}}$,$\mathrm{Bio}_{\mathrm{10}}$'])
plot_lines_with_markers(final_df,[input_files_1[1]]+input_files_2[-5:],['#f7a800','#08519c', '#3182bd', '#6baed6', '#9ecae1', '#c6dbef'],['$\mathrm{GHG}_{\mathrm{low}}$','$\mathrm{GHG}_{\mathrm{low}}$,$\mathrm{Bio}_{\mathrm{50}}$','$\mathrm{GHG}_{\mathrm{low}}$,$\mathrm{Bio}_{\mathrm{40}}$','$\mathrm{GHG}_{\mathrm{low}}$,$\mathrm{Bio}_{\mathrm{30}}$','$\mathrm{GHG}_{\mathrm{low}}$,$\mathrm{Bio}_{\mathrm{20}}$','$\mathrm{GHG}_{\mathrm{low}}$,$\mathrm{Bio}_{\mathrm{10}}$'])

# --- 2. 预处理数据并计算全局 Y 轴范围 (修正版) ---
print("正在预处理数据并计算全局Y轴范围...")
all_dfs = []
global_ymin = float('inf')
global_ymax = float('-inf')

for input_file in input_files:
    df_raw = pd.read_excel(f"{input_dir}/0_Origin_economic_{input_file}.xlsx", sheet_name='Sheet1', index_col=0)  / 1e3
    df_raw = df_raw.loc[df_raw.index >= config.START_YEAR].copy()
    df_processed = calculate_total_profit(df_raw)
    df_stacked = df_processed.iloc[:, :-1]
    all_dfs.append(df_processed)

    # --- 正确的 Y 轴范围计算逻辑 ---
    # 1. 分离堆叠数据和总和线数据
    df_stack = df_stacked.iloc[:, :-1]
    total_line = df_stacked.iloc[:, -1]

    # 2. 计算每行正/负值的累积和
    # clip(lower=0) 将所有负数变为0，然后求和
    positive_sum_per_row = df_stack.clip(lower=0).sum(axis=1)
    # clip(upper=0) 将所有正数变为0，然后求和
    negative_sum_per_row = df_stack.clip(upper=0).sum(axis=1)

    # 3. 确定当前 DataFrame 的 Y 轴边界
    # 上边界是正向堆叠的最高点和总和线最高点中的较大者
    current_max = max(positive_sum_per_row.max(), total_line.max())
    # 下边界是负向堆叠的最低点和总和线最低点中的较小者
    current_min = min(negative_sum_per_row.min(), total_line.min())

    # 4. 更新全局 Y 轴范围
    global_ymin = min(global_ymin, current_min)
    global_ymax = max(global_ymax, current_max)

print(f"全局Y轴范围确定: [{global_ymin:,.0f}, {global_ymax:,.0f}]")

# --- 3. 创建复杂的子图布局 ---
fig = plt.figure(figsize=(22, 12))
gs = gridspec.GridSpec(3, 5, figure=fig, hspace=0.3, wspace=0.3)

# 创建一个列表来存放所有的 axes
axes = []
# 第一行: 3个图 + 1个图例区域
axes.append(fig.add_subplot(gs[0, 0]))
axes.append(fig.add_subplot(gs[0, 1], sharey=axes[0]))
axes.append(fig.add_subplot(gs[0, 2], sharey=axes[0]))
legend_ax = fig.add_subplot(gs[0, 3:])  # 图例专用
legend_ax.axis('off')  # 关闭图例区域的坐标轴

# 第二行: 5个图
for i in range(5):
    axes.append(fig.add_subplot(gs[1, i], sharey=axes[0]))

# 第三行: 5个图
for i in range(5):
    axes.append(fig.add_subplot(gs[2, i], sharey=axes[0]))

# --- 4. 循环绘图并进行美化 ---
colors = ['#fab431', '#ec7951', '#cd4975', '#9f0e9e', '#6200ac', '#2d688f', '#19928e', '#35b876']
title_names = ['Counterfactual',r'$\mathrm{GHG}_{\mathrm{low}}$', r'$\mathrm{GHG}_{\mathrm{high}}$',

               r'$\mathrm{GHG}_{\mathrm{low}}$,$\mathrm{Bio}_{\mathrm{10}}$',
               r'$\mathrm{GHG}_{\mathrm{low}}$,$\mathrm{Bio}_{\mathrm{20}}$',
               r'$\mathrm{GHG}_{\mathrm{low}}$,$\mathrm{Bio}_{\mathrm{30}}$',
               r'$\mathrm{GHG}_{\mathrm{low}}$,$\mathrm{Bio}_{\mathrm{40}}$',
               r'$\mathrm{GHG}_{\mathrm{low}}$,$\mathrm{Bio}_{\mathrm{50}}$',
               
               r'$\mathrm{GHG}_{\mathrm{high}}$,$\mathrm{Bio}_{\mathrm{10}}$',
               r'$\mathrm{GHG}_{\mathrm{high}}$,$\mathrm{Bio}_{\mathrm{20}}$',
               r'$\mathrm{GHG}_{\mathrm{high}}$,$\mathrm{Bio}_{\mathrm{30}}$',
               r'$\mathrm{GHG}_{\mathrm{high}}$,$\mathrm{Bio}_{\mathrm{40}}$',
               r'$\mathrm{GHG}_{\mathrm{high}}$,$\mathrm{Bio}_{\mathrm{50}}$',
               ]

legend_handles, legend_labels = None, None

for i, (ax, df, title) in enumerate(zip(axes, all_dfs, title_names)):
    stacked_area_pos_neg(
        ax, df, colors=colors,
        title_name=title,
        ylabel='',  # Y轴标签只在第一个图上手动设置
        show_legend=False  # 我们将手动创建全局图例
    )

    x_data = df.index
    start_tick, middle_tick, end_tick = x_data.min(), x_data[len(x_data) // 2], x_data.max()
    tick_positions = [start_tick, middle_tick, end_tick]

    ax.set_xticks(tick_positions)
    if i in [8, 9, 10, 11, 12]:
        # 如果是，则设置三个刻度并微调对齐方式

        ax.tick_params(axis='x')  # 在这里设置您想要的字体大小

        # 获取标签并修改对齐方式
        x_labels = ax.get_xticklabels()
        if len(x_labels) >= 3:
            x_labels[0].set_horizontalalignment('left')
            x_labels[-1].set_horizontalalignment('right')
    else:
        # **关键补充**：如果不是最下面一行，则明确地隐藏X轴的刻度标签
        ax.tick_params(axis='x', labelbottom=False)

    # 控制Y轴刻度标签的显示
    # 每行的第一个图 (索引 0, 3, 8) 显示Y轴标签
    if i not in [0, 3, 8]:
        ax.tick_params(axis='y', labelleft=False)

    # 仅为2行的第一个图设置Y轴标签
    if i in [3]:
        # 为每行第一个图设置Y轴标签
        ax.set_ylabel('Net economic returns (Billion AU$)')

    # 在第一次循环时，获取图例信息
    if i == 0:
        legend_handles, legend_labels = ax.get_legend_handles_labels()

# --- 5. 设置全局Y轴和创建统一图例 ---
# 设置统一的Y轴范围
axes[0].set_ylim(global_ymin, global_ymax)
# 1. 获取当前的位置对象
pos = legend_ax.get_position()
# pos.x0 的值大约是 0.6125
# 2. 直接指定一个新的、更小的 x0 值来实现左移
#    您可以根据需要微调这个数字
new_x0 = pos.y0 - 0.08
new_y0 = pos.x0 + 0.08
# 3. 创建新的位置列表 [left, bottom, width, height]
#    我们只改变 left (x0)，其它保持不变
new_position = [
    new_x0,  # 使用我们新指定的左边界
    new_y0,  # 使用原始的下边界
    pos.width,  # 使用原始的宽度
    pos.height  # 使用原始的高度
]
# 4. 应用这个新位置
legend_ax.set_position(new_position)

# 创建并放置图例
if legend_handles:
    draw_legend(axes[0], bbox_to_anchor=legend_ax.get_position(), ncol=2, column_spacing=-7)

# --- 6. 保存最终图表 ---
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
output_path = os.path.join(output_dir, '03_Profit.png')
plt.savefig(output_path, dpi=300)
plt.show()