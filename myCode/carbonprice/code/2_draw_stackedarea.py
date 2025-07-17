import os


import tools.config as config

import statsmodels.api as sm
import pandas as pd
import seaborn as sns

import matplotlib.patches as mpatches
import matplotlib as mpl
from matplotlib.lines import Line2D

sns.set_theme(style="ticks")


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator
from matplotlib.patches import Patch
import re

def stacked_area_pos_neg(ax, df, colors=None, alpha=0.85,
                         title_name='', ylabel='',
                         show_legend=False):
    """
    在指定 ax 上绘制支持正/负值的堆叠面积图，并可选画出总和的虚线。
    最后总是把所有图例收集到 ax.legend() 中。
    """

    # 3) 画 Sum 虚线
    total = df.sum(axis=1)
    ax.plot(
        df.index,
        total,
        linestyle='-',  # 实线
        marker='o',  # 圆点标记
        color='black',
        linewidth=2,
        markersize=5,
        markeredgewidth=1,
        markerfacecolor='black',
        markeredgecolor='black',
        label='Sum'
    )

    # 1) 颜色准备
    if colors is None:
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    if len(colors) < df.shape[1]:
        colors = (colors * (df.shape[1] // len(colors) + 1))[:df.shape[1]]



    # 2) 画面积
    cum_pos = np.zeros(len(df))
    cum_neg = np.zeros(len(df))
    for idx, col in enumerate(df.columns):
        y    = df[col].values
        pos  = np.clip(y, 0, None)
        neg  = np.clip(y, None, 0)
        colr = colors[idx]

        ax.fill_between(df.index, cum_pos, cum_pos + pos,
                        facecolor=colr, alpha=alpha, linewidth=0, label=col)
        cum_pos += pos

        ax.fill_between(df.index, cum_neg, cum_neg + neg,
                        facecolor=colr, alpha=alpha, linewidth=0)
        cum_neg += neg


    # 4) 美化坐标轴
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
    ax.grid(False)

    # 5) 收集所有 handles/labels，并去重、清理括号
    handles, labels = ax.get_legend_handles_labels()
    clean_labels = [re.sub(r'\s*\(.*?\)', '', lbl) for lbl in labels]

    # 去重（保持第一次出现的顺序）
    unique = {}
    for h, l in zip(handles, clean_labels):
        if l not in unique:
            unique[l] = h

    final_labels  = list(unique.keys())
    final_handles = list(unique.values())

    # 6) 放到 legend
    # 如果 show_legend=True，就放在 ax 内侧左上；否则放到外面
    if show_legend:
        loc, bbox = 'upper left', None
    else:
        loc, bbox = 'upper left', (1.02, 1.0)

    ax.legend(
        handles=final_handles,
        labels=final_labels,
        loc=loc,
        bbox_to_anchor=bbox,
        frameon=False,
        ncol=1,
        handlelength=1.0,
        handleheight=1.0,
        handletextpad=0.4,
        labelspacing=0.3
    )

    return ax




def set_plot_style(font_size=12, font_family='Arial'):
    """
    设置 matplotlib 的统一字体风格和大小（适用于 ax 和 fig.text）

    参数:
    - font_size: int，字体大小
    - font_family: str，字体名称，如 'Arial', 'DejaVu Sans', 'Times New Roman'
    """
    mpl.rcParams.update({
        'font.size': font_size,
        'font.family': font_family,
        'axes.titlesize': font_size,
        'axes.labelsize': font_size,
        'xtick.labelsize': font_size,
        'ytick.labelsize': font_size,
        'legend.fontsize': font_size,
        'figure.titlesize': font_size
    })


def plot_line_with_points(ax, df, color='#1f77b4',title_name='',ylabel=''):
    """
    在指定 ax 上绘制只有一列的点线图（带圆点的折线图）。

    参数：
    - ax: matplotlib 的轴对象
    - df: DataFrame，仅包含 index 和一个值列
    - color: 折线和点的颜色（默认为蓝色）
    - font_size: 坐标轴字体大小
    """
    # 安全检查
    if df.shape[1] != 1:
        raise ValueError("DataFrame 只能包含一列值。")

    col_name = df.columns[0]
    ax.plot(df.index, df[col_name],
            color=color, linewidth=1.5, marker='o',
            markersize=4, markeredgecolor=color)

    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:,.0f}'))
    ax.set_xlim(df.index.min(), df.index.max())

    # 样式
    ax.set_title(title_name,  pad=6)
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.tick_params(axis='both', direction='out', length=4, width=1.2, color='black')
    ax.set_ylabel(ylabel)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    for spine in ax.spines.values():
        spine.set_color('black')
        spine.set_linewidth(1.2)

    ax.grid(False)

def plot_scatter_with_fit(ax, df, title_name='', ylabel='',legend_postiton=(0.3, 1)):
    if df.shape[1] != 1:
        raise ValueError("DataFrame 只能包含一列数值。")

    x = df.index.values.astype(float)
    y = df.iloc[:, 0].values

    # 散点
    ax.scatter(x, y, marker='o', color='C0')

    # 线性拟合
    X = sm.add_constant(x)
    model = sm.OLS(y, X).fit()
    pred = model.get_prediction(X)
    pred_df = pred.summary_frame(alpha=0.05)

    # 拟合直线
    ax.plot(x, pred_df['mean'], color='black', linewidth=2)

    # 置信区间
    ax.fill_between(
        x,
        pred_df['mean_ci_lower'],
        pred_df['mean_ci_upper'],
        color='gray', alpha=0.3
    )

    # 组装回归公式文本
    slope, intercept = model.params[1], model.params[0]
    r2 = model.rsquared
    eq_text = f'y = {slope:.2f}x{intercept:+.2f}\n$R^2$ = {r2:.3f}'

    # 坐标轴格式化
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:,.0f}'))
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.set_xlim(x.min()-0.5, x.max()+0.5)
    start, end = int(df.index.min()), int(df.index.max())
    ax.set_xticks(range(start, end + 1, 5))

    ax.set_title(title_name, pad=6)
    ax.set_xlabel('')
    ax.set_ylabel(ylabel)
    ax.tick_params(axis='both', direction='out', length=4, width=1.2)
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)
    ax.grid(False)

    # —— 新增：只创建一个图例框，把 CI 图例和公式标题放一起 —— #
    # 构造 CI 补丁
    ci_patch = mpatches.Patch(facecolor='gray', alpha=0.3, label='95% CI')

    # 添加图例：标题右对齐，图例框放到子图右上角外一点
    leg = ax.legend(
        handles=[ci_patch],
        title=eq_text,
        loc='upper left',
        bbox_to_anchor=legend_postiton,  # (x, y) 以 axes fraction 为单位，(1,1) 是右上角
        frameon=True,
    )

    # 3) 取出那个标题的 Text 对象，右对齐它的所有行
    title_obj = leg.get_title()
    title_obj.set_ha('left')  # 水平对齐方式
    title_obj.set_multialignment('left')  # 多行也左对齐

    # 让标题文本右对齐
    for txt in leg.get_texts():
        txt.set_ha('left')
        # 可选：把 x 坐标移到最右
        txt.set_x(1.0)

    # 可选：调小 handle 和 label 之间的间距
    leg._legend_box.align = "left"  # 整体右对齐
    leg._legend_handle_box.pad = 0.1  # handle 与 label 之间
    leg._legend_title_box.pad = 0.3  # title 与 entries 之间

    # 4) （可选）微调 legend 框线
    frame = leg.get_frame()
    frame.set_linewidth(0.5)
    frame.set_edgecolor('black')

# def draw_legend(ax,bbox_to_anchor=(0.98, 0.69),ncol=4):
#     # 获取 legend 句柄
#     handles, labels = ax.get_legend_handles_labels()
#     clean_labels = [re.sub(r'\s*\(.*?\)', '', label) for label in labels]
#     colors = [h.get_facecolor()[0] for h in handles]
#     patch_handles = [Patch(facecolor=color) for color in colors]
#     # 加入整个 figure 的图例（例如放在整个图底部中间）
#     fig.legend(patch_handles, clean_labels, bbox_to_anchor=bbox_to_anchor,
#                ncol=ncol, frameon=False,labelspacing=0.3,handlelength=0.9,
#                handleheight=1.0,
#                handletextpad=0.4)

from matplotlib.collections import PathCollection
def draw_legend(ax, bbox_to_anchor=(0.98, 0.69), ncol=5):
    """
    从单个 Axes 收集 legend handles/labels，清理标签中的括号内容，
    支持 Patch、Line2D、PathCollection 三种类型，最后在 Figure 层
    面绘制统一的 legend。

    参数
    ----
    ax              : matplotlib.axes.Axes
        要收集图例的那个子图
    bbox_to_anchor  : tuple
        图例在 Figure 坐标系中的定位 (x, y)
    ncol            : int
        图例分几列
    """
    # 取出对应的 Figure
    fig = ax.get_figure()

    # 1) 从 ax 收集原始 handles/labels
    handles, labels = ax.get_legend_handles_labels()
    clean_labels = [re.sub(r'\s*\(.*?\)', '', lbl) for lbl in labels]

    # 2) 隐藏这个 ax 自带的 legend
    if ax.get_legend() is not None:
        ax.get_legend().remove()

    # 3) 针对不同类型的 handle，重建一个 new_handle 列表
    new_handles = []
    for h in handles:
        if isinstance(h, Patch):
            fc = h.get_facecolor()[0]
            ec = h.get_edgecolor()[0] if h.get_edgecolor().size else 'black'
            lw = h.get_linewidth()
            new_handles.append(Patch(facecolor=fc, edgecolor=ec, linewidth=lw))
        elif isinstance(h, Line2D):
            new_handles.append(Line2D(
                [0], [0],
                color=h.get_color(),
                linestyle=h.get_linestyle(),
                linewidth=h.get_linewidth(),
                marker=h.get_marker(),
                markersize=h.get_markersize(),
                markerfacecolor=h.get_markerfacecolor(),
                markeredgecolor=h.get_markeredgecolor()
            ))
        elif isinstance(h, PathCollection):
            # 散点：重建为带 marker 的 Line2D
            # 取 facecolor 或 edgecolor
            fc = h.get_facecolor()
            ec = h.get_edgecolor()
            color = tuple(fc[0]) if len(fc) else tuple(ec[0]) if len(ec) else 'black'
            size = (h.get_sizes()[0] ** 0.5) if hasattr(h, "get_sizes") and h.get_sizes().size else 6
            new_handles.append(Line2D(
                [0], [0],
                linestyle='',
                marker='o',
                markersize=size,
                markerfacecolor=color,
                markeredgecolor=color
            ))
        else:
            # 其它类型保留原 handle
            new_handles.append(h)

    # 4) 在 Figure 上绘制最终 legend
    fig.legend(
        handles=new_handles,
        labels=clean_labels,
        loc='upper left',
        bbox_to_anchor=bbox_to_anchor,
        ncol=ncol,
        frameon=False,
        handlelength=1.0,
        handleheight=1.0,
        handletextpad=0.4,
        labelspacing=0.3
    )

# ====== 使用方法 ======
set_plot_style(font_size=12, font_family='Arial')

df_ghg = pd.read_excel(f"{config.TASK_DIR}/carbon_price/excel/02_process_{config.INPUT_FILES[1]}.xlsx", index_col=0)
df_ghg_bio = pd.read_excel(f"{config.TASK_DIR}/carbon_price/excel/02_process_{config.INPUT_FILES[0]}.xlsx", index_col=0)
df_price = pd.read_excel(f"{config.TASK_DIR}/carbon_price/excel/03_price.xlsx", index_col=0)
df_bio = df_ghg_bio - df_ghg
df_bio.to_excel(f"{config.TASK_DIR}/carbon_price/excel/02_process_bio.xlsx")

df_ghg = df_ghg.loc[df_ghg.index >= config.START_YEAR].copy()
df_bio = df_bio.loc[df_bio.index >= config.START_YEAR].copy()
df_price = df_price[df_price.index >= config.START_YEAR].copy()

# --- 参数 ---
n_cols = 2
n_rows = 3

# --- 建立画布 ---
fig, axes = plt.subplots(n_rows, n_cols,
                         figsize=(4.5 * n_cols, 4 * (n_rows)),
                         constrained_layout=False)
axes = axes.flatten()
# --- 逐列作图 ---
stacked_area_pos_neg(axes[0], df_ghg.iloc[:, :4], colors=['#DD847E', '#DFDD89','#A7D398', '#74A3D4'],title_name='GHG reductions and removals cost',ylabel='Million AU$')
stacked_area_pos_neg(axes[1], df_bio.iloc[:, :4], colors=['#DD847E', '#DFDD89','#A7D398', '#74A3D4'], title_name='Biodiversity restoration cost',ylabel='')
stacked_area_pos_neg(axes[2], df_ghg.iloc[:, 5:9], colors=['#f7dc68', '#f46c3f', '#2e9599', '#a7226f'],title_name='GHG reductions and removals',ylabel='MtCO2e')
stacked_area_pos_neg(axes[3], df_bio.iloc[:, 10:13], colors=['#ffe3b3', '#92de8b', '#0bb68c'],title_name='Biodiversity restoration',ylabel='Mha')
# plot_line_with_points(axes[4], df_price.iloc[:, 4:5], color='black', title_name='Shadow carbon price',ylabel='AU$ tCO2e-1')
# plot_line_with_points(axes[5], df_price.iloc[:, 5:6], color='black',title_name='Shadow biodiversity price',ylabel='AU$ ha-1')

# plot_scatter_with_fit(axes[4], df_price.iloc[:, 4:5],title_name='Shadow carbon price',ylabel='AU$ tCO2e-1',legend_postiton=(0.5, 1))
# plot_scatter_with_fit(axes[5], df_price.iloc[:, 5:6],title_name='Shadow biodiversity price',ylabel='AU$ ha-1',legend_postiton=(0.0, 1))

draw_legend(axes[0],bbox_to_anchor=(0.11, 0.7)) #
draw_legend(axes[2],bbox_to_anchor=(0.13, 0.4), ncol=1)
draw_legend(axes[3],bbox_to_anchor=(0.63, 0.4), ncol=1)

plt.subplots_adjust(left=0.1,  # 图像左边界（0 = 最左，1 = 最右）
                    right=0.98,  # 图像右边界
                    top=0.95,  # 图像上边界
                    bottom=0.05,  # 图像下边界
                    hspace=0.42,  # 子图上下间距
                    wspace=0.2)  # 子图左右间距

# 手动上移第二行图像，增大第一、二行之间的间距
for i in [2, 3]:  # 第二行的两个图
    pos = axes[i].get_position()
    axes[i].set_position([pos.x0, pos.y0 + 0.03, pos.width, pos.height])


# # 手动下移第三行图像，增大第二、三行之间的间距
# for i in [4, 5]:  # 第三行的两个图
#     pos = axes[i].get_position()
#     axes[i].set_position([pos.x0, pos.y0+0.015, pos.width, pos.height])

for i in [4, 5]:
    fig.delaxes(axes[i])

os.makedirs(f"{config.TASK_DIR}/carbon_price/Paper_figure", exist_ok=True)
plt.savefig(f"{config.TASK_DIR}/carbon_price/Paper_figure/02_draw_stackedarea.png", dpi=300, bbox_inches='tight')
# fig.align_ylabels(axes)
fig.show()