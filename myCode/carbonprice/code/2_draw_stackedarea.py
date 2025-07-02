import numpy as np
import tools.config as config

import statsmodels.api as sm
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from matplotlib.ticker import MaxNLocator
import matplotlib as mpl
import re
from matplotlib.patches import Patch
import math

sns.set_theme(style="ticks")

def stacked_area_pos_neg(ax, df, colors=None, alpha=0.85, title_name='', ylabel='', show_legend=False):
    """
    在指定 `ax` 上绘制支持正/负值的堆叠面积图。

    参数
    ----
    ax      : matplotlib.axes.Axes
        目标子图
    df      : pd.DataFrame
        行为 X 轴（需为数值或日期），列为分类；可含正负值
    colors  : list-like[str] | None
        每列对应的颜色；若 None 使用 matplotlib 默认循环色
    alpha   : float
        填充透明度
    title_name : str
        子图标题
    ylabel  : str
        Y轴标题
    show_legend : bool
        是否显示图例（默认 False）
    """
    if colors is None:
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    if len(colors) < df.shape[1]:
        colors = (colors * (df.shape[1] // len(colors) + 1))[:df.shape[1]]

    cum_pos = np.zeros(len(df))
    cum_neg = np.zeros(len(df))

    for idx, col in enumerate(df.columns):
        y = df[col].values
        color = colors[idx]
        pos = np.where(y > 0, y, 0)
        neg = np.where(y < 0, y, 0)

        # 正值
        ax.fill_between(df.index, cum_pos, cum_pos + pos, facecolor=color, alpha=alpha, linewidth=0, label=col)
        cum_pos += pos
        # 负值
        ax.fill_between(df.index, cum_neg, cum_neg + neg, facecolor=color, alpha=alpha, linewidth=0)
        cum_neg += neg

    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:,.0f}'))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.tick_params(direction='out')
    for spine in ax.spines.values():
        spine.set_color('black')
        spine.set_linewidth(1.2)
    ax.grid(False)
    ax.set_title(title_name, pad=6)
    ax.set_ylabel(ylabel)
    ax.set_xlim(df.index.min(), df.index.max())

    if show_legend:
        # 正则表达式去除括号及内容
        def clean_label(label):
            return re.sub(r'\s*\(.*?\)', '', label)

        handles, labels = ax.get_legend_handles_labels()
        labels = [clean_label(lbl) for lbl in labels]

        # 生成方形 Patch 作为图例符号
        patch_handles = [Patch(facecolor=h.get_facecolor()[0], edgecolor='black') for h in handles]

        ax.legend(handles=patch_handles,
                  labels=labels,
                  loc='upper center',
                  bbox_to_anchor=(0.5, -0.1),
                  ncol=1,
                  frameon=False,
                  handlelength=0.9,
                  handleheight=1.0,
                  handletextpad=0.4)



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

def plot_scatter_with_fit(ax, df, title_name='', ylabel=''):
    """
    在指定 ax 上绘制散点图并添加线性拟合直线及95%置信区间，以及显示回归公式和R²。

    参数：
    - ax: matplotlib 的轴对象
    - df: pd.DataFrame，仅包含 index（x 值）和一列数值（y 值）
    - title_name: 图表标题
    - ylabel: y 轴标签
    """
    if df.shape[1] != 1:
        raise ValueError("DataFrame 只能包含一列数值。")

    # 准备数据
    x = df.index.values.astype(float)
    y = df.iloc[:, 0].values

    # 绘制散点
    ax.scatter(x, y, marker='o')

    # 拟合线性模型
    X = sm.add_constant(x)
    model = sm.OLS(y, X).fit()

    # 获取预测值及置信区间
    pred = model.get_prediction(X)
    pred_df = pred.summary_frame(alpha=0.05)

    # 绘制拟合直线
    ax.plot(x, pred_df['mean'], linewidth=2)

    # 绘制95%置信区间
    ax.fill_between(x,
                    pred_df['mean_ci_lower'],
                    pred_df['mean_ci_upper'],
                    color='gray', alpha=0.3)

    # 显示回归公式和R²
    slope = model.params[1]
    intercept = model.params[0]
    r2 = model.rsquared
    eq_text = f'y = {slope:.2f}x{intercept:+.2f}\n$R^2$ = {r2:.3f}'
    ax.text(0.15, 0.95, eq_text, transform=ax.transAxes,
            va='top', ha='left',
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", lw=0.5))

    # 格式化坐标轴
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:,.0f}'))
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.set_xlim(x.min(), x.max())
    start, end = int(df.index.min()), int(df.index.max())
    ax.set_xticks(range(start, end + 1, 5))
    ax.set_title(title_name, pad=6)
    ax.set_xlabel('')
    ax.set_ylabel(ylabel)
    ax.tick_params(axis='both', direction='out', length=4, width=1.2)
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)

    ax.grid(False)


def draw_legend(ax,bbox_to_anchor=(0.98, 0.69),ncol=4):
    # 获取 legend 句柄
    handles, labels = ax.get_legend_handles_labels()
    clean_labels = [re.sub(r'\s*\(.*?\)', '', label) for label in labels]
    colors = [h.get_facecolor()[0] for h in handles]
    patch_handles = [Patch(facecolor=color) for color in colors]
    # 加入整个 figure 的图例（例如放在整个图底部中间）
    fig.legend(patch_handles, clean_labels, bbox_to_anchor=bbox_to_anchor,
               ncol=ncol, frameon=False,labelspacing=0.3,handlelength=0.9,
               handleheight=1.0,
               handletextpad=0.4)

# ====== 使用方法 ======
set_plot_style(font_size=12, font_family='Arial')

df_ghg = pd.read_excel(f"{config.TASK_DIR}/carbon_price/excel/02_process_{config.INPUT_FILES[1]}.xlsx", index_col=0)
df_ghg_bio = pd.read_excel(f"{config.TASK_DIR}/carbon_price/excel/02_process_{config.INPUT_FILES[0]}.xlsx", index_col=0)
df_price = pd.read_excel(f"{config.TASK_DIR}/carbon_price/excel/03_price.xlsx", index_col=0)

df_ghg = df_ghg.loc[df_ghg.index >= config.START_YEAR].copy()
df_ghg_bio = df_ghg_bio.loc[df_ghg_bio.index >= config.START_YEAR].copy()
df_price = df_price[df_price.index >= config.START_YEAR].copy()

df_ghg_bio = df_ghg_bio - df_ghg
# --- 参数 ---
n_cols = 2
n_rows = 3

# --- 建立画布 ---
fig, axes = plt.subplots(n_rows, n_cols,
                         figsize=(4.5 * n_cols, 4 * (n_rows)),
                         constrained_layout=False)
axes = axes.flatten()
# --- 逐列作图 ---
stacked_area_pos_neg(axes[0], df_ghg.iloc[:, :4], colors=['#DD847E', '#DFDD89','#A7D398', '#74A3D4'],title_name='Carbon sequestration cost',ylabel='Million AU$')
stacked_area_pos_neg(axes[1], df_ghg_bio.iloc[:, :4], colors=['#DD847E', '#DFDD89','#A7D398', '#74A3D4'], title_name='Biodiversity restoration cost',ylabel='')
stacked_area_pos_neg(axes[2], df_ghg.iloc[:, 5:9], colors=['#f7dc68', '#f46c3f', '#2e9599', '#a7226f'],title_name='Carbon sequestration',ylabel='MtCO2e')
stacked_area_pos_neg(axes[3], df_ghg_bio.iloc[:, 10:13], colors=['#ffe3b3', '#92de8b', '#0bb68c'],title_name='Biodiversity restoration',ylabel='Mha')
# plot_line_with_points(axes[4], df_price.iloc[:, 4:5], color='black', title_name='Shadow carbon price',ylabel='AU$ tCO2e-1')
# plot_line_with_points(axes[5], df_price.iloc[:, 5:6], color='black',title_name='Shadow biodiversity price',ylabel='AU$ ha-1')

plot_scatter_with_fit(axes[4], df_price.iloc[:, 4:5],title_name='Shadow carbon price',ylabel='AU$ tCO2e-1')
plot_scatter_with_fit(axes[5], df_price.iloc[:, 5:6],title_name='Shadow biodiversity price',ylabel='AU$ ha-1')

draw_legend(axes[0],bbox_to_anchor=(0.92, 0.7)) #
draw_legend(axes[2],bbox_to_anchor=(0.45, 0.4), ncol=1)
draw_legend(axes[3],bbox_to_anchor=(0.94, 0.4), ncol=1)

plt.subplots_adjust(left=0.1,  # 图像左边界（0 = 最左，1 = 最右）
                    right=0.98,  # 图像右边界
                    top=0.95,  # 图像上边界
                    bottom=0.05,  # 图像下边界
                    hspace=0.42,  # 子图上下间距
                    wspace=0.3)  # 子图左右间距

# 手动上移第二行图像，增大第一、二行之间的间距
for i in [2, 3]:  # 第二行的两个图
    pos = axes[i].get_position()
    axes[i].set_position([pos.x0, pos.y0 + 0.03, pos.width, pos.height])


# 手动下移第三行图像，增大第二、三行之间的间距
for i in [4, 5]:  # 第三行的两个图
    pos = axes[i].get_position()
    axes[i].set_position([pos.x0, pos.y0+0.015, pos.width, pos.height])

plt.savefig(f"{config.TASK_DIR}/carbon_price/Paper_figure/2_draw_stackedarea.png", dpi=300, bbox_inches='tight')
# fig.align_ylabels(axes)
fig.show()