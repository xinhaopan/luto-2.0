import numpy as np
import tools.config as config

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from matplotlib.ticker import MaxNLocator
import matplotlib as mpl
import math

sns.set_theme(style="ticks")

def stacked_area_pos_neg(ax, df, colors=None, alpha=0.85):
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
    font_size : int
        轴刻度与标题字号
    """
    if colors is None:
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    # 若列多于颜色数，自动重复
    if len(colors) < df.shape[1]:
        colors = (colors * (df.shape[1] // len(colors) + 1))[:df.shape[1]]

    # 准备累积数组
    cum_pos = np.zeros(len(df))
    cum_neg = np.zeros(len(df))

    # 依次绘制每列
    for idx, col in enumerate(df.columns):
        y = df[col].values
        color = colors[idx]

        pos = np.where(y > 0, y, 0)       # 正值部分
        neg = np.where(y < 0, y, 0)       # 负值部分 (负数)

        # --- 正值向上堆叠 ---
        ax.fill_between(df.index,
                        cum_pos,
                        cum_pos + pos,
                        facecolor=color, alpha=alpha, label=col if pos.any() else None,
                        linewidth=0)
        cum_pos += pos

        # --- 负值向下堆叠 ---
        ax.fill_between(df.index,
                        cum_neg,
                        cum_neg + neg,     # 注意 neg 为负
                        facecolor=color, alpha=alpha,
                        linewidth=0)
        cum_neg += neg

    # 美化
    # ax.axhline(0, color='black', linewidth=1)             # 0 基线
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.tick_params(direction='out')
    for spine in ax.spines.values():
        spine.set_color('black');  spine.set_linewidth(1.2)
    ax.grid(False)

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




# ====== 使用方法 ======
set_plot_style(font_size=12, font_family='Arial')

df_ghg = pd.read_excel(f"{config.TASK_DIR}/carbon_price/excel/02_process_Run_3_GHG_high_BIO_off.xlsx", index_col=0)
df_ghg_bio = pd.read_excel(f"{config.TASK_DIR}/carbon_price/excel/02_process_Run_4_GHG_high_BIO_high.xlsx", index_col=0)
df_ghg_bio = df_ghg_bio - df_ghg

# --- 参数 ---
n_cols = 2
n_rows = 3

# --- 建立画布 ---
fig, axes = plt.subplots(n_rows, n_cols,
                         figsize=(4 * n_cols, 3.2 * (n_rows)),
                         constrained_layout=False)
axes = axes.flatten()
# --- 逐列作图 ---
stacked_area_pos_neg(axes[0], df_ghg.iloc[:, :4], colors=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
stacked_area_pos_neg(axes[1], df_ghg_bio.iloc[:, :4], colors=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
stacked_area_pos_neg(axes[2], df_ghg.iloc[:, 5:9], colors=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
stacked_area_pos_neg(axes[3], df_ghg_bio.iloc[:, 10:13], colors=['#1f77b4', '#ff7f0e', '#2ca02c'])

fig.show()