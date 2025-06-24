import tools.config as config

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from matplotlib.ticker import MaxNLocator
import matplotlib as mpl
import math

sns.set_theme(style="ticks")

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


import matplotlib.pyplot as plt
import pandas as pd


def plot_stacked_area(ax, df, colors=None, font_size=10):
    """
    在指定 ax 上绘制堆叠面积图。

    参数：
    - ax: matplotlib 的轴对象
    - df: DataFrame，行是 x 轴（例如年份），列是各分类的值
    - colors: 自定义颜色列表
    - font_size: 字体大小
    """
    # 使用 seaborn 风格
    plt.style.use('seaborn-ticks')

    # 堆叠图
    ax.stackplot(df.index, df.values.T, labels=df.columns, colors=colors)

    # 样式
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.tick_params(axis='both', labelsize=font_size, direction='out')
    for spine in ax.spines.values():
        spine.set_color('black')
        spine.set_linewidth(1.2)

    ax.grid(False)
    ax.legend(loc='upper left', fontsize=font_size - 2)


# ====== 使用方法 ======
set_plot_style(font_size=12, font_family='Arial')

df_ghg = pd.read_excel(f"{config.TASK_DIR}/carbon_price/excel/02_process_Run_3_GHG_high_BIO_off.xlsx", index_col=0)
df_ghg_bio = pd.read_excel(f"{config.TASK_DIR}/carbon_price/excel/02_process_Run_3_GHG_high_BIO_off.xlsx", index_col=0)
df_ghg_bio = df_ghg_bio - df_ghg

# --- 参数 ---
n_cols = 2
cols_list = df_ghg.columns  # 需要绘制的列
n_rows = math.ceil(len(cols_list) / n_cols)  # 不包括图例行

# --- 建立画布 ---
fig, axes = plt.subplots(n_rows, n_cols,
                         figsize=(4 * n_cols, 3.2 * (n_rows)),
                         constrained_layout=False)
axes = axes.flatten()
# 把多余的 Axes 关掉（不显示坐标轴、不占空间）
for ax in axes[len(cols_list):]:  # len(cols_list) == 15
    ax.axis('off')
# --- 逐列作图 ---
for idx, col in enumerate(cols_list):
