import os

import numpy as np
import tools.config as config

import statsmodels.api as sm
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from matplotlib.ticker import MaxNLocator
import matplotlib.patches as mpatches
import matplotlib as mpl
import re
from matplotlib.patches import Patch
import math

sns.set_theme(
    style="white",
    palette=None,
    rc={
        'font.family': 'Arial',
        'font.size': 12,            # 全局默认字体大小
        'axes.titlesize': 12,       # 子图标题
        'axes.labelsize': 12,       # x/y 轴标签
        'xtick.labelsize': 12,      # 刻度标签
        'ytick.labelsize': 12,      # 刻度标签
        'legend.fontsize': 12,      # 图例文字
        'figure.titlesize': 12,     # suptitle（如果你有的话）

        "mathtext.fontset": "custom",
        "mathtext.rm":      "Arial",
        "mathtext.it":      "Arial:italic",
        "mathtext.bf":      "Arial:italic",

        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 5,   # 刻度长度可选
        "ytick.major.size": 5,
    }
)

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
    eq_text = f'y = {slope:.2f}x{intercept:+.2f}\n$R^2$ = {r2:.2f}'

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

df_price = pd.read_excel(f"{config.TASK_DIR}/carbon_price/excel/04_price.xlsx", index_col=0)
df_price = df_price.loc[df_price.index >= config.START_YEAR].copy()


fig = plt.figure(figsize=(12, 5))
plt.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.1)

# 第一个子图（左边）
ax1 = fig.add_subplot(1, 2, 1)
position_ax1=ax1.get_position() #返回Bbox对象
left=position_ax1.x0
bottom=position_ax1.y0
width=position_ax1.width
height=position_ax1.height

ax2 = fig.add_axes([left+0.48,bottom,width,height])  # 第二个子图（右边），与第一个子图共享 y 轴

plot_scatter_with_fit(ax1, df_price.iloc[:, 4:5],title_name='Carbon price',ylabel=r"$\mathrm{AU\$}\ \mathrm{tCO_2e^{-1}}$",legend_postiton=(0.64, 1))
plot_scatter_with_fit(ax2, df_price.iloc[:, 5:6],title_name='Biodiversity price',ylabel=r"$\mathrm{AU\$}\ \mathrm{ha^{-1}}$",legend_postiton=(0.0, 1))
plt.show()
