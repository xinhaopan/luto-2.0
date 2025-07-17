import os

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


# ====== 使用方法 ======
set_plot_style(font_size=12, font_family='Arial')  # 你可以换成你想要的字体
# def sns_two_df_comparison(ax, df1, df2, col, colors=('#ffb216', '#6db50a'),
#                           label1='DF1', label2='DF2', y_thousands=True, tick_length=4, border_width=1.5):
#     """
#     在给定 `ax` 上画 df1 与 df2 指定列的对比折线图，统一字体、刻度样式、边框加粗、去掉图例与网格。
#     """
#     # 画 DF1（如果有）
#     if df1 is not None and col in df1.columns:
#         sns.lineplot(x=df1.index, y=df1[col], ax=ax,
#                      color=colors[0], markersize=6, linewidth=1.5, marker='o',
#                      markeredgecolor=colors[0], label=label1)
#         df = df1  # 这里 df1 是主要数据源
#
#     # 画 DF2（如果有）
#     if df2 is not None and col in df2.columns:
#         sns.lineplot(x=df2.index, y=df2[col], ax=ax,
#                      color=colors[1], markersize=6, linewidth=1.5, marker='o',
#                      markeredgecolor=colors[1], label=label2)
#         df = df2  # 如果 df2 存在，则覆盖 df1 的数据源
#
#     ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
#     ylim = ax.get_ylim()
#     if abs(ylim[1] - ylim[0]) < 3:
#         ax.yaxis.set_major_formatter(
#             FuncFormatter(lambda x, _: f'{x:.2f}')
#         )
#     else:
#         if y_thousands:
#             ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:,.0f}'))
#
#     x = df.index.values.astype(float)
#     ax.set_xlim(x.min(), x.max())
#     start, end = int(df.index.min()), int(df.index.max())
#     ax.set_xticks(range(start, end + 1, 5))
#     # 标题、坐标轴标签
#     ax.set_title(col, pad=6)
#     ax.set_xlabel('')
#     ax.set_ylabel('')
#
#     # 刻度字体和方向
#     ax.tick_params(axis='both', direction='out', length=tick_length, width=1.2, color='black')
#
#     # 去网格
#     ax.grid(False)
#
#     # 去图例
#     legend = ax.get_legend()
#     if legend is not None:
#         legend.remove()
#
#     # 加粗边框
#     for spine in ax.spines.values():
#         spine.set_color('black')  # 边框颜色
#         spine.set_linewidth(border_width)


def sns_two_df_comparison(ax, df1, df2, col, colors=('#ffb216', '#6db50a'),
                          label1='DF1', label2='DF2', y_thousands=True,
                          tick_length=4, border_width=1.5):
    # 画 DF1（如果有）
    if df1 is not None and col in df1.columns:
        sns.lineplot(x=df1.index, y=df1[col], ax=ax,
                     color=colors[0], markersize=6, linewidth=1.5, marker='o',
                     markeredgecolor=colors[0], label=label1)
        df = df1  # 这里 df1 是主要数据源

    # 画 DF2（如果有）
    if df2 is not None and col in df2.columns:
        sns.lineplot(x=df2.index, y=df2[col], ax=ax,
                     color=colors[1], markersize=6, linewidth=1.5, marker='o',
                     markeredgecolor=colors[1], label=label2)
        df = df2  # 如果 df2 存在，则覆盖 df1 的数据源

    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ylim = ax.get_ylim()
    if abs(ylim[1] - ylim[0]) < 3:
        ax.yaxis.set_major_formatter(
            FuncFormatter(lambda x, _: f'{x:.2f}')
        )
    else:
        if y_thousands:
            ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:,.0f}'))

    x = df.index.values.astype(float)
    ax.set_xlim(x.min()-0.5, x.max()+0.5)
    start, end = int(df.index.min()), int(df.index.max())
    ax.set_xticks(range(start, end + 1, 5))

    # —— 美化部分 —— #
    # 1) 设背景色
    ax.set_facecolor('#E4E4E4')  # 浅灰
    #
    # # 2) 添加白色网格线（只要主刻度）
    ax.grid(True, color='white', linewidth=2)
    #
    # 3) 加粗主刻度线，让它们更显眼
    ax.tick_params(which='major', length=tick_length, width=1.5, color='black')
    ax.set_title(col, pad=6)

    ax.set_xlabel('')
    ax.set_ylabel('')

    # # 5) 坐标轴标签依旧留空，轴线加粗为白色
    for spine in ax.spines.values():
        spine.set_linewidth(border_width)
        spine.set_edgecolor('black')

    # 最后移除图例、网格外的其它装饰时，如有需要，可再微调
    legend = ax.get_legend()
    if legend:
        legend.remove()




columns_name = ["cost_ag(M$)", "cost_am(M$)", "cost_non-ag(M$)", "cost_transition_ag2ag(M$)",
                "cost_amortised_transition_ag2non-ag(M$)", "revenue_ag(M$)", "revenue_am(M$)", "revenue_non-ag(M$)",
                "GHG_ag(MtCOe2)", "GHG_am(MtCOe2)", "GHG_non-ag(MtCOe2)", "GHG_transition(MtCOe2)",
                "BIO_ag(M ha)", "BIO_am(M ha)", "BIO_non-ag(M ha)"]
title_name = ["Ag cost", "AM cost", "Non-ag cost", "Transition cost (AG to AG)",
              "Amortised transition cost (AG to NON-AG)", "Ag revenue", "AM revenue", "Non-ag revenue",
              "Ag GHG emission", "AM GHG emission", "Non-ag GHG emission", "Transition GHG emission",
              "Ag biodiversity", "AM biodiversity", "Non-ag biodiversity "]
col_map = dict(zip(columns_name, title_name))
df_ghg = pd.read_excel(f"{config.TASK_DIR}/carbon_price/excel/01_origin_{config.INPUT_FILES[1]}.xlsx", index_col=0)
df_ghg_bio = pd.read_excel(f"{config.TASK_DIR}/carbon_price/excel/01_origin_{config.INPUT_FILES[0]}.xlsx", index_col=0)
df_bio = df_ghg_bio-df_ghg
df_bio.to_excel(f"{config.TASK_DIR}/carbon_price/excel/01_origin_bio.xlsx")

df_ghg = df_ghg.loc[df_ghg.index >= config.START_YEAR].copy()
df_bio = df_bio.loc[df_bio.index >= config.START_YEAR].copy()
# 只保留匹配的列，并重命名
df_ghg = df_ghg[[col for col in df_ghg.columns if col in col_map]]
df_ghg = df_ghg.rename(columns=col_map)

df_bio = df_bio[[col for col in df_bio.columns if col in col_map]]
df_bio = df_bio.rename(columns=col_map)
# %%

# --- 参数 ---
n_cols = 4
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
    if 'GHG' in col:
        sns_two_df_comparison(ax=axes[idx], df1=df_ghg, df2=None, col=col, label1='Emission targets',
                              label2='Emission & biodiversity targets \n- emission targets')
    elif 'Biodiversity' in col:
        sns_two_df_comparison(ax=axes[idx], df1=None, df2=df_bio, col=col, label1='Emission targets',
                              label2='Emission & biodiversity targets \n- emission targets')
    else:
        sns_two_df_comparison(ax=axes[idx], df1=df_ghg, df2=df_bio, col=col, label1='Emission targets',
                              label2='Emission & biodiversity targets \n- emission targets')

# --- 全局微调间距 ---
plt.subplots_adjust(left=0.1,  # 图像左边界（0 = 最左，1 = 最右）
                    right=0.95,  # 图像右边界
                    top=0.95,  # 图像上边界
                    bottom=0.05,  # 图像下边界
                    hspace=0.32,  # 子图上下间距
                    wspace=0.25)  # 子图左右间距

# --- 添加图例 ---
# 从第一个有效 ax 提取 legend（这里假设第一个图是双线图）
handles, labels = axes[0].get_legend_handles_labels()

# 添加一个新子图用于图例
legend_ax = fig.add_axes([0.77, 0.08, 0.15, 0.12])  # [left, bottom, width, height]
legend_ax.axis('off')
legend_ax.legend(handles, labels, loc='center', frameon=False)

fig.text(0.05, 0.75, 'Million AU$',
         rotation='vertical',
         va='center', ha='center')

fig.text(0.075, 0.38, 'tCO2e',
         rotation='vertical',
         va='center', ha='center')

fig.text(0.07, 0.15, 'mha',
         rotation='vertical',
         va='center', ha='center')

# --- 保存或显示 ---
output_path = f'{config.TASK_DIR}/carbon_price/Paper_figure'
os.makedirs(output_path, exist_ok=True)  # 确保输出目录存在
fig.savefig(f'{output_path}/01_origin_output.png', dpi=300)
plt.show()
