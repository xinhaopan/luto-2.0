import os
import tools.config as config
import statsmodels.api as sm
import pandas as pd
import seaborn as sns
import matplotlib.patches as mpatches
import matplotlib as mpl
from matplotlib.lines import Line2D
import xarray as xr

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator
from matplotlib.patches import Patch
import re
import matplotlib


sns.set_theme(style="darkgrid",font="Arial", font_scale=2)
plt.rcParams.update({
    "xtick.bottom": True, "ytick.left": True,   # 打开刻度
    "xtick.top": False,  "ytick.right": False,  # 需要的话也可开
    "xtick.direction": "out", "ytick.direction": "out",
    "xtick.major.size": 4, "ytick.major.size": 4,
    "xtick.major.width": 1.2, "ytick.major.width": 1.2,
})

# Function definitions (unchanged from your original code)
def stacked_area_pos_neg(
    ax, df, colors=None, alpha=0.60,
    title_name='', ylabel='',
    add_line=True, n_col=1, show_legend=False,bbox_to_anchor=(0.5, -0.25)
):
    # ---- 可选总和线 ----
    if add_line:
        total = df.sum(axis=1)
        ax.plot(
            df.index, total,
            linestyle='-', marker='o',
            color='black', linewidth=2,
            markersize=5, markeredgewidth=1,
            markerfacecolor='black', markeredgecolor='black',
            label='Sum'
        )

    # ---- 颜色 ----
    if colors is None:
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    if len(colors) < df.shape[1]:
        colors = (colors * (df.shape[1] // len(colors) + 1))[:df.shape[1]]

    # ---- 堆叠面积 ----
    cum_pos = np.zeros(len(df))
    cum_neg = np.zeros(len(df))
    for idx, col in enumerate(df.columns):
        y = df[col].values
        pos = np.clip(y, 0, None)
        neg = np.clip(y, None, 0)
        colr = colors[idx]
        ax.fill_between(df.index, cum_pos, cum_pos + pos,
                        facecolor=colr, alpha=alpha, linewidth=0, label=col)
        cum_pos += pos
        ax.fill_between(df.index, cum_neg, cum_neg + neg,
                        facecolor=colr, alpha=alpha, linewidth=0)
        cum_neg += neg

    # ---- 轴和外观 ----
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

    # ---- 图例 ----
    handles, labels = ax.get_legend_handles_labels()
    unique = {}
    for h, l in zip(handles, labels):
        if l not in unique:
            unique[l] = h
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


def draw_legend(ax, bbox_to_anchor=(0.98, 0.69), ncol=6):
    fig = ax.get_figure()
    handles, labels = ax.get_legend_handles_labels()
    clean_labels = labels
    if ax.get_legend() is not None:
        ax.get_legend().remove()

    new_handles = []
    for h in handles:
        if isinstance(h, Patch):
            fc = h.get_facecolor()[0]
            ec = h.get_edgecolor()[0] if h.get_edgecolor().size else 'black'
            lw = h.get_linewidth()
            new_handles.append(Patch(facecolor=fc, edgecolor=ec, linewidth=lw))
        elif isinstance(h, Line2D):
            new_handles.append(Line2D([0], [0], color=h.get_color(), linestyle=h.get_linestyle(),
                                      linewidth=h.get_linewidth(), marker=h.get_marker(),
                                      markersize=h.get_markersize(), markerfacecolor=h.get_markerfacecolor(),
                                      markeredgecolor=h.get_markeredgecolor()))
        else:
            new_handles.append(h)

    fig.legend(handles=new_handles, labels=clean_labels, loc='upper left', bbox_to_anchor=bbox_to_anchor,
               ncol=ncol, frameon=False, handlelength=1.0, handleheight=1.0, handletextpad=0.4, labelspacing=0.3)


# Main script
set_plot_style(font_size=20, font_family='Arial')

input_dir = f'../../../output/{config.TASK_NAME}/carbon_price/0_base_data/Results'
output_dir = f"../../../output/{config.TASK_NAME}/carbon_price/3_Paper_figure"

fig = plt.figure(figsize=(13, 13))
ax1 = fig.add_subplot(2, 2, 1)

xr_category = xr.open_dataset(f'{input_dir}/xr_carbon_cost_category.nc') / 1e6
columns_name = ['Ag','AgMgt','Non-ag','Transition(ag→ag)','Transition(ag→non-ag)']
colors=['#f39b8b', '#9A8AB3', '#6eabb1', '#eb9132','#84a374']
# Adjust subplot spacing
df =  xr_category['data'].to_pandas()
# Load data
df = df.loc[df.index >= config.START_YEAR].copy()
df.columns = columns_name
stacked_area_pos_neg(ax1, df, colors=colors,
                     title_name='GHG reductions and removals cost', ylabel='',show_legend=True,bbox_to_anchor=(0.01, 0.98))

xr_category = xr.open_dataset(f'{input_dir}/xr_carbon_cost_tn.nc') / 1e6
df = xr_category['data'].to_pandas().iloc[:, 1:]
df_map = pd.read_excel('tools/land use colors.xlsx', sheet_name='non_ag')

df_map = df_map.copy()
df_map["desc_clean"] = df_map["desc"].str.lower().str.strip()
cols = pd.DataFrame({"desc_original": df.columns})
cols["desc_clean"] = cols["desc_original"].str.lower().str.strip()

# 2) 把映射信息（desc_new, color）并到每个原列上
joined = cols.merge(
    df_map[["desc_clean", "desc_new", "color"]],
    on="desc_clean", how="left"
)

# 3) 校验是否有匹配不到的列
missing = joined[joined["desc_new"].isna()]
if not missing.empty:
    raise ValueError(f"映射表里找不到这些列：{missing['desc_original'].tolist()}")

# 4) 按 desc_new 的长度排序（长在前），同长再按字母序
joined["len"] = joined["desc_new"].str.len()
joined = joined.sort_values(["len", "desc_new"], ascending=[False, True]).reset_index(drop=True)
# 5) 依新顺序重排 df 的列
new_order_original = joined["desc_original"].tolist()
df_sorted = df[new_order_original].copy()
# 6) 重命名为 desc_new
rename_map = dict(zip(joined["desc_original"], joined["desc_new"]))
df_sorted = df_sorted.rename(columns=rename_map)
# 7) 生成与新列顺序一致的列表
columns_name = joined["desc_new"].tolist()
colors = joined["color"].tolist()

df = df_sorted.loc[df.index >= config.START_YEAR].copy()
df.columns = columns_name
ax2 = fig.add_subplot(2, 2, 2)
stacked_area_pos_neg(ax2, df, colors=colors,add_line=False,
                     title_name='Transition(ag→ag) cost for GHG', ylabel='',show_legend=True,bbox_to_anchor=(0.01, 0.98))

xr_category = xr.open_dataset(f'{input_dir}/xr_bio_cost_category.nc') / 1e6
columns_name = ['Ag','AgMgt','Non-ag','Transition(ag→ag)','Transition(ag→non-ag)']
colors=['#f39b8b', '#9A8AB3', '#6eabb1', '#eb9132','#84a374']
# Adjust subplot spacing
df =  xr_category['data'].to_pandas()
# Load data
df = df.loc[df.index >= config.START_YEAR].copy()
df.columns = columns_name
ax3 = fig.add_subplot(2, 2, 3)
stacked_area_pos_neg(ax3, df, colors=colors,
                     title_name='Biodiversity restoration cost', ylabel='',show_legend=False,bbox_to_anchor=(0.01, 0.98))

xr_category = xr.open_dataset(f'{input_dir}/xr_bio_cost_tn.nc') / 1e6
df = xr_category['data'].to_pandas().iloc[:, 1:]
df_map = pd.read_excel('tools/land use colors.xlsx', sheet_name='non_ag')

df_map = df_map.copy()
df_map["desc_clean"] = df_map["desc"].str.lower().str.strip()
cols = pd.DataFrame({"desc_original": df.columns})
cols["desc_clean"] = cols["desc_original"].str.lower().str.strip()

# 2) 把映射信息（desc_new, color）并到每个原列上
joined = cols.merge(
    df_map[["desc_clean", "desc_new", "color"]],
    on="desc_clean", how="left"
)

# 3) 校验是否有匹配不到的列
missing = joined[joined["desc_new"].isna()]
if not missing.empty:
    raise ValueError(f"映射表里找不到这些列：{missing['desc_original'].tolist()}")

# 4) 按 desc_new 的长度排序（长在前），同长再按字母序
joined["len"] = joined["desc_new"].str.len()
joined = joined.sort_values(["len", "desc_new"], ascending=[False, True]).reset_index(drop=True)
# 5) 依新顺序重排 df 的列
new_order_original = joined["desc_original"].tolist()
df_sorted = df[new_order_original].copy()
# 6) 重命名为 desc_new
rename_map = dict(zip(joined["desc_original"], joined["desc_new"]))
df_sorted = df_sorted.rename(columns=rename_map)
# 7) 生成与新列顺序一致的列表
columns_name = joined["desc_new"].tolist()
colors = joined["color"].tolist()

df = df_sorted.loc[df.index >= config.START_YEAR].copy()
df.columns = columns_name
ax4 = fig.add_subplot(2, 2, 4)
stacked_area_pos_neg(ax4, df, colors=colors,add_line=False,
                     title_name='Transition(ag→ag) cost for biodiversity', ylabel='',show_legend=False,bbox_to_anchor=(0.01, 0.98))

plt.subplots_adjust(left=0.1, right=0.95, top=0.9, bottom=0.2, wspace=0.15)
fig.text(
    0.04, 0.55,                     # x=0.02 靠左，y=0.5 垂直居中
    "MAU$",                 # 文字内容
    va="center", ha="center",      # 垂直水平对齐方式
    rotation="vertical",           # 旋转 90 度
)

# Save and show the figure
os.makedirs(output_dir, exist_ok=True)
plt.savefig(f"{output_dir}/02_draw_stackedarea.png", dpi=300)
plt.show()