import os
import tools.config as config
import statsmodels.api as sm
import pandas as pd
import seaborn as sns
import matplotlib.patches as mpatches
import matplotlib as mpl
from matplotlib.lines import Line2D

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator
from matplotlib.patches import Patch
import re

sns.set_theme(style="darkgrid")
plt.rcParams.update({
    "xtick.bottom": True, "ytick.left": True,   # 打开刻度
    "xtick.top": False,  "ytick.right": False,  # 需要的话也可开
    "xtick.direction": "out", "ytick.direction": "out",
    "xtick.major.size": 4, "ytick.major.size": 4,
    "xtick.major.width": 1.2, "ytick.major.width": 1.2,
})

# Function definitions (unchanged from your original code)
def stacked_area_pos_neg(ax, df, colors=None, alpha=0.60, title_name='', ylabel='', show_legend=False):
    total = df.sum(axis=1)
    ax.plot(df.index, total, linestyle='-', marker='o', color='black', linewidth=2,
            markersize=5, markeredgewidth=1, markerfacecolor='black', markeredgecolor='black', label='Sum')

    if colors is None:
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    if len(colors) < df.shape[1]:
        colors = (colors * (df.shape[1] // len(colors) + 1))[:df.shape[1]]

    cum_pos = np.zeros(len(df))
    cum_neg = np.zeros(len(df))
    for idx, col in enumerate(df.columns):
        y = df[col].values
        pos = np.clip(y, 0, None)
        neg = np.clip(y, None, 0)
        colr = colors[idx]
        ax.fill_between(df.index, cum_pos, cum_pos + pos, facecolor=colr, alpha=alpha, linewidth=0, label=col)
        cum_pos += pos
        ax.fill_between(df.index, cum_neg, cum_neg + neg, facecolor=colr, alpha=alpha, linewidth=0)
        cum_neg += neg

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
    # ax.grid(False)

    handles, labels = ax.get_legend_handles_labels()
    clean_labels = labels
    unique = {}
    for h, l in zip(handles, clean_labels):
        if l not in unique:
            unique[l] = h
    final_labels = list(unique.keys())
    final_handles = list(unique.values())

    if show_legend:
        ax.legend(handles=final_handles, labels=final_labels, loc='upper left', bbox_to_anchor=(1.02, 1.0),
                  frameon=False, ncol=1, handlelength=1.0, handleheight=1.0, handletextpad=0.4, labelspacing=0.3)
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
set_plot_style(font_size=12, font_family='Arial')

# Load data
df_all = pd.read_excel(f"{config.TASK_DIR}/carbon_price/excel/03_cost.xlsx", index_col=0)
# 按列索引拆分
df_ghg = df_all.iloc[:, 0:5]       # 第1到第5列
df_bio = df_all.iloc[:, 6:11]      # 第7到第12列
df_ghg = df_ghg.loc[df_ghg.index >= config.START_YEAR].copy()
df_bio = df_bio.loc[df_bio.index >= config.START_YEAR].copy()
df_ghg.columns = ['Ag','AM','Non-ag','Transition(ag→ag)','Transition(ag→non-ag)']
df_bio.columns = ['Ag','AM','Non-ag','Transition(ag→ag)','Transition(ag→non-ag)']

# Set up parameters for 1 row, 2 columns
n_cols = 2
n_rows = 1

# Create figure and axes
fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.5 * n_cols, 4 * n_rows), constrained_layout=False)
axes = axes.flatten()

# Plot only GHG cost and Biodiversity restoration cost
stacked_area_pos_neg(axes[0], df_ghg, colors=['#f39b8b', '#9A8AB3', '#6eabb1', '#eb9132','#84a374'],
                     title_name='GHG reductions and removals cost', ylabel='Cost (Million AU$)')
stacked_area_pos_neg(axes[1], df_bio, colors=['#f39b8b', '#9A8AB3', '#6eabb1', '#eb9132','#84a374'],
                     title_name='Biodiversity restoration cost', ylabel='')

# Draw legend for the first plot
draw_legend(axes[0], bbox_to_anchor=(0.08, 0.15))

# Adjust subplot spacing
plt.subplots_adjust(left=0.1, right=0.95, top=0.9, bottom=0.2, wspace=0.3)

# Save and show the figure
os.makedirs(f"{config.TASK_DIR}/carbon_price/Paper_figure", exist_ok=True)
plt.savefig(f"{config.TASK_DIR}/carbon_price/Paper_figure/02_draw_stackedarea.png", dpi=300, bbox_inches='tight')
fig.show()