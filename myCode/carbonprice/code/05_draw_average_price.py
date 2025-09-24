import xarray as xr
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib.lines as mlines
import matplotlib.patches as Patch
from matplotlib.ticker import MaxNLocator, FuncFormatter

import tools.config as config
from tools.helper_plot import set_plot_style

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
    sns.regplot(
        data=df_plot,
        x="x",
        y="y",
        order=order,
        ci=ci,
        scatter_kws=scatter_kws,
        line_kws=line_kws,
        color=color,
        ax=ax
    )
    ax.set_title(title_name, pad=6)
    ax.set_xlabel('')
    ax.set_ylabel('')
    for spine in ax.spines.values():
        spine.set_color('black')
        spine.set_linewidth(1.2)

task_name = config.TASK_NAME
# task_name = '20250922_Paper2_Results_HPC_test'
input_dir = f'../../../output/{task_name }/carbon_price/1_draw_data'
output_dir = f"../../../output/{task_name }/carbon_price/3_Paper_figure"
carbon_price_da = xr.open_dataarray(f"{input_dir}/xr_carbon_price.nc")
bio_price_da = xr.open_dataarray(f"{input_dir}/xr_bio_price.nc")

df1 = pd.concat([carbon_price_da.to_pandas().T.iloc[:, :2],bio_price_da.to_pandas().T.iloc[:, 2:12]], axis=1, join='inner')
df1 = df1[df1.index >= config.START_YEAR]
df2 = carbon_price_da.to_pandas().T.iloc[:, 12:]
df2 = df2[df2.index >= config.START_YEAR]
color = 'green'

set_plot_style(20)



fig = plt.figure(figsize=(24, 12))
color = 'green'
gs = gridspec.GridSpec(3, 5, figure=fig, hspace=0.5, wspace=0.2)

# ------ Carbon部分y轴范围（只用前两列，从0开始） ------
carbon_y = np.concatenate([df1.iloc[:, i].values for i in range(2)])
carbon_ymax = np.nanmax(carbon_y)
# carbon_ylim = (0, carbon_ymax)  # 加10%空间
carbon_ylim = (0,20)

# ------ Bio部分y轴范围（只用后10列，从0开始） ------
bio_y = np.concatenate([df1.iloc[:, i+2].values for i in range(10)])
bio_ymax = np.nanmax(bio_y)
bio_ylim = (0, bio_ymax * 1.2)  # 加10%空间

# ------ x轴刻度（所有数据统一） ------
x_data = df1.index
x_min, x_max = x_data.min(), x_data.max()
x_middle = x_data[int(len(x_data) // 2)]
tick_positions = [x_min, x_middle, x_max]

def int_fmt(x, pos):
    return f"{int(x)}"
int_formatter = FuncFormatter(int_fmt)

# ------ Carbon图（第一行前两个） ------
ax_carbon_list = []
for i in range(2):
    ax = fig.add_subplot(gs[0, i])
    df_input = df1.iloc[:, i].to_frame()
    draw_fit_line_ax(ax, df_input, color=color, title_name=config.PRICE_TITLE_MAP.get(df1.columns[i]))
    ax.set_ylim(*carbon_ylim)
    # ax.yaxis.set_major_locator(MaxNLocator(nbins=3, integer=True))
    ax.set_yticks([0, 10, 20])
    ax.yaxis.set_major_formatter(int_formatter)
    ax.set_xticks(tick_positions)
    ax.tick_params(axis='x')
    x_labels = ax.get_xticklabels()
    if len(x_labels) >= 3:
        x_labels[0].set_horizontalalignment('left')
        x_labels[-1].set_horizontalalignment('right')
    if i != 0:
        ax.tick_params(axis='y', labelleft=False)
    ax_carbon_list.append(ax)

# ------ 图例区 ------
legend_ax = fig.add_subplot(gs[0, 2:])
legend_ax.axis('off')

# ------ Bio图（后两行） ------
ax_bio_list = []
for i in range(10):
    row, col = i // 5 + 1, i % 5
    ax = fig.add_subplot(gs[row, col])
    df_input = df1.iloc[:, i+2].to_frame()
    draw_fit_line_ax(ax, df_input, color=color, title_name=config.PRICE_TITLE_MAP.get(df1.columns[i+2]))
    ax.set_ylim(*bio_ylim)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4, integer=True))
    ax.yaxis.set_major_formatter(int_formatter)
    ax.set_xticks(tick_positions)
    ax.tick_params(axis='x')
    x_labels = ax.get_xticklabels()
    if len(x_labels) >= 3:
        x_labels[0].set_horizontalalignment('left')
        x_labels[-1].set_horizontalalignment('right')
    if col != 0:
        ax.tick_params(axis='y', labelleft=False)
    ax_bio_list.append(ax)

# ------ Y轴标签 ------
ax_carbon_list[0].set_ylabel(r"Carbon price (AU\$ tCO$_2$e$^{-1}$)")
ax_bio_list[0].set_ylabel(r"Biodiversity price (AU\$ ha$^{-1}$)")
ax_carbon_list[0].yaxis.set_label_coords(-0.3, 0.3)
ax_bio_list[0].yaxis.set_label_coords(-0.3, -0.4)

# ------ 图例 ------
line_handle = mlines.Line2D([], [], color=color, linewidth=2, label="Quadratic fit")
shade_handle = Patch.Patch(color=color, alpha=0.25, label="95% CI")
leg = fig.legend(
    handles=[line_handle, shade_handle],
    bbox_to_anchor=(0.8, 0.8),
    ncol=2
)
leg.get_frame().set_facecolor('none')
leg.get_frame().set_edgecolor('none')
plt.savefig(f"{output_dir}/05_Carbon_Bio_price.png", dpi=300)
plt.show()

fig = plt.figure(figsize=(24, 8))
color = 'green'
gs = gridspec.GridSpec(2, 5, figure=fig, hspace=0.5, wspace=0.2)

# ------ Bio部分y轴范围（df2全部10列，从0开始） ------
bio_y = np.concatenate([df2.iloc[:, i].values for i in range(10)])
bio_ymax = np.nanmax(bio_y)
bio_ylim = (0, bio_ymax * 1.2)  # 加20%空间

# ------ x轴刻度（所有数据统一） ------
x_data = df2.index
x_min, x_max = x_data.min(), x_data.max()
x_middle = x_data[int(len(x_data) // 2)]
tick_positions = [x_min, x_middle, x_max]

def int_fmt(x, pos):
    return f"{int(x)}"
int_formatter = FuncFormatter(int_fmt)

# ------ Bio图（两行共10张，分别对应df2的10列） ------
ax_bio_list = []
for i in range(10):
    row, col = i // 5, i % 5
    ax = fig.add_subplot(gs[row, col])
    df_input = df2.iloc[:, i].to_frame()
    draw_fit_line_ax(ax, df_input, color=color, title_name=config.PRICE_TITLE_MAP.get(df2.columns[i]))
    ax.set_ylim(*bio_ylim)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4, integer=True))
    ax.yaxis.set_major_formatter(int_formatter)
    ax.set_xticks(tick_positions)
    ax.tick_params(axis='x')
    x_labels = ax.get_xticklabels()
    if len(x_labels) >= 3:
        x_labels[0].set_horizontalalignment('left')
        x_labels[-1].set_horizontalalignment('right')
    if col != 0:
        ax.tick_params(axis='y', labelleft=False)
    ax_bio_list.append(ax)

# ------ Y轴标签 ------
ax_bio_list[0].set_ylabel(r"carbon price for GHG and biodiversity (AU\$ tCO$_2$e$^{-1}$)")
ax_bio_list[0].yaxis.set_label_coords(-0.3, -0.2)

# ------ 图例 ------
line_handle = mlines.Line2D([], [], color=color, linewidth=2, label="Quadratic fit")
shade_handle = Patch.Patch(color=color, alpha=0.25, label="95% CI")
leg = ax_bio_list[0].legend(
    handles=[line_handle, shade_handle],
    loc='best',
    ncol=1,
    frameon=False
)
plt.savefig(f"{output_dir}/05_Carbon_price_for_GHG_and_bio.png", dpi=300)
plt.show()

