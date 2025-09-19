import os
import xarray as xr
import pandas as pd
from joblib import Parallel, delayed
import numpy as np
import seaborn as sns
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FuncFormatter, MaxNLocator
import matplotlib as mpl
from tools.tools import save2nc
import tools.config as config
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import matplotlib.patches as Patch
import matplotlib.lines as mlines

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
})


def set_plot_style(font_size=12, font_family='Arial'):
    """
    统一设置 Matplotlib 的全局绘图风格。

    这个函数会更新 rcParams，以确保图表中的所有文本元素
    （标题、标签、刻度、图例等）都使用一致的字体和大小。

    参数:
        font_size (int): 基础字号，应用于所有文本元素。
        font_family (str): 字体族，例如 'Arial', 'Helvetica', 'Times New Roman'。
    """
    # mpl.rcParams.update() 是一种高效地一次性更新多个参数的方法。
    mpl.rcParams.update({
        # --- 全局字体设置 ---
        'font.size': font_size,  # 设置基础字号
        'font.family': 'sans-serif',  # 推荐设置字体“族”为无衬线
        'font.sans-serif': [font_family, 'DejaVu Sans'],  # 提供一个字体备选列表，Arial优先

        # --- 坐标轴(Axes)相关字体大小 ---
        'axes.titlesize': font_size,  # 子图标题 (ax.set_title())
        'axes.labelsize': font_size,  # 子图X/Y轴标签 (ax.set_xlabel())

        # --- 刻度(Ticks)相关字体大小 ---
        'xtick.labelsize': font_size,  # X轴刻度标签
        'ytick.labelsize': font_size,  # Y轴刻度标签

        # --- 图例(Legend)字体大小 ---
        'legend.fontsize': font_size,

        # --- 图表(Figure)级别字体大小 ---
        # 这个参数同时控制 fig.suptitle() 和 fig.supylabel()
        'figure.titlesize': font_size
    })


def create_profit(excel_path: str) -> pd.DataFrame:
    original_df = pd.read_excel(excel_path, index_col=0)
    profit_df = pd.DataFrame()

    # 规则 1: Ag profit = Ag revenue - Ag cost
    profit_df['Ag profit'] = original_df['Ag revenue'] - original_df['Ag cost']

    # 规则 2: Agmgt profit = Agmgt revenue - Agmgt cost
    profit_df['Agmgt profit'] = original_df['Agmgt revenue'] - original_df['Agmgt cost']

    # 规则 3: Non-ag profit = Non-ag revenue - Non-ag cost
    profit_df['Non-ag profit'] = original_df['Non-ag revenue'] - original_df['Non-ag cost']

    # 规则 4: Transition(ag→ag) profit = 0 - Transition(ag→ag) cost
    profit_df['Transition(ag→ag) profit'] = 0 - original_df['Transition(ag→ag) cost']

    # 规则 5: Transition(ag→non-ag) amortised profit = 0 - Transition(ag→non-ag) amortised cost
    profit_df['Transition(ag→non-ag) amortised profit'] = 0 - original_df['Transition(ag→non-ag) amortised cost']
    return profit_df

def drwa_fit_line(ax, df, color='black',title_name = ''):
    x = df.index.values
    y_cols = df.columns[:3]

    df_plot = df.reset_index()
    x_col = df.index.name or "year"
    if df_plot.columns[0] != x_col:
        df_plot.rename(columns={df_plot.columns[0]: x_col}, inplace=True)

    sns.regplot(
        data=df_plot,
        x=x_col,
        y=y_cols[2],  # 第3列
        order=2,
        ci=95,
        scatter_kws=dict(s=18),
        line_kws=dict(linewidth=2),
        color=color,
        ax=ax  # ✅ 关键：指定当前子图
    )
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_title(title_name, pad=6)
    ax.set_xlim(x.min(), x.max())

    for spine in ax.spines.values():
        spine.set_color('black')
        spine.set_linewidth(1.2)

    return ax



task_name = config.TASK_NAME
base_path = f"../../../output/{task_name}/carbon_price/0_base_data"
excel_path = f"../../../output/{task_name}/carbon_price/1_excel"
figure_path = f"../../../output/{task_name}/carbon_price/3_Paper_figure"

carbon_names = config.carbon_names
carbon_bio_names = config.carbon_bio_names

title_carbon_names = [
    '$\mathrm{GHG}_{\mathrm{low}}$',
    '$\mathrm{GHG}_{\mathrm{high}}$']

title_bio_names = [
    '$\mathrm{GHG}_{\mathrm{low}}$,$\mathrm{Bio}_{\mathrm{10}}$',
    '$\mathrm{GHG}_{\mathrm{low}}$,$\mathrm{Bio}_{\mathrm{20}}$',
    '$\mathrm{GHG}_{\mathrm{low}}$,$\mathrm{Bio}_{\mathrm{30}}$',
    '$\mathrm{GHG}_{\mathrm{low}}$,$\mathrm{Bio}_{\mathrm{40}}$',
    '$\mathrm{GHG}_{\mathrm{low}}$,$\mathrm{Bio}_{\mathrm{50}}$',

    '$\mathrm{GHG}_{\mathrm{high}}$,$\mathrm{Bio}_{\mathrm{10}}$',
    '$\mathrm{GHG}_{\mathrm{high}}$,$\mathrm{Bio}_{\mathrm{20}}$',
    '$\mathrm{GHG}_{\mathrm{high}}$,$\mathrm{Bio}_{\mathrm{30}}$',
    '$\mathrm{GHG}_{\mathrm{high}}$,$\mathrm{Bio}_{\mathrm{40}}$',
    '$\mathrm{GHG}_{\mathrm{high}}$,$\mathrm{Bio}_{\mathrm{50}}$'
]

all_carbon = []
for input_file in carbon_names:
    df = pd.read_excel(os.path.join(excel_path, f'2_{input_file}_cost_series.xlsx'), index_col=0)
    df = df.loc[df.index >= config.START_YEAR].copy()
    all_carbon.append(df)

all_bio = []
for input_file in carbon_bio_names:
    df = pd.read_excel(os.path.join(excel_path, f'2_{input_file}_cost_series.xlsx'), index_col=0)
    df = df.loc[df.index >= config.START_YEAR].copy()
    all_bio.append(df)

carbon_ymin = min(df.columns[:3].min() for df in all_carbon)
carbon_ymax = max(df.columns[:3].max() for df in all_carbon)

bio_ymin = min(df.columns[:3].min() for df in all_bio)
bio_ymax = max(df.columns[:3].max() for df in all_bio)

# --- 4. 创建 3x5 子图布局 ---
set_plot_style(font_size=24)
fig = plt.figure(figsize=(20, 12))
color = 'green'
# 步骤 1: 创建一个 3x5 的主网格
gs = gridspec.GridSpec(3, 5, figure=fig, hspace=0.5, wspace=0.2)

# --- 绘制 Carbon 图 (第一行前两个) ---
ax_carbon_list = []
for i in range(2):
    ax = fig.add_subplot(gs[0, i])
    drwa_fit_line(ax, all_carbon[i],color=color,title_name=title_carbon_names[i])
    ax.set_ylim(carbon_ymin, carbon_ymax)
    x_data = all_bio[i].index
    start_tick, middle_tick, end_tick = x_data.min(), x_data[len(x_data) // 2], x_data.max()
    tick_positions = [start_tick, middle_tick, end_tick]
    ax.tick_params(axis='x', labelbottom=False)
    ax_carbon_list.append(ax)

# 设置共享Y轴，并控制刻度标签
# 修改 GHGhigh 图的 Y 轴刻度
ax_carbon_list[1].set_yticks([0,10,20])
ax_carbon_list[1].set_yticklabels(["0", "10", "20"])
ax_carbon_list[0].sharey(ax_carbon_list[1])
ax_carbon_list[1].tick_params(axis='y', labelleft=False)  # 只让最左边的显示Y刻度

# --- 创建图例区域 (第一行后三个合并) ---
legend_ax = fig.add_subplot(gs[0, 2:])  # 使用切片 gs[0, 2:] 来合并单元格
legend_ax.axis('off')  # 关闭坐标轴

# --- 绘制 Bio 图 (后两行) ---
ax_bio_list = []
shared_bio_ax = None  # 用于共享Y轴的参考轴

for i in range(10):
    row = i // 5 + 1  # +1 让行号从 1 和 2 开始 (主网格的第二、三行)
    col = i % 5

    # 共享Y轴的设置
    if shared_bio_ax is None:
        ax = fig.add_subplot(gs[row, col])
        shared_bio_ax = ax  # 第一个bio图作为共享Y轴的基准
    else:
        ax = fig.add_subplot(gs[row, col], sharey=shared_bio_ax)

    drwa_fit_line(ax, all_bio[i],color=color,title_name=title_bio_names[i])

    # 控制Y轴刻度：只在最左边一列 (col==0) 显示
    if col != 0:
        ax.tick_params(axis='y', labelleft=False)

    # 控制X轴刻度：只在最下面一行 (row==2) 显示
    x_data = all_bio[i].index
    start_tick, middle_tick, end_tick = x_data.min(), x_data[len(x_data) // 2], x_data.max()
    tick_positions = [start_tick, middle_tick, end_tick]

    ax.set_xticks(tick_positions)
    if row == 2:
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

    ax_bio_list.append(ax)

# 在基准轴上设置Y轴范围
# shared_bio_ax.set_ylim(bio_ymin, bio_ymax)
shared_bio_ax.set_ylim(0, 600)
shared_bio_ax.set_yticks([0, 200,400,600])

# --- 5. 添加全局Y轴标签 ---
ax_carbon_list[0].set_ylabel(r"Carbon price (AU\$ tCO$_2$e$^{-1}$)")
ax_bio_list[0].set_ylabel(r"Biodiversity price (AU\$ ha$^{-1}$)")
ax_carbon_list[0].yaxis.set_label_coords(-0.3, 0.3)
ax_bio_list[0].yaxis.set_label_coords(-0.3, -0.4)

# fig.align_ylabels([ax_carbon_list[0], ax_bio_list[0]])

line_handle = mlines.Line2D([], [], color=color, linewidth=2, label="Quadratic fit")
shade_handle = Patch.Patch(color=color, alpha=0.25, label="95% CI")
leg = fig.legend(
    handles=[line_handle, shade_handle],
    bbox_to_anchor=(0.8, 0.8),
    ncol=2
)

# 设置无背景 & 无边框
leg.get_frame().set_facecolor('none')   # 背景透明
leg.get_frame().set_edgecolor('none')   # 去掉边框

plt.savefig(os.path.join(figure_path, '06_average price.png'), dpi=300, bbox_inches='tight')
plt.show()