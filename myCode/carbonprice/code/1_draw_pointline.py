import os
import math
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import FuncFormatter, MaxNLocator
import tools.config as config

# ------------------- Style -------------------
# 先设 darkgrid 背景
sns.set_theme(style="darkgrid")
plt.rcParams.update({
    "xtick.bottom": True, "ytick.left": True,   # 打开刻度
    "xtick.top": False,  "ytick.right": False,  # 需要的话也可开
    "xtick.direction": "out", "ytick.direction": "out",
    "xtick.major.size": 4, "ytick.major.size": 4,
    "xtick.major.width": 1.2, "ytick.major.width": 1.2,
})


def set_plot_style(font_size=12, font_family='Arial'):
    mpl.rcParams.update({
        'font.size': font_size, 'font.family': font_family,
        'axes.titlesize': font_size, 'axes.labelsize': font_size,
        'xtick.labelsize': font_size, 'ytick.labelsize': font_size,
        'legend.fontsize': font_size, 'figure.titlesize': font_size
    })

set_plot_style(font_size=12, font_family='Arial')


def _format_yaxis(ax):
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    y0, y1 = ax.get_ylim()
    if abs(y1 - y0) < 3:
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.2f}"))
    else:
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:,.0f}"))

def plot_data(df):
    df = df[df.index >= config.START_YEAR]

    y_columns = df.columns.tolist()

    n_cols = 4
    n_rows = (len(y_columns) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(4.0 * n_cols, 3.2 * n_rows + 0.6),
                             constrained_layout=False)
    axes = axes.flatten()


    years_all = df.index
    xmin, xmax = int(years_all.min()), int(years_all.max())

    for i, col in enumerate(y_columns):
        ax = axes[i]
        data = df[col]
        ax.plot(years_all, data, marker='o', markersize=4, linewidth=2, label=col,color='red')


        # 轴格式
        ax.set_title(col, pad=6)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_xlim(xmin - 0.5, xmax + 0.5)
        ax.set_xticks(range(xmin, xmax + 1, 5))
        _format_yaxis(ax)

        # # 视觉样式
        # ax.set_facecolor("#E4E4E4")
        # ax.grid(True, color="white", linewidth=2)
        ax.tick_params(bottom=True, left=True)
        # 设置绘图区四条边框（spines）的样式
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)
            spine.set_edgecolor("black")

        # 去局部图例
        leg = ax.get_legend()
        if leg:
            leg.remove()

    # 关闭空轴
    for k in range(len(y_columns), len(axes)):
        axes[k].axis("off")


    fig.text(0.05, 0.5, "Cost (Million AU$)", rotation="vertical", va="center", ha="center")
    # 布局
    plt.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.12,
                        hspace=0.32, wspace=0.25)
    return fig



# ------------------- Plot & save -------------------
path = f"../../../output/{config.TASK_NAME}/carbon_price"
file_name = 'Run_13_GHG_off_BIO_off_CUT_50'
df = pd.read_excel(f"{path}/1_excel/0_Origin_economic_{file_name}.xlsx",index_col=0)

cols_to_drop = [col for col in df.columns if 'Transition(ag→non-ag) cost' in col]
df = df.drop(columns=cols_to_drop)

fig = plot_data(df)

outdir = f"{path}/carbon_price/2_figure"
os.makedirs(outdir, exist_ok=True)
fig.savefig(os.path.join(outdir, f"01_cost_{file_name}.png"),
            dpi=300, bbox_inches="tight")
plt.show()
