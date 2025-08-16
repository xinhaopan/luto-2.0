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

# 原始列名 -> 展示标题（并据此分组）
COLUMNS_NAME = [
    "cost_ag(M$)", "cost_am(M$)", "cost_non-ag(M$)", "cost_transition_ag2ag(M$)",
    "cost_amortised_transition_ag2non-ag(M$)", "revenue_ag(M$)", "revenue_am(M$)", "revenue_non-ag(M$)",
    "GHG_ag(MtCOe2)", "GHG_am(MtCOe2)", "GHG_non-ag(MtCOe2)", "GHG_transition(MtCOe2)",
    "BIO_ag(M ha)", "BIO_am(M ha)", "BIO_non-ag(M ha)"
]
TITLE_NAME = [
    "Ag cost", "AM cost", "Non-ag cost", "Transition cost (AG→AG)",
    "Amortised transition cost (AG→Non-ag)", "Ag revenue", "AM revenue", "Non-ag revenue",
    "Ag GHG emissions", "AM GHG emissions", "Non-ag GHG emissions", "Transition GHG emissions",
    "Ag biodiversity", "AM biodiversity", "Non-ag biodiversity"
]
COL_MAP = dict(zip(COLUMNS_NAME, TITLE_NAME))

def _format_yaxis(ax):
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    y0, y1 = ax.get_ylim()
    if abs(y1 - y0) < 3:
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.2f}"))
    else:
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:,.0f}"))

def plot_data(df0, df1, df2, name0, name1, name2):
    """三套数据对比的多子图 + 底部统一图例 + 左侧单位标签"""
    start_year = getattr(config, "START_YEAR", None)
    if start_year is not None:
        df0 = df0[df0["Year"] >= start_year]
        df1 = df1[df1["Year"] >= start_year]
        df2 = df2[df2["Year"] >= start_year]
    # 只保留需要的列，并按既定顺序排列（便于单位分组）
    keep = [c for c in COLUMNS_NAME if c in df0.columns]
    y_columns = [c for c in keep if c != "Year"]

    # 展示标题列表（等长于 y_columns）
    y_titles = [COL_MAP.get(c, c) for c in y_columns]

    n_cols = 4
    n_rows = (len(y_columns) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(4.0 * n_cols, 3.2 * n_rows + 0.6),
                             constrained_layout=False)
    axes = axes.flatten()

    series = [
        {"df": df0, "label": name0, "color": "#e74c3c", "marker": "o"},
        {"df": df1, "label": name1, "color": "#ffb216", "marker": "s"},
        {"df": df2, "label": name2, "color": "#6db50a", "marker": "^"},
    ]

    legend_handles, legend_labels = [], []

    # 起止年用于设 tick
    if "Year" in df0.columns:
        years_all = df0["Year"].dropna()
        xmin, xmax = int(years_all.min()), int(years_all.max())
    else:
        raise ValueError("Input dataframes must contain a 'Year' column.")

    for i, (col, title) in enumerate(zip(y_columns, y_titles)):
        ax = axes[i]

        for s in series:
            d = s["df"]
            if col not in d.columns or d[col].isna().all():
                continue
            p = sns.lineplot(
                data=d, x="Year", y=col, ax=ax,
                linewidth=1.6, marker=s["marker"], markersize=6,
                color=s["color"], markeredgecolor=s["color"], label=s["label"]
            )
            # 收集唯一图例
            if s["label"] not in legend_labels:
                legend_handles.append(p.lines[-1])
                legend_labels.append(s["label"])

        # 轴格式
        ax.set_title(title, pad=6)
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

    # 底部统一图例
    if len(y_columns) < len(axes):
        legend_ax = axes[len(y_columns)]  # 右下空白轴
        legend_ax.axis("off")
        if legend_handles:
            legend_ax.legend(
                legend_handles, legend_labels,
                loc="center", frameon=False,
                handlelength=2.0
            )

    # ------- 按类别添加左侧单位标签 -------
    # 找到每一类的“第一张图”的行号，用该行的轴位置确定文字的 y 坐标
    def row_center_y(row_idx):
        # 该行有效轴的 bbox 求中位
        row_axes = [axes[row_idx * n_cols + j] for j in range(n_cols)
                    if row_idx * n_cols + j < len(y_columns)]
        y0 = min(ax.get_position().y0 for ax in row_axes)
        y1 = max(ax.get_position().y1 for ax in row_axes)
        return (y0 + y1) / 2.0

    # 按标题判断类别位置
    first_cost = next((i for i, t in enumerate(y_titles) if "cost" in t.lower() or "revenue" in t.lower()), None)
    first_ghg  = next((i for i, t in enumerate(y_titles) if "ghg"  in t.lower()), None)
    first_bio  = next((i for i, t in enumerate(y_titles) if "biodiversity" in t.lower()), None)

    if first_cost is not None:
        rc = first_cost // n_cols
        fig.text(0.05, row_center_y(rc), "Cost (Million AU$)", rotation="vertical", va="center", ha="center")
    if first_ghg is not None:
        rg = first_ghg // n_cols
        fig.text(0.05, row_center_y(rg), "GHG emission (tCO2e)", rotation="vertical", va="center", ha="center")
    if first_bio is not None:
        rb = first_bio // n_cols
        fig.text(0.05, row_center_y(rb), "Biodiversity restoration (Mha)", rotation="vertical", va="center", ha="center")

    # 布局
    plt.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.12,
                        hspace=0.32, wspace=0.25)
    return fig

# ------------------- Load & merge -------------------
input_files = config.INPUT_FILES
name0, name1, name2 = input_files[2], input_files[1], input_files[0]
excel_path = f"{config.TASK_DIR}/carbon_price/excel"

df0 = pd.read_excel(os.path.join(excel_path, f"01_origin_{name0}.xlsx"))
df1 = pd.read_excel(os.path.join(excel_path, f"01_origin_{name1}.xlsx"))
df2 = pd.read_excel(os.path.join(excel_path, f"01_origin_{name2}.xlsx"))
df2 = df2.drop(columns=df2.columns[df2.columns.str.contains('GHG_')])
# 合并 process 文件（若存在）
def merge_process(df, name, usecols=range(5)):
    proc = os.path.join(excel_path, f"02_process_{name}.xlsx")
    if os.path.exists(proc):
        df_proc = pd.read_excel(proc, usecols=usecols)
        df = df.merge(df_proc, on="Year", how="left")
    return df

df0 = merge_process(df0, name0)
df1 = merge_process(df1, name1)
df2 = merge_process(df2, name2)

# 如需置空某些列（按你之前的索引需求，做存在性检查）
if df0.shape[1] >= 17:
    df0.iloc[:, 10:17] = None
if df1.shape[1] >= 17:
    df1.iloc[:, 14:17] = None

# ------------------- Plot & save -------------------
fig = plot_data(df0, df1, df2, 'Counterfactual', 'GHG', 'GHG & Biodiversity')

outdir = f"{config.TASK_DIR}/carbon_price/Paper_figure"
os.makedirs(outdir, exist_ok=True)
fig.savefig(os.path.join(outdir, "01_point_line.png"),
            dpi=300, bbox_inches="tight")
plt.show()
