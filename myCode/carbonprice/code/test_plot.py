# --- 导入必要的库 ---
import os
import math
import pandas as pd
import matplotlib.pyplot as plt
import patchworklib as pw
from plotnine import *
import tools.config as config
# 设置 Matplotlib 参数
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 10

# --- 数据准备 ---
# 定义列名和标题的映射
columns_name = ["cost_ag(M$)", "cost_am(M$)", "cost_non-ag(M$)", "cost_transition_ag2ag(M$)",
                "cost_amortised_transition_ag2non-ag(M$)", "revenue_ag(M$)", "revenue_am(M$)", "revenue_non-ag(M$)",
                "GHG_ag(MtCOe2)", "GHG_am(MtCOe2)", "GHG_non-ag(MtCOe2)", "GHG_transition(MtCOe2)",
                "BIO_ag(M ha)", "BIO_am(M ha)", "BIO_non-ag(M ha)"]
title_name = ["AG Cost", "AM Cost", "NON-AG Cost", "Transition Cost (AG to AG)",
              "Amortised Transition Cost (AG to NON-AG)", "AG Revenue", "AM Revenue", "NON-AG Revenue",
              "AG GHG Emission", "AM GHG Emission", "NON-AG GHG Emission", "Transition GHG Emission",
              "AG Biodiversity", "AM Biodiversity", "NON-AG Biodiversity"]
col_map = dict(zip(columns_name, title_name))

# 读取数据（请替换为你的实际文件路径）
df_ghg = pd.read_excel(f"{config.TASK_DIR}/carbon_price/excel/01_origin_Run_3_GHG_high_BIO_off.xlsx", index_col=0)
df_ghg_bio = pd.read_excel(f"{config.TASK_DIR}/carbon_price/excel/01_origin_Run_4_GHG_high_BIO_high.xlsx", index_col=0)

# 重命名列
df_ghg = df_ghg[[col for col in df_ghg.columns if col in col_map]].rename(columns=col_map)
df_ghg_bio = df_ghg_bio[[col for col in df_ghg_bio.columns if col in col_map]].rename(columns=col_map)

# 定义情景颜色
scenario_colors = {
    'GHG only': '#1f77b4',
    'GHG & BIO': '#d62728'
}


# --- 绘图函数 ---
def plot_single_column(df1, df2, column, scenario_colors, y_lab, figure_size=(3, 2.5)):
    frames = []
    # 如果 df1 中有该列，添加到 frames
    if df1 is not None and column in df1.columns:
        tmp = df1[[column]].copy()
        tmp['x'] = df1.index
        frames.append(
            tmp.rename(columns={column: 'value'}).assign(Scenario='GHG only')
        )
    # 如果 df2 中有该列，添加到 frames
    if df2 is not None and column in df2.columns:
        tmp = df2[[column]].copy()
        tmp['x'] = df2.index
        frames.append(
            tmp.rename(columns={column: 'value'}).assign(Scenario='GHG & BIO')
        )


    # 如果列在两个 DataFrame 中都不存在，返回 None
    if not frames:
        return None

    # 合并数据
    df_long = pd.concat(frames, ignore_index=True)

    # 绘制图表
    p = (
            ggplot(df_long, aes('x', 'value', color='Scenario'))
            + geom_line()
            + geom_point()
            + theme_bw()
            + theme(
        figure_size=figure_size,
        text=element_text(family='Arial', size=8.5),
        legend_position='none',  # 默认不显示图例
        panel_grid_major=element_blank(),
        panel_grid_minor=element_blank(),
        plot_title=element_text(size=10, ha='center')
    )
            + labs(x='', y=y_lab, title=column)
            + scale_y_continuous(labels=lambda l: [f'{v:,.0f}' for v in l])
            + scale_color_manual(values=scenario_colors)
    )
    return p


# --- 绘制所有列的图 ---
all_columns = list(set(df_ghg.columns) | set(df_ghg_bio.columns))
plots = []

for col in all_columns:
    # 根据列名确定 Y 轴单位
    if 'cost' in col.lower() or 'revenue' in col.lower():
        y_lab = 'Million AU$'
    elif 'ghg' in col.lower():
        y_lab = 'MtCO2e'
    elif 'bio' in col.lower():
        y_lab = 'Mha'
    else:
        y_lab = ''

    # 绘制单列图
    p = plot_single_column(
        df1=df_ghg,
        df2=df_ghg_bio,
        column=col,
        scenario_colors=scenario_colors,
        y_lab=y_lab
    )
    if p is not None:
        plots.append(p)

# 在第一张图中添加图例
plots[0] = plots[0] + theme(legend_position=(0.5, -0.1), legend_box='horizontal')

# --- 使用 Patchworklib 拼接 ---
# 转换为 Patchworklib 对象
bricks = [pw.load_ggplot(p) for p in plots]

# 每行四张图
n_col = 4
n_row = math.ceil(len(bricks) / n_col)
fig = pw.stack(bricks, operator='/', n_row=n_row, n_col=n_col, margin=0.1)

# --- 保存图像 ---
fig.savefig("combined_plot.png", dpi=300, bbox_inches='tight')
print("图像已保存为 combined_plot.png")