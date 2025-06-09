import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import matplotlib as mpl

import numpy as np
import os
import gzip
import dill

from tools.plot_helper import *

# Set SVG output to embed fonts as text
mpl.rcParams['svg.fonttype'] = 'none'

# Set global font to Arial
matplotlib.rcParams['font.family'] = 'Arial'

def beautify_transition_matrix(tmat, cell_colors=None, output_png="output/15_transition_S2"):
    """
    按指定规则美化并输出 transition matrix 的二值可视化表格（不扩展行列）

    Parameters
    ----------
    tmat : 2D array-like
        待展示的转移矩阵（可含NaN）
    cell_colors : dict, optional
        颜色映射。例如 {1: "#C1DDB2", 0: "#FBD5D5"}
    output_png : str
        输出文件路径（不含扩展名）
    """
    n_rows, n_cols = tmat.shape
    # 二值化（1=有值，0=NaN）
    binary_array = np.where(np.isnan(tmat), 0, 1)
    df = pd.DataFrame(binary_array)

    colors = cell_colors or {1: "#C1DDB2", 0: "#FBD5D5"}

    fig, ax = plt.subplots(figsize=(min(2 + 0.35 * df.shape[1], 40), min(6 + 0.6 * df.shape[0], 40)))

    table = ax.table(
        cellText=[[""] * len(df.columns) for _ in range(len(df))],
        rowLabels=[f"{i+1}" for i in df.index],
        colLabels=[f"{j+1}" for j in df.columns],
        cellLoc='center',
        loc='center'
    )

    table.auto_set_font_size(False)
    table.set_fontsize(20)

    for (row, col), cell in table.get_celld().items():
        if row > 0 and col >= 0:
            value = df.iloc[row-1, col]
            cell.set_facecolor(colors.get(value, "white"))
            cell.set_edgecolor("white")
            cell.get_text().set_text("")
        if row == 0 or col == -1:
            cell.set_linewidth(0)
        if col == -1:
            cell.get_text().set_horizontalalignment('right')
        if row == 0 and col >= 0:
            cell.get_text().set_rotation(90)
            cell.get_text().set_horizontalalignment('center')
            cell.get_text().set_verticalalignment('bottom')

    table.scale(1.3, 2.5)
    ax.axis('off')
    # 你自己的保存函数
    if 'save_figure' in globals():
        save_figure(fig, output_png)
    else:
        fig.savefig(output_png + ".png", bbox_inches='tight', dpi=300)
        fig.savefig(output_png + ".svg", bbox_inches='tight')
    plt.close(fig)


gz_path = '../../../output/2025_06_06__17_21_25_RF9_2010-2050/data_with_solution.gz'
with gzip.open(gz_path, 'rb') as f:
    data = dill.load(f)
AG_TMATRIX = data.T_MAT
AG_TMATRIX = np.delete(AG_TMATRIX, -2, axis=0)  # 移除倒数第二行BECCS
AG_TMATRIX = np.delete(AG_TMATRIX, -2, axis=1)  # 移除倒数第二列BECCS

beautify_transition_matrix(
    AG_TMATRIX,
    cell_colors={1: "#C1DDB2", 0: "#FBD5D5"},
    output_png="../output/15_transition_S2"
)
print("输出完成！")