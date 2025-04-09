import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import matplotlib as mpl

import numpy as np
import os


from tools.plot_helper import *

# Set SVG output to embed fonts as text
mpl.rcParams['svg.fonttype'] = 'none'

# Set global font to Arial
matplotlib.rcParams['font.family'] = 'Arial'


AG_TMATRIX = np.load(os.path.join('../../../input', "ag_tmatrix.npy"))
n_rows, n_cols = AG_TMATRIX.shape
# 创建扩展后的数组
extended_array = np.full((n_rows + 8, n_cols + 8), np.nan)  # 默认填充 NaN

# 复制原始数据到新数组
extended_array[:n_rows, :n_cols] = AG_TMATRIX

# 设置新增的 8 列的值为 1
extended_array[:n_rows, n_cols:] = 1

# 设置新增的8列值为NaN（保持原状，不需要修改）

# 设置交接部分（右下角8x8）为1
extended_array[n_rows:, n_cols:] = 1
 # 转换规则：非 NaN 变 1，NaN 变 0
binary_array = np.where(np.isnan(extended_array), 0, 1)

# 转换为 DataFrame
df = pd.DataFrame(binary_array)


# Define colors for cells
colors = {1: "#C1DDB2", 0: "#FBD5D5"}  # Green and Red

# Create figure
fig, ax = plt.subplots(figsize=(12, 30))

# Create a table
table = ax.table(
    cellText=[[""] * len(df.columns) for _ in range(len(df))],  # Empty text cells
    rowLabels=df.index,
    colLabels=df.columns,
    cellLoc='center',
    loc='center'
)

# Adjust font size
table.auto_set_font_size(False)
table.set_fontsize(35)

# Color the cells and add white grid lines
for (row, col), cell in table.get_celld().items():
    if row > 0 and col >= 0:  # Avoid coloring headers
        value = df.iloc[row-1, col]  # Adjust row index
        cell.set_facecolor(colors.get(value, "white"))  # Default white if no match
        cell.set_edgecolor("white")  # White grid lines
        cell.get_text().set_text("")  # Remove text content

# Remove border for first row (column headers) and first column (row labels)
for (row, col), cell in table.get_celld().items():
    if row == 0 or col == -1:
        cell.set_linewidth(0)  # Remove border

# Align row labels to the right
for (row, col), cell in table.get_celld().items():
    if col == -1:
        cell.get_text().set_horizontalalignment('right')

# Rotate column labels 90°, align center and bottom
for col_idx, col_label in enumerate(df.columns):
    header_cell = table[(0, col_idx)]
    header_cell.get_text().set_rotation(90)
    header_cell.get_text().set_horizontalalignment('center')
    header_cell.get_text().set_verticalalignment('bottom')

# Adjust table cell size
table.scale(1.3, 2.5)

# Remove axes
ax.axis('off')

# Save figure
output_png = "../output/15_transition_S2"
# fig.savefig(output_png + ".png", bbox_inches='tight', dpi=300)
save_figure(fig, output_png)
# Show plot (optional)
plt.show()
