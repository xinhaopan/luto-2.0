import matplotlib.patches as mpatches
from itertools import cycle
import matplotlib.pyplot as plt
import numpy as np
from itertools import cycle
import pandas as pd

from tools.parameters import *
from tools.data_helper import *
from tools.plot_helper import *



def calculate_transition_matrix(land_use_data_path, mapping_data_path, area_column='Area (ha)',
                                        from_column='From land-use', to_column='To land-use',
                                        desc_column='desc', ag_non_ag_group_column='ag_non_ag_group'):
    """
    Calculate a transition matrix for land-use changes, ordered by the 'ag_non_ag_group' column in the mapping data.
    NaN values are removed from the final transition matrix.

    Parameters:
    - land_use_data_path (str): Path to the CSV file containing land-use data with 'From land-use', 'To land-use', and 'Area (ha)' columns.
    - mapping_data_path (str): Path to the CSV file containing the mapping with 'desc' and 'ag_non_ag_group' columns.
    - area_column (str): Column name for area values. Default is 'Area (ha)'.
    - from_column (str): Column name for the initial land-use type. Default is 'From land-use'.
    - to_column (str): Column name for the target land-use type. Default is 'To land-use'.
    - desc_column (str): Column name in the mapping file for detailed land-use types. Default is 'desc'.
    - ag_non_ag_group_column (str): Column name in the mapping file for grouped land-use types. Default is 'ag_non_ag_group'.

    Returns:
    - pd.DataFrame: Transition matrix with aggregated land-use change areas, ordered by the specified grouping column.
    """
    # Load the main data and the mapping table
    land_use_data = pd.read_csv(land_use_data_path)
    mapping_data = pd.read_excel(mapping_data_path)

    # Create dictionaries for grouping based on mapping
    group_mapping = mapping_data.set_index(desc_column)[ag_non_ag_group_column].to_dict()
    ordered_groups = mapping_data[ag_non_ag_group_column].dropna().drop_duplicates().tolist()

    # Map the grouping to 'From' and 'To' columns
    land_use_data['From_group'] = land_use_data[from_column].map(group_mapping)
    land_use_data['To_group'] = land_use_data[to_column].map(group_mapping)

    # Drop rows where mapping was not found
    land_use_data.dropna(subset=['From_group', 'To_group'], inplace=True)

    # Group by the new grouped land-use types and sum the areas
    transition_matrix = land_use_data.groupby(['From_group', 'To_group'])[area_column].sum().unstack(fill_value=0)

    # Reindex the matrix to ensure it follows the order in `ag_non_ag_group`
    transition_matrix = transition_matrix.reindex(index=ordered_groups, columns=ordered_groups, fill_value=0)

    return transition_matrix

# def plot_all_transition_matrices(matrices, labels_list, label_mapping, output_path="transition_matrix_all.png", font_size=20):
#     """
#     Plots multiple transition matrices with shorthand labels on a grid layout.
#
#     Parameters:
#     - matrices: List[np.array or pd.DataFrame], transition matrices to plot.
#     - labels_list: List[List[str]], list of label sets (one for each matrix).
#     - label_mapping: Dict[str, str], maps full land-use names to shorthand labels.
#     - output_path: str, path to save the output image.
#     - font_size: int, font size for labels and tick marks.
#     """
#     plt.rcParams.update({'font.size': font_size, 'font.family': 'Arial'})
#
#     fig, axes = plt.subplots(3, 3, figsize=(18, 18))
#     fig.subplots_adjust(hspace=0.15, wspace=0.15)
#
#     for i, (matrix, labels) in enumerate(zip(matrices, cycle(labels_list))):
#         # Ensure matrix is a NumPy array for imshow
#         if isinstance(matrix, pd.DataFrame):
#             full_labels = matrix.columns.tolist()
#             short_labels = [label_mapping.get(label, label) for label in full_labels]
#             matrix = matrix.values
#         else:
#             short_labels = [label_mapping.get(label, label) for label in labels]
#
#         # Normalize the matrix by row sums to calculate proportions
#         row_sums = matrix.sum(axis=1, keepdims=True)
#         normalized_matrix = np.divide(
#             matrix,
#             row_sums,
#             out=np.zeros_like(matrix, dtype=float),
#             where=row_sums != 0
#         )
#
#         ax = axes[i // 3, i % 3]
#         # Use normalized matrix for color mapping, original matrix for text
#         cax = ax.imshow(normalized_matrix, cmap='YlOrRd', interpolation='nearest')
#
#         # Set the tick labels
#         ax.set_xticks(np.arange(len(short_labels)))
#         ax.set_yticks(np.arange(len(short_labels)))
#         ax.set_xticklabels(short_labels, ha="center", fontsize=font_size)
#         ax.set_yticklabels(short_labels, fontsize=font_size)
#
#         # Display original values (area) within each cell
#         for x in range(len(short_labels)):
#             for y in range(len(short_labels)):
#                 value = matrix[x, y]  # Original value (area)
#                 proportion = normalized_matrix[x, y]  # Normalized value (proportion)
#                 color = "white" if proportion > 0.5 else "black"
#                 ax.text(y, x, f"{value:.1f}", ha="center", va="center", color=color, fontsize=font_size)
#
#     # Add color bar
#     cbar_ax = fig.add_axes([0.15, 0.06, 0.7, 0.02])  # 设置颜色条的位置
#     cbar = fig.colorbar(cax, cax=cbar_ax, orientation='horizontal')
#
#     # 手动设置颜色条的范围，使其映射到 0-1 的数据比例范围
#     cbar.mappable.set_clim(0, 1)  # 设置颜色条的颜色范围为 0 到 1
#
#     # 设置刻度为归一化的比例
#     cbar.set_ticks([0, 0.25, 0.5, 0.75, 1.0])  # 颜色条刻度位置
#     cbar.set_ticklabels(['0%', '25%', '50%', '75%', '100%'])  # 设置对应的刻度标签
#
#     # 设置颜色条标签
#     cbar.set_label('Proportion of Transition', fontsize=font_size)
#
#     # 设置颜色条刻度字体大小
#     cbar.ax.tick_params(labelsize=font_size)
#
#     # Save the plot
#     plt.savefig(f'{output_path}.png', bbox_inches='tight', dpi=300)
#     save_figure(fig, output_path)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from itertools import cycle


def save_colorbar(cax, output_path, font_size=20):
    """
    保存单独的颜色条为图片文件。

    参数:
    - cax: matplotlib colorbar 对象
    - output_path: str, 保存颜色条的路径
    - font_size: int, 字体大小
    """
    # 创建新图用于保存颜色条
    fig, ax = plt.subplots(figsize=(6, 1))
    fig.subplots_adjust(bottom=0.5)

    # 创建颜色条
    cbar = fig.colorbar(cax, cax=ax, orientation='horizontal')
    cbar.mappable.set_clim(0, 1)  # 设置颜色条范围为 0 到 1
    cbar.set_ticks([0, 0.25, 0.5, 0.75, 1.0])  # 颜色条刻度位置
    cbar.ax.tick_params(labelbottom=False, labeltop=False, labelsize=font_size)
    # cbar.set_ticklabels(['0%', '25%', '50%', '75%', '100%'])  # 设置刻度标签
    # cbar.set_label('Proportion of Transition', fontsize=font_size)  # 设置标签
    # cbar.ax.tick_params(labelsize=font_size)  # 设置刻度字体大小

    # 保存颜色条为单独文件
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close(fig)


def plot_all_transition_matrices(matrices, labels_list, label_mapping, output_path="transition_matrix_all.png",
                                 font_size=20):
    """
    绘制多个转移矩阵，并将图例单独保存。

    参数:
    - matrices: List[np.array or pd.DataFrame], 转移矩阵列表。
    - labels_list: List[List[str]], 标签列表（每个矩阵一组标签）。
    - label_mapping: Dict[str, str], 标签映射，用于将完整名称映射为短名称。
    - output_path: str, 主图保存路径。
    - font_size: int, 字体大小。
    """
    plt.rcParams.update({'font.size': font_size, 'font.family': 'Arial'})

    fig, axes = plt.subplots(3, 3, figsize=(18, 18))
    fig.subplots_adjust(hspace=0.15, wspace=0.15)

    cax = None  # 初始化颜色条对象
    for i, (matrix, labels) in enumerate(zip(matrices, cycle(labels_list))):
        # 确保矩阵是 NumPy 数组
        if isinstance(matrix, pd.DataFrame):
            full_labels = matrix.columns.tolist()
            short_labels = [label_mapping.get(label, label) for label in full_labels]
            matrix = matrix.values
        else:
            short_labels = [label_mapping.get(label, label) for label in labels]

        # 归一化矩阵
        row_sums = matrix.sum(axis=1, keepdims=True)
        normalized_matrix = np.divide(
            matrix,
            row_sums,
            out=np.zeros_like(matrix, dtype=float),
            where=row_sums != 0
        )

        ax = axes[i // 3, i % 3]
        # 使用归一化矩阵作为颜色映射，原始矩阵用于显示数值
        cax = ax.imshow(normalized_matrix, cmap='YlOrRd', interpolation='nearest')

        # 设置刻度标签
        ax.set_xticks(np.arange(len(short_labels)))
        ax.set_yticks(np.arange(len(short_labels)))
        ax.set_xticklabels(short_labels, ha="center", fontsize=font_size)
        ax.set_yticklabels(short_labels, fontsize=font_size)

        # 在每个单元格中显示原始值
        for x in range(len(short_labels)):
            for y in range(len(short_labels)):
                value = matrix[x, y]
                proportion = normalized_matrix[x, y]
                color = "white" if proportion > 0.5 else "black"
                ax.text(y, x, f"{value:.1f}", ha="center", va="center", color=color, fontsize=font_size)

    # 保存主图
    plt.savefig(f'{output_path}', bbox_inches='tight', dpi=300)

    # 将颜色条保存为单独文件
    if cax is not None:  # 确保cax非空
        legend_output_path = f'{output_path}_legend.png'
        save_colorbar(cax, legend_output_path, font_size)
    plt.show()
    plt.close(fig)

mapping_data_path = 'tools/land use group.xlsx'
matrices = []
for input_dir in input_files:
    file_dir = get_path(input_dir)
    land_use_data_path = os.path.join(file_dir, 'out_2050/transition_matrix_2010_2050.csv')
    matrice = calculate_transition_matrix(land_use_data_path, mapping_data_path)
    matrices.append(matrice / 1e6)
labels = list(matrice)

plt.rcParams['font.family'] = 'Arial'
label_mapping = {
    'Crops': 'C',
    'Livestock': 'L',
    'Unallocated - modified land': 'UM',
    'Unallocated - natural land': 'UN',
    'Non-agricultural land-use': 'NA'
}
plot_all_transition_matrices(matrices, labels,label_mapping, output_path="../output/06_transition_matrix_all")
