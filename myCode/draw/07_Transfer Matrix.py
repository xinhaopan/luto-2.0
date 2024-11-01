import numpy as np
import pandas as pd
import os
import openchord as ocd
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import cairosvg
import matplotlib.image as mpimg
from matplotlib.gridspec import GridSpec
from tools.helper import *
from tools.parameters import *
from itertools import cycle

plt.rcParams['font.family'] = 'Arial'


def generate_transition_matrix_grouped(input_dir, lumap_csv):
    # 读取lumap_grouped.csv文件，获取lu_desc_0和lu_desc列
    lumap_df = pd.read_csv(lumap_csv)
    lu_desc_0_to_desc = dict(zip(lumap_df['lu_desc_0'], lumap_df['lu_desc']))

    # 读取2010年的农业和非农业用地数据
    df_agri_start = pd.read_csv(f"{input_dir}/out_2010/area_agricultural_landuse_2010.csv")
    df_non_agri_start = pd.read_csv(f"{input_dir}/out_2010/area_non_agricultural_landuse_2010.csv")
    df_start = pd.concat([df_agri_start, df_non_agri_start], ignore_index=True)
    df_start['New Land-use'] = df_start['Land-use'].map(lu_desc_0_to_desc)  # 映射为新的类别名称
    area_start = df_start.groupby('New Land-use', as_index=False)['Area (ha)'].sum()
    area_start['Area (m ha)'] = area_start['Area (ha)'] / 1_000_000  # 转换为百万公顷

    # 读取2050年的农业和非农业用地数据
    df_agri_end = pd.read_csv(f"{input_dir}/out_2050/area_agricultural_landuse_2050.csv")
    df_non_agri_end = pd.read_csv(f"{input_dir}/out_2050/area_non_agricultural_landuse_2050.csv")
    df_end = pd.concat([df_agri_end, df_non_agri_end], ignore_index=True)
    df_end['New Land-use'] = df_end['Land-use'].map(lu_desc_0_to_desc)  # 同样映射为新的类别名称
    area_end = df_end.groupby('New Land-use', as_index=False)['Area (ha)'].sum()
    area_end['Area (m ha)'] = area_end['Area (ha)'] / 1_000_000  # 转换为百万公顷

    # 保持新的类别合并2010和2050的数据
    merged_area = pd.merge(
        area_start[['New Land-use', 'Area (m ha)']],
        area_end[['New Land-use', 'Area (m ha)']],
        on='New Land-use', how='outer', suffixes=('_2010', '_2050')
    ).fillna(0)

    # 根据合并后的新类别名称创建转移矩阵
    labels = merged_area['New Land-use'].tolist()
    n = len(labels)
    transition_matrix = np.zeros((n, n))

    # 填充转移矩阵，计算2010和2050面积的变化
    for i in range(n):
        for j in range(n):
            if i != j:
                transition_matrix[i][j] = abs(merged_area['Area (m ha)_2050'][j] - merged_area['Area (m ha)_2010'][i])

    return transition_matrix, labels

def generate_transition_matrix(input_dir):
    # 读取2010年的农业和非农业用地数据
    df_agri_start = pd.read_csv(f"{input_dir}/out_2010/area_agricultural_landuse_2010.csv")
    df_non_agri_start = pd.read_csv(f"{input_dir}/out_2010/area_non_agricultural_landuse_2010.csv")
    df_start = pd.concat([df_agri_start, df_non_agri_start], ignore_index=True)
    area_start = df_start.groupby('Land-use', as_index=False)['Area (ha)'].sum()
    area_start['Area (m ha)'] = area_start['Area (ha)'] / 1_000_000

    # 读取2050年的农业和非农业用地数据
    df_agri_end = pd.read_csv(f"{input_dir}/out_2050/area_agricultural_landuse_2050.csv")
    df_non_agri_end = pd.read_csv(f"{input_dir}/out_2050/area_non_agricultural_landuse_2050.csv")
    df_end = pd.concat([df_agri_end, df_non_agri_end], ignore_index=True)
    area_end = df_end.groupby('Land-use', as_index=False)['Area (ha)'].sum()
    area_end['Area (m ha)'] = area_end['Area (ha)'] / 1_000_000

    merged_area = pd.merge(
        area_start[['Land-use', 'Area (m ha)']],
        area_end[['Land-use', 'Area (m ha)']],
        on='Land-use', how='outer', suffixes=('_2010', '_2050')
    ).fillna(0)

    labels = merged_area['Land-use'].tolist()
    n = len(labels)
    transition_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if i != j:
                transition_matrix[i][j] = abs(merged_area['Area (m ha)_2050'][j] - merged_area['Area (m ha)_2010'][i])

    return transition_matrix, labels


def plot_transition_matrix(matrix, labels, title="", output_path="transition_matrix.png", cmap='YlOrRd', font_size=12):
    # 创建标签缩写和完整名称的映射
    label_mapping = {
        'C': 'Crops',
        'L': 'Livestock',
        'NA': 'Non-Agricultural',
        'UM': 'Urban Managed',
        'UN': 'Urban Natural'
    }
    # 使用缩写作为标签
    short_labels = ['C', 'L', 'NA', 'UM', 'UN']

    fig, ax = plt.subplots(figsize=(12, 10))

    # 绘制矩阵
    cax = ax.imshow(matrix, cmap=cmap, interpolation='nearest')

    # 设置颜色条
    cbar = fig.colorbar(cax, ax=ax, orientation='vertical', fraction=0.02, pad=0.04)
    cbar.set_label('Transition Area (M ha)', fontsize=font_size)
    cbar.ax.tick_params(labelsize=font_size)

    # 设置x轴和y轴标签
    ax.set_xticks(np.arange(len(short_labels)))
    ax.set_yticks(np.arange(len(short_labels)))
    ax.set_xticklabels(short_labels, ha="center", fontsize=font_size)  # 不旋转
    ax.set_yticklabels(short_labels, fontsize=font_size)
    ax.set_xlabel("2050年土地利用类型", fontsize=font_size)
    ax.set_ylabel("2010年土地利用类型", fontsize=font_size)
    ax.set_title(title, fontsize=font_size)

    # 在每个格子中显示数值
    for i in range(len(short_labels)):
        for j in range(len(short_labels)):
            value = matrix[i, j]
            color = "white" if value > np.max(matrix) / 2 else "black"
            ax.text(j, i, f"{value:.1f}", ha="center", va="center", color=color, fontsize=font_size)

    plt.tight_layout()


import matplotlib.pyplot as plt
import numpy as np

def plot_all_transition_matrices(matrices, labels_list, output_path="transition_matrix_all.png", font_size=15):
    fig, axes = plt.subplots(3, 3, figsize=(18, 18))
    fig.subplots_adjust(hspace=0.15, wspace=0.15)

    # 缩写说明映射
    label_mapping = {
        'C': 'Crops',
        'L': 'Livestock',
        'NA': 'Non-Agricultural',
        'UM': 'Urban Managed',
        'UN': 'Urban Natural'
    }
    short_labels = ['C', 'L', 'NA', 'UM', 'UN']

    for i, (matrix, labels) in enumerate(zip(matrices, cycle(labels_list))):
        ax = axes[i // 3, i % 3]
        cax = ax.imshow(matrix, cmap='YlOrRd', interpolation='nearest')

        # 设置标签
        ax.set_xticks(np.arange(len(short_labels)))
        ax.set_yticks(np.arange(len(short_labels)))
        ax.set_xticklabels(short_labels, ha="center", fontsize=font_size)
        ax.set_yticklabels(short_labels, fontsize=font_size)
        ax.set_title(f"({chr(97 + i)})", fontsize=font_size+5)

        # 在每个格子中显示数值
        for x in range(len(short_labels)):
            for y in range(len(short_labels)):
                value = matrix[x, y]
                color = "white" if value > np.max(matrix) / 2 else "black"
                ax.text(y, x, f"{value:.1f}", ha="center", va="center", color=color, fontsize=font_size)

    cbar_ax = fig.add_axes([0.15, 0.06, 0.7, 0.02])  # 将颜色条放在图形下方
    cbar = fig.colorbar(cax, cax=cbar_ax, orientation='horizontal')

    # 设置颜色条标签和刻度大小
    cbar.set_label('Transition Area (M ha)', fontsize=font_size+5)  # 标签的字体大小
    cbar.ax.tick_params(labelsize=font_size+5)  # 刻度标签的字体大小

    # 添加缩写说明文本
    # description_text = ' '.join([f"{abbr}: {full_name}" for abbr, full_name in label_mapping.items()])
    # plt.figtext(0.5, 0.03, description_text, ha="center", va="top", fontsize=font_size)

    plt.savefig(output_path, dpi=300, bbox_inches='tight', transparent=True)
    plt.show()


input_dir = get_path(input_files[0])
lumap_csv = 'tools/lumap_grouped.csv'
# transition_matrix, labels = generate_transition_matrix(input_dir)
# plot_transition_matrix(transition_matrix, labels,output_path='transition_matrix.png')

matrices = []
labels = None

# Assuming you have multiple directories or conditions to generate each matrix.
for input_dir in input_files:
    matrice, label = generate_transition_matrix_grouped(get_path(input_dir), lumap_csv)
    matrices.append(matrice)
    # Only set `labels` once if all matrices have the same labels
    if labels is None:
        labels = label

# Call the function to plot all matrices
plot_all_transition_matrices(matrices, labels, output_path="07_transition_matrix_all.png")
