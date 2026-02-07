import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

# 假设这些导入已在您的代码中定义
from tools.parameters import *
from tools.data_helper import *
from tools.plot_helper import *

import matplotlib.patches as patches
import pandas as pd
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.ticker as mticker

def calculate_transition_matrix(land_use_data_path, area_column='Area (ha)',
                               from_column='From-land-use', to_column='To-land-use'):
    """
    计算土地使用变化的转移矩阵，直接使用原始土地使用类型，不进行分组。
    保持 From 和 To 列中土地使用类型的原始顺序。
    最终转移矩阵中会移除 NaN 值。

    参数:
    - land_use_data_path (str): 包含土地使用数据的 CSV 文件路径。
    - area_column (str): 面积列的名称，默认是 'Area (ha)'。
    - from_column (str): 初始土地使用类型的列名，默认是 'From land-use'。
    - to_column (str): 目标土地使用类型的列名，默认是 'To land-use'。

    返回:
    - pd.DataFrame: 使用原始土地使用类型聚合的转移矩阵，保持原始顺序。
    """
    # 加载数据
    land_use_data = pd.read_csv(land_use_data_path)
    land_use_data['From'] = land_use_data[from_column]
    land_use_data['To'] = land_use_data[to_column]

    # 删除 From 或 To 为空的行
    land_use_data.dropna(subset=['From', 'To'], inplace=True)

    # 从 To 列中移除 BECCS
    land_use_data = land_use_data[land_use_data['To'] != 'BECCS']
    land_use_data = land_use_data[land_use_data['From'] != 'BECCS']

    land_use_data = land_use_data[land_use_data['To'] != 'ALL']
    land_use_data = land_use_data[land_use_data['From'] != 'ALL']

    # 获取 From 和 To 列中土地使用类型的原始顺序
    unique_land_uses = []
    seen = set()
    for land_use in pd.concat([land_use_data['From'], land_use_data['To']]).values:
        if land_use not in seen:
            seen.add(land_use)
            unique_land_uses.append(land_use)

    # 按原始土地使用类型分组并计算面积总和
    transition_matrix = land_use_data.groupby(['From', 'To'])[area_column].sum().unstack(fill_value=0)

    # 确保矩阵包含所有唯一的土地使用类型，并保持原始顺序
    transition_matrix = transition_matrix.reindex(index=unique_land_uses, columns=unique_land_uses, fill_value=0)
    # print(transition_matrix)
    return transition_matrix

def save_colorbar(cax, output_path, font_size=10):
    """
    将颜色条保存为单独的图片文件。
    """
    fig, ax = plt.subplots(figsize=(6, 1))
    fig.subplots_adjust(bottom=0.5)
    cbar = fig.colorbar(cax, cax=ax, orientation='horizontal')
    cbar.mappable.set_clim(0, 1)
    cbar.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
    cbar.ax.tick_params(labelbottom=False, labeltop=False, labelsize=font_size)
    plt.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_single_transition_matrix(matrix, labels=None, output_path="transition_matrix.png", font_size=6, figsize=(40, 6), unit_name="Proportion"):
    """
    绘制单个转移矩阵（使用前 N-8 行），x轴标签在顶部，颜色条在底部并上移，颜色条两端为尖端，刻度显示为百分比，图表更宽。
    """
    plt.rcParams.update({'font.size': font_size, 'font.family': 'Arial'})

    # 处理输入矩阵和标签
    if isinstance(matrix, pd.DataFrame):
        # 选择除最后 8 行以外的数据
        matrix = matrix.iloc[:-8]
        labels = matrix.columns.tolist()
        matrix = matrix.values
    elif labels is None:
        raise ValueError("如果矩阵不是 DataFrame，必须提供标签")
    else:
        # 如果 matrix 是 numpy 数组，选择除最后 8 行
        matrix = matrix[:-8]

    # 按行总和归一化矩阵
    row_sums = matrix.sum(axis=1, keepdims=True)
    normalized_matrix = np.divide(
        matrix,
        row_sums,
        out=np.zeros_like(matrix, dtype=float),
        where=row_sums != 0
    )

    # 创建单个图表，宽度更长
    fig, ax = plt.subplots(figsize=figsize)

    # 调整子图布局，增加左右边距以适应更宽的图表
    plt.subplots_adjust(top=0.8, bottom=0.15, left=0.15, right=0.95)

    # 绘制热力图
    cax = ax.imshow(normalized_matrix, cmap='YlOrRd', interpolation='nearest')
    aspect_ratio = matrix.shape[1]*5 / matrix.shape[0] * 0.1  # 增加宽度 1.5 倍
    ax.set_aspect(aspect_ratio)

    # 设置 x 轴标签在顶部
    ax.xaxis.set_label_position('top')
    ax.xaxis.set_ticks_position('top')
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, ha="center", fontsize=font_size, rotation=90)

    # 设置 y 轴标签（行数减少 8）
    ax.set_yticks(np.arange(matrix.shape[0]))
    ax.set_yticklabels(labels[:-8], fontsize=font_size)

    for x in range(matrix.shape[0]):
        for y in range(len(labels)):
            value = matrix[x, y]
            proportion = normalized_matrix[x, y]
            color = "white" if proportion > 0.5 else "black"
            # 使用 np.isclose 检查值是否接近 0
            text = "0" if np.isclose(value, 0, atol=1e-8) else f"{value:.1f}"
            ax.text(y, x, text, ha="center", va="center", color=color, fontsize=font_size)

    # 去掉边框（移除轴的 spines）
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    # 添加 "From:" 标签在 y 轴最上方
    ax.text(-1, -0.5, 'From:', ha='right', va='bottom', fontsize=font_size, fontweight='bold')

    # 添加 "To:" 标签在 x 轴最左侧（顶部），旋转 90 度
    ax.text(-0.5, -1.5, 'To:', ha='right', va='center', fontsize=font_size, fontweight='bold', rotation=90)

    # 添加颜色条在图表底部并上移，缩短宽度，两端为尖端
    cbar_ax = inset_axes(
        ax,
        width="60%",  # 缩短宽度
        height="5%",
        loc='lower center',
        borderpad=1,
        bbox_to_anchor=(0, -0.09, 1, 0.8),  # 上移
        bbox_transform=ax.transAxes,
    )
    cbar = fig.colorbar(
        cax, cax=cbar_ax, orientation='horizontal',
        extend='both', extendfrac=0.1, extendrect=False
    )
    cbar.mappable.set_clim(0, 1)
    cbar.set_ticks([0, 0.25, 0.5, 0.75, 1])
    cbar.set_ticklabels(['0%', '25%', '50%', '75%', '100%'])
    cbar.outline.set_visible(False)
    cbar.ax.xaxis.set_label_position('top')
    cbar.ax.tick_params(labelsize=font_size)

    # 保存主图
    # plt.tight_layout()
    plt.savefig(output_path, dpi=300)

    # 保存颜色条为单独文件
    legend_output_path = f'{output_path}_legend.png'
    # save_colorbar(cax, legend_output_path, font_size, unit_name)

    plt.show()
    plt.close(fig)

# 示例用法
file_dir = get_path(input_files[4])
pd.concat([
        pd.read_csv(f"{file_dir}/out_2050/transition_matrix_ag2ag_start_end.csv"),
        pd.read_csv(f"{file_dir}/out_2050/transition_matrix_ag2non_ag_start_end.csv")
    ]).to_csv(f"{file_dir}/out_2050/transition_matrix_2010_2050.csv", index=False)
land_use_data_path = os.path.join(file_dir, 'out_2050/transition_matrix_2010_2050.csv')
matrice = calculate_transition_matrix(land_use_data_path)

# 绘制单个转移矩阵
plot_single_transition_matrix(matrice / 10000, output_path="../output/06_transition_matrix_all", font_size=10, figsize=(14, 12))