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
from helper import *

plt.rcParams['font.family'] = 'Arial'

def generate_transition_matrix(INPUT_DIR):
    # 读取2010年的农业和非农业用地数据
    df_agri_start = pd.read_csv(f"{INPUT_DIR}/out_2010/area_agricultural_landuse_2010.csv")
    df_non_agri_start = pd.read_csv(f"{INPUT_DIR}/out_2010/area_non_agricultural_landuse_2010.csv")

    # 合并2010年的农业和非农业用地数据
    df_start = pd.concat([df_agri_start, df_non_agri_start], ignore_index=True)
    area_start = df_start.groupby('Land-use', as_index=False, sort=False)['Area (ha)'].sum()
    area_start['Area (mkm²)'] = area_start['Area (ha)'] / 10_000_000  # 转换为百万平方公里

    # 读取2050年的农业和非农业用地数据
    df_agri_end = pd.read_csv(f"{INPUT_DIR}/out_2050/area_agricultural_landuse_2050.csv")
    df_non_agri_end = pd.read_csv(f"{INPUT_DIR}/out_2050/area_non_agricultural_landuse_2050.csv")

    # 合并2050年的农业和非农业用地数据
    df_end = pd.concat([df_agri_end, df_non_agri_end], ignore_index=True)
    area_end = df_end.groupby('Land-use', as_index=False, sort=False)['Area (ha)'].sum()
    area_end['Area (mkm²)'] = area_end['Area (ha)'] / 10_000_000  # 转换为百万平方公里

    # 保留原始顺序进行合并
    merged_area = pd.merge(
        area_start[['Land-use', 'Area (mkm²)']],
        area_end[['Land-use', 'Area (mkm²)']],
        on='Land-use', how='outer', suffixes=('_2010', '_2050'), sort=False
    ).fillna(0)

    # 创建转移矩阵
    labels = merged_area['Land-use'].tolist()  # 获取土地类型标签
    n = len(labels)
    adjacency_matrix = np.zeros((n, n))

    # 填充转移矩阵，根据2010和2050的面积差异
    for i in range(n):
        for j in range(n):
            if i != j:
                adjacency_matrix[i][j] = abs(merged_area['Area (mkm²)_2050'][j] - merged_area['Area (mkm²)_2010'][i])

    return adjacency_matrix, labels


# 删除SVG文件中的所有文本
def remove_text_from_svg(input_svg, output_svg):
    tree = ET.parse(input_svg)
    root = tree.getroot()
    namespace = {'svg': 'http://www.w3.org/2000/svg'}

    # 删除所有 <text> 元素
    for text_element in root.findall('.//svg:text', namespace):
        root.remove(text_element)

    tree.write(output_svg)

# 生成图例
def create_legend(labels, colors, output_file):
    fig, ax = plt.subplots(figsize=(10, 1.5))
    patches = [mpatches.Patch(color=color, label=label, edgecolor='black') for label, color in zip(labels, colors)]
    plt.legend(handles=patches, ncol=5, frameon=False, handleheight=0.5, handlelength=0.5)
    plt.axis('off')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)  # 减少图例四周的空白
    plt.savefig(output_file, dpi=300)
    plt.close()

# 将 SVG 转换为 PNG
def convert_svg_to_png(input_svg, output_png):
    cairosvg.svg2png(url=input_svg, write_to=output_png)

# 主程序逻辑
colors = [
    '#ff6b6b', '#ff8060', '#ff9156', '#ffa34d', '#ffb544',  # 红色到橙色过渡
    '#f97173', '#f9837d', '#f99287', '#fcc83c', '#f6db36',  # 红橙色到黄色
    '#e6ea32', '#cbe734', '#ade439', '#8fe23e', '#72da51',  # 黄色到绿色
    '#63c970', '#56b68e', '#4aa4a5', '#4892ab', '#487fae',  # 绿色到蓝色
    '#506cb1', '#5a5aa8', '#705aa3', '#673ab7', '#6b5b95',  # 蓝色到紫色
    '#895a9e', '#a15b99', '#bb5c8e', '#d55f84', '#ee6279',  # 紫色到粉红色
    '#ffb6c1', '#ff9a8d', '#ff8571', '#7ed3c4', '#4dc3ba',  # 浅粉到青色过渡
    '#2ba9a0', '#1d7a74', '#005f56'                         # 青色到深棕色过渡
]

# 处理每个输入文件
INPUT_NAMEs = ['ON_MAXPROFIT_GHG_15C_67_R5', 'ON_MAXPROFIT_GHG_15C_50_R5', 'ON_MAXPROFIT_GHG_18C_67_R5']
for INPUT_NAME in INPUT_NAMEs:
    INPUT_DIR = get_path(INPUT_NAME)

    # 生成转移矩阵
    adjacency_matrix, labels = generate_transition_matrix(INPUT_DIR)

    # 创建和保存Chord图
    fig = ocd.Chord(adjacency_matrix, labels)
    fig.colormap = colors
    fig.show_colormap()
    fig.save_svg(f'{INPUT_NAME}_chord.svg')

    # 删除SVG中的文本
    remove_text_from_svg(f'{INPUT_NAME}_chord.svg', f'{INPUT_NAME}_chord_no_text.svg')

    # 转换SVG为PNG
    convert_svg_to_png(f'{INPUT_NAME}_chord_no_text.svg', f'{INPUT_NAME}_openchord.png')

# 生成图例
create_legend(labels, colors, 'label.png')

# 加载图例和转换后的图片
legend_img = mpimg.imread('label.png')
chord_imgs = [mpimg.imread(f'{INPUT_NAME}_openchord.png') for INPUT_NAME in INPUT_NAMEs]

# 创建一个画布，指定布局
fig = plt.figure(figsize=(12, 8))
gs = GridSpec(2, 3, height_ratios=[2, 1])  # 上面放三个图，下面放图例

# 标签 (a), (b), (c)
labels = ['(a)', '(b)', '(c)']

# 绘制前三个 Chord 图
for i in range(3):
    ax = fig.add_subplot(gs[0, i])
    ax.imshow(chord_imgs[i])
    ax.axis('off')  # 不显示坐标轴
    ax.text(0.05, 0.85, labels[i], transform=ax.transAxes, fontsize=12, va='top', ha='left')

# 在第二行跨越所有三列绘制图例
ax_legend = fig.add_subplot(gs[1, :])
ax_legend.imshow(legend_img)
ax_legend.axis('off')  # 不显示坐标轴

# 调整图例的高度位置
plt.subplots_adjust(hspace=-0.55)  # hspace控制上下子图之间的间距，负值可以将图例向上移动

# 保存并显示最终合并图像
plt.savefig('06_Openchord.png', dpi=300)
plt.show()
