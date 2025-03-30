import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd  # 用于处理Shapefile
from mpl_toolkits.axes_grid1.inset_locator import inset_axes  # 用于嵌入地图
from PIL import Image, ImageDraw, ImageFont  # 用于图像处理和绘制标签
import json
from tools.plot_helper import *

# 获取到文件的绝对路径，并将其父目录添加到 sys.path
sys.path.append(os.path.abspath('../../../luto'))

# 导入 settings.py（假设 settings 文件位于指定路径下）
import settings

# 导入本地 helper 模块的函数
# from helper import *


def plot_shapefile_map(shapefile_path, gdf_column, df_colors, output_file='shapefile_plot.pdf', add_labels=True):
    """读取Shapefile并绘制地图，可选择是否标注区域名称，输出PDF格式."""
    import geopandas as gpd
    import matplotlib.pyplot as plt

    df_colors = dict(zip(df_colors['desc'], df_colors['color']))
    font_size = 10

    # 读取Shapefile
    gdf = gpd.read_file(shapefile_path)

    # 创建图表
    fig, ax = plt.subplots(figsize=(12, 8))

    # 绘制地图
    gdf.assign(color=gdf[gdf_column].map(df_colors)).plot(
        ax=ax, color=gdf[gdf_column].map(df_colors), edgecolor='gray'
    )

    # 隐藏轴刻度
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)

    # 是否添加区域名称标注
    if add_labels:
        for idx, row in gdf.iterrows():
            centroid = row.geometry.centroid
            name = row[gdf_column]

            # 特殊区域位置调整
            adjustments = {
                'Murray-Darling Basin': (0, 1),
                'South East Coast (NSW)': (0, 0),
                'South Australian Gulf': (0, -1.5),
                'North East Coast (QLD)': (0, -0.5),
                'North Western Plateau': (0, -0.8),
                'Timor Sea': (-1, 0.3),
                'Carpentaria Coast': (0, 0.5),
                'South West Coast': (1, -1),
                'Pilbara-Gascoyne': (0, 0),
                'Tanami': (0, -0.5),
                'Lake Eyre Basin': (0.5, 0.5)
            }

            dx, dy = adjustments.get(name, (0, 0))
            ax.text(centroid.x + dx, centroid.y + dy, name,
                    fontsize=font_size, ha='center', color='black')

    plt.tight_layout()

    # 存储为PDF
    plt.savefig(output_file, dpi=300, bbox_inches='tight', format='pdf')
    plt.close(fig)  # 关闭figure避免内存占用


def add_label_to_image(image, label, position, font_size=60, color="black"):
    """在图像上添加标签"""
    font_path = "C:/Windows/Fonts/arial.ttf"  # 设置默认字体路径
    font = ImageFont.truetype(font_path, font_size)
    # 使用 ImageDraw 在图像上绘制文本
    draw = ImageDraw.Draw(image)
    draw.text(position, label, fill=color, font=font)

    return image

plt.rcParams['font.family'] = 'Arial'
INPUT_DIR = "../../../input"

df_colors = pd.read_excel('tools/land use colors.xlsx', sheet_name='water')
shapefile_path = '../../../../Map/Data/shp/Drainage Division/ADD_2016_AUST.shp'
# plot_shapefile_map(shapefile_path, 'ADD_NAME16', df_colors, output_file='../output/03_drainage divisions')

# # 存储带text标注的地图
# plot_shapefile_map(
#     shapefile_path,
#     'ADD_NAME16',
#     df_colors,
#     output_file='../output/03_drainage divisions.pdf',
#     add_labels=False
# )
#
# # 存储不带text标注的地图
# plot_shapefile_map(
#     shapefile_path,
#     'ADD_NAME16',
#     df_colors,
#     output_file='../output/03_drainage divisions_with_text.pdf',
#     add_labels=True
# )

# water
dd = pd.read_hdf(os.path.join(INPUT_DIR, "draindiv_lut.h5"), index_col='HR_DRAINDIV_ID')
# 替换单位并新增计算列
dd['WATER_YIELD_HIST_BASELINE_GL'] = dd['WATER_YIELD_HIST_BASELINE_ML'] / 1000
# 增加一列 "Water yield limits"
dd['Water yield limits (GL)'] = dd['WATER_YIELD_HIST_BASELINE_GL'] * (1 - settings.WATER_STRESS * settings.AG_SHARE_OF_WATER_USE)
df = dd.round(2)
dd.to_excel('../output/03_limits_water.xlsx')

# 查看结果
print(dd.head())