import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from PIL import Image, ImageDraw, ImageFont
import json

# 设置字体
plt.rcParams['font.family'] = 'Arial'

# 获取到文件的绝对路径，并将其父目录添加到 sys.path
sys.path.append(os.path.abspath('../../luto'))

# 导入 settings.py（假设 settings 文件位于指定路径下）
import settings

# 导入本地 helper 模块的函数
from helper import *

def save_colors(unique_names):
    """保存颜色映射并返回列名到颜色的映射."""
    num_colors = len(unique_names)
    cmap = plt.get_cmap('tab20', num_colors)  # 使用 'tab20' 颜色映射
    return {name: cmap(i / num_colors) for i, name in enumerate(unique_names)}

def plot_line_chart(df, df_colors, output_file='line_plot.png'):
    """绘制 df 的点线图，使用指定颜色."""
    fig, ax = plt.subplots(figsize=(12, 8))
    df.plot(kind='line', marker='o', linestyle='-', markersize=4, color=df_colors, ax=ax, legend=False)

    # 设置 X 和 Y 轴标签
    font_num = 22
    ax.set_xlabel('Year', fontsize=font_num)
    ylabel = 'Water limits (Billion L)' if output_file == 'water limits.png' else 'Water yield (Billion L)'
    ax.set_ylabel(ylabel, fontsize=font_num)

    # 设置 X 和 Y 轴刻度和范围
    ax.set_xticks(range(2010, 2051))
    ax.set_xticklabels(range(2010, 2051), rotation=90, fontsize=font_num)
    ax.set_xlim(2010, 2050)
    ax.set_yticks(range(0, 65, 16))
    ax.set_ylim(0, 64)
    ax.tick_params(axis='y', direction='in')
    ax.tick_params(axis='x', direction='in')

    # 保存图像
    fig.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close(fig)

def plot_legend(labels, df_colors, output_file='legend.png'):
    """创建并保存单独的图例，使用点线图样式."""
    fig_legend = plt.figure(figsize=(3, 2))  # 调整图例大小
    handles = [plt.Line2D([0], [0], color=color, lw=2, label=label, marker='o', linestyle='-', markersize=6)
               for label, color in zip(labels, df_colors)]
    fig_legend.legend(handles=handles, loc='center', frameon=False)
    fig_legend.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close(fig_legend)

def plot_shapefile_map(shapefile_path, gdf_column, df_colors, output_file='shapefile_plot.png'):
    """读取 Shapefile 并绘制地图，使用与点线图相同的颜色并标注区域."""
    font_size = 20
    gdf = gpd.read_file(shapefile_path)
    fig, ax_inset = plt.subplots(figsize=(12, 8))
    gdf.assign(color=gdf[gdf_column].map(df_colors)).plot(ax=ax_inset, color=gdf[gdf_column].map(df_colors), edgecolor='gray')

    # 隐藏轴刻度
    ax_inset.set_xticks([])
    ax_inset.set_yticks([])
    ax_inset.set_frame_on(False)

    # 标注区域名称
    pos_dict = {
        'Murray-Darling Basin': (0, 1), 'South Australian Gulf': (0, -1.5),
        'South West Coast': (1, -1), 'Lake Eyre Basin': (0.5, 0.5),
        'North Western Plateau': (0, -0.8), 'Timor Sea': (-1, 0.3),
        'Carpentaria Coast': (0, 0.5), 'North East Coast (QLD)': (0, -0.5),
        'Tanami': (0, -0.5), 'Pilbara-Gascoyne': (0, 0),
    }
    for idx, row in gdf.iterrows():
        centroid = row.geometry.centroid
        name = row[gdf_column]
        offset = pos_dict.get(name, (0, 0))
        ax_inset.text(centroid.x + offset[0], centroid.y + offset[1], name, fontsize=font_size, ha='center', color='black')

    plt.tight_layout()
    fig.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close(fig)

def add_label_to_image(image, label, position, font_size=60, color="black"):
    """在图像上添加标签."""
    font_path = "C:/Windows/Fonts/arial.ttf"
    font = ImageFont.truetype(font_path, font_size)
    draw = ImageDraw.Draw(image)
    draw.text(position, label, fill=color, font=font)
    return image

def concatenate_images_with_labels(image_files, labels, output_file='combined_image.png'):
    """拼接图像并为前四个图添加标签."""
    images = [Image.open(img) for img in image_files]
    img_width, img_height = images[0].size
    total_width = 2 * img_width
    total_height = 3 * img_height
    final_image = Image.new('RGBA', (total_width, total_height), (255, 255, 255, 0))

    label_positions = [(300, 100) for _ in range(4)]
    for i, img in enumerate(images[:4]):
        x_offset = (i % 2) * img_width
        y_offset = (i // 2) * img_height
        labeled_img = add_label_to_image(img, labels[i], label_positions[i], font_size=150)
        final_image.paste(labeled_img, (x_offset, y_offset))

    for i, img in enumerate(images[4:], start=4):
        if i == 4:
            x_offset = img_width - img.width
            y_offset = (i // 2) * img_height
        elif i == 5:
            aspect_ratio = img.width / img.height
            new_width = int(img_height * aspect_ratio)
            img = img.resize((new_width, img_height), Image.Resampling.LANCZOS)
            x_offset = 0
            y_offset = (i // 2) * img_height
        final_image.paste(img, (x_offset, y_offset))

    final_image.save(output_file, dpi=(300, 300))

# 主流程
plt.rcParams['font.family'] = 'Arial'
INPUT_DIR = "../../input"
dd_ccimpact_df = pd.read_hdf(os.path.join(INPUT_DIR, 'water_yield_2010_2100_cc_dd_ml.h5'))

# 列处理和索引修正
dd_ccimpact_columns = dd_ccimpact_df.columns
dd_ccimpact_df.columns = dd_ccimpact_df.columns.droplevel("Region_name")
dd_ccimpact_df = dd_ccimpact_df.loc[:, pd.IndexSlice[:, settings.SSP]]
dd_ccimpact_df.columns = dd_ccimpact_df.columns.droplevel('ssp')

# 创建列映射
processed_columns = dd_ccimpact_df.columns
original_filtered_columns = dd_ccimpact_columns.droplevel(['Region_name', 'ssp'])
column_mapping = {col: dd_ccimpact_columns[original_filtered_columns == col].get_level_values(0)[0]
                  for col in processed_columns}

# 更新列名
dd_ccimpact_df.columns = [column_mapping[col] for col in dd_ccimpact_df.columns]

# 计算差值并合并数据
dd_ccimpact_delta_df = (dd_ccimpact_df.loc[2050, :] - dd_ccimpact_df) / 1e6
dd_ccimpact_delta_df = dd_ccimpact_delta_df.loc[range(2010, 2051), :]
dd_ccimpact_df = dd_ccimpact_df.loc[range(2010, 2051), :] + dd_ccimpact_delta_df

# 读取额外数据
dd = pd.read_hdf(os.path.join(INPUT_DIR, "draindiv_lut.h5"), index_col='HR_DRAINDIV_ID')
for name, value in zip(dd['HR_DRAINDIV_NAME'], dd['WATER_YIELD_HIST_BASELINE_ML']):
    dd_ccimpact_df[name] = value * settings.WATER_YIELD_TARGET_AG_SHARE / 1e6

# 处理结果数据
input_files = ['ON_MAXPROFIT_GHG_15C_67_R5', 'ON_MAXPROFIT_GHG_15C_50_R5', 'ON_MAXPROFIT_GHG_18C_67_R5']
result_dict = get_value(input_files, 'water_yiled_vs_hiotorical_baseline', 'TOT_WATER_NET_YIELD_ML', 'Value (ML)')
result_dict = {'water limits': dd_ccimpact_df, **result_dict}

# 颜色处理与绘图
unique_names = dd_ccimpact_df.columns
colors = save_colors(unique_names)
df_colors = [colors.get(col, 'gray') for col in dd_ccimpact_df.columns]

plot_legend(dd_ccimpact_df.columns, df_colors, output_file='water_legend.png')
shapefile_path = '../../../Map/Data/shp/Drainage Division/ADD_2016_AUST.shp'
plot_shapefile_map(shapefile_path, 'ADD_NAME16', colors, output_file='shapefile_plot.png')

for output_file, df in result_dict.items():
    plot_line_chart(df, df_colors, output_file=f'{output_file}.png')

# 拼接图像并保存
image_files = [f'{key}.png' for key in list(result_dict.keys())[:4]]
image_files += ['shapefile_plot.png', 'water_legend.png']
labels = ['(a)', '(b)', '(c)', '(d)']
concatenate_images_with_labels(image_files, labels, output_file='05_Water.png')
