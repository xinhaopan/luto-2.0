import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd  # 用于处理Shapefile
from mpl_toolkits.axes_grid1.inset_locator import inset_axes  # 用于嵌入地图
from PIL import Image, ImageDraw, ImageFont  # 用于图像处理和绘制标签
import json

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
    colors = {name: cmap(i / num_colors) for i, name in enumerate(unique_names)}  # 将列名映射为颜色
    return colors


def plot_line_chart(df, df_colors, output_file='line_plot.png'):
    """绘制 df 的点线图，使用指定颜色."""
    fig, ax = plt.subplots(figsize=(12, 8))

    # 绘制 df 点线图，使用相同的颜色映射
    df.plot(kind='line', marker='o', linestyle='-', markersize=4, color=df_colors, ax=ax, legend=False)

    # 设置 X 和 Y 轴标签
    font_num = 22
    ax.set_xlabel('Year', fontsize=font_num)
    if output_file == 'water limits.png':
        ax.set_ylabel('Water limits (Billion L)', fontsize=font_num)
    ax.set_ylabel('Water yield (Billion L)', fontsize=font_num)

    # 设置 X 轴刻度和范围
    ax.set_xticks(range(2010, 2051))
    ax.set_xticklabels(range(2010, 2051), rotation=90, fontsize=font_num)
    ax.set_xlim(2010, 2050)

    # 设置 Y 轴刻度和范围
    ax.set_yticks(range(0, 65, 16))
    ax.set_yticklabels(ax.get_yticks(), fontsize=font_num)
    ax.set_ylim(0, 64)

    # 设置轴刻度方向
    ax.tick_params(axis='y', direction='in')
    ax.tick_params(axis='x', direction='in')

    # 保存图像
    fig.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_legend(labels, df_colors, output_file='legend.png'):
    """创建并保存单独的图例，使用点线图样式."""
    fig_legend = plt.figure(figsize=(3, 2))  # 调整图例大小

    # 创建点线图样式的图例句柄
    handles = [
        plt.Line2D(
            [0], [0], color=color, lw=2, label=label, marker='o', linestyle='-', markersize=6
        ) for label, color in zip(labels, df_colors)
    ]

    # 创建图例对象并单独保存
    fig_legend.legend(handles=handles, loc='center', frameon=False)

    # 保存图例为文件
    fig_legend.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close(fig_legend)


def plot_shapefile_map(shapefile_path, gdf_column, df_colors, output_file='shapefile_plot.png'):
    """读取 Shapefile 并绘制地图，使用与点线图相同的颜色并标注区域."""
    fontize = 20
    # 读取 Shapefile 文件
    gdf = gpd.read_file(shapefile_path)

    # 创建主图表
    fig, ax_inset = plt.subplots(figsize=(12, 8))

    # 绘制 Shapefile 图并分配颜色
    gdf.assign(color=gdf[gdf_column].map(df_colors)).plot(
        ax=ax_inset, color=gdf[gdf_column].map(df_colors), edgecolor='gray'
    )

    # 隐藏轴刻度
    ax_inset.set_xticks([])
    ax_inset.set_yticks([])
    ax_inset.set_frame_on(False)

    # 在地图上标注区域名称
    for idx, row in gdf.iterrows():
        centroid = row.geometry.centroid
        name = row[gdf_column]

        if name == 'Murray-Darling Basin':
            ax_inset.text(centroid.x, centroid.y + 1, name, fontsize=fontize, ha='center', color='black')  # 上移
        elif name == 'South East Coast (NSW)':
            ax_inset.text(centroid.x, centroid.y, name, fontsize=fontize, ha='center', color='black')
        elif name == 'South Australian Gulf':
            ax_inset.text(centroid.x, centroid.y - 1.5, name, fontsize=fontize, ha='center', color='black')  # 下移
        elif name == 'North East Coast (QLD)':
            ax_inset.text(centroid.x, centroid.y - 0.5, name, fontsize=fontize, ha='center', color='black')  # 轻微下移
        elif name == 'North Western Plateau':
            ax_inset.text(centroid.x, centroid.y - 0.8, name, fontsize=fontize, ha='center', color='black')  # 适度下移
        elif name == 'Timor Sea':
            ax_inset.text(centroid.x - 1, centroid.y + 0.3, name, fontsize=fontize, ha='center', color='black')  # 左移和上移
        elif name == 'Carpentaria Coast':
            ax_inset.text(centroid.x, centroid.y + 0.5, name, fontsize=fontize, ha='center', color='black')  # 上移
        elif name == 'South West Coast':
            ax_inset.text(centroid.x + 1, centroid.y - 1, name, fontsize=fontize, ha='center', color='black')  # 右移和下移
        elif name == 'Pilbara-Gascoyne':
            ax_inset.text(centroid.x, centroid.y, name, fontsize=fontize, ha='center', color='black')  # 上移
        elif name == 'Tanami':
            ax_inset.text(centroid.x, centroid.y - 0.5, name, fontsize=fontize, ha='center', color='black')  # 下移
        elif name == 'Lake Eyre Basin':
            ax_inset.text(centroid.x + 0.5, centroid.y + 0.5, name, fontsize=fontize, ha='center',
                          color='black')  # 右移和上移
        else:
            # 对于其他区域，保持默认位置
            ax_inset.text(centroid.x, centroid.y, name, fontsize=fontize, ha='center', color='black')

    # 调整布局，确保图例不重叠
    plt.tight_layout()

    # 保存绘制的地图
    fig.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close(fig)


def add_label_to_image(image, label, position, font_size=60, color="black"):
    """在图像上添加标签"""
    font_path = "C:/Windows/Fonts/arial.ttf"  # 设置默认字体路径
    font = ImageFont.truetype(font_path, font_size)
    # 使用 ImageDraw 在图像上绘制文本
    draw = ImageDraw.Draw(image)
    draw.text(position, label, fill=color, font=font)

    return image


def concatenate_images_with_labels(image_files, labels, output_file='combined_image.png'):
    """拼接图像并为前四个图添加标签"""

    # 加载所有图像
    images = [Image.open(img) for img in image_files]

    # 获取单个图像的宽度和高度（假设所有图像大小相同）
    img_width, img_height = images[0].size

    # 计算拼接后的总宽度和高度（两列三行）
    total_width = 2 * img_width  # 两列
    total_height = 3 * img_height  # 三行

    # 创建一个新的空白图像
    final_image = Image.new('RGBA', (total_width, total_height), (255, 255, 255, 0))

    # 添加标签到每个图像
    label_positions = [(300, 100) for _ in range(4)]

    # 拼接前四个图像并添加标签
    for i, img in enumerate(images[:4]):
        # 计算要粘贴图像的位置
        x_offset = (i % 2) * img_width  # 第一列或第二列
        y_offset = (i // 2) * img_height  # 第一行或第二行

        # 添加标签
        labeled_img = add_label_to_image(img, labels[i], label_positions[i], font_size=150)

        # 粘贴带标签的图像到最终图像
        final_image.paste(labeled_img, (x_offset, y_offset))

        for i, img in enumerate(images[4:], start=4):
            if i == 4:
                # 倒数第二个图（第5个图）与第三个图右对齐
                third_img_x_offset = img_width  # 获取第三个图的 x 偏移（右边）
                # second_last_img_width = img_width
                second_last_img_height = img_height
                x_offset = third_img_x_offset - img.width  # 右对齐
                y_offset = (i // 2) * img_height
            elif i == 5:
                # 最后一个图高度与倒数第二个图一致，保持等比放大
                aspect_ratio = img.width / img.height
                new_width = int(second_last_img_height * aspect_ratio)

                # 放大最后一个图到和倒数第二个图的高度相同，保持等比放大
                img = img.resize((new_width, second_last_img_height), Image.Resampling.LANCZOS)

                # 最后一个图与第四个图左对齐
                x_offset = third_img_x_offset
                y_offset = (i // 2) * img_height

            final_image.paste(img, (x_offset, y_offset))
    final_image.save(output_file, dpi=(300, 300))


plt.rcParams['font.family'] = 'Arial'
INPUT_DIR = "../../input"
# 读取初始的 dd_ccimpact_df 数据
dd_ccimpact_df = pd.read_hdf(os.path.join(INPUT_DIR, 'water_yield_2010_2100_cc_dd_ml.h5'))

# 获取并处理原始的列和行数据，删除不需要的索引层
dd_ccimpact_columns = dd_ccimpact_df.columns
dd_ccimpact_df.columns = dd_ccimpact_df.columns.droplevel("Region_name")
dd_ccimpact_df = dd_ccimpact_df.loc[:, pd.IndexSlice[:, settings.SSP]]
dd_ccimpact_df.columns = dd_ccimpact_df.columns.droplevel('ssp')

# 获取处理后的列（不再是MultiIndex）
processed_columns = dd_ccimpact_df.columns

# 找到原始MultiIndex中去掉'Region_name'和'ssp'后的列
original_filtered_columns = dd_ccimpact_columns.droplevel(['Region_name', 'ssp'])

# 创建一个映射关系：处理后的列 -> 原始第一层index
column_mapping = {col: dd_ccimpact_columns[original_filtered_columns == col].get_level_values(0)[0]
                  for col in processed_columns}

# 使用 column_mapping 修改 dd_ccimpact_df 的列名
dd_ccimpact_df.columns = [column_mapping[col] for col in dd_ccimpact_df.columns]

# 将行索引从字符串转换为整数类型
dd_ccimpact_df.index = dd_ccimpact_df.index.astype(int)

# 计算 2050 年行与其他行的差值
dd_ccimpact_delta_df = (dd_ccimpact_df.loc[2050, :] - dd_ccimpact_df) / 1e6

# 保留 2010-2050 年的数据
dd_ccimpact_delta_df = dd_ccimpact_delta_df.loc[range(2010, 2051), :]

# 读取其他所需数据
dd = pd.read_hdf(os.path.join(INPUT_DIR, "draindiv_lut.h5"), index_col='HR_DRAINDIV_ID')

# 生成 2010 到 2050 的年份作为行索引
years = pd.Index(range(2010, 2051), name='Year')

# 创建一个以年份为索引的 DataFrame
dd_yield_df = pd.DataFrame(index=years)

# 构建最终的 dd_yield_df
for name, value in zip(dd['HR_DRAINDIV_NAME'], dd['WATER_YIELD_HIST_BASELINE_ML']):
    dd_yield_df[name] = value * settings.WATER_YIELD_TARGET_AG_SHARE / 1e6

dd_water_limit_df = dd_yield_df - dd_ccimpact_delta_df


input_files = ['ON_MAXPROFIT_GHG_15C_67_R5', 'ON_MAXPROFIT_GHG_15C_50_R5', 'ON_MAXPROFIT_GHG_18C_67_R5']
csv_name = 'water_yiled_vs_hiotorical_baseline'
row_name = 'TOT_WATER_NET_YIELD_ML'
column_name = 'Value (ML)'

result_dict = get_value(input_files, csv_name, row_name, column_name)
result_dict={'water limits':dd_water_limit_df, **result_dict}

unique_names = dd_water_limit_df.columns
colors = save_colors(unique_names)
df_colors = [colors.get(col, 'gray') for col in dd_water_limit_df.columns]
# 绘制单独图例
plot_legend(dd_water_limit_df.columns, df_colors, output_file='water_legend.png')
# 绘制 Shapefile 地图
shapefile_path = '../../../Map/Data/shp/Drainage Division/ADD_2016_AUST.shp'
plot_shapefile_map(shapefile_path, 'ADD_NAME16', colors, output_file='shapefile_plot.png')
# 绘制点线图
for output_file,df in result_dict.items():
    plot_line_chart(dd_water_limit_df, df_colors, output_file=f'{output_file}.png')

image_files = [f'{key}.png' for i, key in enumerate(list(result_dict.keys())[:4])]
image_files += [ 'shapefile_plot.png', 'water_legend.png']
labels = ['(a)', '(b)', '(c)', '(d)']

# 拼接图像并保存
concatenate_images_with_labels(image_files, labels, output_file='05_Water.png')

