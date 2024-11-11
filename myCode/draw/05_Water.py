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
from tools.helper import *

def plot_line_chart(df, df_colors, output_file='line_plot.png'):
    """绘制 df 的点线图，使用指定颜色，并统一图像布局."""
    fig, ax = plt.subplots(figsize=(12, 8))

    # 绘制 df 点线图，使用相同的颜色映射
    df.plot(kind='line', marker='o', linestyle='-', markersize=4, color=df_colors, ax=ax, legend=False)

    # 设置 X 和 Y 轴标签
    font_num = 22
    ax.set_xlabel('Year', fontsize=font_num)
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

    # 统一布局设置，确保图例、标签等占用空间一致
    plt.subplots_adjust(left=0.1, right=0.95, top=0.9, bottom=0.15)

    # 保存图像
    fig.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close(fig)


def plot_stacked_bar_and_line(merged_dict, input_name, font_size=10):
    merged_df = merged_dict[input_name]

    # 确保索引是 int 类型
    merged_df.index = merged_df.index.astype(int)

    # 设置图形大小
    fig, ax = plt.subplots(figsize=(12, 8))

    # 提取需要堆积的列
    categories = ['Agricultural landuse', 'Agricultural management', 'Non-agricultural landuse', 'Other landuse']

    # 准备数据
    years = merged_df.index
    data = np.array([merged_df[category] for category in categories])

    # 设置颜色列表，使用指定的五个颜色
    color_list = ['#F9C0B7', '#FCD071', '#B4A7D6', '#85C6BE', '#D2E0FB']

    # 分开处理正数和负数
    bars_pos = []
    bars_neg = []

    # 绘制正数的堆积柱状图
    pos_data = np.maximum(data, 0)
    bars_pos.append(ax.bar(years, pos_data[0], label=categories[0], color=color_list[0], width=0.8))
    for i in range(1, len(categories)):
        bars_pos.append(
            ax.bar(years, pos_data[i], bottom=pos_data[:i].sum(axis=0), label=categories[i], color=color_list[i],
                   width=0.8))

    # 绘制负数的堆积柱状图
    neg_data = np.minimum(data, 0)
    bars_neg.append(ax.bar(years, neg_data[0], label=categories[0], color=color_list[0], width=0.8))
    for i in range(1, len(categories)):
        bars_neg.append(
            ax.bar(years, neg_data[i], bottom=neg_data[:i].sum(axis=0), label=categories[i], color=color_list[i],
                   width=0.8))

    # 绘制 Net emissions 红色点线图
    line, = ax.plot(years, merged_df['Water limit'], color='#1E90FF', marker='o', linewidth=2, label='Water limit')

    # 设置刻度朝内
    ax.tick_params(axis='both', which='both', direction='in', labelsize=font_size)

    # 设置 x 轴刻度为年份，并确保标签旋转
    ax.set_xlim(2010 - 0.5, 2050 + 0.5)
    ax.set_xticks(years)
    ax.set_xticklabels(years, rotation=90, fontsize=font_size)

    # 设置 y 轴范围和刻度
    ax.set_ylim(0, 300)
    ax.set_yticks(range(0, 301, 100))
    ax.set_ylabel('Water yeild (Billion L)', fontsize=font_size)

    # 统一布局设置，确保图例、标签等占用空间一致
    plt.subplots_adjust(left=0.1, right=0.95, top=0.9, bottom=0.15)

    # 保存主图
    plt.tight_layout()
    plt.savefig(f"{input_name}_water.png", bbox_inches='tight')
    plt.show()

    # 创建单独的图例
    fig_legend = plt.figure(figsize=(8, 1))
    handles =  [line] + [bar[0] for bar in bars_pos]
    labels =['Water limit'] + categories
    legend = fig_legend.legend(handles, labels, loc='center', fontsize=font_size, frameon=False)

    # 保存图例
    fig_legend.savefig("water_legend.png", bbox_inches='tight')

def save_colors(unique_names):
    """保存颜色映射并返回列名到颜色的映射."""
    num_colors = len(unique_names)
    cmap = plt.get_cmap('tab20', num_colors)  # 使用 'tab20' 颜色映射
    colors = {name: cmap(i / num_colors) for i, name in enumerate(unique_names)}  # 将列名映射为颜色
    return colors


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
    fig_legend.legend(handles=handles, loc='center', frameon=False,ncol=2)

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

from PIL import Image, ImageDraw, ImageFont

def concatenate_images_with_labels(image_files, labels, output_file='combined_image.png'):
    """拼接图像并为前四个图添加标签，并调整前四个图的宽度保持一致"""

    # 加载所有图像
    images = [Image.open(img) for img in image_files]

    # 设置前四个图像的统一宽度
    uniform_width = min([img.width for img in images[:4]])  # 找出前四个图中最小的宽度

    # 调整前四个图像的宽度，同时保持纵横比
    images[:4] = [img.resize((uniform_width, int(img.height * (uniform_width / img.width))), Image.Resampling.LANCZOS)
                  for img in images[:4]]

    # 获取调整后的图像的宽度和高度
    img_width, img_height = images[0].size

    # 计算拼接后的总宽度和高度（两列两行 + 横向拼接的最后一行）
    total_width = 2 * img_width  # 两列
    total_height = 2 * img_height  # 前四个图两行

    # 拼接后三个图横向的总宽度
    horizontal_width = sum(img.width for img in images[4:])  # 后三个图的总宽度

    # 调整后三个图的总宽度等比例缩放到与前四个图的总宽度一致
    scale_factor = total_width / horizontal_width
    images[4:] = [img.resize((int(img.width * scale_factor), int(img.height * scale_factor)), Image.Resampling.LANCZOS)
                  for img in images[4:]]

    # 创建一个新的空白图像（包含三行，第三行为后三个图横向拼接后的结果）
    # final_image = Image.new('RGBA', (total_width, total_height + images[4].height), (255, 255, 255, 0))
    final_image = Image.new('RGBA', (total_width, total_height), (255, 255, 255, 0))

    # 添加标签到前四个图像
    label_positions = [(105, 20) for _ in range(4)]  # 调整标签位置

    # 拼接前四个图像并添加标签
    for i, img in enumerate(images[:4]):
        # 计算要粘贴图像的位置
        x_offset = (i % 2) * img_width  # 第一列或第二列
        y_offset = (i // 2) * img_height  # 第一行或第二行

        # 添加标签
        labeled_img = add_label_to_image(img, labels[i], label_positions[i], font_size=50)

        # 粘贴带标签的图像到最终图像
        final_image.paste(labeled_img, (x_offset, y_offset))

    # # 拼接后三个图（横向拼接在第三行）
    # current_x_offset = 0
    # for img in images[4:]:
    #     final_image.paste(img, (current_x_offset, 2 * img_height))  # 在第三行拼接
    #     current_x_offset += img.width  # 更新 x 偏移量

    final_image.show()
    # 保存最终拼接的图像
    final_image.save(output_file, dpi=(300, 300))




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
input_files = ['ON_MAXPROFIT_GHG_15C_67_R10', 'ON_MAXPROFIT_GHG_15C_50_R10', 'ON_MAXPROFIT_GHG_18C_67_R10']
# input_files = ['ON_MAXPROFIT_GHG_15C_67_R5', 'ON_MAXPROFIT_GHG_15C_50_R5', 'ON_MAXPROFIT_GHG_18C_67_R5']
csv_name, filter_column_name, value_column_name = 'water_yield_separate', 'Landuse Type', 'Water Net Yield (ML)'

table_dict = get_value_sum(input_files,csv_name, filter_column_name, value_column_name )
public_water_dict = get_value_sum(input_files, 'water_yield_of_climate_change_impacts_outside_LUTO', '*', 'Climate Change Impact (ML)')
# waterlimit_dict = get_value_sum(input_files, 'water_yield_of_climate_change_impacts_outside_LUTO', '*', 'Climate Change Impact (ML)')

# 拼接两个字典中 key 一样的表，并添加 Water limit 列
merged_dict = {}
for key in table_dict.keys():
    if key in public_water_dict:  # 确保两个字典中都有相同的 key
        # 拼接两个 DataFrame
        merged_df = pd.concat([table_dict[key], public_water_dict[key]], axis=1)

        # 修改列名为指定的名称
        merged_df.columns = ['Agricultural landuse', 'Non-agricultural landuse', 'Agricultural management', 'Other landuse']

        # 将 dd_water_limit_df 按行求和并添加为新的一列
        water_limit_sum = dd_water_limit_df.sum(axis=1)  # 按行求和
        merged_df['Water limit'] = water_limit_sum  # 添加到最后一列

        # 将拼接好的表存入新的字典
        merged_dict[key] = merged_df

# merged_dict 现在包含了拼接并重命名的表，并附加了 Water limit 列
for input_name in input_files:
    plot_stacked_bar_and_line(merged_dict, input_name, font_size=20)

unique_names = dd_water_limit_df.columns
colors = save_colors(unique_names)
df_colors = [colors.get(col, 'gray') for col in dd_water_limit_df.columns]
plot_line_chart(dd_water_limit_df, df_colors, output_file='water_limit.png')

# 绘制单独图例
plot_legend(dd_water_limit_df.columns, df_colors, output_file='water_limit_legend.png')
# 绘制 Shapefile 地图
shapefile_path = '../../../Map/Data/shp/Drainage Division/ADD_2016_AUST.shp'
plot_shapefile_map(shapefile_path, 'ADD_NAME16', colors, output_file='shapefile_plot.png')

image_files = ['water_limit.png'] + [f'{key}_water.png' for key in input_files]
image_files += [ 'shapefile_plot.png','water_limit_legend.png', 'water_legend.png']
labels = ['(a)', '(b)', '(c)', '(d)']

# 拼接图像并保存
concatenate_images_with_labels(image_files, labels, output_file='05_Water.png')

