from tools.helper import *
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from PIL import Image, ImageDraw, ImageFont
plt.rcParams['font.family'] = 'Arial'

def concatenate_images_with_labels(image_files, output_file, labels, font_size=60, label_position=(50, 50)):
    """
    将多张图片上下拼接，并在每张图片左上角添加标签（如 "(a)", "(b)"）。

    参数：
    image_files: 图片文件路径列表
    output_file: 保存拼接后的图片路径
    labels: 每张图片的左上角标签
    font_size: 标签的字体大小
    label_position: 标签在每张图片中的位置，默认在左上角
    """
    # 打开所有图像
    images = [Image.open(img) for img in image_files]

    # 计算总高度和最大宽度
    total_height = sum(img.height for img in images)
    max_width = max(img.width for img in images)

    # 创建新图像用于拼接
    combined_image = Image.new('RGB', (max_width, total_height))

    # 定义字体（使用系统默认字体）
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except IOError:
        # 如果字体加载失败，使用默认字体
        font = ImageFont.load_default()

    # 逐张粘贴图像并添加标签
    current_height = 0
    for i, img in enumerate(images):
        combined_image.paste(img, (0, current_height))

        # 添加标签
        draw = ImageDraw.Draw(combined_image)
        draw.text((label_position[0], current_height + label_position[1]), labels[i], font=font, fill="black")

        current_height += img.height

    # 保存拼接后的图片
    combined_image.save(output_file)
    print(f"Image saved as {output_file}")

# 添加颜色图例（表示不同列）
def add_color_legend(name, ax, df, colors, font_size=14):
    handles_colors = []
    for j, column in enumerate(df.columns):
        patch = mpatches.Patch(color=colors[j], label=column)
        handles_colors.append(patch)

    color_legend = ax.legend(handles=handles_colors, loc='upper left', bbox_to_anchor=(0.05, 1),
                             frameon=False, fontsize=font_size)
    # ax.add_artist(color_legend)  # 保持颜色图例并添加到图中


    # 根据图例项目数量动态设置图例高度
    num_items = len(df.columns)
    legend_height = num_items * 0.5  # 动态调整高度

    # 创建一个新的空白图形，用于单独保存图例
    fig_legend = plt.figure(figsize=(6, legend_height))
    fig_legend.legend(handles=handles_colors, loc='center', frameon=False, fontsize=font_size)

    # 保存单独的图例
    fig_legend.savefig(f'area_{name}_legend.png', bbox_inches='tight')

    # 关闭新建的图例 Figure，释放内存
    plt.close(fig_legend)


# 添加填充模式图例（表示不同表）
def add_hatch_legend(ax, labels, hatch_patterns, font_size=14):
    handles_hatches = []
    labels = ['1.5°C(67%)', '1.5°C(50%)', '1.8°C(67%)']  # 表示表的标签
    for i in range(len(labels)):
        patch = mpatches.Patch(facecolor='white', edgecolor='black', hatch=hatch_patterns[i], label=labels[i])
        handles_hatches.append(patch)

    # ax.legend(handles=handles_hatches, loc='upper left', bbox_to_anchor=(0.45, 1),  # 调整位置
    #           frameon=False, fontsize=font_size)

    # 创建一个新的空白图形，用于单独保存图例
    fig_legend = plt.figure(figsize=(6, 1.5))
    fig_legend.legend(handles=handles_hatches, loc='center', frameon=False, fontsize=font_size)

    # 保存单独的图例
    fig_legend.savefig('area_legend.png', bbox_inches='tight')

    # 关闭新建的图例 Figure，释放内存
    plt.close(fig_legend)


def plot_multiple_tables(tables_dict, font_size=14, colors=None, hatch_patterns=None, yticks_range=None,
                         output_file='stacked_bar_chart.png'):
    num_columns = len(list(tables_dict.values())[0].columns)
    if colors is None:
        cmap = plt.get_cmap('Set3', num_columns)
        colors = [cmap(i / num_columns) for i in range(num_columns)]

    if hatch_patterns is None:
        hatch_patterns = ['/', '\\', '-', 'o', '.', '*', '|', '+', 'x', 'O', ]

    index = list(tables_dict.values())[0].index
    fig, ax = plt.subplots(figsize=(24, 12))
    num_tables = len(tables_dict)
    width = 0.2  # 柱子的宽度
    positions = np.arange(len(index))  # 位置索引

    for i, (table_name, df) in enumerate(tables_dict.items()):
        bottom = np.zeros(len(index))
        df = df.astype(float)
        for j, column in enumerate(df.columns):
            ax.bar(df.index + i * (width * 1.1) - width, df[column], width=width, bottom=bottom, edgecolor='white',
                   color=colors[j], alpha=0.8, hatch=hatch_patterns[i])
            bottom += df[column].values

    # 设置轴标签、刻度等
    ax.set_xlabel('Year', fontsize=font_size)
    ax.set_ylabel('Area (Million ha)', fontsize=font_size)
    ax.set_xticks(range(2010, 2051))
    ax.set_xlim(2009.5, 2050.5)
    ax.set_xticklabels(range(2010, 2051), rotation=90, fontsize=font_size, ha='center')

    # 根据传入的 yticks_range 设置 Y 轴
    if yticks_range:
        start, end, step = yticks_range
        ax.set_yticks(range(start, end + 1, step))
        ax.set_ylim(start, end)
        ax.set_yticklabels(range(start, end + 1, step), fontsize=font_size)
    else:
        # 如果没有传入 yticks_range，则使用默认的
        ax.set_yticks(range(0, 201, 50))
        ax.set_ylim(0, 200)
        ax.set_yticklabels(range(0, 201, 50), fontsize=font_size)

    ax.tick_params(axis='both', direction='in')

    # 添加颜色图例和填充图例
    add_color_legend(output_file[:-4], ax, df, colors, font_size=font_size)
    add_hatch_legend(ax, list(tables_dict.keys()), hatch_patterns, font_size=font_size)
    ax.get_legend().remove()
    plt.tight_layout()
    plt.savefig(output_file, bbox_inches='tight')
    plt.show()

input_files = ['ON_MAXPROFIT_GHG_15C_67_R10', 'ON_MAXPROFIT_GHG_15C_50_R10', 'ON_MAXPROFIT_GHG_18C_67_R10']
font_size = 50
# ag
csv_name, filter_column_name, value_column_name =  'area_agricultural_landuse', 'Land-use', 'Area (ha)'
tables_ag_dict = get_value_sum(input_files, csv_name, filter_column_name, value_column_name)
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
plot_multiple_tables(tables_ag_dict, font_size=font_size,colors=colors, yticks_range=[0,500,100], output_file='ag.png')

# am
csv_name, filter_column_name, value_column_name =  'area_agricultural_management', 'Type', 'Area (ha)'
tables_am_dict = get_value_sum(input_files, csv_name, filter_column_name, value_column_name)
# colors = ['#D4895F', '#E1A25D', '#C481A1', '#A3C369', '#7FB8C6']
colors = ['#00A9E6', '#E69800', '#00A884', '#D69DBC', '#343434']
plot_multiple_tables(tables_am_dict, font_size=font_size,colors=colors,yticks_range=[0,120,40], output_file='am.png')

# non-ag
csv_name, filter_column_name, value_column_name =  'area_non_agricultural_landuse', 'Land-use', 'Area (ha)'
tables_no_ag_dict = get_value_sum(input_files, csv_name, filter_column_name, value_column_name)
# colors = ['#264653', '#27736F', '#299D92', '#89AF7D', '#E8C56B', '#F2A361', '#E56E51', '#FFB715']
colors = ['#AACC66', '#005CE6', '#C500FF', '#FF0000', '#267300', '#F2A361', '#E56E51', '#FFB715']
plot_multiple_tables(tables_no_ag_dict, font_size=font_size, colors=colors,yticks_range=[0,50,10], output_file='no_ag.png')

# 调用函数，拼接两张图片并添加 (a), (b) 标签
image_files = ['ag.png','am.png', 'no_ag.png']
labels = ['(a)', '(b)','(c)']  # 标签
concatenate_images_with_labels(image_files, output_file='08_ag_am_non_ag.png', labels=labels,font_size=font_size*1.8, label_position=(200, 30))