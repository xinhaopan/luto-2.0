import os
import pandas as pd
import matplotlib.pyplot as plt
import sys
import numpy as np
from tools.helper import *
from PIL import Image, ImageDraw, ImageFont
import matplotlib.image as mpimg

# 获取到文件的绝对路径，并将其父目录添加到 sys.path
sys.path.append(os.path.abspath('../../luto'))

# 导入 settings.py
import settings

plt.rcParams['font.family'] = 'Arial'


# 为每个图像添加标签的函数
def add_label_to_image(image, label, position, font, font_size=60, color="black", font_path=None):
    font_path = "C:/Windows/Fonts/arial.ttf"
    font = ImageFont.truetype(font_path, font_size) if font_path else ImageFont.load_default()
    # 使用 ImageDraw 在图像上绘制文本
    draw = ImageDraw.Draw(image)
    draw.text(position, label, fill=color, font=font)
    return image


# 拼接图像函数
def concatenate_images_with_legend(image_files, legend_file, output_file, labels, font_path=None):
    # 打开所有的图像
    images = [Image.open(img) for img in image_files]

    # 设置字体大小（根据图像尺寸调整）
    font_size = 600
    font = ImageFont.truetype(font_path, font_size) if font_path else ImageFont.load_default()

    # 获取每个图像的宽度和高度
    img_width, img_height = images[0].size

    # 添加标签到每个图像
    label_positions = [(300, 100) for _ in range(4)]
    for i, img in enumerate(images):
        images[i] = add_label_to_image(img, labels[i], label_positions[i], font, font_size=150)

    # 调整图例宽度为与拼接后的图像总宽度一致，保持纵横比
    legend = Image.open(legend_file)
    legend_width = 2 * img_width  # 拼接后图像的总宽度
    legend_height = int(legend.size[1] * (legend_width / legend.size[0]))  # 根据宽度保持比例缩放图例
    legend = legend.resize((legend_width, legend_height), Image.Resampling.LANCZOS)

    # 设置拼接后的图像大小 (2x2) 并将图例放在底部
    total_width = legend_width  # 2列图的总宽度
    total_height = 2 * img_height + legend_height  # 两行图片高度 + 图例高度

    # 创建透明的背景图
    final_image = Image.new('RGBA', (total_width, total_height), (255, 255, 255, 0))  # 透明背景

    # 将四个图像拼接到 2x2 网格中
    final_image.paste(images[0], (0, 0))  # (a)
    final_image.paste(images[1], (img_width, 0))  # (b)
    final_image.paste(images[2], (0, img_height))  # (c)
    final_image.paste(images[3], (img_width, img_height))  # (d)

    # 拼接图例到底部
    final_image.paste(legend, (0, 2 * img_height))  # 将放大后的图例拼接到底部

    # 保存最终拼接后的图像，背景透明
    final_image.save(output_file, 'PNG', dpi=(300, 300))

# 数据处理函数
def process_demand_data(input_dir):
    dd = pd.read_hdf(os.path.join(input_dir, 'demand_projections.h5'))

    demand_data = dd.loc[(settings.SCENARIO,
                          settings.DIET_DOM,
                          settings.DIET_GLOB,
                          settings.CONVERGENCE,
                          settings.IMPORT_TREND,
                          settings.WASTE,
                          settings.FEED_EFFICIENCY),
    'PRODUCTION'].copy()

    # 处理 'eggs' 数据
    demand_data.loc['eggs'] = demand_data.loc['eggs'] * settings.EGGS_AVG_WEIGHT / 1000 / 1000
    demand_data = demand_data.T / 1e6
    demand_data = demand_data.loc[2010:2050]

    return demand_data


# 绘图函数
def plot_demand_data(demand_data, colors, output_file, fontsize=22):
    # 创建新的图形
    fig, ax = plt.subplots(figsize=(10, 7))

    # 绘制堆叠柱状图，不显示图例
    demand_data.plot(kind='bar', stacked=True, ax=ax, color=colors, legend=False)

    # 设置 Y 轴范围和刻度
    ax.set_ylim(0, 200)
    ax.set_yticks(np.arange(0, 201, 50))  # 刻度从 0 到 200，间隔为 50

    # 添加标题和标签
    ax.set_xlabel('Year', fontsize=fontsize)  # x轴标签字体大小
    ax.yaxis.set_label_coords(-0.06, -1)  # 调整 y 轴标签位置
    ax.set_ylabel('Food Demand (million tonnes/kilolitres [milk])', fontsize=fontsize)  # y轴标签字体大小

    # 设置刻度朝内
    ax.tick_params(axis='y', direction='in', labelsize=fontsize)  # y轴刻度字体大小
    ax.tick_params(axis='x', direction='in', labelsize=fontsize)  # x轴刻度字体大小
    # plt.xticks(rotation=45, ha='right', fontsize=fontsize)  # 旋转45度，水平对齐为右对齐

    # 调整布局
    plt.tight_layout()

    # 保存图表（不含图例）
    plt.savefig(output_file, dpi=300)
    plt.show()

    # 关闭当前图形
    plt.close()



# 生成颜色函数
def generate_colors(num_columns):
    # 组合多个 colormap: 'tab20', 'tab20b', 'tab20c'
    colormaps = [plt.get_cmap('tab20'), plt.get_cmap('tab20b'), plt.get_cmap('tab20c')]
    colors = []

    # 循环生成所需数量的颜色
    for cmap in colormaps:
        for i in range(cmap.N):
            colors.append(cmap(i / cmap.N))

    # 使用前 num_columns 种颜色
    return colors[:num_columns]


# 绘制图例并保存为单独图片
def save_legend(ax, output_legend_file='04_Food_legend.png'):
    # 创建一个新的图形来存放图例，调整图形尺寸为小正方形
    fig_legend = plt.figure(figsize=(10, 10))  # 调整尺寸为小正方形

    # 获取图例句柄和标签
    handles, labels = ax.get_legend_handles_labels()

    # 设置图例的列数为3列，自动计算需要的行数
    num_plots = len(labels)
    nrows = 6
    ncols = (num_plots + nrows - 1) // nrows  # 动态计算行数

    # 创建图例，并调整图例形状
    legend = fig_legend.legend(handles, labels, loc='center', frameon=False,
                               handlelength=1,  # 设置图例项的宽度，较短接近正方形
                               handleheight=1,  # 设置图例项的高度，确保接近正方形
                               ncol=ncols)  # 设置图例为 3 列布局

    # 设置图例文本大小，不改变字体大小
    plt.setp(legend.get_texts(), fontsize=12)  # 根据需要调整字体大小

    # 保存图例为图片，确保边界不会截掉
    fig_legend.savefig(output_legend_file, dpi=300, bbox_inches='tight')

    # 关闭图例窗口，释放资源
    plt.close(fig_legend)



# 主程序
if __name__ == "__main__":
    INPUT_DIR = "../../input"
    demand_data = process_demand_data(INPUT_DIR)
    demand_data = demand_data.drop(columns=['aquaculture', 'chicken', 'eggs', 'pork'])

    # input_files = {
    #     ('ON_MAXPROFIT_GHG_15C_67_R5', 'quantity_comparison', 'Prod_targ_year (tonnes, KL)'),
    #     ('ON_MAXPROFIT_GHG_15C_50_R5', 'quantity_comparison', 'Prod_targ_year (tonnes, KL)'),
    #     ('ON_MAXPROFIT_GHG_18C_67_R5', 'quantity_comparison', 'Prod_targ_year (tonnes, KL)')
    # }
    input_files = {
        ('0_BIO_0_GHG_1_5C_67_1', 'quantity_comparison', 'Prod_targ_year (tonnes, KL)'),
        ('0_BIO_0_GHG_1_5C_67_1', 'quantity_comparison', 'Prod_targ_year (tonnes, KL)'),
        ('0_BIO_0_GHG_1_5C_67_1', 'quantity_comparison', 'Prod_targ_year (tonnes, KL)')
    }
    # 图像文件列表
    image_files = ['Food Demand.png', 'ON_MAXPROFIT_GHG_15C_67_R10.png', 'ON_MAXPROFIT_GHG_15C_50_R10.png',
                   'ON_MAXPROFIT_GHG_18C_67_R10.png']

    production_data = process_input_files(input_files)
    expanded_tables = expand_tables(production_data)
    expanded_tables = {'04_Food_Target': demand_data, **expanded_tables}

    num_columns = demand_data.shape[1]
    colors = generate_colors(num_columns)
    for key, value in expanded_tables.items():
        # 创建图表
        fig, ax = plt.subplots()
        value.plot(kind='bar', stacked=True, ax=ax, figsize=(10, 7), color=colors)

        # 保存图表（不带图例）
        plot_demand_data(demand_data, colors, f'04_Food_{key}.png')
        if key == list(expanded_tables.keys())[0]:
            # 保存图例
            save_legend(ax)
        save_legend(ax)
    # 输出文件名
    output_file = '04_Food.png'

    # 图例文件
    legend_file = '04_Food_legend.png'


    # 图像标签 (a), (b), (c), (d)
    labels = ['(a)', '(b)', '(c)', '(d)']

    # 调用拼接函数，生成最终带有图例和标签的透明图像
    # concatenate_images_with_legend(image_files, legend_file, output_file, labels)
