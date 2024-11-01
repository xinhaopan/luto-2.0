import numpy as np
import sys
import os
import pandas as pd
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from tools.helper import *
from tools.parameters import *

plt.rcParams['font.family'] = 'Arial'

# 获取到文件的绝对路径，并将其父目录添加到 sys.path
sys.path.append(os.path.abspath('../../luto'))

# 导入 settings.py
import settings

csv_name, filter_column_name, value_column_name = 'GHG_emissions_separate_agricultural_landuse', 'Land-use', 'Value (t CO2e)'
ag_dict = get_value_sum(input_files, csv_name, filter_column_name, value_column_name)
merged_dict = process_land_use_tables(ag_dict)

categories = list(next(iter(merged_dict.values())).columns)
color_list = get_colors_for_names(categories, mapping_file='tools/lumap_grouped.csv')
output_png = '12_GHG agricultural land.png'
# 调用绘图函数，动态指定图的数量和布局，控制图例显示
plot_Combination_figures(merged_dict, output_png=output_png, input_names=input_files, plot_func=plot_stacked_bar,
                         categories=categories, color_list=color_list,
                         n_rows=3, n_cols=3, font_size=11,
                         x_range=(2010, 2050), y_range=(-1, 100),
                         x_ticks=10, y_ticks=25, x_label='Year', y_label='GHG Emission (Mt CO2e)',
                         legend_position=(0.5, -0.03), show_legend='last', label_positions=(0.005, 0.99))  # 标签位置