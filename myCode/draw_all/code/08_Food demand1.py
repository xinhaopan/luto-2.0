import numpy as np
import sys
import os
import pandas as pd
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

from tools.data_helper import *
from tools.plot_helper import *
from tools.parameters import *

plt.rcParams['font.family'] = 'Arial'

# 获取到文件的绝对路径，并将其父目录添加到 sys.path
sys.path.append(os.path.abspath('../../../luto'))

# 导入 settings.py
import settings

csv_name, value_column_name, filter_column_name = "quantity_production_kt_separate", 'Production (tonnes, KL)', 'Landuse subtype'
food_am_dict = get_dict_data(input_files, csv_name, value_column_name, filter_column_name,use_parallel=False)
food_am_dict,legend_colors = get_colors(food_am_dict, 'tools/land use colors.xlsx', sheet_name='am')
output_png = '../output/09_food_am_group'
y_range, y_ticks = calculate_y_axis_range(food_am_dict,desired_ticks=5,n_jobs=0)
plot_Combination_figures(food_am_dict, output_png, input_files, plot_stacked_bar, legend_colors,
                            n_rows=3, n_cols=3, font_size=font_size, x_range=(2010, 2050), y_range=y_range,
                             x_ticks=20, y_ticks=y_ticks,
                             legend_position=(0.5, -0.25), show_legend='last', legend_n_rows=2)

csv_name, value_column_name, filter_column_name = "quantity_production_kt_separate", 'Production (tonnes, KL)', 'Landuse'
food_non_ag_dict = get_dict_data(input_files, csv_name, value_column_name, filter_column_name,use_parallel=False)
food_non_ag_dict,legend_colors = get_colors(food_non_ag_dict, 'tools/land use colors.xlsx', sheet_name='non_ag')
output_png = '../output/09_food_non_ag_group'
y_range, y_ticks = calculate_y_axis_range(food_non_ag_dict,n_jobs=0)
plot_Combination_figures(food_non_ag_dict, output_png, input_files, plot_stacked_bar, legend_colors,
                            n_rows=3, n_cols=3, font_size=font_size, x_range=(2010, 2050), y_range=y_range,
                             x_ticks=20, y_ticks=y_ticks,
                             legend_position=(0.5, -0.25), show_legend='last', legend_n_rows=2)
