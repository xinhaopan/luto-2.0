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

font_size = 15
csv_name, value_column_name, filter_column_name = 'water_yield_separate', 'Water Net Yield (ML)', 'region'
area_dict = get_dict_data(input_files, csv_name, value_column_name, filter_column_name)
area_ag_group_dict,legend_colors = get_colors(area_dict, 'tools/land use colors.xlsx', sheet_name='water')
output_png = '../output/09_S3_water.png'
plot_Combination_figures(area_ag_group_dict, output_png, input_files, plot_stacked_bar, legend_colors,
                            n_rows=3, n_cols=3, font_size=font_size, x_range=(2010, 2050), y_range=(0, 300),
                             x_ticks=10, y_ticks=100,
                             legend_position=(0.5, -0.25), show_legend='last', legend_n_rows=4)
