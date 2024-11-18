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

font_size = 35
csv_name, value_column_name, filter_column_name = 'area_agricultural_landuse', 'Area (ha)', 'Land-use'
area_dict = get_dict_data(input_files, csv_name, value_column_name, filter_column_name)
ag_dict,legend_colors = get_colors(area_dict, 'tools/land use colors.xlsx', sheet_name='ag')
output_png = '../output/07_S1_area_ag.png'
plot_Combination_figures(ag_dict, output_png, input_files, plot_stacked_bar, legend_colors,
                            n_rows=3, n_cols=3, font_size=font_size, x_range=(2010, 2050), y_range=(0, 500),
                             x_ticks=10, y_ticks=125,
                             legend_position=(0.5, -0.25), show_legend='last', legend_n_rows=7)

csv_name, value_column_name, filter_column_name = 'GHG_emissions_separate_agricultural_landuse','Value (t CO2e)',  'Land-use'
area_dict = get_dict_data(input_files, csv_name, value_column_name, filter_column_name)
ag_dict,legend_colors = get_colors(area_dict, 'tools/land use colors.xlsx', sheet_name='ag')
output_png = '../output/07_S1_GHG_ag.png'
plot_Combination_figures(ag_dict, output_png, input_files, plot_stacked_bar, legend_colors,
                             point_data=None, n_rows=3, n_cols=3, font_size=font_size, x_range=(2010, 2050), y_range=(-1, 100),
                             x_ticks=10, y_ticks=25,
                             legend_position=(0.5, -0.25), show_legend='last', legend_n_rows=2)

csv_name, value_column_name, filter_column_name = 'biodiversity_separate', 'Biodiversity score', 'Landuse'
area_dict = get_dict_data(input_files, csv_name, value_column_name, filter_column_name)
ag_dict,legend_colors = get_colors(area_dict, 'tools/land use colors.xlsx', sheet_name='ag')
output_png = '../output/07_S1_BIO_ag.png'
plot_Combination_figures(ag_dict, output_png, input_files, plot_stacked_bar, legend_colors,
                            n_rows=3, n_cols=3, font_size=font_size, x_range=(2010, 2050), y_range=(0, 105),
                             x_ticks=10, y_ticks=35,
                             legend_position=(0.5, -0.25), show_legend='last', legend_n_rows=2)

csv_name, value_column_name, filter_column_name = 'water_yield_separate', 'Water Net Yield (ML)', 'Landuse'
area_dict = get_dict_data(input_files, csv_name, value_column_name, filter_column_name)
ag_dict,legend_colors = get_colors(area_dict, 'tools/land use colors.xlsx', sheet_name='ag')
output_png = '../output/07_S1_Water_ag.png'
plot_Combination_figures(ag_dict, output_png, input_files, plot_stacked_bar, legend_colors,
                            n_rows=3, n_cols=3, font_size=font_size, x_range=(2010, 2050), y_range=(0, 120),
                             x_ticks=10, y_ticks=40,
                             legend_position=(0.5, -0.25), show_legend='last', legend_n_rows=2)
