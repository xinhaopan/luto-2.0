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

csv_name, value_column_name, filter_column_name = 'biodiversity_separate', 'Biodiversity score', 'Landuse type'
bio_dict = get_dict_data(input_files, csv_name, value_column_name, filter_column_name)
bio_dict,legend_colors = get_colors(bio_dict, 'tools/land use colors.xlsx', sheet_name='lu')
output_png = '../output/05_bio.png'
plot_Combination_figures(bio_dict, output_png, input_files, plot_stacked_bar, legend_colors,
                            n_rows=3, n_cols=3, font_size=font_size, x_range=(2010, 2050), y_range=(0, 105),
                             x_ticks=20, y_ticks=35,
                             legend_position=(0.5, -0.25), show_legend='last', legend_n_rows=2)

csv_name, value_column_name, filter_column_name = 'biodiversity_separate', 'Biodiversity score', 'Landuse'
bio_ag_dict = get_dict_data(input_files, csv_name, value_column_name, filter_column_name)
bio_ag_group_dict = aggregate_by_mapping(bio_ag_dict, 'tools/land use group.xlsx', 'desc', 'ag_group')
bio_dict,legend_colors = get_colors(bio_ag_group_dict, 'tools/land use colors.xlsx', sheet_name='ag_group')
output_png = '../output/05_bio_ag_group.png'
plot_Combination_figures(bio_dict, output_png, input_files, plot_stacked_bar, legend_colors,
                            n_rows=3, n_cols=3, font_size=font_size, x_range=(2010, 2050), y_range=(0, 105),
                             x_ticks=20, y_ticks=35,
                             legend_position=(0.5, -0.25), show_legend='last', legend_n_rows=2)

csv_name, value_column_name, filter_column_name = 'biodiversity_separate', 'Biodiversity score', 'Landuse subtype'
bio_am_dict = get_dict_data(input_files, csv_name, value_column_name, filter_column_name)
bio_dict,legend_colors = get_colors(bio_am_dict, 'tools/land use colors.xlsx', sheet_name='am')
output_png = '../output/05_bio_am.png'
plot_Combination_figures(bio_dict, output_png, input_files, plot_stacked_bar, legend_colors,
                            n_rows=3, n_cols=3, font_size=font_size, x_range=(2010, 2050), y_range=(0, 3),
                             x_ticks=20, y_ticks=1,
                             legend_position=(0.5, -0.25), show_legend='last', legend_n_rows=2)

csv_name, value_column_name, filter_column_name = 'biodiversity_separate', 'Biodiversity score', 'Landuse'
bio_am_dict = get_dict_data(input_files, csv_name, value_column_name, filter_column_name)
bio_dict,legend_colors = get_colors(bio_am_dict, 'tools/land use colors.xlsx', sheet_name='non_ag')
output_png = '../output/05_bio_non-ag.png'
plot_Combination_figures(bio_dict, output_png, input_files, plot_stacked_bar, legend_colors,
                            n_rows=3, n_cols=3, font_size=font_size, x_range=(2010, 2050), y_range=(0, 16),
                             x_ticks=20, y_ticks=4,
                             legend_position=(0.5, -0.25), show_legend='last', legend_n_rows=2)