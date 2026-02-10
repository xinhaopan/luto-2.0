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

csv_name, value_column_name, filter_column_name = 'area_agricultural_landuse', 'Area (ha)', 'Land-use'
area_dict = get_dict_data(input_files, csv_name, value_column_name, filter_column_name,['Water_supply'],['Dryland'])
area_group_dict = aggregate_by_mapping(area_dict, 'tools/land use group.xlsx', 'desc', 'ag_group')
area_ag_group_dict,legend_colors = get_colors(area_group_dict, 'tools/land use colors.xlsx', sheet_name='ag_group')
y_range, y_ticks = calculate_y_axis_range(area_ag_group_dict,5)
output_png = '../output/09_area_dry_ag_group'
plot_Combination_figures(area_ag_group_dict, output_png, input_files, plot_stacked_area, legend_colors,
                            n_rows=3, n_cols=3, font_size=font_size, x_range=(2010, 2050), y_range=y_range,
                             x_ticks=20, y_ticks=y_ticks,
                             legend_position=(0.5, -0.25), show_legend='last', legend_n_rows=2)

csv_name, value_column_name, filter_column_name = 'area_agricultural_landuse', 'Area (ha)', 'Land-use'
area_dict = get_dict_data(input_files, csv_name, value_column_name, filter_column_name,['Water_supply'],['Irrigated'])
area_group_dict = aggregate_by_mapping(area_dict, 'tools/land use group.xlsx', 'desc', 'ag_group')
area_ag_group_dict,legend_colors = get_colors(area_group_dict, 'tools/land use colors.xlsx', sheet_name='ag_group')
y_range, y_ticks = calculate_y_axis_range(area_ag_group_dict,3)
output_png = '../output/09_area_irr_ag_group'
plot_Combination_figures(area_ag_group_dict, output_png, input_files, plot_stacked_area, legend_colors,
                            n_rows=3, n_cols=3, font_size=font_size, x_range=(2010, 2050), y_range=y_range,
                             x_ticks=20, y_ticks=y_ticks,
                             legend_position=(0.5, -0.25), show_legend='last', legend_n_rows=2)


csv_name, value_column_name, filter_column_name = 'water_yield_separate_watershed', 'Water Net Yield (ML)', 'Landuse'
area_dict = get_dict_data(input_files, csv_name, value_column_name, filter_column_name,['Water Supply','Type'],['Dryland','Agricultural Landuse'])
area_group_dict = aggregate_by_mapping(area_dict, 'tools/land use group.xlsx', 'desc', 'ag_group')
area_ag_group_dict,legend_colors = get_colors(area_group_dict, 'tools/land use colors.xlsx', sheet_name='ag_group')
y_range, y_ticks = calculate_y_axis_range(area_ag_group_dict,4)
output_png = '../output/09_water_dry_ag_group'
plot_Combination_figures(area_ag_group_dict, output_png, input_files, plot_stacked_area, legend_colors,
                            n_rows=3, n_cols=3, font_size=font_size, x_range=(2010, 2050), y_range=y_range,
                             x_ticks=20, y_ticks=y_ticks,
                             legend_position=(0.5, -0.25), show_legend='last', legend_n_rows=2)

csv_name, value_column_name, filter_column_name = 'water_yield_separate_watershed', 'Water Net Yield (ML)', 'Landuse'
area_dict = get_dict_data(input_files, csv_name, value_column_name, filter_column_name,['Water Supply','Type'],['Irrigated','Agricultural Landuse'])
area_group_dict = aggregate_by_mapping(area_dict, 'tools/land use group.xlsx', 'desc', 'ag_group')
area_ag_group_dict,legend_colors = get_colors(area_group_dict, 'tools/land use colors.xlsx', sheet_name='ag_group')
y_range, y_ticks = calculate_y_axis_range(area_ag_group_dict,3)
output_png = '../output/09_water_irr_ag_group'
plot_Combination_figures(area_ag_group_dict, output_png, input_files, plot_stacked_area, legend_colors,
                            n_rows=3, n_cols=3, font_size=font_size, x_range=(2010, 2050), y_range=y_range,
                             x_ticks=20, y_ticks=y_ticks,
                             legend_position=(0.5, -0.25), show_legend='last', legend_n_rows=2)