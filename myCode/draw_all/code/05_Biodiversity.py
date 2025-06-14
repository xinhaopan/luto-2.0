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

csv_name, value_column_name, filter_column_name = 'biodiversity_GBF2_priority_scores', 'Contribution Relative to Pre-1750 Level (%)', 'Type'
bio_dict = get_dict_data(input_files, csv_name, value_column_name, filter_column_name,unit_adopt=False)
bio_dict,legend_colors = get_colors(bio_dict, 'tools/land use colors.xlsx', sheet_name='lu')

bio_target_dict = get_dict_data(input_files, "biodiversity_GBF2_priority_scores", 'Priority Target (%)', 'Landuse',['Landuse','Type'],['Apples','Agricultural Landuse'],unit_adopt=False)
# point_dict = rename_and_filter_columns(bio_target_dict, ['Biodiversity score limit','Solve biodiversity score'], ['Constraints','Score'],-10)
point_dict = rename_and_filter_columns(bio_target_dict, ['Apples'], ['Targets'])

point_colors = ['red']
# y_range, y_ticks = calculate_y_axis_range(bio_dict)
y_range, y_ticks = calculate_y_axis_range(bio_dict,3)
output_png = '../output/05_bio.png'
plot_Combination_figures(bio_dict, output_png, input_files, plot_stacked_area_and_line, legend_colors,point_dict=point_dict,point_colors=point_colors,
                            n_rows=3, n_cols=3, font_size=font_size, x_range=(2010, 2050), y_range=y_range,
                             x_ticks=20, y_ticks=y_ticks,
                             legend_position=(0.5, -0.25), show_legend='last', legend_n_rows=2)


csv_name, value_column_name, filter_column_name = 'biodiversity_GBF2_priority_scores', 'Contribution Relative to Pre-1750 Level (%)', 'Landuse'
bio_ag_dict = get_dict_data(input_files, csv_name, value_column_name, filter_column_name,['Type'],['Agricultural Landuse'],unit_adopt=False)
bio_ag_group_dict = aggregate_by_mapping(bio_ag_dict, 'tools/land use group.xlsx', 'desc', 'ag_group')
bio_ag_group_dict,legend_colors = get_colors(bio_ag_group_dict, 'tools/land use colors.xlsx', sheet_name='ag_group')
y_range, y_ticks = calculate_y_axis_range(bio_ag_group_dict,4)
output_png = '../output/05_bio_ag_group.png'
plot_Combination_figures(bio_ag_group_dict, output_png, input_files, plot_stacked_area, legend_colors,
                            n_rows=3, n_cols=3, font_size=font_size, x_range=(2010, 2050), y_range=y_range,
                             x_ticks=20, y_ticks=y_ticks,
                             legend_position=(0.5, -0.25), show_legend='last', legend_n_rows=2)

csv_name, value_column_name, filter_column_name = 'biodiversity_GBF2_priority_scores', 'Contribution Relative to Pre-1750 Level (%)', 'Agri-Management'
bio_am_dict = get_dict_data(input_files, csv_name, value_column_name, filter_column_name, ['Type'],['Agricultural Management'],unit_adopt=False)
bio_am_dict,legend_colors = get_colors(bio_am_dict, 'tools/land use colors.xlsx', sheet_name='am')
output_png = '../output/05_bio_am.png'
y_range, y_ticks = (0,2),[0,1,2]
plot_Combination_figures(bio_am_dict, output_png, input_files, plot_stacked_area, legend_colors,
                            n_rows=3, n_cols=3, font_size=font_size, x_range=(2010, 2050), y_range=y_range,
                             x_ticks=20, y_ticks=y_ticks,
                             legend_position=(0.5, -0.25), show_legend='last', legend_n_rows=2)

csv_name, value_column_name, filter_column_name = 'biodiversity_GBF2_priority_scores', 'Contribution Relative to Pre-1750 Level (%)', 'Landuse'
bio_non_ag_dict = get_dict_data(input_files, csv_name, value_column_name, filter_column_name, ['Type'],['Non-Agricultural land-use'],unit_adopt=False)
bio_non_ag_dict,legend_colors = get_colors(bio_non_ag_dict, 'tools/land use colors.xlsx', sheet_name='non_ag')
output_png = '../output/05_bio_non-ag.png'
y_range, y_ticks = calculate_y_axis_range(bio_non_ag_dict,3)
plot_Combination_figures(bio_non_ag_dict, output_png, input_files, plot_stacked_area, legend_colors,
                            n_rows=3, n_cols=3, font_size=font_size, x_range=(2010, 2050), y_range=y_range,
                             x_ticks=20, y_ticks=y_ticks,
                             legend_position=(0.5, -0.25), show_legend='last', legend_n_rows=2)