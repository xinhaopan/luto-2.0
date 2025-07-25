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

csv_name, value_column_name, filter_column_name = "quantity_production_kt_separate", 'Production (tonnes, KL)', 'Landuse Type'
demand_dict = get_dict_data(input_files, csv_name, value_column_name, filter_column_name)
demand_ag_group_dict,legend_colors = get_colors(demand_dict, 'tools/land use colors.xlsx', sheet_name='lu')

Objective_demand_dict = get_dict_sum_data(input_files, "quantity_comparison", 'Demand (tonnes, KL)', 'Objectives')
# Product_demand_dict = get_dict_sum_data(input_files, "quantity_production_kt_separate", 'Production (tonnes, KL)', 'Production')
# point_dict = concatenate_dicts_by_year([Objective_demand_dict, Product_demand_dict])
point_dict = Objective_demand_dict
point_colors = ['red']
y_range, y_ticks = calculate_y_axis_range(demand_dict)

output_png = '../output/08_food.png'
plot_Combination_figures(demand_ag_group_dict, output_png, input_files, plot_stacked_area_and_line, legend_colors,point_dict=point_dict,point_colors=point_colors,
                            n_rows=3, n_cols=3, font_size=font_size, x_range=(2010, 2050), y_range=y_range,
                             x_ticks=20, y_ticks=y_ticks,
                             legend_position=(0.5, -0.25), show_legend='last', legend_n_rows=7)

csv_name, value_column_name, filter_column_name = "quantity_production_kt_separate", 'Production (tonnes, KL)', 'Landuse'
food_ag_dict = get_dict_data(input_files, csv_name, value_column_name, filter_column_name,['Landuse subtype','Land management'],['Agricultural Landuse','dry'])
# food_ag_group_dict = aggregate_by_mapping(food_ag_dict, 'tools/land use group.xlsx', 'desc', 'ag')
food_ag_dict,legend_colors = get_colors(food_ag_dict, 'tools/land use colors.xlsx', sheet_name='ag')
output_png = '../output/09_food_ag_dry'
y_range, y_ticks = calculate_y_axis_range(food_ag_dict,4)
plot_Combination_figures(food_ag_dict, output_png, input_files, plot_stacked_area, legend_colors,
                            n_rows=3, n_cols=3, font_size=font_size, x_range=(2010, 2050), y_range=y_range,
                             x_ticks=20, y_ticks=y_ticks,
                             legend_position=(0.5, -0.25), show_legend='last', legend_n_rows=2)

csv_name, value_column_name, filter_column_name = "quantity_production_kt_separate", 'Production (tonnes, KL)', 'Landuse'
food_ag_dict = get_dict_data(input_files, csv_name, value_column_name, filter_column_name,['Landuse subtype','Land management'],['Agricultural Landuse','irr'])
# food_ag_group_dict = aggregate_by_mapping(food_ag_dict, 'tools/land use group.xlsx', 'desc', 'ag')
food_ag_dict,legend_colors = get_colors(food_ag_dict, 'tools/land use colors.xlsx', sheet_name='ag')
output_png = '../output/09_food_ag_irr'
y_range, y_ticks = calculate_y_axis_range(food_ag_dict,4)
plot_Combination_figures(food_ag_dict, output_png, input_files, plot_stacked_area, legend_colors,
                            n_rows=3, n_cols=3, font_size=font_size, x_range=(2010, 2050), y_range=y_range,
                             x_ticks=20, y_ticks=y_ticks,
                             legend_position=(0.5, -0.25), show_legend='last', legend_n_rows=2)




