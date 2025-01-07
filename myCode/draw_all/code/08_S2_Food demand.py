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

font_size = 25

Objective_demand_dict = get_dict_sum_data(input_files, "quantity_comparison", 'Demand (tonnes, KL)', 'Objectives')
Product_demand_dict = get_dict_sum_data(input_files, "quantity_comparison", 'Prod_targ_year (tonnes, KL)', 'Production')
point_dict = concatenate_dicts_by_year([Objective_demand_dict, Product_demand_dict])
point_colors = ['red','black']

csv_name, value_column_name, filter_column_name = 'quantity_comparison', 'Prod_targ_year (tonnes, KL)', 'Commodity'
demand_dict = get_dict_data(input_files, csv_name, value_column_name, filter_column_name)
demand_ag_group_dict,legend_colors = get_colors(demand_dict, 'tools/land use colors.xlsx', sheet_name='food')
output_png = '../output/08_S2_food.png'
plot_Combination_figures(demand_ag_group_dict, output_png, input_files, plot_stacked_bar_and_line, legend_colors,point_dict=point_dict,point_colors=point_colors,
                            n_rows=3, n_cols=3, font_size=font_size, x_range=(2015, 2050), y_range=(0, 200),
                             x_ticks=10, y_ticks=50,
                             legend_position=(0.5, -0.25), show_legend='last', legend_n_rows=7)

