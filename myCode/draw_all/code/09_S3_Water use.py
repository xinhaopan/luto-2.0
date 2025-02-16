# import numpy as np
# import sys
# import os
# import pandas as pd
# from scipy.interpolate import interp1d
# import matplotlib.pyplot as plt
# from PIL import Image, ImageDraw, ImageFont
#
# from tools.data_helper import *
# from tools.plot_helper import *
# from tools.parameters import *
#
# plt.rcParams['font.family'] = 'Arial'
#
# # 获取到文件的绝对路径，并将其父目录添加到 sys.path
# sys.path.append(os.path.abspath('../../../luto'))
#
# # 导入 settings.py
# import settings
#
# font_size = 15
# csv_name, value_column_name, filter_column_name = 'water_yield_separate', 'Water Net Yield (ML)', 'region'
# area_dict = get_dict_data(input_files, csv_name, value_column_name, filter_column_name)
# area_ag_group_dict,legend_colors = get_colors(area_dict, 'tools/land use colors.xlsx', sheet_name='water')
# output_png = '../output/09_S3_water.png'
# plot_Combination_figures(area_ag_group_dict, output_png, input_files, plot_stacked_bar, legend_colors,
#                             n_rows=3, n_cols=3, font_size=font_size, x_range=(2010, 2050), y_range=(0, 300),
#                              x_ticks=10, y_ticks=100,
#                              legend_position=(0.5, -0.25), show_legend='last', legend_n_rows=4)

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

csv_name, value_column_name, filter_column_name = 'water_yield_separate', 'Value (ML)', 'Landuse Type'
water_dict = get_dict_from_json(input_files, 'water_1_water_net_use_by_broader_category')
water_dict,legend_colors = get_colors(water_dict, 'tools/land use colors.xlsx', sheet_name='lu')

water_limit_dict = get_dict_data(input_files, 'water_yield_limits_and_public_land', 'Value (ML)', 'Type')
water_yeild_dict ={key: df.assign(**{'Yield': df.sum(axis=1)}) for key, df in water_dict.items()}
water_point_dict = concatenate_dicts_by_year([water_limit_dict, water_yeild_dict])
water_point_dict = rename_and_filter_columns(water_point_dict, ['WNY LIMIT','Yield'], ['Constarints','Yield'])
point_colors = ['red','black']

output_png = '../output/09_S3_water.png'
y_range, y_ticks = calculate_y_axis_range(water_dict)
plot_Combination_figures(water_dict, output_png, input_files, plot_stacked_bar_and_line, legend_colors,point_dict=water_point_dict,point_colors=point_colors,
                            n_rows=3, n_cols=3, font_size=font_size, x_range=(2010, 2050), y_range=y_range,
                             x_ticks=20, y_ticks=y_ticks,
                             legend_position=(0.5, -0.25), show_legend='last', legend_n_rows=2)


csv_name, value_column_name, filter_column_name = 'water_yield_separate', 'Value (ML)',  'Landuse'
water_ag_dict = get_dict_data(input_files, csv_name, value_column_name, filter_column_name)
water_ag_group_dict = aggregate_by_mapping(water_ag_dict, 'tools/land use group.xlsx', 'desc', 'ag_group')
water_ag_group_dict,legend_colors = get_colors(water_ag_group_dict, 'tools/land use colors.xlsx', sheet_name='ag_group')
output_png = '../output/09_S3_water_ag_group.png'
y_range, y_ticks = calculate_y_axis_range(water_ag_group_dict)
plot_Combination_figures(water_ag_group_dict, output_png, input_files, plot_stacked_bar, legend_colors,
                            n_rows=3, n_cols=3, font_size=font_size, x_range=(2010, 2050), y_range=y_range,
                             x_ticks=20, y_ticks=y_ticks,
                             legend_position=(0.5, -0.25), show_legend='last', legend_n_rows=2)

csv_name, value_column_name, filter_column_name = 'water_yield_separate', 'Value (ML)',  'Landuse subtype'
water_am_dict = get_dict_data(input_files, csv_name, value_column_name, filter_column_name)
water_am_dict,legend_colors = get_colors(water_am_dict, 'tools/land use colors.xlsx', sheet_name='am')
output_png = '../output/09_S3_water_am.png'
y_range, y_ticks = calculate_y_axis_range(water_am_dict,  multiplier=1)
plot_Combination_figures(water_am_dict, output_png, input_files, plot_stacked_bar, legend_colors,
                            n_rows=3, n_cols=3, font_size=font_size, x_range=(2010, 2050), y_range=y_range,
                             x_ticks=20, y_ticks=y_ticks,
                             legend_position=(0.5, -0.25), show_legend='last', legend_n_rows=2)

csv_name, value_column_name, filter_column_name = 'water_yield_separate', 'Value (ML)', 'Landuse'
water_non_ag_dict = get_dict_data(input_files, csv_name, value_column_name, filter_column_name)
water_non_ag_dict,legend_colors = get_colors(water_non_ag_dict, 'tools/land use colors.xlsx', sheet_name='non_ag')
output_png = '../output/09_S3_water_non-ag.png'
y_range, y_ticks = calculate_y_axis_range(water_dict)
plot_Combination_figures(water_non_ag_dict, output_png, input_files, plot_stacked_bar, legend_colors,
                            n_rows=3, n_cols=3, font_size=font_size, x_range=(2010, 2050), y_range=y_range,
                             x_ticks=20, y_ticks=y_ticks,
                             legend_position=(0.5, -0.25), show_legend='last', legend_n_rows=2)



