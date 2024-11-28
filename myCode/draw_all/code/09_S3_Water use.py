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

font_size = 35
csv_name, value_column_name, filter_column_name = 'water_yield_separate', 'Value (ML)', 'Landuse Type'
water_all_dict = get_dict_data(input_files, csv_name, value_column_name, filter_column_name)
water_public_dict = get_dict_sum_data(input_files, 'water_yield_of_climate_change_impacts_outside_LUTO', 'Climate Change Impact (ML)', 'Public land')
water_dict = concatenate_dicts_by_year([water_all_dict, water_public_dict])
water_dict,legend_colors = get_colors(water_dict, 'tools/land use colors.xlsx', sheet_name='lu')
output_png = '../output/09_S3_water.png'
plot_Combination_figures(water_dict, output_png, input_files, plot_stacked_bar, legend_colors,
                            n_rows=3, n_cols=3, font_size=font_size, x_range=(2010, 2050), y_range=(0, 300),
                             x_ticks=10, y_ticks=100,
                             legend_position=(0.5, -0.25), show_legend='last', legend_n_rows=2)

csv_name, value_column_name, filter_column_name = 'water_yield_separate', 'Value (ML)',  'Landuse'
water_ag_dict = get_dict_data(input_files, csv_name, value_column_name, filter_column_name)
water_ag_group_dict = aggregate_by_mapping(water_ag_dict, 'tools/land use group.xlsx', 'desc', 'ag_group')
water_dict,legend_colors = get_colors(water_ag_group_dict, 'tools/land use colors.xlsx', sheet_name='ag_group')
output_png = '../output/09_S3_water_ag_group.png'
plot_Combination_figures(water_dict, output_png, input_files, plot_stacked_bar, legend_colors,
                            n_rows=3, n_cols=3, font_size=font_size, x_range=(2010, 2050), y_range=(0, 120),
                             x_ticks=10, y_ticks=40,
                             legend_position=(0.5, -0.25), show_legend='last', legend_n_rows=2)

csv_name, value_column_name, filter_column_name = 'water_yield_separate', 'Value (ML)',  'Landuse subtype'
water_am_dict = get_dict_data(input_files, csv_name, value_column_name, filter_column_name)
water_dict,legend_colors = get_colors(water_am_dict, 'tools/land use colors.xlsx', sheet_name='am')
output_png = '../output/09_S3_water_am.png'
plot_Combination_figures(water_dict, output_png, input_files, plot_stacked_bar, legend_colors,
                            n_rows=3, n_cols=3, font_size=font_size, x_range=(2010, 2050), y_range=(0, 4),
                             x_ticks=10, y_ticks=1,
                             legend_position=(0.5, -0.25), show_legend='last', legend_n_rows=2)

csv_name, value_column_name, filter_column_name = 'water_yield_separate', 'Value (ML)', 'Landuse'
water_am_dict = get_dict_data(input_files, csv_name, value_column_name, filter_column_name)
water_dict,legend_colors = get_colors(water_am_dict, 'tools/land use colors.xlsx', sheet_name='non_ag')
output_png = '../output/09_S3_water_non-ag.png'
plot_Combination_figures(water_dict, output_png, input_files, plot_stacked_bar, legend_colors,
                            n_rows=3, n_cols=3, font_size=font_size, x_range=(2010, 2050), y_range=(0, 24),
                             x_ticks=10, y_ticks=8,
                             legend_position=(0.5, -0.25), show_legend='last', legend_n_rows=2)



