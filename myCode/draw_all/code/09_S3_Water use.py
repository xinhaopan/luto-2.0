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

csv_name, value_column_name, filter_column_name = 'water_yield_separate', 'Water Net Yield (ML)', 'Type'
water_dict = get_dict_data(input_files, csv_name, value_column_name, filter_column_name)
water_public_dict = get_dict_data(input_files, 'water_yield_limits_and_public_land',
                                 'Water yield outside LUTO (ML)')
water_dict = {
    k: pd.merge(
        v,
        water_public_dict[k][['Total']].rename(columns={'Total': 'Public land'}),
        left_index=True,
        right_index=True,
        how='left'
    )
    for k, v in water_dict.items()
}
water_dict,legend_colors = get_colors(water_dict, 'tools/land use colors.xlsx', sheet_name='lu')
water_yeild_dict ={key: df.assign(**{'Yield': df.sum(axis=1)}) for key, df in water_dict.items()}
water_limit_dict = get_dict_data(input_files, 'water_yield_limits_and_public_land',
                                 'Water Yield Limit (ML)')
water_point_dict = rename_and_filter_columns(water_limit_dict, ['Total'], ['Targets'])
point_colors = ['red']

output_png = '../output/09_S3_water'
y_range, y_ticks = calculate_y_axis_range(water_dict,4)
plot_Combination_figures(water_dict, output_png, input_files, plot_stacked_area_and_line, legend_colors,point_dict=water_point_dict,point_colors=point_colors,
                            n_rows=3, n_cols=3, font_size=font_size, x_range=(2010, 2050), y_range=y_range,
                             x_ticks=20, y_ticks=y_ticks,
                             legend_position=(0.5, -0.25), show_legend='last', legend_n_rows=2)


csv_name, value_column_name, filter_column_name = 'water_yield_separate', 'Water Net Yield (ML)',  'Landuse'
water_ag_dict = get_dict_data(input_files, csv_name, value_column_name, filter_column_name,condition_column_name=['Type'], condition_value=['Agricultural Landuse'])
water_ag_group_dict = aggregate_by_mapping(water_ag_dict, 'tools/land use group.xlsx', 'desc', 'ag_group')
water_ag_group_dict,legend_colors = get_colors(water_ag_group_dict, 'tools/land use colors.xlsx', sheet_name='ag_group')
output_png = '../output/09_S3_water_ag_group'
y_range, y_ticks = calculate_y_axis_range(water_ag_group_dict,4)
plot_Combination_figures(water_ag_group_dict, output_png, input_files, plot_stacked_area, legend_colors,
                            n_rows=3, n_cols=3, font_size=font_size, x_range=(2010, 2050), y_range=y_range,
                             x_ticks=20, y_ticks=y_ticks,
                             legend_position=(0.5, -0.25), show_legend='last', legend_n_rows=2)

csv_name, value_column_name, filter_column_name = 'water_yield_separate', 'Water Net Yield (ML)',  'Agri-Management'
water_am_dict = get_dict_data(input_files, csv_name, value_column_name, filter_column_name,condition_column_name=['Type'], condition_value=['Agricultural Management'])
water_am_dict,legend_colors = get_colors(water_am_dict, 'tools/land use colors.xlsx', sheet_name='am')
output_png = '../output/09_S3_water_am'
y_range, y_ticks = calculate_y_axis_range(water_am_dict,4)
plot_Combination_figures(water_am_dict, output_png, input_files, plot_stacked_area, legend_colors,
                            n_rows=3, n_cols=3, font_size=font_size, x_range=(2010, 2050), y_range=y_range,
                             x_ticks=20, y_ticks=y_ticks,
                             legend_position=(0.5, -0.25), show_legend='last', legend_n_rows=2)

csv_name, value_column_name, filter_column_name = 'water_yield_separate', 'Water Net Yield (ML)', 'Landuse'
water_non_ag_dict = get_dict_data(input_files, csv_name, value_column_name, filter_column_name,condition_column_name=['Type'], condition_value=['Non-Agricultural Landuse'])
water_non_ag_dict,legend_colors = get_colors(water_non_ag_dict, 'tools/land use colors.xlsx', sheet_name='non_ag')
output_png = '../output/09_S3_water_non-ag'
y_range, y_ticks = calculate_y_axis_range(water_non_ag_dict,3)
plot_Combination_figures(water_non_ag_dict, output_png, input_files, plot_stacked_area, legend_colors,
                            n_rows=3, n_cols=3, font_size=font_size, x_range=(2010, 2050), y_range=y_range,
                             x_ticks=20, y_ticks=y_ticks,
                             legend_position=(0.5, -0.25), show_legend='last', legend_n_rows=2)
