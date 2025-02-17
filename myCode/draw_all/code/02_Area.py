import sys

from tools.data_helper import *
from tools.plot_helper import *
from tools.parameters import *

# 获取到文件的绝对路径，并将其父目录添加到 sys.path
sys.path.append(os.path.abspath('../../../luto'))

# 导入 settings.py
import settings

font_size = 30
csv_name, value_column_name, filter_column_name = 'area_agricultural_landuse', 'Area (ha)', 'Land-use'
area_dict = get_dict_data(input_files, csv_name, value_column_name, filter_column_name)
area_group_dict = aggregate_by_mapping(area_dict, 'tools/land use group.xlsx', 'desc', 'ag_group')
area_ag_group_dict,legend_colors = get_colors(area_group_dict, 'tools/land use colors.xlsx', sheet_name='ag_group')
output_png = '../output/02_area_ag_group'
plot_Combination_figures(area_ag_group_dict, output_png, input_files, plot_stacked_bar, legend_colors,
                            n_rows=3, n_cols=3, font_size=font_size, x_range=(2010, 2050), y_range=(0, 480),
                             x_ticks=20, y_ticks=160,
                             legend_position=(0.5, -0.25), show_legend='last', legend_n_rows=2)

csv_name, value_column_name, filter_column_name = 'area_agricultural_management', 'Area (ha)', 'Type'
area_am_dict = get_dict_data(input_files, csv_name, value_column_name, filter_column_name)
area_am_dict,legend_colors = get_colors(area_am_dict, 'tools/land use colors.xlsx', sheet_name='am')
y_range, y_ticks = calculate_y_axis_range(area_am_dict)
output_png = '../output/02_area_am_group'
plot_Combination_figures(area_am_dict, output_png, input_files, plot_stacked_bar, legend_colors,
                            n_rows=3, n_cols=3, font_size=font_size, x_range=(2010, 2050), y_range=y_range,
                             x_ticks=20, y_ticks=y_ticks,
                             legend_position=(0.5, -0.25), show_legend='last', legend_n_rows=2)

csv_name, value_column_name, filter_column_name = 'area_non_agricultural_landuse', 'Area (ha)', 'Land-use'
area_non_ag_dict = get_dict_data(input_files, csv_name, value_column_name, filter_column_name)
area_non_ag_dict,legend_colors = get_colors(area_non_ag_dict, 'tools/land use colors.xlsx', sheet_name='non_ag')
output_png = '../output/02_area_non_ag_group'
plot_Combination_figures(area_non_ag_dict, output_png, input_files, plot_stacked_bar, legend_colors,
                            n_rows=3, n_cols=3, font_size=font_size, x_range=(2010, 2050), y_range=(0, 75),
                             x_ticks=20, y_ticks=25,
                             legend_position=(0.5, -0.25), show_legend='last', legend_n_rows=2)

