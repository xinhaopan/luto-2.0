from tools.data_helper import *
from tools.plot_helper import *
from tools.parameters import *
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import sys
from scipy.interpolate import interp1d

# 获取到文件的绝对路径，并将其父目录添加到 sys.path
sys.path.append(os.path.abspath('../../../luto'))

# 导入 settings.py
import settings
plt.rcParams['font.family'] = 'Arial'

ag_dict = get_dict_data(input_files, 'GHG_emissions_separate_agricultural_landuse','Value (t CO2e)',  'Land-use')
ag_group_dict = aggregate_by_mapping(ag_dict, 'tools/land use group.xlsx', 'desc', 'ag_group')
am_dict = get_dict_data(input_files, 'GHG_emissions_separate_agricultural_management', 'Value (t CO2e)', 'Agricultural Management Type')
# off_land_dict = get_value_sum(input_files, 'GHG_emissions_offland_commodity',  '*', 'Total GHG Emissions (tCO2e)')
non_ag_dict = get_dict_data(input_files, 'GHG_emissions_separate_no_ag_reduction', 'Value (t CO2e)', 'Land-use')
transition_dict = get_dict_data(input_files, 'GHG_emissions_separate_transition_penalty', 'Value (t CO2e)','Type')

ghg_dict = concatenate_dicts_by_year([ag_dict, am_dict, non_ag_dict, transition_dict])
ghg_group_dict = aggregate_by_mapping(ghg_dict, 'tools/land use group.xlsx', 'desc', 'lu_group', summary_col_name='Net emissions')

font_size = 35
ghg_group_dict,legend_colors = get_colors(ghg_group_dict, 'tools/land use colors.xlsx', sheet_name='lu')
output_png = '../output/04_ghg_emissions.png'
plot_Combination_figures(ghg_group_dict, output_png, input_files, plot_stacked_bar_and_line, legend_colors,
                             point_data='Net emissions', n_rows=3, n_cols=3, font_size=font_size, x_range=(2010, 2050), y_range=(-800, 201),
                             x_ticks=10, y_ticks=200,
                             legend_position=(0.5, -0.25), show_legend='last', legend_n_rows=2)

draw_dict,legend_colors = get_colors(ag_group_dict, 'tools/land use colors.xlsx', sheet_name='ag_group')
output_png = '../output/04_ghg_ag_emissions.png'
plot_Combination_figures(draw_dict, output_png, input_files, plot_stacked_bar, legend_colors,
                             point_data=None, n_rows=3, n_cols=3, font_size=font_size, x_range=(2010, 2050), y_range=(-1, 100),
                             x_ticks=10, y_ticks=25,
                             legend_position=(0.5, -0.25), show_legend='last', legend_n_rows=2)

draw_dict,legend_colors = get_colors(am_dict, 'tools/land use colors.xlsx', sheet_name='am')
output_png = '../output/04_ghg_am_emissions.png'
plot_Combination_figures(draw_dict, output_png, input_files, plot_stacked_bar, legend_colors,
                             point_data=None, n_rows=3, n_cols=3, font_size=font_size, x_range=(2010, 2050), y_range=(-80, 20),
                             x_ticks=10, y_ticks=20,
                             legend_position=(0.5, -0.3), show_legend='last', legend_n_rows=3)

draw_dict,legend_colors = get_colors(non_ag_dict, 'tools/land use colors.xlsx', sheet_name='non_ag')
output_png = '../output/04_ghg_non_ag_emissions.png'
plot_Combination_figures(draw_dict, output_png, input_files, plot_stacked_bar, legend_colors,
                             point_data=None, n_rows=3, n_cols=3, font_size=font_size, x_range=(2010, 2050), y_range=(-800, 0),
                             x_ticks=10, y_ticks=200,
                             legend_position=(0.5, -0.45), show_legend='last', legend_n_rows=4)