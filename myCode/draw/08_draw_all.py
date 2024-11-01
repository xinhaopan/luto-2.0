import numpy as np
import sys
import os
import pandas as pd
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from tools.helper import *
from tools.parameters import *

plt.rcParams['font.family'] = 'Arial'

# 设置输入路径
sys.path.append(os.path.abspath('../../luto'))
import settings

# 定义绘图函数，包含数据合并步骤
def plot_data(input_files, csv_name, filter_column_name, value_column_name, mapping_file, output_png,
              y_range=None, y_ticks=None, y_label='', combine_data=False, legend_n_rows=1, legend_position=(0.5, -0.05)):
    # 获取数据并根据需求进行合并
    data_dict = get_value_sum(input_files, csv_name, filter_column_name, value_column_name)
    if combine_data:
        data_dict = process_land_use_tables(data_dict)  # 合并相关类别

    # 准备绘图所需的颜色和分类数据
    draw_dict, categories, color_list = prepare_draw_data(data_dict, mapping_file)

    plot_Combination_figures(
        draw_dict, output_png=output_png, input_names=input_files, plot_func=plot_stacked_bar,
        categories=categories, color_list=color_list,
        n_rows=3, n_cols=3, font_size=11,
        x_range=(2010, 2050), y_range=y_range,
        x_ticks=10, y_ticks=y_ticks, x_label='Year', y_label=y_label,
        legend_position=legend_position, show_legend='last', label_positions=(0.005, 0.99),
        legend_n_rows=legend_n_rows  # 设置图例行数
    )

# 绘制各项图表，结合合并步骤

# 1. 农业用地面积（包含合并步骤）
plot_data(input_files, 'area_agricultural_landuse', 'Land-use', 'Area (ha)', 'tools/ag_colors.csv',
          '08_Agricultural_land_area.png', y_range=(0, 500), y_ticks=100, y_label='Area (Million ha)',
          combine_data=True, legend_position=(0.5, -0.03))

# 2. 农业管理面积（无合并）
plot_data(input_files, 'area_agricultural_management', 'Type', 'Area (ha)', 'tools/ammap_colors.csv',
          '09_Agricultural_management_area.png', y_range=(0, 120), y_ticks=40, y_label='Area (Million ha)',
          legend_position=(0.5, -0.03))

# 3. 非农业用地面积（无合并，图例为两行，位置微调）
plot_data(input_files, 'area_non_agricultural_landuse', 'Land-use', 'Area (ha)', 'tools/non_ag_colors.csv',
          '10_Non-agricultural_area.png', y_range=(0, 75), y_ticks=25, y_label='Area (Million ha)',
          legend_n_rows=2, legend_position=(0.5, -0.05))

# 4. 温室气体排放 - 农业土地使用（包含合并步骤）
plot_data(input_files, 'GHG_emissions_separate_agricultural_landuse', 'Land-use', 'Value (t CO2e)', 'tools/ag_colors.csv',
          '12_GHG_agricultural_land.png', y_range=(-1, 100), y_ticks=25, y_label='GHG Emission (Mt CO2e)',
          combine_data=True, legend_position=(0.5, -0.03))

# 5. 温室气体排放 - 农业管理（无合并）
plot_data(input_files, 'GHG_emissions_separate_agricultural_management', 'Agricultural Management Type', 'Value (t CO2e)', 'tools/ammap_colors.csv',
          '13_GHG_agricultural_management_area.png', y_range=(-80, 0), y_ticks=20, y_label='GHG Emission (Mt CO2e)',
          legend_position=(0.5, -0.03))

# 6. 温室气体排放 - 非农业（无合并，图例为两行，位置微调）
plot_data(input_files, 'GHG_emissions_separate_no_ag_reduction', 'Land-use', 'Value (t CO2e)', 'tools/non_ag_colors.csv',
          '14_GHG_non-agricultural.png', y_range=(-800, 0), y_ticks=200, y_label='GHG Emission (Mt CO2e)',
          legend_n_rows=2, legend_position=(0.5, -0.05))

# 7. 生物多样性（包含合并步骤）
plot_data(input_files, 'biodiversity_separate', 'Landuse type', 'Biodiversity score', 'tools/land_colors.csv',
          '15_Biodiversity.png', y_range=(0, 120), y_ticks=40, y_label='Quality-weighted Area (Million ha)',
           legend_position=(0.5, -0.03))

# 8. 生物多样性 - 农业土地（包含合并步骤）
plot_data(input_files, 'biodiversity_separate', 'Landuse', 'Biodiversity score', 'tools/ag_colors.csv',
          '16_Biodiversity_agricultural_land.png', y_range=(0, 105), y_ticks=35, y_label='Quality-weighted Area (Million ha)',
          combine_data=True, legend_position=(0.5, -0.03))

# 9. 生物多样性 - 农业管理（无合并）
plot_data(input_files, 'biodiversity_separate', 'Landuse subtype', 'Biodiversity score', 'tools/ammap_colors.csv',
          '17_Biodiversity_agricultural_management.png', y_range=(0, 3), y_ticks=1, y_label='Quality-weighted Area (Million ha)',
          legend_position=(0.5, -0.03))

# 10. 生物多样性 - 非农业（包含合并步骤，图例为两行，位置微调）
plot_data(input_files, 'biodiversity_separate', 'Landuse', 'Biodiversity score', 'tools/non_ag_colors.csv',
          '18_Biodiversity_non-agricultural.png', y_range=(0, 16), y_ticks=4, y_label='Quality-weighted Area (Million ha)',
           legend_n_rows=2, legend_position=(0.5, -0.05))

