import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import sys
from scipy.interpolate import interp1d
from tools.parameters import *
from tools.data_helper import *
from tools.plot_helper import *

# 获取到文件的绝对路径，并将其父目录添加到 sys.path
sys.path.append(os.path.abspath('../../../luto'))

# 导入 settings.py
import settings

plt.rcParams['font.family'] = 'Arial'

font_size = 20
axis_linewidth = 1
# 生成颜色函数
def draw_plot_lines(df, colors, ylabel, y_range, y_ticks, output_file, font_size=12):
    """绘制 GHG Emissions 点线图."""

    # 创建图形和轴对象
    fig, ax = plt.subplots(figsize=(10, 8))

    ax.spines['left'].set_linewidth(axis_linewidth)  # 左边框
    ax.spines['bottom'].set_linewidth(axis_linewidth)  # 底边框
    ax.spines['right'].set_linewidth(axis_linewidth)  # 右边框
    ax.spines['top'].set_linewidth(axis_linewidth)  # 顶边框

    # 遍历 df 的列名和颜色列表，绘制每条线
    for i, column in enumerate(df.columns):
        ax.plot(df.index, df[column], marker='o', color=colors[i], linewidth=2.5, label=column)

    # 设置刻度朝内
    ax.tick_params(axis='both', which='both', labelsize=font_size, pad=10, direction='out')

    # 设置 x 轴刻度为年份，旋转 90 度
    ax.set_xlim(2010, 2050)
    ax.set_xticks(range(2010, 2051, 10))  # 假设 df 的索引是年份
    # ax.set_xlabel('Year', fontsize=font_size)
    ax.tick_params(axis='x', labelsize=font_size, pad=10, direction='out')

    # 设置 y 轴范围和间隔，传入的范围参数和间隔
    ax.set_ylim(y_range[0], y_range[1])
    ax.set_yticks(y_ticks)
    ax.set_ylabel(ylabel, fontsize=font_size)
    ax.tick_params(axis='y', labelsize=font_size, pad=10, direction='out')

    # # 设置图例，并去掉框
    # ax.legend(fontsize=font_size, frameon=False,loc='lower left')
    #
    # # 显示图形并保存
    # plt.tight_layout()
    # plt.savefig(output_file, bbox_inches='tight', transparent=True, dpi=300)
    # plt.show()
    handles, labels = ax.get_legend_handles_labels()
    legend_file = f"{output_file}" + "_legend.pdf"
    save_legend_as_image(handles, labels, legend_file,ncol=1, font_size=10,format='pdf')
    # 调整布局
    plt.tight_layout()
    save_figure(fig, output_file)
    plt.show()

def draw_coloum(data, legend_colors, output_file, fontsize=22, y_range=(0, 200), y_tick_interval=50, ylabel='Food Demand (million tonnes/kilolitres [milk])'):
    # 创建新的图形
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.spines['left'].set_linewidth(axis_linewidth)  # 左边框
    ax.spines['bottom'].set_linewidth(axis_linewidth)  # 底边框
    ax.spines['right'].set_linewidth(axis_linewidth)  # 右边框
    ax.spines['top'].set_linewidth(axis_linewidth)  # 顶边框

    bottom = np.zeros(len(data))  # 初始位置

    for column, color in legend_colors.items():
        ax.bar(data.index, data[column], bottom=bottom, label=column, color=color)
        bottom += data[column].values  # 更新每个系列的底部位置

    # 设置 Y 轴范围和刻度，传入的范围参数和间隔
    ax.set_ylim(y_range[0], y_range[1])
    ax.set_yticks(np.arange(y_range[0], y_range[1] + 1, y_tick_interval))

    # 添加标题和标签
    ax.yaxis.set_label_coords(-0.06, -1)  # 调整 y 轴标签位置
    ax.set_ylabel(ylabel, fontsize=fontsize)  # y轴标签字体大小

    ax.set_xlim(2009.5, 2050.5)
    ax.set_xticks(range(2010, 2051, 10))  # 假设 df 的索引是年份
    # ax.set_xlabel('Year', fontsize=font_size)
    ax.tick_params(axis='x', labelsize=font_size, pad=10, direction='out')

    # 设置刻度朝内
    ax.tick_params(axis='y', labelsize=fontsize, pad=10, direction='out')  # y轴刻度字体大小
    ax.tick_params(axis='x', labelsize=fontsize, pad=10, direction='out')  # x轴刻度字体大小

    handles, labels = ax.get_legend_handles_labels()
    legend_file = f"{output_file}" + "_legend.svg"
    save_legend_as_image(handles, labels, legend_file,ncol=2, font_size=10)
    # 调整布局
    plt.tight_layout()
    save_figure(fig, output_file)
    plt.show()


# Create a dictionary to hold the annual biodiversity target proportion data for GBF Target 2
def get_biodiversity_target(input_files):
    # Get biodiversity target data from input files
    # Extract dictionary of DataFrames
    bio_target_dict = get_dict_data(input_files, "biodiversity_targets", 'Score', 'Variable')

    # Find keys that contain the specified patterns
    # bio_0_key = next((k for k in bio_target_dict if '1_5C_67_BIO_0' in k), None)
    bio_30_key = next((k for k in bio_target_dict if 'BIO_Moderate' in k), None)
    bio_50_key = next((k for k in bio_target_dict if 'BIO_High' in k), None)

    # Extract the DataFrames corresponding to these keys
    # df_0 = bio_target_dict.get(bio_0_key)
    df_30 = bio_target_dict.get(bio_30_key)
    df_50 = bio_target_dict.get(bio_50_key)

    # Create a new DataFrame using the index from the first DataFrame
    new_df = pd.DataFrame(index=df_30.index)

    # Extract 'Biodiversity score limit' column from each DataFrame
    # and add to the new DataFrame with the appropriate column names
    # new_df['0%'] = df_0['Biodiversity score limit']
    new_df['30%'] = df_30['Biodiversity score limit']
    new_df['50%'] = df_50['Biodiversity score limit']

    return new_df


font_size = 25
# Bio
df = get_biodiversity_target(input_files)
# df = pd.read_csv('biodiversity_targets.csv', index_col=0)
min_v, max_v, ticks = get_y_axis_ticks(df.min().min(), df.max().max(), desired_ticks=5)
colors = ['#2ECC71', '#3498DB','#E74C3C']  # 根据数据列数调整颜色列表
draw_plot_lines(df, colors, ' ', (min_v, max_v), ticks, "../output/03_biodiversity_limit.png", font_size=font_size)


# GHG
# 读取 Excel 文件
df = pd.read_excel(INPUT_DIR + '/GHG_targets.xlsx', index_col=0)

# 选择 2010 到 2050 年的数据
df_filtered = df.loc[2010:2050,
              ['1.5C (67%) excl. avoided emis SCOPE1', '1.5C (50%) excl. avoided emis SCOPE1', '1.8C (67%) excl. avoided emis SCOPE1']]

# 将所有数据单位转换为 million (除以 1,000,000)
df_filtered = df_filtered / 1e6
df_filtered.columns = ['1.5°C (67%)', '1.5°C (50%)', '1.8°C (67%)']
colors = ['#E74C3C', '#3498DB', '#2ECC71']  # 根据数据列数调整颜色列表
min_v, max_v, ticks = get_y_axis_ticks(df_filtered.min().min(), df_filtered.max().max(), desired_ticks=6)
draw_plot_lines(df_filtered, colors, ' ', (min_v, max_v), ticks, "../output/03_GHG_limit.png", font_size=font_size)


# Food demand
dd = pd.read_hdf(os.path.join(INPUT_DIR, 'demand_projections.h5'))
demand_data = dd.loc[(settings.SCENARIO,
                      settings.DIET_DOM,
                      settings.DIET_GLOB,
                      settings.CONVERGENCE,
                      settings.IMPORT_TREND,
                      settings.WASTE,
                      settings.FEED_EFFICIENCY),
'PRODUCTION'].copy()

# 处理 'eggs' 数据
demand_data.loc['eggs'] = demand_data.loc['eggs'] * settings.EGGS_AVG_WEIGHT / 1000 / 1000
demand_data = demand_data.T / 1e6
demand_data = demand_data.loc[2010:2050]
demand_data = demand_data.drop(columns=['aquaculture', 'chicken', 'eggs', 'pork'])

mapping_df = pd.read_excel('tools/land use colors.xlsx', sheet_name='food')
demand_data, legend_colors = process_single_df(demand_data, mapping_df)
draw_coloum(demand_data, legend_colors, '../output/03_Food_demand.png', fontsize=font_size, y_range=(0, 200), y_tick_interval=50, ylabel=' ')

# water
dd = pd.read_hdf(os.path.join(INPUT_DIR, "draindiv_lut.h5"), index_col='HR_DRAINDIV_ID')

# 生成 2010 到 2050 的年份作为行索引
years = pd.Index(range(2010, 2051), name='Year')

# 创建一个以年份为索引的 DataFrame
dd_yield_df = pd.DataFrame(index=years)

# 构建最终的 dd_yield_df
for name, value in zip(dd['HR_DRAINDIV_NAME'], dd['WATER_YIELD_HIST_BASELINE_ML']):
    dd_yield_df[name] = value * (1 - settings.WATER_STRESS * settings.AG_SHARE_OF_WATER_USE) / 1e6

dd_outside_luto = pd.read_hdf(os.path.join(INPUT_DIR, 'water_yield_outside_LUTO_study_area_2010_2100_dd_ml.h5'))
water_yield_natural_land = dd_outside_luto.loc[:, pd.IndexSlice[:, settings.SSP]] / 1e6
water_cci_delta = pd.DataFrame()
for col_name, col_data in water_yield_natural_land.items():
    min_gap = col_data - col_data.min()                 # Climate change impact is the difference between the minimum water yield and the current value
    before_min = col_data.index < col_data.idxmin()     # The impact is only applied to years before the minimum value
    min_gap = min_gap * before_min                      # Apply the impact only to years before the minimum value
    min_gap_df = pd.DataFrame({col_name: min_gap})
    water_cci_delta = pd.concat([water_cci_delta, min_gap_df ], axis=1)

water_cci_delta.columns = dd_yield_df.columns
# 对两个 DataFrame 逐列相加
dd_water_limit_df = dd_yield_df + water_cci_delta
# dd_water_limit_df = dd_yield_df

mapping_df = pd.read_excel('tools/land use colors.xlsx', sheet_name='water')
dd_water_limit_df, legend_colors = process_single_df(dd_water_limit_df, mapping_df)
draw_coloum(dd_water_limit_df, legend_colors, '../output/03_water_limit.png', fontsize=font_size, y_range=(0, 300), y_tick_interval=100, ylabel=' ')

