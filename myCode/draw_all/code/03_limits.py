import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import sys
from scipy.interpolate import interp1d
from tools.parameters import *

from tools.plot_helper import process_single_df

# 获取到文件的绝对路径，并将其父目录添加到 sys.path
sys.path.append(os.path.abspath('../../../luto'))

# 导入 settings.py
import settings

plt.rcParams['font.family'] = 'Arial'

font_size = 20
# 生成颜色函数
def draw_plot_lines(df, colors, ylabel, y_range, y_tick_interval, output_file, font_size=12):
    """绘制 GHG Emissions 点线图."""

    # 创建图形和轴对象
    fig, ax = plt.subplots(figsize=(10, 8))

    # 遍历 df 的列名和颜色列表，绘制每条线
    for i, column in enumerate(df.columns):
        ax.plot(df.index, df[column], marker='o', color=colors[i], linewidth=2.5, label=column)

    # 设置刻度朝内
    ax.tick_params(axis='both', which='both', direction='in', labelsize=font_size, pad=10)

    # 设置 x 轴刻度为年份，旋转 90 度
    ax.set_xlim(2010, 2050)
    ax.set_xticks(range(2010, 2051, 20))  # 假设 df 的索引是年份
    ax.set_xlabel('Year', fontsize=font_size)
    ax.tick_params(axis='x', labelsize=font_size, pad=10)

    # 设置 y 轴范围和间隔，传入的范围参数和间隔
    ax.set_ylim(y_range[0], y_range[1])
    ax.set_yticks(range(y_range[0], y_range[1] + 1, y_tick_interval))  # 以传入的间隔为步长
    ax.set_ylabel(ylabel, fontsize=font_size)
    ax.tick_params(axis='y', labelsize=font_size, pad=10)

    # 设置图例，并去掉框
    ax.legend(fontsize=font_size, frameon=False,loc='lower left')

    # 显示图形并保存
    plt.tight_layout()
    plt.savefig(output_file, bbox_inches='tight', transparent=True, dpi=300)
    plt.show()

def draw_coloum(data, legend_colors, output_file, fontsize=22, y_range=(0, 200), y_tick_interval=50, ylabel='Food Demand (million tonnes/kilolitres [milk])'):
    # 创建新的图形
    fig, ax = plt.subplots(figsize=(10, 8))
    bottom = np.zeros(len(data))  # 初始位置

    for column, color in legend_colors.items():
        ax.bar(data.index, data[column], bottom=bottom, label=column, color=color)
        bottom += data[column].values  # 更新每个系列的底部位置

    # 设置 Y 轴范围和刻度，传入的范围参数和间隔
    ax.set_ylim(y_range[0], y_range[1])
    ax.set_yticks(np.arange(y_range[0], y_range[1] + 1, y_tick_interval))

    # 添加标题和标签
    ax.set_xlabel('Year', fontsize=fontsize)  # x轴标签字体大小
    ax.yaxis.set_label_coords(-0.06, -1)  # 调整 y 轴标签位置
    ax.set_ylabel(ylabel, fontsize=fontsize)  # y轴标签字体大小

    ax.set_xlim(2009.5, 2050.5)
    ax.set_xticks(range(2010, 2051, 20))  # 假设 df 的索引是年份
    ax.set_xlabel('Year', fontsize=font_size)
    ax.tick_params(axis='x', labelsize=font_size, pad=10)

    # 设置刻度朝内
    ax.tick_params(axis='y', direction='in', labelsize=fontsize, pad=10)  # y轴刻度字体大小
    ax.tick_params(axis='x', direction='in', labelsize=fontsize, pad=10)  # x轴刻度字体大小

    # 调整布局
    plt.tight_layout()

    # 保存图表（不含图例）

    plt.savefig(output_file, dpi=300, transparent=True)
    plt.show()

    # 关闭当前图形
    plt.close()

    save_legend(ax, f'{output_file[:-4]}_legend.png',fontsize=fontsize)

def save_legend(ax, output_legend_file='../output/03_Food_legend.png',fontsize=20):
    # 创建一个新的图形来存放图例，调整图形尺寸为小正方形
    fig_legend = plt.figure(figsize=(10, 10))  # 调整尺寸为小正方形

    # 获取图例句柄和标签
    handles, labels = ax.get_legend_handles_labels()

    # 设置图例的列数为3列，自动计算需要的行数
    num_plots = len(labels)
    nrows = 6
    ncols = 2

    # 创建图例，并调整图例形状
    legend = fig_legend.legend(handles, labels, loc='center', frameon=False,
                               handlelength=1,  # 设置图例项的宽度，较短接近正方形
                               handleheight=1,  # 设置图例项的高度，确保接近正方形
                               ncol=ncols)  # 设置图例为 3 列布局

    # 设置图例文本大小，不改变字体大小
    plt.setp(legend.get_texts(), fontsize=fontsize)  # 根据需要调整字体大小

    # 保存图例为图片，确保边界不会截掉
    fig_legend.savefig(output_legend_file, dpi=300, bbox_inches='tight', transparent=True)

    # 关闭图例窗口，释放资源
    plt.close(fig_legend)

# Create a dictionary to hold the annual biodiversity target proportion data for GBF Target 2
def get_biodiversity_target(INPUT_DIR, BIODIV_GBF_TARGET_2_DICTs):
    BIODIV_GBF_TARGET = []
    for i in range(3):
        BIODIV_GBF_TARGET_2_DICT = BIODIV_GBF_TARGET_2_DICTs[i]
        f = interp1d(
            list(BIODIV_GBF_TARGET_2_DICT.keys()),
            list(BIODIV_GBF_TARGET_2_DICT.values()),
            kind="linear",
            fill_value="extrapolate",
        )
        biodiv_GBF_target_2_proportions_2010_2100 = {yr: f(yr).item() for yr in range(2010, 2101)}

        # Get the connectivity score between 0 and 1, where 1 is the highest connectivity
        biodiv_priorities = pd.read_hdf(os.path.join(INPUT_DIR, 'biodiv_priorities.h5'))
        LUMAP_NO_RESFACTOR = pd.read_hdf(os.path.join(INPUT_DIR, "lumap.h5")).to_numpy()
        MASK_LU_CODE = -1
        LUMASK = LUMAP_NO_RESFACTOR != MASK_LU_CODE
        if settings.CONNECTIVITY_SOURCE == 'NCI':
            connectivity_score = biodiv_priorities['DCCEEW_NCI'].to_numpy(dtype=np.float32)
            connectivity_score = np.where(LUMASK, connectivity_score,
                                          1)  # Set the connectivity score to 1 for cells outside the LUMASK
            connectivity_score = np.interp(connectivity_score, (connectivity_score.min(), connectivity_score.max()),
                                           (settings.CONNECTIVITY_LB, 1)).astype('float32')
        elif settings.CONNECTIVITY_SOURCE == 'DWI':
            connectivity_score = biodiv_priorities['NATURAL_AREA_CONNECTIVITY'].to_numpy(dtype=np.float32)
            connectivity_score = np.interp(connectivity_score, (connectivity_score.min(), connectivity_score.max()),
                                           (1, settings.CONNECTIVITY_LB)).astype('float32')
        elif settings.CONNECTIVITY_SOURCE == 'NONE':
            connectivity_score = 1
        else:
            raise ValueError(
                f"Invalid connectivity source: {settings.CONNECTIVITY_SOURCE}, must be 'NCI', 'DWI' or 'NONE'")

        # Get the Zonation output score between 0 and 1. biodiv_score_raw.sum() = 153 million
        biodiv_score_raw = biodiv_priorities['BIODIV_PRIORITY_SSP' + str(settings.SSP)].to_numpy(dtype=np.float32)
        # Weight the biodiversity score by the connectivity score
        BIODIV_SCORE_RAW_WEIGHTED = biodiv_score_raw * connectivity_score

        # Habitat degradation scale for agricultural land-use
        biodiv_degrade_df = pd.read_csv(
            os.path.join(INPUT_DIR, 'HABITAT_CONDITION.csv'))  # Load the HCAS percentile data (pd.DataFrame)

        AGRICULTURAL_LANDUSES = pd.read_csv((os.path.join(INPUT_DIR, 'ag_landuses.csv')), header=None)[0].to_list()
        AGLU2DESC = {i: lu for i, lu in enumerate(AGRICULTURAL_LANDUSES)}
        DESC2AGLU = {value: key for key, value in AGLU2DESC.items()}

        if settings.HABITAT_CONDITION == 'HCAS':
            ''' 
            The degradation weight score of "HCAS" are float values range between 0-1 indicating the suitability for wild animals survival. 
            Here we average this dataset in year 2009, 2010, and 2011, then calculate the percentiles of the average score under each land-use type. 
            '''
            BIODIV_HABITAT_DEGRADE_LOOK_UP = biodiv_degrade_df[['lu',
                                                                f'PERCENTILE_{settings.HCAS_PERCENTILE}']]  # Get the biodiversity degradation score at specified percentile (pd.DataFrame)
            BIODIV_HABITAT_DEGRADE_LOOK_UP = {int(k): v for k, v in dict(
                BIODIV_HABITAT_DEGRADE_LOOK_UP.values).items()}  # Convert the biodiversity degradation score to a dictionary {land-use-code: score}
            unalloc_nat_land_bio_score = BIODIV_HABITAT_DEGRADE_LOOK_UP[DESC2AGLU[
                'Unallocated - natural land']]  # Get the biodiversity degradation score for unallocated natural land (float)
            BIODIV_HABITAT_DEGRADE_LOOK_UP = {k: v * (1 / unalloc_nat_land_bio_score) for k, v in
                                              BIODIV_HABITAT_DEGRADE_LOOK_UP.items()}  # Normalise the biodiversity degradation score to the unallocated natural land score

        elif settings.HABITAT_CONDITION == 'USER_DEFINED':
            BIODIV_HABITAT_DEGRADE_LOOK_UP = biodiv_degrade_df[['lu', 'USER_DEFINED']]
            BIODIV_HABITAT_DEGRADE_LOOK_UP = {int(k): v for k, v in dict(
                BIODIV_HABITAT_DEGRADE_LOOK_UP.values).items()}  # Convert the biodiversity degradation score to a dictionary {land-use-code: score}

        else:
            raise ValueError(
                f"Invalid habitat condition source: {settings.HABITAT_CONDITION}, must be 'HCAS' or 'USER_DEFINED'")

        # Get the biodiversity degradation score (0-1) for each cell
        '''
        The degradation scores are float values range between 0-1 indicating the discount of biodiversity value for each cell.
        E.g., 0.8 means the biodiversity value of the cell is 80% of the original raw biodiversity value.
        '''
        savburn_df = pd.read_hdf(os.path.join(INPUT_DIR, 'cell_savanna_burning.h5'))
        SAVBURN_ELIGIBLE = savburn_df.ELIGIBLE_AREA.to_numpy()
        biodiv_degrade_LDS = np.where(SAVBURN_ELIGIBLE, settings.LDS_BIODIVERSITY_VALUE,
                                      1)  # Get the biodiversity degradation score for LDS burning (1D numpy array)
        biodiv_degrade_habitat = np.vectorize(BIODIV_HABITAT_DEGRADE_LOOK_UP.get)(LUMAP_NO_RESFACTOR).astype(
            np.float32)  # Get the biodiversity degradation score for each cell (1D numpy array)

        # Get the biodiversity damage under LDS burning (0-1) for each cell
        biodiv_degradation_raw_weighted_LDS = BIODIV_SCORE_RAW_WEIGHTED * (
                    1 - biodiv_degrade_LDS)  # Biodiversity damage under LDS burning (1D numpy array)
        biodiv_degradation_raw_weighted_habitat = BIODIV_SCORE_RAW_WEIGHTED * (
                    1 - biodiv_degrade_habitat)  # Biodiversity damage under under HCAS (1D numpy array)

        # Get the biodiversity value at the beginning of the simulation
        BIODIV_RAW_WEIGHTED_LDS = BIODIV_SCORE_RAW_WEIGHTED - biodiv_degradation_raw_weighted_LDS  # Biodiversity value under LDS burning (1D numpy array); will be used as base score for calculating ag/non-ag stratagies impacts on biodiversity
        REAL_AREA = pd.read_hdf(os.path.join(INPUT_DIR, "real_area.h5")).to_numpy()
        REAL_AREA_NO_RESFACTOR = REAL_AREA.copy()
        biodiv_current_val = BIODIV_RAW_WEIGHTED_LDS - biodiv_degradation_raw_weighted_habitat  # Biodiversity value at the beginning year (1D numpy array)
        biodiv_current_val = np.nansum(biodiv_current_val[LUMASK] * REAL_AREA_NO_RESFACTOR[
            LUMASK])  # Sum the biodiversity value within the LUMASK

        # Biodiversity values need to be restored under the GBF Target 2
        '''
        The biodiversity value to be restored is calculated as the difference between the 'Unallocated - natural land' 
        and 'current land-use' regarding their biodiversity degradation scale.
        '''
        biodiv_degradation_val = (
                biodiv_degradation_raw_weighted_LDS +  # Biodiversity degradation from HCAS
                biodiv_degradation_raw_weighted_habitat  # Biodiversity degradation from LDS burning
        )

        biodiv_degradation_val = np.nansum(biodiv_degradation_val[LUMASK] * REAL_AREA_NO_RESFACTOR[
            LUMASK])  # Sum the biodiversity degradation value within the LUMASK

        # Multiply by biodiversity target to get the additional biodiversity score required to achieve the target
        BIODIV_GBF_TARGET_2 = {
            yr: biodiv_current_val + biodiv_degradation_val * biodiv_GBF_target_2_proportions_2010_2100[yr]
            for yr in range(2010, 2101)
        }
        BIODIV_GBF_TARGET.append(BIODIV_GBF_TARGET_2)
    df = pd.DataFrame(BIODIV_GBF_TARGET)
    df_biodiversity = df.T.loc[2010:2050]
    df_biodiversity.index.name = 'Year'
    df_biodiversity.columns = ['0%', '30%', '50%']
    return df_biodiversity

font_size = 25
# GHG
# 读取 Excel 文件
df = pd.read_excel(INPUT_DIR + '/GHG_targets.xlsx', index_col=0)

# 选择 2010 到 2050 年的数据
df_filtered = df.loc[2010:2050,
              ['1.5C (67%) excl. avoided emis', '1.5C (50%) excl. avoided emis', '1.8C (67%) excl. avoided emis']]

# 将所有数据单位转换为 million (除以 1,000,000)
df_filtered = df_filtered / 1e6
df_filtered.columns = ['1.5°C (67%)', '1.5°C (50%)', '1.8°C (67%)']
colors = ['#E74C3C', '#3498DB', '#2ECC71']  # 根据数据列数调整颜色列表
draw_plot_lines(df_filtered, colors, ' ', (-300, 100), 100, "../output/03_GHG_limit.png", font_size=font_size)


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

# Biodiversity
BIODIV_GBF_TARGET_2_DICTs = [
                            {2010: 0,  2030: 0,  2050: 0, 2100: 0},
                            {2010: 0,  2030: 0.3,  2050: 0.3, 2100: 0.3},
                            {2010: 0,  2030: 0.3,  2050: 0.5, 2100: 0.5},
                            ]



df = get_biodiversity_target(INPUT_DIR, BIODIV_GBF_TARGET_2_DICTs ) / 1e6
colors = ['#E74C3C', '#3498DB', '#2ECC71']  # 根据数据列数调整颜色列表
draw_plot_lines(df, colors, ' ', (0, 120), 40, "../output/03_biodiversity_limit.png", font_size=font_size)


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

mapping_df = pd.read_excel('tools/land use colors.xlsx', sheet_name='water')
dd_water_limit_df, legend_colors = process_single_df(dd_water_limit_df, mapping_df)
draw_coloum(dd_water_limit_df, legend_colors, '../output/03_water_limit.png', fontsize=font_size, y_range=(0, 300), y_tick_interval=100, ylabel=' ')

