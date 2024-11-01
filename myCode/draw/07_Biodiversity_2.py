'''
Targets and results
'''
import numpy as np
import sys
import os
import pandas as pd
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from helper import *

plt.rcParams['font.family'] = 'Arial'

# 获取到文件的绝对路径，并将其父目录添加到 sys.path
sys.path.append(os.path.abspath('../../luto'))

# 导入 settings.py
import settings


# 定义数据读取函数，避免重复代码
def load_data(input_dir):
    biodiv_priorities = pd.read_hdf(os.path.join(input_dir, 'biodiv_priorities.h5'))
    lumap_no_resfactor = pd.read_hdf(os.path.join(input_dir, "lumap.h5")).to_numpy()
    real_area_no_resfactor = pd.read_hdf(os.path.join(input_dir, "real_area.h5")).to_numpy()
    savburn_df = pd.read_hdf(os.path.join(input_dir, 'cell_savanna_burning.h5'))
    return biodiv_priorities, lumap_no_resfactor, real_area_no_resfactor, savburn_df


# 预处理数据，统一处理所需的数组
def preprocess_data(biodiv_priorities, settings):
    # 提取生物多样性得分和自然区域连通性
    biodiv_score_raw = biodiv_priorities['BIODIV_PRIORITY_SSP' + str(settings.SSP)].to_numpy(dtype=np.float32)
    dist_to_natural = biodiv_priorities['NATURAL_AREA_CONNECTIVITY'].to_numpy(dtype=np.float32)

    # 连通性得分计算
    conn_score = 1 - np.interp(dist_to_natural, (dist_to_natural.min(), dist_to_natural.max()), (0, 1)).astype('float32')

    # 加权生物多样性得分
    biodiv_score_weighted = biodiv_score_raw * conn_score
    return biodiv_score_raw, biodiv_score_weighted


# 构建 land-use 索引字典，并处理需要的列表
def build_land_use_dicts(ag_landuses, non_ag_landuses):
    # 农业和非农业土地用途的索引字典
    aglu2desc = {i: lu for i, lu in enumerate(ag_landuses)}
    desc2aglu = {value: key for key, value in aglu2desc.items()}
    aglu2desc[-1] = 'Non-agricultural land'

    nonaglu2desc = dict(zip(range(settings.NON_AGRICULTURAL_LU_BASE_CODE,
                                  settings.NON_AGRICULTURAL_LU_BASE_CODE + len(non_ag_landuses)),
                            non_ag_landuses))

    return aglu2desc, desc2aglu, nonaglu2desc


# 计算退化土地生物多样性损失
def calculate_biodiv_value_degraded_ag_land(
        biodiv_score_raw,
        biodiv_value_pristine_natural_land,
        biodiv_value_current_natural_land,
        lumap_no_resfactor,
        real_area_no_resfactor,
        lu_modified_land
):
    # 自然土地损失
    natural_land_loss = biodiv_value_pristine_natural_land - biodiv_value_current_natural_land

    # 其它农业土地利用造成的损失:这些土地利用损失了全部的生物多样性
    modified_land_loss = np.isin(lumap_no_resfactor, lu_modified_land) * biodiv_score_raw * real_area_no_resfactor

    # 总生物多样性损失
    biodiv_value_degraded_ag_land = natural_land_loss + modified_land_loss

    return biodiv_value_degraded_ag_land, biodiv_value_current_natural_land


# 主函数
def calculate_biodiv_gbf_target_2(input_dir):
    # 加载所有数据
    biodiv_priorities, lumap_no_resfactor, real_area_no_resfactor, savburn_df = load_data(input_dir)

    # 预处理数据
    biodiv_score_raw, biodiv_score_weighted = preprocess_data(biodiv_priorities, settings)

    # 读取农业和非农业土地用途
    ag_landuses = pd.read_csv(os.path.join(input_dir, 'ag_landuses.csv'), header=None)[0].to_list()
    non_ag_landuses = list(settings.NON_AG_LAND_USES.keys())

    # 构建土地用途字典
    aglu2desc, desc2aglu, nonaglu2desc = build_land_use_dicts(ag_landuses, non_ag_landuses)

    # Savanna burning eligibility
    savburn_eligible = savburn_df.ELIGIBLE_AREA.to_numpy()

    # 加权后的 LDS 生物多样性得分
    biodiv_score_weighted_lds_burning = biodiv_score_raw * np.where(savburn_eligible, settings.LDS_BIODIVERSITY_VALUE, 1)

    # 当前自然土地的生物多样性值计算
    biodiv_value_current_natural_land = (
                                                np.isin(lumap_no_resfactor,
                                                        [2, 6, 15, 23]) * biodiv_score_weighted_lds_burning -
                                                np.isin(lumap_no_resfactor, [2, 6,
                                                                             15]) * biodiv_score_weighted * settings.BIODIV_LIVESTOCK_IMPACT
                                        ) * real_area_no_resfactor

    # 自然土地生物多样性原始值
    biodiv_value_pristine_natural_land = np.isin(lumap_no_resfactor,
                                                 [2, 6, 15, 23]) * biodiv_score_weighted * real_area_no_resfactor

    lu_modified_land = [desc2aglu[lu] for lu in ag_landuses if
                        desc2aglu[lu] not in [desc2aglu["Beef - natural land"], desc2aglu["Dairy - natural land"],
                                              desc2aglu["Sheep - natural land"],
                                              desc2aglu["Unallocated - natural land"]]]

    # 计算退化农业土地的生物多样性值和当前自然土地的生物多样性值
    biodiv_value_degraded_ag_land, biodiv_value_current_natural_land = calculate_biodiv_value_degraded_ag_land(
        biodiv_score_raw,
        biodiv_value_pristine_natural_land,
        biodiv_value_current_natural_land,
        lumap_no_resfactor,
        real_area_no_resfactor,
        lu_modified_land
    )

    # 创建一个字典用于存储2010到2100年GBF目标2的生物多样性比例数据
    biodiv_gbf_target_2_proportions_2010_2100 = {}

    # 创建线性插值函数，并使用 settings 中的目标字典进行插值
    f = interp1d(
        list(settings.BIODIV_GBF_TARGET_2_DICT.keys()),
        list(settings.BIODIV_GBF_TARGET_2_DICT.values()),
        kind="linear",
        fill_value="extrapolate",
    )

    # 使用插值函数填充每年的生物多样性目标比例
    for yr in range(2010, 2101):
        biodiv_gbf_target_2_proportions_2010_2100[yr] = f(yr).item()

    # 计算当前自然土地的生物多样性总得分和退化土地的总得分
    biodiv_value_current_total = biodiv_value_current_natural_land.sum()
    biodiv_value_degraded_total = biodiv_value_degraded_ag_land.sum()

    # 创建字典存储每年GBF目标2的生物多样性总得分
    biodiv_gbf_target_2 = {}

    # 计算每年的生物多样性总目标，并将结果存储在字典中
    for yr in range(2010, 2101):
        # 当前的生物多样性总得分加上需要恢复的退化土地生物多样性得分
        biodiv_gbf_target_2[yr] = biodiv_value_current_total + (
            biodiv_value_degraded_total * biodiv_gbf_target_2_proportions_2010_2100[yr]
        )

    return biodiv_gbf_target_2


def plot_line_chart(df, ax, font_size=20):
    """
    在指定轴上绘制点线图。

    参数：
    df: 第一个表的 DataFrame，包含要绘制的数据
    ax: 用于绘制的轴对象
    font_size: 设置字体大小
    """
    ax.plot(df.index, df, marker='o', linestyle='-', color='green', label='Biodiversity target')
    ax.set_xlabel('Year', fontsize=font_size)
    ax.set_ylabel('Quality-weighted Area (Million ha)', fontsize=font_size)
    ax.set_xticks(range(2010, 2051))
    ax.set_xticklabels(range(2010, 2051), rotation=90, fontsize=font_size)
    ax.set_xlim(2010, 2050)
    ax.set_yticks(range(0, 101, 25))
    ax.set_ylim(0, 100)
    ax.set_yticklabels(range(0, 101, 25), fontsize=font_size)
    ax.tick_params(axis='both', direction='in')
    ax.legend(loc='lower left', fontsize=font_size, frameon=False)


def plot_stacked_bar_chart(tables_dict, input_file, ax, font_size=14):
    """
    在指定轴上绘制累积柱状图，不显示图例。
    """
    colors = ['#FFB715', '#79BE92', '#008EAB']
    df_stacked = tables_dict[input_file].astype(float)
    bottom = np.zeros(len(df_stacked.index))
    for i, column in enumerate(df_stacked.columns):
        ax.bar(df_stacked.index, df_stacked[column], bottom=bottom, color=colors[i], alpha=0.8, width=0.7)
        bottom += df_stacked[column].values
    ax.set_xlabel('Year', fontsize=font_size)
    ax.set_ylabel('Quality-weighted Area (Million ha)', fontsize=font_size)
    ax.set_xticks(range(2010, 2051))
    ax.set_xlim(2010, 2050)
    ax.set_xticklabels(range(2010, 2051), rotation=90, fontsize=font_size, ha='center')
    ax.set_yticks(range(0, 101, 25))
    ax.set_ylim(0, 100)
    ax.set_yticklabels(range(0, 101, 25), fontsize=font_size)
    ax.tick_params(axis='both', direction='in')


def save_legend_for_bars(df_stacked, colors, output_file='legend.png', font_size=14):
    """
    生成针对柱状图的图例并单独保存为 PNG 图像。
    """
    fig_legend = plt.figure(figsize=(8, 2))
    handles = [plt.Rectangle((0, 0), 1, 1, color=color, label=column) for column, color in
               zip(df_stacked.columns, colors)]
    fig_legend.legend(handles=handles, loc='center', frameon=False, ncol=3, fontsize=font_size)
    fig_legend.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close(fig_legend)


def save_combined_image_with_legend(tables_dict, input_files, font_size=25,
                                    output_file='combined_biodiversity_image.png'):
    """
    生成点线图和三个累积柱状图，合并为一张图片，保存主图像并将图例垂直拼接到图片下方。

    参数：
    tables_dict: 包含多个 DataFrame 的字典
    input_files: 要绘制的表名列表
    font_size: 设置字体大小
    output_file: 保存最终拼接图像的路径
    """
    # 创建绘图
    fig, axs = plt.subplots(2, 2, figsize=(24, 16))
    labels = ['(a)', '(b)', '(c)', '(d)']

    # 绘制点线图
    plot_line_chart(tables_dict['Target'], axs[0, 0], font_size)
    axs[0, 0].text(0.01, 0.99, labels[0], transform=axs[0, 0].transAxes, fontsize=font_size + 5,
                   va='top')

    # 分别绘制三个累积柱状图
    for i, input_file in enumerate(input_files):
        plot_stacked_bar_chart(tables_dict, input_file, axs[(i + 1) // 2, (i + 1) % 2], font_size)
        axs[(i + 1) // 2, (i + 1) % 2].text(0.01, 0.99, labels[i + 1],
                                            transform=axs[(i + 1) // 2, (i + 1) % 2].transAxes,
                                            fontsize=font_size + 5, va='top')

    # 调整布局并保存主图像（不再需要单独输出 result.png）
    plt.tight_layout()
    fig.savefig('temp_main_image.png', bbox_inches='tight')
    plt.close()

    # 保存图例
    save_legend_for_bars(tables_dict[input_files[0]], ['#FFB715', '#79BE92', '#008EAB'],
                         output_file='biodiversity_legend.png', font_size=font_size)

    # 拼接主图像和图例
    add_legend_to_image('temp_main_image.png', 'biodiversity_legend.png', output_file=output_file)


def add_legend_to_image(main_image_file, legend_image_file, output_file='combined_biodiversity_image.png'):
    """
    将图例和主图像垂直拼接，按比例调整图例的宽度与主图像一致。
    """
    main_image = Image.open(main_image_file)
    legend_image = Image.open(legend_image_file)
    aspect_ratio = legend_image.height / legend_image.width
    new_legend_height = int(main_image.width * aspect_ratio)
    legend_image = legend_image.resize((main_image.width, new_legend_height), Image.LANCZOS)
    total_height = main_image.height + legend_image.height
    combined_image = Image.new('RGB', (main_image.width, total_height))
    combined_image.paste(main_image, (0, 0))
    combined_image.paste(legend_image, (0, main_image.height))
    combined_image.save(output_file)
    print(f"Image saved as {output_file}")

# 调用主函数并输出结果
input_dir = '../../input'
input_files = ['ON_MAXPROFIT_GHG_15C_67_R10', 'ON_MAXPROFIT_GHG_15C_50_R10', 'ON_MAXPROFIT_GHG_18C_67_R10']
csv_name = 'biodiversity_separate'
filter_column_name = 'Landuse type'
value_column_name = 'Biodiversity score'

biodiv_gbf_target_2 = get_unique_value([input_files[0]], 'biodiversity', 0, 'Score') # input_files, csv_name, row_name, column_name
df = biodiv_gbf_target_2[input_files[0]]
df.columns = ['Target']
df = df.rename_axis('Year')
tables_dict = get_value_sum(input_files, csv_name, filter_column_name, value_column_name)
tables_dict = {"Target": df, **tables_dict}

save_combined_image_with_legend(tables_dict, input_files, font_size=25, output_file='07_biodiversity.png')

