from tools.data_helper import *
from tools.plot_helper import *
from tools.parameters import *
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import sys
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import math

# 获取到文件的绝对路径，并将其父目录添加到 sys.path
sys.path.append(os.path.abspath('../../../luto'))

# 导入 settings.py
import settings


plt.rcParams['font.family'] = 'Arial'


def plot_line_charts(result_dict, save_name="line_charts.png", font_size=14):
    """
    为 result_dict 中的每个 DataFrame 绘制点线图。
    每个 DataFrame 的每列数据绘制为一条线。

    Parameters:
        result_dict (dict): 包含 DataFrame 的字典。
        save_name (str): 保存的图像文件名。
        font_size (int): 全局字体大小。

    Returns:
        None
    """
    # 设置全局字体大小
    plt.rcParams.update({'font.size': font_size})

    color_mapping = {
        "Target": "red",
        "0%": "#9BD608",
        "30%": "#8A9515",
        "50%": "#1D4B2A"
    }

    # 创建横向拼接的图表
    fig, axes = plt.subplots(1, len(result_dict), figsize=(18, 5.8), sharey=True)  # 横向子图

    for ax, (key, df) in zip(axes, result_dict.items()):
        # 为每列数据绘制一条线
        for column in df.columns:
            color = color_mapping.get(column, "black")  # 默认黑色，如果列名不在字典中
            ax.plot(df.index, df[column], marker='o', label=column, color=color)

        # 设置 X 轴范围和刻度
        ax.set_xlim(2009.5, 2050.5)
        ax.set_xticks(range(2010, 2051, 20))  # 每隔 20 年一个刻度

        # 设置 Y 轴范围和刻度
        y_min = int(df.min().min() // 10 * 10)  # 找到最小值，并向下取整为 10 的倍数
        y_max = math.ceil(df.max().max() / 10) * 10  # 找到最大值，并向上取整为最近的 10 的倍数
        ax.set_ylim(y_min, y_max)
        ax.set_yticks(range(y_min, y_max + 1, int((y_max - y_min) // 4)))  # 分成 4 段

        # 设置刻度方向
        # 设置 X 轴和 Y 轴刻度
        ax.tick_params(axis='x', which='both', direction='in', pad=10)  # 下移 X 轴刻度标签
        ax.tick_params(axis='y', which='both', direction='in')  # Y 轴刻度标签向内

        # 设置标题
        ax.set_xlabel("Year")

    # 调整布局
    plt.tight_layout()

    # 保存图表为透明背景
    plt.savefig(save_name, dpi=300, transparent=True)
    plt.show()

    # 创建单独的图例
    fig_legend, ax_legend = plt.subplots(figsize=(6, 1))
    handles, labels = [], []
    for column, color in color_mapping.items():
        handles.append(plt.Line2D([0], [0], color=color, marker='o', linestyle='-', label=column))
        labels.append(column)
    ax_legend.legend(
        handles=handles,
        labels=labels,
        loc="center",
        ncol=len(color_mapping),
        frameon=False,
        fontsize=math.ceil(font_size / 3)  # 使用正确的参数名 fontsize
    )
    ax_legend.axis("off")
    fig_legend.tight_layout()
    # 保存图例为透明背景
    fig_legend.savefig(save_name[:-4] + "_legend.png", dpi=300, transparent=True)


def make_data(data_dict, figure_keys, GHG_TARGETS):
    """
    从字典中提取键名包含 figure_keys 中任一值的 DataFrame，按键名重命名列，并将结果存入新的字典中。
    每个结果 DataFrame 与 additional_df 按索引拼接。

    Parameters:
        data_dict (dict): 包含 DataFrame 的字典。
        figure_keys (list): 用于筛选键名的关键字列表。
        additional_df (pd.DataFrame): 需要与合并结果按索引拼接的 DataFrame。

    Returns:
        dict: 包含合并结果 DataFrame 的字典，键名为 figure_keys 的值。
    """
    result_dict = {}

    for figure_key in figure_keys:
        GHG_LIMITS_FIELD = format_ghg_string(figure_key)
        ghg_targets_df = GHG_TARGETS[GHG_LIMITS_FIELD] / 1e6
        ghg_targets_df.index.name = "Year"
        additional_df = ghg_targets_df.to_frame(name="Target")
        # 提取键名包含当前 figure_key 的 DataFrame
        filtered_dfs = []
        for key, df in data_dict.items():
            if figure_key in key:
                # 提取 BIO_ 后的数字并转换为百分比
                column_name = f"{int(key.split('_BIO_')[1].split('_')[0]) * 10}%"
                # 重命名列并添加到结果列表
                filtered_dfs.append(df.rename(columns=lambda col: column_name))

        # 合并所有提取的 DataFrame
        if filtered_dfs:
            merged_df = pd.concat(filtered_dfs, axis=1)  # 按列合并，保持索引
            # 按照列名的大小顺序重新排序
            merged_df = merged_df[sorted(merged_df.columns, key=lambda x: float(x.strip('%')))]

            # 与 additional_df 按索引拼接
            final_df = pd.concat([merged_df, additional_df.reindex(merged_df.index)], axis=1)
            result_dict[figure_key] = final_df
        else:
            print(f"Warning: No keys found containing the keyword '{figure_key}'")

    return result_dict


def format_ghg_string(input_string):
    # 提取 GHG_ 后的部分，例如 "1_5C_67"
    match = re.search(r"GHG_(\d+)_(\d+C)_(\d+)", input_string)
    if match:
        # 提取数字部分并格式化
        ghg_value = match.group(1)  # "1"
        temp_value = match.group(2)  # "5C"
        percentage = match.group(3)  # "67"

        # 转换为目标格式
        result = f"{ghg_value}.{temp_value} ({percentage}%) excl. avoided emis"
        return result
    else:
        raise ValueError("Input string does not match the expected format")

plt.rcParams['font.family'] = 'Arial'

ag_dict = get_dict_data(input_files, 'GHG_emissions_separate_agricultural_landuse','Value (t CO2e)',  'Land-use')
ag_group_dict = aggregate_by_mapping(ag_dict, 'tools/land use group.xlsx', 'desc', 'ag_group')
am_dict = get_dict_data(input_files, 'GHG_emissions_separate_agricultural_management', 'Value (t CO2e)', 'Agricultural Management Type')
# off_land_dict = get_value_sum(input_files, 'GHG_emissions_offland_commodity',  '*', 'Total GHG Emissions (tCO2e)')
non_ag_dict = get_dict_data(input_files, 'GHG_emissions_separate_no_ag_reduction', 'Value (t CO2e)', 'Land-use')
transition_dict = get_dict_data(input_files, 'GHG_emissions_separate_transition_penalty', 'Value (t CO2e)','Type')

ghg_dict = concatenate_dicts_by_year([ag_dict, am_dict, non_ag_dict, transition_dict])
ghg_group_dict = aggregate_by_mapping(ghg_dict, 'tools/land use group.xlsx', 'desc', 'lu_group', summary_col_name='Net emissions')
point_dict = {key: df[['Net emissions']] for key, df in ghg_group_dict.items()}

figure_keys = ['GHG_1_8C_67','GHG_1_5C_50','GHG_1_5C_67']
figure_key = figure_keys[0]

GHG_TARGETS = pd.read_excel(
                os.path.join(INPUT_DIR, "GHG_targets.xlsx"), sheet_name="Data", index_col="YEAR"
            )
result_dict = make_data(point_dict, figure_keys, GHG_TARGETS)
plot_line_charts(result_dict, save_name="../output/04_GHG_target.png", font_size=25)
