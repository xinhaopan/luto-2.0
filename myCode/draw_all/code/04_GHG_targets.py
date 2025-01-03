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
point_dict = {key: df[['Net emissions']] for key, df in ghg_group_dict.items()}

figure_keys = ['GHG_1_8C_67','GHG_1_5C_50','GHG_1_5C_67']
figure_key = figure_keys[0]

GHG_TARGETS = pd.read_excel(
                os.path.join(INPUT_DIR, "GHG_targets.xlsx"), sheet_name="Data", index_col="YEAR"
            )

result_dict = make_data(point_dict, figure_keys, GHG_TARGETS)

result_dict,legend_colors = get_colors(result_dict, 'tools/land use colors.xlsx', sheet_name='biodiversity')
output_png = '../output/04_ghg_targets.png'
y_range, y_ticks = calculate_y_axis_range(result_dict)
plot_Combination_figures(result_dict, output_png, figure_keys, plot_line_chart, legend_colors,
                             point_data=None, n_rows=1, n_cols=3, font_size=font_size, x_range=(2010, 2050), y_range=y_range,
                             x_ticks=20, y_ticks=y_ticks,
                             legend_position=(0.5, -0.25), show_legend='last', legend_n_rows=3)