# --- 标准库 ---
import os
import re
import math

# --- 第三方库 ---import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from plotnine import *
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from datetime import datetime

import cairosvg
from lxml import etree

# --- 本地模块 ---
from .config import *
from .tools import get_path, get_year


plt.rcParams['font.sans-serif'] = ['Arial']

def get_y_axis_ticks(min_value, max_value, desired_ticks=5):
    """
    生成Y轴刻度，根据数据范围智能调整刻度间隔和范围。
    优化版本，提高运行速度，并处理0-100特殊情况。
    """
    # 1. 快速处理特殊情况
    if min_value > 0 and max_value > 0:
        min_value = 0
    elif min_value < 0 and max_value < 0:
        max_value = 0

    range_value = max_value - min_value
    if range_value <= 0:
        return 0, 1, np.array([0, 0.5, 1])  # 使用numpy数组替代列表

    # 2. 一次性计算间隔
    ideal_interval = range_value / (desired_ticks - 1)
    # 根据理想间隔选择“nice”间隔
    e = math.floor(math.log10(ideal_interval))  # 计算数量级
    base = 10 ** e
    normalized_interval = ideal_interval / base

    # 定义“nice”间隔选项
    nice_intervals = [1, 2, 5, 10]
    # 选择最接近理想间隔的“nice”值
    interval = min(nice_intervals, key=lambda x: abs(x - normalized_interval)) * base

    # 3. 整合计算，减少中间变量
    min_tick = math.floor(min_value / interval) * interval
    max_tick = math.ceil(max_value / interval) * interval

    # 4. 使用numpy直接生成数组，避免Python列表操作
    tick_count = int((max_tick - min_tick) / interval) + 1
    ticks = np.linspace(min_tick, max_tick, tick_count)

    # 5. 高效处理0的插入
    if min_value < 0 < max_value and 0 not in ticks:
        # numpy的searchsorted比Python的排序更高效
        zero_idx = np.searchsorted(ticks, 0)
        ticks = np.insert(ticks, zero_idx, 0)

    # 6. 预计算共享变量，避免重复计算
    close_threshold = 0.3 * interval

    # 7. 简化逻辑，减少条件分支
    max_v = max_tick
    min_v = min_tick

    # 处理刻度和范围调整（仅当有足够刻度且最值不是0时）
    if len(ticks) >= 2:
        # 处理最大值
        if ticks[-1] != 0 and (max_value - ticks[-2]) < close_threshold and (ticks[-1] - max_value) > close_threshold:
            ticks = ticks[:-1]  # 移除最后一个刻度
            max_v = max_value + 0.1 * interval

        # 处理最小值
        if ticks[0] != 0 and (ticks[1] - min_value) < close_threshold and (min_value - ticks[0]) > close_threshold:
            ticks = ticks[1:]  # 移除第一个刻度
            min_v = min_value - 0.1 * interval
        elif abs(min_value) < interval:
            min_v = math.floor(min_value)

    # 8. 特殊情况：当刻度范围是0到100时，使用规则的25间隔
    if (abs(ticks[0]) < 1e-10 and abs(ticks[-1] - 100) < 1e-10) or (min_tick == 0 and max_tick == 100):
        ticks = np.array([0, 25, 50, 75, 100])
        min_v = 0
        max_v = 100

    return min_v, max_v, ticks.tolist()  # 根据需要转回列表

def draw_histogram(path_name,year=2050, mask_use=True):
    print("Start to draw histogram.")
    path_name = get_path(input_file)
    # 加载 2050 年的 .npy 文件
    payment_arr_2050 = np.load(os.path.join(path_name, "data_for_carbon_price", f"cost_{year}.npy"))
    ghg_arr_2050 = np.load(os.path.join(path_name, "data_for_carbon_price", f"ghg_{year}.npy"))
    cp_arr_2050 = np.load(os.path.join(path_name, "data_for_carbon_price", f"cp_{year}.npy"))

    # 使用 mask 过滤掉 ghg_arr 小于 1 的值
    output_name =  f"histogram_{year}_nomask.png"
    if mask_use:
        mask = ghg_arr_2050 >= 1
        filtered_cp_arr_with_mask = cp_arr_2050[mask]
        filtered_ghg_arr_with_mask = ghg_arr_2050[mask]
        filtered_payment_arr_with_mask = payment_arr_2050[mask]
        output_name = f"histogram_{year}_mask.png"

    # 绘制直方图
    plt.figure(figsize=(18, 6))

    plt.subplot(1, 3, 1)
    plt.hist(filtered_cp_arr_with_mask, bins=50)
    plt.yscale('log')
    plt.title(f'Carbon Price for Uniform Payment ($/tCO2e) - {year}')
    plt.xlabel('Price ($/tCO2e)')
    plt.ylabel('Frequency')

    plt.subplot(1, 3, 2)
    plt.hist(filtered_ghg_arr_with_mask, bins=50)
    plt.yscale('log')
    plt.title(f'GHG with mask - {year}')
    plt.xlabel('GHG')
    plt.ylabel('Frequency')

    plt.subplot(1, 3, 3)
    plt.hist(filtered_payment_arr_with_mask, bins=50)
    plt.yscale('log')
    plt.title(f'Cost with mask - {year}')
    plt.xlabel('Cost')
    plt.ylabel('Frequency')

    plt.tight_layout()
    plt.savefig(os.path.join(r"output\Carbon_Price", path_name.split("/")[1] + "_" + f"histogram_{year}.png"))
    plt.show()


def summarize_from_csv(input_file):
    print("Start to summarize from csv.")
    path_name = f"../output/02_{input_file}_price.xlsx"
    # 初始化路径和列名
    categories = {
        'Cost_ag($)': ['cost_agricultural_commodity'],
        'Cost_am($)': ['cost_agricultural_management'],
        'Cost_non_ag($)': ['cost_non_ag'],
        'Cost_transition_ag2ag($)': ['cost_transition_ag2ag'],
        'Cost_transition_ag2non_ag($)': ['cost_transition_ag2non_ag'],
        'Cost_transition_non_ag2ag($)': ['cost_transition_non_ag2_ag'],
        'Revenue_ag($)': ['revenue_agricultural_commodity'],
        'Revenue_am($)': ['revenue_agricultural_management'],
        'Revenue_non_ag($)': ['revenue_non_ag'],
        'GHG_ag(tCO2e)': ['GHG_emissions_separate_agricultural_landuse'],
        'GHG_am(tCO2e)': ['GHG_emissions_separate_agricultural_management'],
        'GHG_non_ag(tCO2e)': ['GHG_emissions_separate_no_ag_reduction'],
        'GHG_off_land(tCO2e)': ['GHG_emissions_offland_commodity'],
        'GHG_transition(tCO2e)': ['GHG_emissions_separate_transition_penalty']
    }

    # 定义列名关键字的映射字典
    column_keywords = {
        'cost_transition_ag2ag': 'Cost ($)',
        'cost_transition_ag2non_ag': 'Cost ($)',
        'cost_transition_non_ag2_ag': 'Cost ($)',
        'GHG_emissions_offland_commodity': 'Total GHG Emissions',
        'default': 'Value'
    }

    # 创建DataFrame并保存到CSV文件
    # 处理所有年份的数据
    folder_name = next((f for f in os.listdir(path_name) if
                        os.path.isdir(os.path.join(path_name, f)) and f.startswith("begin_end_compare_")), None)
    years = [int(year) for year in re.findall(r'\d{4}', folder_name)]
    results = [process_year_data(year, path_name, categories, column_keywords) for year in range(years[0], years[1] + 1)]
    df_result = pd.DataFrame(results)

    df_result.to_excel("output/Carbon_Price/" + path_name.split("/")[1] + "_" + 'calculated_carbon_price_fromcsv.xlsx', index=False)

    # 用于计算的字典，包含四种计算情况
    calculations = {
        'Carbon_price1($/tCO2e)': {
            'cost_columns': ['Cost_ag($)', 'Cost_am($)', 'Cost_non_ag($)', 'Cost_transition_ag2ag($)',
                             'Cost_transition_ag2non_ag($)', 'Cost_transition_non_ag2ag($)'],
            'ghg_columns': ['GHG_ag(tCO2e)', 'GHG_am(tCO2e)', 'GHG_non_ag(tCO2e)', 'GHG_off_land(tCO2e)',
                            'GHG_transition(tCO2e)'],
            'revenue_columns': []
        },
        'Carbon_price2($/tCO2e)': {
            'cost_columns': ['Cost_ag($)', 'Cost_am($)', 'Cost_non_ag($)', 'Cost_transition_ag2ag($)',
                             'Cost_transition_ag2non_ag($)', 'Cost_transition_non_ag2ag($)'],
            'ghg_columns': ['GHG_ag(tCO2e)', 'GHG_am(tCO2e)', 'GHG_non_ag(tCO2e)', 'GHG_off_land(tCO2e)',
                            'GHG_transition(tCO2e)'],
            'revenue_columns': ['Revenue_ag($)', 'Revenue_am($)']
        },
        'Carbon_price3($/tCO2e)': {
            'cost_columns': ['Cost_am($)', 'Cost_non_ag($)', 'Cost_transition_ag2non_ag($)',
                             'Cost_transition_non_ag2ag($)'],
            'ghg_columns': ['GHG_am(tCO2e)', 'GHG_non_ag(tCO2e)', 'GHG_transition(tCO2e)'],
            'revenue_columns': []
        },
        'Carbon_price4($/tCO2e)': {
            'cost_columns': ['Cost_am($)', 'Cost_non_ag($)', 'Cost_transition_ag2non_ag($)',
                             'Cost_transition_non_ag2ag($)'],
            'ghg_columns': ['GHG_am(tCO2e)', 'GHG_non_ag(tCO2e)', 'GHG_transition(tCO2e)'],
            'revenue_columns': ['Revenue_am($)']
        }
    }

    with pd.ExcelWriter("output/Carbon_Price/" + path_name.split("/")[1] + "_" + 'calculated_carbon_price_fromcsv.xlsx', engine='openpyxl', mode='a',
                        if_sheet_exists='new') as writer:
        start_sheet = len(writer.sheets) + 1  # 获取现有工作表数量，加1以从下一个空白工作表开始
        for i, (key, value) in enumerate(calculations.items(), start=start_sheet):
            # 计算成本总和，温室气体总和和收入总和
            df_result1 = df_result.copy()
            df_result1['Cost_all(M$)'] = df_result1[value['cost_columns']].sum(axis=1) / 1e6
            df_result1['GHG_all(MtCO2e)'] = - df_result1[value['ghg_columns']].sum(axis=1) / 1e6

            if value['revenue_columns']:
                df_result1['Revenue_all(M$)'] = df_result1[value['revenue_columns']].sum(axis=1) / 1e6
                df_result1[key] = (df_result1['Cost_all(M$)'] - df_result1['Revenue_all(M$)']) / df_result1[
                    'GHG_all(MtCO2e)']
            else:
                df_result1[key] = df_result1['Cost_all(M$)'] / df_result1['GHG_all(MtCO2e)']

            # 选择要写入的列
            columns_to_write = ['Year', 'Cost_all(M$)', 'GHG_all(MtCO2e)', key]
            if 'Revenue_all(M$)' in df_result1.columns:
                columns_to_write.insert(2, 'Revenue_all(M$)')

            # 写入Excel的对应工作表
            df_result1.to_excel(writer, sheet_name=key[:-9], index=False, columns=columns_to_write)

def sanitize_filename(filename):
    return filename.replace('/', '_').replace('\\', '_').replace(':', '_')


def draw_figure(input_file):
    # 读取表格数据
    file_path = f'../output/03_{input_file}_shadow_price.xlsx'  # 替换为你的文件路径
    df = pd.read_excel(file_path)

    # 设置 Year 列为索引
    df.set_index('Year', inplace=True)

    # 遍历每一列并绘制点线图
    for column in df.columns:
        # 计算均值和标准差
        mean = df[column].mean()
        std = df[column].std()

        # 将超出均值±3倍标准差的值替换为NaN
        df[column] = df[column].where((df[column] <= mean + 3*std) & (df[column] >= mean - 3*std), np.nan)

        plt.figure()
        plt.plot(df.index, df[column], marker='o', linestyle='-', label=column, color='b')

        # 检查是否有NaN值
        has_nan = df[column].isna().any()

        if has_nan:
            # 绘制带断点的图
            plt.plot(df.index, df[column].interpolate(), color='r')  # 插值显示断点

        plt.xlabel('Year')
        plt.ylabel(column)
        plt.title(f'{input_file} {column}')
        plt.legend()
        plt.grid(True)

        sanitized_column = sanitize_filename(column)
        plt.savefig(f'../Figure/{input_file}_{sanitized_column}.png')  # 保存图像为文件
        plt.close()  # 关闭当前图像，避免内存问题
        # plt.show()
# def draw_figure(input_file):
#     # 读取表格数据
#     file_path = f'../output/03_{input_file}_shadow_price.xlsx'  # 替换为你的文件路径
#     df = pd.read_excel(file_path)
# 
#     # 设置 Year 列为索引
#     df.set_index('Year', inplace=True)
# 
#     # 遍历每一列并绘制点线图
#     for column in df.columns:
#         # 计算均值和标准差
#         mean = df[column].mean()
#         std = df[column].std()
# 
#         # 将超出均值±3倍标准差的值标记为异常值
#         is_outlier = (df[column] > mean + 3*std) | (df[column] < mean - 3*std)
# 
#         # 获取非异常值的最大最小值
#         non_outlier_max = df[column][~is_outlier].max()
#         non_outlier_min = df[column][~is_outlier].min()
# 
#         # 设置断点范围（略低于断点的值，略高于去掉断点后的最大值）
#         break_start = mean + 3 * std - 0.1 * std
#         break_end = non_outlier_max + 0.1 * std
# 
#         # 创建带有断点的双轴图
#         fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10, 6), gridspec_kw={'height_ratios': [3, 1]})
# 
#         # 绘制正常值
#         ax1.plot(df.index, df[column], marker='o', linestyle='-', label=column, color='b')
#         ax2.plot(df.index, df[column], marker='o', linestyle='-', label=column, color='b')
# 
#         # 隐藏异常值
#         ax1.plot(df.index[is_outlier], df[column][is_outlier], 'o', color='white')
#         ax2.plot(df.index[~is_outlier], df[column][~is_outlier], 'o', color='white')
# 
#         # 设置轴范围
#         ax1.set_ylim(break_start, df[column].max() + std)
#         ax2.set_ylim(df[column].min() - std, break_end)
# 
#         # 添加断点标记
#         d = .015  # 断点大小
#         kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False)
#         ax1.plot((-d, +d), (-d, +d), **kwargs)        # top-left diagonal
#         ax1.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal
# 
#         kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
#         ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
#         ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal
# 
#         plt.xlabel('Year')
#         plt.ylabel(column)
#         plt.title(f'{input_file} {column}')
#         plt.legend()
#         plt.grid(True)
# 
#         sanitized_column = sanitize_filename(column)
#         plt.savefig(f'../Figure/{input_file}_{sanitized_column}.png')  # 保存图像为文件
#         plt.close()  # 关闭当前图像，避免内存问题
# 
# 
# 
def plot_cost_vs_price(input_file: str):
    # 构造文件路径并读取数据
    file_path = f"../output/02_{input_file}_price.xlsx"
    df = pd.read_excel(file_path)

    # 删除 2010 年并重置年份
    df = df[df["Year"] >= start_year]
    # df["Year"] = list(range(start_year, start_year + len(df)))

    # 原始列名与对应图例名
    cost_columns = {
        'Opportunity cost(M$)': 'Opportunity cost($/tCOe2)',
        'Transition cost(M$)': 'Transition cost($/tCOe2)',
        'AM net cost(M$)': 'AM net cost($/tCOe2)',
        'Non_AG net cost(M$)': 'Non_AG net cost($/tCOe2)'
    }
    ghg_column = 'GHG Abatement(MtCOe2)'
    price_column = 'carbon price($/tCOe2)'

    # 自定义颜色映射
    color_map = {
        'Opportunity cost($/tCOe2)': '#1F77B4',
        'Transition cost($/tCOe2)': '#FF7F0E',
        'AM net cost($/tCOe2)': '#2CA02C',
        'Non_AG net cost($/tCOe2)': '#FFFF00',
    }

    # 计算单位减排成本列
    stacked_labels = []
    for raw_col, new_col in cost_columns.items():
        df[new_col] = df[raw_col] / df[ghg_column]
        stacked_labels.append(new_col)

    # 设置横轴
    x = df["Year"].values
    x_labels = df["Year"].tolist()

    # 提取堆叠数据
    stacked_values = np.array([df[label].values for label in stacked_labels])

    # 创建图形
    fig, ax = plt.subplots(figsize=(12, 6))
    bottom = np.zeros(len(df))
    positive_bottom = np.zeros(len(df))
    negative_bottom = np.zeros(len(df))

    # 绘制堆叠柱状图，支持正负值
    for i, label in enumerate(stacked_labels):
        values = stacked_values[i]
        pos_vals = np.where(values > 0, values, 0)
        neg_vals = np.where(values < 0, values, 0)

        ax.bar(x, pos_vals, bottom=positive_bottom, label=label, color=color_map[label])
        ax.bar(x, neg_vals, bottom=negative_bottom, color=color_map[label])

        positive_bottom += pos_vals
        negative_bottom += neg_vals

    # 画碳价红色点线图
    ax.plot(x, df[price_column].values, color='red', marker='D',
            linewidth=2, markersize=6, label='Carbon Price', zorder=5)

    # 坐标轴 & 图例
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=45)
    ax.set_xlabel("Year")
    ax.set_ylabel("Price ($/tCO2e)")
    ax.set_title(f"GHG: {input_file.split('_')[3]} & BIO {input_file.split('_')[4]}")
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
              ncol=5, fancybox=True, shadow=False)

    ax.grid(True)

    plt.tight_layout()
    plt.savefig(f"../output/{input_file}_price.png")
    plt.show()


def plot_absolute_cost_stack(input_file: str):
    # 构造文件路径并读取数据
    file_path = f"../output/02_{input_file}_price.xlsx"
    df = pd.read_excel(file_path)

    # 删除 2010 年并重置年份
    df = df[df["Year"] >= start_year]
    # df["Year"] = list(range(start_year, start_year + len(df)))

    # 原始列名与图例名
    cost_columns = {
        'Opportunity cost(M$)': 'Opportunity cost',
        'Transition cost(M$)': 'Transition cost',
        'AM net cost(M$)': 'AM net cost',
        'Non_AG net cost(M$)': 'Non_AG net cost'
    }

    # 自定义颜色映射
    color_map = {
        'Opportunity cost': '#1F77B4',
        'Transition cost': '#FF7F0E',
        'AM net cost': '#2CA02C',
        'Non_AG net cost': '#FFFF00',
    }

    # 设置绘图数据
    x = df["Year"].values
    x_labels = df["Year"].tolist()

    # 创建图形
    fig, ax = plt.subplots(figsize=(12, 6))
    positive_bottom = np.zeros(len(df))
    negative_bottom = np.zeros(len(df))

    total_cost = np.zeros(len(df))

    for raw_col, legend_name in cost_columns.items():
        values = df[raw_col].values
        pos_vals = np.where(values > 0, values, 0)
        neg_vals = np.where(values < 0, values, 0)

        ax.bar(x, pos_vals, bottom=positive_bottom, color=color_map[legend_name], label=legend_name)
        ax.bar(x, neg_vals, bottom=negative_bottom, color=color_map[legend_name])

        positive_bottom += pos_vals
        negative_bottom += neg_vals

        total_cost += values  # 累加到总成本

    # 添加总成本折线图（黑色点线）
    ax.plot(x, total_cost, color='black', marker='o',
            linewidth=2, markersize=5, label='Total Cost')

    # 坐标轴 & 图例
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=45)
    ax.set_xlabel("Year")
    ax.set_ylabel("Cost (M$)")
    ax.set_title(f"GHG: {input_file.split('_')[3]} & BIO {input_file.split('_')[4]}")
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
              ncol=5, fancybox=True, shadow=False)

    ax.grid(True)

    plt.tight_layout()
    plt.savefig(f"../output/{input_file}_cost.png")
    plt.show()


def plot_ghg_stack(input_file: str):
    # 构造文件路径并读取数据
    file_path = f"../output/02_{input_file}_price.xlsx"
    df = pd.read_excel(file_path)

    # 计算差值：
    # 对 'GHG_ag(MtCOe2)' 计算年度差值，生成新列，不取负数
    df["GHG_ag difference(MtCO2e)"] = df["GHG_ag(MtCOe2)"].diff().fillna(0)

    # 对除了 'Year'  的所有列进行 -diff() 操作
    cols_to_diff = df.columns.difference(["Year"])
    df.loc[:, cols_to_diff] = -df.loc[:, cols_to_diff]


    # 只保留起始年份及之后的数据
    df = df[df["Year"] >= start_year]

    # 定义成本列与图例名（注意：这里的键名要与经过 diff() 处理后的列名一致）
    cost_columns = {
        'GHG_ag difference(MtCO2e)': 'GHG_ag difference',
        'GHG_am(MtCOe2)': 'GHG_am',
        'GHG_non-ag(MtCOe2)': 'GHG_non-ag',
        'GHG_transition(MtCOe2)': 'GHG_transition'
    }

    # 自定义颜色映射
    color_map = {
        'GHG_ag difference': '#1F77B4',
        'GHG_am': '#FF7F0E',
        'GHG_non-ag': '#2CA02C',
        'GHG_transition': '#FFFF00',
    }

    # 设置绘图数据：使用 Year 列作为 x 轴
    x = df["Year"].values
    x_labels = df["Year"].tolist()

    # 初始化正负堆叠底部和总成本数组
    positive_bottom = np.zeros(len(df))
    negative_bottom = np.zeros(len(df))
    total_cost = np.zeros(len(df))

    # 绘制堆叠柱状图，支持正负值
    for raw_col, legend_name in cost_columns.items():
        values = df[raw_col].values
        pos_vals = np.where(values > 0, values, 0)
        neg_vals = np.where(values < 0, values, 0)

        ax_bar = plt.gca()
        ax_bar.bar(x, pos_vals, bottom=positive_bottom, label=legend_name, color=color_map[legend_name])
        ax_bar.bar(x, neg_vals, bottom=negative_bottom, color=color_map[legend_name])

        positive_bottom += pos_vals
        negative_bottom += neg_vals
        total_cost += values

    # 获取当前轴对象
    ax = plt.gca()

    # 添加总成本折线图（黑色点线图）
    ax.plot(x, total_cost, color='black', marker='o',
            linewidth=2, markersize=5, label='GHG')

    # 设置坐标轴与图例
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=45)
    ax.set_xlabel("Year")
    ax.set_ylabel("GHG abatement (MtCO2e)")
    ax.set_title(f"GHG abatement: {input_file.split('_')[3]} & BIO {input_file.split('_')[4]}")
    ax.grid(True)

    # 将图例显示在图下方一行
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
              ncol=3, fancybox=True, shadow=False)

    plt.tight_layout()
    plt.savefig(f"../output/{input_file}_ghg.png")
    plt.show()


def create_price_data(df_price, metric_class='cp'):
    """
    Prepare data for plotting:
      - metric_class = 'cp' (cost per GHG)
      - metric_class = 'bp' (cost per BIO)
    Steps:
      1. 过滤年份
      2. 宽表 (Year as index)
      3. 计算每个成本 ÷ GHG 或 BIO，列名用 f"{cm[:-4]}{unit}"
      4. melt 回长表
    """
    # 1. 过滤
    df = df_price[df_price["Year"] >= start_year].copy()

    # 2. 宽表
    wide = df.set_index("Year")

    if metric_class == 'cp':
        divisor_col = "GHG Abatement(MtCOe2)"
    elif metric_class == 'bp':
        divisor_col = "BIO(Mha)"
    else:
        raise ValueError("metric_class must be 'cp' or 'bp'")

    unit = name_dict[metric_class]["unit"]

    # 5. 计算并生成新列
    per_cols = []
    for cm in cost_columns:
        # cm[:-4] 会把 "(M$)" 去掉最后四个字符
        base_name = cm[:-4].strip()
        new_col = f"{base_name}{unit}"
        wide[new_col] = wide[cm] / wide[divisor_col]
        per_cols.append(new_col)

    # 6. melt 回长表
    long_df = (
        wide.reset_index()
            .melt(
                id_vars=["Year"],
                value_vars=per_cols,
                var_name="Metric",
                value_name="Value"
            )
    )

    return long_df




def plot_price(input_file: str):
    """
    Draw cp and bp cost-per-unit together in facets.
    """
    # 1) Read in your price data
    file_path_price = f"../output/02_{input_file}_price.xlsx"
    df_price = pd.read_excel(file_path_price)

    # 2) Create two long‐format DataFrames, one for cp and one for bp
    df_cp = create_price_data(df_price, metric_class='cp')
    df_bp = create_price_data(df_price, metric_class='bp')

    # 3) Label them and concatenate
    df_cp['Class'] = name_dict['cp']['title'] + name_dict['cp']['unit']      # "Carbon Price"
    df_bp['Class'] = name_dict['bp']['title'] + name_dict['bp']['unit']      # "Biodiversity Price"
    df_all = pd.concat([df_cp, df_bp], ignore_index=True)

    # 4) Compute total per Year+Class for the line overlay
    total_by_year = (
        df_all
        .groupby(['Class','Year'])['Value']
        .sum()
        .reset_index(name='TotalValue')
    )

    # 5) Build the plot with facet_wrap on Class
    # 按 'Class' 分组并提取每个组的最后一个点
    last_points = total_by_year.groupby('Class').apply(lambda group: group.iloc[[-1]]).reset_index(drop=True)

    # 保留两位小数的 TotalValue，并添加到新的列中
    last_points['Label'] = last_points['TotalValue'].map(lambda x: f"{x:.2f}")

    # 使用 ggplot 绘图
    p = (
            ggplot() +
            geom_bar(
                data=df_all,
                mapping=aes(x='Year', y='Value', fill='Metric'),
                stat='identity',
                position='stack'
            ) +
            geom_line(
                total_by_year,
                aes(x='Year', y='TotalValue', group='1'),
                color='black',
                size=1.2
            ) +
            geom_point(
                total_by_year,
                aes(x='Year', y='TotalValue'),
                color='black',
                size=2
            ) +
            # 添加标记每个图中最后一个点的值
            geom_text(
                data=last_points,
                mapping=aes(x='Year', y='TotalValue', label='Label'),
                color='black',
                va='bottom',  # 垂直方向对齐在点上方
                size=13
            ) +
            facet_wrap('~Class') +
            labs(
                title=input_file,
                x="",
                y="Value"
            ) +
            theme_minimal() +
            theme(
                legend_position='bottom',
                legend_title=element_blank(),
                axis_text_x=element_text(rotation=45, hjust=1),
                text=element_text(family="Arial", size=13),
                figure_size=(15, 8),
            )
    )

    p.show()
    p.save(f"../Graphs/{input_file}_price.png", dpi=300)  # 保存图像为文件
    return p


def create_cost_data(input_file: str):
    """
    将df_cost表中所有包含（M$）的列转换为长表格式
    """
    file_path_cost = f"../output/01_{input_file}_summary.xlsx"
    df_cost = pd.read_excel(file_path_cost)
    # 1. 过滤
    df_cost = df_cost[df_cost["Year"] >= start_year].copy()
    # 筛选包含（M$）的列
    dollar_cols = [col for col in df_cost.columns if '（M$）' in col]

    if not dollar_cols:
        # 如果没有找到包含（M$）的列，也检查包含(M$)的列（不同括号类型）
        dollar_cols = [col for col in df_cost.columns if '(M$)' in col]

    if not dollar_cols:
        raise ValueError("没有找到包含（M$）或(M$)的列")

    print(f"找到的包含M$的列: {dollar_cols}")

    # 假设df_cost有一个年份列，如果没有，需要添加

    # 创建一个空的DataFrame用于存储长格式数据
    long_df = pd.DataFrame()

    # 将每一列转为长格式并添加到long_df
    for col in dollar_cols:
        # 提取指标名称（去掉M$部分）
        metric_name = col.replace('（M$）', '').replace('(M$)', '').strip()

        # 创建临时数据框
        temp_df = df_cost[['Year', col]].copy()
        temp_df.columns = ['Year', 'Value']
        temp_df['Metric'] = metric_name

        # 添加到长格式数据框
        long_df = pd.concat([long_df, temp_df], ignore_index=True)

    return long_df


def plot_cost(input_file: str):
    """
    将df_cost中包含（M$）的列转为长表并绘制分面图
    """
    long_df = create_cost_data(input_file)

    # 设置matplotlib参数
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.size'] = 12

    # 创建分面图
    p = (
            ggplot() +
            geom_line(
                data=long_df,
                mapping=aes(x='Year', y='Value', group='1'),
                size=1.2
            ) +
            geom_point(
                data=long_df,
                mapping=aes(x='Year', y='Value'),
                size=3
            ) +
            facet_wrap('~Metric', scales='free_y') +  # 使用free_y让每个分面有独立的y轴比例
            labs(
                title=input_file,
                x="",
                y="Value (M$)"
            ) +
            theme_minimal() +
            theme(
                figure_size=(16, 10),  # 较大的图形尺寸以适应多个分面
                text=element_text(family="Arial", size=13),
                plot_title=element_text(family="Arial", size=18, face="bold"),
                axis_title=element_text(family="Arial", size=14),
                axis_text=element_text(family="Arial", size=12),
                strip_text=element_text(family="Arial", size=16, face="bold"),
                axis_text_x=element_text(rotation=45, hjust=1)
            )
    )
    p.show()
    p.save(f"../Graphs/{input_file}_cost.png", dpi=300)  # 保存图像为文件
    return p


def plot_revenue_cost_stacked(input_file, figure_size=(14, 8)):
    """
    读取表格，提取含有(M$)的列（去除cost_transition_ag2non_ag）
    将revenue相关列变为负数，绘制累积柱形图并添加总和折线

    参数:
        input_file: 输入文件名（不含路径和前缀）
        start_year: 起始年份
        figure_size: 图表尺寸

    返回:
        p: plotnine图表对象
        df_long: 处理后的长格式数据
    """
    # 读取数据

    file_path = f"../output/02_{input_file}_price.xlsx"
    df = pd.read_excel(file_path)

    # 筛选年份
    df = df[df["Year"] >= start_year].copy()

    # 筛选包含(M$)的列
    dollar_cols = []
    for col in cost_columns:
        dollar_cols.append(col)

    if not dollar_cols:
        raise ValueError("没有找到符合条件的(M$)列")

    print(f"找到的符合条件的M$列: {dollar_cols}")

    # 创建宽格式数据框，包含Year和提取的列
    df_wide = df[['Year'] + dollar_cols].copy()

    # 创建长格式数据
    df_long = pd.DataFrame()
    for col in dollar_cols:
        # 提取指标名称（去掉M$部分）
        metric_name = col.replace('（M$）', '').replace('(M$)', '').strip()

        # 创建临时数据框
        temp_df = df_wide[['Year', col]].copy()
        temp_df.columns = ['Year', 'Value']
        temp_df['Metric'] = metric_name

        # revenue相关列转为负值
        if 'revenue' in metric_name.lower():
            temp_df['Value'] = -temp_df['Value']

        # 添加到长格式数据框
        df_long = pd.concat([df_long, temp_df], ignore_index=True)

    # 计算每年的总和用于折线图
    total_by_year = (
        df_long
        .groupby('Year')['Value']
        .sum()
        .reset_index(name='TotalValue')
    )

    # 设置matplotlib参数
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.size'] = 12

    last_points = total_by_year.apply(lambda group: group.iloc[[-1]]).reset_index(drop=True)

    # 保留两位小数的 TotalValue，并添加到新的列中
    last_points['Label'] = last_points['TotalValue'].map(lambda x: f"{x:.2f}")

    # 绘制图表
    p = (
            ggplot() +
            geom_bar(
                data=df_long,
                mapping=aes(x='Year', y='Value', fill='Metric'),
                stat='identity',
                position='stack'
            ) +
            geom_line(
                total_by_year,
                aes(x='Year', y='TotalValue', group='1'),
                color='black',
                size=1.2
            ) +
            geom_point(
                total_by_year,
                aes(x='Year', y='TotalValue'),
                color='black',
                size=3
            ) +
            # 添加标记每个图中最后一个点的值
            geom_text(
                data=last_points,
                mapping=aes(x='Year', y='TotalValue', label='Label'),
                color='black',
                va='bottom',  # 垂直方向对齐在点上方
                size=13
            ) +
            labs(
                title=input_file,
                x="",
                y="Value (M$)",
                fill="Components"
            ) +
            theme_minimal() +
            theme(
                figure_size=figure_size,
                text=element_text(family="Arial", size=13),
                legend_position='bottom',
                axis_text_x=element_text(rotation=45, hjust=1)
            ) +
            scale_fill_brewer(type='qual', palette='Set2')  # 使用色盲友好的调色板
    )
    p.show()
    p.save(f"../Graphs/{input_file}_revenue_cost_stacked.png", dpi=300)  # 保存图像为文件
    return p, df_long


def plot_specified_columns(input_file, columns_to_plot, x_col='Year',
                           figure_size=(12, 8), stacked=False, colors=None,
                            y_label="Value", legend_title="Legend"):
    """
    读取表格并使用指定列绘制柱形图

    参数:
        input_file: 输入文件名（不含路径和前缀）
        columns_to_plot: 要绘制的列名列表
        title: 图表标题
        x_col: X轴所用的列名，默认为'Year'
        start_year: 起始年份，如果为None则使用所有数据
        figure_size: 图表尺寸，默认为(12, 8)
        stacked: 是否使用堆叠柱形图，默认为False（分组柱形图）
        colors: 自定义颜色列表，默认为None（使用默认颜色）
        x_label: X轴标签
        y_label: Y轴标签
        legend_title: 图例标题

    返回:
        p: plotnine图表对象
        df_long: 处理后的长格式数据
    """

    # 构建文件路径并读取数据
    file_path = f"../output/02_{input_file}_price.xlsx"
    df = pd.read_excel(file_path)

    # 筛选数据
    if start_year is not None and x_col == 'Year':
        df = df[df[x_col] >= start_year].copy()

    # 创建长格式数据
    df_selected = df[[x_col] + columns_to_plot].copy()
    df_long = pd.melt(
        df_selected,
        id_vars=[x_col],
        value_vars=columns_to_plot,
        var_name='Metric',
        value_name='Value'
    )

    # 设置matplotlib参数
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.size'] = 12

    # 确定位置参数
    position = 'stack' if stacked else 'dodge'

    # 绘制图表
    p = (
            ggplot(
                data=df_long,
                mapping=aes(x=x_col, y='Value', fill='Metric')
            ) +
            geom_bar(stat='identity', position=position, width=0.7) +
            labs(
                title=input_file,
                x="",
                y=y_label,
                fill=legend_title
            ) +
            theme_minimal() +
            theme(
                figure_size=figure_size,
                text=element_text(family="Arial", size=13),
                legend_position='bottom',
                axis_text_x=element_text(rotation=45, hjust=1)
            )
    )

    # 应用自定义颜色
    if colors:
        if len(colors) < len(columns_to_plot):
            print(f"警告: 提供的颜色数量({len(colors)})少于列数({len(columns_to_plot)})，将循环使用")
        p = p + scale_fill_manual(values=colors)
    p.show()
    p.save(f"../Graphs/{input_file}_{columns_to_plot[0]}.png", dpi=300)  # 保存图像为文件
    return p, df_long


def merge_png_images(image_paths_class, images_per_row=2, padding=10,
                     background_color=(255, 255, 255), add_filename=False,
                     filename_font_size=12, title=None, index=None):
    """
    将多个PNG图像拼接到一起，可指定每行显示的图像数量

    参数:
        image_paths: 图像文件路径列表或包含PNG图像的文件夹路径
        output_path: 输出图像的保存路径，默认为当前时间戳命名
        images_per_row: 每行显示的图像数量，默认为2
        padding: 图像之间的间距（像素），默认为10
        background_color: 背景颜色，默认为白色
        add_filename: 是否在图像下方添加文件名，默认为True
        filename_font_size: 文件名字体大小，默认为12
        title: 拼接图像的标题，默认为None

    返回:
        拼接后的PIL图像对象
    """
    # 处理输入路径
    image_paths = []
    for input_file in input_files:
        image_paths.append(os.path.join("..","Graphs",f"{input_file}_{image_paths_class}.png"))
    if isinstance(image_paths, str) and os.path.isdir(image_paths):
        # 如果提供的是文件夹路径，获取所有PNG文件
        folder_path = image_paths
        image_paths = [
            os.path.join(folder_path, f) for f in os.listdir(folder_path)
            if f.lower().endswith('.png')
        ]
        image_paths.sort()  # 按文件名排序

    if not image_paths:
        raise ValueError("未找到PNG图像文件")

    # 加载所有图像
    images = []
    max_width = 0
    max_height = 0

    for path in image_paths:
        try:
            img = Image.open(path)
            # 确保图像是RGB模式
            if img.mode != 'RGB':
                img = img.convert('RGB')
            images.append((img, os.path.basename(path)))
            max_width = max(max_width, img.width)
            max_height = max(max_height, img.height)
        except Exception as e:
            print(f"警告: 无法加载图像 {path}: {e}")

    if not images:
        raise ValueError("没有可用的图像可以拼接")

    # 计算行数
    num_images = len(images)
    num_rows = (num_images + images_per_row - 1) // images_per_row

    # 计算文件名区域的高度
    filename_height = filename_font_size + 10 if add_filename else 0

    # 创建新图像
    total_width = images_per_row * max_width + (images_per_row + 1) * padding
    total_height = num_rows * (max_height + filename_height) + (num_rows + 1) * padding

    # 如果有标题，添加标题空间
    title_height = 40 if title else 0
    total_height += title_height

    # 创建新的白色背景图像
    merged_image = Image.new('RGB', (total_width, total_height), background_color)

    # 在图像上绘制
    for idx, (img, filename) in enumerate(images):
        row = idx // images_per_row
        col = idx % images_per_row

        # 计算位置，使图像在单元格中居中
        x = col * (max_width + padding) + padding
        y = row * (max_height + filename_height + padding) + padding + title_height

        # 计算图像在单元格中的居中位置
        x_offset = (max_width - img.width) // 2
        y_offset = (max_height - img.height) // 2

        # 粘贴图像
        merged_image.paste(img, (x + x_offset, y + y_offset))

        # 如果需要，添加文件名
        if add_filename:
            # 使用matplotlib在图像上添加文本
            plt.figure(figsize=(total_width / 100, total_height / 100), dpi=100)
            plt.imshow(np.array(merged_image))
            plt.text(
                x + max_width / 2,
                y + max_height + 5,
                os.path.splitext(filename)[0],  # 去掉扩展名
                ha='center',
                va='top',
                fontsize=filename_font_size,
                color='black'
            )
            plt.axis('off')
            plt.tight_layout(pad=0)
            plt.savefig('temp.png', bbox_inches='tight', pad_inches=0)
            plt.close()
            merged_image = Image.open('temp.png')
            os.remove('temp.png')

    # 如果有标题，添加标题
    if title:
        plt.figure(figsize=(total_width / 100, total_height / 100), dpi=100)
        plt.imshow(np.array(merged_image))
        plt.text(
            total_width / 2,
            padding / 2,
            title,
            ha='center',
            va='top',
            fontsize=filename_font_size + 4,
            fontweight='bold',
            color='black'
        )
        plt.axis('off')
        plt.tight_layout(pad=0)
        plt.savefig('temp.png', bbox_inches='tight', pad_inches=0)
        plt.close()
        merged_image = Image.open('temp.png')
        os.remove('temp.png')

    # 保存结果
    output_path = os.path.join("..","Pannels",f"{index}{image_paths_class}_{suffix}.png")
    if output_path is None:
        # 使用当前时间创建文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"merged_images_{timestamp}.png"

    merged_image.save(output_path)
    # 显示拼接后的图像
    plt.figure(figsize=(10, 10))
    plt.imshow(np.array(merged_image))
    plt.axis('off')
    plt.tight_layout()
    plt.show()


    print(f"已将{len(images)}张图像拼接并保存到: {output_path}")

    return merged_image



def plot_histogram(input_file, arr_name, bins=30):
    """
    绘制直方图函数。

    :param input_file: 输入文件路径
    :param arr_name: 数据数组的名称
    :param bins: 直方图的分箱数量，默认值为 30
    """
    path_name = get_path(input_file)
    years = [2050]  # 你可以替换成其他年份或动态获取年份
    data = pd.concat(
        pd.DataFrame({'year': year,
                      'value': np.load(os.path.join(path_name, "data_for_carbon_price", f"{arr_name}_{year}.npy")) / 1e6})
        for year in years
    )

    # 绘制直方图
    plot = (
        ggplot(data, aes(x='value')) +
        geom_histogram(bins=bins, fill='blue', color='black', alpha=0.7) +
        labs(title=name_dict[arr_name]["title"], x=name_dict[arr_name]["unit"], y="Frequency") +
        theme(axis_text_x=element_text(angle=90))  # 修复旋转 x 轴标签
    )
    plot.show()
    print(plot)


def plot_violin_box(input_file: str,
                    arr_name: str,
                    year: int = 2050,
                    thr: list[float] = (0.25, 0.75)):
    """
    在指定年份用小提琴+箱线图展示 cost 分布，按分位数 [0,thr[0],thr[1],1] 划 Low/Moderate/High 三组，
    三个面板并排，各面板只显示组名，不带数量。
    """
    # 1. 载入数据并缩放
    path_name = get_path(input_file)
    arr_path = os.path.join(
        path_name, "data_for_carbon_price", f"{arr_name}_{year}.npy"
    )
    vals = np.load(arr_path) / 1e6

    # 2. 准备 DataFrame
    df = pd.DataFrame({"Value": vals})

    # 3. 按分位数分组
    df['Group'] = pd.qcut(
        df['Value'],
        q=[0, thr[0], thr[1], 1.0],
        labels=['Low', 'Moderate', 'High'],
        duplicates='drop'
    )

    # 4. 强制顺序
    df['Group'] = pd.Categorical(
        df['Group'],
        categories=['Low', 'Moderate', 'High'],
        ordered=True
    )

    # 5. 绘图
    p = (
        ggplot(df, aes(x='Group', y='Value', fill='Group'))
      + geom_violin(alpha=0.7, scale='width', trim=True)
      # 箱线宽度调小一点，背景填白
      + geom_boxplot(width=0.1, fill='white', outlier_size=1, alpha=0.8)
      + facet_wrap('~Group', scales='free_y', nrow=1)
      + labs(
            title=f"{arr_name} Distribution in {year}",
            x="Group",
            y="Value (M$)"
        )
      + theme(
            axis_text_x=element_text(angle=0, hjust=0.5),
            legend_position='none'
        )
    )
    p.show()
    return p


import os
import numpy as np
import pandas as pd
from plotnine import (
    ggplot, aes, geom_boxplot, geom_violin, facet_wrap,
    scale_x_discrete, labs, theme, element_text
)

def plot_box(input_file: str,
             arr_name: str,
             year: int = 2050,
             thr: tuple[float,float] = (0.25, 0.75)):
    """
    箱型图：按分位数 [0,thr[0],thr[1],1] 划 Low/Moderate/High 三组，
    三个面板并排，面板标签为 "组名 (样本数)"，X 轴仅显示组名。
    """
    # 1. 载入并缩放到 M$
    path_name = get_path(input_file)
    arr_file = os.path.join(
        path_name, "data_for_carbon_price", f"{arr_name}_{year}.npy"
    )
    vals = np.load(arr_file)
    df = pd.DataFrame({"Value": vals})

    # 2. 分组并强制顺序
    df['Group'] = pd.qcut(
        df['Value'],
        q=[0, thr[0], thr[1], 1.0],
        labels=['Low','Moderate','High'],
        duplicates='drop'
    )
    df['Group'] = df['Group'].astype(
        pd.CategoricalDtype(['Low','Moderate','High'], ordered=True)
    )

    # 3. 统计样本数，生成面板标签
    counts = df['Group'].value_counts().reindex(['Low','Moderate','High']).fillna(0).astype(int)
    df['GroupLabel'] = df['Group'].map(lambda g: f"{g} ({counts[g]})")
    df['GroupLabel'] = df['GroupLabel'].astype(
        pd.CategoricalDtype(
            [f"Low ({counts['Low']})",
             f"Moderate ({counts['Moderate']})",
             f"High ({counts['High']})"],
            ordered=True
        )
    )

    # 4. 绘图
    p = (
        ggplot(df, aes(x='Group', y='Value', fill='Group'))
      + geom_boxplot(width=0.6, outlier_size=1, alpha=0.8)
      + facet_wrap('~GroupLabel', scales='free_y', nrow=1)
      + scale_x_discrete(labels=['Low','Moderate','High'])
      + labs(
            title=f"{input_file} {arr_name} {year}",
            x="Group",
            y="Value (M$)"
        )
      + theme(
            axis_text_x=element_text(angle=0, hjust=0.5),
            legend_position='none'
        )
    )
    p.show()
    return p


def plot_violin(input_file: str,
                arr_name: str,
                year: int = 2050,
                thr: tuple[float,float] = (0.25, 0.75)):
    """
    小提琴+箱型图：按分位数 [0,thr[0],thr[1],1] 划 Low/Moderate/High 三组，
    三个面板并排，面板标签为 "组名 (样本数)"，X 轴仅显示组名。
    """
    # 1. 载入并缩放到 M$
    path_name = get_path(input_file)
    arr_file = os.path.join(
        path_name, "data_for_carbon_price", f"{arr_name}_{year}.npy"
    )
    vals = np.load(arr_file) / 1e6
    df = pd.DataFrame({"Value": vals})

    # 2. 分组并强制顺序
    df['Group'] = pd.qcut(
        df['Value'],
        q=[0, thr[0], thr[1], 1.0],
        labels=['Low','Moderate','High'],
        duplicates='drop'
    )
    df['Group'] = df['Group'].astype(
        pd.CategoricalDtype(['Low','Moderate','High'], ordered=True)
    )

    # 3. 统计样本数，生成面板标签
    counts = df['Group'].value_counts().reindex(['Low','Moderate','High']).fillna(0).astype(int)
    df['GroupLabel'] = df['Group'].map(lambda g: f"{g} ({counts[g]})")
    df['GroupLabel'] = df['GroupLabel'].astype(
        pd.CategoricalDtype(
            [f"Low ({counts['Low']})",
             f"Moderate ({counts['Moderate']})",
             f"High ({counts['High']})"],
            ordered=True
        )
    )

    # 4. 绘图
    p = (
            ggplot(df, aes(x='Group', y='Value', fill='Group'))
            + geom_violin(alpha=0.7, scale='width', trim=True)
            + facet_wrap('~GroupLabel', scales='free_y', nrow=1)
            + scale_x_discrete(labels=['Low', 'Moderate', 'High'])
            + labs(
        title=f"{arr_name} Violin in {year}",
        x="Group",
        y="Value (M$)"
    )
            + theme(
        axis_text_x=element_text(angle=0, hjust=0.5),
        legend_position='none'
    )
    )
    p.show()
    return p

import os
import numpy as np
import pandas as pd
from plotnine import (
    ggplot, aes, geom_violin, facet_wrap,
    labs, theme, element_text
)

def plot_violin_quantile_groups(input_file: str,
                                arr_name: str,
                                year: int = 2050,
                                quantile_thresholds: list[float] = [0.2,0.4,0.6,0.8]):
    """
    Split the data into five quantile‐based groups and draw side‐by‐side violin plots,
    with each facet only showing its own group on the x‐axis.
    """
    # 1. load and scale to millions
    data_dir = get_path(input_file)
    arr_path = os.path.join(
        data_dir,
        "data_for_carbon_price",
        f"{arr_name}_{year}.npy"
    )
    vals = np.load(arr_path) / 1e6
    df = pd.DataFrame({"Value": vals})

    # 2. validate thresholds
    if quantile_thresholds is None or len(quantile_thresholds) != 4:
        raise ValueError("Please supply four quantile thresholds in (0,1), e.g. [0.2,0.4,0.6,0.8]")

    # 3. compute quantile cut points
    q_values = df["Value"].quantile(quantile_thresholds).tolist()

    # 4. bin into five ordered groups
    bins = [-np.inf, *q_values, np.inf]
    group_names = ['Low', 'Lower-Mid', 'Mid', 'Upper-Mid', 'High']
    df['Group'] = pd.cut(
        df['Value'], bins=bins, labels=group_names,
        include_lowest=True, right=False
    ).astype(pd.CategoricalDtype(group_names, ordered=True))

    # 5. count and make facet labels
    counts = df['Group'].value_counts().reindex(group_names).fillna(0).astype(int)
    label_map = {g: f"{g} ({counts[g]})" for g in group_names}
    df['GroupLabel'] = df['Group'].map(label_map)
    df['GroupLabel'] = df['GroupLabel'].astype(
        pd.CategoricalDtype([label_map[g] for g in group_names], ordered=True)
    )

    # 6. draw violins with free x‐ and y‐scales
    p = (
        ggplot(df, aes(x='Group', y='Value', fill='Group'))
      + geom_violin(alpha=0.7, scale='width', trim=True)
      + facet_wrap('~GroupLabel', scales='free', nrow=1)
      + labs(
            title=f"{input_file} {arr_name} ({year})",
            x="Group", y="Value ($M)"
        )
      + theme(
            axis_text_x=element_text(angle=0, hjust=0.5),
            legend_position='none'
        )
    )
    p.show()
    return p

import os
import numpy as np
import pandas as pd
from plotnine import (
    ggplot, aes, geom_boxplot, facet_wrap,
    labs, theme, element_text
)

def plot_box_quantile_groups(input_file: str,
                             arr_name: str,
                             year: int = 2050,
                             quantile_thresholds: list[float] = [0.2,0.4,0.6,0.8]):
    """
    Split the data into five quantile‐based groups and plot side‐by‐side boxplots,
    with each facet only showing its own group on the x‐axis and titled "GroupName (count)".
    """
    # 1. load and scale to millions
    data_dir = get_path(input_file)
    arr_path = os.path.join(
        data_dir,
        "data_for_carbon_price",
        f"{arr_name}_{year}.npy"
    )
    vals = np.load(arr_path)
    df = pd.DataFrame({"Value": vals})
    if arr_name == 'bp' or arr_name == 'cp':
        df = df[df["Value"] > 0].copy()

    # 2. validate thresholds
    if quantile_thresholds is None or len(quantile_thresholds) != 4:
        raise ValueError("Please supply four quantile thresholds in (0,1), e.g. [0.2,0.4,0.6,0.8]")

    # 3. compute cut points
    q_values = df["Value"].quantile(quantile_thresholds).tolist()

    # 4. bin into 5 ordered groups
    bins = [-np.inf, *q_values, np.inf]
    group_names = ['Low', 'Lower-Mid', 'Mid', 'Upper-Mid', 'High']
    df['Group'] = pd.cut(
        df['Value'],
        bins=bins,
        labels=group_names,
        include_lowest=True,
        right=False
    ).astype(pd.CategoricalDtype(group_names, ordered=True))

    # 5. count and create facet labels
    counts = df['Group'].value_counts().reindex(group_names).fillna(0).astype(int)
    label_map = {g: f"{g} ({counts[g]})" for g in group_names}
    df['GroupLabel'] = df['Group'].map(label_map)
    df['GroupLabel'] = df['GroupLabel'].astype(
        pd.CategoricalDtype([label_map[g] for g in group_names], ordered=True)
    )

    # 6. plot boxplots with free x and y scales
    p = (
        ggplot(df, aes(x='Group', y='Value', fill='Group'))
      + geom_boxplot(width=0.6, outlier_size=1, alpha=0.8)
      + facet_wrap('~GroupLabel', scales='free', nrow=1)
      + labs(
            title=f"{input_file} {arr_name} ({year})",
            x="Group",
            y="Value ($M)"
        )
      + theme(
            axis_text_x=element_text(angle=0, hjust=0.5),
            legend_position='none'
        )
    )
    p.show()
    return p


def plot_box_trimmed_equal_bins(input_file: str,
                                arr_name: str,
                                year: int = 2050,
                                trim_pct: float = 0.05) -> None:
    """
    1. Load the .npy array (divide by 1e6 if 'cost'),
    2. Trim lowest & highest `trim_pct` fraction,
    3. Bin the remainder into 5 equal–width bins,
    4. Save a *_bins.npy mask matching original shape (nan where trimmed),
    5. Plot side–by–side boxplots with free y–axes.
    """
    # 1) load
    data_dir = get_path(input_file)
    arr_path = os.path.join(data_dir,
                            "data_for_carbon_price",
                            f"{arr_name}_{year}.npy")
    vals = np.load(arr_path)
    if arr_name == 'cost':
        vals = vals / 1e6  # scale

    # 2) trim extremes
    mask = np.ones_like(vals, dtype=bool)
    if arr_name in ('bp', 'cp'):
        # keep only positive and drop top trim_pct
        positive = vals > 0
        lo, hi = np.quantile(vals[positive], [trim_pct, 1 - trim_pct])
        mask &= positive & (vals >= lo) & (vals <= hi)

    # 3) compute bins on the trimmed values
    kept = vals[mask]
    labels = ['Low', 'Lower-Mid', 'Mid', 'Upper-Mid', 'High']
    if kept.size:
        lo, hi = kept.min(), kept.max()
        bins = np.linspace(lo, hi, 6)  # 5 intervals

        # digitize -> 0..4
        inds = np.digitize(kept, bins, right=False) - 1
        inds = np.clip(inds, 0, 4)
    else:
        # nothing kept
        bins = np.linspace(0, 1, 6)
        inds = np.array([], dtype=int)

    # 4) build full_codes array and save
    full_codes = np.full(vals.shape, np.nan)
    full_codes[mask] = inds
    out_path = os.path.join(data_dir,
                            "data_for_carbon_price",
                            f"{arr_name}_{year}_bins.npy")
    np.save(out_path, full_codes)

    # 5) prepare DataFrame for plotting
    df = pd.DataFrame({
        'Value': kept,
        'Group': pd.Categorical.from_codes(
            codes=inds,
            categories=labels,
            ordered=True
        )
    })
    # make nice facet labels "Name (count)"
    counts = df['Group'].value_counts().reindex(labels, fill_value=0)
    df['GroupLabel'] = df['Group'].map(lambda g: f"{g} ({counts[g]})")
    df['GroupLabel'] = pd.Categorical(
        df['GroupLabel'],
        categories=[f"{g} ({counts[g]})" for g in labels],
        ordered=True
    )

    # 6) plot
    p = (
            ggplot(df, aes(x='Group', y='Value', fill='Group'))
            + geom_boxplot(width=0.6, outlier_size=1, alpha=0.8)
            + facet_wrap('~GroupLabel', scales='free_y', nrow=1)
            + labs(
        title=f"{name_dict[arr_name]['title']} after trimming ±{int(trim_pct * 100)}%",
        x="Group",
        y=name_dict[arr_name]['unit']
    )
            + theme(
        axis_text_x=element_blank(),
        legend_position='none'
    )
    )
    p.show()
    return p