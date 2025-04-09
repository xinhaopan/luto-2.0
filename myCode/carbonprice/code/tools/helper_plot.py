# --- 标准库 ---
import os
import re
import math

# --- 第三方库 ---
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy_financial as npf
from joblib import Parallel, delayed
from tqdm import tqdm  # 更推荐只导入 tqdm 本体
import rasterio
from sklearn.preprocessing import MinMaxScaler
from PIL import Image

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


def plot_cost_vs_price(input_file: str, start_year: int = 2021):
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
    plt.show()


def plot_absolute_cost_stack(input_file: str, start_year: int = 2021):
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
    plt.show()


def plot_ghg_stack(input_file: str, start_year: int = 2021):
    # 构造文件路径并读取数据
    file_path = f"../output/01_{input_file}_summary.xlsx"
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
    plt.show()

