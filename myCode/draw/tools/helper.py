import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import os

def get_path(path_name):
    output_path = f"../../output/{path_name}/output"
    try:
        if os.path.exists(output_path):
            subdirectories = os.listdir(output_path)
            numeric_starting_subdir = [s for s in subdirectories if s[0].isdigit()][0]
            subdirectory_path = os.path.join(output_path, numeric_starting_subdir)
            return subdirectory_path
        else:
            raise FileNotFoundError(f"The specified output path does not exist: {output_path}")
    except (IndexError, FileNotFoundError) as e:
        print(f"Error occurred while getting path for {path_name}: {e}")
        print(f"Current directory content for {output_path}: {os.listdir(output_path) if os.path.exists(output_path) else 'Directory not found'}")


def process_input_files(input_files):
    """
    处理多个输入文件并返回合并后的 DataFrame。

    参数:
    - input_files: 包含 (INPUT_NAME, file_name, column_name) 的元组的集合
    - get_path: 一个函数，用于获取每个 INPUT_NAME 对应的路径

    返回:
    - results: 合并后的 DataFrame，按年份索引
    """

    # 创建一个字典，存储所有的输入数据，key 是年份，value 是对应的各个 INPUT_NAME 的值
    data_dict = {'Year': list(range(2010, 2051))}

    # 循环每个 INPUT_NAME 来处理
    for INPUT_NAME, file_name, column_name in input_files:
        # 初始化每个 INPUT_NAME 的数据存储字典
        data_dict[INPUT_NAME] = []

        base_path = get_path(INPUT_NAME)

        # 遍历2010到2050年的文件
        for year in range(2010, 2051):
            # 构建每个CSV文件的路径
            file_path = os.path.join(base_path, f'out_{year}', f'{file_name}_{year}.csv')

            # 检查文件是否存在
            if not os.path.exists(file_path):
                # 如果2010年的文件不存在，使用2011年的文件
                if year == 2010 and column_name == 'Prod_targ_year (tonnes, KL)':
                    file_path = os.path.join(base_path, f'out_{2011}', f'{file_name}_{2011}.csv')
                    column_name = 'Prod_base_year (tonnes, KL)'

            # 读取CSV文件
            df = pd.read_csv(file_path)

            # 提取列的值
            value = df[column_name] / 1e6

            data_dict[INPUT_NAME].append(value)

    # 将字典转换为 DataFrame
    results = pd.DataFrame(data_dict)

    return results


def expand_tables(results):
    """
    将 results 中的每个列（Series 或单个值）展开为新的 DataFrame，年份保持为行。

    参数:
    - results: 包含多个列的 DataFrame，其中每列可能是 Series 或单个值

    返回:
    - expanded_tables: 一个字典，键是原来的列名，值是展开后的 DataFrame
    """

    # 存储每个展开后的 DataFrame
    expanded_tables = {}

    # 遍历 results 中的每一列
    for column in results.columns:
        if column == 'Year':  # 跳过 Year 列
            continue

        # 初始化一个新的 DataFrame，用于存储展开后的数据
        temp_df = pd.DataFrame()

        # 遍历每一行，展开 Series
        for i, data in results[column].items():
            year = results.loc[i, 'Year']

            if isinstance(data, pd.Series):  # 如果是 Series，则展开
                expanded_data = pd.DataFrame(data).T  # 转置使 Series 成为一行
                expanded_data['Year'] = year  # 保留年份
            else:  # 如果是单个值，直接添加
                expanded_data = pd.DataFrame({0: [data], 'Year': [year]})

            # 将展开的数据添加到临时 DataFrame 中
            temp_df = pd.concat([temp_df, expanded_data], ignore_index=True)

        # 将展开后的 DataFrame 存入字典
        expanded_tables[column] = temp_df.set_index('Year')  # 保持年份作为索引

    return expanded_tables


def get_value(input_files, csv_name, row_name, column_name):
    # 创建主字典，用于存储各年结果
    data_dict = {}

    # 循环每个 input_file 来处理
    for input_name in input_files:
        base_path = get_path(input_name)

        # 初始化存储每个 INPUT_NAME 的结果
        temp_results = pd.DataFrame(index=list(range(2010, 2051)))  # 以年份为索引

        # 遍历2010到2050年的文件
        for year in range(2010, 2051):
            # 构建每个CSV文件的路径
            file_path = os.path.join(base_path, f'out_{year}', f'{csv_name}_{year}.csv')

            # 确保文件存在，防止文件缺失导致错误
            if not os.path.exists(file_path):
                print(f"File not found: {file_path}, skipping...")
                continue

            # 读取CSV文件
            df = pd.read_csv(file_path)

            # 如果是第一个年份（2010），初始化 unique_values
            if year == 2010:
                unique_values = df.iloc[:, 0].unique()  # 第一列的唯一值
                # 初始化列名为 unique_values 的临时 DataFrame
                for value in unique_values:
                    temp_results[value] = None  # 为每个 unique_value 添加列

            # 为每个 unique_value 提取对应的值
            values = []
            for row1_name in unique_values:
                # 提取第一列为 row1_name, 第二列为 row_name 对应的第三列值
                value = df[(df.iloc[:, 0] == row1_name) & (df.iloc[:, 1] == row_name)][
                            column_name].sum() / 1e6  # 转换单位为百万
                values.append(value)

            # 在 temp_results 中添加结果, 使用年份作为索引
            temp_results.loc[year] = values

        # 将每个 input_file 的结果添加到 data_dict 中，使用 input_name 作为键
        data_dict[input_name] = temp_results

    return data_dict

def get_unique_value(input_files, csv_name, row_name, column_name):
    # 创建主字典，用于存储各年结果
    data_dict = {}

    # 循环每个 input_file 来处理
    for input_name in input_files:
        base_path = get_path(input_name)

        # 初始化存储每个 INPUT_NAME 的结果
        temp_results = pd.DataFrame(index=list(range(2010, 2051)),columns=[column_name])  # 以年份为索引

        # 遍历2010到2050年的文件
        for year in range(2010, 2051):
            # 构建每个CSV文件的路径
            file_path = os.path.join(base_path, f'out_{year}', f'{csv_name}_{year}.csv')

            # 确保文件存在，防止文件缺失导致错误
            if not os.path.exists(file_path):
                print(f"File not found: {file_path}, skipping...")
                continue

            # 读取CSV文件
            df = pd.read_csv(file_path, index_col=0)

            value = df.loc[row_name, column_name] / 1e6  # 转换单位为百万

            # 在 temp_results 中添加结果, 使用年份作为索引
            temp_results.loc[year] = value

        # 将每个 input_file 的结果添加到 data_dict 中，使用 input_name 作为键
        data_dict[input_name] = temp_results

    return data_dict


def get_value_sum(input_files, csv_name, filter_column_name, value_column_name):
    # 创建主字典，用于存储各年结果
    data_dict = {}

    # 循环每个 input_file 来处理
    for input_name in input_files:
        base_path = get_path(input_name)

        # 初始化存储每个 INPUT_NAME 的结果
        temp_results = pd.DataFrame(index=list(range(2010, 2051)))  # 以年份为索引
        temp_results.index.name = 'Year'

        # 遍历2010到2050年的文件
        for year in range(2010, 2051):
            # 构建每个CSV文件的路径
            file_path = os.path.join(base_path, f'out_{year}', f'{csv_name}_{year}.csv')

            # 确保文件存在，防止文件缺失导致错误
            if not os.path.exists(file_path):
                print(f"File not found: {file_path}, skipping...")
                continue

            # 读取CSV文件
            df = pd.read_csv(file_path)

            # 如果传入的 filter_column_name 是 "*"，直接求 value_column_name 列的和
            if filter_column_name == "*":
                total_value = df[value_column_name].sum() / 1e6  # 转换单位为百万
                temp_results.loc[year, 'Total'] = total_value  # 只添加一列 Total 表示每年的总和
            else:
                # 如果是第一个年份（2010），初始化 unique_values
                if year == 2010:
                    unique_values = df[filter_column_name].unique()  # 第一列的唯一值
                    # 初始化列名为 unique_values 的临时 DataFrame
                    for value in unique_values:
                        temp_results[value] = None  # 为每个 unique_value 添加列

                # 为每个 unique_value 提取对应的值
                values = []
                for row1_name in unique_values:
                    # 提取 filter_column_name 列为 row1_name 对应的第三列值
                    value = df[df[filter_column_name] == row1_name][value_column_name].sum() / 1e6  # 转换单位为百万
                    values.append(value)

                # 在 temp_results 中添加结果, 使用年份作为索引
                temp_results.loc[year] = values

        # 将每个 input_file 的结果添加到 data_dict 中，使用 input_name 作为键
        data_dict[input_name] = temp_results

    return data_dict


# 通用函数，用于创建任意数量的图，并设置每行图的数量
def plot_Combination_figures(merged_dict, output_png, input_names, plot_func, categories=None, color_list=None,
                             point_data=None,
                             n_rows=3, n_cols=3, font_size=10, x_range=(2010, 2050), y_range=(-600, 100),
                             x_ticks=5, y_ticks=100, x_label='Year', y_label='GHG Emissions (Mt CO2e)',
                             legend_position=(0.5, -0.03), show_legend='last', label_positions=(0.05, 0.95),legend_n_rows=1):
    total_plots = len(input_names)
    # 按照 A4 纸的比例动态调整 figsize
    fig_width = 12  # 根据列数调整宽度
    fig_height = 8  # 根据行数调整高度
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height))  # 动态调整图的大小

    # 扁平化 axs 以支持单个子图或多图
    axs = axs.flat if total_plots > 1 else [axs]

    all_handles = []
    all_labels = []

    # 根据图的数量动态生成标签 (a), (b), (c), ...
    labels = [f'({chr(97 + i)})' for i in range(total_plots)]  # 生成 (a), (b), (c), ...

    # 循环绘制每个图
    for i, ax in enumerate(axs):
        if i < total_plots:
            if point_data is not None:
                # 绘制每个子图，获取句柄
                bar_list, line = plot_func(ax, merged_dict=merged_dict, input_name=input_names[i], categories=categories, color_list=color_list, point_data=point_data,
                              font_size=font_size, x_range=x_range, y_range=y_range, x_ticks=x_ticks, y_ticks=y_ticks,
                              x_label=x_label, y_label=y_label, show_legend=show_legend)
                if i == len(axs)-1:
                    categories.insert(0, point_data)
            else:
                bar_list = plot_func(ax, merged_dict=merged_dict, input_name=input_names[i], categories=categories, color_list=color_list,
                              font_size=font_size, x_range=x_range, y_range=y_range, x_ticks=x_ticks, y_ticks=y_ticks,
                              x_label=x_label, y_label=y_label, show_legend=show_legend)
            # 在左上角添加标签，例如 (a), (b), (c)
            ax.text(label_positions[0], label_positions[1], labels[i], transform=ax.transAxes, fontsize=font_size,
                    verticalalignment='top', horizontalalignment='left')

            # 只在最后一个图获取图例句柄和标签
            if i == total_plots - 1:
                handles, labels_legend = ax.get_legend_handles_labels()
                all_handles.extend(handles[:len(categories)])  # 只添加堆积柱状图的图例
                all_labels.extend(labels_legend[:len(categories)])

    ncol = len(categories) // legend_n_rows  # 计算每行的列数
    # 在整体图的下方显示图例，设置为一行
    fig.legend(all_handles, all_labels, loc='lower center', bbox_to_anchor=legend_position, ncol=ncol,
               frameon=False, fontsize=font_size)

    # 调整布局
    plt.tight_layout()
    plt.savefig(output_png, bbox_inches='tight', dpi=300)  # 保存图片
    plt.show()

def plot_stacked_bar(ax, merged_dict, input_name, categories, color_list, font_size=10,
                     x_range=(2010, 2050), y_range=(-600, 100), x_ticks=None, y_ticks=None,
                     x_label='Year', y_label='GHG Emissions (Mt CO2e)', show_legend=False):
    merged_df = merged_dict[input_name]
    merged_df.index = merged_df.index.astype(int)

    # 准备数据
    years = merged_df.index
    data = np.array([merged_df[category] for category in categories])

    # 绘制正数的堆积柱状图
    pos_data = np.maximum(data, 0)  # 将负数设置为0
    bar_list = []
    bar_list.append(ax.bar(years, pos_data[0], label=categories[0], color=color_list[0], width=0.8))
    for i in range(1, len(categories)):
        bar_list.append(
            ax.bar(years, pos_data[i], bottom=pos_data[:i].sum(axis=0), label=categories[i], color=color_list[i],
                   width=0.8))

    # 绘制负数的堆积柱状图
    neg_data = np.minimum(data, 0)  # 将正数设置为0
    for i in range(len(categories)):
        bar_list.append(
            ax.bar(years, neg_data[i], bottom=neg_data[:i].sum(axis=0), label=categories[i], color=color_list[i],
                   width=0.8))

    # 设置刻度朝内
    ax.tick_params(axis='both', which='both', direction='in', labelsize=font_size)

    # 设置 x 轴刻度和间距
    ax.set_xlim(x_range[0] - 0.5, x_range[1] + 0.5)
    if x_ticks is not None:
        ax.set_xticks(np.arange(x_range[0], x_range[1] + 1, x_ticks))
    ax.set_xticklabels(ax.get_xticks(), fontsize=font_size)

    # 设置 y 轴范围和刻度
    ax.set_ylim(y_range[0], y_range[1])
    if y_ticks is not None:
        ax.set_yticks(np.arange(y_range[0], y_range[1] + 1, y_ticks))

    # 设置自定义 x 轴和 y 轴标签
    ax.set_xlabel(x_label, fontsize=font_size)
    ax.set_ylabel(y_label, fontsize=font_size)

    # 返回柱状图句柄
    return bar_list


def process_land_use_tables(tables_dict, mapping_file='tools/lumap_grouped.csv'):
    # 从 Excel 文件中读取映射表
    mapping_df = pd.read_csv(mapping_file)
    column_mapping = dict(zip(mapping_df['lu_desc_0'], mapping_df['lu_desc']))

    grouped_tables = {}

    # 遍历每个表并进行分组求和
    for name, table in tables_dict.items():
        # 按照映射关系重命名列
        table = table.rename(columns=column_mapping)
        # 按新类别分组求和
        grouped_table = table.groupby(by=table.columns, axis=1).sum()
        # 将结果存入字典
        grouped_tables[name] = grouped_table

    return grouped_tables


import pandas as pd


def prepare_draw_data(merged_dict, mapping_file):
    """
    根据 merged_dict 提取所需的 draw_dict、categories 和 color_list。

    参数:
    merged_dict (dict): 包含多个 DataFrame 的字典。
    mapping_file (str): 映射文件路径，包含 lu_desc 和 lu_color_HEX 列。

    返回:
    tuple: 包含 draw_dict（按名字重命名的列）、categories 和 color_list。
    """
    # 获取第一个 DataFrame 的列名作为初始 categories
    categories = list(next(iter(merged_dict.values())).columns)

    # 获取映射关系和对应颜色
    name_to_lu_desc, color_list = get_names_to_lu_desc_mapping(categories, mapping_file)

    # 使用映射关系重命名字典中的列名
    draw_dict = rename_columns_in_dict(merged_dict, name_to_lu_desc)

    # 更新 categories 为重命名后的列名顺序
    categories = list(name_to_lu_desc.values())

    return draw_dict, categories, color_list


# 定义的帮助函数
def get_names_to_lu_desc_mapping(names, mapping_file):
    mapping_df = pd.read_csv(mapping_file)
    unique_colors_df = mapping_df[['lu_desc', 'lu_color_HEX']].drop_duplicates()
    unique_colors_df['lu_desc_lower'] = unique_colors_df['lu_desc'].str.lower()
    names_lower = [name.lower() for name in names]
    filtered_df = unique_colors_df[unique_colors_df['lu_desc_lower'].isin(names_lower)]

    name_to_lu_desc = {}
    color_list = []
    for name in names:
        lower_name = name.lower()
        match = filtered_df[filtered_df['lu_desc_lower'] == lower_name]
        if not match.empty:
            name_to_lu_desc[name] = match['lu_desc'].iloc[0]
            color_list.append(match['lu_color_HEX'].iloc[0])

    return name_to_lu_desc, color_list


def rename_columns_in_dict(data_dict, name_to_lu_desc):
    renamed_data_dict = {}
    for key, df in data_dict.items():
        new_columns = [name_to_lu_desc.get(col, col) for col in df.columns]
        renamed_data_dict[key] = df.copy()
        renamed_data_dict[key].columns = new_columns
    return renamed_data_dict






