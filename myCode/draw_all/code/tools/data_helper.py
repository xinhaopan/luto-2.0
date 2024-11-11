import numpy as np
import os

import os
import pandas as pd

def get_path(path_name):
    output_path = f"../../../output/{path_name}/output"
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


def get_dict_data(input_files, csv_name, value_column_name, filter_column_name):
    """
    从多个文件中读取数据并按指定列分组求和。

    参数:
    - input_files (list): 输入文件的列表。
    - csv_name (str): 目标 CSV 文件的名称前缀（不含年份）。
    - value_column_name (str): 要求和的列名。
    - filter_column_name (str): 过滤列名，根据此列的唯一值进行分组求和。

    返回:
    - dict: 每个输入文件的汇总数据字典，每个文件对应一个 DataFrame。
    """
    data_dict = {}

    # 遍历每个 input_file 进行处理
    for input_name in input_files:
        base_path = get_path(input_name)

        # 创建以年份为索引的 DataFrame
        temp_results = pd.DataFrame(index=range(2010, 2051))
        temp_results.index.name = 'Year'

        # 遍历2010到2050年的文件
        for year in range(2010, 2051):
            file_path = os.path.join(base_path, f'out_{year}', f'{csv_name}_{year}.csv')

            # 如果文件存在，进行处理
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)

                # 按 `filter_column_name` 中的唯一值进行分组求和
                unique_values = df[filter_column_name].unique()
                for value in unique_values:
                    total_value = df[df[filter_column_name] == value][value_column_name].sum() / 1e6
                    temp_results.loc[year, value] = total_value  # 将结果填入年份索引中

        # 将结果 DataFrame 添加到字典
        data_dict[input_name] = temp_results

    return data_dict


def aggregate_by_mapping(data_dict, mapping_file, from_field, to_field, sheet_name=None, summary_col_name=None):
    """
    根据映射文件对 data_dict 中的每个 DataFrame 按映射后的字段分组并求和，并将 Year 列设置为索引。

    参数:
    data_dict (dict): 包含多个 DataFrame 的字典。
    mapping_file (str): 映射文件路径，包含映射前字段和映射后字段。
    from_field (str): 映射前字段名。
    to_field (str): 映射后字段名。
    sheet_name (str, 可选): 如果指定，读取映射文件中的指定 sheet。
    summary_col_name (str, 可选): 如果指定，对每个表的每行求和，并将结果存储为新的列，默认值为 None。

    返回:
    dict: 包含按映射后字段聚合的 DataFrame 的新字典。
    """
    # 读取映射文件
    if sheet_name:
        mapping_df = pd.read_excel(mapping_file, sheet_name=sheet_name)
    else:
        mapping_df = pd.read_excel(mapping_file)

    # 构建映射字典
    mapping_dict = mapping_df.set_index(from_field)[to_field].to_dict()

    # 创建新的字典存储聚合后的结果
    aggregated_dict = {}

    # 对每个 DataFrame 进行映射和聚合
    for key, df in data_dict.items():
        # 确保 Year 列存在且将其设置为索引
        if 'Year' in df.columns:
            df = df.set_index('Year')

        # 使用映射将列名替换为映射后的字段
        renamed_df = df.rename(columns=mapping_dict)

        # 按映射后字段分组并求和
        aggregated_df = renamed_df.T.groupby(renamed_df.columns).sum().T

        # 如果指定了 summary_col_name 参数，添加一列对每行求和
        if summary_col_name:
            aggregated_df[summary_col_name] = aggregated_df.sum(axis=1)

        # 将聚合后的 DataFrame 添加到新字典中
        aggregated_dict[key] = aggregated_df

    return aggregated_dict




def concatenate_dicts_by_year(data_dicts):
    """
    将多个字典中的键相同的 DataFrame 横向拼接，以 Year 列或索引为主键。

    参数:
    data_dicts (list of dict): 包含多个字典，每个字典的值都是以 Year 为主键的 DataFrame。

    返回:
    dict: 按键相同的表横向拼接后的新字典。
    """
    # 初始化新字典，用于存储拼接后的表
    concatenated_dict = {}

    # 获取所有字典中的公共键
    keys = set(data_dicts[0].keys())
    for data_dict in data_dicts[1:]:
        keys &= set(data_dict.keys())

    # 对于每个共同的键，将对应的 DataFrame 进行拼接
    for key in keys:
        # 获取当前键对应的所有 DataFrame
        dfs = []
        for i, data_dict in enumerate(data_dicts):
            if key in data_dict:
                df = data_dict[key]
                # 检查是否存在 Year 列或索引，如果都不存在则输出提示信息
                if 'Year' not in df.columns and df.index.name != 'Year':
                    print(f"Warning: DataFrame in dictionary {i} for key '{key}' is missing 'Year' column or index.")
                    continue  # 跳过此 DataFrame

                # 如果 Year 是列，将其设置为索引
                if 'Year' in df.columns:
                    df = df.set_index('Year')

                dfs.append(df)

        # 横向拼接
        concatenated_df = pd.concat(dfs, axis=1)

        # 将拼接结果存入新字典
        concatenated_dict[key] = concatenated_df.reset_index()

    return concatenated_dict




