import os
import re
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import xy
from pyproj import CRS, Transformer
from joblib import Parallel, delayed
from tools.parameters import TASK_ROOT


def get_path(path_name):
    output_path = f"../../../output/{TASK_ROOT}/{path_name}/output"
    try:
        if os.path.exists(output_path):
            subdirectories = os.listdir(output_path)
            # numeric_starting_subdir = [s for s in subdirectories if s[0].isdigit()][0]
            numeric_starting_subdir = [s for s in subdirectories if "2010-2050" in s][0]
            subdirectory_path = os.path.join(output_path, numeric_starting_subdir)
            return subdirectory_path
        else:
            raise FileNotFoundError(f"The specified output path does not exist: {output_path}")
    except (IndexError, FileNotFoundError) as e:
        print(f"Error occurred while getting path for {path_name}: {e}")
        print(f"Current directory content for {output_path}: {os.listdir(output_path) if os.path.exists(output_path) else 'Directory not found'}")

import pandas as pd

def fill_empty_dataframes(data_dict, fill_value=None, columns_to_fill=None):
    """
    对于字典中为空的 DataFrame，填充指定列，所有值设为 fill_value，保留原始索引。

    Parameters:
        data_dict (dict): 包含 DataFrame 的字典。
        fill_value (any): 用于填充的值，例如 None、0、-10 等。
        columns_to_fill (list): 要填充的列名列表。如果未提供，则跳过空表。

    Returns:
        dict: 处理后的 DataFrame 字典。
    """
    new_data_dict = {}

    for key, df in data_dict.items():
        if df.empty or df.shape[1] == 0:
            if columns_to_fill is not None:
                index = df.index if not df.index.empty else range(2010, 2051)
                filled_df = pd.DataFrame(
                    {col: [fill_value] * len(index) for col in columns_to_fill},
                    index=index
                )
                new_data_dict[key] = filled_df
            else:
                # 如果没有提供 columns_to_fill，就跳过
                new_data_dict[key] = df.copy()
        else:
            new_data_dict[key] = df.copy()

    return new_data_dict


import pandas as pd

def rename_and_filter_columns(data_dict, columns_to_keep, new_column_names=None, default_value=None):
    """
    保留并重命名字典中每个 DataFrame 的指定列，如果缺列则添加并填充默认值。

    Parameters:
        data_dict (dict): 包含 DataFrame 的字典。
        columns_to_keep (list): 要保留的列名。
        new_column_names (list, optional): 新列名（与 columns_to_keep 对应）。
        default_value (any): 如果某列不存在，用该值填充新列。

    Returns:
        dict: 处理后的新 DataFrame 字典。
    """
    if new_column_names and len(columns_to_keep) != len(new_column_names):
        raise ValueError("The lengths of `columns_to_keep` and `new_column_names` must match.")

    new_data_dict = {}
    for key, df in data_dict.items():
        df_copy = df.copy()

        for col in columns_to_keep:
            if col not in df_copy.columns:
                df_copy[col] = default_value  # 添加缺失列并赋默认值

        filtered_df = df_copy[columns_to_keep]

        if new_column_names:
            filtered_df.columns = new_column_names

        new_data_dict[key] = filtered_df

    return new_data_dict




def get_dict_data(input_files, csv_name, value_column_name, filter_column_name=None,
                   condition_column_name=None, condition_value=None, use_parallel=True, n_jobs=-1,unit_adopt=True):
    """
    从多个文件中读取数据并按指定列分组求和，并可根据条件列进行筛选。

    参数:
    - input_files (list): 输入文件的列表。
    - csv_name (str): 目标 CSV 文件的名称前缀（不含年份）。
    - value_column_name (str): 要求和的列名。
    - filter_column_name (str): 过滤列名，根据此列的唯一值进行分组求和。
    - condition_column_name (str or list, optional): 一个或多个筛选列名。
    - condition_value (any or list, optional): 对应的一个或多个筛选值。
    - use_parallel (bool): 是否启用并行处理，默认启用。
    - n_jobs (int): 并行作业数，默认使用所有可用核心。

    返回:
    - dict: 每个输入文件的汇总数据字典，每个文件对应一个 DataFrame。
    """

    def process_single_file(input_name,unit_adopt):
        base_path = get_path(input_name)
        file_list = os.listdir(base_path)

        out_numbers = sorted([
            int(re.search(r"out_(\d+)", filename).group(1))
            for filename in file_list
            if "out_" in filename and re.search(r"out_(\d+)", filename)
        ])

        temp_results = pd.DataFrame(index=out_numbers)
        temp_results.index.name = 'Year'

        for year in out_numbers:
            file_path = os.path.join(base_path, f'out_{year}', f'{csv_name}_{year}.csv')

            if os.path.exists(file_path):
                df = pd.read_csv(file_path)

                # 多列筛选支持
                if condition_column_name and condition_value is not None:
                    if isinstance(condition_column_name, list) and isinstance(condition_value, list):
                        for col, val in zip(condition_column_name, condition_value):
                            df = df[df[col] == val]
                    else:
                        df = df[df[condition_column_name] == condition_value]

                if filter_column_name is None:
                    # 不分组，直接求和
                    if unit_adopt:
                        total_value = df[value_column_name].sum() / 1e6
                    else:
                        total_value = df[value_column_name].sum()
                    temp_results.loc[year, 'Total'] = total_value
                else:
                    unique_values = df[filter_column_name].unique()
                    for value in unique_values:
                        if unit_adopt:
                            total_value = df[df[filter_column_name] == value][value_column_name].sum() / 1e6
                        else:
                            total_value = df[df[filter_column_name] == value][value_column_name].sum()
                        temp_results.loc[year, value] = total_value

        return input_name, temp_results

    if use_parallel:
        results = Parallel(n_jobs=n_jobs)(
            delayed(process_single_file)(input_name,unit_adopt) for input_name in input_files
        )
    else:
        results = [process_single_file(input_name,unit_adopt) for input_name in input_files]

    return dict(results)






def get_dict_sum_data(input_files, csv_name, value_column_name, give_column_name):
    """
    从多个文件中读取数据，对指定列求和，并将结果保存为新的 CSV 文件。

    参数:
    - input_files (list): 输入文件的列表。
    - csv_name (str): 目标 CSV 文件的名称前缀（不含年份）。
    - value_column_name (str): 要求和的列名。
    - give_column_name (str): 保存求和结果的新列名。

    返回:
    - dict: 包含每个输入文件的汇总数据的字典。
    """
    data_dict = {}

    # 遍历每个 input_file 进行处理
    for input_name in input_files:
        # print(f"Processing {input_name}...")
        base_path = get_path(input_name)
        file_list = os.listdir(base_path)  # 确保从当前路径获取文件列表

        # 提取包含年份的文件名中的数字
        out_numbers = sorted([int(re.search(r"out_(\d+)", filename).group(1)) for filename in file_list if
                              "out_" in filename and re.search(r"out_(\d+)", filename)])

        # 创建以提取的年份为索引的 DataFrame
        temp_results = pd.DataFrame(index=out_numbers)
        temp_results.index.name = 'Year'

        # 遍历提取的年份
        for year in out_numbers:
            file_path = os.path.join(base_path, f'out_{year}', f'{csv_name}_{year}.csv')

            # 如果文件存在，进行处理
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)

                # 对指定列求和
                total_value = df[value_column_name].sum() / 1e6  # 将结果单位转换为百万
                temp_results.loc[year, give_column_name] = total_value

        # 保存结果到新的 CSV 文件
        output_file = os.path.join(base_path, f'{csv_name}_summed_results.csv')
        temp_results.to_csv(output_file)

        # 将结果 DataFrame 添加到字典
        data_dict[input_name] = temp_results

    return data_dict

def get_dict_from_json(input_files, json_name):
    data_dict = {}

    # 遍历每个 input_file 进行处理
    for input_name in input_files:
        # 获取输入文件的基本路径
        base_path = get_path(input_name)

        # 创建以年份为索引的 DataFrame
        file_path = os.path.join(base_path, 'DATA_REPORT', 'data', f'{json_name}.json')
        df = pd.read_json(file_path)
        df_expanded = df.explode('data')
        df_expanded[['Year', 'value']] = pd.DataFrame(df_expanded['data'].tolist(), index=df_expanded.index)
        df_transformed = df_expanded.pivot(index='Year', columns='name', values='value')

        # Display the transformed DataFrame
        data_dict[input_name] = df_transformed / 1e6
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
        concatenated_dict[key] = concatenated_df

    return concatenated_dict


# 提取列信息并重新排序
def parse_column_name(col):
    try:
        temp = col.split('_GHG_')[1].split('_BIO_')[0]  # 提取温度目标
        bio = col.split('_BIO_')[1].split('_')[0]  # 提取 BIO 百分比
        return f"{temp}", f"{bio}"  # 返回多级索引 (温度目标, BIO 百分比)
    except IndexError:
        raise ValueError(f"Invalid column format: {col}")


def sort_columns_by_priority(df):
    """
    按照温度目标和 BIO 百分比对 DataFrame 的列进行排序。

    Parameters:
        df (pd.DataFrame): 输入的 DataFrame，列名应为解析后格式的元组。

    Returns:
        pd.DataFrame: 排列好列名的 DataFrame。
    """
    # 解析列名
    parsed_columns = [parse_column_name(col) for col in df.columns]
    order = {'low': 0, 'medium': 1, 'high': 2}

    # 对列按温度目标和百分比排序
    sorted_columns = sorted(parsed_columns, key=lambda x: (order[x[0]], order[x[1]]))

    # 根据排序结果重新排列列
    sorted_indices = [df.columns[list(parsed_columns).index(col)] for col in sorted_columns]
    sorted_df = df[sorted_indices]

    # 设置多级索引
    sorted_df.columns = pd.MultiIndex.from_tuples(sorted_columns, names=["GHG Target", "BIO Percentage"])

    return sorted_df

def merge_transposed_dict(data_dict):
    """
    将字典中的 DataFrame 转置，提取 2050 列，并合并为一个新的 DataFrame，用原来的字典键作为列名。

    Parameters:
        data_dict (dict): 包含 DataFrame 的字典。

    Returns:
        pd.DataFrame: 合并后的 DataFrame，带多级列索引。
    """
    # 提取 2050 列并转置
    transposed_dfs = {}
    missing_keys = []
    for key, df in data_dict.items():
        if 2050 in df.index:
            transposed_dfs[key] = df.loc[2050].T
        else:
            missing_keys.append(key)

    if missing_keys:
        print(f"Warning: The following keys are missing 2050: {missing_keys}")

    # 合并所有 DataFrame
    merged_df = pd.concat(transposed_dfs.values(), axis=1)
    merged_df.columns = transposed_dfs.keys()  # 设置列名为字典的键

    # 调用排序函数
    sorted_df = sort_columns_by_priority(merged_df)

    return sorted_df


def get_lon_lat(tif_path):
    """
    计算面积加权经纬度，并返回重心经纬度。
    这里使用每个像元的值作为权重，对经纬度取加权平均。
    """
    with rasterio.open(tif_path) as dataset:
        data = dataset.read(1)
        transform = dataset.transform

    valid_mask = data > 0
    values = data[valid_mask]
    rows, cols = np.where(valid_mask)
    lon, lat = rasterio.transform.xy(transform, rows, cols)
    lon = np.array(lon)
    lat = np.array(lat)
    centroid_lon = np.sum(lon * values) / np.sum(values)
    centroid_lat = np.sum(lat * values) / np.sum(values)
    return centroid_lon, centroid_lat

def compute_land_use_change_metrics(input_file, use_parallel=True):
    """
    计算土地利用变化指标，包括：
      - 重心移动距离（km）
      - 移动角度（degrees）
      - 面积（单位：ha，根据 CSV 文件中各 Water_supply 的面积求和）
      - 面积×距离
    """
    path = get_path(input_file)
    pattern = re.compile(r'(?<!Non-)Ag_LU.*\.tif{1,2}$', re.IGNORECASE)
    years = list(range(2010, 2051, 5))

    # 获取2050年文件夹中的文件，提取所有 land_use 名称
    folder_path_2050 = os.path.join(path, "out_2050", "lucc_separate")
    sample_files = [f for f in os.listdir(folder_path_2050) if pattern.match(f)]
    names = [f.split("_")[3] for f in sample_files]

    # 初始化存储各年份重心坐标的 DataFrame，列使用 MultiIndex（Land Use, Coordinate）
    coord_columns = pd.MultiIndex.from_product([names, ['Centroid Lon', 'Centroid Lat']],
                                               names=["Land Use", "Coordinate"])
    coord_df = pd.DataFrame(index=years, columns=coord_columns)

    # 遍历每年，计算每个 land_use 的重心经纬度
    for year in years:
        folder_path = os.path.join(path, f"out_{year}", "lucc_separate")
        # 仅处理非 mercator 的 tif 文件
        year_files = [f for f in os.listdir(folder_path) if pattern.match(f) and "mercator" not in f]
        file_paths = [os.path.join(folder_path, f) for f in year_files]
        if use_parallel:
            centroid_results = Parallel(n_jobs=30)(delayed(get_lon_lat)(fp) for fp in file_paths)
        else:
            centroid_results = [get_lon_lat(fp) for fp in file_paths]
        # 构造字典：land_use -> (centroid_lon, centroid_lat)
        year_land_use = {f.split("_")[3]: centroid for f, centroid in zip(year_files, centroid_results)}
        for land_use in names:
            if land_use in year_land_use:
                c_lon, c_lat = year_land_use[land_use]
                coord_df.loc[year, (land_use, 'Centroid Lon')] = c_lon
                coord_df.loc[year, (land_use, 'Centroid Lat')] = c_lat
            else:
                coord_df.loc[year, (land_use, 'Centroid Lon')] = np.nan
                coord_df.loc[year, (land_use, 'Centroid Lat')] = np.nan

    # 构造坐标转换器（假设各年份投影一致，以2050年的文件为例）
    with rasterio.open(os.path.join(folder_path_2050, sample_files[0])) as dataset:
        crs_from = dataset.crs
    crs_to = CRS.from_epsg(3577)
    transformer = Transformer.from_crs(crs_from, crs_to, always_xy=True)

    # 定义结果 DataFrame，多级索引列包含 "Distance (km)"、"Angle (degrees)"、"Area" 和 "Area×Distance"
    metric_columns = pd.MultiIndex.from_product([names, ['Distance (km)', 'Angle (degrees)', 'Area', 'Area×Distance']],
                                                names=["Land Use", "Metric"])
    result_df = pd.DataFrame(index=years, columns=metric_columns)

    # 针对每个年份，从 CSV 文件中获取面积数据（单位：ha）
    for year in years:
        folder_path = os.path.join(path, f"out_{year}", "lucc_separate")
        area_file = os.path.join(path, f"out_{year}", f"area_agricultural_landuse_{year}.csv")
        if os.path.exists(area_file):
            df_area = pd.read_csv(area_file)
            # 按照 "Land-use" 分组，计算各地类所有 Water_supply 的面积之和
            area_group = df_area.groupby("Land-use")["Area (ha)"].sum()
            area_dict = area_group.to_dict()
        else:
            area_dict = {}

        # 对每个 land_use 计算重心移动距离、角度及新增指标“Area×Distance”
        for land_use in names:
            # 获取基准年 2010 的重心
            origin_lon = coord_df.loc[2010, (land_use, 'Centroid Lon')]
            origin_lat = coord_df.loc[2010, (land_use, 'Centroid Lat')]
            if pd.isna(origin_lon) or pd.isna(origin_lat):
                origin_x, origin_y = np.nan, np.nan
            else:
                origin_x, origin_y = transformer.transform(origin_lon, origin_lat)
            # 获取当前年份的重心
            curr_lon = coord_df.loc[year, (land_use, 'Centroid Lon')]
            curr_lat = coord_df.loc[year, (land_use, 'Centroid Lat')]
            if pd.isna(curr_lon) or pd.isna(curr_lat):
                curr_x, curr_y = np.nan, np.nan
            else:
                curr_x, curr_y = transformer.transform(curr_lon, curr_lat)
            # 计算移动距离与角度（当坐标有效时）
            if not (pd.isna(origin_x) or pd.isna(curr_x)):
                distance = np.sqrt((curr_x - origin_x) ** 2 + (curr_y - origin_y) ** 2) / 1000.0
                angle_deg = (np.degrees(np.arctan2(curr_x - origin_x, curr_y - origin_y)) + 360) % 360
            else:
                distance = np.nan
                angle_deg = np.nan

            # 获取面积：单位为 ha，从 CSV 中读取后转换为 Mkm²（1 ha = 1e-8 Mkm²）
            area_val = area_dict.get(land_use, np.nan)
            # 转换面积单位
            area_conv = area_val / 1e8 if not pd.isna(area_val) else np.nan

            # 计算面积×距离（基于转换后的面积和距离）
            area_distance = area_conv * distance if not (pd.isna(area_conv) or pd.isna(distance)) else np.nan

            result_df.loc[year, (land_use, 'Distance (km)')] = round(distance, 3) if not pd.isna(distance) else np.nan
            result_df.loc[year, (land_use, 'Angle (degrees)')] = round(angle_deg, 2) if not pd.isna(
                angle_deg) else np.nan
            result_df.loc[year, (land_use, 'Area (km2)')] = area_conv if not pd.isna(area_conv) else np.nan
            result_df.loc[year, (land_use, 'Area×Distance')] = round(area_distance, 3) if not pd.isna(
                area_distance) else np.nan

    # 假设已经完成了原始DataFrame的计算
    # 使用stack和unstack重塑数据
    flat_df = result_df.stack(level=[0, 1], future_stack=True).reset_index()
    flat_df.columns = ['Year', 'Land Use', 'Metric', 'Value']

    # 然后使用pivot_table重组
    new_result_df = flat_df.pivot_table(
        index='Land Use',
        columns=['Year', 'Metric'],
        values='Value'
    )
    excel_path = os.path.join("..", "output", "12_land_use_movement_all.xlsx")
    # 检查文件是否存在，根据情况选择写入模式
    if os.path.exists(excel_path):
        # 文件存在，使用追加模式
        with pd.ExcelWriter(excel_path, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
            new_result_df.to_excel(writer, sheet_name=input_file)
    else:
        # 文件不存在，创建新文件
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            new_result_df.to_excel(writer, sheet_name=input_file)

def align_columns_with_legend(data_dict, legend_colors):
    """
    确保字典中每个 DataFrame 都包含 legend_colors 中的所有列

    参数:
        data_dict: 包含多个 DataFrame 的字典
        legend_colors: 包含列名的字典

    返回:
        更新后的字典
    """
    required_columns = list(legend_colors.keys())

    for table_name, df in data_dict.items():
        # 找出缺失的列
        missing_cols = set(required_columns) - set(df.columns)

        if missing_cols:
            for col in sorted(missing_cols):
                df[col] = 0
    return data_dict