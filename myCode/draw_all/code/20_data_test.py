

def filter_dfs_by_substring(group_dict, substring):
    """
    根据主键名称筛选包含指定子字符串的 DataFrame。

    :param group_dict: 包含多个 DataFrame 的字典。
    :param substring: 用于匹配的主键子字符串。如果为 "*"，使用所有 DataFrame。
                      如果为列表，则匹配列表中的所有子字符串。
    :return: 筛选后的 DataFrame 字典。
    """
    if substring == "*":
        return group_dict

    if isinstance(substring, list):
        return {name: df for name, df in group_dict.items() if any(sub in name for sub in substring)}

    return {name: df for name, df in group_dict.items() if substring in name}


def find_global_min_max(filtered_dict="*", target_column="*", index_filter="*", decimal_places=2, ):
    """
    在筛选后的所有 DataFrame 中，找出每列的全局最大值和最小值，以及它们的来源。

    :param filtered_dict: 筛选后的 DataFrame 字典。
    :param target_column: 计算最大值和最小值的目标列。如果为 "*"，比较所有列。
    :param decimal_places: 保留的小数位，默认两位。
    :param index_filter: 用于筛选的索引值。如果为 "*"，使用所有索引。
    :return: DataFrame 包含每列的全局最大值和最小值及其来源。
    """
    global_results = {}

    # 如果 target_column 为 "*"，比较所有列
    for name, df in filtered_dict.items():
        # 根据 index_filter 筛选
        if index_filter != "*":
            if index_filter not in df.index:
                print(f"Skipping {name}: Index '{index_filter}' not found.")
                continue
            df = df.loc[[index_filter]]

        columns_to_check = df.columns if target_column == "*" else [target_column]

        for column in columns_to_check:
            if column not in df.columns:
                print(f"Skipping {name}: Target column '{column}' not found.")
                continue

            max_value = df[column].max()
            min_value = df[column].min()

            max_index = df[column].idxmax()
            min_index = df[column].idxmin()

            # 更新全局最大值
            if column not in global_results or max_value > global_results[column].get("Max", {}).get("Value",
                                                                                                     float('-inf')):
                global_results[column] = global_results.get(column, {})
                global_results[column]["Max"] = {
                    "Source": name,
                    "Value": round(max_value, decimal_places),
                    "Index": max_index
                }

            # 更新全局最小值
            if column not in global_results or min_value < global_results[column].get("Min", {}).get("Value",
                                                                                                     float('inf')):
                global_results[column] = global_results.get(column, {})
                global_results[column]["Min"] = {
                    "Source": name,
                    "Value": round(min_value, decimal_places),
                    "Index": min_index
                }

    # 整理结果为 DataFrame
    results = []
    for column, stats in global_results.items():
        results.append({
            "Statistic": "Min",
            "Source": stats["Min"]["Source"],
            "Column": column,
            "Index": stats["Min"]["Index"],
            "Value": stats["Min"]["Value"]
        })
        results.append({
            "Statistic": "Max",
            "Source": stats["Max"]["Source"],
            "Column": column,
            "Index": stats["Max"]["Index"],
            "Value": stats["Max"]["Value"]
        })

    result_df = pd.DataFrame(results, columns=["Statistic", "Source", "Column", "Index", "Value"])
    result_df["Value"] = result_df["Value"].map(lambda x: f"{x:.{decimal_places}f}")
    return result_df


import pandas as pd
import re


def add_value_label_column(df, unit):
    def extract_labels(source):
        match = re.search(r'GHG_([A-Za-z]+)_BIO_([A-Za-z]+)', source)
        if match:
            ghg, bio = match.groups()
            return f"{ghg}GHG, {bio}BIO"
        else:
            return "Unknown"

    # 创建新列
    df['Value Label'] = df.apply(lambda row: f"{row['Value']} {unit} ({extract_labels(row['Source'])})", axis=1)

    # 重新排列列顺序
    desired_order = ['Statistic', 'Value Label', 'Column', 'Value', 'Index', 'Source']
    df = df[desired_order]

    return df



import pandas as pd


def extract_land_use_stats(result_file, filter_str=None):
    """
    从Excel文件中提取每种土地利用类型在2050年的最大值和最小值，并找到对应的工作表名称。
    额外计算一级列名是2050，二级列名是'Area×Distance'的最大值和最小值。

    参数:
    result_file (str): Excel文件路径。
    filter_str (str, optional): 过滤包含此字符串的工作表。如果未提供，则比较全部工作表。

    返回:
    pd.DataFrame: 包含每种土地利用类型最大值和最小值及其对应工作表名称的DataFrame。
    """
    # 读取Excel文件中的所有工作表
    all_sheets = pd.read_excel(result_file, sheet_name=None, header=[0, 1])

    # 存储结果的列表
    results = []

    # 遍历每个工作表
    for sheet_name, df in all_sheets.items():
        # 如果提供了过滤字符串并且当前工作表名称不包含该字符串，则跳过该工作表
        if filter_str and filter_str not in sheet_name:
            continue

        # 确保列索引是MultiIndex
        if isinstance(df.columns, pd.MultiIndex):
            # 过滤出2050年的数据
            year_2050_cols = [col for col in df.columns if col[0] == 2050]
            if year_2050_cols:
                df_2050 = df.loc[:, year_2050_cols]
                # 获取土地利用类型
                land_use_column = df[('Year', 'Metric')]
                df_2050 = df_2050.copy()
                df_2050['Land Use'] = land_use_column

                # 对于每种土地利用类型，找到最大值和最小值
                for land_use in df_2050['Land Use'].unique():
                    df_land_use = df_2050[df_2050['Land Use'] == land_use]

                    # 仅选择数值列进行计算
                    numeric_cols = df_land_use.select_dtypes(include='number')

                    max_value = numeric_cols.max().max()
                    min_value = numeric_cols.min().min()

                    # 计算 'Area×Distance' 的最大值和最小值
                    area_distance_col = (2050, 'Area×Distance')
                    if area_distance_col in df_land_use.columns:
                        max_area_distance = df_land_use[area_distance_col].max()
                        min_area_distance = df_land_use[area_distance_col].min()
                    else:
                        max_area_distance = None
                        min_area_distance = None

                    # 将结果添加到列表中
                    results.append({
                        'Land Use': land_use,
                        'Max Value': max_value,
                        'Max Sheet Name': sheet_name,
                        'Min Value': min_value,
                        'Min Sheet Name': sheet_name,
                        'Max Area×Distance': max_area_distance,
                        'Min Area×Distance': min_area_distance
                    })

    # 将结果转换为DataFrame
    results_df = pd.DataFrame(results)
    return results_df

result_file = "../output/12_land_use_movement_all.xlsx"
filter_str = None
results_df = extract_land_use_stats(result_file, filter_str)