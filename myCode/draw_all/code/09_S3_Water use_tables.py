import pandas as pd

from tools.data_helper import *
from tools.plot_helper import *
from tools.parameters import *


def compute_2050_water_difference_series(water_yield_dict, water_public_dict, water_limit_dict):
    result_dict = {}

    common_keys = (
        set(water_yield_dict.keys()) &
        set(water_public_dict.keys()) &
        set(water_limit_dict.keys())
    )

    for key in common_keys:
        yield_df = water_yield_dict[key]
        public_df = water_public_dict[key]
        limit_df = water_limit_dict[key]

        # 确保2050存在
        if 2050 not in yield_df.index or 2050 not in public_df.index or 2050 not in limit_df.index:
            continue

        # 提取2050年行
        yield_2050 = yield_df.loc[2050]
        public_2050 = public_df.loc[2050]
        limit_2050 = limit_df.loc[2050]

        # 正确求交集
        common_columns = yield_2050.index.intersection(public_2050.index).intersection(limit_2050.index)

        # 计算差值
        diff_series = yield_2050[common_columns] + public_2050[common_columns] - limit_2050[common_columns]
        diff_series.name = "Difference"
        diff_series.index.name = "Region Name"

        result_dict[key] = diff_series

    return result_dict



csv_name, value_column_name, filter_column_name = 'water_yield_separate_watershed', 'Water Net Yield (ML)', 'Region'
water_yield_dict = get_dict_data(input_files, csv_name, value_column_name, filter_column_name)
water_public_dict = get_dict_data(input_files, 'water_yield_limits_and_public_land',
                                 'Water yield outside LUTO (ML)',filter_column_name)

water_limit_dict = get_dict_data(input_files, 'water_yield_limits_and_public_land',
                                 'Water Yield Limit (ML)', filter_column_name)
summary_dict = compute_2050_water_difference_series(water_yield_dict, water_public_dict, water_limit_dict)

# 合并所有 summary_df 的 Difference 列为一个 DataFrame
merged_summary_df = pd.DataFrame(summary_dict)
df_result = sort_columns_by_priority(merged_summary_df)
df_result.round(2).to_excel('../output/09_S3_water_summary.xlsx')