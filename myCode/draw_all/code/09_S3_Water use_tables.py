import pandas as pd

from tools.data_helper import *
from tools.plot_helper import *
from tools.parameters import *


def create_summary_df(df):
    data = []
    for _, row in df.iterrows():
        # 获取 Water Net Yield 的 2050 值
        yield_2050 = next(
            value
            for entry in row['data']
            if entry['name'] == "Water Net Yield"
            for year, value in entry['data']
            if year == 2050
        )

        # 获取 Historical Limit 的 2050 值
        limit = next(
            value
            for entry in row['data']
            if entry['name'] == "Historical Limit"
            for year, value in entry['data']
            if year == 2050
        )

        # 计算差值
        difference = yield_2050 - limit

        # 添加数据行
        data.append({
            'Region Name': row['name'],
            'Difference': difference / 1e6
        })

    # 创建 DataFrame
    summary_df = pd.DataFrame(data)
    return summary_df

json_name = 'water_3_water_net_yield_by_region'
summary_dict = {}
for input_name in input_files:
    # 获取输入文件的基本路径
    base_path = get_path(input_name)

    # 创建以年份为索引的 DataFrame
    file_path = os.path.join(base_path, 'DATA_REPORT', 'data', f'{json_name}.json')
    df = pd.read_json(file_path)
    summary_df = create_summary_df(df)

    # 将 summary_df 的数据存储到字典中，键为 input_name
    summary_dict[input_name] = summary_df.set_index("Region Name")["Difference"]

# 合并所有 summary_df 的 Difference 列为一个 DataFrame
merged_summary_df = pd.DataFrame(summary_dict)
df_result = sort_columns_by_priority(merged_summary_df)
df_result.round(2).to_excel('../output/09_S3_water_summary.xlsx')