import pandas as pd
import sys

from tools.data_helper import *
from tools.plot_helper import *
from tools.parameters import *

# 获取到文件的绝对路径，并将其父目录添加到 sys.path
sys.path.append(os.path.abspath('../../../luto'))

# 导入 settings.py
import settings

csv_name, value_column_name, filter_column_name = 'quantity_comparison', 'Abs_diff (tonnes, KL)', 'Commodity'
demand_dict = get_dict_data(input_files, csv_name, value_column_name, filter_column_name)
demand_ag_group_dict,legend_colors = get_colors(demand_dict, 'tools/land use colors.xlsx', sheet_name='food')
result_df = merge_transposed_dict(demand_ag_group_dict)

out_path = '../output/08_S2_Food_demand_Abs_diff.xlsx'
result_df.round(2).to_excel(out_path)

csv_name, value_column_name, filter_column_name = 'quantity_comparison', 'Prop_diff (%)', 'Commodity'
demand_dict = get_dict_data(input_files, csv_name, value_column_name, filter_column_name)
demand_ag_group_dict,legend_colors = get_colors(demand_dict, 'tools/land use colors.xlsx', sheet_name='food')
demand_ag_group_dict = {key: df * 1e6 for key, df in demand_ag_group_dict.items()}
result_df = merge_transposed_dict(demand_ag_group_dict)

out_path = '../output/08_S2_Food_demand_Prop_diff.xlsx'
result_df.round(2).to_excel(out_path)