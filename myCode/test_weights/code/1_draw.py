import os
import pandas as pd
from tools.data_helper import process_data
from tools.plot_helper import create_combined_plot

df_runs = pd.read_csv('../../tasks_run/Custom_runs/setting_0410_test_weight1.csv')
file_names = [name.replace('.', '_') for name in df_runs.columns[2:]]
output_dir = '../output'

for path_name in file_names:
    try:
        # 1) 数据处理
        df_all = process_data(path_name)
        # 2) 导出 Excel
        df_all.to_excel(os.path.join(output_dir, f'{path_name}.xlsx'), index=False)
        # 3) 画图（并打印或保存）
        plot_obj = create_combined_plot(df_all, path_name)
        print(plot_obj)
    except Exception as e:
        # 如果出错，就打印信息并跳过
        print(f"Skipping '{path_name}' due to error: {e}")
        continue
