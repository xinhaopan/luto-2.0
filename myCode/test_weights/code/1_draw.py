import os

from tools.data_helper import process_data
from tools.plot_helper import create_combined_plot

file_path = '../../../output'
file_names = os.listdir(file_path)
for path_name in file_names:
    df_all = process_data(path_name)
    df_all.to_excel(f'../output/{path_name}.xlsx', index=False)
    plot_obj = create_combined_plot(df_all, path_name)
    print(plot_obj)