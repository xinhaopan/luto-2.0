from tools.data_helper import process_data
from tools.plot_helper import create_combined_plot

path_name = 'test_Run_1_0_5_0'
df_all = process_data(path_name)
df_all.to_excel(f'../output/{path_name}.xlsx', index=False)
plot_obj = create_combined_plot(df_all, path_name)
print(plot_obj)