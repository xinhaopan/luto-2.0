import os
import tools.config as config
from tools.helper_plot import set_plot_style,xarray_to_dict, get_global_ylim, plot_13_layout

def process_dict(data_dict):
    for k, df in data_dict.items():
        # 先移除名为 'Transition(ag2non→ag) cost' 的列（如果存在）
        if 'Transition(ag→non-ag) cost' in df.columns:
            df = df.drop(columns=['Transition(ag→non-ag) cost'])
        # 找到所有列名包含 "cost" 的列
        cost_cols = [col for col in df.columns if "cost" in col]
        # 将这些列取负值
        df[cost_cols] = df[cost_cols] * -1
        # 新建 Total 列，为每行所有列的和
        df['Total'] = df.sum(axis=1)
        # 更新字典
        data_dict[k] = df
    return data_dict

# Main script
set_plot_style(font_size=24, font_family='Arial')

# task_name = config.TASK_NAME
task_name = '20250922_Paper2_Results_HPC_test'
input_dir = f'../../../output/{task_name }/carbon_price/1_draw_data'
output_dir = f"../../../output/{task_name }/carbon_price/3_Paper_figure"

data_dict = xarray_to_dict(f"{input_dir}/xr_cost_for_profit.nc",1e3)
data_dict = process_dict(data_dict)
summary_ylim = get_global_ylim(data_dict)

colors = ['#fab431', '#ec7951', '#cd4975', '#9f0e9e', '#6200ac', '#2d688f', '#19928e', '#35b876']
output_path = os.path.join(output_dir, '03_Profit.png')
plot_13_layout(data_dict,config.ORIGINAL_TITLE_MAP,colors,output_path,summary_ylim,bbox_to_anchor=[0.58, 0.82, 0.4, 0.1],dividing_line=1)