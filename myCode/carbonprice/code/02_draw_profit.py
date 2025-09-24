import os
import tools.config as config
from tools.helper_plot import set_plot_style,xarray_to_dict, get_global_ylim, plot_13_layout,get_colors

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

task_name = config.TASK_NAME
input_dir = f'../../../output/{task_name }/carbon_price/1_draw_data'
output_dir = f"../../../output/{task_name }/carbon_price/3_Paper_figure"

data_dict = xarray_to_dict(f"{input_dir}/xr_cost_for_profit.nc",1e3)
data_dict = process_dict(data_dict)

data_dict, colors = get_colors(data_dict,'tools/land use colors.xlsx',sheet_name='cost_revenue')
summary_ylim = get_global_ylim(data_dict)

output_path = os.path.join(output_dir, '03_Profit.png')
plot_13_layout(data_dict,config.ORIGINAL_TITLE_MAP,colors,output_path,summary_ylim,bbox_to_anchor=[0.58, 0.82, 0.4, 0.1],dividing_line=1,column_spacing=-7)

input_files = ['xr_total_carbon','xr_total_bio','xr_area_agricultural_management','xr_area_non_agricultural_landuse',
             'xr_biodiversity_GBF2_priority_ag_management','xr_biodiversity_GBF2_priority_non_ag',
             'xr_GHG_ag_management','xr_GHG_non_ag']
sheet_names = ["cost","cost",'am','non_ag','am','non_ag','am','non_ag']
ylabels = [r"GHG emission (MtCO$_2$e yr$^{-1}$)",r"Biodiversity (Mha yr$^{-1}$)",r"Area (Mha yr$^{-1}$)",r"Area (Mha yr$^{-1}$)",
           r"Biodiversity (Mha yr$^{-1}$)",r"Biodiversity (Mha yr$^{-1}$)",
            r"GHG emission (MtCO$_2$e yr$^{-1}$)",r"GHG emission (MtCO$_2$e yr$^{-1}$)"]

for input_file, sheet_name,ylabel in zip(input_files, sheet_names,ylabels):
    print(input_file)
    data_dict = xarray_to_dict(f"{input_dir}/{input_file}_original.nc",1,add_total=True)
    data_dict, colors = get_colors(data_dict,'tools/land use colors.xlsx',sheet_name=sheet_name)
    summary_ylim = get_global_ylim(data_dict)

    output_path = os.path.join(output_dir, f'03_{input_file}.png')
    plot_13_layout(
        all_dfs=data_dict,
        title_map=config.ORIGINAL_TITLE_MAP,
        colors=colors,
        output_path=output_path,
        summary_ylim=summary_ylim,
        ylabel=ylabel,
        bbox_to_anchor = [0.58, 0.87, 0.4, 0.1],
        ncol = 1,
        ghost_legend_num=0
    )
