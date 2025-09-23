import os
import tools.config as config
from tools.helper_plot import set_plot_style,xarray_to_dict, get_global_ylim, plot_22_layout, get_colors

set_plot_style(font_size=20, font_family='Arial')

task_name = config.TASK_NAME
# task_name = '20250922_Paper2_Results_HPC_test'
input_dir = f'../../../output/{task_name }/carbon_price/1_draw_data'
output_dir = f"../../../output/{task_name }/carbon_price/3_Paper_figure"

input_files = ['xr_total_cost', 'xr_cost_agricultural_management', 'xr_cost_non_ag', 'xr_transition_cost_ag2non_ag_amortised_diff',
               'xr_total_carbon','xr_total_bio','xr_GHG_ag_management', 'xr_GHG_non_ag',
               'xr_biodiversity_GBF2_priority_ag_management','xr_biodiversity_GBF2_priority_non_ag']
sheet_names = ['cost','am','non_ag','non_ag',
               'cost','cost','am','non_ag',
               'am','non_ag']
scales = [1e3,1e3,1e3,1e3,1,1,1,1,1,1]  # cost 需要乘 1e3 转为 billion
ylabels = [r"Cost (Billion AU\$ yr$^{-1}$)",r"Cost (Billion AU\$ yr$^{-1}$)",r"Cost (Billion AU\$ yr$^{-1}$)",r"Cost (Billion AU\$ yr$^{-1}$)",
           r"Carbon benefit (MtCO$_2$e yr$^{-1}$)",r"Biodiversity benefit (Mha yr$^{-1}$)",r"Carbon benefit (MtCO$_2$e yr$^{-1}$)",r"Carbon benefit (MtCO$_2$e yr$^{-1}$)",
            r"Biodiversity benefit (Mha yr$^{-1}$)",r"Biodiversity benefit (Mha yr$^{-1}$)"]
column_spacings = [1,1,-3,-3,1,1,1,-3,1,-3]

for input_file, sheet_name,scale,ylabel,column_spacing in zip(input_files, sheet_names,scales,ylabels,column_spacings):
    data_dict = xarray_to_dict(f"{input_dir}/{input_file}.nc",scale,add_total=True)
    data_dict, colors = get_colors(data_dict,'tools/land use colors.xlsx',sheet_name=sheet_name)
    summary_ylim = get_global_ylim(data_dict)

    output_path = os.path.join(output_dir, f'04_{input_file}.png')
    plot_22_layout(
        all_dfs=data_dict,
        title_map=config.PRICE_TITLE_MAP,
        colors=colors,
        output_path=output_path,
        summary_ylim=summary_ylim,
        bbox_to_anchor=(0.42, 0.89),
        ylabel=ylabel,
        column_spacing=column_spacing
    )

