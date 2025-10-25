import os
import pandas as pd
from collections import namedtuple
import tools.config as config
from tools.helper_plot import set_plot_style,xarray_to_dict, get_global_ylim, plot_13_layout,get_colors, plot_25_layout

def process_profit_dict(data_dict):
    for k, df in data_dict.items():
        # 先移除名为 'Transition(ag2non→ag) cost' 的列（如果存在）
        if 'Transition(ag→non-ag) cost' in df.columns:
            df = df.drop(columns=['Transition(ag→non-ag) cost'])
        # 找到所有列名包含 "cost" 的列
        cost_cols = [col for col in df.columns if "cost" in col]
        # 将这些列取负值
        df[cost_cols] = df[cost_cols] * -1
        # 新建 Total 列，为每行所有列的和
        df['Net economic return'] = df.sum(axis=1)
        # 更新字典
        data_dict[k] = df

    return data_dict

def add_targets_line(axes):
    df_path = '../../../input/GHG_targets.xlsx'
    df = pd.read_excel(df_path, sheet_name="Data", index_col="YEAR")
    cols = [
        '1.5C (67%) excl. avoided emis SCOPE1',
        '1.8C (67%) excl. avoided emis SCOPE1'
    ]

    new_df = df[cols] / 1e6
    new_df.columns = ['high', 'low']
    new_df = new_df[(new_df.index >= config.START_YEAR) & (new_df.index <= 2050)]
    # 需要画 high 的子图 index
    high_axes = [2, 8, 9,10, 11, 12]
    # 需要画 low 的子图 index
    low_axes = [1, 3, 4, 5, 6, 7]

    # 统一样式：藏青色、实线、无点
    line_kw = dict(color='#0078d4', linestyle='-', linewidth=3, zorder=60)

    # high 子图
    for idx in high_axes:
        axes[idx].plot(new_df.index, new_df['high'], **line_kw, label='_nolegend_')

    # low 子图
    for idx in low_axes:
        axes[idx].plot(new_df.index, new_df['low'], **line_kw, label='_nolegend_')

    # 图例只在最后一个轴上保留
    axes[-1].plot(new_df.index, new_df['high'], **line_kw, label='GHG emissions targets')

    return axes


# Main script
set_plot_style(font_size=20, font_family='Arial')

task_name = config.TASK_NAME
input_dir = f'../../../output/{task_name }/carbon_price/1_draw_data'
output_dir = f"../../../output/{task_name }/carbon_price/3_Paper_figure"

data_dict = xarray_to_dict(f"{input_dir}/xr_cost_for_profit.nc",1e3)
data_dict = process_profit_dict(data_dict)

data_dict, colors = get_colors(data_dict,'tools/land use colors.xlsx',sheet_name='cost_revenue',total_name='Net economic return')
summary_ylim = get_global_ylim(data_dict)

output_path = os.path.join(output_dir, '03_Profit.png')
plot_13_layout(data_dict,config.ORIGINAL_TITLE_MAP,colors,output_path,summary_ylim,bbox_to_anchor=[0.63, 0.80, 0.4, 0.1],dividing_line=1,column_spacing=-7,total_name='Net economic return')
#
PlotSpec = namedtuple("PlotSpec", ["file", "sheet", "ylabel","total_name","bbox_to_anchor","post_process"])
#
# # 把三元组写在一起，避免对不上的问题
plot_specs = [
    PlotSpec("xr_total_carbon", "carbon_total", r"GHG emissions (MtCO$_2$e yr$^{-1}$)", 'Total',[0.63, 0.80, 0.4, 0.1],add_targets_line),
    # PlotSpec("xr_total_bio", "biodiversity_total", r"Biodiversity (contribution-weighted area, Mha yr$^{-1}$)", 'Total',[0.63, 0.80, 0.4, 0.1],None),
    PlotSpec("xr_area_agricultural_management", "am", r"Area (Mha yr$^{-1}$)", None,[0.63, 0.80, 0.4, 0.1],None),
    PlotSpec("xr_area_non_agricultural_landuse", "non_ag", r"Area (Mha yr$^{-1}$)", None,[0.63, 0.82, 0.4, 0.1],None),
    # PlotSpec("xr_biodiversity_GBF2_priority_ag_management", "am", r"Biodiversity (contribution-weighted area, Mha yr$^{-1}$)", None,[0.63, 0.80, 0.4, 0.1],None),
    # PlotSpec("xr_biodiversity_GBF2_priority_non_ag", "non_ag", r"Biodiversity (contribution-weighted area, Mha yr$^{-1}$)", None,[0.63, 0.82, 0.4, 0.1],None),
    PlotSpec("xr_GHG_ag_management", "am", r"GHG emissions (MtCO$_2$e yr$^{-1}$)", None,[0.63, 0.80, 0.4, 0.1],None),
    PlotSpec("xr_GHG_non_ag", "non_ag", r"GHG emissions (MtCO$_2$e yr$^{-1}$)", None,[0.63, 0.84, 0.4, 0.1],None)
]

for spec in plot_specs:
    input_file, sheet_name,ylabel,total_name,bbox_to_anchor,post_process = spec.file, spec.sheet, spec.ylabel, spec.total_name, spec.bbox_to_anchor,spec.post_process
    print(input_file)
    data_dict = xarray_to_dict(f"{input_dir}/{input_file}_original.nc",1,total_name=total_name)
    data_dict, colors = get_colors(data_dict,'tools/land use colors.xlsx',sheet_name=sheet_name)
    summary_ylim = get_global_ylim(data_dict)

    output_path = os.path.join(output_dir, f'03_{input_file}.png')
    plot_13_layout(
        all_dfs=data_dict,
        title_map=config.ORIGINAL_TITLE_MAP,
        colors=colors,
        output_path=output_path,
        summary_ylim=summary_ylim,
        total_name=total_name,
        ylabel=ylabel,
        bbox_to_anchor = bbox_to_anchor,
        ncol = 1,
        ghost_legend_num=0,
        post_process=post_process
    )

plot_specs = [
    PlotSpec("xr_total_bio", "biodiversity_total", r"Biodiversity (contribution-weighted area, Mha yr$^{-1}$)", 'Total',[0.25, -0.068, 0.4, 0.1],None),
    PlotSpec("xr_biodiversity_GBF2_priority_ag_management", "am", r"Biodiversity (contribution-weighted area, Mha yr$^{-1}$)", None,[0.12, -0.068, 0.4, 0.1],None),
    PlotSpec("xr_biodiversity_GBF2_priority_non_ag", "non_ag", r"Biodiversity (contribution-weighted area, Mha yr$^{-1}$)", None,[0.12, -0.068, 0.4, 0.1],None),
]

for spec in plot_specs:
    input_file, sheet_name,ylabel,total_name,bbox_to_anchor,post_process = spec.file, spec.sheet, spec.ylabel, spec.total_name, spec.bbox_to_anchor,spec.post_process
    print(input_file)
    data_dict = xarray_to_dict(f"{input_dir}/{input_file}_original.nc",1,total_name=total_name)
    data_dict, colors = get_colors(data_dict,'tools/land use colors.xlsx',sheet_name=sheet_name)
    summary_ylim = get_global_ylim(data_dict)

    output_path = os.path.join(output_dir, f'03_{input_file}.png')
    n_col = 4
    if input_file == "xr_biodiversity_GBF2_priority_non_ag":
        n_col = 3
    plot_25_layout(
        all_dfs=data_dict,
        title_map=config.ALL_TITLE_MAP,
        colors=colors,
        output_path=output_path,
        summary_ylim=summary_ylim,
        total_name=total_name,
        ylabel=ylabel,
        bbox_to_anchor = bbox_to_anchor,
        ncol = n_col,
        ghost_legend_num=0,
        post_process=post_process,
        y_labelpad=20
    )