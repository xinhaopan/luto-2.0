from tools import *
from helper import *

INPUT_NAMEs = ['ON_MAXPROFIT_GHG_18C_67_R5']
source_name = "non_ag_2050" # lumap_2050, ammap_2050, non_ag_2050
colors_name = "non_ag_colors.csv" # lumap_colors_grouped.csv, ammap_colors.csv,non_ag_colors.csv

for source_dir in INPUT_NAMEs:
    # 判断是否是 'ON_MAXPROFIT_GHG_18C_67_R5'，设置 add_legend
    add_legend_value = True if source_dir == INPUT_NAMEs[-1] else False

    draw_maps(
        overlay_geo_tif=get_path(source_dir, source_name),
        colors_csv=f"Assets/{colors_name}",
        output_png=f"{source_dir}_{source_name}.png",
        add_legend=add_legend_value,  # 动态设置 add_legend
        delete_intermediate_files=False,
        write_png=False,
        legend_png_name=f"{source_name}_legend.png",

        legend_title="Legend",
        legend_title_fontsize=20,
        legend_fontsize=18,
        legend_location=(0.01, 0.01),
        legend_ncol=1,
        legend_figsize=(20.5, 1),
        scalebar_fontsize=18,
    )
