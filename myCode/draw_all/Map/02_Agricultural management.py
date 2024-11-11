from tools import *

INPUT_NAMEs = ['ON_MAXPROFIT_GHG_15C_67_R5', 'ON_MAXPROFIT_GHG_15C_50_R5', 'ON_MAXPROFIT_GHG_18C_67_R5']
source_name = "ammap_2050" # lumap_2050, ammap_2050, non_ag_2050
colors_name = "ammap_colors.csv" # lumap_colors_grouped.csv, ammap_colors.csv,non_ag_colors.csv

for source_dir in INPUT_NAMEs:
    draw_maps(
        overlay_geo_tif=get_path(source_dir, source_name),
        colors_csv=f"Assets/{colors_name}",
        output_png=f"{source_dir}_{source_name}.png",
        legend_title="", legend_fontsize=18,
        legend_title_fontsize=20, legend_ncol=1, legend_figsize=(20.5, 1),
        add_legend=False,
        delete_intermediate_files=False,
        write_png=False,
        legend_png_name=f"{source_name}_legend.png"
    )