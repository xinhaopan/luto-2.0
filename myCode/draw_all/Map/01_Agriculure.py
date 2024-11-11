from tools import *
import config as cfg

INPUT_NAMEs = ['ON_MAXPROFIT_GHG_15C_67_R5', 'ON_MAXPROFIT_GHG_15C_50_R5', 'ON_MAXPROFIT_GHG_18C_67_R5']
source_name = "lumap_2050"
colors_name = "lumap_colors_grouped.csv" # lumap_colors_grouped.csv

for source_dir in INPUT_NAMEs:
    draw_maps(
        overlay_geo_tif=get_path(source_dir, source_name),
        colors_csv=f"Assets/{colors_name}",
        output_png=f"{source_dir}_{source_name}.png",
    )
    break