from helper import *
import config as cfg
from tools import *
from parameters import *


cfg.write_png = False
cfg.intermediate_files = False
cfg.add_legend = False
cfg.north_arrow_zoom = 0.3
cfg.north_arrow_location = (0.9, 0.85)
cfg.scalebar_fontsize = 18

for INPUT_NAME, source_name, colors_name in tasks:
    # 生成输出文件名和颜色文件路径
    colors_csv = f"Assets/{colors_name}"
    output_png = f"{INPUT_NAME}_{source_name}.png"
    legend_png_name = f'{source_name}_legend.png'

    # 调用绘图函数
    draw_maps(
        overlay_geo_tif=get_path(INPUT_NAME, source_name),
        colors_csv=colors_csv,
        output_png=output_png,
        legend_png_name = legend_png_name
    )
