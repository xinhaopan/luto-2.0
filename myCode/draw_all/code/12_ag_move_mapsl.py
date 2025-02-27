from tools.map_tools import *
from tools.map_helper import *
from tools.parameters import *
from tools.data_helper import *
from tools.config import *

from joblib import Parallel, delayed

cfg = MapConfig(input_name=input_files[4], source_name="Ag_LU_06_Dairy - natural land_2050.png")
folder_path = os.path.join(get_path(input_files[4]), f"out_2050", "lucc_separate","Ag_LU_06_Dairy - natural land_2050.tiff")
output_png = f"../output/Ag_LU_06_Dairy - natural land_2050.png"
legend_png_name = f'../output/Ag_LU_06_Dairy - natural land_legend.png'
draw_proportion_maps(
        overlay_geo_tif=folder_path,
        output_png=output_png,
        legend_png_name=legend_png_name,
        cfg=cfg
    )

print("Finish all mapping.")
