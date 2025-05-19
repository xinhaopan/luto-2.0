from tools.helper_map import plot_bivariate_rgb_map, npy_to_map
from tools.helper_plot import *
from tools.config import input_files,suffix
from tools.helper_data import *


# compute_zonal_stats_to_shp(
#     shapefile_path="../Hexagon/Hexagonal100.shp",
#     raster_paths={
#         "ghg": "../data/ghg_2050.tif",
#         "carbon_cost": "../data/carbon_cost_2050.tif",
#         "bio": "../data/bio_2050.tif",
#         "bio_cost": "../data/bio_cost_2050.tif"
#     },
#     output_shp_path="../Hexagon/zonal_stats_result.shp",
#     id_field="id"
# )

rasterize_fields_from_shp(
    shapefile_path="../Hexagon/zonal_stats_result.shp",
    output_dir="../Hexagon",
    fields=[
        "ghg_sum", "carbon_cos", "bio_sum", "bio_cost_s",
        "carbon_pri", "bio_price"
    ],
    resolution=5000
)