from tools.helper_map import plot_bivariate_rgb_map
from tools.helper_plot import *

# plot_bivariate_rgb_map(
#     input_file="20250407_Run_4_on_on_20",
#     arr_path1="cost_2050.npy",
#     arr_path2="bio_2050.npy",
#     output_png="../output/carbon_bio_rgb_map.png",
#     proj_file="lumap_2050.tiff",
#     quantile=0.005,
# )

# plot_combined_with_facets(input_file)
for input_file in input_files:
    plot_cost_vs_price(input_file)
    plot_absolute_cost_stack(input_file)
    plot_ghg_stack(input_file)