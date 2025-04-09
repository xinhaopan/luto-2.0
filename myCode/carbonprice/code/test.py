from tools.helper_map import plot_bivariate_rgb_map
from tools.helper_plot import plot_cost_vs_price, plot_absolute_cost_stack, plot_ghg_stack

# plot_bivariate_rgb_map(
#     input_file="20250407_Run_4_on_on_20",
#     arr_path1="cost_2050.npy",
#     arr_path2="bio_2050.npy",
#     output_png="../output/carbon_bio_rgb_map.png",
#     proj_file="lumap_2050.tiff",
#     quantile=0.005,
# )

input_file = "20250408_Run_4_on_on_20"
plot_cost_vs_price(input_file)
plot_absolute_cost_stack(input_file)
plot_ghg_stack(input_file)