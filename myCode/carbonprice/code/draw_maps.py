from tools.tools import get_path,npy_to_map
from tools.helper_plot import *
from tools.helper_map import plot_bivariate_rgb_map
import os

proj_file = os.path.join(get_path(input_files[0]), 'out_2050', 'lumap_2050.tiff')
for input_file in input_files:
    arr_path = os.path.join(get_path(input_file), "out_2050","data_for_carbon_price")
    arr_names = os.listdir(arr_path)

    for arr_name in arr_names:
        input_arr = os.path.join(arr_path, arr_name)
        output_tif = input_arr.replace('.npy', '_restored.tif')
        npy_to_map(input_arr, output_tif, proj_file)

    arr_path = os.path.join(get_path(input_file), "data_for_carbon_price")
    arr_names = os.listdir(arr_path)
    for arr_name in arr_names:
        input_arr = os.path.join(arr_path, arr_name)
        output_tif = input_arr.replace('.npy', '_restored.tif')
        npy_to_map(input_arr, output_tif, proj_file)