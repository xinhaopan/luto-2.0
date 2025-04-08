from joblib import Parallel, delayed
from tools.config import n_jobs, input_files
from tools import run_analysis_pipeline

# Parallel(n_jobs=n_jobs)(
#     delayed(run_analysis_pipeline)(input_file, True) for input_file in input_files
# )

plot_bivariate_rgb_map(
    input_file="20250407_Run_4_on_on_20",
    arr_path1="cp_2050.npy",
    arr_path2="bp_2050.npy",
    output_png="../outputs/carbon_bio_rgb_map.png",
    proj_file="ammap_2050.tiff"
)

