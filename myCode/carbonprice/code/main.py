from joblib import Parallel, delayed
from tools.config import n_jobs, input_files,suffix
from tools import run_analysis_pipeline, draw_plots
from tools.helper_plot import merge_png_images,plot_all_columns,plot_revenue_cost_faceted
from tools.helper_data import calculate_bio_price

Parallel(n_jobs=n_jobs)(
    delayed(lambda input_file: (run_analysis_pipeline(input_file, True), draw_plots(input_file)))(input_file)
    for input_file in input_files
)

# for input_file in input_files:
#     # run_analysis_pipeline(input_file, False)
#     draw_plots(input_file)
calculate_bio_price(input_files)

file_path = f"../output/03_price_{suffix}.xlsx"
plot_all_columns(file_path)
merge_png_images('cost',index="01_")
plot_revenue_cost_faceted(input_files)
