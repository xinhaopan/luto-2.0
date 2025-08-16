from joblib import Parallel, delayed
import tools.config as config
from tools import run_analysis_pipeline, draw_plots
from tools.helper_plot import merge_png_images,plot_all_columns,plot_revenue_cost_faceted
from tools.helper_data import calculate_cost, calculate_price

Parallel(n_jobs=config.N_JOBS)(
    delayed(lambda input_file: (run_analysis_pipeline(input_file, True), draw_plots(input_file)))(input_file)
    for input_file in config.INPUT_FILES
)
#
# for input_file in config.INPUT_FILES:
#     run_analysis_pipeline(input_file, False)
#     draw_plots(input_file)

# df = calculate_bio_price(config.INPUT_FILES)
# plot_all_columns(df)
