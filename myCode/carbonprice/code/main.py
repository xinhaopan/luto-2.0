from joblib import Parallel, delayed
from tools.config import n_jobs, input_files
from tools import run_analysis_pipeline, draw_plots
from tools.helper_plot import merge_png_images

# Parallel(n_jobs=n_jobs)(
#     delayed(lambda input_file: (run_analysis_pipeline(input_file, True), draw_plots(input_file)))(input_file)
#     for input_file in input_files
# )
#
for input_file in input_files:
    # run_analysis_pipeline(input_file, False)
    draw_plots(input_file)
merge_png_images('cost',index="01_")
merge_png_images('price',index="02_")
merge_png_images('revenue_cost_stacked',index="03_")
merge_png_images('GHG Abatement(MtCOe2)',index="04_")
merge_png_images('BIO(Mha)',index="05_")
