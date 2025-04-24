from joblib import Parallel, delayed
from tools.config import n_jobs, input_files
from tools import run_analysis_pipeline

Parallel(n_jobs=n_jobs)(
    delayed(run_analysis_pipeline)(input_file, True) for input_file in input_files
)
#
# for input_file in input_files:
#     run_analysis_pipeline(input_file, False)
