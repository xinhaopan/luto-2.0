from joblib import Parallel, delayed
from tools.desc import input_files,n_jobs
from tools.tools import *
for input_file in input_files:
    draw_figure(input_file)

# Parallel(n_jobs=n_jobs)(
#     delayed(calculate_price)(input_file, True) for input_file in input_files
# )
