
from tools.data_helper import *
from tools.plot_helper import *
from tools.parameters import *


# ----------------------- 函数定义 -----------------------
for input_file in input_files:
    compute_land_use_change_metrics(input_file)
    plot_land_use_polar(input_file)