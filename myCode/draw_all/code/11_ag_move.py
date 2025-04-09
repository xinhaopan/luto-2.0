import os
import re
import numpy as np
import pandas as pd
import rasterio
from joblib import Parallel, delayed
from pyproj import CRS, Transformer
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from tools.data_helper import *
from tools.plot_helper import *
from tools.parameters import *


# ----------------------- 函数定义 -----------------------
# compute_land_use_change_metrics(input_files[4],use_parallel=False)
plot_land_use_polar(f"{input_files[4]}", f"12_{input_files[4]}",fontsize=30,x_offset=1.35)