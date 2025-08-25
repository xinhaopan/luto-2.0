from joblib import Parallel, delayed
import time
from tools.tools import get_path, get_year
import shutil
import os
import xarray as xr
import numpy_financial as npf
import numpy as np

import tools.config as config

input_files = config.INPUT_FILES
path_name_0 = get_path(input_files[2])
path_name_1 = get_path(input_files[1])
path_name_2 = get_path(input_files[0])

