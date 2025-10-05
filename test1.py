import numpy as np
import xarray as xr
from typing import Union

path = r"N:\LUF-Modelling\LUTO2_XH\LUTO2\output\20251003_Paper2_Results_test\carbon_price\0_base_data\Run_01_GHG_high_BIO_high_CUT_50\2050\xr_GHG_non_ag_2050.nc"
da = xr.open_dataarray(path)
da