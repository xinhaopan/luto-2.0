"""Find the MASK by reading agec_crops.h5 index, then check SAVBURN."""
import io, sys, zipfile
import cf_xarray as cfxr
import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from tools.price_slice_utils import build_run_map

run_map, cp_vals, bp_vals = build_run_map()
YEAR = 2050
MAX_BP = max(bp_vals)
input_dir = Path(__file__).parents[3] / "input"

# Check how many rows savanna burning has
sb = pd.read_hdf(input_dir / "cell_savanna_burning.h5")
print(f"savanna_burning rows: {len(sb)}, columns: {list(sb.columns)}")
print(sb.head(3))
print(f"ELIGIBLE_AREA values: {sb['ELIGIBLE_AREA'].value_counts().to_dict()}")

# Check agec_crops for its index
agec = pd.read_hdf(input_dir / "agec_crops.h5")
print(f"\nagec_crops rows: {len(agec)}")
print(f"agec_crops index type: {type(agec.index)}")
print(f"agec_crops index range: {agec.index.min()} to {agec.index.max()}")
