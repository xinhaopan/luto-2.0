"""
Find MASK, align output cells to input cells, check SAVBURN eligibility
for cells that lost Unallocated natural land.
"""
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

# ── Find what files are in input to locate MASK ──────────────────────────────
print("=== Input files (looking for MASK) ===")
for f in sorted(input_dir.glob("*.npy"))[:20]:
    print(f"  {f.name}: shape={np.load(f, allow_pickle=True).shape}")

print()
for f in sorted(input_dir.glob("*.h5"))[:15]:
    print(f"  {f.name}")
