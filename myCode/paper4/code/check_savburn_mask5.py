"""Find correct cell mapping via LUTO's settings and NLUM_MASK."""
import io, sys, zipfile
import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path

# Add luto to path
repo_root = Path(__file__).parents[3]
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(Path(__file__).parent))

# Check what other files are in output zip that give spatial info
from tools.price_slice_utils import build_run_map

run_map, cp_vals, bp_vals = build_run_map()
baseline_zip = run_map[(0.0, 0.0)]

print("=== Files in baseline zip that mention coord/meta/cell ===")
with zipfile.ZipFile(baseline_zip) as arc:
    for n in arc.namelist():
        if any(k in n.lower() for k in ['coord', 'meta', 'nlum', 'mask', 'geo', 'lut', 'cell_id']):
            print(f"  {n}")

print("\n=== All nc files (non-dvar, non-area) ===")
with zipfile.ZipFile(baseline_zip) as arc:
    for n in arc.namelist():
        if n.endswith('.nc') and 'dvar' not in n and 'area' not in n and '2050' in n:
            print(f"  {n}")

# Try to load LUTO settings and data to get MASK
print("\n=== Trying to load LUTO MASK directly ===")
import luto.settings as settings
print(f"RESFACTOR={settings.RESFACTOR}")
input_dir = Path(settings.INPUT_DIR)

# Load NLUM_MASK (the full-resolution land/non-land mask)
nlum_tif = input_dir / "NLUM_SPREAD_LU_ID_250m_2km.tif"
nlum_npy = input_dir / "NLUM_MASK.npy"
for f in [nlum_npy] + list(input_dir.glob("NLUM*.npy")):
    if f.exists():
        arr = np.load(f)
        print(f"  {f.name}: shape={arr.shape}, dtype={arr.dtype}")

# Try reading NLUM mask from the tif
for tif_file in input_dir.glob("*.tif"):
    print(f"  tif: {tif_file.name}")
