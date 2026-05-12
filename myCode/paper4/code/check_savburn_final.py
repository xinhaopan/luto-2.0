"""
Use xr_map_template to get lat/lon of output cells,
then find their SAVBURN_ELIGIBLE status.
"""
import io, sys, zipfile
import numpy as np
import pandas as pd
import xarray as xr
import cf_xarray as cfxr
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from tools.price_slice_utils import build_run_map

run_map, cp_vals, bp_vals = build_run_map()
YEAR = 2050
MAX_BP = max(bp_vals)
input_dir = Path(__file__).parents[3] / "input"

baseline_zip = run_map[(0.0, 0.0)]
high_bp_zip  = run_map.get((0.0, MAX_BP))

# ── Read map_template for lat/lon ────────────────────────────────────────────
def read_nc_raw(zip_path, suffix, year=None):
    target_year = year or YEAR
    with zipfile.ZipFile(zip_path) as arc:
        matches = [n for n in arc.namelist() if suffix in n and str(target_year) in n]
        if not matches:
            return None
        with arc.open(matches[0]) as f:
            return xr.open_dataset(io.BytesIO(f.read()), engine="h5netcdf").load()

ds_tmpl = read_nc_raw(baseline_zip, "xr_map_template", year=2015)
print("=== map_template coords & dims ===")
print(f"dims: {list(ds_tmpl.dims)}")
print(f"coords: {list(ds_tmpl.coords)}")
for var in ds_tmpl.data_vars:
    print(f"  var: {var}, shape={ds_tmpl[var].shape}")

# ── Read lumap for spatial reference ─────────────────────────────────────────
ds_lumap = read_nc_raw(baseline_zip, "xr_map_lumap", year=2010)
print("\n=== lumap 2010 ===")
print(f"dims: {list(ds_lumap.dims)}")
print(f"coords: {list(ds_lumap.coords)}")

# ── Decode compressed area file to get cell → (lat,lon) mapping ─────────────
# The area files have 'cell' coord 0..186647; map_template might have lat/lon in 2D
# Check if map_template has a mapping cell->lat,lon

# Alternative: use agec_crops which has 6.9M rows.
# Since RESFACTOR=5, the MASK selects every 5th cell (offset by RESFACTOR//2)
# from the full Australian land grid.
# The full land grid has 6.9M cells (NLUM land cells).
# The RF5 mask: from each 5x5 block, picks the center cell if any land use exists.
# This gives ~186K cells. Their indices in the 6.9M array are the MASK.

# Strategy: find MASK by reading the settings from the zip
print("\n=== Settings from inside baseline zip ===")
with zipfile.ZipFile(baseline_zip) as arc:
    if 'luto/settings.py' in arc.namelist():
        with arc.open('luto/settings.py') as f:
            content = f.read().decode('utf-8')
        for line in content.split('\n'):
            if 'RESFACTOR' in line and '=' in line and not line.strip().startswith('#'):
                print(f"  {line.strip()}")

# ── Load LUTO data object to get the real MASK ───────────────────────────────
# This is the most reliable approach: use LUTO's own data loading
print("\n=== Loading LUTO data to get MASK ===")
try:
    import luto.settings as settings_module
    # Save original RESFACTOR
    orig_rf = settings_module.RESFACTOR
    # Temporarily set RF to match this run
    settings_module.RESFACTOR = 5

    from luto.data import Data
    data = Data()
    mask = data.MASK
    print(f"MASK: {mask.sum()} True values out of {len(mask)}")

    # Now we have the correct MASK
    # Map output cells to full-grid indices
    mask_indices = np.where(mask)[0]  # full-grid indices of output cells

    # Read area files and find lost cells
    def read_area(zip_path, suffix):
        with zipfile.ZipFile(zip_path) as arc:
            matches = [n for n in arc.namelist() if suffix in n and str(YEAR) in n]
            if not matches:
                return None
            with arc.open(matches[0]) as f:
                ds = xr.open_dataset(io.BytesIO(f.read()), engine="h5netcdf")
        if "layer" in ds.dims and "compress" in ds["layer"].attrs:
            ds = cfxr.decode_compress_to_multi_index(ds, "layer")
        return ds.load()

    da_base = next(iter(read_area(baseline_zip, "xr_area_agricultural_landuse").data_vars.values()))
    da_high = next(iter(read_area(high_bp_zip,  "xr_area_agricultural_landuse").data_vars.values()))

    unalloc_base = da_base.sel(lu='Unallocated - natural land').sum('lm').values
    unalloc_high = da_high.sel(lu='Unallocated - natural land').sum('lm').values
    lost_mask = (unalloc_base - unalloc_high) > 10
    lost_full_idx = mask_indices[lost_mask]

    sb = pd.read_hdf(input_dir / "cell_savanna_burning.h5")
    sb_elig = sb['ELIGIBLE_AREA'].values
    sb_in_lost = sb_elig[lost_full_idx]
    print(f"\nOf {lost_mask.sum()} cells losing Unallocated natural land:")
    print(f"  SAVBURN_ELIGIBLE: {sb_in_lost.sum():.0f} ({100*sb_in_lost.mean():.1f}%)")
    print(f"  NOT in SAVBURN:   {(~sb_in_lost.astype(bool)).sum()} ({100*(1-sb_in_lost.mean()):.1f}%)")

    # Latitude
    lat = data.LATS
    lat_lost = lat[lost_mask]
    print(f"\nLatitude of lost cells: mean={lat_lost.mean():.1f}, "
          f"range [{lat_lost.min():.1f}, {lat_lost.max():.1f}]")
    print(f"Cells north of -20° (tropical/northern): {(lat_lost > -20).sum()} / {len(lat_lost)}")
    print(f"Cells south of -20°: {(lat_lost <= -20).sum()} / {len(lat_lost)}")

except Exception as e:
    print(f"Error loading LUTO data: {e}")
    import traceback; traceback.print_exc()
