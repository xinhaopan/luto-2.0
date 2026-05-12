"""
Use xr_map_lumap to get 2D spatial grid, then map output cells to lat/lon
and find SAVBURN status via the GHG output (which already has savburn info).
"""
import io, sys, zipfile
import numpy as np
import pandas as pd
import xarray as xr
import cf_xarray as cfxr
import rasterio
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from tools.price_slice_utils import build_run_map

run_map, cp_vals, bp_vals = build_run_map()
YEAR = 2050
MAX_BP = max(bp_vals)
input_dir = Path(__file__).parents[3] / "input"
baseline_zip = run_map[(0.0, 0.0)]
high_bp_zip  = run_map.get((0.0, MAX_BP))

def read_nc(zip_path, suffix, year=None):
    yr = year or YEAR
    with zipfile.ZipFile(zip_path) as arc:
        matches = [n for n in arc.namelist() if suffix in n and str(yr) in n]
        if not matches: return None
        with arc.open(matches[0]) as f:
            ds = xr.open_dataset(io.BytesIO(f.read()), engine="h5netcdf")
    if "layer" in ds.dims and "compress" in ds["layer"].attrs:
        ds = cfxr.decode_compress_to_multi_index(ds, "layer")
    return ds.load()

# ── Read NLUM_MASK.tif to get the 2D->1D mapping ────────────────────────────
nlum_tif = input_dir / "NLUM_2010-11_mask.tif"
print(f"Reading {nlum_tif.name}...")
with rasterio.open(nlum_tif) as src:
    nlum = src.read(1)          # 2D array, non-null = land cells
    transform = src.transform
    crs = src.crs
    print(f"  NLUM shape: {nlum.shape}, dtype={nlum.dtype}")
    print(f"  Transform: {transform}")
    print(f"  CRS: {crs}")
    nrows, ncols = nlum.shape

# Land cells in 2D
land_mask_2d = nlum > 0
print(f"  Land cells (NLUM>0): {land_mask_2d.sum()}")

# Build (row, col) for each land cell → 1D index
land_rows, land_cols = np.where(land_mask_2d)
n_land = len(land_rows)  # should be ~6.9M
print(f"  n_land (1D): {n_land}")

# RF5: center of each 5x5 block → row = 2, 7, 12, ...; col = 2, 7, 12, ...
RF = 5
offset = RF // 2  # = 2

# For each land cell (land_rows[i], land_cols[i]):
# Is it the center of a 5x5 block AND does it have land?
# Center cells: row % RF == offset AND col % RF == offset
rf_center_mask = (land_rows % RF == offset) & (land_cols % RF == offset)
print(f"\nRF{RF} center cells among land cells: {rf_center_mask.sum()}")

# These should be the ~186K output cells
# Their 1D index in the land array is where rf_center_mask is True
rf_center_1d = np.where(rf_center_mask)[0]  # indices in the land array

# Compute lat/lon for these cells
center_rows = land_rows[rf_center_1d]
center_cols = land_cols[rf_center_1d]
# rasterio: (col, row) → (lon, lat)
lons = transform[2] + center_cols * transform[0] + 0.5 * transform[0]
lats = transform[5] + center_rows * transform[4] + 0.5 * transform[4]

print(f"Output cells: {len(rf_center_1d)}")
print(f"Lat range: [{lats.min():.1f}, {lats.max():.1f}]")
print(f"Lon range: [{lons.min():.1f}, {lons.max():.1f}]")

# ── Read area files and find lost Unallocated natural land cells ─────────────
da_base = next(iter(read_nc(baseline_zip, "xr_area_agricultural_landuse").data_vars.values()))
da_high = next(iter(read_nc(high_bp_zip,  "xr_area_agricultural_landuse").data_vars.values()))
unalloc_base = da_base.sel(lu='Unallocated - natural land').sum('lm').values
unalloc_high = da_high.sel(lu='Unallocated - natural land').sum('lm').values
lost_mask_out = (unalloc_base - unalloc_high) > 10  # output cell mask

print(f"\nOutput cells: {len(unalloc_base)}, RF centers: {len(rf_center_1d)}")

if len(rf_center_1d) == len(unalloc_base):
    # Perfect match - use RF center lat/lon for lost cells
    lat_lost = lats[lost_mask_out]
    lon_lost = lons[lost_mask_out]
    print(f"\nOf {lost_mask_out.sum()} cells losing Unallocated natural land:")
    print(f"  Lat: mean={lat_lost.mean():.1f}, range [{lat_lost.min():.1f}, {lat_lost.max():.1f}]")
    print(f"  Lon: mean={lon_lost.mean():.1f}, range [{lon_lost.min():.1f}, {lon_lost.max():.1f}]")
    print(f"  Cells north of -20° lat (tropical): {(lat_lost > -20).sum()} / {len(lat_lost)}")
    print(f"  Cells south of -20°: {(lat_lost <= -20).sum()} / {len(lat_lost)}")

    # SAVBURN check using 1D land indices
    lost_land_1d = rf_center_1d[lost_mask_out]  # indices in the 6.9M land array
    sb = pd.read_hdf(input_dir / "cell_savanna_burning.h5")
    sb_elig = sb['ELIGIBLE_AREA'].values
    sb_in_lost = sb_elig[lost_land_1d]
    print(f"\n  SAVBURN_ELIGIBLE: {sb_in_lost.sum():.0f} ({100*sb_in_lost.mean():.1f}%)")
    print(f"  NOT in SAVBURN:   {(~sb_in_lost.astype(bool)).sum()} ({100*(1-sb_in_lost.mean()):.1f}%)")
else:
    print(f"Mismatch: RF centers={len(rf_center_1d)} vs output={len(unalloc_base)}")
    # Maybe RF5 doesn't simply pick centers - check data.py more carefully
