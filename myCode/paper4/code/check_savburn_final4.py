"""Use xr_map_lumap 2D output to get exact cell lat/lon, then check SAVBURN."""
import io, sys, zipfile
import numpy as np
import pandas as pd
import xarray as xr
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

# ── Read xr_map_lumap 2010 to find valid cell positions ─────────────────────
with zipfile.ZipFile(baseline_zip) as arc:
    matches = [n for n in arc.namelist() if 'xr_map_lumap' in n and '2010' in n]
    print(f"lumap files: {matches}")
    with arc.open(matches[0]) as f:
        ds_lumap = xr.open_dataset(io.BytesIO(f.read()), engine="h5netcdf").load()

print(f"lumap dims: {list(ds_lumap.dims)}")
print(f"lumap coords: {list(ds_lumap.coords)}")
print(f"lumap vars: {list(ds_lumap.data_vars)}")
for v in ds_lumap.data_vars:
    print(f"  {v}: shape={ds_lumap[v].shape}, dtype={ds_lumap[v].dtype}")
    print(f"    x range: [{float(ds_lumap.coords['x'].min()):.3f}, {float(ds_lumap.coords['x'].max()):.3f}]")
    print(f"    y range: [{float(ds_lumap.coords['y'].min()):.3f}, {float(ds_lumap.coords['y'].max()):.3f}]")
    print(f"    unique vals (first 10): {np.unique(ds_lumap[v].values)[:10]}")
    break

# ── Find valid cells (non-null/non-nodata) in the 2D lumap ──────────────────
lumap_2d = next(iter(ds_lumap.data_vars.values())).values  # (673, 814)
print(f"\nlumap 2D shape: {lumap_2d.shape}")
print(f"nodata values: {lumap_2d.min()}")

# Valid (non-nodata) cells
nodata = lumap_2d.min()  # assume minimum is nodata (-1 or 255 or similar)
valid_2d = lumap_2d != nodata
print(f"Valid cells in 2D: {valid_2d.sum()}")

# x, y coordinates
x_vals = ds_lumap.coords['x'].values  # longitude
y_vals = ds_lumap.coords['y'].values  # latitude

# For each valid cell, get its lat/lon
valid_rows, valid_cols = np.where(valid_2d)
valid_lons = x_vals[valid_cols]
valid_lats = y_vals[valid_rows]

print(f"Lat range: [{valid_lats.min():.2f}, {valid_lats.max():.2f}]")
print(f"Lon range: [{valid_lons.min():.2f}, {valid_lons.max():.2f}]")

# ── If 186648 valid cells found, check SAVBURN ────────────────────────────────
import cf_xarray as cfxr

def read_nc(zip_path, suffix):
    with zipfile.ZipFile(zip_path) as arc:
        matches = [n for n in arc.namelist() if suffix in n and str(YEAR) in n]
        if not matches: return None
        with arc.open(matches[0]) as f:
            ds = xr.open_dataset(io.BytesIO(f.read()), engine="h5netcdf")
    if "layer" in ds.dims and "compress" in ds["layer"].attrs:
        ds = cfxr.decode_compress_to_multi_index(ds, "layer")
    return ds.load()

da_base = next(iter(read_nc(baseline_zip, "xr_area_agricultural_landuse").data_vars.values()))
da_high = next(iter(read_nc(high_bp_zip,  "xr_area_agricultural_landuse").data_vars.values()))
unalloc_base = da_base.sel(lu='Unallocated - natural land').sum('lm').values
unalloc_high = da_high.sel(lu='Unallocated - natural land').sum('lm').values
lost_out = (unalloc_base - unalloc_high) > 10

print(f"\nOutput cells: {len(unalloc_base)}, valid 2D cells: {len(valid_lats)}")

if len(valid_lats) == len(unalloc_base):
    # Match found - use these lat/lon for spatial analysis
    lat_lost = valid_lats[lost_out]
    lon_lost = valid_lons[lost_out]
    print(f"\nOf {lost_out.sum()} cells losing Unallocated natural land:")
    print(f"  Lat: mean={lat_lost.mean():.1f}, range [{lat_lost.min():.1f}, {lat_lost.max():.1f}]")
    print(f"  Lon: mean={lon_lost.mean():.1f}, range [{lon_lost.min():.1f}, {lon_lost.max():.1f}]")
    print(f"  Cells north of -20° (tropical/north): {(lat_lost > -20).sum()} / {len(lat_lost)}")
    print(f"  Cells south of -20°: {(lat_lost <= -20).sum()} / {len(lat_lost)}")

    # Map back to NLUM land cell indices for SAVBURN check
    # Each valid_2d cell at (row, col) in RF5 grid corresponds to
    # NLUM cell at (row*5+2, col*5+2)
    RF = 5; offset = 2
    nlum_rows = valid_rows * RF + offset
    nlum_cols = valid_cols * RF + offset
    nlum_flat = nlum_rows * 4071 + nlum_cols

    # Map to 1D land index
    with rasterio.open(input_dir / "NLUM_2010-11_mask.tif") as src:
        nlum_2d_arr = src.read(1)
    nlum_flat_all = nlum_2d_arr.flatten()
    cumsum = np.cumsum(nlum_flat_all > 0) - 1
    land_1d = cumsum[nlum_flat]

    land_lost_1d = land_1d[lost_out]
    sb = pd.read_hdf(input_dir / "cell_savanna_burning.h5")
    sb_elig = sb['ELIGIBLE_AREA'].values
    sb_in_lost = sb_elig[land_lost_1d]
    print(f"\n  SAVBURN_ELIGIBLE=1: {sb_in_lost.sum():.0f} ({100*sb_in_lost.mean():.1f}%)")
    print(f"  NOT in SAVBURN:     {(~sb_in_lost.astype(bool)).sum()} ({100*(1-sb_in_lost.mean()):.1f}%)")
else:
    print(f"Cell count mismatch: {len(valid_lats)} valid in 2D vs {len(unalloc_base)} output cells")
    # Check what nodata really is
    print(f"nodata used: {nodata}")
    print(f"All unique lumap values: {np.unique(lumap_2d)}")
