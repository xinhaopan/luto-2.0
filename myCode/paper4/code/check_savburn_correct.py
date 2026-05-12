"""Reconstruct exact LUTO MASK and check SAVBURN for lost natural land cells."""
import io, sys, zipfile
import numpy as np
import pandas as pd
import xarray as xr
import cf_xarray as cfxr
import rasterio
from scipy.ndimage import maximum_filter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from tools.price_slice_utils import build_run_map

run_map, cp_vals, bp_vals = build_run_map()
YEAR = 2050
MAX_BP = max(bp_vals)
input_dir = Path(__file__).parents[3] / "input"
baseline_zip = run_map[(0.0, 0.0)]
high_bp_zip  = run_map.get((0.0, MAX_BP))
RF = 5; offset = RF // 2

# ── Reconstruct LUTO MASK (same logic as data.py) ────────────────────────────
print("Loading NLUM mask...")
with rasterio.open(input_dir / "NLUM_2010-11_mask.tif") as src:
    nlum_2d = src.read(1).astype(np.int8)   # (3364, 4071), 0=ocean 1=land
    transform = src.transform

print(f"NLUM shape: {nlum_2d.shape}, land cells: {(nlum_2d>0).sum()}")

print("Loading lumap.h5...")
lumap_1d = pd.read_hdf(input_dir / "lumap.h5").to_numpy().astype(np.int8)
print(f"lumap shape: {lumap_1d.shape}, unique: {np.unique(lumap_1d)}")

# Build LUMASK_2D_FULLRES (True = valid ag cell)
MASK_LU_CODE = -1
lumask_1d = lumap_1d != MASK_LU_CODE          # (6956407,) True=ag
lumask_2d = np.zeros_like(nlum_2d, dtype=np.float32)
land_rows, land_cols = np.where(nlum_2d > 0)
lumask_2d[land_rows, land_cols] = lumask_1d.astype(np.float32)

print("Applying maximum_filter...")
have_lu = maximum_filter(lumask_2d, size=RF)   # True where any 5×5 has ag

# RF5 center positions in full-res 2D
rf_mask = nlum_2d.copy().astype(np.int8)
have_lu_ds = have_lu[offset::RF, offset::RF]   # (672, 814)

lu_mask_fullres = np.zeros_like(nlum_2d, dtype=bool)
lu_mask_fullres[offset::RF, offset::RF] = have_lu_ds.astype(bool)

# COORD_ROW_COL_FULLRES: RF5 center cells on land AND with lu in neighborhood
rf_center_land = (nlum_2d == 1) & lu_mask_fullres
coord_rc = np.argwhere(rf_center_land).T  # (2, n_centers)
print(f"Valid RF5 centers: {coord_rc.shape[1]}")

# Build 1D MASK (True where RF5 center is on land)
rf_mask_2d = nlum_2d.copy().astype(np.int8)
rf_mask_2d[coord_rc[0], coord_rc[1]] = 2
mask_1d = rf_mask_2d[nlum_2d > 0] == 2         # (6956407,) bool
print(f"MASK True count: {mask_1d.sum()}")

# Map output cell index (0..186647) → full 1D land index
mask_indices = np.where(mask_1d)[0]             # (186648,) full-grid indices

# ── Lat/Lon of each output cell ──────────────────────────────────────────────
# Center row/col in full-res 2D
out_rows = coord_rc[0]   # NLUM rows of output cells
out_cols = coord_rc[1]   # NLUM cols of output cells
# lat/lon from rasterio transform: lon = transform[2] + col*transform[0]
out_lons = transform[2] + out_cols * transform[0]
out_lats = transform[5] + out_rows * transform[4]

# ── Read area files ───────────────────────────────────────────────────────────
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
lost_out = (unalloc_base - unalloc_high) > 10   # output-space bool mask

print(f"\nOutput cells: {len(unalloc_base)}, MASK cells: {len(mask_indices)}")

if len(mask_indices) == len(unalloc_base):
    lat_lost = out_lats[lost_out]
    lon_lost = out_lons[lost_out]
    land_lost_idx = mask_indices[lost_out]

    print(f"\n=== Cells losing Unallocated natural land ({lost_out.sum()} cells) ===")
    print(f"  Lat: mean={lat_lost.mean():.2f}, range [{lat_lost.min():.2f}, {lat_lost.max():.2f}]")
    print(f"  Lon: mean={lon_lost.mean():.2f}")
    print(f"  Cells north of -20° (tropical/north): {(lat_lost > -20).sum()} / {len(lat_lost)}")
    print(f"  Cells south of -20° (temperate):      {(lat_lost <= -20).sum()} / {len(lat_lost)}")

    # SAVBURN check
    sb = pd.read_hdf(input_dir / "cell_savanna_burning.h5")
    sb_elig = sb['ELIGIBLE_AREA'].values
    sb_in_lost = sb_elig[land_lost_idx]
    print(f"\n  SAVBURN_ELIGIBLE=1: {sb_in_lost.sum():.0f} ({100*sb_in_lost.mean():.1f}%)")
    print(f"  NOT in SAVBURN:     {(~sb_in_lost.astype(bool)).sum()} ({100*(1-sb_in_lost.mean()):.1f}%)")

    # Overall SAVBURN rate for all output cells (for reference)
    sb_all = sb_elig[mask_indices]
    print(f"\n  (Overall SAVBURN rate in all output cells: {100*sb_all.mean():.1f}%)")
else:
    print(f"MISMATCH: output={len(unalloc_base)}, mask={len(mask_indices)}")
