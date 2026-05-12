"""Use lumap 2D output to get lat/lon of output cells, then check SAVBURN."""
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

# ── Read NLUM tif + lumap to build exact MASK ────────────────────────────────
# NLUM: 3364×4071 at 0.01° resolution, starting at lon=112.92, lat=-10.02
# RF5: center = rows 2,7,12,...; cols 2,7,12,... in the 2D NLUM grid
# Valid RF5 cells: those with lumap != -1

# Check for lumap file
print("Looking for lumap .npy files...")
for f in sorted(input_dir.glob("*.npy")):
    arr = np.load(f, allow_pickle=True)
    if len(arr.shape) == 1 and arr.shape[0] == 6956407:
        uniq = np.unique(arr)[:8]
        print(f"  {f.name}: unique vals={uniq}")

# Use state_id to understand what valid cells look like
state_id = np.load(input_dir / "state_id.npy")
print(f"\nstate_id unique: {np.unique(state_id)}")
print(f"state_id=0 (invalid?): {(state_id==0).sum()}")

# ── Alternative: use lat/lon from bio_OVERALL_PRIORITY file ─────────────────
bio_h5 = input_dir / "bio_OVERALL_PRIORITY_RANK_AND_AREA_CONNECTIVITY.h5"
if bio_h5.exists():
    bio_df = pd.read_hdf(bio_h5)
    print(f"\nbio h5: rows={len(bio_df)}, cols={list(bio_df.columns)[:10]}")

# ── Key insight: use agec_crops LAT/LON if present ──────────────────────────
agec = pd.read_hdf(input_dir / "agec_crops.h5")
print(f"\nagec_crops: rows={len(agec)}, col levels={agec.columns.nlevels}")
if agec.columns.nlevels > 1:
    print(f"  top-level cols: {list(agec.columns.get_level_values(0).unique())[:10]}")

# ── Use map_template + known RF5 transform to get cell lat/lon ──────────────
# map_template shape: (673, 814), covering RF5 grid
# RF5 pixel size = 5 × 0.01° = 0.05°
# RF5 grid origin: NLUM origin + RF5_offset × pixel_size
# NLUM: top-left = (lon=112.92, lat=-10.02), pixel=0.01°
# RF5 centers: lon = 112.92 + (2 + 5k) * 0.01 for k=0,1,...
#              lat = -10.02 - (2 + 5k) * 0.01 for k=0,1,...
RF = 5
offset = RF // 2
nlum_lon0, nlum_lat0 = 112.92, -10.02
nlum_res = 0.01

# Generate RF5 center coordinates (2D grid)
n_rf_rows = 3364 // RF  # = 672
n_rf_cols = 4071 // RF  # = 814
print(f"\nRF5 grid: {n_rf_rows} × {n_rf_cols} = {n_rf_rows*n_rf_cols}")

# RF5 centers in NLUM row/col
rf_row_idx = np.arange(n_rf_rows) * RF + offset  # rows in NLUM (2,7,12,...)
rf_col_idx = np.arange(n_rf_cols) * RF + offset  # cols in NLUM (2,7,12,...)

# 2D lat/lon grids
rf_lons_2d = nlum_lon0 + rf_col_idx[np.newaxis, :] * nlum_res  # (1, ncols)
rf_lats_2d = nlum_lat0 - rf_row_idx[:, np.newaxis] * nlum_res  # (nrows, 1)
rf_lons_2d = np.broadcast_to(rf_lons_2d, (n_rf_rows, n_rf_cols))
rf_lats_2d = np.broadcast_to(rf_lats_2d, (n_rf_rows, n_rf_cols))

# Read NLUM to find which RF5 centers have land
with rasterio.open(input_dir / "NLUM_2010-11_mask.tif") as src:
    nlum_2d = src.read(1)  # (3364, 4071)

# RF5 centers that fall on land
rf_nlum_vals = nlum_2d[rf_row_idx[:, np.newaxis], rf_col_idx[np.newaxis, :]]
rf_land = rf_nlum_vals > 0  # (n_rf_rows, n_rf_cols)
n_rf_land = rf_land.sum()
print(f"RF5 centers on land: {n_rf_land}")

# Now need to further filter by lumap != -1 (valid LUTO cells)
# The 186648 output cells are RF5 land centers with a valid lumap
# We can check this using x_mrj: cells where any lu has area
x_mrj = np.load(input_dir / "x_mrj.npy")  # (2, 6956407, 28)
# 1D land cell indices for each RF5 land center:
# RF5 center at (rf_row, rf_col) → NLUM 2D → 1D land cell index

# Map 2D NLUM position to 1D land index
nlum_flat = nlum_2d.flatten()
nlum_land_cumsum = np.cumsum(nlum_flat > 0) - 1  # 1D land index for each 2D pixel

# Get 1D NLUM position for each RF5 land center
rf_row_2d, rf_col_2d = np.where(rf_land)
# These are indices in the RF5 grid; get corresponding NLUM positions
nlum_rows = rf_row_idx[rf_row_2d]
nlum_cols = rf_col_idx[rf_col_2d]
nlum_flat_idx = nlum_rows * 4071 + nlum_cols
land_1d_idx = nlum_land_cumsum[nlum_flat_idx]

# Check if these cells have any x_mrj > 0
x_sum = x_mrj[:, land_1d_idx, :].sum(axis=(0, 2))  # (n_rf_land,)
valid_mask = x_sum > 0
print(f"RF5 land centers with valid x_mrj: {valid_mask.sum()}")

# These should be the 186648 output cells
valid_lat = rf_lats_2d[rf_land][valid_mask]
valid_lon = rf_lons_2d[rf_land][valid_mask]
valid_land_1d = land_1d_idx[valid_mask]
print(f"Valid cells: {len(valid_lat)}")
print(f"Lat range: [{valid_lat.min():.1f}, {valid_lat.max():.1f}]")

# ── Now load area files and find lost cells ──────────────────────────────────
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
lost_out = (unalloc_base - unalloc_high) > 10  # output-space mask

if len(valid_lat) == len(unalloc_base):
    lat_lost = valid_lat[lost_out]
    lon_lost = valid_lon[lost_out]
    land_lost = valid_land_1d[lost_out]

    print(f"\nOf {lost_out.sum()} cells losing Unallocated natural land:")
    print(f"  Lat: mean={lat_lost.mean():.1f}, range [{lat_lost.min():.1f}, {lat_lost.max():.1f}]")
    print(f"  Cells north of -20° (tropical): {(lat_lost > -20).sum()} / {len(lat_lost)}")
    print(f"  Cells south of -20°: {(lat_lost <= -20).sum()} / {len(lat_lost)}")

    # SAVBURN check
    sb = pd.read_hdf(input_dir / "cell_savanna_burning.h5")
    sb_elig = sb['ELIGIBLE_AREA'].values
    sb_in_lost = sb_elig[land_lost]
    print(f"\n  SAVBURN_ELIGIBLE=1: {sb_in_lost.sum():.0f} ({100*sb_in_lost.mean():.1f}%)")
    print(f"  NOT in SAVBURN:     {(~sb_in_lost.astype(bool)).sum()} ({100*(1-sb_in_lost.mean()):.1f}%)")
else:
    print(f"\nMismatch: valid={len(valid_lat)} vs output={len(unalloc_base)}")
    print("Need to adjust filtering approach")
