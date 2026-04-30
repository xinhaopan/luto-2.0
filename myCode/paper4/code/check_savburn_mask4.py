"""Find MASK via NLUM_MASK and check SAVBURN for lost natural land cells."""
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

# ── Find NLUM_MASK / LUMAP to derive MASK ────────────────────────────────────
# Look for tiff or npy files that could be NLUM_MASK
print("Looking for NLUM or LUMAP files...")
for f in input_dir.glob("*.npy"):
    arr = np.load(f)
    if arr.shape == (6956407,):
        print(f"  {f.name}: shape={arr.shape}, unique={np.unique(arr)[:10]}")

# Check if there's a lumap file
lumap_file = input_dir / "lumap_2010.npy"
if lumap_file.exists():
    lumap = np.load(lumap_file)
    print(f"\nlumap_2010.npy: shape={lumap.shape}, unique values={np.unique(lumap)}")

# ── Try reading agec_crops with RESFACTOR=5 selection ───────────────────────
# The MASK at RF5 selects every 5th cell in lat/lon, only where land exists
# Let's check the agec_lvstk to see if it has LAT/LON
agec = pd.read_hdf(input_dir / "agec_lvstk.h5")
print(f"\nagec_lvstk columns: {list(agec.columns)[:10]}")
print(f"agec_lvstk index range: {agec.index.min()} - {agec.index.max()}")

# ── Alternative: read output nc to get cell coordinate (index in LUTO space)
# and find MASK indices from the output zip ──────────────────────────────────
baseline_zip = run_map[(0.0, 0.0)]
high_bp_zip  = run_map.get((0.0, MAX_BP))

# Check if output nc has lat/lon coords
with zipfile.ZipFile(baseline_zip) as arc:
    matches = [n for n in arc.namelist() if "xr_area_agricultural" in n and str(YEAR) in n]
    with arc.open(matches[0]) as f:
        ds_raw = xr.open_dataset(io.BytesIO(f.read()), engine="h5netcdf")

print(f"\nRaw output nc coords: {list(ds_raw.coords)}")
print(f"Raw output nc dims: {list(ds_raw.dims)}")
# Check if 'cell' coord has actual spatial info
if 'cell' in ds_raw.coords:
    cell_vals = ds_raw.coords['cell'].values
    print(f"Cell values: min={cell_vals.min()}, max={cell_vals.max()}, n={len(cell_vals)}")
    print(f"First 5 cell values: {cell_vals[:5]}")

# If cell values are indices into the full 6.9M grid, we can use them directly!
ds_raw.close()

# ── Use cell indices as MASK indices ────────────────────────────────────────
def read_nc(zip_path, suffix):
    with zipfile.ZipFile(zip_path) as arc:
        matches = [n for n in arc.namelist() if suffix in n and str(YEAR) in n]
        if not matches:
            return None
        with arc.open(matches[0]) as f:
            ds = xr.open_dataset(io.BytesIO(f.read()), engine="h5netcdf")
    if "layer" in ds.dims and "compress" in ds["layer"].attrs:
        ds = cfxr.decode_compress_to_multi_index(ds, "layer")
    return ds.load()

ds_ag_base = read_nc(baseline_zip, "xr_area_agricultural_landuse")
ds_ag_high = read_nc(high_bp_zip,  "xr_area_agricultural_landuse")
da_ag_base = next(iter(ds_ag_base.data_vars.values()))
da_ag_high = next(iter(ds_ag_high.data_vars.values()))

# Get cell coordinate values (these should be indices into full grid if RF>1)
cell_indices = da_ag_base.coords['cell'].values
print(f"\nOutput cell coord values: min={cell_indices.min()}, max={cell_indices.max()}, n={len(cell_indices)}")

# Check: are these the full grid indices?
unalloc_base = da_ag_base.sel(lu='Unallocated - natural land').sum('lm').values
unalloc_high = da_ag_high.sel(lu='Unallocated - natural land').sum('lm').values
lost_mask = (unalloc_base - unalloc_high) > 10
lost_cell_coords = cell_indices[lost_mask]
print(f"Lost cell coord values (first 10): {lost_cell_coords[:10]}")

# Try using these as indices into savanna_burning.h5
sb = pd.read_hdf(input_dir / "cell_savanna_burning.h5")
sb_eligible = sb['ELIGIBLE_AREA'].values

if lost_cell_coords.max() < len(sb_eligible):
    sb_in_lost = sb_eligible[lost_cell_coords]
    print(f"\nOf {lost_mask.sum()} cells losing Unallocated natural land:")
    print(f"  SAVBURN_ELIGIBLE=1: {sb_in_lost.sum():.0f} ({100*sb_in_lost.mean():.1f}%)")
    print(f"  NOT in SAVBURN: {(~sb_in_lost.astype(bool)).sum()} ({100*(1-sb_in_lost.mean()):.1f}%)")

    # Latitude check via agec_crops LAT/LON if available
    agec_crops = pd.read_hdf(input_dir / "agec_crops.h5")
    lat_col = next((c for c in agec_crops.columns if 'lat' in c.lower()), None)
    if lat_col:
        lat = agec_crops[lat_col].values
        lat_lost = lat[lost_cell_coords]
        print(f"\nLatitude of lost cells: mean={lat_lost.mean():.1f}, "
              f"range [{lat_lost.min():.1f}, {lat_lost.max():.1f}]")
        print(f"Cells north of -20°: {(lat_lost > -20).sum()} / {len(lat_lost)}")
    else:
        print(f"\nagec_crops has no LAT column. Columns: {list(agec_crops.columns)[:10]}")
else:
    print(f"cell coords ({lost_cell_coords.max()}) exceed savburn rows ({len(sb_eligible)})")
    print("These are NOT full-grid indices — need different MASK approach")
