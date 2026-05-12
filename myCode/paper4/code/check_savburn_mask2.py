"""Check SAVBURN eligibility of cells losing Unallocated natural land."""
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

baseline_zip = run_map[(0.0, 0.0)]
high_bp_zip  = run_map.get((0.0, MAX_BP))

# ── Find MASK from x_mrj ─────────────────────────────────────────────────────
print("Loading x_mrj to find MASK...")
x_mrj = np.load(input_dir / "x_mrj.npy")        # (2, 6956407, 28)
# MASK = cells with any land use assigned
mask = x_mrj.sum(axis=(0,2)) > 0                  # (6956407,) bool
print(f"Total cells: {len(mask)}, MASK cells: {mask.sum()}")

# ── Read areas ───────────────────────────────────────────────────────────────
ds_ag_base = read_nc(baseline_zip, "xr_area_agricultural_landuse")
ds_ag_high = read_nc(high_bp_zip,  "xr_area_agricultural_landuse")

da_ag_base = next(iter(ds_ag_base.data_vars.values()))
da_ag_high = next(iter(ds_ag_high.data_vars.values()))

unalloc_base = da_ag_base.sel(lu='Unallocated - natural land').sum('lm').values
unalloc_high = da_ag_high.sel(lu='Unallocated - natural land').sum('lm').values

lost_ha   = unalloc_base - unalloc_high
lost_mask = lost_ha > 10  # output cell indices
print(f"Output cells losing Unallocated nat land: {lost_mask.sum()}")
print(f"Total lost: {lost_ha[lost_mask].sum()/1e6:.3f} Mha")

# ── Map output cell indices to full-grid indices ─────────────────────────────
full_indices = np.where(mask)[0]      # shape (186648,), maps output→full grid
if len(full_indices) == len(unalloc_base):
    lost_full_idx = full_indices[lost_mask]
    print(f"Mapped {len(lost_full_idx)} lost cells to full grid indices")

    # ── SAVBURN check ────────────────────────────────────────────────────────
    savburn_df = pd.read_hdf(input_dir / "cell_savanna_burning.h5")
    savburn_elig = savburn_df['ELIGIBLE_AREA'].values  # (6956407,)
    print(f"Savburn rows: {len(savburn_elig)}")

    sb_in_lost = savburn_elig[lost_full_idx]
    print(f"\nOf {lost_mask.sum()} cells losing Unallocated natural land:")
    print(f"  In SAVBURN_ELIGIBLE (=1): {sb_in_lost.sum():.0f} ({100*sb_in_lost.mean():.1f}%)")
    print(f"  NOT in SAVBURN areas:     {(~sb_in_lost.astype(bool)).sum()} ({100*(1-sb_in_lost.mean()):.1f}%)")

    # ── Latitude/longitude of lost cells ────────────────────────────────────
    cell_df = pd.read_hdf(input_dir / "cell_df.h5")
    print(f"\ncell_df columns: {[c for c in cell_df.columns if 'LAT' in c.upper() or 'LON' in c.upper() or 'lat' in c or 'lon' in c]}")
    lat_col = next((c for c in cell_df.columns if 'LAT' in c.upper()), None)
    lon_col = next((c for c in cell_df.columns if 'LON' in c.upper()), None)
    if lat_col:
        lat = cell_df[lat_col].values
        lon = cell_df[lon_col].values if lon_col else None
        lat_lost = lat[lost_full_idx]
        print(f"\nLatitude of lost cells: mean={lat_lost.mean():.1f}, "
              f"range [{lat_lost.min():.1f}, {lat_lost.max():.1f}]")
        print(f"Cells north of -20° (tropical): {(lat_lost > -20).sum()} / {len(lat_lost)}")
        print(f"Cells south of -20°: {(lat_lost <= -20).sum()} / {len(lat_lost)}")
        if lon is not None:
            lon_lost = lon[lost_full_idx]
            print(f"Longitude: mean={lon_lost.mean():.1f}, range [{lon_lost.min():.1f}, {lon_lost.max():.1f}]")

    # ── What did these cells convert TO? ────────────────────────────────────
    print("\n--- What these cells converted TO (all land uses with change): ---")
    ds_na_base = read_nc(baseline_zip, "xr_area_non_agricultural_landuse")
    ds_na_high = read_nc(high_bp_zip,  "xr_area_non_agricultural_landuse")
    da_na_base = next(iter(ds_na_base.data_vars.values()))
    da_na_high = next(iter(ds_na_high.data_vars.values()))

    base_na_lus = set(da_na_base.coords['lu'].values)
    print("  Non-ag gains in lost cells:")
    for lu in da_na_high.coords['lu'].values:
        if lu == 'ALL':
            continue
        h = da_na_high.sel(lu=lu).values
        b = da_na_base.sel(lu=lu).values if lu in base_na_lus else np.zeros_like(h)
        delta = (h - b)[lost_mask].sum()
        if abs(delta) > 100:
            print(f"    {lu}: {delta/1e6:+.4f} Mha")

    print("  Ag changes in lost cells:")
    for lu in da_ag_high.coords['lu'].values:
        if lu in ('ALL', 'Unallocated - natural land'):
            continue
        h = da_ag_high.sel(lu=lu).sum('lm').values
        b = da_ag_base.sel(lu=lu).sum('lm').values
        delta = (h - b)[lost_mask].sum()
        if abs(delta) > 100:
            print(f"    {lu}: {delta/1e6:+.4f} Mha")
else:
    print(f"Cell count mismatch: full_indices={len(full_indices)}, output={len(unalloc_base)}")
    print("Trying cell_df approach...")
    cell_df = pd.read_hdf(input_dir / "cell_df.h5")
    print(f"cell_df rows: {len(cell_df)}, columns: {list(cell_df.columns)[:15]}")
