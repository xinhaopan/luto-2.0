"""
Q1: Where does Unallocated natural land go at high bio price? LDS areas?
Q2: Why EP not destocked? Check which cells can be destocked vs EP.
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
input_dir    = Path(__file__).parents[3] / "input"

# ── Read area data ───────────────────────────────────────────────────────────
ds_ag_base = read_nc(baseline_zip, "xr_area_agricultural_landuse")
ds_ag_high = read_nc(high_bp_zip,  "xr_area_agricultural_landuse")
ds_na_base = read_nc(baseline_zip, "xr_area_non_agricultural_landuse")
ds_na_high = read_nc(high_bp_zip,  "xr_area_non_agricultural_landuse")

da_ag_base = next(iter(ds_ag_base.data_vars.values()))
da_ag_high = next(iter(ds_ag_high.data_vars.values()))
da_na_base = next(iter(ds_na_base.data_vars.values()))
da_na_high = next(iter(ds_na_high.data_vars.values()))

# ── Q1: Which cells lose Unallocated natural land? ───────────────────────────
print("=== Q1: Unallocated natural land → where does it go? ===")

# sum over lm for ag (lm is inside layer multiindex)
unalloc_base = da_ag_base.sel(lu='Unallocated - natural land').sum('lm').values  # (cell,)
unalloc_high = da_ag_high.sel(lu='Unallocated - natural land').sum('lm').values

lost_ha  = unalloc_base - unalloc_high
lost_mask = lost_ha > 10   # cells losing >10 ha of unalloc nat land
print(f"Cells losing Unallocated natural land (>10 ha): {lost_mask.sum()}")
print(f"Total lost: {lost_ha[lost_mask].sum()/1e6:.3f} Mha")

# What did these cells gain in non-ag?
base_na_lus = set(da_na_base.coords['lu'].values)
print("\nWhat these cells gained in non-ag:")
for lu in da_na_high.coords['lu'].values:
    if lu == 'ALL':
        continue
    high_val = da_na_high.sel(lu=lu).values
    base_val = da_na_base.sel(lu=lu).values if lu in base_na_lus else np.zeros_like(high_val)
    gained = high_val - base_val
    gained_in_lost_cells = gained[lost_mask].sum()
    if abs(gained_in_lost_cells) > 1e3:
        print(f"  {lu}: +{gained_in_lost_cells/1e6:.4f} Mha")

# What did these cells gain in ag (other lu)?
print("\nWhat these cells gained in ag (top changes):")
for lu in da_ag_high.coords['lu'].values:
    if lu in ('ALL', 'Unallocated - natural land'):
        continue
    gained = (da_ag_high.sel(lu=lu).sum('lm') - da_ag_base.sel(lu=lu).sum('lm')).values
    gained_in_lost_cells = gained[lost_mask].sum()
    if abs(gained_in_lost_cells) > 1e3:
        print(f"  {lu}: {gained_in_lost_cells/1e6:+.4f} Mha")

# Check SAVBURN_ELIGIBLE for lost cells
savburn_file = input_dir / "cell_savanna_burning.h5"
cell_df_file = input_dir / "cell_df.h5"
if savburn_file.exists():
    savburn_df = pd.read_hdf(savburn_file)
    savburn_elig = savburn_df['ELIGIBLE_AREA'].values
    n_input = len(savburn_elig)
    n_output = len(unalloc_base)
    print(f"\ncell count: input={n_input}, output={n_output}")
    if n_input == n_output:
        pct = 100 * savburn_elig[lost_mask].mean() if lost_mask.sum() > 0 else 0
        print(f"Lost-unalloc cells in SAVBURN_ELIGIBLE: {savburn_elig[lost_mask].sum():.0f} "
              f"({pct:.1f}%)")
        # Latitude
        if cell_df_file.exists():
            cell_df = pd.read_hdf(cell_df_file)
            if 'LAT' in cell_df.columns:
                lat = cell_df['LAT'].values
                print(f"Lat of lost cells: mean={lat[lost_mask].mean():.1f}, "
                      f"range [{lat[lost_mask].min():.1f}, {lat[lost_mask].max():.1f}]")
                print(f"Cells north of -20°: {(lat[lost_mask] > -20).sum()} / {lost_mask.sum()}")
else:
    print("SAVBURN file not found in input dir")

# ── Q2: Why EP not destocked for modified land? ──────────────────────────────
print("\n\n=== Q2: Why EP not destocked? ===")
print("Destocked transition cost: only available from Beef/Dairy/Sheep natural land")
print("Modified land (crops, modified pasture) → cannot be destocked → must use EP\n")

# Show baseline area by land use type to confirm
print("Baseline ag area by category:")
for lu in da_ag_base.coords['lu'].values:
    if lu == 'ALL':
        continue
    area = float(da_ag_base.sel(lu=lu).sum('lm').sum())
    if area > 1e6:
        print(f"  {lu}: {area/1e6:.1f} Mha")

# Load destocked cost array with lu names
destock_npy = input_dir / "ag_to_destock_tmatrix.npy"
if destock_npy.exists():
    dc = np.load(destock_npy)
    ag_lus_area = [lu for lu in da_ag_base.coords['lu'].values
                   if lu != 'ALL' and 'lm' not in str(lu)]
    # Remove ALL and deduplicate (lu coord may include lm x lu combos)
    # Get unique lu values
    all_lu = list(da_ag_base.coords['lu'].values)
    unique_lu = []
    seen = set()
    for lu in all_lu:
        if lu not in seen and lu != 'ALL':
            unique_lu.append(lu)
            seen.add(lu)
    print(f"\nDestocked option available for ({len(dc)} ag lus):")
    allowed = [(lu, c) for lu, c in zip(unique_lu, dc) if not np.isnan(c)]
    not_allowed = [(lu, c) for lu, c in zip(unique_lu, dc) if np.isnan(c)]
    print("  CAN be destocked:")
    for lu, c in allowed:
        area = float(da_ag_base.sel(lu=lu).sum('lm').sum())
        print(f"    {lu}: {c:.0f} AUD/ha, baseline area={area/1e6:.1f} Mha")
    print(f"  CANNOT be destocked ({len(not_allowed)} lu types)")
    # Show large ones
    for lu, c in not_allowed:
        area = float(da_ag_base.sel(lu=lu).sum('lm').sum())
        if area > 1e6:
            print(f"    {lu}: baseline area={area/1e6:.1f} Mha → must use EP if converting")
