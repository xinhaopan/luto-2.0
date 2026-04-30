"""
Check: which cells convert from natural land (ag) to other uses at high bio price?
Are they in SAVBURN_ELIGIBLE areas?
Also print transition cost stats.
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

def read_nc_var(zip_path, suffix):
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

# -- read ag dvar --
ds_ag_base = read_nc_var(baseline_zip, "xr_dvar_ag")
ds_ag_high = read_nc_var(high_bp_zip,  "xr_dvar_ag")
ds_na_base = read_nc_var(baseline_zip, "xr_dvar_non_ag")
ds_na_high = read_nc_var(high_bp_zip,  "xr_dvar_non_ag")

ag_base = next(iter(ds_ag_base.data_vars.values()))   # (lm, cell, lu)
ag_high = next(iter(ds_ag_high.data_vars.values()))
na_base = next(iter(ds_na_base.data_vars.values()))   # (cell, lu)
na_high = next(iter(ds_na_high.data_vars.values()))

print(f"ag dims: {list(ag_base.dims)}")
print(f"non_ag dims: {list(na_base.dims)}")
print(f"ag lu: {list(ag_base.coords['lu'].values)}")
print(f"non_ag lu: {list(na_base.coords['lu'].values)}")

# Natural land area per cell (sum over lm)
NAT_LU = 'Unallocated - natural land'
nat_base = ag_base.sel(lu=NAT_LU).sum('lm').values   # (cell,)
nat_high = ag_high.sel(lu=NAT_LU).sum('lm').values

# Cells that lost natural land (ha)
lost_ha = nat_base - nat_high
lost_mask = lost_ha > 10  # lost more than 10 ha
print(f"\nTotal cells in output: {len(nat_base)}")
print(f"Cells with natural land in baseline (>10 ha): {(nat_base > 10).sum()}")
print(f"Cells that lost natural land at bp={MAX_BP}: {lost_mask.sum()}")
print(f"Total natural land lost: {lost_ha[lost_mask].sum()/1e6:.3f} Mha")

# Where did those cells go? Check non-ag gains
DESTOCK_LU = 'Destocked - natural land'
EP_LU = 'Environmental Plantings'
# find actual lu names
na_lus = list(na_base.coords['lu'].values)
destock_lu_name = [x for x in na_lus if 'estocked' in x or 'destock' in x.lower()]
ep_lu_names = [x for x in na_lus if 'nvironmental' in x or 'planting' in x.lower()]
print(f"\nDestocked lu names: {destock_lu_name}")
print(f"EP lu names: {ep_lu_names}")

# For cells that lost natural land, what did they gain in non-ag?
if destock_lu_name:
    d_name = destock_lu_name[0]
    destock_gain = (na_high.sel(lu=d_name) - na_base.sel(lu=d_name)).values  # (cell,)
    print(f"\nFor cells that lost natural land:")
    print(f"  Gained destocked land: {destock_gain[lost_mask].sum()/1e6:.4f} Mha")
    for ep in ep_lu_names:
        ep_gain = (na_high.sel(lu=ep) - na_base.sel(lu=ep)).values
        print(f"  Gained {ep}: {ep_gain[lost_mask].sum()/1e6:.4f} Mha")

# Load SAVBURN_ELIGIBLE for these cells
input_dir = Path(__file__).parents[3] / "input"
savburn_file = input_dir / "cell_savanna_burning.h5"
cell_df_file = input_dir / "cell_df.h5"

if savburn_file.exists() and cell_df_file.exists():
    cell_df   = pd.read_hdf(cell_df_file)
    savburn_df = pd.read_hdf(savburn_file)

    # The output cells correspond to MASK rows in cell_df
    # Find which column indicates the mask
    print(f"\ncell_df columns: {list(cell_df.columns)}")
    print(f"savburn_df columns: {list(savburn_df.columns)}")

    # Align: output has n_output_cells rows, input has n_all_cells rows
    n_all   = len(cell_df)
    n_out   = len(nat_base)
    print(f"cell_df rows: {n_all}, output cells: {n_out}")

    if n_all == n_out:
        savburn_elig = savburn_df['ELIGIBLE_AREA'].values
        print(f"\nOf {lost_mask.sum()} cells that lost natural land:")
        print(f"  SAVBURN_ELIGIBLE=1: {savburn_elig[lost_mask].sum():.0f} cells ({100*savburn_elig[lost_mask].mean():.1f}%)")
        print(f"  SAVBURN_ELIGIBLE=0: {(1-savburn_elig[lost_mask]).sum():.0f} cells ({100*(1-savburn_elig[lost_mask]).mean():.1f}%)")

        # Latitude check
        if 'LAT' in cell_df.columns:
            lat = cell_df['LAT'].values
            print(f"\nLatitude of lost-natural-land cells:")
            print(f"  Mean lat: {lat[lost_mask].mean():.2f}")
            print(f"  Min lat:  {lat[lost_mask].min():.2f}")
            print(f"  Max lat:  {lat[lost_mask].max():.2f}")
            print(f"  Cells north of -20°: {(lat[lost_mask] > -20).sum()}")
            print(f"  Cells south of -20°: {(lat[lost_mask] < -20).sum()}")
    else:
        print("Cell count mismatch - skipping SAVBURN check")
else:
    print(f"Input files not found")

# -- Transition costs --
print("\n=== Transition costs ===")
destock_npy = input_dir / "ag_to_destock_tmatrix.npy"
if destock_npy.exists():
    dc = np.load(destock_npy)
    print(f"Destocked transition costs (AUD/ha) by ag lu:")
    print(f"  values: {dc}")
    print(f"  mean={dc.mean():.1f}, min={dc.min():.1f}, max={dc.max():.1f}")
