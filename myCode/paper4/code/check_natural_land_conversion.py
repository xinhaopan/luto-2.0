"""
Check: where are natural land cells being converted to other uses?
Are they in SAVBURN_ELIGIBLE areas?
Also check transition costs: destocked vs EP.
"""
import io, sys, zipfile
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).parent))
from tools.price_slice_utils import build_run_map

run_map, cp_vals, bp_vals = build_run_map()
YEAR = 2050

def read_nc(zip_path, suffix):
    with zipfile.ZipFile(zip_path) as arc:
        matches = [n for n in arc.namelist() if n.endswith(suffix)]
        if not matches:
            return None
        with arc.open(matches[0]) as f:
            ds = xr.open_dataset(io.BytesIO(f.read()), engine="h5netcdf")
    return ds.load()

baseline_zip = run_map[(0.0, 0.0)]

# Read DVAR (land use decision variable) for baseline and high bio price
print("Reading baseline DVAR...")
ds_base = read_nc(baseline_zip, f"xr_ag_dvar_{YEAR}.nc")

# Use highest bio price
MAX_BP = max(bp_vals)
high_bp_zip = run_map.get((0.0, MAX_BP))
print(f"Reading high bp={MAX_BP} DVAR...")
ds_high = read_nc(high_bp_zip, f"xr_ag_dvar_{YEAR}.nc")

if ds_base is None or ds_high is None:
    print("Could not read DVAR files")
    sys.exit(1)

# Get the dominant land use per cell in baseline and high bp
dvar_base = next(iter(ds_base.data_vars.values()))  # (lm, cell, lu)
dvar_high = next(iter(ds_high.data_vars.values()))

# Sum over lm to get area per cell per lu
area_base = dvar_base.sum('lm')  # (cell, lu)
area_high = dvar_high.sum('lm')

# Find cells that are natural land in baseline (>0 area) and less in high bp
nat_lu = 'Unallocated - natural land'
if nat_lu in area_base.coords['lu'].values:
    nat_base = area_base.sel(lu=nat_lu).values  # fraction per cell
    nat_high = area_high.sel(lu=nat_lu).values

    # Cells that lost natural land
    lost_mask = (nat_base > 0.1) & (nat_high < nat_base - 0.05)
    print(f"\nCells with natural land at baseline: {(nat_base > 0.1).sum()}")
    print(f"Cells that lost natural land at bp={MAX_BP}: {lost_mask.sum()}")

    # Now check SAVBURN_ELIGIBLE for these cells
    # Read from zip - look for savburn or cell info
    # Try to find SAVBURN info from the output
    print("\nChecking if we can get SAVBURN info from output...")

    # Read non-ag dvar for high bp to see what they converted to
    ds_nonag_high = read_nc(high_bp_zip, f"xr_non_ag_dvar_{YEAR}.nc")
    if ds_nonag_high is not None:
        dvar_nonag = next(iter(ds_nonag_high.data_vars.values()))  # (cell, lu)
        print(f"Non-ag land uses: {list(dvar_nonag.coords['lu'].values)}")

        ds_nonag_base = read_nc(baseline_zip, f"xr_non_ag_dvar_{YEAR}.nc")
        dvar_nonag_base = next(iter(ds_nonag_base.data_vars.values()))

        for lu in dvar_nonag.coords['lu'].values:
            delta = float((dvar_nonag.sel(lu=lu) - dvar_nonag_base.sel(lu=lu)).sum())
            if abs(delta) > 100:  # ha
                print(f"  {lu}: delta = {delta/1e6:.3f} Mha")

    # Check cell indices of lost natural land and cross-check with savburn
    lost_cell_indices = np.where(lost_mask)[0]
    print(f"\nLost natural land cell indices (first 10): {lost_cell_indices[:10]}")
    print(f"These cells cover area: {lost_mask.sum()} cells")

    # Try to read SAVBURN eligible from a data file
    input_dir = Path(__file__).parents[3] / "input"
    savburn_file = input_dir / "cell_savanna_burning.h5"
    if savburn_file.exists():
        savburn_df = pd.read_hdf(savburn_file)
        savburn_eligible = savburn_df['ELIGIBLE_AREA'].values

        # Match cell indices - need to know if these are masked cells
        # The cell index in the output is within the MASK, not all cells
        # Let's check if lost cells overlap with SAVBURN_ELIGIBLE
        n_cells_total = len(savburn_eligible)
        n_cells_output = len(nat_base)
        print(f"\nTotal cells in input: {n_cells_total}, cells in output: {n_cells_output}")

        if n_cells_total == n_cells_output:
            savburn_in_lost = savburn_eligible[lost_cell_indices]
            print(f"Of {lost_mask.sum()} lost-natural-land cells:")
            print(f"  In SAVBURN_ELIGIBLE areas: {savburn_in_lost.sum()} ({100*savburn_in_lost.mean():.1f}%)")
            print(f"  NOT in SAVBURN_ELIGIBLE: {(1-savburn_in_lost).sum()} ({100*(1-savburn_in_lost).mean():.1f}%)")
        else:
            print(f"Cell count mismatch - need MASK to align")
            # Try to find MASK info
            cell_df_file = input_dir / "cell_df.h5"
            if cell_df_file.exists():
                cell_df = pd.read_hdf(cell_df_file)
                mask = cell_df['MASK'].values if 'MASK' in cell_df.columns else None
                if mask is not None:
                    masked_savburn = savburn_eligible[mask.astype(bool)]
                    savburn_in_lost = masked_savburn[lost_cell_indices]
                    print(f"Of {lost_mask.sum()} lost-natural-land cells:")
                    print(f"  In SAVBURN_ELIGIBLE areas: {savburn_in_lost.sum()} ({100*savburn_in_lost.mean():.1f}%)")
    else:
        print(f"SAVBURN file not found at {savburn_file}")

        # Alternative: check latitude of cells using coordinates
        cell_lat_file = input_dir / "cell_df.h5"
        if cell_lat_file.exists():
            cell_df = pd.read_hdf(cell_lat_file)
            print(f"Cell df columns: {list(cell_df.columns)[:10]}")

# Also check transition costs: destocked vs EP
print("\n=== Transition costs: Destocked vs EP ===")
input_dir = Path(__file__).parents[3] / "input"
destock_file = input_dir / "ag_to_destock_tmatrix.npy"
if destock_file.exists():
    destock_costs = np.load(destock_file)
    print(f"Destocked transition costs (per ag lu): {destock_costs}")
    print(f"  Mean: {destock_costs.mean():.1f}, Min: {destock_costs.min():.1f}, Max: {destock_costs.max():.1f}")
else:
    print(f"Destock tmatrix not found at {destock_file}")
