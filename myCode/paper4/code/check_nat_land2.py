"""Check dvar structure and natural land conversion more carefully."""
import io, sys, zipfile
import cf_xarray as cfxr
import numpy as np
import xarray as xr
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from tools.price_slice_utils import build_run_map

run_map, cp_vals, bp_vals = build_run_map()
YEAR = 2050
MAX_BP = max(bp_vals)

def read_dvar(zip_path, suffix):
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

# Check baseline dvar_ag
ds = read_dvar(baseline_zip, "xr_dvar_ag")
da = next(iter(ds.data_vars.values()))
print("=== Baseline ag dvar ===")
print(f"dims: {list(da.dims)}")
print(f"coords: {list(da.coords)}")
print(f"value range: {float(da.min()):.4f} to {float(da.max()):.4f}")

# Check if 'lm' and 'lu' are separate dims
if 'lm' in da.dims and 'lu' in da.dims:
    nat = da.sel(lu='Unallocated - natural land')
    print(f"\nNatural land (lm x cell): shape={nat.shape}")
    print(f"Natural land total area: {float(nat.sum())/1e6:.2f} Mha")
    nat_total = nat.sum('lm')  # per cell
    print(f"Cells with nat land > 0: {(nat_total > 0.01).sum().item()}")
    print(f"Cells with nat land > 10: {(nat_total > 10).sum().item()}")
    print(f"Max nat land per cell: {float(nat_total.max()):.1f}")
elif 'layer' in da.dims:
    # Multi-index layer
    print(f"layer coords: {da.coords}")
    # Try to access by level
    if hasattr(da, 'indexes') and 'layer' in da.indexes:
        idx = da.indexes['layer']
        print(f"layer index names: {idx.names}")
        print(f"layer index levels: {[list(l) for l in idx.levels]}")

# Check non-ag dvar for both baseline and high bp
print("\n=== Non-ag land uses in baseline vs high bp ===")
for label, zpath in [("baseline", baseline_zip), (f"bp={MAX_BP}", high_bp_zip)]:
    ds_na = read_dvar(zpath, "xr_dvar_non_ag")
    if ds_na is None:
        print(f"{label}: no non-ag dvar")
        continue
    da_na = next(iter(ds_na.data_vars.values()))
    print(f"\n{label} non-ag dims: {list(da_na.dims)}")
    if 'lu' in da_na.dims:
        lus = list(da_na.coords['lu'].values)
        print(f"  lu values: {lus}")
        for lu in lus:
            if lu == 'ALL':
                continue
            area = float(da_na.sel(lu=lu).sum())
            print(f"  {lu}: {area/1e6:.3f} Mha")
    elif 'layer' in da_na.dims:
        print(f"  layer: {da_na.coords}")

# Also check area outputs
print("\n=== Area from area files ===")
for label, zpath in [("baseline", baseline_zip), (f"bp={MAX_BP}", high_bp_zip)]:
    with zipfile.ZipFile(zpath) as arc:
        area_files = [n for n in arc.namelist() if 'xr_area' in n and str(YEAR) in n]
    print(f"{label} area files: {area_files[:5]}")

# Check transition cost - what does it cost to go natural land -> destocked?
print("\n=== Destocked transition costs ===")
input_dir = Path(__file__).parents[3] / "input"
destock_npy = input_dir / "ag_to_destock_tmatrix.npy"
if destock_npy.exists():
    dc = np.load(destock_npy)
    # Load ag lu names from somewhere
    ag_lu_file = input_dir / "ag_landuses.txt"
    if ag_lu_file.exists():
        lus = [l.strip() for l in ag_lu_file.read_text().split('\n') if l.strip()]
    else:
        lus = [f"lu_{i}" for i in range(len(dc))]
    for i, (lu, cost) in enumerate(zip(lus, dc)):
        if not np.isnan(cost):
            print(f"  {lu}: {cost:.0f} AUD/ha")
