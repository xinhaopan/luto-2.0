"""Check non-ag lu names and area files directly."""
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

def print_nonag_area(label, zpath):
    ds = read_nc(zpath, "xr_area_non_agricultural_landuse")
    if ds is None:
        print(f"  {label}: not found")
        return
    da = next(iter(ds.data_vars.values()))
    print(f"\n=== {label} Non-ag area, dims={list(da.dims)} ===")
    lu_coord = da.coords['lu'] if 'lu' in da.coords else None
    if lu_coord is None:
        print("  no lu coord")
        return
    for lu in lu_coord.values:
        if lu == 'ALL':
            continue
        area = float(da.sel(lu=lu).sum())
        print(f"  {lu}: {area/1e6:.4f} Mha")

def print_ag_nat(label, zpath):
    ds = read_nc(zpath, "xr_area_agricultural_landuse")
    if ds is None:
        return
    da = next(iter(ds.data_vars.values()))
    lu_coord = da.coords['lu'] if 'lu' in da.coords else None
    if lu_coord is None:
        return
    for lu in lu_coord.values:
        if 'natural' in lu.lower():
            area = float(da.sel(lu=lu).sum())
            print(f"  {label} - {lu}: {area/1e6:.4f} Mha")

print("=== Agricultural natural land area ===")
for label, zpath in [("baseline (cp=0,bp=0)", baseline_zip), (f"high bp={MAX_BP}", high_bp_zip)]:
    print_ag_nat(label, zpath)

print_nonag_area("baseline (cp=0,bp=0)", baseline_zip)
print_nonag_area(f"high bp={MAX_BP}", high_bp_zip)

# Destocked transition costs
print("\n=== Destocked transition costs by lu ===")
input_dir = Path(__file__).parents[3] / "input"
destock_npy = input_dir / "ag_to_destock_tmatrix.npy"

ds_ag = read_nc(baseline_zip, "xr_area_agricultural_landuse")
da_ag = next(iter(ds_ag.data_vars.values()))
lu_coord = da_ag.coords['lu'] if 'lu' in da_ag.coords else None
ag_lus = [lu for lu in lu_coord.values if lu != 'ALL'] if lu_coord is not None else None

if destock_npy.exists() and ag_lus is not None:
    dc = np.load(destock_npy)
    print(f"ag lus ({len(ag_lus)}), destock costs ({len(dc)})")
    for lu, cost in zip(ag_lus, dc):
        tag = f"{cost:.0f} AUD/ha" if not np.isnan(cost) else "NaN (free or not allowed)"
        if not np.isnan(cost):
            print(f"  {lu}: {tag}")
