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
            return None, None
        fname = matches[0]
        with arc.open(fname) as f:
            ds = xr.open_dataset(io.BytesIO(f.read()), engine="h5netcdf")
    if "layer" in ds.dims and "compress" in ds["layer"].attrs:
        ds = cfxr.decode_compress_to_multi_index(ds, "layer")
    return ds.load(), fname

baseline_zip = run_map[(0.0, 0.0)]
high_bp_zip  = run_map.get((0.0, MAX_BP))

for label, zpath in [("baseline", baseline_zip), (f"bp={MAX_BP}", high_bp_zip)]:
    print(f"\n=== {label} Non-ag area ===")
    ds, fname = read_nc(zpath, "xr_area_non_agricultural_landuse")
    if ds is None:
        print("  not found")
        continue
    da = next(iter(ds.data_vars.values()))
    print(f"  file: {fname}")
    print(f"  dims: {list(da.dims)}")
    print(f"  lu values: {list(da.coords.get('lu', da.coords).get('lu', [None]))}")
    # Try to get lu coord
    if 'lu' in da.coords:
        for lu in da.coords['lu'].values:
            if lu == 'ALL':
                continue
            area = float(da.sel(lu=lu).sum())
            if abs(area) > 1000:
                print(f"  {lu}: {area/1e6:.4f} Mha")
    elif 'layer' in da.dims:
        lu_vals = da.coords['lu'].values if 'lu' in da.coords else []
        print(f"  layer lu values: {list(lu_vals)}")
        for lu in lu_vals:
            if lu == 'ALL':
                continue
            try:
                area = float(da.sel(lu=lu).sum())
                if abs(area) > 1000:
                    print(f"  {lu}: {area/1e6:.4f} Mha")
            except Exception as e:
                print(f"  {lu}: error - {e}")

print("\n=== Natural land area (ag) baseline vs high bp ===")
for label, zpath in [("baseline", baseline_zip), (f"bp={MAX_BP}", high_bp_zip)]:
    ds, fname = read_nc(zpath, "xr_area_agricultural_landuse")
    if ds is None:
        continue
    da = next(iter(ds.data_vars.values()))
    if 'lu' in da.coords:
        nat = float(da.sel(lu='Unallocated - natural land').sum())
        print(f"  {label}: natural land = {nat/1e6:.4f} Mha")
    elif 'layer' in da.dims:
        lu_vals = da.coords['lu'].values if 'lu' in da.coords else []
        nat_found = [lu for lu in lu_vals if 'natural' in lu.lower() and 'unalloc' in lu.lower()]
        for lu in nat_found:
            try:
                area = float(da.sel(lu=lu).sum())
                print(f"  {label} {lu}: {area/1e6:.4f} Mha")
            except:
                pass

# Check destocked transition cost by lu name
print("\n=== Destocked transition costs by lu ===")
input_dir = Path(__file__).parents[3] / "input"
destock_npy = input_dir / "ag_to_destock_tmatrix.npy"

# Read ag landuse names from data
ds_ag, _ = read_nc(baseline_zip, "xr_area_agricultural_landuse")
if ds_ag is not None:
    da_ag = next(iter(ds_ag.data_vars.values()))
    if 'lu' in da_ag.coords:
        ag_lus = [lu for lu in da_ag.coords['lu'].values if lu != 'ALL']
    elif 'layer' in da_ag.dims and 'lu' in da_ag.coords:
        ag_lus = [lu for lu in da_ag.coords['lu'].values if lu != 'ALL']
    else:
        ag_lus = None

if destock_npy.exists() and ag_lus:
    dc = np.load(destock_npy)
    print(f"Number of ag lus: {len(ag_lus)}, destock cost array length: {len(dc)}")
    for lu, cost in zip(ag_lus, dc):
        if not np.isnan(cost):
            print(f"  {lu}: {cost:.0f} AUD/ha")
        else:
            print(f"  {lu}: 0 (no cost / NaN)")
