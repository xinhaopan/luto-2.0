"""Check actual dims in GHG/bio/profit NetCDF files."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tools.price_slice_utils import build_run_map
import io, zipfile, xarray as xr, cf_xarray as cfxr

YEAR = 2025
run_map, cp_vals, bp_vals = build_run_map()
cp_max, bp_max = max(cp_vals), max(bp_vals)

def open_da(zip_path, fname):
    with zipfile.ZipFile(zip_path) as arc:
        m = [n for n in arc.namelist() if n.endswith(fname)]
        if not m: return None
        with arc.open(m[0]) as f:
            ds = xr.open_dataset(io.BytesIO(f.read()), engine='h5netcdf')
    try:
        if 'layer' in ds.dims and 'compress' in ds['layer'].attrs:
            ds = cfxr.decode_compress_to_multi_index(ds, 'layer')
        da = next(iter(ds.data_vars.values())).load()
        return da
    finally:
        ds.close()

max_zip = run_map[(cp_max, 0.0)]
fnames = [
    f'xr_GHG_non_ag_{YEAR}.nc',
    f'xr_GHG_ag_{YEAR}.nc',
    f'xr_economics_non_ag_profit_{YEAR}.nc',
    f'xr_economics_ag_profit_{YEAR}.nc',
]
print(f"=== Carbon max-price zip: {max_zip} ===")
for fname in fnames:
    da = open_da(max_zip, fname)
    if da is not None:
        print(f"  {fname}: dims={list(da.dims)}, coords={list(da.coords)}")
        for dim in da.dims:
            if dim not in ('cell',):
                print(f"    {dim} values: {list(da.coords[dim].values[:10])}")
    else:
        print(f"  {fname}: NOT FOUND")

print()
max_zip2 = run_map[(0.0, bp_max)]
fnames2 = [
    f'xr_biodiversity_overall_priority_non_ag_{YEAR}.nc',
    f'xr_biodiversity_overall_priority_ag_{YEAR}.nc',
    f'xr_economics_non_ag_profit_{YEAR}.nc',
    f'xr_economics_ag_profit_{YEAR}.nc',
]
print(f"=== Bio max-price zip: {max_zip2} ===")
for fname in fnames2:
    da = open_da(max_zip2, fname)
    if da is not None:
        print(f"  {fname}: dims={list(da.dims)}, coords={list(da.coords)}")
        for dim in da.dims:
            if dim not in ('cell',):
                print(f"    {dim} values: {list(da.coords[dim].values[:10])}")
    else:
        print(f"  {fname}: NOT FOUND")
