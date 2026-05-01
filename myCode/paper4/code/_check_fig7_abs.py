"""Check absolute values from max-price run for Figure 7."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tools.price_slice_utils import build_run_map
import io, zipfile
import cf_xarray as cfxr
import numpy as np, pandas as pd, xarray as xr

YEAR = 2025

def open_da(zip_path, fname):
    with zipfile.ZipFile(zip_path) as arc:
        m = [n for n in arc.namelist() if n.endswith(fname)]
        if not m: return None
        with arc.open(m[0]) as f:
            ds = xr.open_dataset(io.BytesIO(f.read()), engine="h5netcdf")
    try:
        if "layer" in ds.dims and "compress" in ds["layer"].attrs:
            ds = cfxr.decode_compress_to_multi_index(ds, "layer")
        return next(iter(ds.data_vars.values())).load()
    finally:
        ds.close()

def total(da, dim, val):
    sub = da.sel({dim: val})
    for c in list(sub.coords):
        if c in {dim, "cell", "layer"}: continue
        try:
            if "ALL" in sub.coords[c].values: sub = sub.sel({c: "ALL"})
        except: pass
    return float(sub.sum())

run_map, cp_vals, bp_vals = build_run_map()
cp_max, bp_max = max(cp_vals), max(bp_vals)

print(f"=== Carbon (max price cp={cp_max}) ===")
max_zip = run_map[(cp_max, 0.0)]
for fname_g, fname_p in [
    (f"xr_GHG_non_ag_{YEAR}.nc", f"xr_economics_non_ag_profit_{YEAR}.nc"),
    (f"xr_GHG_ag_{YEAR}.nc",     f"xr_economics_ag_profit_{YEAR}.nc"),
]:
    da_g = open_da(max_zip, fname_g)
    da_p = open_da(max_zip, fname_p)
    if da_g is None: continue
    dim = "lu" if "lu" in da_g.dims else "am"
    for val in da_g.coords[dim].values:
        if val == "ALL": continue
        g = total(da_g, dim, val)
        p = total(da_p, dim, val) if da_p is not None else 0
        abatement = -g / 1e6
        if abatement > 0.001:
            pol = cp_max * abatement / 1e3
            cost = -(p/1e9 - pol)*1e3/abatement
            print(f"  {val[:35]:35s}  abatement={abatement:8.2f}Mt  profit={p/1e9:8.2f}B  cost={cost:10.1f}")

print(f"\n=== Biodiversity (max price bp={bp_max}) ===")
max_zip2 = run_map[(0.0, bp_max)]
for fname_b, fname_p in [
    (f"xr_biodiversity_overall_priority_non_ag_{YEAR}.nc", f"xr_economics_non_ag_profit_{YEAR}.nc"),
    (f"xr_biodiversity_overall_priority_ag_{YEAR}.nc",     f"xr_economics_ag_profit_{YEAR}.nc"),
]:
    da_b = open_da(max_zip2, fname_b)
    da_p = open_da(max_zip2, fname_p)
    if da_b is None: continue
    dim = "lu" if "lu" in da_b.dims else "am"
    for val in da_b.coords[dim].values:
        if val == "ALL": continue
        b = total(da_b, dim, val)
        p = total(da_p, dim, val) if da_p is not None else 0
        bio = b / 1e6
        if bio > 0.001:
            pol = bp_max * bio / 1e3
            cost = -(p/1e9 - pol)*1e3/bio
            print(f"  {val[:35]:35s}  bio={bio:8.3f}Mha  profit={p/1e9:8.2f}B  cost={cost:10.1f}")
