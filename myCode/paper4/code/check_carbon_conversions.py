"""
Analyze land use conversions driven by carbon price (bp=0, cp varies).
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
MAX_CP = max(cp_vals)
input_dir = Path(__file__).parents[3] / "input"

def read_nc(zip_path, suffix):
    with zipfile.ZipFile(zip_path) as arc:
        matches = [n for n in arc.namelist() if suffix in n and str(YEAR) in n]
        if not matches: return None
        with arc.open(matches[0]) as f:
            ds = xr.open_dataset(io.BytesIO(f.read()), engine="h5netcdf")
    if "layer" in ds.dims and "compress" in ds["layer"].attrs:
        ds = cfxr.decode_compress_to_multi_index(ds, "layer")
    return ds.load()

baseline_zip = run_map[(0.0, 0.0)]
high_cp_zip  = run_map.get((MAX_CP, 0.0))
print(f"Comparing baseline (cp=0,bp=0) vs high cp={MAX_CP} (bp=0)\n")

# ── Ag area changes ───────────────────────────────────────────────────────────
print("=== Agricultural land-use area changes (Mha) ===")
da_ag_base = next(iter(read_nc(baseline_zip, "xr_area_agricultural_landuse").data_vars.values()))
da_ag_high = next(iter(read_nc(high_cp_zip,  "xr_area_agricultural_landuse").data_vars.values()))

for lu in da_ag_base.coords['lu'].values:
    if lu == 'ALL': continue
    base = float(da_ag_base.sel(lu=lu).sum('lm').sum()) / 1e6
    high = float(da_ag_high.sel(lu=lu).sum('lm').sum()) / 1e6
    delta = high - base
    if abs(delta) > 0.05:
        print(f"  {lu:40s}: {base:7.2f} → {high:7.2f}  (Δ{delta:+.2f} Mha)")

# ── Non-ag area changes ───────────────────────────────────────────────────────
print("\n=== Non-agricultural land-use area changes (Mha) ===")
da_na_base = next(iter(read_nc(baseline_zip, "xr_area_non_agricultural_landuse").data_vars.values()))
da_na_high = next(iter(read_nc(high_cp_zip,  "xr_area_non_agricultural_landuse").data_vars.values()))
base_na_lus = set(da_na_base.coords['lu'].values)

for lu in da_na_high.coords['lu'].values:
    if lu == 'ALL': continue
    high = float(da_na_high.sel(lu=lu).sum()) / 1e6
    base = float(da_na_base.sel(lu=lu).sum()) / 1e6 if lu in base_na_lus else 0.0
    delta = high - base
    if abs(delta) > 0.01:
        print(f"  {lu:40s}: {base:7.2f} → {high:7.2f}  (Δ{delta:+.2f} Mha)")

# ── Ag management changes ─────────────────────────────────────────────────────
print("\n=== Agricultural management changes (Mha) ===")
ds_am_base = read_nc(baseline_zip, "xr_area_agricultural_management")
ds_am_high = read_nc(high_cp_zip,  "xr_area_agricultural_management")
if ds_am_base and ds_am_high:
    da_am_base = next(iter(ds_am_base.data_vars.values()))
    da_am_high = next(iter(ds_am_high.data_vars.values()))
    base_am_lus = set(da_am_base.coords['am'].values) if 'am' in da_am_base.coords else set()
    am_coord = 'am' if 'am' in da_am_high.coords else 'lu'
    for am in da_am_high.coords[am_coord].values:
        if am == 'ALL': continue
        high = float(da_am_high.sel({am_coord: am}).sum()) / 1e6
        base = float(da_am_base.sel({am_coord: am}).sum()) / 1e6 if am in base_am_lus else 0.0
        delta = high - base
        if abs(delta) > 0.05:
            print(f"  {am:40s}: {base:7.2f} → {high:7.2f}  (Δ{delta:+.2f} Mha)")

# ── Compare at multiple price levels ─────────────────────────────────────────
print("\n=== Area trends across carbon prices ===")
key_lus_ag = ['Beef - modified land', 'Beef - natural land', 'Sheep - modified land',
              'Sheep - natural land', 'Unallocated - natural land', 'Unallocated - modified land']
key_lus_na = ['Environmental Plantings', 'Destocked - natural land',
              'Riparian Plantings', 'Carbon Plantings (Block)', 'BECCS']

print(f"{'Land use':<40}", end="")
for cp in sorted(cp_vals):
    print(f"  cp={int(cp):>6}", end="")
print()

for lu in key_lus_ag:
    print(f"  {lu:<38}", end="")
    for cp in sorted(cp_vals):
        zp = run_map.get((cp, 0.0))
        if zp is None:
            print(f"  {'N/A':>9}", end=""); continue
        da = next(iter(read_nc(zp, "xr_area_agricultural_landuse").data_vars.values()))
        if lu in da.coords['lu'].values:
            val = float(da.sel(lu=lu).sum('lm').sum()) / 1e6
            print(f"  {val:9.2f}", end="")
        else:
            print(f"  {'N/A':>9}", end="")
    print()

for lu in key_lus_na:
    print(f"  {lu:<38}", end="")
    for cp in sorted(cp_vals):
        zp = run_map.get((cp, 0.0))
        if zp is None:
            print(f"  {'N/A':>9}", end=""); continue
        da_na = next(iter(read_nc(zp, "xr_area_non_agricultural_landuse").data_vars.values()))
        if lu in da_na.coords['lu'].values:
            val = float(da_na.sel(lu=lu).sum()) / 1e6
            print(f"  {val:9.2f}", end="")
        else:
            print(f"  {0:9.2f}", end="")
    print()

# ── Transition costs for carbon-driven options ────────────────────────────────
print("\n=== Carbon sequestration options transition costs ===")
ep_npy = input_dir / "ag_to_ep_tmatrix.npy"
if ep_npy.exists():
    ep_costs = np.load(ep_npy)
    destock_costs = np.load(input_dir / "ag_to_destock_tmatrix.npy")
    da_ag = next(iter(read_nc(baseline_zip, "xr_area_agricultural_landuse").data_vars.values()))
    ag_lus = [lu for lu in da_ag.coords['lu'].values
              if lu not in ('ALL',) and 'lm' not in str(lu)]
    seen, unique_lus = set(), []
    for lu in ag_lus:
        if lu not in seen: unique_lus.append(lu); seen.add(lu)
    print(f"{'Land use':<40} EP_cost  Destock_cost")
    for lu, ep, ds in zip(unique_lus, ep_costs, destock_costs):
        if not np.isnan(ep) or not np.isnan(ds):
            ep_s  = f"{ep:.0f}" if not np.isnan(ep) else "N/A"
            ds_s  = f"{ds:.0f}" if not np.isnan(ds) else "N/A"
            print(f"  {lu:<40} {ep_s:>8}  {ds_s:>12}")
