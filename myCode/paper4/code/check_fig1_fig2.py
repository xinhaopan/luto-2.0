"""Check fig1 total bio score and fig2 NER breakdown."""
import io, sys, zipfile
import cf_xarray as cfxr
import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from tools.price_slice_utils import build_run_map, read_sum

run_map, cp_vals, bp_vals = build_run_map()
YEAR = 2050
DATA_DIR = Path(__file__).parents[1] / "data"

# ── Fig1: total bio score (absolute) vs bio price ────────────────────────────
BIO_FILES = [
    "xr_biodiversity_overall_priority_ag",
    "xr_biodiversity_overall_priority_ag_management",
    "xr_biodiversity_overall_priority_non_ag",
]

print("=== Fig1: Absolute total bio score vs bio price (bp slice, cp=0) ===")
baseline_zip = run_map[(0.0, 0.0)]
bio_baseline = read_sum(baseline_zip, BIO_FILES, YEAR)
print(f"Baseline total bio score: {bio_baseline/1e6:.2f} Mha yr")

print(f"\n{'bp':>10}  {'total_bio (Mha yr)':>20}  {'change (Mha yr)':>17}  {'%change':>8}  {'converted area (Mha)':>22}")
for bp in sorted(bp_vals):
    zp = run_map.get((0.0, bp))
    if zp is None: continue
    bio = read_sum(zp, BIO_FILES, YEAR)
    delta = bio - bio_baseline

    # Get total non-ag area (proxy for converted area)
    def read_nc(zip_path, suffix):
        with zipfile.ZipFile(zip_path) as arc:
            matches = [n for n in arc.namelist() if suffix in n and str(YEAR) in n]
            if not matches: return None
            with arc.open(matches[0]) as f:
                ds = xr.open_dataset(io.BytesIO(f.read()), engine="h5netcdf")
        if "layer" in ds.dims and "compress" in ds["layer"].attrs:
            ds = cfxr.decode_compress_to_multi_index(ds, "layer")
        return ds.load()

    ds_na = read_nc(zp, "xr_area_non_agricultural_landuse")
    da_na = next(iter(ds_na.data_vars.values()))
    ep_area = 0
    for lu in da_na.coords['lu'].values:
        if lu not in ('ALL', 'Destocked - natural land', 'Sheep Agroforestry', 'Beef Agroforestry'):
            ep_area += float(da_na.sel(lu=lu).sum()) / 1e6

    print(f"  {bp:10.0f}  {bio/1e6:20.2f}  {delta/1e6:17.2f}  {100*delta/bio_baseline:8.2f}%  {ep_area:22.2f}")

# ── Fig2: NER breakdown ───────────────────────────────────────────────────────
print("\n=== Fig2: NER breakdown vs bio price ===")
CACHE = DATA_DIR / f"2_NetEcon_raw_data_{YEAR}.xlsx"
if CACHE.exists():
    df_bp = pd.read_excel(CACHE, sheet_name="Bio_slice")
    print(df_bp.to_string(index=False))
else:
    print("Cache not found, reading from raw data...")
    ECON_FILES_AG = ["xr_economics_ag_revenue", "xr_economics_ag_cost",
                     "xr_economics_ag_transition"]
    ECON_FILES_NA = ["xr_economics_non_ag_revenue", "xr_economics_non_ag_cost",
                     "xr_economics_non_ag_transition"]

    rows = []
    for bp in sorted(bp_vals):
        zp = run_map.get((0.0, bp))
        if zp is None: continue
        ag_rev = read_sum(zp, ["xr_economics_ag_revenue"], YEAR)
        ag_cost = read_sum(zp, ["xr_economics_ag_cost"], YEAR)
        na_rev = read_sum(zp, ["xr_economics_non_ag_revenue"], YEAR)
        na_cost = read_sum(zp, ["xr_economics_non_ag_cost"], YEAR)
        bio_val = read_sum(zp, BIO_FILES, YEAR)
        bio_payment = bp * bio_val / 1e9  # Billion AUD
        ag_econ = (ag_rev - ag_cost) / 1e9
        na_econ = (na_rev - na_cost) / 1e9
        ner = ag_econ + na_econ  # this already includes bio payment if in revenue
        rows.append({'bp': bp, 'ag_econ_B': ag_econ, 'na_econ_B': na_econ,
                     'bio_payment_B': bio_payment, 'bio_Mha_yr': bio_val/1e6})
        print(f"  bp={bp:8.0f}: ag_econ={ag_econ:8.2f}B, na_econ={na_econ:8.2f}B, "
              f"bio_score={bio_val/1e6:.2f}Mha_yr, bio_payment={bio_payment:.1f}B")
