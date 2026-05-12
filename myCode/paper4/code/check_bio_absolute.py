"""
Read the baseline run (cp=0, bp=0) and compute absolute per-ha biodiversity
score for each land-use category.
"""
import io, re, sys, zipfile
from pathlib import Path
import cf_xarray as cfxr
import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).parent))
from tools.price_slice_utils import build_run_map

run_map, cp_vals, bp_vals = build_run_map()
baseline_zip = run_map[(0.0, 0.0)]
YEAR = 2050

GROUP_FILE = Path(__file__).parent.parents[1] / "draw_all" / "code" / "tools" / "land use group.xlsx"
group_df = pd.read_excel(GROUP_FILE)

def normalize(v):
    return re.sub(r"[\s\-]+", "", str(v).strip().lower())

LU_TO_AG_GROUP = {
    normalize(row["desc"]): row["ag_group"]
    for _, row in group_df.iterrows()
    if pd.notna(row.get("desc")) and pd.notna(row.get("ag_group"))
}

def read_da(zip_path, target_suffix):
    with zipfile.ZipFile(zip_path) as arc:
        matches = [n for n in arc.namelist() if n.endswith(target_suffix)]
        if not matches:
            return None
        with arc.open(matches[0]) as f:
            ds = xr.open_dataset(io.BytesIO(f.read()), engine="h5netcdf")
    try:
        if "layer" in ds.dims and "compress" in ds["layer"].attrs:
            ds = cfxr.decode_compress_to_multi_index(ds, "layer")
        return next(iter(ds.data_vars.values())).load()
    finally:
        ds.close()

def sum_sel(da, **sel):
    sub = da.sel(sel)
    for c in list(sub.coords):
        if c in {"cell","layer"} or c in sel:
            continue
        vals = sub.coords[c].values
        try:
            if "ALL" in vals:
                sub = sub.sel({c: "ALL"})
        except TypeError:
            pass
    return float(sub.sum())

# Ag land-use: absolute bio score and area at baseline
print("=== Ag land-use: absolute bio score/ha at BASELINE (cp=0, bp=0) ===")
bio_ag = read_da(baseline_zip, f"xr_biodiversity_overall_priority_ag_{YEAR}.nc")
area_ag = read_da(baseline_zip, f"xr_area_agricultural_landuse_{YEAR}.nc")

if bio_ag is not None and area_ag is not None:
    results = {}
    for lu in pd.unique(area_ag.coords["lu"].values):
        if lu == "ALL":
            continue
        bio_val = sum_sel(bio_ag, lu=lu)
        area_val = sum_sel(area_ag, lu=lu) / 1e6  # Mha
        group = LU_TO_AG_GROUP.get(normalize(lu), lu)
        if group not in results:
            results[group] = {"bio": 0.0, "area": 0.0}
        results[group]["bio"] += bio_val
        results[group]["area"] += area_val
    for g, v in sorted(results.items(), key=lambda x: x[1]["area"], reverse=True):
        if v["area"] > 0.01:
            per_ha = v["bio"] / (v["area"] * 1e6) if v["area"] > 0 else 0
            print(f"  {g:40s} area={v['area']:8.2f} Mha  bio/ha={per_ha:.6f}")

print()
# Non-ag land-use
print("=== Non-ag land-use: absolute bio score/ha at BASELINE ===")
bio_nonag = read_da(baseline_zip, f"xr_biodiversity_overall_priority_non_ag_{YEAR}.nc")
area_nonag = read_da(baseline_zip, f"xr_area_non_agricultural_landuse_{YEAR}.nc")

if bio_nonag is not None and area_nonag is not None:
    NON_AG_EXCLUDE = {"agriculturallanduse", "otherlanduse"}
    for lu in pd.unique(area_nonag.coords["lu"].values):
        if lu == "ALL" or normalize(lu) in NON_AG_EXCLUDE:
            continue
        bio_val = sum_sel(bio_nonag, lu=lu)
        area_val = sum_sel(area_nonag, lu=lu) / 1e6
        if area_val > 0.001:
            per_ha = bio_val / (area_val * 1e6)
            print(f"  {lu:50s} area={area_val:8.2f} Mha  bio/ha={per_ha:.6f}")
