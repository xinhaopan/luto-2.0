# ==============================================================================
# Figure 1: MACC-style figure with two panels
#   Panel A: GHG abatement since 2010 vs carbon price      (BioPrice=0)
#   Panel B: Biodiversity change since 2010 vs bio price   (CarbonPrice=0)
# ==============================================================================

import os
import sys

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tools.price_slice_utils import (
    apply_price_formatter,
    build_run_map,
    DATA_DIR,
    format_thousands,
    OUT_DIR,
    read_sum,
    style_box_axis,
)


YEAR = 2050
CACHE_PATH = DATA_DIR / f"1_MACC_raw_data_{YEAR}.xlsx"

GHG_FILES = [
    "xr_GHG_ag",
    "xr_GHG_ag_management",
    "xr_GHG_non_ag",
    "xr_transition_GHG",
]
BIO_FILES = [
    "xr_biodiversity_overall_priority_ag",
    "xr_biodiversity_overall_priority_ag_management",
    "xr_biodiversity_overall_priority_non_ag",
]

COLOR_GHG = "#1d52a1"
COLOR_BIO = "#72c15a"

FS = 11
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial"],
    "font.size": FS,
    "axes.titlesize": FS,
    "axes.labelsize": FS,
    "xtick.labelsize": FS,
    "ytick.labelsize": FS,
    "legend.fontsize": FS,
    "mathtext.fontset": "stixsans",
})


def collect_slices():
    run_map, cp_vals, bp_vals = build_run_map()
    baseline_zip = run_map.get((0.0, 0.0))

    rows_ghg = []
    print(f"\n--- GHG abatement vs baseline (cp=0, bp=0) at {YEAR} vs carbon price (BioPrice=0) ---")
    ghg_baseline = read_sum(baseline_zip, GHG_FILES, YEAR) / 1e6 if baseline_zip else np.nan
    for cp in cp_vals:
        zip_path = run_map.get((cp, 0.0))
        ghg_2050 = read_sum(zip_path, GHG_FILES, YEAR) / 1e6 if zip_path else np.nan
        abatement = ghg_baseline - ghg_2050 if not np.isnan(ghg_2050) else np.nan
        rows_ghg.append({
            "CarbonPrice": cp,
            "GHG_Baseline_MtCO2e": ghg_baseline,
            "GHG_2050_MtCO2e": ghg_2050,
            "GHGAbatement_vs_Baseline_MtCO2e": abatement,
        })
        print(
            f"  cp={format_thousands(cp)}: "
            f"GHG_baseline={ghg_baseline:.1f}, GHG2050={ghg_2050:.1f}, "
            f"abatement={abatement:.1f} Mt CO2e"
        )

    rows_bio = []
    print(f"\n--- Biodiversity change vs baseline (cp=0, bp=0) at {YEAR} vs bio price (CarbonPrice=0) ---")
    bio_baseline = read_sum(baseline_zip, BIO_FILES, YEAR) if baseline_zip else np.nan
    for bp in bp_vals:
        zip_path = run_map.get((0.0, bp))
        bio_2050 = read_sum(zip_path, BIO_FILES, YEAR) if zip_path else np.nan
        dbio = bio_2050 - bio_baseline if not np.isnan(bio_2050) else np.nan
        rows_bio.append({
            "BioPrice": bp,
            "Bio_Baseline_ha_yr": bio_baseline,
            "Bio_2050_ha_yr": bio_2050,
            "BioChange_vs_Baseline_ha_yr": dbio,
        })
        print(
            f"  bp={format_thousands(bp)}: "
            f"Bio_baseline={bio_baseline:.2f}, Bio2050={bio_2050:.2f}, "
            f"change={dbio:.2f}"
        )

    df_ghg = pd.DataFrame(rows_ghg)
    df_bio = pd.DataFrame(rows_bio)

    with pd.ExcelWriter(CACHE_PATH, engine="openpyxl") as writer:
        df_ghg.to_excel(writer, sheet_name="GHG_slice", index=False)
        df_bio.to_excel(writer, sheet_name="Bio_slice", index=False)

    print(f"\nCache saved: {CACHE_PATH}")
    return df_ghg, df_bio


if CACHE_PATH.is_file():
    print(f"Loading cached data from {CACHE_PATH}")
    df_ghg = pd.read_excel(CACHE_PATH, sheet_name="GHG_slice")
    df_bio = pd.read_excel(CACHE_PATH, sheet_name="Bio_slice")
    if (
        "GHGAbatement_vs_Baseline_MtCO2e" not in df_ghg.columns
        or "BioChange_vs_Baseline_ha_yr" not in df_bio.columns
    ):
        print("Cached schema is outdated; rebuilding.")
        df_ghg, df_bio = collect_slices()
else:
    df_ghg, df_bio = collect_slices()


x_ghg = df_ghg["GHGAbatement_vs_Baseline_MtCO2e"].to_numpy()
y_cp = df_ghg["CarbonPrice"].to_numpy()

x_bio = df_bio["BioChange_vs_Baseline_ha_yr"].to_numpy() / 1e6
y_bp = df_bio["BioPrice"].to_numpy()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

mask_ghg = ~np.isnan(x_ghg)
ax1.plot(
    x_ghg[mask_ghg],
    y_cp[mask_ghg],
    color=COLOR_GHG,
    marker="o",
    linestyle="-",
    linewidth=1.5,
    markersize=5,
)
ax1.set_xlabel(r"GHG abatement vs. 2050 baseline (Mt CO$_2$e)")
ax1.set_ylabel(r"Carbon price (AU\$/tCO$_2$e)")
ax1.set_xlim(left=0)
ax1.set_ylim(bottom=0)
apply_price_formatter(ax1, axis="y")
style_box_axis(ax1)

mask_bio = ~np.isnan(x_bio)
ax2.plot(
    x_bio[mask_bio],
    y_bp[mask_bio],
    color=COLOR_BIO,
    marker="o",
    linestyle="-",
    linewidth=1.5,
    markersize=5,
)
ax2.set_xlabel(r"Biodiversity change vs. 2050 baseline (Mha yr$^{-1}$)")
ax2.set_ylabel(r"Biodiversity price (AU\$/ha)")
ax2.set_ylim(bottom=0)
apply_price_formatter(ax2, axis="y")
style_box_axis(ax2)

plt.tight_layout()
out_path = OUT_DIR / f"1_MACC_GHG_Bio_{YEAR}.png"
fig.savefig(out_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"Saved: {out_path}")
