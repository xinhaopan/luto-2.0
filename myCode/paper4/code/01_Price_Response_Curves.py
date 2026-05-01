# ==============================================================================
# Figure 01: 2025 price response curves
#   Panel A: GHG emissions vs carbon price                 (BioPrice=0)
#   Panel B: Biodiversity contribution vs biodiversity price (CarbonPrice=0)
#
#   Values are absolute 2025 quantities:
#     GHG emissions = net GHG emissions
#     biodiversity contribution = total biodiversity contribution
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


YEAR = 2025
CACHE_PATH = DATA_DIR / f"01_Price_Response_raw_data_{YEAR}.xlsx"

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

    rows_ghg = []
    print(f"\n--- Absolute GHG emissions at {YEAR} vs carbon price (BioPrice=0) ---")
    for cp in cp_vals:
        zip_path = run_map.get((cp, 0.0))
        ghg_2025 = read_sum(zip_path, GHG_FILES, YEAR) / 1e6 if zip_path else np.nan
        rows_ghg.append({
            "CarbonPrice": cp,
            "GHGEmissions_2025_MtCO2e": ghg_2025,
        })
        print(
            f"  cp={format_thousands(cp)}: "
            f"GHG emissions={ghg_2025:.1f} Mt CO2e"
        )

    rows_bio = []
    print(f"\n--- Absolute biodiversity contribution at {YEAR} vs bio price (CarbonPrice=0) ---")
    for bp in bp_vals:
        zip_path = run_map.get((0.0, bp))
        bio_2025 = read_sum(zip_path, BIO_FILES, YEAR) if zip_path else np.nan
        rows_bio.append({
            "BioPrice": bp,
            "Bio_2025_ha_yr": bio_2025,
            "BioContribution_2025_ha_yr": bio_2025,
        })
        print(
            f"  bp={format_thousands(bp)}: "
            f"Bio2025={bio_2025:.2f}"
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
        "GHGEmissions_2025_MtCO2e" not in df_ghg.columns
        or "BioContribution_2025_ha_yr" not in df_bio.columns
    ):
        print("Cached schema is outdated; rebuilding.")
        df_ghg, df_bio = collect_slices()
else:
    df_ghg, df_bio = collect_slices()


x_ghg = df_ghg["GHGEmissions_2025_MtCO2e"].to_numpy()
y_cp = df_ghg["CarbonPrice"].to_numpy()

x_bio = df_bio["BioContribution_2025_ha_yr"].to_numpy() / 1e6
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
ax1.set_xlabel(r"GHG emissions (Mt CO$_2$e yr$^{-1}$)")
ax1.set_ylabel(r"Carbon price (AU\$/tCO$_2$e yr$^{-1}$)")
ax1.set_ylim(bottom=0)
if mask_ghg.any() and np.nanmin(x_ghg[mask_ghg]) < 0.0 < np.nanmax(x_ghg[mask_ghg]):
    ax1.axvline(0.0, color="#444444", linewidth=0.8)
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
ax2.set_xlabel(r"Biodiversity contribution score (Mha yr$^{-1}$)")
ax2.set_ylabel(r"Biodiversity price (AU\$/ha yr$^{-1}$)")
ax2.set_ylim(bottom=0)
apply_price_formatter(ax2, axis="y")
style_box_axis(ax2)

plt.tight_layout()
out_path = OUT_DIR / f"01_Price_Response_Curves_{YEAR}.png"
fig.savefig(out_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"Saved: {out_path}")
