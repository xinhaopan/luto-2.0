# ==============================================================================
# Figure 01: price response curves
#   Panel A: change in GHG emissions vs carbon price
#            (BioPrice=0; price run minus zero-price run)
#   Panel B: change in biodiversity contribution vs biodiversity price
#            (CarbonPrice=0; price run minus zero-price run)
#
#   Values are differences from the zero-price run for YEAR.
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
    apply_compact_ticks,
    DATA_DIR,
    format_thousands,
    OUT_DIR,
    read_sum,
    style_box_axis,
)


YEAR = 2025
CACHE_PATH = DATA_DIR / f"01_Price_Response_Delta_vs_Zero_raw_data_{YEAR}.xlsx"

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
    "axes.facecolor": "#EAEAF2",
    "grid.color": "white",
    "grid.linewidth": 1.0,
})


def collect_slices():
    run_map, cp_vals, bp_vals = build_run_map()
    zero_zip_path = run_map.get((0.0, 0.0))
    if zero_zip_path is None:
        raise FileNotFoundError("Could not find zero-price run (CarbonPrice=0, BioPrice=0).")

    ghg_zero = read_sum(zero_zip_path, GHG_FILES, YEAR) / 1e6
    bio_zero = read_sum(zero_zip_path, BIO_FILES, YEAR)

    rows_ghg = []
    print(f"\n--- GHG emissions change at {YEAR} vs carbon price (BioPrice=0) ---")
    for cp in cp_vals:
        zip_path = run_map.get((cp, 0.0))
        ghg_2025 = read_sum(zip_path, GHG_FILES, YEAR) / 1e6 if zip_path else np.nan
        ghg_delta = ghg_2025 - ghg_zero
        rows_ghg.append({
            "CarbonPrice": cp,
            "GHGEmissions_2025_MtCO2e": ghg_2025,
            "GHGEmissions_ZeroPrice_2025_MtCO2e": ghg_zero,
            "GHGEmissionsChange_vs_ZeroPrice_MtCO2e": ghg_delta,
        })
        print(
            f"  cp={format_thousands(cp)}: "
            f"GHG emissions change={ghg_delta:.1f} Mt CO2e"
        )

    rows_bio = []
    print(f"\n--- Biodiversity contribution change at {YEAR} vs bio price (CarbonPrice=0) ---")
    for bp in bp_vals:
        zip_path = run_map.get((0.0, bp))
        bio_2025 = read_sum(zip_path, BIO_FILES, YEAR) if zip_path else np.nan
        bio_delta = bio_2025 - bio_zero
        rows_bio.append({
            "BioPrice": bp,
            "Bio_2025_ha_yr": bio_2025,
            "BioContribution_2025_ha_yr": bio_2025,
            "BioContribution_ZeroPrice_2025_ha_yr": bio_zero,
            "BioContributionChange_vs_ZeroPrice_ha_yr": bio_delta,
        })
        print(
            f"  bp={format_thousands(bp)}: "
            f"bio contribution change={bio_delta:.2f}"
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
        "GHGEmissionsChange_vs_ZeroPrice_MtCO2e" not in df_ghg.columns
        or "BioContributionChange_vs_ZeroPrice_ha_yr" not in df_bio.columns
    ):
        print("Cached schema is outdated; rebuilding.")
        df_ghg, df_bio = collect_slices()
else:
    df_ghg, df_bio = collect_slices()


x_ghg = df_ghg["GHGEmissionsChange_vs_ZeroPrice_MtCO2e"].to_numpy()
y_cp = df_ghg["CarbonPrice"].to_numpy()

x_bio = df_bio["BioContributionChange_vs_ZeroPrice_ha_yr"].to_numpy() / 1e6
y_bp = df_bio["BioPrice"].to_numpy()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

mask_ghg = ~np.isnan(x_ghg)
ax1.plot(
    x_ghg[mask_ghg],
    y_cp[mask_ghg],
    color=COLOR_GHG,
    marker="o",
    linestyle="-",
    linewidth=2.2,
    markersize=7,
    markeredgewidth=0,
)
ax1.set_title("Carbon price response", pad=8)
ax1.set_xlabel(r"Change in GHG emissions vs zero price (Mt CO$_2$e yr$^{-1}$)")
ax1.set_ylabel(r"Carbon price (AU\$/tCO$_2$e yr$^{-1}$)")
ax1.set_ylim(bottom=0)
apply_price_formatter(ax1, axis="y")
apply_compact_ticks(ax1, x_nbins=7, y_nbins=6)
style_box_axis(ax1)

mask_bio = ~np.isnan(x_bio)
ax2.plot(
    x_bio[mask_bio],
    y_bp[mask_bio],
    color=COLOR_BIO,
    marker="o",
    linestyle="-",
    linewidth=2.2,
    markersize=7,
    markeredgewidth=0,
)
ax2.set_title("Biodiversity price response", pad=8)
ax2.set_xlabel(r"Change in biodiversity contribution score vs zero price (Mha yr$^{-1}$)")
ax2.set_ylabel(r"Biodiversity price (AU\$/ha yr$^{-1}$)")
ax2.set_ylim(bottom=0)
apply_price_formatter(ax2, axis="y")
apply_compact_ticks(ax2, x_nbins=7, y_nbins=6)
style_box_axis(ax2)

plt.tight_layout()
out_path = OUT_DIR / "01_Price_Response_Delta_vs_Zero.png"
fig.savefig(out_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"Saved: {out_path}")
