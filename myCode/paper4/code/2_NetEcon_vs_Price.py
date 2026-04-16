# ==============================================================================
# Figure A: Change in net economic return since 2010 vs carbon price
#           (BioPrice=0)
# Figure B: Change in net economic return since 2010 vs biodiversity price
#           (CarbonPrice=0)
#
# Notes on accounting:
#   - luto/solvers/solver.py builds the economy objective from
#     `self._input_data.economic_contr_mrj`.
#   - Those solver-side economic coefficients are assembled in
#     luto/solvers/input_data.py, where biodiversity price is monetised as
#     bio_score x bio_price and added to agricultural / ag-management / non-ag
#     revenue before the solver objective is formed.
#   - Carbon price is already embedded in the archived `xr_economics_*_profit`
#     outputs used by this figure, so the carbon-price slice can use those
#     profit outputs directly.
#   - For the archived paper4 runs, the biodiversity-price component should not
#     be assumed to be present in `xr_economics_*_profit`, even though the solver
#     objective did include it. To match solver-side accounting, this script adds
#     back biodiversity payment for the biodiversity-price slice.
#   - This figure now shows the 2050 minus 2010 change within each run. The 2010
#     slice is treated as the historical no-policy anchor for these paper4 scans,
#     so the explicit biodiversity-payment add-back is only applied to 2050.
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
    get_price_axis_label,
    OUT_DIR,
    read_sum,
    style_box_axis,
)


BASE_YEAR = 2025
YEAR = 2050
CACHE_PATH = DATA_DIR / f"2_NetEcon_raw_data_{YEAR}.xlsx"

PROFIT_FILES = [
    "xr_economics_ag_profit",
    "xr_economics_am_profit",
    "xr_economics_non_ag_profit",
]
BIO_FILES = [
    "xr_biodiversity_overall_priority_ag",
    "xr_biodiversity_overall_priority_ag_management",
    "xr_biodiversity_overall_priority_non_ag",
]

# Paper4 currently reads archived Run_Archive.zip outputs. Keep this explicit so
# future reruns with fully refreshed archives can disable the add-back instead of
# silently double-counting biodiversity payment.
ADD_BIO_PAYMENT_TO_ARCHIVED_PROFITS = True

COLOR = "#1d52a1"

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


def collect_slice(run_map, price_vals, varying_key):
    price_col = "CarbonPrice" if varying_key == "cp" else "BioPrice"
    rows = []

    for price in price_vals:
        key = (price, 0.0) if varying_key == "cp" else (0.0, price)
        zip_path = run_map.get(key)

        if zip_path:
            base_net_econ_2010 = read_sum(zip_path, PROFIT_FILES, BASE_YEAR) / 1e9
            base_net_econ_2050 = read_sum(zip_path, PROFIT_FILES, YEAR) / 1e9
            bio_score_2010_ha_yr = read_sum(zip_path, BIO_FILES, BASE_YEAR)
            bio_score_2050_ha_yr = read_sum(zip_path, BIO_FILES, YEAR)
        else:
            base_net_econ_2010 = np.nan
            base_net_econ_2050 = np.nan
            bio_score_2010_ha_yr = np.nan
            bio_score_2050_ha_yr = np.nan

        # Carbon price is already included in xr_economics_*_profit. Only the
        # biodiversity-price slice needs an explicit bio_price x bio_score add-back
        # so the plotted value matches the solver objective accounting.
        add_back_bio_payment = (
            varying_key == "bp"
            and ADD_BIO_PAYMENT_TO_ARCHIVED_PROFITS
            and not np.isnan(bio_score_2050_ha_yr)
        )
        bio_payment_2010 = 0.0
        bio_payment_2050 = price * bio_score_2050_ha_yr / 1e9 if add_back_bio_payment else 0.0

        net_econ_2010 = (
            base_net_econ_2010 + bio_payment_2010
            if not np.isnan(base_net_econ_2010)
            else np.nan
        )
        net_econ_2050 = (
            base_net_econ_2050 + bio_payment_2050
            if not np.isnan(base_net_econ_2050)
            else np.nan
        )
        net_econ_change = (
            net_econ_2050 - net_econ_2010
            if not np.isnan(net_econ_2010) and not np.isnan(net_econ_2050)
            else np.nan
        )

        rows.append({
            price_col: price,
            "BaseNetEcon2010_BAUD": base_net_econ_2010,
            "BaseNetEcon2050_BAUD": base_net_econ_2050,
            "BioScore2010_ha_yr": bio_score_2010_ha_yr,
            "BioScore2050_ha_yr": bio_score_2050_ha_yr,
            "BioPayment2010_BAUD": bio_payment_2010,
            "BioPayment2050_BAUD": bio_payment_2050,
            "NetEcon2010_BAUD": net_econ_2010,
            "NetEcon2050_BAUD": net_econ_2050,
            "NetEconChangeSince2010_BAUD": net_econ_change,
        })

        if varying_key == "bp":
            print(
                f"  {varying_key}={format_thousands(price)}: "
                f"2010={net_econ_2010:.2f}, 2050={net_econ_2050:.2f}, "
                f"bio_payment_2050={bio_payment_2050:.2f}, change={net_econ_change:.2f} B AUD"
            )
        else:
            print(
                f"  {varying_key}={format_thousands(price)}: "
                f"2010={net_econ_2010:.2f}, 2050={net_econ_2050:.2f}, "
                f"change={net_econ_change:.2f} B AUD"
            )

    return pd.DataFrame(rows)


def load_cache():
    if not CACHE_PATH.is_file():
        return None, None

    print(f"Loading cached data from {CACHE_PATH}")
    df_cp = pd.read_excel(CACHE_PATH, sheet_name="CarbonPrice")
    df_bp = pd.read_excel(CACHE_PATH, sheet_name="BioPrice")

    required_columns = {
        "BaseNetEcon2010_BAUD",
        "BaseNetEcon2050_BAUD",
        "BioScore2010_ha_yr",
        "BioScore2050_ha_yr",
        "BioPayment2010_BAUD",
        "BioPayment2050_BAUD",
        "NetEcon2010_BAUD",
        "NetEcon2050_BAUD",
        "NetEconChangeSince2010_BAUD",
    }
    if not required_columns.issubset(df_cp.columns) or not required_columns.issubset(df_bp.columns):
        print("Cached schema is outdated; rebuilding net economic return cache.")
        return None, None

    return df_cp, df_bp


df_cp, df_bp = load_cache()

if df_cp is None or df_bp is None:
    run_map, cp_vals, bp_vals = build_run_map()

    print(f"\n--- Slice A: BioPrice=0, carbon price varies ({BASE_YEAR}->{YEAR}) ---")
    df_cp = collect_slice(run_map, cp_vals, "cp")

    print(f"\n--- Slice B: CarbonPrice=0, biodiversity price varies ({BASE_YEAR}->{YEAR}) ---")
    df_bp = collect_slice(run_map, bp_vals, "bp")

    with pd.ExcelWriter(CACHE_PATH, engine="openpyxl") as writer:
        df_cp.to_excel(writer, sheet_name="CarbonPrice", index=False)
        df_bp.to_excel(writer, sheet_name="BioPrice", index=False)
    print(f"\nCache saved: {CACHE_PATH}")


def line_plot(ax, df, price_col):
    x = df[price_col].to_numpy()
    y = df["NetEconChangeSince2010_BAUD"].to_numpy()
    mask = ~np.isnan(y)

    ax.plot(
        x[mask],
        y[mask],
        color=COLOR,
        marker="o",
        linestyle="-",
        linewidth=1.5,
        markersize=5,
    )
    ax.set_xlabel(get_price_axis_label("cp" if price_col == "CarbonPrice" else "bp"))
    ax.set_ylabel("Change in net economic return since 2010 (Billion AU$)")
    ax.set_xlim(left=0)
    if mask.any() and np.nanmin(y[mask]) < 0.0 < np.nanmax(y[mask]):
        ax.axhline(0.0, color="#444444", linewidth=0.8)
    apply_price_formatter(ax, axis="x")
    style_box_axis(ax)


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
line_plot(ax1, df_cp, "CarbonPrice")
line_plot(ax2, df_bp, "BioPrice")

plt.tight_layout()
out_path = OUT_DIR / f"2_NetEcon_vs_Price_{YEAR}.png"
fig.savefig(out_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"Saved: {out_path}")
