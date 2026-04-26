# ==============================================================================
# Figure 6: Change in net economic return since 2010 vs carbon price /
#           biodiversity price
#   Left column:  BioPrice = 0, carbon price varies
#   Right column: CarbonPrice = 0, biodiversity price varies
#   Rows: Agricultural land-use / Ag management / Non-ag
#
# Notes on accounting:
#   - Solver-side economic optimisation uses `economic_contr_mrj`, assembled in
#     `luto/solvers/input_data.py`.
#   - In that solver pathway, biodiversity payment is monetised as
#     `bio_score x bio_price` before the economic objective is formed.
#   - Archived `xr_economics_*_profit` outputs can be used directly for the
#     carbon-price slice because carbon pricing is already embedded there.
#   - For the biodiversity-price slice in these archived paper4 runs, we do not
#     assume biodiversity payment is already included in `xr_economics_*_profit`.
#     To match solver-side accounting, this script adds it back explicitly at the
#     category level for 2050.
#   - This figure now plots the 2050 minus 2010 change within each run. The 2010
#     slice is treated as the historical no-policy anchor for these paper4 scans,
#     so the explicit biodiversity-payment add-back is only applied to 2050.
# ==============================================================================

import io
import os
import re
import sys
import zipfile
from pathlib import Path

import cf_xarray as cfxr
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tools.price_slice_utils import (
    DATA_DIR,
    OUT_DIR,
    build_run_map,
    format_thousands,
    get_price_axis_label,
    style_box_axis,
)


YEAR = 2050
BASE_DIR = Path(__file__).resolve().parent
DRAW_ALL_TOOLS_DIR = BASE_DIR.parents[1] / "draw_all" / "code" / "tools"
COLOR_FILE = DRAW_ALL_TOOLS_DIR / "land use colors.xlsx"
GROUP_FILE = DRAW_ALL_TOOLS_DIR / "land use group.xlsx"
CACHE_PATH = DATA_DIR / f"6_NetEcon_Composition_raw_data_{YEAR}.xlsx"

FS = 11
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial"],
    "font.size": FS,
    "axes.titlesize": FS,
    "axes.labelsize": FS,
    "xtick.labelsize": FS,
    "ytick.labelsize": FS,
    "legend.fontsize": FS - 1,
    "mathtext.fontset": "stixsans",
})

ADD_BIO_PAYMENT_TO_ARCHIVED_PROFITS = True

NON_AG_EXCLUDE = {
    "agriculturallanduse",
    "otherlanduse",
}


def normalize_name(value):
    return re.sub(r"[\s\-]+", "", str(value).strip().lower())


def load_style_table(sheet_name):
    df = pd.read_excel(COLOR_FILE, sheet_name=sheet_name)
    label_col = "desc_new" if "desc_new" in df.columns else "desc"

    order = []
    color_map = {}
    label_map = {}
    for _, row in df.iterrows():
        label = row[label_col]
        order.append(label)
        color_map[label] = row["color"]
        label_map[normalize_name(row["desc"])] = label

    return order, color_map, label_map


AG_ORDER, AG_COLOR_MAP, _ = load_style_table("ag_group")
AM_ORDER, AM_COLOR_MAP, AM_LABEL_MAP = load_style_table("am")
NON_AG_ORDER, NON_AG_COLOR_MAP, NON_AG_LABEL_MAP = load_style_table("non_ag")

group_df = pd.read_excel(GROUP_FILE)
LU_TO_AG_GROUP = {
    normalize_name(row["desc"]): row["ag_group"]
    for _, row in group_df.iterrows()
    if pd.notna(row.get("desc")) and pd.notna(row.get("ag_group"))
}

PANEL_CONFIG = {
    "Agricultural land-use": {
        "order": AG_ORDER,
        "color_map": AG_COLOR_MAP,
        "ylabel": "Agricultural land-use\nNet econ. change vs. 2050 baseline (Billion AU$)",
    },
    "Ag management": {
        "order": AM_ORDER,
        "color_map": AM_COLOR_MAP,
        "ylabel": "Ag management\nNet econ. change vs. 2050 baseline (Billion AU$)",
    },
    "Non-ag": {
        "order": NON_AG_ORDER,
        "color_map": NON_AG_COLOR_MAP,
        "ylabel": "Non-ag\nNet econ. change vs. 2050 baseline (Billion AU$)",
    },
}


def open_metric_da(zip_path, file_name):
    with zipfile.ZipFile(zip_path) as archive:
        matches = [name for name in archive.namelist() if name.endswith(file_name)]
        if not matches:
            return None

        with archive.open(matches[0]) as file_obj:
            ds = xr.open_dataset(io.BytesIO(file_obj.read()), engine="h5netcdf")

    try:
        if "layer" in ds.dims and "compress" in ds["layer"].attrs:
            ds = cfxr.decode_compress_to_multi_index(ds, "layer")

        da = next(iter(ds.data_vars.values()))
        return da.load()
    finally:
        ds.close()


def sum_with_total_coords(da, **selectors):
    sub = da.sel(selectors)

    for coord_name in list(sub.coords):
        if coord_name in {"cell", "layer"} or coord_name in selectors:
            continue

        coord_values = sub.coords[coord_name].values
        try:
            has_all = "ALL" in coord_values
        except TypeError:
            has_all = False

        if has_all:
            sub = sub.sel({coord_name: "ALL"})

    return float(sub.sum())


def read_ag_group_summary(zip_path, file_name):
    da = open_metric_da(zip_path, file_name)
    if da is None:
        return {}

    result = {}
    for lu in pd.unique(da.coords["lu"].values):
        if lu == "ALL":
            continue

        value = sum_with_total_coords(da, lu=lu)
        if np.isclose(value, 0.0):
            continue

        group = LU_TO_AG_GROUP.get(normalize_name(lu), "Other land")
        result[group] = result.get(group, 0.0) + value

    return result


def read_ag_management_summary(zip_path, file_name):
    da = open_metric_da(zip_path, file_name)
    if da is None:
        return {}

    result = {}
    for am in pd.unique(da.coords["am"].values):
        if am == "ALL":
            continue

        value = sum_with_total_coords(da, am=am)
        if np.isclose(value, 0.0):
            continue

        label = AM_LABEL_MAP.get(normalize_name(am), am)
        result[label] = result.get(label, 0.0) + value

    return result


def read_non_ag_summary(zip_path, file_name):
    da = open_metric_da(zip_path, file_name)
    if da is None:
        return {}

    result = {}
    for lu in pd.unique(da.coords["lu"].values):
        if lu == "ALL" or normalize_name(lu) in NON_AG_EXCLUDE:
            continue

        value = sum_with_total_coords(da, lu=lu)
        if np.isclose(value, 0.0):
            continue

        label = NON_AG_LABEL_MAP.get(normalize_name(lu), lu)
        result[label] = result.get(label, 0.0) + value

    return result


def get_profit_summaries(zip_path, year):
    return {
        "Agricultural land-use": read_ag_group_summary(zip_path, f"xr_economics_ag_profit_{year}.nc"),
        "Ag management": read_ag_management_summary(zip_path, f"xr_economics_am_profit_{year}.nc"),
        "Non-ag": read_non_ag_summary(zip_path, f"xr_economics_non_ag_profit_{year}.nc"),
    }


def get_bio_summaries(zip_path, year):
    return {
        "Agricultural land-use": read_ag_group_summary(zip_path, f"xr_biodiversity_overall_priority_ag_{year}.nc"),
        "Ag management": read_ag_management_summary(zip_path, f"xr_biodiversity_overall_priority_ag_management_{year}.nc"),
        "Non-ag": read_non_ag_summary(zip_path, f"xr_biodiversity_overall_priority_non_ag_{year}.nc"),
    }


def get_category_order(area_type, categories_seen):
    base_order = PANEL_CONFIG[area_type]["order"]
    ordered = [category for category in base_order if category in categories_seen]
    ordered += [category for category in categories_seen if category not in base_order]
    return ordered


def collect_slice_rows(run_map, price_vals, varying_key):
    rows = []
    price_type = "CarbonPrice" if varying_key == "cp" else "BioPrice"
    baseline_zip = run_map.get((0.0, 0.0))

    # Baseline: (cp=0, bp=0) run at YEAR — bp=0 so no bio payment add-back needed
    profit_baseline = get_profit_summaries(baseline_zip, YEAR) if baseline_zip else {at: {} for at in PANEL_CONFIG}

    for price in price_vals:
        key = (price, 0.0) if varying_key == "cp" else (0.0, price)
        zip_path = run_map.get(key)
        if zip_path is None:
            continue

        profit_2050 = get_profit_summaries(zip_path, YEAR)
        bio_2050 = get_bio_summaries(zip_path, YEAR) if varying_key == "bp" else {}

        for area_type in PANEL_CONFIG:
            categories = list(dict.fromkeys(
                list(profit_baseline.get(area_type, {})) +
                list(profit_2050[area_type]) +
                list(bio_2050.get(area_type, {}))
            ))
            category_order = get_category_order(area_type, categories)

            total_change = 0.0
            for category in category_order:
                base_net_econ_baseline_aud = profit_baseline.get(area_type, {}).get(category, 0.0)
                base_net_econ_2050_aud = profit_2050[area_type].get(category, 0.0)
                bio_score_2050_ha_yr = bio_2050.get(area_type, {}).get(category, 0.0)

                # Carbon pricing is already embedded in archived profit outputs.
                # For the biodiversity-price slice, add back bio_price x bio_score
                # at 2050 so this figure matches solver-side accounting.
                add_back_bio_payment = varying_key == "bp" and ADD_BIO_PAYMENT_TO_ARCHIVED_PROFITS
                bio_payment_2050_aud = price * bio_score_2050_ha_yr if add_back_bio_payment else 0.0

                # Baseline has bp=0, so no bio payment add-back for baseline
                net_econ_baseline_aud = base_net_econ_baseline_aud
                net_econ_2050_aud = base_net_econ_2050_aud + bio_payment_2050_aud
                net_econ_change_baud = (net_econ_2050_aud - net_econ_baseline_aud) / 1e9
                total_change += net_econ_change_baud

                rows.append({
                    "PriceType": price_type,
                    "Price": price,
                    "AreaType": area_type,
                    "Category": category,
                    "BaseNetEcon_Baseline_BAUD": base_net_econ_baseline_aud / 1e9,
                    "BaseNetEcon2050_BAUD": base_net_econ_2050_aud / 1e9,
                    "BioScore2050_ha_yr": bio_score_2050_ha_yr,
                    "BioPayment2050_BAUD": bio_payment_2050_aud / 1e9,
                    "NetEcon_Baseline_BAUD": net_econ_baseline_aud / 1e9,
                    "NetEcon2050_BAUD": net_econ_2050_aud / 1e9,
                    "NetEconChangevs_Baseline_BAUD": net_econ_change_baud,
                })

            print(
                f"  {varying_key}={format_thousands(price)} | {area_type}: "
                f"change={total_change:.2f} B AUD"
            )

    return rows


def load_cache():
    if not CACHE_PATH.is_file():
        return None

    try:
        print(f"Loading cached data from {CACHE_PATH}")
        df_long = pd.read_excel(CACHE_PATH, sheet_name="NetEconLong")
    except ValueError:
        print("Cached net economic workbook uses an older layout; rebuilding.")
        return None

    required_columns = {
        "PriceType",
        "Price",
        "AreaType",
        "Category",
        "BaseNetEcon_Baseline_BAUD",
        "BaseNetEcon2050_BAUD",
        "BioScore2050_ha_yr",
        "BioPayment2050_BAUD",
        "NetEcon_Baseline_BAUD",
        "NetEcon2050_BAUD",
        "NetEconChangevs_Baseline_BAUD",
    }
    if not required_columns.issubset(df_long.columns):
        print("Cached net economic data schema is outdated; rebuilding.")
        return None

    return df_long


def collect_and_cache():
    run_map, cp_vals, bp_vals = build_run_map()

    print(f"\n--- Slice A: BioPrice=0, carbon price varies (baseline=cp=0,bp=0 at {YEAR}) ---")
    rows_cp = collect_slice_rows(run_map, cp_vals, "cp")

    print(f"\n--- Slice B: CarbonPrice=0, biodiversity price varies (baseline=cp=0,bp=0 at {YEAR}) ---")
    rows_bp = collect_slice_rows(run_map, bp_vals, "bp")

    df_long = pd.DataFrame(rows_cp + rows_bp)
    df_long = df_long.sort_values(["PriceType", "AreaType", "Price", "Category"]).reset_index(drop=True)

    with pd.ExcelWriter(CACHE_PATH, engine="openpyxl") as writer:
        df_long.to_excel(writer, sheet_name="NetEconLong", index=False)
        df_long[df_long["PriceType"] == "CarbonPrice"].to_excel(writer, sheet_name="CarbonPrice", index=False)
        df_long[df_long["PriceType"] == "BioPrice"].to_excel(writer, sheet_name="BioPrice", index=False)

    print(f"\nCache saved: {CACHE_PATH}")
    return df_long


def build_pivot(df_long, price_type, area_type):
    df_subset = df_long[
        (df_long["PriceType"] == price_type) &
        (df_long["AreaType"] == area_type)
    ]

    if df_subset.empty:
        return pd.DataFrame()

    pivot = df_subset.pivot_table(
        index="Price",
        columns="Category",
        values="NetEconChangevs_Baseline_BAUD",
        aggfunc="sum",
        fill_value=0.0,
    ).sort_index()

    category_order = get_category_order(area_type, list(pivot.columns))
    return pivot.reindex(columns=category_order, fill_value=0.0)


def stacked_bar(ax, pivot_df, area_type, varying_key, show_xlabel):
    if pivot_df.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        style_box_axis(ax, linewidth=0.8)
        return []

    color_map = PANEL_CONFIG[area_type]["color_map"]
    price_vals = pivot_df.index.to_list()
    x = np.arange(len(price_vals))
    positive_bottoms = np.zeros(len(price_vals))
    negative_bottoms = np.zeros(len(price_vals))

    visible_categories = []
    for category in pivot_df.columns:
        heights = pivot_df[category].to_numpy()
        if np.isclose(np.abs(heights).sum(), 0.0):
            continue

        positive = np.clip(heights, 0.0, None)
        negative = np.clip(heights, None, 0.0)

        if not np.isclose(positive.sum(), 0.0):
            ax.bar(
                x,
                positive,
                0.75,
                bottom=positive_bottoms,
                color=color_map.get(category, "#888888"),
            )
            positive_bottoms += positive

        if not np.isclose(np.abs(negative).sum(), 0.0):
            ax.bar(
                x,
                negative,
                0.75,
                bottom=negative_bottoms,
                color=color_map.get(category, "#888888"),
            )
            negative_bottoms += negative

        visible_categories.append(category)

    if np.any(pivot_df.to_numpy() < 0.0):
        ax.axhline(0.0, color="#444444", linewidth=0.8)

    ax.set_xticks(x)
    if show_xlabel:
        ax.set_xticklabels([format_thousands(value) for value in price_vals], rotation=90, ha="center")
        ax.set_xlabel(get_price_axis_label(varying_key))
    else:
        ax.tick_params(axis="x", labelbottom=False)

    style_box_axis(ax, linewidth=0.8)
    return visible_categories


df_long = load_cache()
if df_long is None:
    df_long = collect_and_cache()


fig, axes = plt.subplots(3, 2, figsize=(17, 11), sharex="col")

axes[0, 0].set_title("Carbon price")
axes[0, 1].set_title("Biodiversity price")

row_area_types = ["Agricultural land-use", "Ag management", "Non-ag"]
row_legends = {}

for row_idx, area_type in enumerate(row_area_types):
    ax_left = axes[row_idx, 0]
    ax_right = axes[row_idx, 1]

    pivot_cp = build_pivot(df_long, "CarbonPrice", area_type)
    pivot_bp = build_pivot(df_long, "BioPrice", area_type)

    cats_left = stacked_bar(ax_left, pivot_cp, area_type, "cp", show_xlabel=(row_idx == len(row_area_types) - 1))
    cats_right = stacked_bar(ax_right, pivot_bp, area_type, "bp", show_xlabel=(row_idx == len(row_area_types) - 1))

    ax_left.set_ylabel(PANEL_CONFIG[area_type]["ylabel"])

    legend_categories = get_category_order(area_type, list(dict.fromkeys(cats_left + cats_right)))
    row_legends[area_type] = [
        mpatches.Patch(
            facecolor=PANEL_CONFIG[area_type]["color_map"].get(category, "#888888"),
            edgecolor="none",
            label=category,
        )
        for category in legend_categories
    ]

for row_idx, area_type in enumerate(row_area_types):
    handles = row_legends[area_type]
    if not handles:
        continue

    axes[row_idx, 1].legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        frameon=False,
        borderaxespad=0.0,
        handlelength=1.0,
        handleheight=1.0,
    )

plt.tight_layout()
plt.subplots_adjust(right=0.77, hspace=0.24, wspace=0.12)

out_path = OUT_DIR / f"6_NetEcon_Composition_vs_Price_{YEAR}.png"
fig.savefig(out_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"Saved: {out_path}")
