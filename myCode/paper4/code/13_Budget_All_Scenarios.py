# ==============================================================================
# Figure 15: net economic return response vs carbon price /
#            biodiversity price
#   Left column:  BioPrice = 0, carbon price varies
#   Right column: CarbonPrice = 0, biodiversity price varies
#   Rows: Agricultural land-use / Ag management / Non-ag
#
#   Values are absolute net economic returns for YEAR.
#
# Notes on accounting:
#   - Solver-side economic optimisation uses `economic_contr_mrj`, assembled in
#     `luto/solvers/input_data.py`.
#   - In that solver pathway, biodiversity payment is monetised as
#     `bio_score x bio_price` before the economic objective is formed.
#   - Archived `xr_economics_*_profit` outputs already include price-linked
#     revenue, including biodiversity-price revenue, so this absolute composition
#     figure uses those profit layers directly and does not add an extra
#     biodiversity payment.
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
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tools.price_slice_utils import (
    DATA_DIR,
    OUT_DIR,
    apply_paper4_color_overrides_to_style_df,
    standardize_display_label,
    build_run_map,
    format_thousands,
    get_price_axis_label,
    set_sparse_index_price_ticks,
    style_box_axis,
)


YEAR = 2025
BASE_DIR = Path(__file__).resolve().parent
DRAW_ALL_TOOLS_DIR = BASE_DIR.parents[1] / "draw_all" / "code" / "tools"
COLOR_FILE = DRAW_ALL_TOOLS_DIR / "land use colors.xlsx"
GROUP_FILE = DRAW_ALL_TOOLS_DIR / "land use group.xlsx"
CACHE_PATH = DATA_DIR / f"13_Budget_All_Scenarios_raw_data_{YEAR}.xlsx"
DELTA_BUDGET_PATH = DATA_DIR / f"04_Budget_Delta_vs_Zero_raw_data_{YEAR}.xlsx"

FS = 18
SUM_LINE_LABEL = "Sum"
OLD_LIVESTOCK_LABEL = "Livestock"
MODIFIED_LIVESTOCK_LABEL = "Livestock (modified land)"
NATURAL_LIVESTOCK_LABEL = "Livestock (natural land)"
MODIFIED_LIVESTOCK_COLOR = "#762500"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial"],
    "font.size": FS,
    "axes.titlesize": FS + 2,
    "axes.labelsize": FS + 1,
    "xtick.labelsize": FS,
    "ytick.labelsize": FS,
    "legend.fontsize": FS,
    "mathtext.fontset": "stixsans",
    "axes.titleweight": "bold",
    "axes.labelweight": "bold",
})

ADD_BIO_PAYMENT_TO_ARCHIVED_PROFITS = False

NON_AG_EXCLUDE = {
    "agriculturallanduse",
    "otherlanduse",
}


def normalize_name(value):
    return re.sub(r"[\s\-]+", "", str(value).strip().lower())


def load_style_table(sheet_name):
    df = pd.read_excel(COLOR_FILE, sheet_name=sheet_name)
    df = apply_paper4_color_overrides_to_style_df(df)
    label_col = "desc_new" if "desc_new" in df.columns else "desc"

    order = []
    color_map = {}
    label_map = {}
    for _, row in df.iterrows():
        label = standardize_display_label(row[label_col])
        order.append(label)
        color_map[label] = row["color"]
        label_map[normalize_name(row["desc"])] = label

    return order, color_map, label_map


def split_livestock_style(order, color_map):
    previous_livestock_color = color_map.get(OLD_LIVESTOCK_LABEL, "#FFC87C")
    new_order = []
    for label in order:
        if label == OLD_LIVESTOCK_LABEL:
            new_order.extend([MODIFIED_LIVESTOCK_LABEL, NATURAL_LIVESTOCK_LABEL])
        else:
            new_order.append(label)

    new_color_map = {
        label: color
        for label, color in color_map.items()
        if label != OLD_LIVESTOCK_LABEL
    }
    new_color_map[MODIFIED_LIVESTOCK_LABEL] = MODIFIED_LIVESTOCK_COLOR
    new_color_map[NATURAL_LIVESTOCK_LABEL] = previous_livestock_color
    return new_order, new_color_map


def map_ag_group(row):
    group = row["ag_group"]
    if group == OLD_LIVESTOCK_LABEL:
        desc_key = normalize_name(row["desc"])
        if "modifiedland" in desc_key:
            return MODIFIED_LIVESTOCK_LABEL
        if "naturalland" in desc_key:
            return NATURAL_LIVESTOCK_LABEL
    return standardize_display_label(group)


AG_ORDER, AG_COLOR_MAP, _ = load_style_table("ag_group")
AG_ORDER, AG_COLOR_MAP = split_livestock_style(AG_ORDER, AG_COLOR_MAP)
AM_ORDER, AM_COLOR_MAP, AM_LABEL_MAP = load_style_table("am")
NON_AG_ORDER, NON_AG_COLOR_MAP, NON_AG_LABEL_MAP = load_style_table("non_ag")

# Apply Ag2050 naming convention: remap color-table desc_new -> Ag2050 display name
_AG2050_DISPLAY = {
    "Biochar":                                              "Biochar (soil amendment)",
    "Human-Induced Regeneration (beef)":                    "Managed regeneration (beef)",
    "Human-Induced Regeneration (sheep)":                   "Managed regeneration (sheep)",
    "Environmental plantings (mixed local native species)": "Environmental plantings (mixed species)",
    "BECCS (Bioenergy with carbon capture and storage)":    "BECCS (Bioenergy with Carbon Capture and Storage)",
    "Destocked (natural land)":                             "Destocked (natural land)",
}

def _apply_ag2050(order, color_map, label_map):
    new_order = [_AG2050_DISPLAY.get(x, x) for x in order]
    new_color = {_AG2050_DISPLAY.get(k, k): v for k, v in color_map.items()}
    new_label = {k: _AG2050_DISPLAY.get(v, v) for k, v in label_map.items()}
    return new_order, new_color, new_label

AM_ORDER, AM_COLOR_MAP, AM_LABEL_MAP = _apply_ag2050(AM_ORDER, AM_COLOR_MAP, AM_LABEL_MAP)
NON_AG_ORDER, NON_AG_COLOR_MAP, NON_AG_LABEL_MAP = _apply_ag2050(NON_AG_ORDER, NON_AG_COLOR_MAP, NON_AG_LABEL_MAP)
LU_ORDER, LU_COLOR_MAP, LU_LABEL_MAP = load_style_table("lu")

group_df = pd.read_excel(GROUP_FILE)
LU_TO_AG_GROUP = {
    normalize_name(row["desc"]): map_ag_group(row)
    for _, row in group_df.iterrows()
    if pd.notna(row.get("desc")) and pd.notna(row.get("ag_group"))
}

PANEL_CONFIG = {
    "Agricultural land-use": {
        "order": AG_ORDER,
        "color_map": AG_COLOR_MAP,
        "ylabel": "Agricultural land-use",
    },
    "Ag management": {
        "order": AM_ORDER,
        "color_map": AM_COLOR_MAP,
        "ylabel": "Agricultural management",
    },
    "Non-ag": {
        "order": NON_AG_ORDER,
        "color_map": NON_AG_COLOR_MAP,
        "ylabel": "Non-agricultural land-use",
    },
}

TOTAL_CATEGORY_MAP = {
    "Agricultural land-use": LU_LABEL_MAP.get(
        normalize_name("Agricultural land-use"),
        "Agricultural land-use",
    ),
    "Ag management": LU_LABEL_MAP.get(
        normalize_name("Agricultural management"),
        "Agricultural management",
    ),
    "Non-ag": LU_LABEL_MAP.get(
        normalize_name("Non-agricultural land-use"),
        "Non-agricultural land-use",
    ),
}
TOTAL_ORDER = [
    TOTAL_CATEGORY_MAP["Agricultural land-use"],
    TOTAL_CATEGORY_MAP["Ag management"],
    TOTAL_CATEGORY_MAP["Non-ag"],
]
TOTAL_COLOR_MAP = {
    category: LU_COLOR_MAP.get(category, "#888888")
    for category in TOTAL_ORDER
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

    for price in price_vals:
        key = (price, 0.0) if varying_key == "cp" else (0.0, price)
        zip_path = run_map.get(key)
        if zip_path is None:
            continue

        profit_2025 = get_profit_summaries(zip_path, YEAR)
        bio_2025 = get_bio_summaries(zip_path, YEAR) if varying_key == "bp" else {}

        for area_type in PANEL_CONFIG:
            categories = list(dict.fromkeys(
                list(profit_2025[area_type]) +
                list(bio_2025.get(area_type, {}))
            ))
            category_order = get_category_order(area_type, categories)

            total_net_econ = 0.0
            for category in category_order:
                base_net_econ_2025_aud = profit_2025[area_type].get(category, 0.0)
                bio_2025_ha_yr = bio_2025.get(area_type, {}).get(category, 0.0)

                # Carbon pricing is already embedded in archived profit outputs.
                # For the biodiversity-price slice, add back bio_price x absolute
                # biodiversity contribution at YEAR so this figure matches
                # solver-side accounting.
                add_back_bio_payment = varying_key == "bp" and ADD_BIO_PAYMENT_TO_ARCHIVED_PROFITS
                bio_payment_2025_aud = price * bio_2025_ha_yr if add_back_bio_payment else 0.0

                net_econ_2025_aud = base_net_econ_2025_aud + bio_payment_2025_aud
                net_econ_2025_baud = net_econ_2025_aud / 1e9
                total_net_econ += net_econ_2025_baud

                rows.append({
                    "AccountingMode": "Absolute2025ArchivedProfit",
                    "PriceType": price_type,
                    "Price": price,
                    "AreaType": area_type,
                    "Category": category,
                    "BaseNetEcon_2025_BAUD": base_net_econ_2025_aud / 1e9,
                    "Bio_2025_ha_yr": bio_2025_ha_yr,
                    "BioPayment_2025_BAUD": bio_payment_2025_aud / 1e9,
                    "NetEcon_2025_BAUD": net_econ_2025_baud,
                })

            print(
                f"  {varying_key}={format_thousands(price)} | {area_type}: "
                f"net_econ={total_net_econ:.2f} B AUD"
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
        "AccountingMode",
        "PriceType",
        "Price",
        "AreaType",
        "Category",
        "BaseNetEcon_2025_BAUD",
        "NetEcon_2025_BAUD",
    }
    if not required_columns.issubset(df_long.columns):
        print("Cached net economic data schema is outdated; rebuilding.")
        return None
    if set(df_long["AccountingMode"]) != {"BaselinePlusDvarDelta"}:
        print("Cached net economic data uses outdated accounting; rebuilding.")
        return None
    if OLD_LIVESTOCK_LABEL in set(df_long["Category"]):
        print("Cached net economic data uses unsplit livestock categories; rebuilding.")
        return None

    df_long["Category"] = df_long["Category"].map(standardize_display_label)
    return df_long


def collect_and_cache():
    run_map, cp_vals, bp_vals = build_run_map()

    if not DELTA_BUDGET_PATH.is_file():
        raise FileNotFoundError(f"Missing Figure 04 budget differences: {DELTA_BUDGET_PATH}")

    zero_zip = run_map.get((0.0, 0.0))
    if zero_zip is None:
        raise FileNotFoundError("Cannot find zero-price baseline run (CarbonPrice=0, BioPrice=0).")

    baseline_profit = get_profit_summaries(zero_zip, YEAR)
    baseline_summary = {
        area_type: {
            standardize_display_label(category): value / 1e9
            for category, value in baseline_profit.get(area_type, {}).items()
        }
        for area_type in PANEL_CONFIG
    }

    delta_df = pd.read_excel(DELTA_BUDGET_PATH, sheet_name="NetEconLong")
    delta_df["Category"] = delta_df["Category"].map(standardize_display_label)

    rows = []
    for price_type, price_vals in [("CarbonPrice", cp_vals), ("BioPrice", bp_vals)]:
        print(f"\n--- {price_type}: absolute budget at {YEAR}; zero-price baseline + Figure 04 difference ---")
        for price in price_vals:
            subset = delta_df[
                (delta_df["PriceType"] == price_type) &
                (delta_df["Price"] == price)
            ]

            for area_type in PANEL_CONFIG:
                area_delta = subset[subset["AreaType"] == area_type]
                delta_summary = (
                    area_delta.groupby("Category")["NetEconChange_vs_ZeroPrice_BAUD"]
                    .sum()
                    .to_dict()
                )
                categories = list(dict.fromkeys(
                    list(baseline_summary.get(area_type, {})) +
                    list(delta_summary)
                ))
                category_order = get_category_order(area_type, categories)

                total_budget = 0.0
                for category in category_order:
                    base_baud = baseline_summary.get(area_type, {}).get(category, 0.0)
                    delta_baud = delta_summary.get(category, 0.0)
                    budget_baud = base_baud + delta_baud
                    total_budget += budget_baud

                    rows.append({
                        "AccountingMode": "BaselinePlusDvarDelta",
                        "PriceType": price_type,
                        "Price": price,
                        "AreaType": area_type,
                        "Category": category,
                        "BaseNetEcon_2025_BAUD": base_baud,
                        "DvarBudgetChange_BAUD": delta_baud,
                        "NetEcon_2025_BAUD": budget_baud,
                    })

                print(
                    f"  {price_type}={format_thousands(price)} | {area_type}: "
                    f"budget={total_budget:.2f} B AUD"
                )

    df_long = pd.DataFrame(rows)
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
        values="NetEcon_2025_BAUD",
        aggfunc="sum",
        fill_value=0.0,
    ).sort_index()

    category_order = get_category_order(area_type, list(pivot.columns))
    return pivot.reindex(columns=category_order, fill_value=0.0)


def build_total_pivot(df_long, price_type):
    df_subset = df_long[df_long["PriceType"] == price_type].copy()
    if df_subset.empty:
        return pd.DataFrame()

    df_subset["Category"] = df_subset["AreaType"].map(TOTAL_CATEGORY_MAP)
    df_subset = df_subset.dropna(subset=["Category"])

    pivot = df_subset.pivot_table(
        index="Price",
        columns="Category",
        values="NetEcon_2025_BAUD",
        aggfunc="sum",
        fill_value=0.0,
    ).sort_index()

    category_order = [category for category in TOTAL_ORDER if category in pivot.columns]
    category_order += [category for category in pivot.columns if category not in category_order]
    return pivot.reindex(columns=category_order, fill_value=0.0)


def build_patch_handles(categories, color_map):
    return [
        mpatches.Patch(
            facecolor=color_map.get(category, "#888888"),
            edgecolor="none",
            label=category,
        )
        for category in categories
    ]


def build_total_line_handle():
    return Line2D(
        [0],
        [0],
        color="black",
        linestyle="-",
        marker="o",
        linewidth=1.8,
        markersize=4.5,
        solid_capstyle="round",
        label=SUM_LINE_LABEL,
    )


def plot_sum_markers(ax, x, y):
    ax.plot(
        x,
        y,
        color="black",
        linestyle="-",
        marker="o",
        linewidth=1.8,
        markersize=4.5,
        markeredgewidth=0,
        zorder=30,
    )


def stacked_bar(ax, pivot_df, area_type, varying_key, show_xlabel, color_map=None, show_sum_line=False):
    if pivot_df.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        style_box_axis(ax, linewidth=0.8)
        return []

    color_map = PANEL_CONFIG[area_type]["color_map"] if color_map is None else color_map
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

    if show_sum_line:
        totals = pivot_df.sum(axis=1).to_numpy()
        plot_sum_markers(ax, x, totals)

    set_sparse_index_price_ticks(ax, price_vals, max_ticks=8)
    if show_xlabel:
        ax.tick_params(axis="x", labelrotation=90)
        for label in ax.get_xticklabels():
            label.set_ha("center")
        ax.set_xlabel(get_price_axis_label(varying_key))
        ax.xaxis.set_label_coords(0.5, -0.25)
    else:
        ax.tick_params(axis="x", labelbottom=False)

    style_box_axis(ax, linewidth=0.8)
    return visible_categories


df_long = load_cache()
if df_long is None:
    df_long = collect_and_cache()


fig, axes = plt.subplots(4, 2, figsize=(14, 23), sharex="col")
row_area_types = ["Agricultural land-use", "Ag management", "Non-ag"]
row_legends = {}

total_pivot_cp = build_total_pivot(df_long, "CarbonPrice")
total_pivot_bp = build_total_pivot(df_long, "BioPrice")
total_cats_left = stacked_bar(axes[0, 0], total_pivot_cp, "Total", "cp", show_xlabel=False, color_map=TOTAL_COLOR_MAP, show_sum_line=True)
total_cats_right = stacked_bar(axes[0, 1], total_pivot_bp, "Total", "bp", show_xlabel=False, color_map=TOTAL_COLOR_MAP, show_sum_line=True)
axes[0, 0].set_ylabel("Total")

total_legend_categories = [category for category in TOTAL_ORDER if category in dict.fromkeys(total_cats_left + total_cats_right)]
total_handles = build_patch_handles(total_legend_categories, TOTAL_COLOR_MAP)
if total_handles:
    total_handles = [build_total_line_handle()] + total_handles
row_legends["_total"] = total_handles if total_handles else []

for row_idx, area_type in enumerate(row_area_types):
    ax_left = axes[row_idx + 1, 0]
    ax_right = axes[row_idx + 1, 1]

    pivot_cp = build_pivot(df_long, "CarbonPrice", area_type)
    pivot_bp = build_pivot(df_long, "BioPrice", area_type)

    cats_left = stacked_bar(ax_left, pivot_cp, area_type, "cp", show_xlabel=(row_idx == len(row_area_types) - 1))
    cats_right = stacked_bar(ax_right, pivot_bp, area_type, "bp", show_xlabel=(row_idx == len(row_area_types) - 1))

    ax_left.set_ylabel(PANEL_CONFIG[area_type]["ylabel"])

    legend_categories = get_category_order(area_type, list(dict.fromkeys(cats_left + cats_right)))
    row_legends[area_type] = build_patch_handles(
        legend_categories,
        PANEL_CONFIG[area_type]["color_map"],
    )

LEGEND_NCOL = {
    "_total": 2,
    "Agricultural land-use": 2,
    "Ag management": 2,
    "Non-ag": 2,
}
LEGEND_FS = {
    "_total": FS,
    "Agricultural land-use": FS,
    "Ag management": FS,
    "Non-ag": FS,
}

fig.supylabel(r"Budget (AU\$ billion yr$^{-1}$)", x=0.065, y=0.5,
              fontsize=FS + 1, fontweight="bold")
plt.tight_layout(rect=[0.075, 0, 1, 1])
plt.subplots_adjust(hspace=0.62, wspace=0.12)
ROW_UP_SHIFTS = {
    1: 0.024,
    2: 0.048,
    3: 0.048,
}
for row_idx, y_shift in ROW_UP_SHIFTS.items():
    for ax in axes[row_idx, :]:
        pos = ax.get_position()
        ax.set_position([pos.x0, pos.y0 + y_shift, pos.width, pos.height])
fig.canvas.draw()
renderer = fig.canvas.get_renderer()
fig_w_px = fig.get_figwidth() * fig.dpi
fig_h_px = fig.get_figheight() * fig.dpi

all_rows = [("_total", 0)] + [(area_type, i + 1) for i, area_type in enumerate(row_area_types)]
for key, row_idx in all_rows:
    handles = row_legends.get(key, [])
    if not handles:
        continue

    ax_l, ax_r = axes[row_idx, 0], axes[row_idx, 1]
    bb_l = ax_l.get_tightbbox(renderer)
    bb_r = ax_r.get_tightbbox(renderer)
    x_center = (bb_l.x0 + bb_r.x1) / 2 / fig_w_px
    if row_idx == len(all_rows) - 1:
        y_offset = 0.065
    else:
        y_offset = 0.01
    y_anchor = min(ax_l.get_position().y0, ax_r.get_position().y0) - y_offset

    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(x_center, y_anchor),
        bbox_transform=fig.transFigure,
        ncol=LEGEND_NCOL.get(key, 3),
        frameon=False,
        borderaxespad=0.0,
        handlelength=1.0,
        handleheight=1.0,
        columnspacing=1.0,
        fontsize=LEGEND_FS.get(key, FS - 1),
    )

out_path = OUT_DIR / "13_Budget_All_Scenarios.png"
fig.savefig(out_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"Saved: {out_path}")
