# ==============================================================================
# Figure 3: Area composition change since 2010 vs carbon price / biodiversity price
#   Left column:  BioPrice = 0, carbon price varies
#   Right column: CarbonPrice = 0, biodiversity price varies
#   Rows: Agricultural land-use / Ag management / Non-ag
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


BASE_YEAR = 2025
YEAR = 2050
BASE_DIR = Path(__file__).resolve().parent
DRAW_ALL_TOOLS_DIR = BASE_DIR.parents[1] / "draw_all" / "code" / "tools"
COLOR_FILE = DRAW_ALL_TOOLS_DIR / "land use colors.xlsx"
GROUP_FILE = DRAW_ALL_TOOLS_DIR / "land use group.xlsx"
CACHE_PATH = DATA_DIR / f"3_Area_raw_data_{YEAR}.xlsx"

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

AREA_CONFIG = {
    "Agricultural land-use": {
        "order": AG_ORDER,
        "color_map": AG_COLOR_MAP,
        "ylabel": "Agricultural land-use\nArea change since 2010 (Mha)",
    },
    "Ag management": {
        "order": AM_ORDER,
        "color_map": AM_COLOR_MAP,
        "ylabel": "Ag management\nArea change since 2010 (Mha)",
    },
    "Non-ag": {
        "order": NON_AG_ORDER,
        "color_map": NON_AG_COLOR_MAP,
        "ylabel": "Non-ag\nArea change since 2010 (Mha)",
    },
}

NON_AG_EXCLUDE = {
    normalize_name("Agricultural land-use"),
    normalize_name("Other land-use"),
}


def open_area_da(zip_path, file_name):
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


def read_agricultural_area(zip_path, year):
    da = open_area_da(zip_path, f"xr_area_agricultural_landuse_{year}.nc")
    if da is None:
        return {}

    result = {}
    for lu in pd.unique(da.coords["lu"].values):
        if lu == "ALL":
            continue

        area_mha = sum_with_total_coords(da, lu=lu) / 1e6
        if np.isclose(area_mha, 0.0):
            continue

        group = LU_TO_AG_GROUP.get(normalize_name(lu), "Other land")
        result[group] = result.get(group, 0.0) + area_mha

    return result


def read_ag_management_area(zip_path, year):
    da = open_area_da(zip_path, f"xr_area_agricultural_management_{year}.nc")
    if da is None:
        return {}

    result = {}
    for am in pd.unique(da.coords["am"].values):
        if am == "ALL":
            continue

        area_mha = sum_with_total_coords(da, am=am) / 1e6
        if np.isclose(area_mha, 0.0):
            continue

        label = AM_LABEL_MAP.get(normalize_name(am), am)
        result[label] = result.get(label, 0.0) + area_mha

    return result


def read_non_ag_area(zip_path, year):
    da = open_area_da(zip_path, f"xr_area_non_agricultural_landuse_{year}.nc")
    if da is None:
        return {}

    result = {}
    for lu in pd.unique(da.coords["lu"].values):
        if lu == "ALL" or normalize_name(lu) in NON_AG_EXCLUDE:
            continue

        area_mha = sum_with_total_coords(da, lu=lu) / 1e6
        if np.isclose(area_mha, 0.0):
            continue

        label = NON_AG_LABEL_MAP.get(normalize_name(lu), lu)
        result[label] = result.get(label, 0.0) + area_mha

    return result


def collect_slice_rows(run_map, price_vals, varying_key):
    price_type = "CarbonPrice" if varying_key == "cp" else "BioPrice"
    rows = []

    for price in price_vals:
        key = (price, 0.0) if varying_key == "cp" else (0.0, price)
        zip_path = run_map.get(key)
        if zip_path is None:
            continue

        summaries_2010 = {
            "Agricultural land-use": read_agricultural_area(zip_path, BASE_YEAR),
            "Ag management": read_ag_management_area(zip_path, BASE_YEAR),
            "Non-ag": read_non_ag_area(zip_path, BASE_YEAR),
        }
        summaries_2050 = {
            "Agricultural land-use": read_agricultural_area(zip_path, YEAR),
            "Ag management": read_ag_management_area(zip_path, YEAR),
            "Non-ag": read_non_ag_area(zip_path, YEAR),
        }

        print(f"  {varying_key}={format_thousands(price)}:")

        for area_type in AREA_CONFIG:
            summary_2010 = summaries_2010[area_type]
            summary_2050 = summaries_2050[area_type]
            category_order = get_category_order(
                area_type,
                list(dict.fromkeys(list(summary_2010) + list(summary_2050))),
            )

            total_2010 = sum(summary_2010.values())
            total_2050 = sum(summary_2050.values())
            print(
                f"    {area_type}: 2010={total_2010:.2f} Mha, "
                f"2050={total_2050:.2f} Mha, change={total_2050 - total_2010:.2f} Mha"
            )

            for category in category_order:
                area_2010 = summary_2010.get(category, 0.0)
                area_2050 = summary_2050.get(category, 0.0)
                rows.append({
                    "PriceType": price_type,
                    "Price": price,
                    "AreaType": area_type,
                    "Category": category,
                    "Area_2010_Mha": area_2010,
                    "Area_2050_Mha": area_2050,
                    "AreaChangeSince2010_Mha": area_2050 - area_2010,
                })

    return rows


def load_cache():
    if not CACHE_PATH.is_file():
        return None

    try:
        print(f"Loading cached data from {CACHE_PATH}")
        df_long = pd.read_excel(CACHE_PATH, sheet_name="AreaLong")
    except ValueError:
        print("Cached area workbook uses an older layout; rebuilding.")
        return None

    required_columns = {
        "PriceType",
        "Price",
        "AreaType",
        "Category",
        "Area_2010_Mha",
        "Area_2050_Mha",
        "AreaChangeSince2010_Mha",
    }
    if not required_columns.issubset(df_long.columns):
        print("Cached area data schema is outdated; rebuilding.")
        return None

    return df_long


def collect_and_cache():
    run_map, cp_vals, bp_vals = build_run_map()

    print(f"\n--- Slice A: BioPrice=0, carbon price varies ({BASE_YEAR}->{YEAR}) ---")
    rows_cp = collect_slice_rows(run_map, cp_vals, "cp")

    print(f"\n--- Slice B: CarbonPrice=0, biodiversity price varies ({BASE_YEAR}->{YEAR}) ---")
    rows_bp = collect_slice_rows(run_map, bp_vals, "bp")

    df_long = pd.DataFrame(rows_cp + rows_bp)
    df_long = df_long.sort_values(["PriceType", "AreaType", "Price", "Category"]).reset_index(drop=True)

    with pd.ExcelWriter(CACHE_PATH, engine="openpyxl") as writer:
        df_long.to_excel(writer, sheet_name="AreaLong", index=False)
        df_long[df_long["PriceType"] == "CarbonPrice"].to_excel(writer, sheet_name="CarbonPrice", index=False)
        df_long[df_long["PriceType"] == "BioPrice"].to_excel(writer, sheet_name="BioPrice", index=False)

    print(f"\nCache saved: {CACHE_PATH}")
    return df_long


def get_category_order(area_type, categories_seen):
    base_order = AREA_CONFIG[area_type]["order"]
    ordered = [category for category in base_order if category in categories_seen]
    ordered += [category for category in categories_seen if category not in base_order]
    return ordered


def build_area_pivot(df_long, price_type, area_type):
    df_subset = df_long[
        (df_long["PriceType"] == price_type) &
        (df_long["AreaType"] == area_type)
    ]

    if df_subset.empty:
        return pd.DataFrame()

    pivot = df_subset.pivot_table(
        index="Price",
        columns="Category",
        values="AreaChangeSince2010_Mha",
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

    color_map = AREA_CONFIG[area_type]["color_map"]
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
                label=category,
            )
            positive_bottoms += positive

        if not np.isclose(np.abs(negative).sum(), 0.0):
            ax.bar(
                x,
                negative,
                0.75,
                bottom=negative_bottoms,
                color=color_map.get(category, "#888888"),
                label=category,
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


fig, axes = plt.subplots(
    3,
    2,
    figsize=(16, 11),
    sharex="col",
)

axes[0, 0].set_title("Carbon price")
axes[0, 1].set_title("Biodiversity price")

row_area_types = ["Agricultural land-use", "Ag management", "Non-ag"]
row_legends = {}

for row_idx, area_type in enumerate(row_area_types):
    ax_left = axes[row_idx, 0]
    ax_right = axes[row_idx, 1]

    pivot_cp = build_area_pivot(df_long, "CarbonPrice", area_type)
    pivot_bp = build_area_pivot(df_long, "BioPrice", area_type)

    cats_left = stacked_bar(ax_left, pivot_cp, area_type, "cp", show_xlabel=(row_idx == len(row_area_types) - 1))
    cats_right = stacked_bar(ax_right, pivot_bp, area_type, "bp", show_xlabel=(row_idx == len(row_area_types) - 1))

    ax_left.set_ylabel(AREA_CONFIG[area_type]["ylabel"])

    legend_categories = get_category_order(area_type, list(dict.fromkeys(cats_left + cats_right)))
    row_legends[area_type] = [
        mpatches.Patch(
            facecolor=AREA_CONFIG[area_type]["color_map"].get(category, "#888888"),
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
plt.subplots_adjust(right=0.77, hspace=0.22, wspace=0.12)

out_path = OUT_DIR / f"3_Area_vs_Price_{YEAR}.png"
fig.savefig(out_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"Saved: {out_path}")
