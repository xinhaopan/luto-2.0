# ==============================================================================
# Figure 02: area composition across all price scenarios
#   Left column:  BioPrice = 0, carbon price varies
#   Right column: CarbonPrice = 0, biodiversity price varies
#   Rows: Agricultural land-use / Ag management / Non-ag
#
#   Values are differences from the zero-price run for YEAR.
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
    add_zero_line,
    apply_compact_ticks,
    apply_paper4_color_overrides_to_style_df,
    apply_price_formatter,
    build_run_map,
    format_thousands,
    get_price_axis_label,
    stacked_area_pos_neg,
    style_box_axis,
)


YEAR = 2025
BASE_DIR = Path(__file__).resolve().parent
DRAW_ALL_TOOLS_DIR = BASE_DIR.parents[1] / "draw_all" / "code" / "tools"
COLOR_FILE = DRAW_ALL_TOOLS_DIR / "land use colors.xlsx"
GROUP_FILE = DRAW_ALL_TOOLS_DIR / "land use group.xlsx"
CACHE_PATH = DATA_DIR / f"02_Area_Delta_vs_Zero_raw_data_{YEAR}.xlsx"

FS = 11
SUM_LINE_LABEL = "Sum"
OLD_LIVESTOCK_LABEL = "Livestock"
MODIFIED_LIVESTOCK_LABEL = "Modified livestock"
NATURAL_LIVESTOCK_LABEL = "Natural Livestock"
MODIFIED_LIVESTOCK_COLOR = "#762500"

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
        label = row[label_col]
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
    return group


AG_ORDER, AG_COLOR_MAP, _ = load_style_table("ag_group")
AG_ORDER, AG_COLOR_MAP = split_livestock_style(AG_ORDER, AG_COLOR_MAP)
AM_ORDER, AM_COLOR_MAP, AM_LABEL_MAP = load_style_table("am")
NON_AG_ORDER, NON_AG_COLOR_MAP, NON_AG_LABEL_MAP = load_style_table("non_ag")
LU_ORDER, LU_COLOR_MAP, LU_LABEL_MAP = load_style_table("lu")

group_df = pd.read_excel(GROUP_FILE)
LU_TO_AG_GROUP = {
    normalize_name(row["desc"]): map_ag_group(row)
    for _, row in group_df.iterrows()
    if pd.notna(row.get("desc")) and pd.notna(row.get("ag_group"))
}

AREA_CONFIG = {
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


def collect_slice_rows(run_map, price_vals, varying_key, baseline_summaries):
    price_type = "CarbonPrice" if varying_key == "cp" else "BioPrice"
    rows = []

    for price in price_vals:
        key = (price, 0.0) if varying_key == "cp" else (0.0, price)
        zip_path = run_map.get(key)
        if zip_path is None:
            continue

        summaries_2025 = {
            "Agricultural land-use": read_agricultural_area(zip_path, YEAR),
            "Ag management": read_ag_management_area(zip_path, YEAR),
            "Non-ag": read_non_ag_area(zip_path, YEAR),
        }

        print(f"  {varying_key}={format_thousands(price)}:")

        for area_type in AREA_CONFIG:
            summary_2025 = summaries_2025[area_type]
            summary_zero = baseline_summaries[area_type]
            category_order = get_category_order(
                area_type,
                list(dict.fromkeys(list(summary_2025) + list(summary_zero))),
            )

            total_delta = sum(summary_2025.values()) - sum(summary_zero.values())
            print(f"    {area_type}: difference={total_delta:.2f} Mha")

            for category in category_order:
                area_2025 = summary_2025.get(category, 0.0)
                area_zero = summary_zero.get(category, 0.0)
                area_delta = area_2025 - area_zero
                rows.append({
                    "AccountingMode": "DeltaVsZeroPrice",
                    "PriceType": price_type,
                    "Price": price,
                    "AreaType": area_type,
                    "Category": category,
                    "Area_2025_Mha": area_2025,
                    "Area_ZeroPrice_2025_Mha": area_zero,
                    "AreaChange_vs_ZeroPrice_Mha": area_delta,
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
        "AccountingMode",
        "PriceType",
        "Price",
        "AreaType",
        "Category",
        "Area_2025_Mha",
        "Area_ZeroPrice_2025_Mha",
        "AreaChange_vs_ZeroPrice_Mha",
    }
    if not required_columns.issubset(df_long.columns):
        print("Cached area data schema is outdated; rebuilding.")
        return None
    if set(df_long["AccountingMode"]) != {"DeltaVsZeroPrice"}:
        print("Cached area data uses a different accounting mode; rebuilding.")
        return None
    if OLD_LIVESTOCK_LABEL in set(df_long["Category"]):
        print("Cached area data uses unsplit livestock categories; rebuilding.")
        return None

    return df_long


def collect_and_cache():
    run_map, cp_vals, bp_vals = build_run_map()
    zero_zip_path = run_map.get((0.0, 0.0))
    if zero_zip_path is None:
        raise FileNotFoundError("Could not find zero-price run (CarbonPrice=0, BioPrice=0).")

    baseline_summaries = {
        "Agricultural land-use": read_agricultural_area(zero_zip_path, YEAR),
        "Ag management": read_ag_management_area(zero_zip_path, YEAR),
        "Non-ag": read_non_ag_area(zero_zip_path, YEAR),
    }

    print(f"\n--- Slice A: area difference at {YEAR}, BioPrice=0 and carbon price varies ---")
    rows_cp = collect_slice_rows(run_map, cp_vals, "cp", baseline_summaries)

    print(f"\n--- Slice B: area difference at {YEAR}, CarbonPrice=0 and biodiversity price varies ---")
    rows_bp = collect_slice_rows(run_map, bp_vals, "bp", baseline_summaries)

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
        values="AreaChange_vs_ZeroPrice_Mha",
        aggfunc="sum",
        fill_value=0.0,
    ).sort_index()

    category_order = get_category_order(area_type, list(pivot.columns))
    return pivot.reindex(columns=category_order, fill_value=0.0)


def build_total_area_pivot(df_long, price_type):
    df_subset = df_long[df_long["PriceType"] == price_type].copy()
    if df_subset.empty:
        return pd.DataFrame()

    df_subset["Category"] = df_subset["AreaType"].map(TOTAL_CATEGORY_MAP)
    df_subset = df_subset.dropna(subset=["Category"])

    pivot = df_subset.pivot_table(
        index="Price",
        columns="Category",
        values="AreaChange_vs_ZeroPrice_Mha",
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
        linestyle="None",
        marker="_",
        markersize=16,
        markeredgewidth=2.0,
        zorder=30,
    )
    ax.plot(
        x,
        y,
        color="black",
        linestyle="None",
        marker="o",
        markersize=4.5,
        zorder=31,
    )


def stacked_bar(ax, pivot_df, area_type, varying_key, show_xlabel, color_map=None, show_sum_line=False):
    if pivot_df.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        style_box_axis(ax, linewidth=0.8)
        return []

    color_map = AREA_CONFIG[area_type]["color_map"] if color_map is None else color_map
    visible_categories = stacked_area_pos_neg(ax, pivot_df, color_map)

    if show_sum_line:
        totals = pivot_df.sum(axis=1).to_numpy()
        plot_sum_markers(ax, pivot_df.index.to_numpy(dtype=float), totals)

    add_zero_line(ax)
    if show_xlabel:
        ax.set_xlabel(get_price_axis_label(varying_key))
    else:
        ax.tick_params(axis="x", labelbottom=False)

    apply_price_formatter(ax, axis="x")
    apply_compact_ticks(ax, x_nbins=8, y_nbins=5)
    if show_xlabel:
        for label in ax.get_xticklabels():
            label.set_rotation(45)
            label.set_ha("right")
    style_box_axis(ax, linewidth=0.8)
    return visible_categories


def sync_row_y_limits(axes):
    for row_idx in range(axes.shape[0]):
        row_limits = [axes[row_idx, col_idx].get_ylim() for col_idx in range(axes.shape[1])]
        ymin = min(limit[0] for limit in row_limits)
        ymax = max(limit[1] for limit in row_limits)
        for col_idx in range(axes.shape[1]):
            axes[row_idx, col_idx].set_ylim(ymin, ymax)


def hide_redundant_y_ticks(axes):
    for row_idx in range(axes.shape[0]):
        for col_idx in range(1, axes.shape[1]):
            axes[row_idx, col_idx].tick_params(axis="y", left=False, labelleft=False)


df_long = load_cache()
if df_long is None:
    df_long = collect_and_cache()


fig, axes = plt.subplots(
    3,
    2,
    figsize=(10, 13),
    sharex="col",
)

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
    row_legends[area_type] = build_patch_handles(
        legend_categories,
        AREA_CONFIG[area_type]["color_map"],
    )

LEGEND_NCOL = {
    "Agricultural land-use": 5,
    "Ag management": 3,
    "Non-ag": 2,
}
LEGEND_FS = {
    "Agricultural land-use": FS,
    "Ag management": FS,
    "Non-ag": FS - 1,
}

sync_row_y_limits(axes)
hide_redundant_y_ticks(axes)
fig.supylabel(r"Area difference relative to zero price (Mha)", fontsize=FS)
plt.tight_layout()
plt.subplots_adjust(hspace=0.35, wspace=0.12)
fig.canvas.draw()
renderer = fig.canvas.get_renderer()
fig_w_px = fig.get_figwidth() * fig.dpi
fig_h_px = fig.get_figheight() * fig.dpi

for row_idx, area_type in enumerate(row_area_types):
    handles = row_legends[area_type]
    if not handles:
        continue

    ax_l, ax_r = axes[row_idx, 0], axes[row_idx, 1]
    bb_l = ax_l.get_tightbbox(renderer)
    bb_r = ax_r.get_tightbbox(renderer)
    x_center = (bb_l.x0 + bb_r.x1) / 2 / fig_w_px
    y_anchor = min(bb_l.y0, bb_r.y0) / fig_h_px - 0.01

    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(x_center, y_anchor),
        bbox_transform=fig.transFigure,
        ncol=LEGEND_NCOL.get(area_type, 3),
        frameon=False,
        borderaxespad=0.0,
        handlelength=1.0,
        handleheight=1.0,
        columnspacing=1.0,
        fontsize=LEGEND_FS.get(area_type, FS - 1),
    )

out_path = OUT_DIR / "02_Area_Delta_vs_Zero.png"
fig.savefig(out_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"Saved: {out_path}")
