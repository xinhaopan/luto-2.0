# ==============================================================================
# Figure 4: Contribution change since 2010 vs carbon price / biodiversity price
#   Left column:  BioPrice = 0, carbon price varies
#   Right column: CarbonPrice = 0, biodiversity price varies
#   Rows: Agricultural land-use / Ag management / Non-ag
#   Note: transition GHG is folded into the first row (Agricultural land-use)
#         as a separate stacked segment labelled "Transition".
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
CACHE_PATH = DATA_DIR / f"4_Contribution_raw_data_{YEAR}.xlsx"

FS = 11
SUM_LINE_LABEL = "Sum"

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


AG_ORDER, AG_COLOR_MAP, _ = load_style_table("ag_group")
AM_ORDER, AM_COLOR_MAP, AM_LABEL_MAP = load_style_table("am")
NON_AG_ORDER, NON_AG_COLOR_MAP, NON_AG_LABEL_MAP = load_style_table("non_ag")
LU_ORDER, LU_COLOR_MAP, LU_LABEL_MAP = load_style_table("lu")
TRANSITION_LABEL = LU_LABEL_MAP.get(normalize_name("Transition"), "Transition")

group_df = pd.read_excel(GROUP_FILE)
LU_TO_AG_GROUP = {
    normalize_name(row["desc"]): row["ag_group"]
    for _, row in group_df.iterrows()
    if pd.notna(row.get("desc")) and pd.notna(row.get("ag_group"))
}

PANEL_CONFIG = {
    "Agricultural land-use": {
        "order": AG_ORDER + [TRANSITION_LABEL],
        "color_map": {**AG_COLOR_MAP, TRANSITION_LABEL: LU_COLOR_MAP.get(TRANSITION_LABEL, "#D2E0FB")},
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
    TRANSITION_LABEL,
]
TOTAL_COLOR_MAP = {
    category: LU_COLOR_MAP.get(category, "#888888")
    for category in TOTAL_ORDER
}

NON_AG_EXCLUDE = {
    normalize_name("Agricultural land-use"),
    normalize_name("Other land-use"),
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


def read_transition_ghg_total(zip_path, year):
    da = open_metric_da(zip_path, f"xr_transition_GHG_{year}.nc")
    if da is None:
        return 0.0

    return sum_with_total_coords(da)


def get_ghg_summaries(zip_path, year):
    ag_summary = read_ag_group_summary(zip_path, f"xr_GHG_ag_{year}.nc")
    transition_total = read_transition_ghg_total(zip_path, year)
    if not np.isclose(transition_total, 0.0):
        ag_summary[TRANSITION_LABEL] = ag_summary.get(TRANSITION_LABEL, 0.0) + transition_total

    return {
        "Agricultural land-use": ag_summary,
        "Ag management": read_ag_management_summary(zip_path, f"xr_GHG_ag_management_{year}.nc"),
        "Non-ag": read_non_ag_summary(zip_path, f"xr_GHG_non_ag_{year}.nc"),
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


def collect_carbon_rows(run_map, cp_vals):
    rows = []
    baseline_zip = run_map.get((0.0, 0.0))
    summary_baseline = get_ghg_summaries(baseline_zip, YEAR) if baseline_zip else {}

    print(f"\n--- Slice A: BioPrice=0, carbon price varies (baseline=cp=0,bp=0 at {YEAR}) ---")
    for cp in cp_vals:
        zip_path = run_map.get((cp, 0.0))
        if zip_path is None:
            continue

        summary_2050 = get_ghg_summaries(zip_path, YEAR)

        for area_type in PANEL_CONFIG:
            baseline_area = summary_baseline.get(area_type, {})
            categories = list(dict.fromkeys(
                list(baseline_area) + list(summary_2050[area_type])
            ))
            category_order = get_category_order(area_type, categories)

            total_mt = 0.0
            for category in category_order:
                contribution_mt = (
                    baseline_area.get(category, 0.0)
                    - summary_2050[area_type].get(category, 0.0)
                ) / 1e6
                total_mt += contribution_mt
                rows.append({
                    "PriceType": "CarbonPrice",
                    "Price": cp,
                    "AreaType": area_type,
                    "Category": category,
                    "MetricType": "GHGAbatement_vs_Baseline_MtCO2e",
                    "ContributionValue": contribution_mt,
                })

            print(f"  cp={format_thousands(cp)} | {area_type}: {total_mt:.2f} Mt CO2e")

    return rows


def collect_biodiversity_rows(run_map, bp_vals):
    rows = []
    baseline_zip = run_map.get((0.0, 0.0))
    summary_baseline = get_bio_summaries(baseline_zip, YEAR) if baseline_zip else {}

    print(f"\n--- Slice B: CarbonPrice=0, biodiversity price varies (baseline=cp=0,bp=0 at {YEAR}) ---")
    for bp in bp_vals:
        zip_path = run_map.get((0.0, bp))
        if zip_path is None:
            continue

        summary_2050 = get_bio_summaries(zip_path, YEAR)

        for area_type in PANEL_CONFIG:
            baseline_area = summary_baseline.get(area_type, {})
            categories = list(dict.fromkeys(
                list(baseline_area) + list(summary_2050[area_type])
            ))
            category_order = get_category_order(area_type, categories)

            total_mha_yr = 0.0
            for category in category_order:
                contribution_mha_yr = (
                    summary_2050[area_type].get(category, 0.0)
                    - baseline_area.get(category, 0.0)
                ) / 1e6
                total_mha_yr += contribution_mha_yr
                rows.append({
                    "PriceType": "BioPrice",
                    "Price": bp,
                    "AreaType": area_type,
                    "Category": category,
                    "MetricType": "BiodiversityChange_vs_Baseline_MhaYr",
                    "ContributionValue": contribution_mha_yr,
                })

            print(
                f"  bp={format_thousands(bp)} | {area_type}: "
                f"{total_mha_yr:.2f} Mha yr^-1"
            )

    return rows


def load_cache():
    if not CACHE_PATH.is_file():
        return None

    try:
        print(f"Loading cached data from {CACHE_PATH}")
        df_long = pd.read_excel(CACHE_PATH, sheet_name="ContributionLong")
    except ValueError:
        print("Cached contribution workbook uses an older layout; rebuilding.")
        return None

    required_columns = {
        "PriceType",
        "Price",
        "AreaType",
        "Category",
        "MetricType",
        "ContributionValue",
    }
    if not required_columns.issubset(df_long.columns):
        print("Cached contribution data schema is outdated; rebuilding.")
        return None

    return df_long


def collect_and_cache():
    run_map, cp_vals, bp_vals = build_run_map()
    rows = collect_carbon_rows(run_map, cp_vals) + collect_biodiversity_rows(run_map, bp_vals)

    df_long = pd.DataFrame(rows)
    df_long = df_long.sort_values(["PriceType", "AreaType", "Price", "Category"]).reset_index(drop=True)

    with pd.ExcelWriter(CACHE_PATH, engine="openpyxl") as writer:
        df_long.to_excel(writer, sheet_name="ContributionLong", index=False)
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
        values="ContributionValue",
        aggfunc="sum",
        fill_value=0.0,
    ).sort_index()

    category_order = get_category_order(area_type, list(pivot.columns))
    return pivot.reindex(columns=category_order, fill_value=0.0)


def map_total_category(area_type, category):
    if area_type == "Agricultural land-use" and category == TRANSITION_LABEL:
        return TRANSITION_LABEL
    return TOTAL_CATEGORY_MAP.get(area_type)


def build_total_pivot(df_long, price_type):
    df_subset = df_long[df_long["PriceType"] == price_type].copy()
    if df_subset.empty:
        return pd.DataFrame()

    df_subset["Category"] = [
        map_total_category(area_type, category)
        for area_type, category in zip(df_subset["AreaType"], df_subset["Category"])
    ]
    df_subset = df_subset.dropna(subset=["Category"])

    pivot = df_subset.pivot_table(
        index="Price",
        columns="Category",
        values="ContributionValue",
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

    if np.any(pivot_df.to_numpy() < 0.0):
        ax.axhline(0.0, color="#444444", linewidth=0.8)

    if show_sum_line:
        totals = pivot_df.sum(axis=1).to_numpy()
        plot_sum_markers(ax, x, totals)

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


fig, axes = plt.subplots(4, 2, figsize=(10, 16), sharex="col")
row_area_types = ["Agricultural land-use", "Ag management", "Non-ag"]
row_legends = {}

total_pivot_cp = build_total_pivot(df_long, "CarbonPrice")
if not total_pivot_cp.empty:
    total_pivot_cp = -total_pivot_cp
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
    if not pivot_cp.empty:
        pivot_cp = -pivot_cp
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
    "_total": 5,
    "Agricultural land-use": 5,
    "Ag management": 3,
    "Non-ag": 2,
}
LEGEND_FS = {
    "_total": FS,
    "Agricultural land-use": FS,
    "Ag management": FS,
    "Non-ag": FS - 1,
}

axes[0, 0].set_title(r"GHG emissions (Mt CO$_2$e yr$^{-1}$)", fontsize=FS)
axes[0, 1].set_title(r"Biodiversity contribution score (Mha yr$^{-1}$)", fontsize=FS)
plt.tight_layout()
plt.subplots_adjust(hspace=0.35, wspace=0.28)
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
    y_anchor = min(bb_l.y0, bb_r.y0) / fig_h_px - 0.01

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

out_path = OUT_DIR / f"4_Contribution_vs_Price_{YEAR}.png"
fig.savefig(out_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"Saved: {out_path}")
