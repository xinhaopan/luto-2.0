# ==============================================================================
# Figure 12: absolute contribution by carbon price / biodiversity price (bar chart)
#   Columns:
#     1. GHG abatement, BioPrice = 0 and carbon price varies
#     2. GHG abatement, CarbonPrice = 0 and biodiversity price varies (co-benefit)
#     3. Biodiversity contribution, BioPrice = 0 and carbon price varies (co-benefit)
#     4. Biodiversity contribution, CarbonPrice = 0 and biodiversity price varies
#   Rows: Agricultural land-use / Ag management / Non-ag
#   Note: transition GHG is folded into the first row (Agricultural land-use)
#         as a separate stacked segment labelled "Transition".
#
#   Values are ABSOLUTE quantities for YEAR (not differences from zero price).
#   Stacked BAR chart (cf. Fig 03 which shows the relative river plot).
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
    standardize_display_label,
    apply_price_formatter,
    build_run_map,
    format_thousands,
    get_price_axis_label,
    set_sparse_index_price_ticks,
    stacked_area_pos_neg,
    style_box_axis,
)


YEAR = 2025
BASE_DIR = Path(__file__).resolve().parent
DRAW_ALL_TOOLS_DIR = BASE_DIR.parents[1] / "draw_all" / "code" / "tools"
COLOR_FILE = DRAW_ALL_TOOLS_DIR / "land use colors.xlsx"
GROUP_FILE = DRAW_ALL_TOOLS_DIR / "land use group.xlsx"
CACHE_PATH = DATA_DIR / f"12_Contribution_All_Scenarios_raw_data_{YEAR}.xlsx"

FS = 18
SUM_LINE_LABEL = "Sum"
GHG_METRIC = "GHGAbatement_2025_MtCO2e"
BIO_METRIC = "BiodiversityContribution_2025_MhaYr"
EXPECTED_METRIC_TYPES = {GHG_METRIC, BIO_METRIC}
EXPECTED_PRICE_METRIC_PAIRS = {
    ("CarbonPrice", GHG_METRIC),
    ("BioPrice", GHG_METRIC),
    ("CarbonPrice", BIO_METRIC),
    ("BioPrice", BIO_METRIC),
}
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
LU_ORDER, LU_COLOR_MAP, LU_LABEL_MAP = load_style_table("lu")

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

TRANSITION_LABEL = LU_LABEL_MAP.get(normalize_name("Transition"), "Transition")

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


def collect_metric_rows(run_map, price_vals, varying_key, metric_type):
    rows = []

    if metric_type == GHG_METRIC:
        get_summaries = get_ghg_summaries
        units = "Mt CO2e"
        metric_label = "GHG abatement"
    elif metric_type == BIO_METRIC:
        get_summaries = get_bio_summaries
        units = "Mha yr^-1"
        metric_label = "biodiversity contribution"
    else:
        raise ValueError(f"Unsupported metric_type: {metric_type}")

    if varying_key == "cp":
        price_type = "CarbonPrice"
        fixed_label = "BioPrice=0 and carbon price varies"
    elif varying_key == "bp":
        price_type = "BioPrice"
        fixed_label = "CarbonPrice=0 and biodiversity price varies"
    else:
        raise ValueError(f"Unsupported varying_key: {varying_key}")

    print(f"\n--- {metric_label} at {YEAR}; {fixed_label} ---")
    for price in price_vals:
        key = (price, 0.0) if varying_key == "cp" else (0.0, price)
        zip_path = run_map.get(key)
        if zip_path is None:
            continue

        summary_2025 = get_summaries(zip_path, YEAR)

        for area_type in PANEL_CONFIG:
            categories = list(dict.fromkeys(list(summary_2025[area_type])))
            category_order = get_category_order(area_type, categories)

            total_value = 0.0
            for category in category_order:
                raw_value = summary_2025[area_type].get(category, 0.0)
                if metric_type == GHG_METRIC:
                    contribution_value = -raw_value / 1e6
                else:
                    contribution_value = raw_value / 1e6

                total_value += contribution_value
                rows.append({
                    "PriceType": price_type,
                    "Price": price,
                    "AreaType": area_type,
                    "Category": category,
                    "MetricType": metric_type,
                    "ContributionValue": contribution_value,
                })

            print(
                f"  {varying_key}={format_thousands(price)} | {area_type}: "
                f"absolute={total_value:.2f} {units}"
            )

    return rows


def load_cache():
    if not CACHE_PATH.is_file():
        return None

    try:
        print(f"Loading cached data from {CACHE_PATH}")
        df_long = pd.read_excel(CACHE_PATH, sheet_name="ContributionLong")
        df_long["Category"] = df_long["Category"].map(standardize_display_label)
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
    if set(df_long["MetricType"]) != EXPECTED_METRIC_TYPES:
        print("Cached contribution data metric coverage is outdated; rebuilding.")
        return None
    cached_pairs = set(zip(df_long["PriceType"], df_long["MetricType"]))
    if not EXPECTED_PRICE_METRIC_PAIRS.issubset(cached_pairs):
        print("Cached contribution data does not include cross-price metrics; rebuilding.")
        return None
    if OLD_LIVESTOCK_LABEL in set(df_long["Category"]):
        print("Cached contribution data uses unsplit livestock categories; rebuilding.")
        return None

    return df_long


def collect_and_cache():
    run_map, cp_vals, bp_vals = build_run_map()
    rows = (
        collect_metric_rows(run_map, cp_vals, "cp", GHG_METRIC) +
        collect_metric_rows(run_map, bp_vals, "bp", GHG_METRIC) +
        collect_metric_rows(run_map, cp_vals, "cp", BIO_METRIC) +
        collect_metric_rows(run_map, bp_vals, "bp", BIO_METRIC)
    )

    df_long = pd.DataFrame(rows)
    df_long = df_long.sort_values(
        ["MetricType", "PriceType", "AreaType", "Price", "Category"]
    ).reset_index(drop=True)

    with pd.ExcelWriter(CACHE_PATH, engine="openpyxl") as writer:
        df_long.to_excel(writer, sheet_name="ContributionLong", index=False)
        for price_type, metric_type in sorted(EXPECTED_PRICE_METRIC_PAIRS):
            sheet_name = (
                ("CP" if price_type == "CarbonPrice" else "BP") +
                ("_GHG" if metric_type == GHG_METRIC else "_Bio")
            )
            df_long[
                (df_long["PriceType"] == price_type) &
                (df_long["MetricType"] == metric_type)
            ].to_excel(writer, sheet_name=sheet_name, index=False)

    print(f"\nCache saved: {CACHE_PATH}")
    return df_long


def build_pivot(df_long, price_type, metric_type, area_type):
    df_subset = df_long[
        (df_long["PriceType"] == price_type) &
        (df_long["MetricType"] == metric_type) &
        (df_long["AreaType"] == area_type)
    ]
    if area_type == "Agricultural land-use":
        df_subset = df_subset[df_subset["Category"] != TRANSITION_LABEL]

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


def build_total_pivot(df_long, price_type, metric_type):
    df_subset = df_long[
        (df_long["PriceType"] == price_type) &
        (df_long["MetricType"] == metric_type)
    ].copy()
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
        linestyle="-",
        marker="o",
        linewidth=1.8,
        markersize=4.5,
        markeredgewidth=0,
        zorder=30,
    )


def stacked_bar(ax, pivot_df, area_type, varying_key, show_xlabel, show_xticks=None, color_map=None, show_sum_line=False):
    if show_xticks is None:
        show_xticks = show_xlabel
    if pivot_df.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        style_box_axis(ax, linewidth=0.8)
        return []

    color_map = PANEL_CONFIG[area_type]["color_map"] if color_map is None else color_map
    visible_categories = stacked_area_pos_neg(ax, pivot_df, color_map)

    if show_sum_line:
        totals = pivot_df.sum(axis=1).to_numpy()
        plot_sum_markers(ax, pivot_df.index.to_numpy(dtype=float), totals)

    add_zero_line(ax)
    if show_xlabel:
        raw_label = get_price_axis_label(varying_key)
        ax.set_xlabel(raw_label.replace(" (", "\n("))
        ax.xaxis.label.set_fontsize(FS - 3)
    elif not show_xticks:
        ax.tick_params(axis="x", labelbottom=False)

    apply_price_formatter(ax, axis="x")
    apply_compact_ticks(ax, x_nbins=8, y_nbins=5)
    if show_xticks:
        for label in ax.get_xticklabels():
            label.set_rotation(45)
            label.set_ha("right")
    style_box_axis(ax, linewidth=0.8)
    return visible_categories


COLUMN_CONFIG = [
    {
        "price_type": "CarbonPrice",
        "metric_type": GHG_METRIC,
        "varying_key": "cp",
        "metric_group": "ghg",
    },
    {
        "price_type": "BioPrice",
        "metric_type": GHG_METRIC,
        "varying_key": "bp",
        "metric_group": "ghg",
    },
    {
        "price_type": "CarbonPrice",
        "metric_type": BIO_METRIC,
        "varying_key": "cp",
        "metric_group": "bio",
    },
    {
        "price_type": "BioPrice",
        "metric_type": BIO_METRIC,
        "varying_key": "bp",
        "metric_group": "bio",
    },
]


def sync_pair_y_limits(axes, col_pairs):
    for row_idx in range(axes.shape[0]):
        for col_a, col_b in col_pairs:
            limits = [axes[row_idx, col_a].get_ylim(), axes[row_idx, col_b].get_ylim()]
            ymin = min(limit[0] for limit in limits)
            ymax = max(limit[1] for limit in limits)
            axes[row_idx, col_a].set_ylim(ymin, ymax)
            axes[row_idx, col_b].set_ylim(ymin, ymax)


def apply_y_axis_visibility(axes):
    for row_idx in range(axes.shape[0]):
        axes[row_idx, 0].tick_params(axis="y", left=True, labelleft=True, right=False, labelright=False)
        axes[row_idx, 1].tick_params(axis="y", left=False, labelleft=False, right=False, labelright=False)
        axes[row_idx, 2].tick_params(axis="y", left=False, labelleft=False, right=False, labelright=False)
        axes[row_idx, 3].yaxis.tick_right()
        axes[row_idx, 3].tick_params(axis="y", left=False, labelleft=False, right=True, labelright=True)


def add_column_group_title(fig, axes, columns, title):
    left = axes[0, columns[0]].get_position().x0
    right = axes[0, columns[-1]].get_position().x1
    top = max(axes[0, col_idx].get_position().y1 for col_idx in columns)
    fig.text(
        (left + right) / 2,
        top + 0.015,
        title,
        ha="center",
        va="bottom",
        fontsize=FS + 2,
        fontweight="bold",
    )


df_long = load_cache()
if df_long is None:
    df_long = collect_and_cache()


fig, axes = plt.subplots(4, 4, figsize=(18, 22), sharex="col")
row_area_types = ["Agricultural land-use", "Ag management", "Non-ag"]
row_legends = {}

total_cats_all = []
for col_idx, col_cfg in enumerate(COLUMN_CONFIG):
    total_pivot = build_total_pivot(
        df_long,
        col_cfg["price_type"],
        col_cfg["metric_type"],
    )
    total_cats_all.extend(
        stacked_bar(
            axes[0, col_idx],
            total_pivot,
            "Total",
            col_cfg["varying_key"],
            show_xlabel=False,
            color_map=TOTAL_COLOR_MAP,
            show_sum_line=True,
        )
    )
axes[0, 0].set_ylabel("Total")

total_legend_categories = [category for category in TOTAL_ORDER if category in dict.fromkeys(total_cats_all)]
total_handles = build_patch_handles(total_legend_categories, TOTAL_COLOR_MAP)
if total_handles:
    total_handles = [build_total_line_handle()] + total_handles
row_legends["_total"] = total_handles if total_handles else []

for row_idx, area_type in enumerate(row_area_types):
    row_cats_all = []
    for col_idx, col_cfg in enumerate(COLUMN_CONFIG):
        pivot = build_pivot(
            df_long,
            col_cfg["price_type"],
            col_cfg["metric_type"],
            area_type,
        )
        row_cats_all.extend(
            stacked_bar(
                axes[row_idx + 1, col_idx],
                pivot,
                area_type,
                col_cfg["varying_key"],
                show_xlabel=(row_idx == len(row_area_types) - 1),
                show_xticks=(row_idx == len(row_area_types) - 1),
            )
        )

    axes[row_idx + 1, 0].set_ylabel(PANEL_CONFIG[area_type]["ylabel"])

    legend_categories = get_category_order(area_type, list(dict.fromkeys(row_cats_all)))
    row_legends[area_type] = build_patch_handles(
        legend_categories,
        PANEL_CONFIG[area_type]["color_map"],
    )

LEGEND_NCOL = {
    "_total": 5,
    "Agricultural land-use": 5,
    "Ag management": 3,
    "Non-ag": 3,
}
LEGEND_FS = {
    "_total": FS,
    "Agricultural land-use": FS,
    "Ag management": FS,
    "Non-ag": FS,
}

sync_pair_y_limits(axes, [(0, 1), (2, 3)])
apply_y_axis_visibility(axes)
plt.tight_layout()
plt.subplots_adjust(hspace=0.35, wspace=0.18, top=0.94)
for ax in axes[-1, :]:
    ax.xaxis.set_label_coords(0.5, -0.24)
add_column_group_title(fig, axes, (0, 1), "GHG")
add_column_group_title(fig, axes, (2, 3), "Biodiversity")
fig.canvas.draw()
renderer = fig.canvas.get_renderer()
fig_w_px = fig.get_figwidth() * fig.dpi
fig_h_px = fig.get_figheight() * fig.dpi

bb_ghg = [axes[r, c].get_tightbbox(renderer) for r in range(4) for c in (0, 1)]
bb_bio = [axes[r, c].get_tightbbox(renderer) for r in range(4) for c in (2, 3)]
y_mid_l = (max(b.y1 for b in bb_ghg) + min(b.y0 for b in bb_ghg)) / 2 / fig_h_px
y_mid_r = (max(b.y1 for b in bb_bio) + min(b.y0 for b in bb_bio)) / 2 / fig_h_px
x_l = min(b.x0 for b in bb_ghg) / fig_w_px - 0.025
x_r = max(b.x1 for b in bb_bio) / fig_w_px + 0.02
fig.text(x_l, y_mid_l, r"GHG abatement (Mt CO$_2$e yr$^{-1}$)",
         rotation=90, va='center', ha='center', fontsize=FS + 1, fontweight="bold")
fig.text(x_r, y_mid_r, r"Biodiversity contribution (Mha yr$^{-1}$)",
         rotation=270, va='center', ha='center', fontsize=FS + 1, fontweight="bold")

all_rows = [("_total", 0)] + [(area_type, i + 1) for i, area_type in enumerate(row_area_types)]
for key, row_idx in all_rows:
    handles = row_legends.get(key, [])
    if not handles:
        continue

    row_bboxes = [axes[row_idx, col_idx].get_tightbbox(renderer) for col_idx in range(4)]
    x_center = (min(b.x0 for b in row_bboxes) + max(b.x1 for b in row_bboxes)) / 2 / fig_w_px
    # Last row has xlabel; keep the legend below it without adding a large gap.
    y_offset = 0.085 if row_idx == len(all_rows) - 1 else 0.01
    y_anchor = min(axes[row_idx, c].get_position().y0 for c in range(4)) - y_offset

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

out_path = OUT_DIR / "12_Contribution_All_Scenarios.png"
fig.savefig(out_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"Saved: {out_path}")
