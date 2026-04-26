# ==============================================================================
# Figure 7: Average net cost curves under equilibrium outcomes
#   Panel A: Carbon price, comparing cp=0 vs cp=max (BioPrice=0)
#   Panel B: Biodiversity price, comparing bp=0 vs bp=max (CarbonPrice=0)
#
# Curve interpretation:
#   - First calculate the within-run change from 2010 to 2050 for the zero-price
#     scenario and for the highest-price scenario.
#   - Then calculate the additional contribution and additional net return as:
#         (highest-price run change since 2010) - (zero-price run change since 2010)
#   - Width  = additional contribution unlocked beyond the zero-price path.
#   - Height = average net cost excluding policy payment.
#
# Carbon note:
#   Archived economics outputs do not expose a clean, category-level carbon
#   transfer term. For this figure we therefore remove an approximate carbon
#   transfer from the highest-price scenario change using:
#       carbon transfer ~= carbon price x abatement since 2010
#
# Biodiversity note:
#   In paper4 we add biodiversity payment explicitly as bio_price x bio_score.
#   That term can be removed from the highest-price scenario change using:
#       biodiversity payment = bio price x biodiversity change since 2010
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
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tools.price_slice_utils import (
    DATA_DIR,
    OUT_DIR,
    build_run_map,
    format_thousands,
    style_box_axis,
)


YEAR = 2050
BASE_DIR = Path(__file__).resolve().parent
DRAW_ALL_TOOLS_DIR = BASE_DIR.parents[1] / "draw_all" / "code" / "tools"
COLOR_FILE = DRAW_ALL_TOOLS_DIR / "land use colors.xlsx"
GROUP_FILE = DRAW_ALL_TOOLS_DIR / "land use group.xlsx"
CACHE_PATH = DATA_DIR / f"7_Average_Net_Cost_raw_data_{YEAR}.xlsx"

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

NON_AG_EXCLUDE = {
    "agriculturallanduse",
    "otherlanduse",
}
MIN_WIDTH = 1e-3


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
        "order": AG_ORDER,
        "color_map": AG_COLOR_MAP,
    },
    "Ag management": {
        "order": AM_ORDER,
        "color_map": AM_COLOR_MAP,
    },
    "Non-ag": {
        "order": NON_AG_ORDER,
        "color_map": NON_AG_COLOR_MAP,
    },
}

CURVE_COLOR_MAP = {
    **AG_COLOR_MAP,
    **AM_COLOR_MAP,
    **NON_AG_COLOR_MAP,
    TRANSITION_LABEL: LU_COLOR_MAP.get(TRANSITION_LABEL, "#D2E0FB"),
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


def get_profit_summaries(zip_path, year):
    return {
        "Agricultural land-use": read_ag_group_summary(zip_path, f"xr_economics_ag_profit_{year}.nc"),
        "Ag management": read_ag_management_summary(zip_path, f"xr_economics_am_profit_{year}.nc"),
        "Non-ag": read_non_ag_summary(zip_path, f"xr_economics_non_ag_profit_{year}.nc"),
    }


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


def collect_carbon_rows(run_map, cp_max):
    # Baseline: zero-price run at YEAR; max: highest carbon price run at YEAR
    zero_zip = run_map[(0.0, 0.0)]
    max_zip = run_map[(cp_max, 0.0)]

    zero_profit_2050 = get_profit_summaries(zero_zip, YEAR)
    max_profit_2050 = get_profit_summaries(max_zip, YEAR)

    zero_ghg_2050 = get_ghg_summaries(zero_zip, YEAR)
    max_ghg_2050 = get_ghg_summaries(max_zip, YEAR)

    rows = []
    for area_type in ("Agricultural land-use", "Ag management", "Non-ag"):
        categories = list(dict.fromkeys(
            list(zero_profit_2050[area_type]) +
            list(max_profit_2050[area_type]) +
            list(zero_ghg_2050[area_type]) +
            list(max_ghg_2050[area_type])
        ))
        category_order = get_category_order(area_type, categories)

        for category in category_order:
            # zero-price run IS the baseline; its "change vs baseline" = 0
            zero_profit_change_baud = 0.0
            max_profit_change_baud = (
                max_profit_2050[area_type].get(category, 0.0)
                - zero_profit_2050[area_type].get(category, 0.0)
            ) / 1e9

            # GHG abatement vs baseline (positive = less emissions than baseline)
            zero_metric_change_mt = 0.0
            max_metric_change_mt = (
                zero_ghg_2050[area_type].get(category, 0.0)
                - max_ghg_2050[area_type].get(category, 0.0)
            ) / 1e6
            contribution_width = max_metric_change_mt  # zero_metric = 0

            zero_policy_payment_baud = 0.0
            max_policy_payment_baud = cp_max * max_metric_change_mt / 1e3
            max_profit_change_excl_policy_baud = max_profit_change_baud - max_policy_payment_baud
            delta_profit_policy_baud = max_profit_change_baud - zero_profit_change_baud
            delta_profit_excl_policy_baud = max_profit_change_excl_policy_baud

            avg_cost = np.nan
            include_in_curve = (
                contribution_width > MIN_WIDTH
                and category != TRANSITION_LABEL
            )
            if include_in_curve:
                avg_cost = -(delta_profit_excl_policy_baud * 1e3) / contribution_width

            rows.append({
                "Panel": "Carbon",
                "TargetPrice": cp_max,
                "AreaType": area_type,
                "Category": category,
                "ZeroPriceMetricChange": zero_metric_change_mt,
                "MaxPriceMetricChange": max_metric_change_mt,
                "ContributionWidth": contribution_width,
                "ContributionUnit": "Mt CO2e",
                "ZeroPriceProfitChange_BAUD": zero_profit_change_baud,
                "MaxPriceProfitChange_BAUD": max_profit_change_baud,
                "ZeroPricePolicyPayment_BAUD": zero_policy_payment_baud,
                "MaxPricePolicyPayment_BAUD": max_policy_payment_baud,
                "PolicyPaymentMethod": "Approximate: carbon price x abatement vs 2050 baseline in the highest-price run",
                "DeltaProfitPolicy_BAUD": delta_profit_policy_baud,
                "DeltaProfitExclPolicy_BAUD": delta_profit_excl_policy_baud,
                "AverageNetCost": avg_cost,
                "AverageNetCostUnit": "AUD per tCO2e",
                "IncludeInCurve": include_in_curve,
            })

    return rows


def collect_biodiversity_rows(run_map, bp_max):
    # Baseline: zero-price run at YEAR; max: highest bio price run at YEAR
    zero_zip = run_map[(0.0, 0.0)]
    max_zip = run_map[(0.0, bp_max)]

    zero_profit_2050 = get_profit_summaries(zero_zip, YEAR)
    max_profit_2050 = get_profit_summaries(max_zip, YEAR)

    zero_bio_2050 = get_bio_summaries(zero_zip, YEAR)
    max_bio_2050 = get_bio_summaries(max_zip, YEAR)

    rows = []
    for area_type in ("Agricultural land-use", "Ag management", "Non-ag"):
        categories = list(dict.fromkeys(
            list(zero_profit_2050[area_type]) +
            list(max_profit_2050[area_type]) +
            list(zero_bio_2050[area_type]) +
            list(max_bio_2050[area_type])
        ))
        category_order = get_category_order(area_type, categories)

        for category in category_order:
            # zero-price run IS the baseline; its "change vs baseline" = 0
            zero_profit_change_baud = 0.0
            max_profit_change_base_baud = (
                max_profit_2050[area_type].get(category, 0.0)
                - zero_profit_2050[area_type].get(category, 0.0)
            ) / 1e9

            zero_metric_change_mha = 0.0
            max_metric_change_mha = (
                max_bio_2050[area_type].get(category, 0.0)
                - zero_bio_2050[area_type].get(category, 0.0)
            ) / 1e6
            contribution_width = max_metric_change_mha  # zero_metric = 0

            zero_policy_payment_baud = 0.0
            max_policy_payment_baud = bp_max * max_metric_change_mha / 1e3
            max_profit_change_baud = max_profit_change_base_baud + max_policy_payment_baud
            delta_profit_policy_baud = max_profit_change_baud - zero_profit_change_baud
            delta_profit_excl_policy_baud = max_profit_change_base_baud

            avg_cost = np.nan
            include_in_curve = contribution_width > MIN_WIDTH
            if include_in_curve:
                avg_cost = -(delta_profit_excl_policy_baud * 1e3) / contribution_width

            rows.append({
                "Panel": "Biodiversity",
                "TargetPrice": bp_max,
                "AreaType": area_type,
                "Category": category,
                "ZeroPriceMetricChange": zero_metric_change_mha,
                "MaxPriceMetricChange": max_metric_change_mha,
                "ContributionWidth": contribution_width,
                "ContributionUnit": "Mha yr^-1",
                "ZeroPriceProfitChange_BAUD": zero_profit_change_baud,
                "MaxPriceProfitChange_BAUD": max_profit_change_baud,
                "ZeroPricePolicyPayment_BAUD": zero_policy_payment_baud,
                "MaxPricePolicyPayment_BAUD": max_policy_payment_baud,
                "PolicyPaymentMethod": "Explicit in paper4 plotting logic: biodiversity price x biodiversity change vs 2050 baseline in the highest-price run",
                "DeltaProfitPolicy_BAUD": delta_profit_policy_baud,
                "DeltaProfitExclPolicy_BAUD": delta_profit_excl_policy_baud,
                "AverageNetCost": avg_cost,
                "AverageNetCostUnit": "AUD per (ha yr^-1)",
                "IncludeInCurve": include_in_curve,
            })

    return rows


def load_cache():
    if not CACHE_PATH.is_file():
        return None, None

    try:
        print(f"Loading cached data from {CACHE_PATH}")
        df_curve = pd.read_excel(CACHE_PATH, sheet_name="CurveData")
        df_metadata = pd.read_excel(CACHE_PATH, sheet_name="Metadata")
    except ValueError:
        print("Cached average-cost workbook uses an older layout; rebuilding.")
        return None, None

    required_columns = {
        "Panel",
        "AreaType",
        "Category",
        "ZeroPriceMetricChange",
        "MaxPriceMetricChange",
        "ContributionWidth",
        "ZeroPriceProfitChange_BAUD",
        "MaxPriceProfitChange_BAUD",
        "MaxPricePolicyPayment_BAUD",
        "DeltaProfitExclPolicy_BAUD",
        "AverageNetCost",
        "IncludeInCurve",
    }
    if not required_columns.issubset(df_curve.columns):
        print("Cached average-cost data schema is outdated; rebuilding.")
        return None, None

    return df_curve, df_metadata


def collect_and_cache():
    run_map, cp_vals, bp_vals = build_run_map()
    cp_max = max(cp_vals)
    bp_max = max(bp_vals)

    carbon_rows = collect_carbon_rows(run_map, cp_max)
    bio_rows = collect_biodiversity_rows(run_map, bp_max)

    df_curve = pd.DataFrame(carbon_rows + bio_rows)
    df_curve = df_curve.sort_values(["Panel", "AreaType", "Category"]).reset_index(drop=True)
    df_metadata = pd.DataFrame([
        {
            "Panel": "Carbon",
            "TargetPrice": cp_max,
            "Note": f"Curve width is the additional GHG abatement in the max-price run vs the zero-price (baseline) run at {YEAR}. Carbon transfer is removed approximately using price x abatement vs baseline in the highest-price run.",
        },
        {
            "Panel": "Biodiversity",
            "TargetPrice": bp_max,
            "Note": f"Curve width is the additional biodiversity contribution in the max-price run vs the zero-price (baseline) run at {YEAR}. Biodiversity payment is removed using price x biodiversity change vs baseline in the highest-price run.",
        },
    ])

    with pd.ExcelWriter(CACHE_PATH, engine="openpyxl") as writer:
        df_curve.to_excel(writer, sheet_name="CurveData", index=False)
        df_curve[df_curve["Panel"] == "Carbon"].to_excel(writer, sheet_name="Carbon", index=False)
        df_curve[df_curve["Panel"] == "Biodiversity"].to_excel(writer, sheet_name="Biodiversity", index=False)
        df_metadata.to_excel(writer, sheet_name="Metadata", index=False)

    print(f"Cache saved: {CACHE_PATH}")
    return df_curve, df_metadata


def prepare_curve(df_curve, panel_name):
    df_panel = df_curve[(df_curve["Panel"] == panel_name) & (df_curve["IncludeInCurve"])].copy()
    if df_panel.empty:
        return df_panel

    df_panel = df_panel.sort_values(
        ["AverageNetCost", "ContributionWidth", "Category"],
        ascending=[True, False, True],
    ).reset_index(drop=True)
    df_panel["x_left"] = df_panel["ContributionWidth"].cumsum().shift(fill_value=0.0)
    df_panel["x_right"] = df_panel["x_left"] + df_panel["ContributionWidth"]
    return df_panel


VALUE_FORMATTERS = {
    "Carbon": ticker.FuncFormatter(lambda value, _: format_thousands(value)),
    "Biodiversity": ticker.FuncFormatter(lambda value, _: format_thousands(value)),
}

PANEL_SEGMENTS = {
    "Carbon": [(600, 1100), (-650, 350)],
    "Biodiversity": [(2200, 2800), (-100, 900), (-8000, -6000)],
}

THIN_BAR_ABS_THRESHOLDS = {
    "Carbon": 0.5,
    "Biodiversity": 0.1,
}
THIN_BAR_SHARE_THRESHOLD = 0.02

THIN_BAR_UNITS = {
    "Carbon": "Mt CO$_2$e",
    "Biodiversity": "Mha yr$^{-1}$",
}


def get_legend_handles(df_panel):
    handles = []
    seen_categories = set()
    for category in df_panel["Category"]:
        if category in seen_categories:
            continue
        handles.append(
            mpatches.Patch(
                facecolor=CURVE_COLOR_MAP.get(category, "#888888"),
                edgecolor="none",
                label=category,
            )
        )
        seen_categories.add(category)
    return handles


def add_break_marks(ax_upper, ax_lower, d=0.008):
    kwargs_upper = dict(transform=ax_upper.transAxes, color="black", clip_on=False, linewidth=0.8)
    ax_upper.plot((-d, +d), (-d, +d), **kwargs_upper)
    ax_upper.plot((1 - d, 1 + d), (-d, +d), **kwargs_upper)

    kwargs_lower = dict(transform=ax_lower.transAxes, color="black", clip_on=False, linewidth=0.8)
    ax_lower.plot((-d, +d), (1 - d, 1 + d), **kwargs_lower)
    ax_lower.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs_lower)


def get_thin_bar_subset(df_panel, panel_name):
    total_width = float(df_panel["ContributionWidth"].sum())
    threshold = max(
        THIN_BAR_ABS_THRESHOLDS[panel_name],
        total_width * THIN_BAR_SHARE_THRESHOLD,
    )
    df_thin = df_panel[df_panel["ContributionWidth"] < threshold].copy()
    return df_thin, threshold


def draw_thin_bar_zoom(zoom_ax, df_panel, panel_name):
    df_thin, threshold = get_thin_bar_subset(df_panel, panel_name)
    if df_thin.empty:
        zoom_ax.set_visible(False)
        return

    df_thin["zoom_left"] = df_thin["ContributionWidth"].cumsum().shift(fill_value=0.0)
    df_thin["zoom_right"] = df_thin["zoom_left"] + df_thin["ContributionWidth"]

    for _, row in df_thin.iterrows():
        zoom_ax.bar(
            row["zoom_left"],
            row["AverageNetCost"],
            width=row["ContributionWidth"],
            align="edge",
            color=CURVE_COLOR_MAP.get(row["Category"], "#888888"),
            edgecolor="white",
            linewidth=0.4,
        )

    zoom_ax.set_xlim(0.0, float(df_thin["zoom_right"].max()))
    zoom_ax.set_yscale("symlog", linthresh=100.0)
    if df_thin["AverageNetCost"].min() < 0.0 < df_thin["AverageNetCost"].max():
        zoom_ax.axhline(0.0, color="#444444", linewidth=0.6)

    zoom_ax.set_title(
        f"Thin bars zoom ({len(df_thin)})\n(width < {format_thousands(threshold)} {THIN_BAR_UNITS[panel_name]})",
        fontsize=FS - 3,
        pad=2,
    )
    zoom_ax.xaxis.set_major_formatter(VALUE_FORMATTERS[panel_name])
    zoom_ax.tick_params(axis="both", labelsize=FS - 4, length=2)
    style_box_axis(zoom_ax, linewidth=0.6)


def draw_curve(fig, outer_spec, df_panel, panel_name, title, xlabel, ylabel):
    if df_panel.empty:
        ax = fig.add_subplot(outer_spec)
        ax.text(0.5, 0.5, "No positive contribution categories", ha="center", va="center", transform=ax.transAxes)
        style_box_axis(ax, linewidth=0.8)
        return []

    segments = PANEL_SEGMENTS[panel_name]
    df_thin, _ = get_thin_bar_subset(df_panel, panel_name)
    has_zoom = not df_thin.empty

    inner = outer_spec.subgridspec(
        len(segments),
        1,
        hspace=0.05,
    )

    axes = []
    shared_ax = None
    for idx in range(len(segments)):
        ax = fig.add_subplot(inner[idx, 0], sharex=shared_ax)
        if shared_ax is None:
            shared_ax = ax
        axes.append(ax)

    for ax, (ymin, ymax) in zip(axes, segments):
        for _, row in df_panel.iterrows():
            ax.bar(
                row["x_left"],
                row["AverageNetCost"],
                width=row["ContributionWidth"],
                align="edge",
                color=CURVE_COLOR_MAP.get(row["Category"], "#888888"),
                edgecolor="white",
                linewidth=0.5,
            )

        ax.set_ylim(ymin, ymax)
        if ymin < 0.0 < ymax:
            ax.axhline(0.0, color="#444444", linewidth=0.8)
        style_box_axis(ax, linewidth=0.8)

    axes[0].set_title(title)
    axes[-1].set_xlabel(xlabel)
    axes[len(axes) // 2].set_ylabel(ylabel)

    x_max = float(df_panel["x_right"].max())
    for ax in axes:
        ax.set_xlim(0.0, x_max)
        ax.xaxis.set_major_formatter(VALUE_FORMATTERS[panel_name])

    for ax in axes[:-1]:
        ax.tick_params(axis="x", labelbottom=False, bottom=False)
        ax.spines["bottom"].set_visible(False)

    for ax in axes[1:]:
        ax.spines["top"].set_visible(False)

    for ax_upper, ax_lower in zip(axes[:-1], axes[1:]):
        add_break_marks(ax_upper, ax_lower)

    if has_zoom:
        zoom_ax = axes[0].inset_axes([0.08, 0.08, 0.40, 0.52])
        draw_thin_bar_zoom(zoom_ax, df_panel, panel_name)

    return get_legend_handles(df_panel)


df_curve, df_metadata = load_cache()
if df_curve is None or df_metadata is None:
    df_curve, df_metadata = collect_and_cache()

df_carbon = prepare_curve(df_curve, "Carbon")
df_bio = prepare_curve(df_curve, "Biodiversity")

carbon_price = float(df_curve.loc[df_curve["Panel"] == "Carbon", "TargetPrice"].iloc[0])
bio_price = float(df_curve.loc[df_curve["Panel"] == "Biodiversity", "TargetPrice"].iloc[0])

fig = plt.figure(figsize=(18, 8))
outer = fig.add_gridspec(1, 2, wspace=0.22)

handles_carbon = draw_curve(
    fig,
    outer[0, 0],
    df_carbon,
    "Carbon",
    f"Carbon average net cost curve\n(max vs 0 at {YEAR}; cp = {format_thousands(carbon_price)} AUD per tCO$_2$e)",
    r"Additional abatement beyond cp=0 (Mt CO$_2$e)",
    "Average net cost excluding carbon transfer\n(AUD per tCO$_2$e)",
)

handles_bio = draw_curve(
    fig,
    outer[0, 1],
    df_bio,
    "Biodiversity",
    f"Biodiversity average net cost curve\n(max vs 0 at {YEAR}; bp = {format_thousands(bio_price)} AUD per ha)",
    r"Additional biodiversity contribution beyond bp=0 (Mha yr$^{-1}$)",
    "Average net cost excluding biodiversity payment\n(AUD per (ha yr$^{-1}$))",
)

all_handles = []
seen_labels = set()
for handle in handles_carbon + handles_bio:
    if handle.get_label() in seen_labels:
        continue
    all_handles.append(handle)
    seen_labels.add(handle.get_label())

if all_handles:
    fig.legend(
        handles=all_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=4,
        frameon=False,
    )

fig.subplots_adjust(bottom=0.2, top=0.95)

out_path = OUT_DIR / f"7_Average_Net_Cost_Curves_{YEAR}.png"
fig.savefig(out_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"Saved: {out_path}")
