# ==============================================================================
# Figure 05: net economic return response by carbon price /
#            biodiversity price
#   Left column:  BioPrice = 0, carbon price varies
#   Right column: CarbonPrice = 0, biodiversity price varies
#   Rows: Agricultural land-use / Ag management / Non-ag
#
#   Values are differences from the zero-price run for YEAR.
#
# Notes on accounting:
#   - Solver-side economic optimisation uses `economic_contr_mrj`, assembled in
#     `luto/solvers/input_data.py`.
#   - In that solver pathway, biodiversity payment is monetised as
#     `bio_score x bio_price` before the economic objective is formed.
#   - Do not use `xr_economics_*_profit` here because those archived profit
#     layers already mix in absolute biodiversity-price revenue.
#   - Instead, recompute net economic returns from cell-level NC outputs:
#     revenue - cost - transition, removing absolute biodiversity-price revenue.
#   - For the biodiversity-price slice, add only the incremental biodiversity
#     payment:
#     bio_price x (scenario biodiversity contribution - zero-price contribution).
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
CACHE_PATH = DATA_DIR / f"05_NetEcon_Delta_vs_Zero_raw_data_{YEAR}.xlsx"
NC_CACHE_DIR = DATA_DIR / f"05_NetEcon_nc_cache_{YEAR}"

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

ADD_BIO_PAYMENT_CHANGE = True

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


def open_metric_da(zip_path, file_name, keep_coord=None):
    zip_path = Path(zip_path)
    cache_run_dir = NC_CACHE_DIR / f"{zip_path.parents[1].name}_{zip_path.parent.name}"
    cache_run_dir.mkdir(parents=True, exist_ok=True)
    cached_nc = cache_run_dir / file_name

    with zipfile.ZipFile(zip_path) as archive:
        matches = [name for name in archive.namelist() if name.endswith(file_name)]
        if not matches:
            return None

        if not cached_nc.is_file():
            with archive.open(matches[0]) as src, open(cached_nc, "wb") as dst:
                dst.write(src.read())

    ds = xr.open_dataset(cached_nc, engine="h5netcdf")

    try:
        if "layer" in ds.dims and "compress" in ds["layer"].attrs:
            ds = cfxr.decode_compress_to_multi_index(ds, "layer")

        da = next(iter(ds.data_vars.values()))
        if keep_coord is not None:
            for coord_name in list(da.coords):
                if coord_name in {"cell", "layer", keep_coord}:
                    continue

                coord = da.coords[coord_name]
                if "ALL" not in coord.values:
                    continue

                if coord.dims == ("layer",):
                    da = da.isel(layer=(coord.values == "ALL"))
                elif coord_name in da.dims:
                    da = da.sel({coord_name: "ALL"})

        return da.load()
    finally:
        ds.close()


def filter_coord(da, coord_name, value):
    if coord_name not in da.coords:
        return da

    coord = da.coords[coord_name]
    if coord_name in da.dims:
        if value in coord.values:
            return da.sel({coord_name: value})
        return da.isel({coord_name: []})

    if coord.dims == ("layer",):
        return da.isel(layer=(coord.values == value))

    return da


def to_item_cell_da(zip_path, file_name, item_coord, selectors=None, exclude_items=None):
    da = open_metric_da(zip_path, file_name)
    if da is None:
        return None

    selectors = selectors or {}
    exclude_items = set(exclude_items or {"ALL"})

    for coord_name, value in selectors.items():
        da = filter_coord(da, coord_name, value)

    if item_coord in da.coords and da.coords[item_coord].dims == ("layer",):
        item_values = da.coords[item_coord].values
        mask = np.array([item not in exclude_items for item in item_values], dtype=bool)
        da = da.isel(layer=mask)
        item_values = da.coords[item_coord].values
        da = da.assign_coords(item=("layer", item_values)).swap_dims({"layer": "item"})
        da = da.drop_vars("layer")
    elif item_coord in da.dims:
        item_values = da.coords[item_coord].values
        keep_items = [item for item in item_values if item not in exclude_items]
        da = da.sel({item_coord: keep_items}).rename({item_coord: "item"})
    else:
        raise KeyError(f"{file_name} does not contain item coordinate {item_coord!r}")

    if "layer" in da.dims:
        if da.sizes["layer"] == 1:
            da = da.isel(layer=0, drop=True)
        else:
            da = da.sum("layer")

    if "item" not in da.dims:
        raise ValueError(f"{file_name} could not be reduced to an item dimension")

    if da.sizes["item"] == 0:
        return None

    return da.transpose("cell", "item").load()


def zeros_like_items(reference):
    return xr.zeros_like(reference)


def align_item_arrays(*arrays):
    present = [arr for arr in arrays if arr is not None]
    if not present:
        return arrays

    aligned = xr.align(*present, join="outer", fill_value=0)
    result = []
    idx = 0
    template = aligned[0]
    for arr in arrays:
        if arr is None:
            result.append(zeros_like_items(template))
        else:
            result.append(aligned[idx])
            idx += 1
    return result


def item_delta_value(delta_dvar, scenario_value, scenario_dvar, baseline_value, baseline_dvar):
    delta_dvar, scenario_value, scenario_dvar, baseline_value, baseline_dvar = align_item_arrays(
        delta_dvar,
        scenario_value,
        scenario_dvar,
        baseline_value,
        baseline_dvar,
    )

    tol = 1e-10
    scenario_unit = xr.where(np.abs(scenario_dvar) > tol, scenario_value / scenario_dvar, 0)
    baseline_unit = xr.where(np.abs(baseline_dvar) > tol, baseline_value / baseline_dvar, 0)
    positive = xr.where(delta_dvar > tol, delta_dvar * scenario_unit, 0)
    negative = xr.where(delta_dvar < -tol, delta_dvar * baseline_unit, 0)
    return positive + negative


def group_item_values(da, area_type):
    if da is None:
        return {}

    values_by_item = da.sum("cell").to_series()
    result = {}
    for item, value in values_by_item.items():
        if np.isclose(value, 0.0):
            continue

        if area_type == "Agricultural land-use":
            category = LU_TO_AG_GROUP.get(normalize_name(item), "Other land")
        elif area_type == "Ag management":
            category = AM_LABEL_MAP.get(normalize_name(item), item)
        elif area_type == "Non-ag":
            if normalize_name(item) in NON_AG_EXCLUDE:
                continue
            category = NON_AG_LABEL_MAP.get(normalize_name(item), item)
        else:
            raise ValueError(f"Unknown area type: {area_type}")

        result[category] = result.get(category, 0.0) + float(value)
    return result


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
    da = open_metric_da(zip_path, file_name, keep_coord="lu")
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
    da = open_metric_da(zip_path, file_name, keep_coord="am")
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
    da = open_metric_da(zip_path, file_name, keep_coord="lu")
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


def get_bio_summaries(zip_path, year):
    return {
        "Agricultural land-use": read_ag_group_summary(zip_path, f"xr_biodiversity_overall_priority_ag_{year}.nc"),
        "Ag management": read_ag_management_summary(zip_path, f"xr_biodiversity_overall_priority_ag_management_{year}.nc"),
        "Non-ag": read_non_ag_summary(zip_path, f"xr_biodiversity_overall_priority_non_ag_{year}.nc"),
    }


def combine_summaries(*terms):
    result = {}
    for sign, summary in terms:
        for category, value in summary.items():
            result[category] = result.get(category, 0.0) + sign * value
    return {k: v for k, v in result.items() if not np.isclose(v, 0.0)}


def scale_summaries(summary_by_area, scalar):
    return {
        area_type: {
            category: value * scalar
            for category, value in summary.items()
        }
        for area_type, summary in summary_by_area.items()
    }


def sum_item_arrays(*arrays):
    present = [arr for arr in arrays if arr is not None]
    if not present:
        return None

    aligned = xr.align(*present, join="outer", fill_value=0)
    total = aligned[0].copy()
    for arr in aligned[1:]:
        total = total + arr
    return total


def get_area_item_inputs(zip_path, year, area_type, bio_price):
    if area_type == "Agricultural land-use":
        dvar = to_item_cell_da(
            zip_path,
            f"xr_dvar_ag_{year}.nc",
            "lu",
            selectors={"lm": "ALL"},
        )
        bio = to_item_cell_da(
            zip_path,
            f"xr_biodiversity_overall_priority_ag_{year}.nc",
            "lu",
            selectors={"lm": "ALL"},
        )
        revenue = to_item_cell_da(
            zip_path,
            f"xr_economics_ag_revenue_{year}.nc",
            "lu",
            selectors={"lm": "ALL", "source": "ALL"},
        )
        cost = to_item_cell_da(
            zip_path,
            f"xr_economics_ag_cost_{year}.nc",
            "lu",
            selectors={"lm": "ALL", "source": "ALL"},
        )
        transition_ag2ag = to_item_cell_da(
            zip_path,
            f"xr_economics_ag_transition_ag2ag_{year}.nc",
            "lu",
            selectors={"lm": "ALL", "source": "ALL"},
        )
        transition_non_ag2ag = to_item_cell_da(
            zip_path,
            f"xr_economics_ag_transition_non_ag2ag_{year}.nc",
            "lu",
            selectors={"lm": "ALL", "source": "ALL", "from_lu": "ALL"},
        )
        revenue, bio, cost, transition_ag2ag, transition_non_ag2ag = align_item_arrays(
            revenue,
            bio,
            cost,
            transition_ag2ag,
            transition_non_ag2ag,
        )
        net_excl_bio_price = (
            revenue
            - bio * bio_price
            - cost
            - transition_ag2ag
            - transition_non_ag2ag
        )
        return dvar, net_excl_bio_price, bio

    if area_type == "Ag management":
        selectors = {"lm": "ALL", "lu": "ALL"}
        dvar = to_item_cell_da(zip_path, f"xr_dvar_am_{year}.nc", "am", selectors=selectors)
        bio = to_item_cell_da(
            zip_path,
            f"xr_biodiversity_overall_priority_ag_management_{year}.nc",
            "am",
            selectors=selectors,
        )
        revenue = to_item_cell_da(
            zip_path,
            f"xr_economics_am_revenue_{year}.nc",
            "am",
            selectors=selectors,
        )
        cost = to_item_cell_da(
            zip_path,
            f"xr_economics_am_cost_{year}.nc",
            "am",
            selectors=selectors,
        )
        transition = to_item_cell_da(
            zip_path,
            f"xr_economics_am_transition_{year}.nc",
            "am",
            selectors=selectors,
        )
        revenue, bio, cost, transition = align_item_arrays(revenue, bio, cost, transition)
        net_excl_bio_price = revenue - bio * bio_price - cost - transition
        return dvar, net_excl_bio_price, bio

    if area_type == "Non-ag":
        dvar = to_item_cell_da(zip_path, f"xr_dvar_non_ag_{year}.nc", "lu")
        bio = to_item_cell_da(
            zip_path,
            f"xr_biodiversity_overall_priority_non_ag_{year}.nc",
            "lu",
        )
        revenue = to_item_cell_da(
            zip_path,
            f"xr_economics_non_ag_revenue_{year}.nc",
            "lu",
        )
        cost = to_item_cell_da(
            zip_path,
            f"xr_economics_non_ag_cost_{year}.nc",
            "lu",
        )
        transition_non_ag2non_ag = to_item_cell_da(
            zip_path,
            f"xr_economics_non_ag_transition_non_ag2non_ag_{year}.nc",
            "lu",
        )
        transition_non_ag2ag = to_item_cell_da(
            zip_path,
            f"xr_economics_non_ag_transition_non_ag2ag_{year}.nc",
            "lu",
        )
        revenue, bio, cost, transition_non_ag2non_ag, transition_non_ag2ag = align_item_arrays(
            revenue,
            bio,
            cost,
            transition_non_ag2non_ag,
            transition_non_ag2ag,
        )
        net_excl_bio_price = (
            revenue
            - bio * bio_price
            - cost
            - transition_non_ag2non_ag
            - transition_non_ag2ag
        )
        return dvar, net_excl_bio_price, bio

    raise ValueError(f"Unknown area type: {area_type}")


def get_all_area_item_inputs(zip_path, year, bio_price):
    return {
        area_type: get_area_item_inputs(zip_path, year, area_type, bio_price)
        for area_type in PANEL_CONFIG
    }


def get_category_order(area_type, categories_seen):
    base_order = PANEL_CONFIG[area_type]["order"]
    ordered = [category for category in base_order if category in categories_seen]
    ordered += [category for category in categories_seen if category not in base_order]
    return ordered


def collect_slice_rows(run_map, price_vals, varying_key, baseline_inputs):
    rows = []
    price_type = "CarbonPrice" if varying_key == "cp" else "BioPrice"

    for price in price_vals:
        key = (price, 0.0) if varying_key == "cp" else (0.0, price)
        zip_path = run_map.get(key)
        if zip_path is None:
            continue

        bio_price = price if varying_key == "bp" else 0.0
        scenario_inputs = get_all_area_item_inputs(zip_path, YEAR, bio_price)

        for area_type in PANEL_CONFIG:
            scenario_dvar, scenario_net, scenario_bio = scenario_inputs[area_type]
            baseline_dvar, baseline_net, baseline_bio = baseline_inputs[area_type]

            scenario_dvar, baseline_dvar = align_item_arrays(scenario_dvar, baseline_dvar)
            delta_dvar = scenario_dvar - baseline_dvar
            net_econ_change = item_delta_value(
                delta_dvar,
                scenario_net,
                scenario_dvar,
                baseline_net,
                baseline_dvar,
            )
            bio_change = item_delta_value(
                delta_dvar,
                scenario_bio,
                scenario_dvar,
                baseline_bio,
                baseline_dvar,
            )
            bio_payment_change = bio_change * price if varying_key == "bp" and ADD_BIO_PAYMENT_CHANGE else bio_change * 0
            total_change = net_econ_change + bio_payment_change

            net_summary = group_item_values(net_econ_change, area_type)
            bio_summary = group_item_values(bio_change, area_type)
            bio_payment_summary = group_item_values(bio_payment_change, area_type)
            total_summary = group_item_values(total_change, area_type)

            categories = list(dict.fromkeys(
                list(net_summary) +
                list(bio_summary) +
                list(bio_payment_summary) +
                list(total_summary)
            ))
            category_order = get_category_order(area_type, categories)

            total_net_econ = 0.0
            for category in category_order:
                net_econ_change_2025_aud = net_summary.get(category, 0.0)
                bio_change_2025_ha_yr = bio_summary.get(category, 0.0)
                bio_payment_change_2025_aud = bio_payment_summary.get(category, 0.0)
                net_econ_delta_2025_aud = total_summary.get(category, 0.0)
                net_econ_delta_2025_baud = net_econ_delta_2025_aud / 1e9
                total_net_econ += net_econ_delta_2025_baud

                rows.append({
                    "AccountingMode": "DvarDeltaVsZeroPrice",
                    "PriceType": price_type,
                    "Price": price,
                    "AreaType": area_type,
                    "Category": category,
                    "NetEconChangeExclBioPrice_2025_BAUD": net_econ_change_2025_aud / 1e9,
                    "BioChange_vs_ZeroPrice_ha_yr": bio_change_2025_ha_yr,
                    "BioPaymentChange_2025_BAUD": bio_payment_change_2025_aud / 1e9,
                    "NetEconChange_vs_ZeroPrice_BAUD": net_econ_delta_2025_baud,
                })

            print(
                f"  {varying_key}={format_thousands(price)} | {area_type}: "
                f"net_econ_difference={total_net_econ:.2f} B AUD"
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
        "NetEconChangeExclBioPrice_2025_BAUD",
        "BioChange_vs_ZeroPrice_ha_yr",
        "BioPaymentChange_2025_BAUD",
        "NetEconChange_vs_ZeroPrice_BAUD",
    }
    if not required_columns.issubset(df_long.columns):
        print("Cached net economic data schema is outdated; rebuilding.")
        return None
    if set(df_long["AccountingMode"]) != {"DvarDeltaVsZeroPrice"}:
        print("Cached net economic data uses a different accounting mode; rebuilding.")
        return None
    if OLD_LIVESTOCK_LABEL in set(df_long["Category"]):
        print("Cached net economic data uses unsplit livestock categories; rebuilding.")
        return None

    return df_long


def collect_and_cache():
    run_map, cp_vals, bp_vals = build_run_map()
    zero_zip_path = run_map.get((0.0, 0.0))
    if zero_zip_path is None:
        raise FileNotFoundError("Could not find zero-price run (CarbonPrice=0, BioPrice=0).")
    baseline_inputs = get_all_area_item_inputs(zero_zip_path, YEAR, 0.0)

    print(f"\n--- Slice A: net economic return difference at {YEAR}; BioPrice=0 and carbon price varies ---")
    rows_cp = collect_slice_rows(run_map, cp_vals, "cp", baseline_inputs)

    print(f"\n--- Slice B: net economic return difference at {YEAR}; CarbonPrice=0 and biodiversity price varies ---")
    rows_bp = collect_slice_rows(run_map, bp_vals, "bp", baseline_inputs)

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
        values="NetEconChange_vs_ZeroPrice_BAUD",
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
        values="NetEconChange_vs_ZeroPrice_BAUD",
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
    apply_price_formatter(ax, axis="y")
    apply_compact_ticks(ax, x_nbins=8, y_nbins=5)
    if show_xlabel:
        for label in ax.get_xticklabels():
            label.set_rotation(45)
            label.set_ha("right")
    style_box_axis(ax, linewidth=0.8)
    return visible_categories


df_long = load_cache()
if df_long is None:
    df_long = collect_and_cache()


fig, axes = plt.subplots(4, 2, figsize=(10, 17), sharex="col")
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

fig.supylabel(r"Difference in net economic returns relative to zero price (Billion AU\$ yr$^{-1}$)", fontsize=FS)
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

out_path = OUT_DIR / "05_NetEcon_Delta_vs_Zero.png"
fig.savefig(out_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"Saved: {out_path}")
