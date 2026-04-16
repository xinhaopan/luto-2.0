# ==============================================================================
# Figure 5: Spatial distribution of area composition (2050)
#   Left column:  max carbon price slice       (BioPrice=0)
#   Right column: max biodiversity price slice (CarbonPrice=0)
#   Rows: Agricultural land-use / Ag management / Non-ag
# ==============================================================================

import io
import os
import re
import sys
import zipfile
from pathlib import Path

import cf_xarray as cfxr
import geopandas as gpd
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tools.price_slice_utils import (
    build_run_map,
    DATA_DIR,
    format_thousands,
    OUT_DIR,
)


YEAR = 2050
BASE_DIR = Path(__file__).resolve().parent
DRAW_ALL_TOOLS_DIR = BASE_DIR.parents[1] / "draw_all" / "code" / "tools"
COLOR_FILE = DRAW_ALL_TOOLS_DIR / "land use colors.xlsx"
GROUP_FILE = DRAW_ALL_TOOLS_DIR / "land use group.xlsx"
SHP_FILE = "../../paper4/Map/AUS_line1.shp"
CACHE_NPZ = DATA_DIR / f"5_Area_Map_raw_data_{YEAR}.npz"
CACHE_META = DATA_DIR / f"5_Area_Map_meta_{YEAR}.xlsx"

EXTENT = [113.0, 153.6, -43.64, -10.04]
MAP_PAD = 0.8
XLIM = (EXTENT[0] - MAP_PAD, EXTENT[1] + MAP_PAD)
YLIM = (EXTENT[2] - MAP_PAD, EXTENT[3] + MAP_PAD)

FS = 10
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

OUTSIDE_CODE = -2


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


AG_FULL_ORDER, AG_COLOR_MAP, _ = load_style_table("ag_group")
AM_FULL_ORDER, AM_COLOR_MAP, AM_LABEL_MAP = load_style_table("am")
NON_AG_FULL_ORDER, NON_AG_COLOR_MAP, NON_AG_LABEL_MAP = load_style_table("non_ag")

AG_ACTIVE_ORDER = [label for label in AG_FULL_ORDER if label != "Other land"]
AM_ACTIVE_ORDER = [
    label for label in AM_FULL_ORDER
    if label not in ("No agricultural management", "Other land-use")
]
NON_AG_ACTIVE_ORDER = [
    label for label in NON_AG_FULL_ORDER
    if label not in ("Agricultural land-use", "Other land-use")
]

group_df = pd.read_excel(GROUP_FILE)
LU_TO_AG_GROUP = {
    normalize_name(row["desc"]): row["ag_group"]
    for _, row in group_df.iterrows()
    if pd.notna(row.get("desc")) and pd.notna(row.get("ag_group"))
}

MAP_CONFIG = {
    "Agricultural land-use": {
        "active_order": AG_ACTIVE_ORDER,
        "color_map": AG_COLOR_MAP,
        "zero_label": "Other land",
        "zero_color": AG_COLOR_MAP["Other land"],
        "background_label": "Other land",
        "background_color": AG_COLOR_MAP["Other land"],
    },
    "Ag management": {
        "active_order": AM_ACTIVE_ORDER,
        "color_map": AM_COLOR_MAP,
        "zero_label": "No agricultural management",
        "zero_color": AM_COLOR_MAP["No agricultural management"],
        "background_label": "Other land-use",
        "background_color": AM_COLOR_MAP["Other land-use"],
    },
    "Non-ag": {
        "active_order": NON_AG_ACTIVE_ORDER,
        "color_map": NON_AG_COLOR_MAP,
        "zero_label": "Agricultural land-use",
        "zero_color": NON_AG_COLOR_MAP["Agricultural land-use"],
        "background_label": "Other land-use",
        "background_color": NON_AG_COLOR_MAP["Other land-use"],
    },
}


def hex_to_rgb(hex_color):
    return (
        int(hex_color[1:3], 16) / 255,
        int(hex_color[3:5], 16) / 255,
        int(hex_color[5:7], 16) / 255,
    )


def open_dataarray(zip_path, target_name):
    with zipfile.ZipFile(zip_path) as archive:
        matches = [name for name in archive.namelist() if name.endswith(target_name)]
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


def read_template(zip_path):
    with zipfile.ZipFile(zip_path) as archive:
        matches = [
            name for name in archive.namelist()
            if name.endswith(f"xr_map_template_{YEAR}.nc")
        ]
        if not matches:
            return None

        with archive.open(matches[0]) as file_obj:
            ds = xr.open_dataset(io.BytesIO(file_obj.read()), engine="h5netcdf")

    try:
        return ds["layer"].values.astype("float32")
    finally:
        ds.close()


def select_total_coords_da(da, **selectors):
    sub = da.sel(selectors)

    for coord_name in list(sub.coords):
        if coord_name in {"cell", "layer"}:
            continue

        coord_values = np.atleast_1d(sub.coords[coord_name].values).tolist()
        if any(value == "ALL" for value in coord_values):
            sub = sub.sel({coord_name: "ALL"})

    return sub


def to_cell_values(da):
    values = np.asarray(da.values, dtype="float32")
    return values.reshape(values.size)


def build_dominant_map(zip_path, target_name, active_order, builder_fn):
    da = open_dataarray(zip_path, target_name)
    template = read_template(zip_path)
    if da is None or template is None:
        return None

    ncells = da.sizes["cell"]
    category_arrays = {
        label: np.zeros(ncells, dtype="float32")
        for label in active_order
    }

    builder_fn(da, category_arrays)

    stacked = np.stack([category_arrays[label] for label in active_order], axis=1)
    dominant_codes = np.zeros(ncells, dtype="int16")
    if stacked.shape[1] > 0:
        max_values = stacked.max(axis=1)
        max_indices = stacked.argmax(axis=1)
        has_data = max_values > 0
        for idx, label in enumerate(active_order, start=1):
            dominant_codes[has_data & (max_indices == idx - 1)] = idx

    category_map = np.full(template.shape, OUTSIDE_CODE, dtype="int16")
    background_mask = np.isfinite(template) & (template == -1)
    valid_mask = np.isfinite(template) & (template >= 0)
    category_map[background_mask] = -1
    category_map[valid_mask] = dominant_codes
    return category_map


def populate_ag_arrays(da, category_arrays):
    for lu in pd.unique(da.coords["lu"].values):
        if lu == "ALL":
            continue

        group = LU_TO_AG_GROUP.get(normalize_name(lu), "Other land")
        if group not in category_arrays:
            continue

        sub = select_total_coords_da(da, lu=lu)
        category_arrays[group] += to_cell_values(sub)


def populate_am_arrays(da, category_arrays):
    for am in pd.unique(da.coords["am"].values):
        if am == "ALL":
            continue

        label = AM_LABEL_MAP.get(normalize_name(am), am)
        if label not in category_arrays:
            continue

        sub = select_total_coords_da(da, am=am)
        category_arrays[label] += to_cell_values(sub)


def populate_non_ag_arrays(da, category_arrays):
    for lu in pd.unique(da.coords["lu"].values):
        if lu == "ALL":
            continue

        label = NON_AG_LABEL_MAP.get(normalize_name(lu), lu)
        if label not in category_arrays:
            continue

        sub = select_total_coords_da(da, lu=lu)
        category_arrays[label] += to_cell_values(sub)


def build_ag_map(zip_path):
    return build_dominant_map(
        zip_path,
        f"xr_area_agricultural_landuse_{YEAR}.nc",
        AG_ACTIVE_ORDER,
        populate_ag_arrays,
    )


def build_am_map(zip_path):
    return build_dominant_map(
        zip_path,
        f"xr_area_agricultural_management_{YEAR}.nc",
        AM_ACTIVE_ORDER,
        populate_am_arrays,
    )


def build_non_ag_map(zip_path):
    return build_dominant_map(
        zip_path,
        f"xr_area_non_agricultural_landuse_{YEAR}.nc",
        NON_AG_ACTIVE_ORDER,
        populate_non_ag_arrays,
    )


def category_map_to_rgba(cat_2d, config):
    rgba = np.zeros((*cat_2d.shape, 4), dtype="float32")

    rgba[cat_2d == OUTSIDE_CODE] = [1.0, 1.0, 1.0, 1.0]

    zero_rgb = hex_to_rgb(config["zero_color"])
    rgba[cat_2d == 0] = [*zero_rgb, 1.0]

    background_rgb = hex_to_rgb(config["background_color"])
    rgba[cat_2d == -1] = [*background_rgb, 1.0]

    for idx, label in enumerate(config["active_order"], start=1):
        rgb = hex_to_rgb(config["color_map"][label])
        rgba[cat_2d == idx] = [*rgb, 1.0]

    return rgba


def get_row_legend_handles(cat_left, cat_right, config):
    present_codes = set(np.unique(cat_left)).union(set(np.unique(cat_right)))
    handles = []
    seen_labels = set()

    for idx, label in enumerate(config["active_order"], start=1):
        if idx not in present_codes:
            continue
        if label in seen_labels:
            continue
        handles.append(
            mpatches.Patch(
                facecolor=config["color_map"][label],
                edgecolor="none",
                label=label,
            )
        )
        seen_labels.add(label)

    if 0 in present_codes and config["zero_label"] not in seen_labels:
        handles.append(
            mpatches.Patch(
                facecolor=config["zero_color"],
                edgecolor="none",
                label=config["zero_label"],
            )
        )
        seen_labels.add(config["zero_label"])

    if -1 in present_codes and config["background_label"] not in seen_labels:
        handles.append(
            mpatches.Patch(
                facecolor=config["background_color"],
                edgecolor="none",
                label=config["background_label"],
            )
        )

    return handles


def load_states():
    shp_path = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), SHP_FILE))
    if not os.path.exists(shp_path):
        print(f"Shapefile not found: {shp_path}")
        return None

    gdf = gpd.read_file(shp_path)
    try:
        gdf = gdf.to_crs(epsg=4283)
    except Exception:
        pass
    return gdf


def style_map_ax(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(*XLIM)
    ax.set_ylim(*YLIM)
    ax.set_aspect("equal", adjustable="box")
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.6)
        spine.set_edgecolor("#444444")


def load_cache():
    if not CACHE_NPZ.is_file() or not CACHE_META.is_file():
        return None

    print(f"Loading cached data from {CACHE_NPZ}")
    npz = np.load(CACHE_NPZ)
    try:
        maps = {
            "ag_cp": npz["ag_cp"],
            "ag_bp": npz["ag_bp"],
            "am_cp": npz["am_cp"],
            "am_bp": npz["am_bp"],
            "non_ag_cp": npz["non_ag_cp"],
            "non_ag_bp": npz["non_ag_bp"],
        }
    finally:
        npz.close()

    try:
        price_meta = pd.read_excel(CACHE_META, sheet_name="Prices")
    except ValueError:
        print("Cached map metadata uses an older layout; rebuilding.")
        return None

    max_cp = float(price_meta.loc[price_meta["panel"] == "CarbonPrice", "price_value"].iloc[0])
    max_bp = float(price_meta.loc[price_meta["panel"] == "BioPrice", "price_value"].iloc[0])
    return maps, max_cp, max_bp


def build_and_cache():
    run_map, cp_vals, bp_vals = build_run_map()
    max_cp = max(cp_vals)
    max_bp = max(bp_vals)
    print(
        "Building maps for "
        f"CP={format_thousands(max_cp)} (BioPrice=0) and "
        f"BP={format_thousands(max_bp)} (CarbonPrice=0)"
    )

    maps = {
        "ag_cp": build_ag_map(run_map[(max_cp, 0.0)]),
        "ag_bp": build_ag_map(run_map[(0.0, max_bp)]),
        "am_cp": build_am_map(run_map[(max_cp, 0.0)]),
        "am_bp": build_am_map(run_map[(0.0, max_bp)]),
        "non_ag_cp": build_non_ag_map(run_map[(max_cp, 0.0)]),
        "non_ag_bp": build_non_ag_map(run_map[(0.0, max_bp)]),
    }

    np.savez_compressed(CACHE_NPZ, **maps)

    price_meta = pd.DataFrame([
        {"panel": "CarbonPrice", "price_value": max_cp},
        {"panel": "BioPrice", "price_value": max_bp},
    ])

    legend_rows = []
    for area_type, config in MAP_CONFIG.items():
        for idx, label in enumerate(config["active_order"], start=1):
            legend_rows.append({
                "AreaType": area_type,
                "Code": idx,
                "Label": label,
                "Color": config["color_map"][label],
            })
        legend_rows.append({
            "AreaType": area_type,
            "Code": 0,
            "Label": config["zero_label"],
            "Color": config["zero_color"],
        })
        if config["background_label"] != config["zero_label"] or config["background_color"] != config["zero_color"]:
            legend_rows.append({
                "AreaType": area_type,
                "Code": -1,
                "Label": config["background_label"],
                "Color": config["background_color"],
            })

    with pd.ExcelWriter(CACHE_META, engine="openpyxl") as writer:
        price_meta.to_excel(writer, sheet_name="Prices", index=False)
        pd.DataFrame(legend_rows).to_excel(writer, sheet_name="Legend", index=False)

    print(f"Cache saved: {CACHE_NPZ}, {CACHE_META}")
    return maps, max_cp, max_bp


cached = load_cache()
if cached is None:
    MAPS, MAX_CP, MAX_BP = build_and_cache()
else:
    MAPS, MAX_CP, MAX_BP = cached


gdf_states = load_states()

fig, axes = plt.subplots(3, 2, figsize=(16, 15), facecolor="white")

axes[0, 0].set_title(
    rf"Carbon price = {format_thousands(MAX_CP)} AUD/tCO$_2$e",
    pad=4,
)
axes[0, 1].set_title(
    f"Biodiversity price = {format_thousands(MAX_BP)} AUD/ha",
    pad=4,
)

row_specs = [
    ("Agricultural land-use", "ag_cp", "ag_bp"),
    ("Ag management", "am_cp", "am_bp"),
    ("Non-ag", "non_ag_cp", "non_ag_bp"),
]

for row_idx, (area_type, key_cp, key_bp) in enumerate(row_specs):
    config = MAP_CONFIG[area_type]
    cat_cp = MAPS[key_cp]
    cat_bp = MAPS[key_bp]

    for col_idx, cat_map in enumerate((cat_cp, cat_bp)):
        ax = axes[row_idx, col_idx]
        rgba = category_map_to_rgba(cat_map, config)
        ax.imshow(
            rgba,
            extent=[EXTENT[0], EXTENT[1], EXTENT[2], EXTENT[3]],
            origin="upper",
            interpolation="nearest",
            zorder=1,
            aspect="auto",
        )
        if gdf_states is not None:
            gdf_states.plot(ax=ax, edgecolor="#555555", linewidth=0.4, facecolor="none", zorder=3)
        style_map_ax(ax)

    axes[row_idx, 0].set_ylabel(area_type, rotation=90, labelpad=18)

    legend_handles = get_row_legend_handles(cat_cp, cat_bp, config)
    axes[row_idx, 1].legend(
        handles=legend_handles,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        frameon=False,
        borderaxespad=0.0,
        handlelength=1.0,
        handleheight=1.0,
    )

plt.tight_layout()
plt.subplots_adjust(right=0.77, hspace=0.08, wspace=0.06)

out_path = OUT_DIR / f"5_Area_Map_vs_Price_{YEAR}.png"
fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
plt.close()
print(f"Saved: {out_path}")
