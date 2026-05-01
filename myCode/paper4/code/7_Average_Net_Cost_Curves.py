# ==============================================================================
# Figure 7: Average net cost curves — broken-axis 3×3 equal-grid layout
#   Panel A (top):    Carbon price
#   Panel B (bottom): Biodiversity price
#
#   Uses absolute values from the max-price run (not deltas vs zero-price run).
#   Policy payment is removed to obtain true underlying net cost.
#   Both panels use a 3×3 equal-size grid of subplots with broken x and y axes.
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
    apply_paper4_color_overrides_to_style_df,
    build_run_map,
    format_thousands,
    style_box_axis,
)

YEAR     = 2025
N_SEGS   = 3   # Equal 3×3 grid of subplots per panel
BASE_DIR = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Manual axis segment ranges  (set to None to use auto-detection)
#
# X ranges: list of (xmin, xmax) tuples, left → right, units on x-axis label
# Y ranges: list of (ymin, ymax) tuples, bottom → top (smallest y first)
# Number of tuples must equal N_SEGS (3) when not None.
# ---------------------------------------------------------------------------

# Carbon panel  (x unit: Mt CO₂e yr⁻¹,  y unit: AUD/tCO₂e yr⁻¹)
CARBON_X_RANGES = [(0,1),(1,150),(150,165)]   # e.g. [(0, 20), (20, 110), (145, 165)]
CARBON_Y_RANGES = [(-600,0),(0,500),(500,2100)]   # e.g. [(800, 2100), (-50, 700), (-420, -250)]

# Biodiversity panel  (x unit: Mha yr⁻¹,  y unit: AUD/ha yr⁻¹)
BIO_X_RANGES    = None   # e.g. [(0, 20), (20, 80), (80, 115)]
BIO_Y_RANGES    = None   # e.g. [(700, 1100), (-100, 430), (-420, -250)]
DRAW_ALL_TOOLS_DIR = BASE_DIR.parents[1] / "draw_all" / "code" / "tools"
COLOR_FILE  = DRAW_ALL_TOOLS_DIR / "land use colors.xlsx"
GROUP_FILE  = DRAW_ALL_TOOLS_DIR / "land use group.xlsx"
CACHE_PATH  = DATA_DIR / f"7_Average_Net_Cost_raw_data_{YEAR}_abs.xlsx"

FS = 10
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial"],
    "font.size": FS,
    "axes.titlesize": FS,
    "axes.labelsize": FS,
    "xtick.labelsize": FS - 1,
    "ytick.labelsize": FS - 1,
    "legend.fontsize": FS - 1,
    "mathtext.fontset": "stixsans",
})

NON_AG_EXCLUDE = {"agriculturallanduse", "otherlanduse"}
MIN_WIDTH = 1e-3


# ---------------------------------------------------------------------------
# Style helpers
# ---------------------------------------------------------------------------

def normalize_name(v):
    return re.sub(r"[\s\-]+", "", str(v).strip().lower())


def load_style_table(sheet_name):
    df = pd.read_excel(COLOR_FILE, sheet_name=sheet_name)
    df = apply_paper4_color_overrides_to_style_df(df)
    label_col = "desc_new" if "desc_new" in df.columns else "desc"
    order, color_map, label_map = [], {}, {}
    for _, row in df.iterrows():
        lbl = row[label_col]
        order.append(lbl)
        color_map[lbl] = row["color"]
        label_map[normalize_name(row["desc"])] = lbl
    return order, color_map, label_map


AG_ORDER, AG_COLOR_MAP, _                        = load_style_table("ag_group")
AM_ORDER, AM_COLOR_MAP, AM_LABEL_MAP             = load_style_table("am")
NON_AG_ORDER, NON_AG_COLOR_MAP, NON_AG_LABEL_MAP = load_style_table("non_ag")
LU_ORDER, LU_COLOR_MAP, LU_LABEL_MAP             = load_style_table("lu")
TRANSITION_LABEL = LU_LABEL_MAP.get(normalize_name("Transition"), "Transition")

group_df = pd.read_excel(GROUP_FILE)
LU_TO_AG_GROUP = {
    normalize_name(row["desc"]): row["ag_group"]
    for _, row in group_df.iterrows()
    if pd.notna(row.get("desc")) and pd.notna(row.get("ag_group"))
}

PANEL_CONFIG = {
    "Agricultural land-use": {"order": AG_ORDER,     "color_map": AG_COLOR_MAP},
    "Ag management":          {"order": AM_ORDER,     "color_map": AM_COLOR_MAP},
    "Non-ag":                 {"order": NON_AG_ORDER, "color_map": NON_AG_COLOR_MAP},
}
CURVE_COLOR_MAP = {
    **AG_COLOR_MAP, **AM_COLOR_MAP, **NON_AG_COLOR_MAP,
    TRANSITION_LABEL: LU_COLOR_MAP.get(TRANSITION_LABEL, "#D2E0FB"),
}


# ---------------------------------------------------------------------------
# NetCDF / zip helpers
# ---------------------------------------------------------------------------

def open_metric_da(zip_path, file_name):
    with zipfile.ZipFile(zip_path) as archive:
        matches = [n for n in archive.namelist() if n.endswith(file_name)]
        if not matches:
            return None
        with archive.open(matches[0]) as f:
            ds = xr.open_dataset(io.BytesIO(f.read()), engine="h5netcdf")
    try:
        if "layer" in ds.dims and "compress" in ds["layer"].attrs:
            ds = cfxr.decode_compress_to_multi_index(ds, "layer")
        da = next(iter(ds.data_vars.values()))
        return da.load()
    finally:
        ds.close()


def sum_with_total_coords(da, **selectors):
    sub = da.sel(selectors)
    for cname in list(sub.coords):
        if cname in {"cell", "layer"} or cname in selectors:
            continue
        try:
            if "ALL" in sub.coords[cname].values:
                sub = sub.sel({cname: "ALL"})
        except TypeError:
            pass
    return float(sub.sum())


def read_ag_group_summary(zip_path, fname):
    da = open_metric_da(zip_path, fname)
    if da is None:
        return {}
    result = {}
    for lu in pd.unique(da.coords["lu"].values):
        if lu == "ALL":
            continue
        v = sum_with_total_coords(da, lu=lu)
        if np.isclose(v, 0.0):
            continue
        g = LU_TO_AG_GROUP.get(normalize_name(lu), "Other land")
        result[g] = result.get(g, 0.0) + v
    return result


def read_am_summary(zip_path, fname):
    da = open_metric_da(zip_path, fname)
    if da is None:
        return {}
    if "am" not in da.coords:
        return {}
    result = {}
    for am in pd.unique(da.coords["am"].values):
        if am == "ALL":
            continue
        v = sum_with_total_coords(da, am=am)
        if np.isclose(v, 0.0):
            continue
        label = AM_LABEL_MAP.get(normalize_name(am), am)
        result[label] = result.get(label, 0.0) + v
    return result


def read_non_ag_summary(zip_path, fname):
    da = open_metric_da(zip_path, fname)
    if da is None:
        return {}
    result = {}
    for lu in pd.unique(da.coords["lu"].values):
        if lu == "ALL" or normalize_name(lu) in NON_AG_EXCLUDE:
            continue
        v = sum_with_total_coords(da, lu=lu)
        if np.isclose(v, 0.0):
            continue
        label = NON_AG_LABEL_MAP.get(normalize_name(lu), lu)
        result[label] = result.get(label, 0.0) + v
    return result


def read_transition_ghg(zip_path, year):
    da = open_metric_da(zip_path, f"xr_transition_GHG_{year}.nc")
    return 0.0 if da is None else sum_with_total_coords(da)


def get_profit(zip_path, year):
    return {
        "Agricultural land-use": read_ag_group_summary(zip_path, f"xr_economics_ag_profit_{year}.nc"),
        "Ag management":          read_am_summary(zip_path,        f"xr_economics_am_profit_{year}.nc"),
        "Non-ag":                 read_non_ag_summary(zip_path,    f"xr_economics_non_ag_profit_{year}.nc"),
    }


def get_ghg(zip_path, year):
    ag = read_ag_group_summary(zip_path, f"xr_GHG_ag_{year}.nc")
    t  = read_transition_ghg(zip_path, year)
    if not np.isclose(t, 0.0):
        ag[TRANSITION_LABEL] = ag.get(TRANSITION_LABEL, 0.0) + t
    return {
        "Agricultural land-use": ag,
        "Ag management":          read_am_summary(zip_path,     f"xr_GHG_ag_management_{year}.nc"),
        "Non-ag":                 read_non_ag_summary(zip_path, f"xr_GHG_non_ag_{year}.nc"),
    }


def get_bio(zip_path, year):
    return {
        "Agricultural land-use": read_ag_group_summary(zip_path, f"xr_biodiversity_overall_priority_ag_{year}.nc"),
        "Ag management":          read_am_summary(zip_path,        f"xr_biodiversity_overall_priority_ag_management_{year}.nc"),
        "Non-ag":                 read_non_ag_summary(zip_path,    f"xr_biodiversity_overall_priority_non_ag_{year}.nc"),
    }


def get_category_order(area_type, cats):
    base = PANEL_CONFIG[area_type]["order"]
    return [c for c in base if c in cats] + [c for c in cats if c not in base]


# ---------------------------------------------------------------------------
# Data collection — absolute values from max-price run only
# ---------------------------------------------------------------------------

def collect_carbon_rows(run_map, cp_max):
    """
    Use absolute values from the max-price carbon run.
    Subtract the carbon policy payment to obtain underlying net cost.
    Only include categories with positive GHG abatement (sequestration).
    """
    m   = run_map[(cp_max, 0.0)]
    mp  = get_profit(m, YEAR)
    mg  = get_ghg(m, YEAR)
    rows = []
    for at in ("Agricultural land-use", "Ag management", "Non-ag"):
        cats = list(dict.fromkeys(list(mp[at]) + list(mg[at])))
        for cat in get_category_order(at, cats):
            p    = mp[at].get(cat, 0.0) / 1e9    # absolute profit [BAUD]
            g    = -mg[at].get(cat, 0.0) / 1e6   # abatement [Mt CO2e] (+ = seq)
            excl = p - cp_max * g / 1e3           # profit excl. carbon payment [BAUD]
            inc  = g > MIN_WIDTH and cat != TRANSITION_LABEL
            rows.append({
                "Panel": "Carbon",
                "TargetPrice": cp_max,
                "AreaType": at,
                "Category": cat,
                "ContributionWidth": g,
                "ProfitExclPolicy_BAUD": excl,
                "AverageNetCost": -(excl * 1e3) / g if inc else np.nan,
                "IncludeInCurve": inc,
            })
    return rows


def collect_bio_rows(run_map, bp_max):
    """
    Use absolute values from the max-price biodiversity run.
    Subtract the biodiversity policy payment to obtain underlying net cost.
    Only include categories with positive biodiversity contribution.
    """
    m   = run_map[(0.0, bp_max)]
    mp  = get_profit(m, YEAR)
    mb  = get_bio(m, YEAR)
    rows = []
    for at in ("Agricultural land-use", "Ag management", "Non-ag"):
        cats = list(dict.fromkeys(list(mp[at]) + list(mb[at])))
        for cat in get_category_order(at, cats):
            p    = mp[at].get(cat, 0.0) / 1e9    # absolute profit [BAUD]
            b    = mb[at].get(cat, 0.0) / 1e6    # biodiversity score [Mha yr]
            excl = p - bp_max * b / 1e3           # profit excl. bio payment [BAUD]
            inc  = b > MIN_WIDTH
            rows.append({
                "Panel": "Biodiversity",
                "TargetPrice": bp_max,
                "AreaType": at,
                "Category": cat,
                "ContributionWidth": b,
                "ProfitExclPolicy_BAUD": excl,
                "AverageNetCost": -(excl * 1e3) / b if inc else np.nan,
                "IncludeInCurve": inc,
            })
    return rows


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

def load_cache():
    if not CACHE_PATH.is_file():
        return None, None
    try:
        print(f"Loading cached data from {CACHE_PATH}")
        dfc = pd.read_excel(CACHE_PATH, sheet_name="CurveData")
        dfm = pd.read_excel(CACHE_PATH, sheet_name="Metadata")
    except ValueError:
        print("Cache schema outdated; rebuilding.")
        return None, None
    required = {"Panel", "AreaType", "Category", "ContributionWidth",
                 "ProfitExclPolicy_BAUD", "AverageNetCost", "IncludeInCurve"}
    if not required.issubset(dfc.columns):
        print("Cache schema outdated; rebuilding.")
        return None, None
    return dfc, dfm


def collect_and_cache():
    run_map, cp_vals, bp_vals = build_run_map()
    cp_max, bp_max = max(cp_vals), max(bp_vals)
    dfc = pd.DataFrame(
        collect_carbon_rows(run_map, cp_max) + collect_bio_rows(run_map, bp_max)
    ).sort_values(["Panel", "AreaType", "Category"]).reset_index(drop=True)
    dfm = pd.DataFrame([{"Panel": "Carbon", "TargetPrice": cp_max},
                         {"Panel": "Biodiversity", "TargetPrice": bp_max}])
    with pd.ExcelWriter(CACHE_PATH, engine="openpyxl") as w:
        dfc.to_excel(w, sheet_name="CurveData", index=False)
        dfc[dfc["Panel"] == "Carbon"].to_excel(w,       sheet_name="Carbon",       index=False)
        dfc[dfc["Panel"] == "Biodiversity"].to_excel(w, sheet_name="Biodiversity", index=False)
        dfm.to_excel(w, sheet_name="Metadata", index=False)
    print(f"Cache saved: {CACHE_PATH}")
    return dfc, dfm


# ---------------------------------------------------------------------------
# Curve preparation
# ---------------------------------------------------------------------------

def prepare_curve(dfc, panel_name, min_contrib_frac=0.005):
    df = dfc[(dfc["Panel"] == panel_name) & (dfc["IncludeInCurve"])].copy()
    if df.empty:
        return df
    total = df["ContributionWidth"].sum()
    if total > 0:
        df = df[df["ContributionWidth"] >= total * min_contrib_frac].copy()
    df = df.sort_values(["AverageNetCost", "ContributionWidth", "Category"],
                        ascending=[True, False, True]).reset_index(drop=True)
    df["x_left"]  = df["ContributionWidth"].cumsum().shift(fill_value=0.0)
    df["x_right"] = df["x_left"] + df["ContributionWidth"]
    return df


VALUE_FMT = {
    "Carbon":       ticker.FuncFormatter(lambda v, _: format_thousands(v)),
    "Biodiversity": ticker.FuncFormatter(lambda v, _: format_thousands(v)),
}


# ---------------------------------------------------------------------------
# Axis-segment detection — always returns exactly N_SEGS segments
# ---------------------------------------------------------------------------

def _gap_segs(values, n, pad_frac=0.06):
    """
    Split `values` into `n` segments by finding the n-1 largest gaps.
    Returns segments in ascending order (bottom to top for y, left to right for x).
    """
    vals = np.sort(values[~np.isnan(values)])
    if len(vals) == 0:
        return [(-1.0, 1.0)] * n
    total_rng = float(vals[-1] - vals[0])
    pad = max(total_rng * pad_frac, abs(vals[0]) * 0.01 + 1.0)

    if total_rng < 1e-10 or len(vals) < n:
        lo, hi = vals[0] - pad, vals[-1] + pad
        step = (hi - lo) / n
        return [(lo + i * step, lo + (i + 1) * step) for i in range(n)]

    gaps = np.diff(vals)
    n_breaks = n - 1
    top_idx = sorted(np.argsort(gaps)[::-1][:n_breaks].tolist())

    # Boundaries between segments (midpoints of the largest gaps)
    bounds = [vals[i] + gaps[i] / 2.0 for i in top_idx]
    edges = [-np.inf] + bounds + [np.inf]

    segs = []
    for i in range(n):
        mask = (vals >= edges[i]) & (vals < edges[i + 1])
        seg_vals = vals[mask]
        if len(seg_vals) == 0:
            # Edge case: no data in this segment — use boundary midpoints
            lo_b = bounds[i - 1] if i > 0 else vals[0] - pad
            hi_b = bounds[i]     if i < n - 1 else vals[-1] + pad
            segs.append((lo_b, hi_b))
        else:
            segs.append((float(seg_vals[0]) - pad, float(seg_vals[-1]) + pad))
    return segs


def y_segs(df, n=N_SEGS):
    """N y-segments, top (high y) first."""
    heights = df["AverageNetCost"].dropna().values
    segs = _gap_segs(heights, n)
    return list(reversed(segs))   # highest y at index 0


def x_segs(df, n=N_SEGS, pad_frac=0.02):
    """N x-segments, left (low x) first. Equal division over cumulative x range."""
    x_max = float(df["x_right"].max()) * (1.0 + pad_frac)
    step  = x_max / n
    return [(i * step, (i + 1) * step) for i in range(n)]


# ---------------------------------------------------------------------------
# Break marks
# ---------------------------------------------------------------------------

def add_y_break_marks(ax_u, ax_l, d=0.010):
    kw = dict(color="black", clip_on=False, linewidth=0.9)
    for ax, ys in [(ax_u, (-d, +d)), (ax_l, (1 - d, 1 + d))]:
        ax.plot((-d, +d),   ys, transform=ax.transAxes, **kw)
        ax.plot((1 - d, 1 + d), ys, transform=ax.transAxes, **kw)


def add_x_break_marks(ax_l, ax_r, d=0.010):
    kw = dict(color="black", clip_on=False, linewidth=0.9)
    for ax, xs in [(ax_l, (1 - d, 1 + d)), (ax_r, (-d, +d))]:
        ax.plot(xs, (-d, +d),   transform=ax.transAxes, **kw)
        ax.plot(xs, (1 - d, 1 + d), transform=ax.transAxes, **kw)


# ---------------------------------------------------------------------------
# Legend
# ---------------------------------------------------------------------------

def get_legend_handles(df):
    handles, seen = [], set()
    for cat in df["Category"]:
        if cat in seen:
            continue
        handles.append(mpatches.Patch(
            facecolor=CURVE_COLOR_MAP.get(cat, "#888888"),
            edgecolor="none", label=cat))
        seen.add(cat)
    return handles


# ---------------------------------------------------------------------------
# Main draw function — 3×3 equal-size grid with broken axes
# ---------------------------------------------------------------------------

def draw_curve(fig, outer_spec, df, panel_name, title, xlabel, ylabel,
               x_ranges=None, y_ranges=None):
    """
    x_ranges : list of (xmin, xmax), left → right.  None = auto-detect.
    y_ranges : list of (ymin, ymax), top → bottom (highest y first).  None = auto-detect.
    """
    if df.empty:
        ax = fig.add_subplot(outer_spec)
        ax.text(0.5, 0.5, "No data", ha="center", va="center",
                transform=ax.transAxes)
        style_box_axis(ax, linewidth=0.8)
        return []

    y_segs_ = list(reversed(y_ranges)) if y_ranges is not None else y_segs(df)
    x_segs_ = x_ranges if x_ranges is not None else x_segs(df)

    n_y, n_x = len(y_segs_), len(x_segs_)

    # Equal-size subplots — NO height_ratios or width_ratios
    inner = outer_spec.subgridspec(n_y, n_x, hspace=0.08, wspace=0.08)
    axes  = [[fig.add_subplot(inner[iy, ix])
              for ix in range(n_x)] for iy in range(n_y)]

    # Draw every bar in every cell; matplotlib clips to the cell's xlim/ylim
    for iy in range(n_y):
        for ix in range(n_x):
            ax = axes[iy][ix]
            for _, row in df.iterrows():
                ax.bar(
                    row["x_left"], row["AverageNetCost"],
                    width=row["ContributionWidth"],
                    align="edge",
                    color=CURVE_COLOR_MAP.get(row["Category"], "#888888"),
                    edgecolor="white", linewidth=0.5,
                )
            ax.set_xlim(*x_segs_[ix])
            ax.set_ylim(*y_segs_[iy])
            if y_segs_[iy][0] < 0.0 < y_segs_[iy][1]:
                ax.axhline(0.0, color="#444444", linewidth=0.8)
            ax.xaxis.set_major_formatter(VALUE_FMT[panel_name])
            ax.xaxis.set_major_locator(ticker.MaxNLocator(3, integer=False))
            ax.yaxis.set_major_locator(ticker.MaxNLocator(3, integer=False))
            style_box_axis(ax, linewidth=0.8)

    # ── Broken y-axis: hide inner horizontal spines + x-tick labels ──
    for iy in range(n_y - 1):
        for ix in range(n_x):
            axes[iy][ix].spines["bottom"].set_visible(False)
            axes[iy][ix].tick_params(axis="x", labelbottom=False, bottom=False)
    for iy in range(1, n_y):
        for ix in range(n_x):
            axes[iy][ix].spines["top"].set_visible(False)

    # ── Broken x-axis: hide inner vertical spines + y-tick labels ──
    for ix in range(n_x - 1):
        for iy in range(n_y):
            axes[iy][ix].spines["right"].set_visible(False)
            axes[iy][ix].tick_params(axis="y", right=False)
    for ix in range(1, n_x):
        for iy in range(n_y):
            axes[iy][ix].spines["left"].set_visible(False)
            axes[iy][ix].tick_params(axis="y", labelleft=False, left=False)

    # ── Break marks ──
    for iy in range(n_y - 1):
        for ix in range(n_x):
            add_y_break_marks(axes[iy][ix], axes[iy + 1][ix])
    for ix in range(n_x - 1):
        for iy in range(n_y):
            add_x_break_marks(axes[iy][ix], axes[iy][ix + 1])

    # ── Labels: title top-centre, x-label bottom-centre, y-label mid-left ──
    axes[0][n_x // 2].set_title(title, pad=6)
    axes[-1][n_x // 2].set_xlabel(xlabel)
    axes[n_y // 2][0].set_ylabel(ylabel)

    return get_legend_handles(df)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

dfc, dfm = load_cache()
if dfc is None or dfm is None:
    dfc, dfm = collect_and_cache()

df_carbon = prepare_curve(dfc, "Carbon")
df_bio    = prepare_curve(dfc, "Biodiversity")

cp = float(dfc.loc[dfc["Panel"] == "Carbon",       "TargetPrice"].iloc[0])
bp = float(dfc.loc[dfc["Panel"] == "Biodiversity", "TargetPrice"].iloc[0])

fig   = plt.figure(figsize=(14, 16))
outer = fig.add_gridspec(3, 1, hspace=0.45, height_ratios=[10, 10, 1.2])

handles_c = draw_curve(
    fig, outer[0, 0], df_carbon, "Carbon",
    rf"Carbon average net cost curve  (cp = {format_thousands(cp)} AU\$/tCO$_2$e yr$^{{-1}}$)",
    r"GHG abatement at max price (Mt CO$_2$e yr$^{-1}$)",
    r"Emission reduction cost (AU\$/tCO$_2$e yr$^{-1}$)",
    x_ranges=CARBON_X_RANGES,
    y_ranges=CARBON_Y_RANGES,
)

handles_b = draw_curve(
    fig, outer[1, 0], df_bio, "Biodiversity",
    rf"Biodiversity average net cost curve  (bp = {format_thousands(bp)} AU\$/ha yr$^{{-1}}$)",
    r"Biodiversity contribution at max price (Mha yr$^{-1}$)",
    r"Biodiversity restoration cost (AU\$/ha yr$^{-1}$)",
    x_ranges=BIO_X_RANGES,
    y_ranges=BIO_Y_RANGES,
)

all_handles, seen = [], set()
for h in handles_c + handles_b:
    if h.get_label() not in seen:
        all_handles.append(h)
        seen.add(h.get_label())

if all_handles:
    leg_ax = fig.add_subplot(outer[2, 0])
    leg_ax.set_axis_off()
    leg_ax.legend(
        handles=all_handles,
        loc="center",
        ncol=min(len(all_handles), 5),
        frameon=False,
        fontsize=FS - 1,
        handlelength=1.2,
        handleheight=1.0,
        columnspacing=1.0,
    )

out_path = OUT_DIR / f"7_Average_Net_Cost_Curves_{YEAR}.png"
fig.savefig(out_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"Saved: {out_path}")
