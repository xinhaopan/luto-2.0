# ==============================================================================
# Figure 06: What does each Budget level buy?
#
# X-axis: Budget (income transfer) = price × additional quantity above baseline
#         Carbon pathway:  Budget = CarbonPrice × ΔGHG (Mt→B$: divide by 1000)
#         Bio pathway:     Budget = BioPrice    × ΔBio (Mha→B$: divide by 1000)
#
# Layout (3 rows × 2 columns):
#   Row 0: Budget → total ΔGHG (left) and total ΔBio (right)
#           Both pathways overlaid; shows the aggregate response curve.
#   Row 1: Budget → contribution stacked by area type (Ag / AM / Non-ag)
#           Left = carbon pathway, Right = biodiversity pathway.
#           GHG contribution on carbon panel, Bio contribution on bio panel.
#   Row 2: Budget → contribution stacked by land-use category (top categories)
#           Same left/right split as row 1.
# ==============================================================================

import os
import sys
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tools.price_slice_utils import (
    DATA_DIR,
    OUT_DIR,
    apply_compact_ticks,
    apply_paper4_color_overrides_to_style_df,
    stacked_area_pos_neg,
    style_box_axis,
)


YEAR = 2025
BASE_DIR = Path(__file__).resolve().parent
DRAW_ALL_TOOLS_DIR = BASE_DIR.parents[1] / "draw_all" / "code" / "tools"
COLOR_FILE = DRAW_ALL_TOOLS_DIR / "land use colors.xlsx"
CONTRIBUTION_CACHE = DATA_DIR / f"04_Contribution_Delta_vs_Zero_raw_data_{YEAR}.xlsx"
RAW_DATA_PATH = DATA_DIR / f"06_Bio_GHG_Budget_raw_data_{YEAR}.xlsx"

GHG_METRIC = "GHGAbatementChange_vs_ZeroPrice_MtCO2e"
BIO_METRIC  = "BiodiversityContributionChange_vs_ZeroPrice_MhaYr"

COLOR_CARBON = "#fbbc45"
COLOR_BIO    = "#2ca25f"
FS = 11

AREA_TYPE_COLORS = {
    "Agricultural land-use": "#c49a67",
    "Ag management":         "#72c15a",
    "Non-ag":                "#2166ac",
}
AREA_TYPE_LABELS = {
    "Agricultural land-use": "Agricultural land-use",
    "Ag management":         "Agricultural management",
    "Non-ag":                "Non-agricultural land-use",
}
AREA_ORDER = ["Agricultural land-use", "Ag management", "Non-ag"]

LEGEND_LABELS = {
    "Agricultural technology (energy)":              "AgTech energy",
    "Agricultural technology (fertiliser)":          "AgTech fertiliser",
    "Biochar":                                       "Biochar",
    "Early dry-season savanna burning":              "Savanna burning",
    "Human-Induced Regeneration (beef)":             "HIR beef",
    "Human-Induced Regeneration (sheep)":            "HIR sheep",
    "Crops":                                         "Crops",
    "Modified livestock":                            "Modified livestock",
    "Natural Livestock":                             "Natural livestock",
    "Unallocated - modified land":                   "Unallocated modified",
    "Unallocated - natural land":                    "Unallocated natural",
    "Agroforestry (mixed species + beef)":           "Agroforestry beef",
    "Agroforestry (mixed species + sheep)":          "Agroforestry sheep",
    "Carbon plantings (monoculture)":                "Carbon plantings",
    "Destocked (natural land)":                      "Destocked natural",
    "Environmental plantings (mixed local native species)": "Environmental plantings",
    "Farm forestry (hardwood timber + beef)":        "Farm forestry beef",
    "Farm forestry (hardwood timber + sheep)":       "Farm forestry sheep",
    "Riparian buffer restoration (mixed species)":   "Riparian restoration",
    "Methane reduction (livestock)":                 "Methane reduction",
    "Regenerative agriculture (livestock)":          "Regenerative agriculture",
}

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


# ── Style helpers ──────────────────────────────────────────────────────────────

def load_style_table(sheet_name):
    df = pd.read_excel(COLOR_FILE, sheet_name=sheet_name)
    df = apply_paper4_color_overrides_to_style_df(df)
    label_col = "desc_new" if "desc_new" in df.columns else "desc"
    order, color_map = [], {}
    for _, row in df.iterrows():
        label = row[label_col]
        order.append(label)
        color_map[label] = row["color"]
    return order, color_map


def build_category_styles():
    ag_order, ag_colors = load_style_table("ag_group")
    am_order, am_colors = load_style_table("am")
    non_ag_order, non_ag_colors = load_style_table("non_ag")
    skip = {"Other land", "No agricultural management", "Other land-use",
            "Agricultural land-use", "Livestock"}
    order, color_map = [], {}
    for o, c in [(ag_order, ag_colors), (am_order, am_colors), (non_ag_order, non_ag_colors)]:
        for cat in o:
            if cat in skip or cat in order:
                continue
            order.append(cat)
            color_map[cat] = c.get(cat, "#888888")
    return order, color_map


CATEGORY_ORDER, CATEGORY_COLOR_MAP = build_category_styles()


# ── Data loading ───────────────────────────────────────────────────────────────

def load_data():
    if not CONTRIBUTION_CACHE.is_file():
        raise FileNotFoundError(
            f"Contribution cache not found: {CONTRIBUTION_CACHE}\n"
            "Run 04_Contribution_Delta_vs_Zero.py first."
        )
    df = pd.read_excel(CONTRIBUTION_CACHE, sheet_name="ContributionLong")

    # Compute Budget (Billion AUD) = price × additional_quantity / 1000
    # Carbon: price (AUD/tCO2e) × ΔGHG (MtCO2e) × 1e6 / 1e9 = price × ΔGHG / 1000
    # Bio:    price (AUD/ha)    × ΔBio  (Mha)    × 1e6 / 1e9 = price × ΔBio  / 1000
    totals = (
        df.groupby(["PriceType", "Price", "MetricType"], as_index=False)["ContributionValue"]
        .sum()
    )
    carbon_ghg = totals[
        (totals.PriceType == "CarbonPrice") & (totals.MetricType == GHG_METRIC)
    ].rename(columns={"ContributionValue": "total_ghg"})
    bio_bio = totals[
        (totals.PriceType == "BioPrice") & (totals.MetricType == BIO_METRIC)
    ].rename(columns={"ContributionValue": "total_bio"})

    carbon_ghg["Budget_BillionAUD"] = carbon_ghg["Price"] * carbon_ghg["total_ghg"] / 1000
    bio_bio["Budget_BillionAUD"]    = bio_bio["Price"]    * bio_bio["total_bio"]     / 1000

    budget_map = pd.concat([
        carbon_ghg[["PriceType", "Price", "Budget_BillionAUD"]],
        bio_bio[["PriceType", "Price", "Budget_BillionAUD"]],
    ], ignore_index=True)
    # Zero-price baseline has zero budget
    for pt in ["CarbonPrice", "BioPrice"]:
        if not ((budget_map.PriceType == pt) & np.isclose(budget_map.Price, 0)).any():
            budget_map = pd.concat([
                budget_map,
                pd.DataFrame([{"PriceType": pt, "Price": 0.0, "Budget_BillionAUD": 0.0}])
            ], ignore_index=True)

    df = df.merge(budget_map, on=["PriceType", "Price"], how="left")
    # Drop zero-budget rows for area-type / category panels (keep only where budget > 0)
    return df, carbon_ghg, bio_bio


# ── Axis helpers ───────────────────────────────────────────────────────────────

def _budget_xlabel():
    return r"Budget (Billion AU\$ yr$^{-1}$)  =  price × additional quantity"


def _pad_lim(lo, hi, frac=0.08, min_pad=0.0):
    pad = max((hi - lo) * frac, min_pad)
    return lo - pad, hi + pad


def _x_lim(budget_values):
    vals = np.asarray(budget_values, dtype=float)
    vals = vals[np.isfinite(vals)]
    return _pad_lim(0.0, float(vals.max()) if len(vals) else 1.0, frac=0.06, min_pad=0.5)


# ── Row 0: aggregate curves ────────────────────────────────────────────────────

def plot_row0(ax_ghg, ax_bio, carbon_ghg, bio_bio):
    """Budget → total ΔGHG (left) and total ΔBio (right); both pathways on each."""
    # left panel: both pathways → GHG
    # Only carbon pathway has meaningful GHG response; bio pathway shown for reference
    # Each pathway shows its own primary metric
    for ax, df_sub, y_col, y_label, title, color, label in [
        (ax_ghg, carbon_ghg, "total_ghg",
         r"GHG abatement change (Mt CO$_2$e yr$^{-1}$)",
         "Carbon price pathway: GHG abatement vs Budget",
         COLOR_CARBON, "Carbon price"),
        (ax_bio, bio_bio, "total_bio",
         r"Biodiversity contribution change (Mha yr$^{-1}$)",
         "Biodiversity price pathway: bio contribution vs Budget",
         COLOR_BIO, "Biodiversity price"),
    ]:
        sub = df_sub.sort_values("Budget_BillionAUD")
        # insert (0,0) origin if missing
        if not np.isclose(sub["Budget_BillionAUD"].iloc[0], 0):
            sub = pd.concat([
                pd.DataFrame([{"Budget_BillionAUD": 0.0, y_col: 0.0}]), sub
            ], ignore_index=True).sort_values("Budget_BillionAUD")

        ax.plot(
            sub["Budget_BillionAUD"], sub[y_col],
            color=color, linewidth=2.0,
            marker="o", markersize=5.5, markeredgewidth=0,
            label=label, zorder=6,
        )
        ax.set_xlabel(_budget_xlabel())
        ax.set_ylabel(y_label)
        ax.set_title(title, pad=7)
        ax.set_xlim(*_x_lim(sub["Budget_BillionAUD"]))
        ax.set_ylim(0, float(sub[y_col].max()) * 1.12 + 0.2)
        ax.xaxis.set_major_formatter(ticker.StrMethodFormatter("{x:,.0f}"))
        apply_compact_ticks(ax, x_nbins=6, y_nbins=6)
        style_box_axis(ax)
        ax.legend(loc="upper left", frameon=True, framealpha=0.9)


# ── Row 1: stacked area by area type ──────────────────────────────────────────

def plot_row1(ax, df, price_type, metric_type, ylabel, title):
    """Budget → stacked contribution by area type (Ag / AM / Non-ag)."""
    sub = df[
        (df["PriceType"] == price_type) & (df["MetricType"] == metric_type)
    ].copy()
    if sub.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        style_box_axis(ax)
        return

    pivot = (
        sub.groupby(["Budget_BillionAUD", "AreaType"])["ContributionValue"]
        .sum()
        .reset_index()
        .pivot(index="Budget_BillionAUD", columns="AreaType", values="ContributionValue")
        .fillna(0.0)
        .sort_index()
    )
    # Ensure zero row
    if not np.isclose(pivot.index[0], 0.0):
        zero_row = pd.DataFrame([[0.0] * len(pivot.columns)],
                                 columns=pivot.columns, index=[0.0])
        pivot = pd.concat([zero_row, pivot]).sort_index()

    # Only keep area types in AREA_ORDER that are present
    cols = [c for c in AREA_ORDER if c in pivot.columns]
    pivot = pivot[cols]

    color_map = {c: AREA_TYPE_COLORS.get(c, "#888888") for c in cols}
    stacked_area_pos_neg(ax, pivot, color_map, alpha=0.82)

    ax.set_xlabel(_budget_xlabel())
    ax.set_ylabel(ylabel)
    ax.set_title(title, pad=7)
    ax.set_xlim(*_x_lim(pivot.index))
    ax.axhline(0, color="#404040", linewidth=0.8, zorder=8)
    ax.xaxis.set_major_formatter(ticker.StrMethodFormatter("{x:,.0f}"))
    apply_compact_ticks(ax, x_nbins=6, y_nbins=5)
    style_box_axis(ax)

    handles = [
        mpatches.Patch(facecolor=AREA_TYPE_COLORS.get(c, "#888888"),
                       label=AREA_TYPE_LABELS.get(c, c), edgecolor="none")
        for c in cols
    ]
    ax.legend(handles=handles, loc="upper left", frameon=True, framealpha=0.9,
              fontsize=FS - 1)


# ── Row 2: stacked area by land-use category ──────────────────────────────────

def plot_row2(ax, df, price_type, metric_type, ylabel, title, top_n=10):
    """Budget → stacked contribution by land-use category (top N categories)."""
    sub = df[
        (df["PriceType"] == price_type) & (df["MetricType"] == metric_type)
    ].copy()
    if sub.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        style_box_axis(ax)
        return []

    # Pick top_n categories by max absolute contribution across all budget levels
    cat_max = (
        sub.groupby("Category")["ContributionValue"]
        .apply(lambda x: x.abs().max())
        .sort_values(ascending=False)
    )
    top_cats = cat_max.head(top_n).index.tolist()
    # Sort by CATEGORY_ORDER for consistent color ordering
    ordered_cats = [c for c in CATEGORY_ORDER if c in top_cats]
    ordered_cats += [c for c in top_cats if c not in ordered_cats]

    sub_top = sub[sub["Category"].isin(ordered_cats)]
    pivot = (
        sub_top.groupby(["Budget_BillionAUD", "Category"])["ContributionValue"]
        .sum()
        .reset_index()
        .pivot(index="Budget_BillionAUD", columns="Category", values="ContributionValue")
        .fillna(0.0)
        .sort_index()
    )
    if not np.isclose(pivot.index[0], 0.0):
        zero_row = pd.DataFrame([[0.0] * len(pivot.columns)],
                                 columns=pivot.columns, index=[0.0])
        pivot = pd.concat([zero_row, pivot]).sort_index()

    # Reorder columns
    cols = [c for c in ordered_cats if c in pivot.columns]
    pivot = pivot[cols]
    color_map = {c: CATEGORY_COLOR_MAP.get(c, "#888888") for c in cols}

    stacked_area_pos_neg(ax, pivot, color_map, alpha=0.80)

    ax.set_xlabel(_budget_xlabel())
    ax.set_ylabel(ylabel)
    ax.set_title(title, pad=7)
    ax.set_xlim(*_x_lim(pivot.index))
    ax.axhline(0, color="#404040", linewidth=0.8, zorder=8)
    ax.xaxis.set_major_formatter(ticker.StrMethodFormatter("{x:,.0f}"))
    apply_compact_ticks(ax, x_nbins=6, y_nbins=5)
    style_box_axis(ax)

    return cols


# ── Main ───────────────────────────────────────────────────────────────────────

df, carbon_ghg, bio_bio = load_data()

with pd.ExcelWriter(RAW_DATA_PATH, engine="openpyxl") as writer:
    df.to_excel(writer, sheet_name="ContribWithBudget", index=False)
    carbon_ghg.to_excel(writer, sheet_name="CarbonBudgetCurve", index=False)
    bio_bio.to_excel(writer, sheet_name="BioBudgetCurve", index=False)
print(f"Raw data saved: {RAW_DATA_PATH}")

fig, axes = plt.subplots(3, 2, figsize=(14, 14),
                         gridspec_kw={"height_ratios": [1.0, 1.1, 1.1]})

# Row 0
plot_row0(axes[0, 0], axes[0, 1], carbon_ghg, bio_bio)

# Row 1: area-type stacked
plot_row1(
    axes[1, 0], df,
    price_type="CarbonPrice", metric_type=GHG_METRIC,
    ylabel=r"GHG abatement change (Mt CO$_2$e yr$^{-1}$)",
    title="Carbon price: GHG by land-use type vs Budget",
)
plot_row1(
    axes[1, 1], df,
    price_type="BioPrice", metric_type=BIO_METRIC,
    ylabel=r"Biodiversity contribution change (Mha yr$^{-1}$)",
    title="Biodiversity price: Bio by land-use type vs Budget",
)

# Row 2: category-level stacked
cats_left  = plot_row2(
    axes[2, 0], df,
    price_type="CarbonPrice", metric_type=GHG_METRIC,
    ylabel=r"GHG abatement change (Mt CO$_2$e yr$^{-1}$)",
    title="Carbon price: GHG by land-use category vs Budget",
)
cats_right = plot_row2(
    axes[2, 1], df,
    price_type="BioPrice", metric_type=BIO_METRIC,
    ylabel=r"Biodiversity contribution change (Mha yr$^{-1}$)",
    title="Biodiversity price: Bio by land-use category vs Budget",
)

# Shared legend for category rows
all_cats = list(dict.fromkeys(cats_left + cats_right))
ordered_legend = [c for c in CATEGORY_ORDER if c in all_cats]
ordered_legend += [c for c in all_cats if c not in ordered_legend]
handles = [
    mpatches.Patch(
        facecolor=CATEGORY_COLOR_MAP.get(c, "#888888"),
        edgecolor="none",
        label=LEGEND_LABELS.get(c, c),
    )
    for c in ordered_legend
]
fig.legend(
    handles=handles,
    loc="lower center",
    bbox_to_anchor=(0.5, 0.005),
    ncol=4,
    frameon=False,
    handlelength=1.0,
    handleheight=0.9,
    columnspacing=0.9,
    fontsize=FS - 2,
)

plt.tight_layout(rect=[0, 0.12, 1, 1])
plt.subplots_adjust(hspace=0.46, wspace=0.28)

out_path = OUT_DIR / "06_Bio_GHG_Budget.png"
fig.savefig(out_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"Saved: {out_path}")
