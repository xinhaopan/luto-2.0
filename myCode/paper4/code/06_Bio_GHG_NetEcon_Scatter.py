# ==============================================================================
# Figure 06: Budget efficiency and solution mix
#
# Question: For the same budget (income transfer = price × additional quantity),
#   how much GHG abatement AND biodiversity restoration can be achieved?
#   Which land-use subcategories deliver each benefit?
#
# X-axis: Budget (Billion AUD yr⁻¹) = price × additional quantity above baseline
#   Carbon pathway:       Budget = CarbonPrice × ΔGHG (MtCO₂e → B$ : /1000)
#   Biodiversity pathway: Budget = BioPrice    × ΔBio (Mha     → B$ : /1000)
#
# Layout: 3 rows × 2 columns
#   [0,0] Carbon pathway  — Budget → ΔGHG (left y) + ΔBio (right y)  dual-axis
#   [0,1] Bio pathway     — Budget → ΔBio (left y) + ΔGHG (right y)  dual-axis
#   [1,0] Carbon pathway  — Budget → GHG contribution by subcategory  stacked area
#   [1,1] Bio pathway     — Budget → GHG contribution by subcategory  stacked area
#   [2,0] Carbon pathway  — Budget → Bio contribution by subcategory  stacked area
#   [2,1] Bio pathway     — Budget → Bio contribution by subcategory  stacked area
# ==============================================================================

import sys
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from tools.price_slice_utils import (
    DATA_DIR,
    OUT_DIR,
    apply_compact_ticks,
    apply_paper4_color_overrides_to_style_df,
    stacked_area_pos_neg,
    style_box_axis,
)


# ── Config ─────────────────────────────────────────────────────────────────────

YEAR = 2025
BASE_DIR = Path(__file__).resolve().parent
COLOR_FILE  = BASE_DIR.parents[1] / "draw_all" / "code" / "tools" / "land use colors.xlsx"
CONTRIB_CACHE = DATA_DIR / f"04_Contribution_Delta_vs_Zero_raw_data_{YEAR}.xlsx"
RAW_OUT       = DATA_DIR / f"06_Budget_raw_data_{YEAR}.xlsx"

GHG_METRIC = "GHGAbatementChange_vs_ZeroPrice_MtCO2e"
BIO_METRIC  = "BiodiversityContributionChange_vs_ZeroPrice_MhaYr"

COLOR_GHG   = "#1d52a1"   # blue  — GHG axis / line
COLOR_BIO   = "#2ca25f"   # green — Bio axis / line
COLOR_CP    = "#fbbc45"   # amber — carbon price label
COLOR_BP    = "#2ca25f"   # green — bio price label

FS = 11
TOP_N_CATS = 10   # top N subcategories to show in stacked plots

LEGEND_LABELS = {
    "Agricultural technology (energy)":                     "AgTech energy",
    "Agricultural technology (fertiliser)":                 "AgTech fertiliser",
    "Biochar":                                              "Biochar",
    "Early dry-season savanna burning":                     "Savanna burning",
    "Human-Induced Regeneration (beef)":                    "HIR beef",
    "Human-Induced Regeneration (sheep)":                   "HIR sheep",
    "Crops":                                                "Crops",
    "Modified livestock":                                   "Modified livestock",
    "Natural Livestock":                                    "Natural livestock",
    "Unallocated - modified land":                          "Unallocated modified",
    "Unallocated - natural land":                           "Unallocated natural",
    "Agroforestry (mixed species + beef)":                  "Agroforestry beef",
    "Agroforestry (mixed species + sheep)":                 "Agroforestry sheep",
    "Carbon plantings (monoculture)":                       "Carbon plantings",
    "Destocked (natural land)":                             "Destocked natural",
    "Environmental plantings (mixed local native species)": "Environmental plantings",
    "Farm forestry (hardwood timber + beef)":               "Farm forestry beef",
    "Farm forestry (hardwood timber + sheep)":              "Farm forestry sheep",
    "Riparian buffer restoration (mixed species)":          "Riparian restoration",
    "Methane reduction (livestock)":                        "Methane reduction",
    "Regenerative agriculture (livestock)":                 "Regenerative agriculture",
}

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial"],
    "font.size": FS, "axes.titlesize": FS, "axes.labelsize": FS,
    "xtick.labelsize": FS, "ytick.labelsize": FS, "legend.fontsize": FS,
    "mathtext.fontset": "stixsans",
    "axes.facecolor": "#EAEAF2", "grid.color": "white", "grid.linewidth": 1.0,
})


# ── Style ──────────────────────────────────────────────────────────────────────

def _load_style(sheet):
    df = pd.read_excel(COLOR_FILE, sheet_name=sheet)
    df = apply_paper4_color_overrides_to_style_df(df)
    col = "desc_new" if "desc_new" in df.columns else "desc"
    order, cmap = [], {}
    for _, row in df.iterrows():
        label = row[col]
        order.append(label)
        cmap[label] = row["color"]
    return order, cmap


def _build_cat_styles():
    skip = {"Other land", "No agricultural management", "Other land-use",
            "Agricultural land-use", "Livestock"}
    order, cmap = [], {}
    for sheet in ("ag_group", "am", "non_ag"):
        o, c = _load_style(sheet)
        for cat in o:
            if cat in skip or cat in order:
                continue
            order.append(cat)
            cmap[cat] = c.get(cat, "#888888")
    return order, cmap


CAT_ORDER, CAT_COLORS = _build_cat_styles()


# ── Data ───────────────────────────────────────────────────────────────────────

def load_data():
    if not CONTRIB_CACHE.is_file():
        raise FileNotFoundError(
            f"Cache not found: {CONTRIB_CACHE}\n"
            "Run 04_Contribution_Delta_vs_Zero.py first."
        )
    df = pd.read_excel(CONTRIB_CACHE, sheet_name="ContributionLong")

    # Total metric by (PriceType, Price)
    totals = (
        df.groupby(["PriceType", "Price", "MetricType"])["ContributionValue"]
        .sum().reset_index()
    )

    def _get_total(pt, mt):
        return (
            totals[(totals.PriceType == pt) & (totals.MetricType == mt)]
            [["Price", "ContributionValue"]]
            .rename(columns={"ContributionValue": "value"})
            .set_index("Price")["value"]
        )

    cp_ghg = _get_total("CarbonPrice", GHG_METRIC)   # ΔGHG on carbon axis
    cp_bio = _get_total("CarbonPrice", BIO_METRIC)   # ΔBio co-benefit on carbon axis
    bp_bio = _get_total("BioPrice",    BIO_METRIC)   # ΔBio on bio axis
    bp_ghg = _get_total("BioPrice",    GHG_METRIC)   # ΔGHG co-benefit on bio axis

    # Budget (B AUD) = price × primary metric / 1000
    carbon_prices = cp_ghg.index.to_numpy()
    bio_prices    = bp_bio.index.to_numpy()

    cp_budget = pd.Series(carbon_prices * cp_ghg.reindex(carbon_prices).fillna(0).values / 1000,
                          index=carbon_prices, name="Budget_BAud")
    bp_budget = pd.Series(bio_prices    * bp_bio.reindex(bio_prices).fillna(0).values    / 1000,
                          index=bio_prices, name="Budget_BAud")

    # Attach budget to full df
    budget_lookup = pd.concat([
        pd.DataFrame({"PriceType": "CarbonPrice", "Price": carbon_prices,
                      "Budget_BAud": cp_budget.values}),
        pd.DataFrame({"PriceType": "BioPrice",    "Price": bio_prices,
                      "Budget_BAud": bp_budget.values}),
    ], ignore_index=True)
    df = df.merge(budget_lookup, on=["PriceType", "Price"], how="left")

    curves = {
        "cp_ghg": cp_ghg, "cp_bio": cp_bio,
        "bp_bio": bp_bio, "bp_ghg": bp_ghg,
        "cp_budget": cp_budget, "bp_budget": bp_budget,
    }
    return df, curves


# ── Row 0: dual-axis curves ────────────────────────────────────────────────────

def _insert_origin(budget, primary, secondary):
    """Prepend (0, 0, 0) if budget doesn't start at 0."""
    if len(budget) == 0 or not np.isclose(budget[0], 0.0):
        budget    = np.concatenate([[0.0], budget])
        primary   = np.concatenate([[0.0], primary])
        secondary = np.concatenate([[0.0], secondary])
    return budget, primary, secondary


def plot_dual_axis(ax, budget_ser, primary_ser, secondary_ser,
                   primary_label, secondary_label,
                   primary_color, secondary_color,
                   primary_ylabel, secondary_ylabel, title):
    prices = budget_ser.index.to_numpy()
    budget    = budget_ser.reindex(prices).fillna(0).values
    primary   = primary_ser.reindex(prices).fillna(0).values
    secondary = secondary_ser.reindex(prices).fillna(0).values

    order = np.argsort(budget)
    budget, primary, secondary = budget[order], primary[order], secondary[order]
    budget, primary, secondary = _insert_origin(budget, primary, secondary)

    ax2 = ax.twinx()

    line1, = ax.plot(budget, primary, color=primary_color, linewidth=2.0,
                     marker="o", markersize=5.5, markeredgewidth=0, zorder=6,
                     label=primary_label)
    line2, = ax2.plot(budget, secondary, color=secondary_color, linewidth=1.8,
                      marker="s", markersize=4.5, markeredgewidth=0,
                      linestyle="--", zorder=5, label=secondary_label)

    ax.set_xlabel(r"Budget (Billion AU\$ yr$^{-1}$)  =  price × additional quantity")
    ax.set_ylabel(primary_ylabel, color=primary_color)
    ax2.set_ylabel(secondary_ylabel, color=secondary_color)
    ax.tick_params(axis="y", labelcolor=primary_color)
    ax2.tick_params(axis="y", labelcolor=secondary_color)
    ax2.spines["right"].set_visible(True)
    ax.set_title(title, pad=7)
    ax.set_xlim(0, float(budget.max()) * 1.08 + 0.5)
    ax.set_ylim(0, float(primary.max()) * 1.15 + 0.1)
    ax2.set_ylim(0, float(secondary.max()) * 1.15 + 0.02)
    ax.xaxis.set_major_formatter(ticker.StrMethodFormatter("{x:,.0f}"))
    apply_compact_ticks(ax,  x_nbins=6, y_nbins=5)
    apply_compact_ticks(ax2, x_nbins=6, y_nbins=5)
    style_box_axis(ax)

    lines  = [line1, line2]
    labels = [line1.get_label(), line2.get_label()]
    ax.legend(lines, labels, loc="upper left", frameon=True, framealpha=0.9,
              fontsize=FS - 1)


# ── Row 1 & 2: stacked area by subcategory ────────────────────────────────────

def plot_stacked_subcategory(ax, df, price_type, metric_type,
                              budget_ser, ylabel, title):
    sub = df[
        (df["PriceType"] == price_type) &
        (df["MetricType"] == metric_type)
    ].copy()
    if sub.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center",
                transform=ax.transAxes)
        style_box_axis(ax)
        return []

    # Use budget as x
    sub = sub.merge(
        budget_ser.reset_index().rename(columns={"index": "Price", 0: "Budget_BAud"}),
        on="Price", how="left", suffixes=("", "_from_ser"),
    )
    # prefer the Budget_BAud already on df; if merge added _from_ser, drop
    if "Budget_BAud_from_ser" in sub.columns:
        sub = sub.drop(columns=["Budget_BAud_from_ser"])

    # Top-N categories by peak absolute contribution
    cat_peak = (
        sub.groupby("Category")["ContributionValue"]
        .apply(lambda x: x.abs().max())
        .sort_values(ascending=False)
    )
    top_cats = cat_peak.head(TOP_N_CATS).index.tolist()
    ordered  = [c for c in CAT_ORDER if c in top_cats]
    ordered += [c for c in top_cats if c not in ordered]

    sub_top = sub[sub["Category"].isin(ordered)]
    pivot = (
        sub_top.groupby(["Budget_BAud", "Category"])["ContributionValue"]
        .sum().reset_index()
        .pivot(index="Budget_BAud", columns="Category", values="ContributionValue")
        .fillna(0.0)
        .sort_index()
    )
    # Insert zero-budget row
    if not np.isclose(float(pivot.index[0]), 0.0):
        zero = pd.DataFrame([[0.0] * len(pivot.columns)],
                             columns=pivot.columns, index=[0.0])
        pivot = pd.concat([zero, pivot]).sort_index()

    cols     = [c for c in ordered if c in pivot.columns]
    pivot    = pivot[cols]
    cmap     = {c: CAT_COLORS.get(c, "#888888") for c in cols}

    stacked_area_pos_neg(ax, pivot, cmap, alpha=0.82)

    ax.axhline(0, color="#404040", linewidth=0.8, zorder=8)
    ax.set_xlabel(r"Budget (Billion AU\$ yr$^{-1}$)")
    ax.set_ylabel(ylabel)
    ax.set_title(title, pad=7)
    ax.set_xlim(0, float(pivot.index.max()) * 1.06 + 0.5)
    ax.xaxis.set_major_formatter(ticker.StrMethodFormatter("{x:,.0f}"))
    apply_compact_ticks(ax, x_nbins=6, y_nbins=5)
    style_box_axis(ax)
    return cols


# ── Main ───────────────────────────────────────────────────────────────────────

df, curves = load_data()

# Save intermediate data
with pd.ExcelWriter(RAW_OUT, engine="openpyxl") as writer:
    df.to_excel(writer, sheet_name="ContribWithBudget", index=False)
    pd.DataFrame({
        "Budget_BAud": curves["cp_budget"].values,
        "ΔGHG_MtCO2e": curves["cp_ghg"].values,
        "ΔBio_Mha":    curves["cp_bio"].values,
    }, index=curves["cp_budget"].index).to_excel(writer, sheet_name="CarbonCurves")
    pd.DataFrame({
        "Budget_BAud": curves["bp_budget"].values,
        "ΔBio_Mha":    curves["bp_bio"].values,
        "ΔGHG_MtCO2e": curves["bp_ghg"].values,
    }, index=curves["bp_budget"].index).to_excel(writer, sheet_name="BioCurves")
print(f"Raw data saved: {RAW_OUT}")

fig, axes = plt.subplots(3, 2, figsize=(14, 15),
                         gridspec_kw={"height_ratios": [1.0, 1.1, 1.1]})

# ── Row 0: dual-axis curves ────────────────────────────────────────────────────
plot_dual_axis(
    axes[0, 0],
    budget_ser=curves["cp_budget"], primary_ser=curves["cp_ghg"],
    secondary_ser=curves["cp_bio"],
    primary_label=r"GHG abatement (Mt CO$_2$e yr$^{-1}$)",
    secondary_label=r"Biodiversity co-benefit (Mha yr$^{-1}$)",
    primary_color=COLOR_GHG, secondary_color=COLOR_BIO,
    primary_ylabel=r"ΔGHG (Mt CO$_2$e yr$^{-1}$)",
    secondary_ylabel=r"ΔBio (Mha yr$^{-1}$)",
    title="Carbon price pathway: GHG abatement & bio co-benefit vs Budget",
)
plot_dual_axis(
    axes[0, 1],
    budget_ser=curves["bp_budget"], primary_ser=curves["bp_bio"],
    secondary_ser=curves["bp_ghg"],
    primary_label=r"Biodiversity restoration (Mha yr$^{-1}$)",
    secondary_label=r"GHG co-benefit (Mt CO$_2$e yr$^{-1}$)",
    primary_color=COLOR_BIO, secondary_color=COLOR_GHG,
    primary_ylabel=r"ΔBio (Mha yr$^{-1}$)",
    secondary_ylabel=r"ΔGHG (Mt CO$_2$e yr$^{-1}$)",
    title="Biodiversity price pathway: bio restoration & GHG co-benefit vs Budget",
)

# ── Row 1: GHG contribution by subcategory ────────────────────────────────────
ghg_cats_cp = plot_stacked_subcategory(
    axes[1, 0], df, "CarbonPrice", GHG_METRIC,
    budget_ser=curves["cp_budget"],
    ylabel=r"GHG abatement change (Mt CO$_2$e yr$^{-1}$)",
    title="Carbon price: GHG by subcategory vs Budget",
)
ghg_cats_bp = plot_stacked_subcategory(
    axes[1, 1], df, "BioPrice", GHG_METRIC,
    budget_ser=curves["bp_budget"],
    ylabel=r"GHG abatement change (Mt CO$_2$e yr$^{-1}$)",
    title="Biodiversity price: GHG co-benefit by subcategory vs Budget",
)

# ── Row 2: Bio contribution by subcategory ────────────────────────────────────
bio_cats_cp = plot_stacked_subcategory(
    axes[2, 0], df, "CarbonPrice", BIO_METRIC,
    budget_ser=curves["cp_budget"],
    ylabel=r"Biodiversity contribution change (Mha yr$^{-1}$)",
    title="Carbon price: bio co-benefit by subcategory vs Budget",
)
bio_cats_bp = plot_stacked_subcategory(
    axes[2, 1], df, "BioPrice", BIO_METRIC,
    budget_ser=curves["bp_budget"],
    ylabel=r"Biodiversity contribution change (Mha yr$^{-1}$)",
    title="Biodiversity price: bio by subcategory vs Budget",
)

# ── Shared legend ──────────────────────────────────────────────────────────────
all_cats = list(dict.fromkeys(ghg_cats_cp + ghg_cats_bp + bio_cats_cp + bio_cats_bp))
legend_cats = [c for c in CAT_ORDER if c in all_cats]
legend_cats += [c for c in all_cats if c not in legend_cats]
handles = [
    mpatches.Patch(
        facecolor=CAT_COLORS.get(c, "#888888"), edgecolor="none",
        label=LEGEND_LABELS.get(c, c),
    )
    for c in legend_cats
]
fig.legend(
    handles=handles, loc="lower center",
    bbox_to_anchor=(0.5, 0.005), ncol=4,
    frameon=False, handlelength=1.0, handleheight=0.9,
    columnspacing=0.9, fontsize=FS - 2,
)

plt.tight_layout(rect=[0, 0.10, 1, 1])
plt.subplots_adjust(hspace=0.48, wspace=0.32)

out_path = OUT_DIR / "06_Bio_GHG_Budget.png"
fig.savefig(out_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"Saved: {out_path}")
