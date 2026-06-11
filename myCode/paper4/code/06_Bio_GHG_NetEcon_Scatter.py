# ==============================================================================
# Figure 06: Budget efficiency — what does each budget level buy?
#
# Layout (3 rows × 2 columns):
#   [0,0] ZOOM:  x = carbon budget range (0 → ~60 B$)
#   [0,1] FULL:  x = full budget range   (0 → ~320 B$)
#         Both panels carry 4 lines (colour = pathway, marker = metric):
#           Carbon GHG   — blue, solid,  circle  (primary)
#           Carbon Bio   — blue, dashed, square  (co-benefit)
#           Bio    Bio   — green, solid,  square  (primary)
#           Bio    GHG   — green, dashed, circle  (co-benefit)
#         Dashed lines connect [0,0]'s right edge to the matching x-position in [0,1].
#
#   [1,0] Carbon path  — GHG contribution by subcategory vs Budget
#   [1,1] Bio path     — GHG co-benefit by subcategory vs Budget
#   [2,0] Carbon path  — Bio co-benefit by subcategory vs Budget
#   [2,1] Bio path     — Bio contribution by subcategory vs Budget
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
from matplotlib.patches import ConnectionPatch

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
COLOR_FILE    = BASE_DIR.parents[1] / "draw_all" / "code" / "tools" / "land use colors.xlsx"
CONTRIB_CACHE = DATA_DIR / f"04_Contribution_Delta_vs_Zero_raw_data_{YEAR}.xlsx"
RAW_OUT       = DATA_DIR / f"06_Budget_raw_data_{YEAR}.xlsx"

GHG_METRIC = "GHGAbatementChange_vs_ZeroPrice_MtCO2e"
BIO_METRIC  = "BiodiversityContributionChange_vs_ZeroPrice_MhaYr"

COLOR_CP  = "#1d52a1"   # blue  — carbon price pathway
COLOR_BP  = "#2ca25f"   # green — biodiversity price pathway
COLOR_CON = "#888888"   # grey  — connection lines / zoom indicator

FS = 11
TOP_N_CATS = 10

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


# ── Category colour map ────────────────────────────────────────────────────────

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


# ── Data loading ───────────────────────────────────────────────────────────────

def _prepend_zero(budget, *arrays):
    """Insert a zero point at budget=0 if not already present."""
    idx = np.argsort(budget)
    budget = budget[idx]
    arrays = [a[idx] for a in arrays]
    if not np.isclose(budget[0], 0.0):
        budget = np.concatenate([[0.0], budget])
        arrays = [np.concatenate([[0.0], a]) for a in arrays]
    return (budget, *arrays)


def load_data():
    if not CONTRIB_CACHE.is_file():
        raise FileNotFoundError(
            f"Cache not found: {CONTRIB_CACHE}\n"
            "Run 04_Contribution_Delta_vs_Zero.py first."
        )
    df = pd.read_excel(CONTRIB_CACHE, sheet_name="ContributionLong")

    totals = (
        df.groupby(["PriceType", "Price", "MetricType"])["ContributionValue"]
        .sum().reset_index()
    )

    def _get(pt, mt):
        s = totals[(totals.PriceType == pt) & (totals.MetricType == mt)]
        return s.set_index("Price")["ContributionValue"]

    cp_ghg = _get("CarbonPrice", GHG_METRIC)
    cp_bio = _get("CarbonPrice", BIO_METRIC)
    bp_bio = _get("BioPrice",    BIO_METRIC)
    bp_ghg = _get("BioPrice",    GHG_METRIC)

    # Budget (B AUD) = price × primary metric / 1000
    cp_prices = cp_ghg.index.to_numpy(float)
    bp_prices = bp_bio.index.to_numpy(float)

    cp_budget_vals = cp_prices * cp_ghg.reindex(cp_prices).fillna(0).values / 1000
    bp_budget_vals = bp_prices * bp_bio.reindex(bp_prices).fillna(0).values / 1000

    cp_budget = pd.Series(cp_budget_vals, index=cp_prices)
    bp_budget = pd.Series(bp_budget_vals, index=bp_prices)

    # Attach budget to df for subcategory plots
    budget_lookup = pd.concat([
        pd.DataFrame({"PriceType": "CarbonPrice", "Price": cp_prices,
                      "Budget_BAud": cp_budget_vals}),
        pd.DataFrame({"PriceType": "BioPrice",    "Price": bp_prices,
                      "Budget_BAud": bp_budget_vals}),
    ], ignore_index=True)
    df = df.merge(budget_lookup, on=["PriceType", "Price"], how="left")

    curves = dict(cp_budget=cp_budget, cp_ghg=cp_ghg, cp_bio=cp_bio,
                  bp_budget=bp_budget, bp_bio=bp_bio, bp_ghg=bp_ghg)
    return df, curves


# ── Row 0: combined 4-line dual-axis plot ─────────────────────────────────────

def _draw_4lines(ax, curves, xlim):
    """
    Draw 4 lines on ax (twin-y):
      left  y = GHG (MtCO₂e)   colour = pathway colour
      right y = Bio (Mha)       colour = pathway colour
    Return (ax_twin, x_max_cp) for connection-line use.
    """
    ax2 = ax.twinx()

    cp_bud, cp_ghg, cp_bio = _prepend_zero(
        curves["cp_budget"].values,
        curves["cp_ghg"].reindex(curves["cp_budget"].index).fillna(0).values,
        curves["cp_bio"].reindex(curves["cp_budget"].index).fillna(0).values,
    )
    bp_bud, bp_bio, bp_ghg = _prepend_zero(
        curves["bp_budget"].values,
        curves["bp_bio"].reindex(curves["bp_budget"].index).fillna(0).values,
        curves["bp_ghg"].reindex(curves["bp_budget"].index).fillna(0).values,
    )

    kw_primary   = dict(linewidth=2.0, markersize=5.5, markeredgewidth=0, zorder=6)
    kw_cobenefit = dict(linewidth=1.5, markersize=4.5, markeredgewidth=0,
                        linestyle="--", alpha=0.85, zorder=5)

    # Left axis — GHG
    l_cp_ghg, = ax.plot(cp_bud, cp_ghg, color=COLOR_CP, marker="o",
                        label=r"Carbon price → GHG (primary)", **kw_primary)
    l_bp_ghg, = ax.plot(bp_bud, bp_ghg, color=COLOR_BP, marker="o",
                        label=r"Bio price → GHG co-benefit", **kw_cobenefit)
    # Right axis — Bio
    l_cp_bio, = ax2.plot(cp_bud, cp_bio, color=COLOR_CP, marker="s",
                         label=r"Carbon price → Bio co-benefit", **kw_cobenefit)
    l_bp_bio, = ax2.plot(bp_bud, bp_bio, color=COLOR_BP, marker="s",
                         label=r"Bio price → Bio (primary)", **kw_primary)

    # Axis styling
    ax.set_xlim(*xlim)
    ax2.set_xlim(*xlim)
    ax.set_xlabel(r"Budget (Billion AU\$ yr$^{-1}$)  =  price × additional quantity")
    ax.set_ylabel(r"ΔGHG (Mt CO$_2$e yr$^{-1}$)", color=COLOR_CP)
    ax2.set_ylabel(r"ΔBio (Mha yr$^{-1}$)",        color=COLOR_BP)
    ax.tick_params(axis="y", labelcolor=COLOR_CP)
    ax2.tick_params(axis="y", labelcolor=COLOR_BP)
    ax2.spines["right"].set_visible(True)

    # Y limits — start from 0, include all visible data
    visible_ghg = np.concatenate([
        cp_ghg[cp_bud <= xlim[1]], bp_ghg[bp_bud <= xlim[1]]
    ])
    visible_bio = np.concatenate([
        cp_bio[cp_bud <= xlim[1]], bp_bio[bp_bud <= xlim[1]]
    ])
    ax.set_ylim(0,  max(float(np.nanmax(visible_ghg)), 0.1) * 1.15)
    ax2.set_ylim(0, max(float(np.nanmax(visible_bio)), 0.01) * 1.15)
    ax.xaxis.set_major_formatter(ticker.StrMethodFormatter("{x:,.0f}"))
    apply_compact_ticks(ax,  x_nbins=6, y_nbins=5)
    apply_compact_ticks(ax2, x_nbins=6, y_nbins=5)
    style_box_axis(ax)

    # Legend on left axis
    lines  = [l_cp_ghg, l_bp_bio, l_bp_ghg, l_cp_bio]
    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels, loc="upper left", frameon=True,
              framealpha=0.9, fontsize=FS - 1)

    return ax2, float(cp_bud.max())


def sync_ylims(*axes_list):
    """Set all axes to the same y-range and recompute ticks."""
    y_min = min(ax.get_ylim()[0] for ax in axes_list)
    y_max = max(ax.get_ylim()[1] for ax in axes_list)
    for ax in axes_list:
        ax.set_ylim(y_min, y_max)
        ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=5, steps=[1,2,5,10]))


def add_zoom_connectors(fig, ax_zoom, ax_full, x_zoom_max, full_xlim):
    """
    Draw dashed lines from the right edge of ax_zoom to the
    corresponding vertical position in ax_full (in axes-fraction coords).
    Also shade the zoom region in ax_full.
    """
    x_frac = np.clip(
        (x_zoom_max - full_xlim[0]) / (full_xlim[1] - full_xlim[0]), 0, 1
    )

    # Shade zoom region in full panel
    ax_full.axvspan(0, x_zoom_max, alpha=0.07, color="steelblue", zorder=0)

    # Vertical marker line in full panel
    ax_full.axvline(x_zoom_max, color=COLOR_CON, lw=0.9, ls=":", zorder=4)

    # Top connector: top-right of zoom → top of zoom-region in full
    for y_a, y_b in [(1.0, 1.0), (0.0, 0.0)]:
        con = ConnectionPatch(
            xyA=(1.0, y_a), coordsA="axes fraction", axesA=ax_zoom,
            xyB=(x_frac, y_b), coordsB="axes fraction", axesB=ax_full,
            color=COLOR_CON, lw=0.8, linestyle="--", alpha=0.7, zorder=100,
        )
        fig.add_artist(con)


# ── Row 1 & 2: stacked-area subcategory plots ─────────────────────────────────

def plot_stacked_sub(ax, df, price_type, metric_type,
                     budget_col, ylabel, title):
    sub = df[
        (df["PriceType"] == price_type) &
        (df["MetricType"] == metric_type)
    ].copy()
    if sub.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center",
                transform=ax.transAxes)
        style_box_axis(ax)
        return []

    # Top-N by peak absolute contribution
    cat_peak = (
        sub.groupby("Category")["ContributionValue"]
        .apply(lambda x: x.abs().max())
        .sort_values(ascending=False)
    )
    top_cats = cat_peak.head(TOP_N_CATS).index.tolist()
    ordered  = [c for c in CAT_ORDER if c in top_cats]
    ordered += [c for c in top_cats if c not in ordered]

    pivot = (
        sub[sub["Category"].isin(ordered)]
        .groupby([budget_col, "Category"])["ContributionValue"]
        .sum().reset_index()
        .pivot(index=budget_col, columns="Category", values="ContributionValue")
        .fillna(0.0).sort_index()
    )
    if not np.isclose(float(pivot.index[0]), 0.0):
        zero = pd.DataFrame([[0.0] * len(pivot.columns)],
                             columns=pivot.columns, index=[0.0])
        pivot = pd.concat([zero, pivot]).sort_index()

    cols  = [c for c in ordered if c in pivot.columns]
    pivot = pivot[cols]
    cmap  = {c: CAT_COLORS.get(c, "#888888") for c in cols}

    stacked_area_pos_neg(ax, pivot, cmap, alpha=0.85)

    ax.axhline(0, color="#404040", lw=0.8, zorder=8)
    ax.set_xlabel(r"Budget (Billion AU\$ yr$^{-1}$)")
    ax.set_ylabel(ylabel)
    ax.set_title(title, pad=7)
    x_max = float(pivot.index.max())
    ax.set_xlim(0, x_max * 1.06 + 0.5)
    ax.xaxis.set_major_formatter(ticker.StrMethodFormatter("{x:,.0f}"))
    apply_compact_ticks(ax, x_nbins=6, y_nbins=5)
    style_box_axis(ax)
    return cols


# ── Main ───────────────────────────────────────────────────────────────────────

df, curves = load_data()

# Determine x limits
x_max_cp = float(curves["cp_budget"].max()) * 1.08 + 1.0
x_max_bp = float(curves["bp_budget"].max()) * 1.06 + 2.0

# Save intermediate
with pd.ExcelWriter(RAW_OUT, engine="openpyxl") as writer:
    df.to_excel(writer, sheet_name="ContribWithBudget", index=False)
print(f"Raw data saved: {RAW_OUT}")

fig, axes = plt.subplots(3, 2, figsize=(14, 15),
                         gridspec_kw={"height_ratios": [1.0, 1.1, 1.1]})

# ── Row 0 ──────────────────────────────────────────────────────────────────────
ax_zoom2, x_zoom_max = _draw_4lines(axes[0, 0], curves, xlim=(0, x_max_cp))
axes[0, 0].set_title("Carbon budget range (zoom)", pad=7)

ax_full2, _ = _draw_4lines(axes[0, 1], curves, xlim=(0, x_max_bp))
axes[0, 1].set_title("Full budget range", pad=7)
axes[0, 1].get_legend().remove()   # keep legend only on zoom panel

# Sync row 0: primary y (GHG left axis) and secondary y (Bio right axis)
sync_ylims(axes[0, 0], axes[0, 1])
sync_ylims(ax_zoom2, ax_full2)

# ── Row 1: GHG by subcategory ──────────────────────────────────────────────────
ghg_cats_cp = plot_stacked_sub(
    axes[1, 0], df, "CarbonPrice", GHG_METRIC, "Budget_BAud",
    ylabel=r"GHG abatement change (Mt CO$_2$e yr$^{-1}$)",
    title="Carbon price: GHG by subcategory vs Budget",
)
ghg_cats_bp = plot_stacked_sub(
    axes[1, 1], df, "BioPrice", GHG_METRIC, "Budget_BAud",
    ylabel=r"GHG abatement change (Mt CO$_2$e yr$^{-1}$)",
    title="Biodiversity price: GHG co-benefit by subcategory vs Budget",
)
# Sync row 1
sync_ylims(axes[1, 0], axes[1, 1])

# ── Row 2: Bio by subcategory ──────────────────────────────────────────────────
bio_cats_cp = plot_stacked_sub(
    axes[2, 0], df, "CarbonPrice", BIO_METRIC, "Budget_BAud",
    ylabel=r"Biodiversity contribution change (Mha yr$^{-1}$)",
    title="Carbon price: Bio co-benefit by subcategory vs Budget",
)
bio_cats_bp = plot_stacked_sub(
    axes[2, 1], df, "BioPrice", BIO_METRIC, "Budget_BAud",
    ylabel=r"Biodiversity contribution change (Mha yr$^{-1}$)",
    title="Biodiversity price: Bio by subcategory vs Budget",
)
# Sync row 2
sync_ylims(axes[2, 0], axes[2, 1])

# ── Shared legend ──────────────────────────────────────────────────────────────
all_cats = list(dict.fromkeys(
    ghg_cats_cp + ghg_cats_bp + bio_cats_cp + bio_cats_bp
))
legend_cats = [c for c in CAT_ORDER if c in all_cats]
legend_cats += [c for c in all_cats if c not in legend_cats]

handles = [
    mpatches.Patch(
        facecolor=CAT_COLORS.get(c, "#888888"),
        edgecolor="none",
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

out_path = OUT_DIR / "06_Bio_GHG_Budget.png"
fig.savefig(out_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"Saved: {out_path}")
