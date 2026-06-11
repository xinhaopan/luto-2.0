# ==============================================================================
# Figure 06: Budget response of GHG abatement and biodiversity contribution
#
# Layout (3 rows × 2 columns):
#   [0,0] ZOOM:  x = carbon budget range (0 → ~60 B$)
#   [0,1] FULL:  x = full budget range   (0 → ~320 B$)
#         Both panels carry 4 lines (colour = pathway, marker = metric):
#           GHG response          = blue
#           Biodiversity response = green
#           Primary response      = solid line
#           Co-benefit response   = dashed line
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
CONTRIB_CACHE = DATA_DIR / f"03_Contribution_Delta_vs_Zero_raw_data_{YEAR}.xlsx"
RAW_OUT       = DATA_DIR / f"05_Bio_GHG_Budget_raw_data_{YEAR}.xlsx"

GHG_METRIC = "GHGAbatementChange_vs_ZeroPrice_MtCO2e"
BIO_METRIC  = "BiodiversityContributionChange_vs_ZeroPrice_MhaYr"

COLOR_GHG = "#1d52a1"
COLOR_BIO = "#2ca25f"
COLOR_CON = "#888888"

FS = 11
TOP_N_CATS = None

LEGEND_LABELS = {
    # AM — Ag2050 naming
    "Methane reduction (livestock)":                        "Methane reduction",
    "Agricultural technology (fertiliser)":                 "AgTech fertiliser",
    "Regenerative agriculture (livestock)":                 "Regenerative agriculture",
    "Early dry-season savanna burning":                     "Savanna burning",
    "Agricultural technology (energy)":                     "AgTech energy",
    "Biochar (soil amendment)":                             "Biochar",
    "Managed regeneration (beef)":                          "Managed regeneration (beef)",
    "Managed regeneration (sheep)":                         "Managed regeneration (sheep)",
    # Ag land-use groups
    "Crops":                                                "Crops",
    "Modified livestock":                                   "Modified livestock",
    "Natural Livestock":                                    "Natural livestock",
    "Unallocated - modified land":                          "Unallocated modified",
    "Unallocated - natural land":                           "Unallocated natural",
    # Non-ag — Ag2050 naming
    "Environmental plantings (mixed species)":              "Environmental plantings",
    "Riparian buffer restoration (mixed species)":          "Riparian restoration",
    "Agroforestry (mixed species + sheep)":                 "Agroforestry sheep",
    "Agroforestry (mixed species + beef)":                  "Agroforestry beef",
    "Carbon plantings (monoculture)":                       "Carbon plantings",
    "Farm forestry (hardwood timber + sheep)":              "Farm forestry sheep",
    "Farm forestry (hardwood timber + beef)":               "Farm forestry beef",
    "BECCS (Bioenergy with Carbon Capture and Storage)":    "BECCS",
    "Destocked - natural land":                             "Destocked natural",
}

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial"],
    "font.size": FS, "axes.titlesize": FS, "axes.labelsize": FS,
    "xtick.labelsize": FS, "ytick.labelsize": FS, "legend.fontsize": FS,
    "mathtext.fontset": "stixsans",
    "axes.facecolor": "#EAEAF2", "grid.color": "white", "grid.linewidth": 1.0,
})


# ── Category colour map ────────────────────────────────────────────────────────

# Ag2050 naming: remap color-table desc_new -> Ag2050 display name
_AG2050_DISPLAY = {
    "Biochar":                                              "Biochar (soil amendment)",
    "Human-Induced Regeneration (beef)":                    "Managed regeneration (beef)",
    "Human-Induced Regeneration (sheep)":                   "Managed regeneration (sheep)",
    "Environmental plantings (mixed local native species)": "Environmental plantings (mixed species)",
    "BECCS (Bioenergy with carbon capture and storage)":    "BECCS (Bioenergy with Carbon Capture and Storage)",
    "Destocked (natural land)":                             "Destocked - natural land",
}


def _load_style(sheet):
    df = pd.read_excel(COLOR_FILE, sheet_name=sheet)
    df = apply_paper4_color_overrides_to_style_df(df)
    col = "desc_new" if "desc_new" in df.columns else "desc"
    order, cmap = [], {}
    for _, row in df.iterrows():
        label = _AG2050_DISPLAY.get(row[col], row[col])
        order.append(label)
        cmap[label] = row["color"]
    return order, cmap


def _split_livestock_style(order, cmap):
    livestock_color = cmap.get("Livestock", "#c49a67")
    new_order = []
    for cat in order:
        if cat == "Livestock":
            new_order.extend(["Modified livestock", "Natural Livestock"])
        else:
            new_order.append(cat)

    new_cmap = {
        cat: color
        for cat, color in cmap.items()
        if cat != "Livestock"
    }
    new_cmap["Modified livestock"] = "#762500"
    new_cmap["Natural Livestock"] = livestock_color
    return new_order, new_cmap


def _ordered_without(order, excluded):
    return [cat for cat in order if cat not in excluded]


def _build_cat_styles():
    ag_order, ag_cmap = _load_style("ag_group")
    ag_order, ag_cmap = _split_livestock_style(ag_order, ag_cmap)

    am_order, am_cmap = _load_style("am")
    non_ag_order, non_ag_cmap = _load_style("non_ag")
    lu_order, lu_cmap = _load_style("lu")

    ag_order = _ordered_without(ag_order, {"Other land"})
    am_order = _ordered_without(am_order, {"No agricultural management", "Other land-use"})
    non_ag_order = _ordered_without(non_ag_order, {"Agricultural land-use", "Other land-use"})

    # Transition is reported with agricultural land-use GHG in Figure 04.
    if "Transition" not in ag_order:
        ag_order.append("Transition")
    ag_cmap["Transition"] = lu_cmap.get("Transition", "#D2E0FB")

    cat_order = ag_order + am_order + non_ag_order
    cat_cmap = {}
    for cmap in (ag_cmap, am_cmap, non_ag_cmap):
        cat_cmap.update(cmap)
    cat_cmap["Transition"] = ag_cmap["Transition"]

    group_order = {
        "Ag": ag_order,
        "AM": am_order,
        "Non-ag": non_ag_order,
    }
    return cat_order, cat_cmap, group_order


CAT_ORDER, CAT_COLORS, CAT_GROUP_ORDER = _build_cat_styles()


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

def _draw_drive_lines(ax, curves, drive, xlim):
    """
    Draw budget response lines for one drive on twin y-axes.
      left  y-axis = GHG abatement difference
      right y-axis = biodiversity contribution difference
      color        = response metric
      line style   = primary-vs-co-benefit role
    Return ax_twin.
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

    kw_primary = dict(
        linewidth=2.1,
        markersize=7.4,
        markeredgewidth=0,
        linestyle="-",
        alpha=0.96,
        zorder=7,
    )
    kw_cobenefit = dict(
        linewidth=1.65,
        markersize=6.5,
        markeredgewidth=0,
        linestyle="--",
        alpha=0.9,
        zorder=6,
    )

    if drive == "carbon":
        l_ghg, = ax.plot(
            cp_bud,
            cp_ghg,
            color=COLOR_GHG,
            marker="o",
            label="GHG abatement",
            **kw_primary,
        )
        l_bio, = ax2.plot(
            cp_bud,
            cp_bio,
            color=COLOR_BIO,
            marker="D",
            label="Biodiversity co-benefit",
            **kw_cobenefit,
        )
        visible_ghg = cp_ghg[cp_bud <= xlim[1]]
        visible_bio = cp_bio[cp_bud <= xlim[1]]
    elif drive == "biodiversity":
        l_ghg, = ax.plot(
            bp_bud,
            bp_ghg,
            color=COLOR_GHG,
            marker="^",
            label="GHG co-benefit",
            **kw_cobenefit,
        )
        l_bio, = ax2.plot(
            bp_bud,
            bp_bio,
            color=COLOR_BIO,
            marker="s",
            label="Biodiversity contribution",
            **kw_primary,
        )
        visible_ghg = bp_ghg[bp_bud <= xlim[1]]
        visible_bio = bp_bio[bp_bud <= xlim[1]]
    else:
        raise ValueError(f"Unsupported drive: {drive}")

    # Axis styling
    ax.set_xlim(*xlim)
    ax2.set_xlim(*xlim)
    ax.set_xlabel("")
    ax.set_ylabel(r"GHG abatement difference (Mt CO$_2$e yr$^{-1}$)", color="black")
    ax2.set_ylabel(r"Biodiversity contribution difference (Mha yr$^{-1}$)", color="black")
    ax.tick_params(axis="y", colors="black", labelcolor="black")
    ax2.tick_params(axis="y", colors="black", labelcolor="black")
    ax.yaxis.label.set_color("black")
    ax2.yaxis.label.set_color("black")
    ax2.spines["right"].set_visible(True)
    ax2.spines["right"].set_color("black")

    # Y limits — start from 0, include all visible data
    ax.set_ylim(0,  max(float(np.nanmax(visible_ghg)), 0.1) * 1.15)
    ax2.set_ylim(0, max(float(np.nanmax(visible_bio)), 0.01) * 1.15)
    ax.xaxis.set_major_formatter(ticker.StrMethodFormatter("{x:,.0f}"))
    apply_compact_ticks(ax,  x_nbins=6, y_nbins=5)
    apply_compact_ticks(ax2, x_nbins=6, y_nbins=5)
    style_box_axis(ax)

    lines = [l_ghg, l_bio]
    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels, loc="upper left", frameon=True,
              framealpha=0.9, fontsize=FS - 1)

    return ax2


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
    ax_full.text(
        x_zoom_max * 0.5,
        0.965,
        "zoomed range",
        ha="center",
        va="top",
        color=COLOR_CON,
        fontsize=FS - 1,
        transform=ax_full.get_xaxis_transform(),
    )

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

    # Include all non-zero contribution categories. Some management options are
    # tiny, but retaining them keeps the legend accounting complete.
    cat_peak = (
        sub.groupby("Category")["ContributionValue"]
        .apply(lambda x: x.abs().max())
        .sort_values(ascending=False)
    )
    if TOP_N_CATS is None:
        visible_cats = cat_peak[cat_peak > 0].index.tolist()
    else:
        visible_cats = cat_peak[cat_peak > 0].head(TOP_N_CATS).index.tolist()
    ordered  = [c for c in CAT_ORDER if c in visible_cats]
    ordered += [c for c in visible_cats if c not in ordered]

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
    ax.set_xlabel("")
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

fig, axes = plt.subplots(3, 2, figsize=(10, 14.8),
                         gridspec_kw={"height_ratios": [0.95, 1.0, 1.0]})

# ── Row 0 ──────────────────────────────────────────────────────────────────────
ax_left2 = _draw_drive_lines(axes[0, 0], curves, drive="carbon", xlim=(0, x_max_cp))
axes[0, 0].set_title("Carbon Price Drive", pad=7)

ax_right2 = _draw_drive_lines(axes[0, 1], curves, drive="biodiversity", xlim=(0, x_max_bp))
axes[0, 1].set_title("Biodiversity Price Drive", pad=7)

# Sync row 0: primary y (GHG left axis) and secondary y (Bio right axis)
sync_ylims(axes[0, 0], axes[0, 1])
sync_ylims(ax_left2, ax_right2)
axes[0, 0].yaxis.label.set_rotation(270)
axes[0, 0].yaxis.label.set_va("bottom")
axes[0, 0].yaxis.set_label_coords(-0.12, 0.5)

# Keep only the outer y-axes in the first row:
# left outer axis for GHG and right outer axis for biodiversity.
ax_left2.set_ylabel("")
ax_left2.tick_params(axis="y", right=False, labelright=False)
ax_left2.spines["right"].set_visible(False)
axes[0, 1].set_ylabel("")
axes[0, 1].tick_params(axis="y", left=False, labelleft=False)
axes[0, 1].spines["left"].set_visible(False)

# ── Row 1: GHG by subcategory ──────────────────────────────────────────────────
ghg_cats_cp = plot_stacked_sub(
    axes[1, 0], df, "CarbonPrice", GHG_METRIC, "Budget_BAud",
    ylabel=r"GHG abatement difference (Mt CO$_2$e yr$^{-1}$)",
    title="GHG Contribution",
)
ghg_cats_bp = plot_stacked_sub(
    axes[1, 1], df, "BioPrice", GHG_METRIC, "Budget_BAud",
    ylabel=r"GHG abatement difference (Mt CO$_2$e yr$^{-1}$)",
    title="GHG Co-benefit",
)
# Sync row 1
sync_ylims(axes[1, 0], axes[1, 1])
axes[1, 1].set_ylabel("")
axes[1, 1].tick_params(axis="y", left=False, labelleft=False)

# ── Row 2: Bio by subcategory ──────────────────────────────────────────────────
bio_cats_cp = plot_stacked_sub(
    axes[2, 0], df, "CarbonPrice", BIO_METRIC, "Budget_BAud",
    ylabel=r"Biodiversity contribution difference (Mha yr$^{-1}$)",
    title="Biodiversity Co-benefit",
)
bio_cats_bp = plot_stacked_sub(
    axes[2, 1], df, "BioPrice", BIO_METRIC, "Budget_BAud",
    ylabel=r"Biodiversity contribution difference (Mha yr$^{-1}$)",
    title="Biodiversity Contribution",
)
# Sync row 2
sync_ylims(axes[2, 0], axes[2, 1])
axes[2, 1].set_ylabel("")
axes[2, 1].tick_params(axis="y", left=False, labelleft=False)

# Shared x-axis treatment by column.
for row_idx in range(3):
    axes[row_idx, 0].set_xlim(0, x_max_cp)
    axes[row_idx, 1].set_xlim(0, x_max_bp)
    axes[row_idx, 0].xaxis.set_major_formatter(ticker.StrMethodFormatter("{x:,.0f}"))
    axes[row_idx, 1].xaxis.set_major_formatter(ticker.StrMethodFormatter("{x:,.0f}"))

ax_left2.set_xlim(0, x_max_cp)
ax_right2.set_xlim(0, x_max_bp)

for ax in axes[:2, :].flat:
    ax.tick_params(axis="x", labelbottom=False)

for ax in axes[2, :]:
    ax.set_xlabel("")

for ax in list(axes.flat) + [ax_left2, ax_right2]:
    ax.tick_params(axis="both", colors="black", labelcolor="black")
    ax.xaxis.label.set_color("black")
    ax.yaxis.label.set_color("black")
    ax.title.set_color("black")
    for spine in ax.spines.values():
        spine.set_color("black")

# ── Shared legend ──────────────────────────────────────────────────────────────
all_cats = list(dict.fromkeys(
    ghg_cats_cp + ghg_cats_bp + bio_cats_cp + bio_cats_bp
))

def _group_visible_cats(group_name):
    cats = [cat for cat in CAT_GROUP_ORDER[group_name] if cat in all_cats]
    if group_name == "Non-ag":
        known_cats = set(sum(CAT_GROUP_ORDER.values(), []))
        cats += [cat for cat in all_cats if cat not in known_cats]
    return cats


def _draw_grouped_legend():
    legend_ax = fig.add_axes([0.04, 0.035, 0.92, 0.18])
    legend_ax.set_axis_off()
    legend_ax.set_xlim(0, 1)
    legend_ax.set_ylim(0, 1)

    col_x = [0.26, 0.52, 0.76]
    y_rows = [0.91, 0.79, 0.65, 0.53, 0.39, 0.27, 0.15]
    marker_size = 66
    text_dx = 0.014

    group_labels = {
        "Ag": "Agricultural land-use",
        "AM": "Agricultural management",
        "Non-ag": "Non-agricultural land-use",
    }

    row_idx = 0
    for group_name in ["Ag", "AM", "Non-ag"]:
        cats = _group_visible_cats(group_name)
        chunks = [cats[i:i + len(col_x)] for i in range(0, len(cats), len(col_x))]
        if row_idx >= len(y_rows):
            break
        group_ys = y_rows[row_idx:row_idx + len(chunks)]
        legend_ax.text(
            0.19, float(np.mean(group_ys)), group_labels[group_name],
            ha="right", va="center", fontsize=FS,
            color="black",
            transform=legend_ax.transAxes,
        )
        for chunk in chunks:
            if row_idx >= len(y_rows):
                break
            y = y_rows[row_idx]
            for x, cat in zip(col_x, chunk):
                legend_ax.scatter(
                    [x], [y], s=marker_size, marker="s",
                    color=CAT_COLORS.get(cat, "#888888"),
                    edgecolors="none",
                    transform=legend_ax.transAxes,
                    clip_on=False,
                )
                legend_ax.text(
                    x + text_dx, y, LEGEND_LABELS.get(cat, cat),
                    ha="left", va="center", fontsize=FS,
                    color="black", transform=legend_ax.transAxes,
                )
            row_idx += 1


fig.supxlabel(r"Budget (Billion AU\$ yr$^{-1}$)", y=0.212, fontsize=FS)

plt.tight_layout(rect=[0, 0.20, 1, 1])
plt.subplots_adjust(hspace=0.16, wspace=0.10)
_draw_grouped_legend()

out_path = OUT_DIR / "05_Bio_GHG_Budget.png"
fig.savefig(out_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"Saved: {out_path}")
