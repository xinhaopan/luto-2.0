# ==============================================================================
# Figure 06: GHG and biodiversity changes against economic return response
#   Left panel:  x = NER difference, y = GHG abatement difference
#   Right panel: x = NER difference, y = biodiversity contribution difference
#   Lines show carbon-price and biodiversity-price pathways.
# ==============================================================================

import os
import sys

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tools.price_slice_utils import (
    DATA_DIR,
    OUT_DIR,
    apply_compact_ticks,
    style_box_axis,
)


YEAR = 2025
CONTRIBUTION_CACHE = DATA_DIR / f"04_Contribution_Delta_vs_Zero_raw_data_{YEAR}.xlsx"
NET_ECON_CACHE = DATA_DIR / f"05_NetEcon_Delta_vs_Zero_raw_data_{YEAR}.xlsx"
RAW_DATA_PATH = DATA_DIR / f"06_Bio_GHG_NetEcon_Scatter_raw_data_{YEAR}.xlsx"

GHG_METRIC = "GHGAbatementChange_vs_ZeroPrice_MtCO2e"
BIO_METRIC = "BiodiversityContributionChange_vs_ZeroPrice_MhaYr"
NER_COLUMN = "NetEconChange_vs_ZeroPrice_BAUD"

COLOR_CARBON = "#fbbc45"
COLOR_BIO = "#2ca25f"
FS = 11

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


def require_file(path):
    if not path.is_file():
        raise FileNotFoundError(f"Required cache not found: {path}")


def load_pathway_data():
    require_file(CONTRIBUTION_CACHE)
    require_file(NET_ECON_CACHE)

    df_contrib = pd.read_excel(CONTRIBUTION_CACHE, sheet_name="ContributionLong")
    df_ner = pd.read_excel(NET_ECON_CACHE, sheet_name="NetEconLong")

    metric_totals = (
        df_contrib
        .groupby(["PriceType", "Price", "MetricType"], as_index=False)["ContributionValue"]
        .sum()
    )
    metric_wide = (
        metric_totals
        .pivot_table(
            index=["PriceType", "Price"],
            columns="MetricType",
            values="ContributionValue",
            aggfunc="sum",
            fill_value=0.0,
        )
        .reset_index()
        .rename(columns={
            GHG_METRIC: "GHGAbatementChange_MtCO2e",
            BIO_METRIC: "BioContributionChange_MhaYr",
        })
    )

    ner_totals = (
        df_ner
        .groupby(["PriceType", "Price"], as_index=False)[NER_COLUMN]
        .sum()
        .rename(columns={NER_COLUMN: "NetEconomicReturnChange_BillionAUD"})
    )

    merged = metric_wide.merge(ner_totals, on=["PriceType", "Price"], how="inner")
    merged["Pathway"] = merged["PriceType"].map({
        "CarbonPrice": "Carbon price drive",
        "BioPrice": "Biodiversity price drive",
    })

    missing_zero_rows = []
    for price_type, pathway in [
        ("CarbonPrice", "Carbon price drive"),
        ("BioPrice", "Biodiversity price drive"),
    ]:
        has_zero = (
            (merged["PriceType"] == price_type) &
            np.isclose(merged["Price"], 0.0)
        ).any()
        if not has_zero:
            missing_zero_rows.append({
                "PriceType": price_type,
                "Price": 0.0,
                "GHGAbatementChange_MtCO2e": 0.0,
                "BioContributionChange_MhaYr": 0.0,
                "NetEconomicReturnChange_BillionAUD": 0.0,
                "Pathway": pathway,
            })
    if missing_zero_rows:
        merged = pd.concat([pd.DataFrame(missing_zero_rows), merged], ignore_index=True)

    return merged.sort_values(["PriceType", "Price"]).reset_index(drop=True)


def plot_pathway(ax, df, y_column, color, label):
    sub = df.sort_values("Price")
    x = sub["NetEconomicReturnChange_BillionAUD"].to_numpy(dtype=float)
    y = sub[y_column].to_numpy(dtype=float)

    ax.plot(
        x,
        y,
        color=color,
        linewidth=1.8,
        marker="o",
        markersize=5.2,
        markerfacecolor=color,
        markeredgecolor="none",
        markeredgewidth=0.0,
        alpha=0.94,
        label=label,
        zorder=6,
    )


def zero_based_limits(values, min_pad):
    values = np.asarray(values, dtype=float)
    vmax = float(np.nanmax(values))
    pad = max(vmax * 0.08, min_pad)
    return 0.0, vmax + pad


df = load_pathway_data()
df.to_excel(RAW_DATA_PATH, sheet_name="ResponseData", index=False)
print(f"Raw data saved: {RAW_DATA_PATH}")

fig, axes = plt.subplots(1, 2, figsize=(12, 5.4))
panel_config = [
    (
        axes[0],
        "GHGAbatementChange_MtCO2e",
        r"GHG abatement difference (Mt CO$_2$e yr$^{-1}$)",
        "GHG abatement response to NER",
        8.0,
    ),
    (
        axes[1],
        "BioContributionChange_MhaYr",
        r"Biodiversity contribution difference (Mha yr$^{-1}$)",
        "Biodiversity response to NER",
        0.5,
    ),
]

pathway_config = [
    ("CarbonPrice", COLOR_CARBON, "Carbon price drive"),
    ("BioPrice", COLOR_BIO, "Biodiversity price drive"),
]

for ax, y_column, y_label, title, min_pad in panel_config:
    for price_type, color, label in pathway_config:
        plot_pathway(
            ax,
            df[df["PriceType"] == price_type],
            y_column,
            color,
            label,
        )

    ax.set_ylabel(y_label)
    ax.set_title(title, pad=8)
    style_box_axis(ax, linewidth=0.9)
    ax.set_ylim(*zero_based_limits(df[y_column], min_pad))
    apply_compact_ticks(ax, x_nbins=6, y_nbins=6)

x_limits = zero_based_limits(df["NetEconomicReturnChange_BillionAUD"], 15.0)
for ax in axes:
    ax.set_xlim(*x_limits)

axes[0].legend(
    loc="lower right",
    frameon=True,
    framealpha=0.92,
    borderpad=0.8,
)
axes[1].yaxis.tick_right()
axes[1].yaxis.set_label_position("right")

fig.supxlabel(r"Net economic return difference (Billion AU\$ yr$^{-1}$)", y=0.03)
plt.tight_layout(rect=[0, 0.05, 1, 1])
plt.subplots_adjust(wspace=0.08)

out_path = OUT_DIR / "06_Bio_GHG_NetEcon_Scatter.png"
fig.savefig(out_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"Saved: {out_path}")

summary = (
    df.loc[df.groupby("PriceType")["Price"].idxmax()]
    [["Pathway", "Price", "BioContributionChange_MhaYr", "GHGAbatementChange_MtCO2e", "NetEconomicReturnChange_BillionAUD"]]
)
print(summary.to_string(index=False))
