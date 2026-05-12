"""
MACC (Marginal Abatement Cost Curve): Reference → NZhigh

For each agricultural management (am) and non-agricultural (non-ag) solution,
computes:
  - GHG abatement (Mt CO2e yr-1): Reference_GHG - NZhigh_GHG  (positive = reduction)
  - Net cost (M AUD yr-1): profit loss + transition cost

Plots a classic MACC waterfall chart:
  - X-axis: cumulative GHG abatement (Mt CO2e yr-1)
  - Y-axis: marginal abatement cost (AU$/tCO2e yr-1)
  - Bar width  = abatement of each solution
  - Bars sorted from lowest to highest MAC
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tools.config as config

# ── Settings ───────────────────────────────────────────────────────────────────
YEAR = 2050
SCENARIO = "carbon_high_50"   # Reference → NZhigh, full adoption (top 50%)

INPUT_DIR = os.path.join(
    "..", "..", "..", "output", config.TASK_NAME, config.CARBON_PRICE_DIR, "1_draw_data"
)
OUTPUT_DIR = os.path.join(
    "..", "..", "..", "output", config.TASK_NAME, config.CARBON_PRICE_DIR, "3_Paper_figure"
)
os.makedirs(OUTPUT_DIR, exist_ok=True)

AM_SOLUTIONS    = config.COST_DICT["cost_am"]
NONAG_SOLUTIONS = config.COST_DICT["cost_non-ag"]

FS = 13
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial"],
    "font.size": FS,
    "axes.labelsize": FS,
    "axes.titlesize": FS + 1,
    "xtick.labelsize": FS - 1,
    "ytick.labelsize": FS - 1,
    "legend.fontsize": FS - 1,
    "mathtext.fontset": "stixsans",
})


# ── Helpers ────────────────────────────────────────────────────────────────────

def load_by_scenario(fname: str, scenario: str = SCENARIO) -> xr.DataArray:
    """Open a (scenario, Year, type) DataArray and select the given scenario."""
    path = os.path.join(INPUT_DIR, fname)
    da = xr.open_dataarray(path)
    avail = list(da.coords["scenario"].values)
    if scenario not in avail:
        raise KeyError(f"Scenario '{scenario}' not found in '{fname}'.\nAvailable: {avail}")
    return da.sel(scenario=scenario)


def at_year(da: xr.DataArray, year: int) -> pd.Series:
    """Select a single year and return as Series indexed by 'type'."""
    arr = da.sel(Year=year)
    return pd.Series(arr.values, index=arr.coords["type"].values.astype(str))


def read_colors(sheet: str) -> dict[str, str]:
    """Read solution → hex-color mapping (case-insensitive key matching)."""
    colors_path = os.path.join(os.path.dirname(__file__), "tools", "land use colors.xlsx")
    df = pd.read_excel(colors_path, sheet_name=sheet)
    mapping = {}
    for _, row in df.iterrows():
        name  = str(row.get("desc", row.get("desc_new", ""))).strip()
        color = str(row.get("color", "")).strip()
        if name and color and color.lower() != "nan":
            mapping[name.lower()] = color
    return mapping


def get_color(name: str, color_map: dict, fallback: str) -> str:
    return color_map.get(name.lower(), fallback)


# ── Load GHG abatement (Reference_GHG − NZhigh_GHG, already positive) ─────────
am_ghg_s    = at_year(load_by_scenario("xr_GHG_ag_management.nc"), YEAR)
nonag_ghg_s = at_year(load_by_scenario("xr_GHG_non_ag.nc"),        YEAR)

# ── Load net cost (profit_Reference − profit_NZhigh) ──────────────────────────
# Positive = policy costs money; negative = policy saves money (win-win)
am_cost_s    = at_year(load_by_scenario("xr_cost_agricultural_management.nc"), YEAR)
nonag_cost_s = at_year(load_by_scenario("xr_cost_non_ag.nc"),                  YEAR)

# Transition cost diff = NZhigh_transition − Reference_transition (extra CAPEX)
tran_cost_s  = at_year(
    load_by_scenario("xr_transition_cost_ag2non_ag_amortised_diff.nc"), YEAR
)

# Non-ag total cost = operating profit loss + transition cost
nonag_total_cost_s = nonag_cost_s.add(tran_cost_s, fill_value=0)

# ── Reindex to defined solution lists ─────────────────────────────────────────
am_ghg_s           = am_ghg_s.reindex(AM_SOLUTIONS,    fill_value=0.0)
am_cost_s          = am_cost_s.reindex(AM_SOLUTIONS,    fill_value=0.0)
nonag_ghg_s        = nonag_ghg_s.reindex(NONAG_SOLUTIONS, fill_value=0.0)
nonag_total_cost_s = nonag_total_cost_s.reindex(NONAG_SOLUTIONS, fill_value=0.0)

# ── Load colors ────────────────────────────────────────────────────────────────
am_colors    = read_colors("am")
nonag_colors = read_colors("non_ag")

# ── Compute MAC per solution ───────────────────────────────────────────────────
# abatement [Mt CO2e/yr], cost [M AUD/yr]
# MAC = M AUD / Mt CO2e = AU$/tCO2e  (unit check: M/M = 1)
records = []

for name in AM_SOLUTIONS:
    abatement = float(am_ghg_s[name])      # already positive = reduction
    cost      = float(am_cost_s[name])
    if abatement <= 0:
        continue
    records.append({
        "solution":     name,
        "category":     "AgMgt",
        "abatement_Mt": abatement,
        "cost_MAUD":    cost,
        "mac":          cost / abatement,
        "color":        get_color(name, am_colors, "#2196F3"),
    })

for name in NONAG_SOLUTIONS:
    abatement = float(nonag_ghg_s[name])
    cost      = float(nonag_total_cost_s[name])
    if abatement <= 0:
        continue
    records.append({
        "solution":     name,
        "category":     "Non-ag",
        "abatement_Mt": abatement,
        "cost_MAUD":    cost,
        "mac":          cost / abatement,
        "color":        get_color(name, nonag_colors, "#4CAF50"),
    })

df = pd.DataFrame(records)
if df.empty:
    raise RuntimeError("No solutions with positive abatement found.")

df = df.sort_values("mac").reset_index(drop=True)

# ── Print summary ──────────────────────────────────────────────────────────────
print(f"\nMACC — {SCENARIO}, year {YEAR}")
hdr = f"{'Solution':<40} {'Cat':<8} {'Abatement':>12} {'Cost':>12} {'MAC':>12}"
print(hdr)
print("-" * len(hdr))
for _, r in df.iterrows():
    print(
        f"{r['solution']:<40} {r['category']:<8}"
        f" {r['abatement_Mt']:>11.3f}"
        f" {r['cost_MAUD']:>11.2f}"
        f" {r['mac']:>11.1f}"
    )
print(f"{'Units':<48} {'Mt CO2e/yr':>12} {'M AUD/yr':>12} {'AU$/tCO2e':>12}\n")

# ── Broken-axis MACC plot ──────────────────────────────────────────────────────
# Three Y panels (top→bottom in figure order):
#   ax_top : MAC ∈ [550, 600]   — Biochar
#   ax_mid : MAC ∈ [  0, 300]   — main cluster  (largest panel)
#   ax_bot : MAC ∈ [-1500,-500] — win-win (negative cost)
Y_TOP = (550, 600)
Y_MID = (0, 300)
Y_BOT = (-1500, -500)

# Height ratios: mid gets the most visual space
fig = plt.figure(figsize=(16, 9))
gs  = gridspec.GridSpec(3, 1, height_ratios=[1, 5, 2], hspace=0.05, figure=fig)

ax_bot = fig.add_subplot(gs[2])
ax_mid = fig.add_subplot(gs[1], sharex=ax_bot)
ax_top = fig.add_subplot(gs[0], sharex=ax_bot)

axes  = [ax_top, ax_mid, ax_bot]
ylims = [Y_TOP,  Y_MID,  Y_BOT]

# ── Draw bars (each bar only in panels where it overlaps) ─────────────────────
x_left = 0.0
bar_meta = []   # (xcen, width, mac, solution, color, category)
for _, row in df.iterrows():
    w    = row["abatement_Mt"]
    h    = row["mac"]
    xcen = x_left + w / 2.0
    bar_ylo, bar_yhi = min(0.0, h), max(0.0, h)

    for ax, (ymin, ymax) in zip(axes, ylims):
        if bar_yhi > ymin and bar_ylo < ymax:      # bar overlaps this panel
            ax.bar(
                x=xcen, height=h, width=w,
                color=row["color"],
                edgecolor="white", linewidth=0.6,
                align="center", zorder=3,
            )
    bar_meta.append((xcen, w, h, row["solution"], row["color"], row["category"]))
    x_left += w

x_total = x_left

# ── Axis limits ────────────────────────────────────────────────────────────────
for ax, (ymin, ymax) in zip(axes, ylims):
    ax.set_xlim(0, x_total)
    ax.set_ylim(ymin, ymax)

# Zero baseline in mid panel (bottom edge)
ax_mid.axhline(0, color="black", linewidth=1.0, zorder=4)

# ── Tick intervals (different per panel, as requested) ────────────────────────
ax_top.yaxis.set_major_locator(plt.MultipleLocator(25))
ax_mid.yaxis.set_major_locator(plt.MultipleLocator(50))
ax_bot.yaxis.set_major_locator(plt.MultipleLocator(500))
ax_bot.xaxis.set_major_locator(plt.MultipleLocator(50))

# Hide x-tick labels on the two upper panels
ax_top.tick_params(labelbottom=False, bottom=False)
ax_mid.tick_params(labelbottom=False, bottom=False)
ax_bot.tick_params(direction="out", length=4)
for ax in axes:
    ax.tick_params(axis="y", direction="out", length=4)

# ── Spine styling for break ────────────────────────────────────────────────────
ax_top.spines["bottom"].set_visible(False)
ax_top.spines["top"].set_visible(False)
ax_top.spines["right"].set_visible(False)

ax_mid.spines["top"].set_visible(False)
ax_mid.spines["bottom"].set_visible(False)
ax_mid.spines["right"].set_visible(False)

ax_bot.spines["top"].set_visible(False)
ax_bot.spines["right"].set_visible(False)

# ── Break markers (diagonal slashes at panel junctions) ───────────────────────
def _break_marks(ax_upper, ax_lower, d: float = 0.018) -> None:
    """Draw double-diagonal break marks at the boundary between two axes."""
    kw = dict(color="k", clip_on=False, linewidth=1.2, zorder=10)
    # Bottom of upper axes (y=0 in transAxes)
    for x0 in (0.0, 1.0):
        ax_upper.plot([x0 - d, x0 + d], [-d, d],
                      transform=ax_upper.transAxes, **kw)
    # Top of lower axes (y=1 in transAxes)
    for x0 in (0.0, 1.0):
        ax_lower.plot([x0 - d, x0 + d], [1 - d, 1 + d],
                      transform=ax_lower.transAxes, **kw)

_break_marks(ax_top, ax_mid)
_break_marks(ax_mid, ax_bot)

# ── Solution name annotations ─────────────────────────────────────────────────
MID_RANGE = Y_MID[1] - Y_MID[0]   # 300 units
BOT_RANGE = abs(Y_BOT[1] - Y_BOT[0])  # 1000 units

for xcen, w, h, name, color, cat in bar_meta:
    if h >= Y_TOP[0]:                       # Biochar → annotate in top panel
        ax_top.text(xcen, h + 2, name, ha="center", va="bottom",
                    fontsize=8, rotation=90)
    elif h >= Y_MID[0]:                     # main cluster → mid panel
        pad = MID_RANGE * 0.03
        ax_mid.text(xcen, h + pad, name, ha="center", va="bottom",
                    fontsize=8, rotation=90)
    else:                                   # negative → bot panel
        pad = BOT_RANGE * 0.02
        ax_bot.text(xcen, h - pad, name, ha="center", va="top",
                    fontsize=8, rotation=90)

# ── Legend ────────────────────────────────────────────────────────────────────
legend_handles = []
for cat_label, color_map, fallback, cat_key in [
    ("Agricultural management", am_colors,    "#2196F3", "AgMgt"),
    ("Non-agricultural solution", nonag_colors, "#4CAF50", "Non-ag"),
]:
    solutions = AM_SOLUTIONS if cat_key == "AgMgt" else NONAG_SOLUTIONS
    in_df     = df[df["category"] == cat_key]
    for name in solutions:
        if name in in_df["solution"].values:
            legend_handles.append(
                mpatches.Patch(color=get_color(name, color_map, fallback), label=name)
            )

ax_top.legend(
    handles=legend_handles, frameon=False,
    loc="upper left", ncol=2,
    fontsize=8.5, handlelength=1.0, handleheight=0.8, columnspacing=0.8,
)

# ── Axis labels & title ───────────────────────────────────────────────────────
ax_bot.set_xlabel(r"Cumulative GHG abatement (Mt CO$_2$e yr$^{-1}$)", labelpad=8)
ax_mid.set_ylabel(r"Marginal abatement cost (AU\$/tCO$_2$e yr$^{-1}$)", labelpad=8)
ax_top.set_title(
    rf"MACC: Reference $\rightarrow$ NZ$_{{\rm high}}$ ({YEAR})", pad=10
)

# Extend y-label visually to all panels by centering in figure coordinates
ax_mid.yaxis.set_label_coords(-0.07, 0.5)

fig.subplots_adjust(left=0.10, right=0.97, top=0.93, bottom=0.09, hspace=0.05)

out_png = os.path.join(OUTPUT_DIR, f"MACC_NZhigh_{YEAR}.png")
fig.savefig(out_png, dpi=300, bbox_inches="tight")
plt.close()
print(f"Saved figure : {out_png}")

out_csv = os.path.join(OUTPUT_DIR, f"MACC_NZhigh_{YEAR}.csv")
try:
    df[["solution", "category", "abatement_Mt", "cost_MAUD", "mac"]].to_csv(out_csv, index=False)
    print(f"Saved data   : {out_csv}")
except PermissionError:
    print(f"[Warning] CSV locked (close it first): {out_csv}")
