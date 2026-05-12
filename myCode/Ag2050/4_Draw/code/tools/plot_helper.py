"""
plot_helper.py
Stacked-area chart utilities for LUTO2 Ag2050 analysis.
Drawing style mirrors myCode/carbonprice/code/tools/helper_plot.py.
All plot functions accept long-format DataFrames (year/scenario/category/value).
"""
import os
import math
import re
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import gridspec
from matplotlib.ticker import FuncFormatter, MaxNLocator
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

from tools.parameters import SCENARIO_LABELS


# ── Global style ──────────────────────────────────────────────────────────────

def set_plot_style(font_size=12, font_family='Arial'):
    """Apply seaborn darkgrid style + Arial font — same as carbonprice."""
    sns.set_theme(style="darkgrid")
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = [font_family]
    plt.rcParams['mathtext.fontset'] = 'stixsans'
    plt.rcParams.update({
        "xtick.bottom": True, "ytick.left": True,
        "xtick.top": False,   "ytick.right": False,
        "xtick.direction": "out", "ytick.direction": "out",
        "xtick.major.size": 4, "ytick.major.size": 4,
        "xtick.major.width": 1.2, "ytick.major.width": 1.2,
        "font.size": font_size,
        'axes.titlesize': font_size * 1.0,
        'axes.labelsize': font_size,
        'xtick.labelsize': font_size,
        'ytick.labelsize': font_size,
        'legend.fontsize': font_size,
        'pdf.fonttype': 42,
        'svg.fonttype': 'none',
    })


# ── Color helpers ─────────────────────────────────────────────────────────────

def get_colors(excel_file, sheet=None):
    """Return ordered {label: hex_color} from an Excel color-mapping sheet."""
    kw = {"sheet_name": sheet} if sheet else {}
    df = pd.read_excel(excel_file, **kw)
    if "desc_new" in df.columns:
        df["desc"] = df["desc_new"].fillna(df["desc"])
    df = df[df["color"].notna() & (df["color"].astype(str) != "-")]
    return dict(zip(df["desc"], df["color"]))


def _normalize(s):
    return re.sub(r"[-\s]", "", str(s)).lower()


def match_colors(df_long, colors):
    """Normalise category names to match *colors* keys (case/hyphen insensitive).
    Rows with no match are dropped."""
    norm_map = {_normalize(k): k for k in colors}
    df = df_long.copy()
    df["category"] = df["category"].map(lambda c: norm_map.get(_normalize(c)))
    return df.dropna(subset=["category"])


# ── Y-axis helpers ────────────────────────────────────────────────────────────

def get_y_axis_ticks(min_value, max_value, desired_ticks=5):
    """Return (min_v, max_v, ticks) with nice rounded tick values.
    Identical logic to carbonprice helper_plot.get_y_axis_ticks."""
    if min_value > 0 and max_value > 0:
        min_value = 0
    elif min_value < 0 and max_value < 0:
        max_value = 0

    span = max_value - min_value
    if span <= 0:
        return (0.0, 1.0, [0, 0.5, 1])

    ideal = span / (desired_ticks - 1)
    e     = math.floor(math.log10(ideal))
    base  = 10 ** e
    interval = min([1, 2, 5, 10], key=lambda x: abs(x - ideal / base)) * base

    min_tick = math.floor(min_value / interval) * interval
    max_tick = math.ceil(max_value  / interval) * interval
    tick_count = int(round((max_tick - min_tick) / interval)) + 1
    ticks = np.linspace(min_tick, max_tick, tick_count)

    if len(ticks) > desired_ticks + 1:
        scale = math.ceil((len(ticks) - 1) / (desired_ticks - 1))
        interval *= scale
        min_tick  = math.floor(min_value / interval) * interval
        max_tick  = math.ceil(max_value  / interval) * interval
        ticks     = np.linspace(min_tick, max_tick,
                                int(round((max_tick - min_tick) / interval)) + 1)

    if min_value < 0 < max_value and 0 not in ticks:
        ticks = np.insert(ticks, np.searchsorted(ticks, 0), 0)

    close = 0.3 * interval
    min_v, max_v = min_tick, max_tick
    if len(ticks) >= 2:
        if ticks[-1] != 0 and (max_value - ticks[-2]) < close and (ticks[-1] - max_value) > close:
            ticks = ticks[:-1]
            max_v = max_value + 0.1 * interval
        if ticks[0] != 0 and (ticks[1] - min_value) < close and (min_value - ticks[0]) > close:
            ticks = ticks[1:]
            min_v = min_value - 0.1 * interval
        elif abs(min_value) < interval:
            min_v = math.floor(min_value)

    if (abs(ticks[0]) < 1e-10 and abs(ticks[-1] - 100) < 1e-10) or (min_tick == 0 and max_tick == 100):
        ticks = np.array([0, 25, 50, 75, 100])

    return (float(min_v), float(max_v), ticks.tolist())


def calc_y_range(df_long, n_ticks=5):
    """Compute global y-range from stacked data, return (min_v, max_v, ticks)."""
    pos = df_long[df_long["value"] > 0].groupby(["year", "scenario"])["value"].sum()
    neg = df_long[df_long["value"] < 0].groupby(["year", "scenario"])["value"].sum()
    vmax = float(pos.max()) if not pos.empty else 0.0
    vmin = float(neg.min()) if not neg.empty else 0.0
    return get_y_axis_ticks(vmin, vmax, desired_ticks=n_ticks)


# ── Core stacked area (carbonprice style) ─────────────────────────────────────

def stacked_area_pos_neg(ax, df, colors=None, alpha=0.60,
                          title_name='', ylabel='', y_ticks_all=None,
                          total_name=None, dividing_line=0.25,
                          show_legend=False, n_col=1,
                          bbox_to_anchor=(0.5, -0.25), y_labelpad=6):
    """Stacked area chart with positive/negative separation and white dividers.
    Identical interface to carbonprice helper_plot.stacked_area_pos_neg."""
    if total_name and total_name in df.columns:
        df_stack = df.drop(columns=[total_name])
        ax.plot(df.index, df[total_name], linestyle='-', color='#404040',
                linewidth=3, label=total_name, zorder=20)
    else:
        df_stack = df

    default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    if colors is None:
        colors_list = default_colors
    elif isinstance(colors, dict):
        colors_list = [colors.get(col, default_colors[i % len(default_colors)])
                       for i, col in enumerate(df_stack.columns)]
    else:
        colors_list = colors

    cum_pos = np.zeros(len(df_stack))
    cum_neg = np.zeros(len(df_stack))
    for idx, col in enumerate(df_stack.columns):
        y   = df_stack[col].fillna(0).values
        pos = np.clip(y, 0, None)
        neg = np.clip(y, None, 0)
        clr = colors_list[idx] if idx < len(colors_list) else default_colors[idx % len(default_colors)]
        ax.fill_between(df_stack.index, cum_pos, cum_pos + pos,
                        facecolor=clr, alpha=alpha, linewidth=0, label=col)
        ax.fill_between(df_stack.index, cum_neg, cum_neg + neg,
                        facecolor=clr, alpha=alpha, linewidth=0)
        if dividing_line:
            ax.plot(df_stack.index, cum_pos + pos, color='white', linewidth=dividing_line, zorder=10)
            ax.plot(df_stack.index, cum_neg + neg, color='white', linewidth=dividing_line, zorder=10)
            if idx > 0:
                ax.plot(df_stack.index, cum_pos, color='white', linewidth=dividing_line, zorder=10)
                ax.plot(df_stack.index, cum_neg, color='white', linewidth=dividing_line, zorder=10)
        cum_pos += pos
        cum_neg += neg

    if y_ticks_all is not None:
        ax.set_ylim(y_ticks_all[0], y_ticks_all[1])
        ax.set_yticks(y_ticks_all[2])
        ticks = y_ticks_all[2]
        if ticks and abs(ticks[-1]) > 1000:
            ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:,.0f}'))
        elif len(ticks) > 1 and abs(ticks[1]) < 1 and ticks[1] != 0:
            ax.yaxis.set_major_formatter(
                FuncFormatter(lambda x, _: f'{int(x)}' if abs(x) < 1e-10 else f'{x:.1f}'))
    else:
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))

    ax.set_xlim(df.index.min(), df.index.max())
    ax.set_title(title_name, pad=6)
    ax.set_ylabel(ylabel, labelpad=y_labelpad)
    ax.set_xlabel('')
    ax.tick_params(direction='out')
    for spine in ax.spines.values():
        spine.set_color('black')
        spine.set_linewidth(1.2)

    if show_legend:
        handles, labels = ax.get_legend_handles_labels()
        seen = {}
        for h, l in zip(handles, labels):
            if l not in seen:
                seen[l] = h
        ax.legend(handles=list(seen.values()), labels=list(seen.keys()),
                  loc='upper left', bbox_to_anchor=bbox_to_anchor,
                  frameon=False, ncol=n_col,
                  handlelength=1.0, handleheight=1.0,
                  handletextpad=0.4, labelspacing=0.3)
    return ax


# ── Figure legend ─────────────────────────────────────────────────────────────

def draw_legend(ax, bbox_to_anchor=(0.5, -0.05), ncol=4, column_spacing=1.0):
    """Draw a figure-level legend, lines first then patches."""
    fig = ax.get_figure()
    handles, labels = ax.get_legend_handles_labels()
    lines   = [(h, l) for h, l in zip(handles, labels) if isinstance(h, Line2D)]
    patches = [(h, l) for h, l in zip(handles, labels) if not isinstance(h, Line2D)]
    ordered = lines + patches

    def clone(h):
        if isinstance(h, Line2D):
            return Line2D([0], [0], color=h.get_color(), linestyle=h.get_linestyle(),
                          linewidth=h.get_linewidth(), marker=h.get_marker(),
                          markersize=h.get_markersize())
        return Patch(facecolor=h.get_facecolor(), edgecolor='none')

    new_handles = [clone(h) for h, _ in ordered]
    new_labels  = [l for _, l in ordered]
    # deduplicate
    seen, dedup_h, dedup_l = set(), [], []
    for h, l in zip(new_handles, new_labels):
        if l not in seen:
            seen.add(l); dedup_h.append(h); dedup_l.append(l)

    fig.legend(dedup_h, dedup_l, loc='upper left', bbox_to_anchor=bbox_to_anchor,
               ncol=ncol, frameon=False,
               handlelength=1.0, handleheight=1.0,
               handletextpad=0.4, labelspacing=0.3,
               columnspacing=column_spacing)


# ── Main subplot function ─────────────────────────────────────────────────────

def plot_subplots(df_long, output_path, scenarios, colors,
                  targets=None, target_colors=('red',),
                  n_rows=2, n_cols=2, font_size=12,
                  x_range=(2010, 2050), y_range=None, x_ticks=5, y_ticks=None,
                  scenario_labels=None, legend_n_rows=2,
                  ylabel='', total_name=None, alpha=0.60):
    """Draw stacked-area subplots (one per scenario) and save as PNG.

    Parameters
    ----------
    df_long  : DataFrame — columns: year, scenario, category, value
    output_path : str — full output file path (with .png extension)
    scenarios : list — scenario keys in display order
    colors : dict — {category: hex_color}
    targets : DataFrame — overlay lines (columns: year, scenario, category, value)
    y_range  : (min_v, max_v, [ticks]) tuple from get_y_axis_ticks / calc_y_range,
               or None for auto
    """
    set_plot_style(font_size=font_size)

    df_long  = match_colors(df_long, colors)
    # Use x_range to define the year axis; fall back to data years if x_range not given
    if x_range is not None:
        years = list(range(x_range[0], x_range[1] + 1))
    else:
        years = sorted(df_long["year"].unique()) if not df_long.empty else [2010, 2050]
    cats     = list(colors.keys())
    clrs     = list(colors.values())

    if y_range is None:
        y_range = calc_y_range(df_long)  # returns (min, max, ticks)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    fig = plt.figure(figsize=(n_cols * 5.5, n_rows * 4.5))
    gs  = gridspec.GridSpec(n_rows, n_cols, figure=fig, hspace=0.20, wspace=0.08)
    fig.subplots_adjust(left=0.08, right=0.97, top=0.94, bottom=0.12)

    axes = []
    for r in range(n_rows):
        for c in range(n_cols):
            if r == 0 and c == 0:
                axes.append(fig.add_subplot(gs[r, c]))
            else:
                axes.append(fig.add_subplot(gs[r, c], sharey=axes[0]))

    labels_map = scenario_labels or SCENARIO_LABELS

    for i, scenario in enumerate(scenarios):
        ax   = axes[i]
        row  = i // n_cols
        col  = i % n_cols
        is_bottom = (row == n_rows - 1)
        is_left   = (col == 0)

        # pivot to wide format for this scenario
        df_s = df_long[df_long["scenario"] == scenario]
        df_wide = (
            df_s.pivot_table(index="year", columns="category", values="value", aggfunc="sum")
               .reindex(index=years)
               .reindex(columns=cats, fill_value=0)
        )

        title = labels_map.get(scenario, scenario).replace('\n', ' ')
        stacked_area_pos_neg(ax, df_wide,
                             colors=dict(zip(cats, clrs)),
                             alpha=alpha,
                             title_name=title,
                             ylabel=ylabel if is_left and row == n_rows // 2 else '',
                             y_ticks_all=y_range,
                             total_name=total_name,
                             show_legend=False)

        # overlay target/reference lines
        if targets is not None:
            t_s = targets[targets["scenario"] == scenario]
            for j, tcat in enumerate(sorted(t_s["category"].unique())):
                t_vals = t_s[t_s["category"] == tcat].set_index("year")["value"].reindex(years)
                clr = target_colors[j % len(target_colors)]
                ax.plot(years, t_vals.values, color=clr, linewidth=2,
                        linestyle='--', marker='o', markersize=3, label=tcat, zorder=5)

        # x-axis ticks: every x_ticks years
        tick_pos = [y for y in range(x_range[0], x_range[1] + 1, x_ticks)
                    if y in years]
        ax.set_xticks(tick_pos)
        if is_bottom:
            ax.tick_params(axis='x', labelrotation=45)
        else:
            ax.tick_params(axis='x', labelbottom=False)

        if not is_left:
            ax.tick_params(axis='y', labelleft=False)

        # hide leftmost x-label for non-left bottom subplots (avoids overlap)
        if is_bottom and col != 0:
            ax.figure.canvas.draw()
            xlabs = ax.get_xticklabels()
            if xlabs:
                xlabs[0].set_visible(False)

    # hide unused axes
    for ax in axes[len(scenarios):]:
        ax.set_visible(False)

    # figure-level legend from last visible subplot
    ncol = math.ceil(
        (len(cats) + (len(targets["category"].unique()) if targets is not None else 0))
        / legend_n_rows
    )
    draw_legend(axes[len(scenarios) - 1],
                bbox_to_anchor=(0.5 / n_cols, -0.04),
                ncol=ncol)

    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {output_path}")


# ── Map grid assembly ─────────────────────────────────────────────────────────

def assemble_map_grid(png_files, output_path, n_rows, n_cols, labels=None, dpi=200):
    """Assemble individual map PNGs into a matplotlib subplot grid."""
    from PIL import Image
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig, axs = plt.subplots(n_rows, n_cols,
                            figsize=(n_cols * 8, n_rows * 6),
                            gridspec_kw={"wspace": 0.02, "hspace": 0.05})
    fig.subplots_adjust(left=0.02, right=0.98, top=0.96, bottom=0.02)
    for i, (ax, png) in enumerate(zip(axs.flat, png_files)):
        ax.imshow(Image.open(png))
        ax.axis("off")
        if labels:
            ax.set_title(labels[i], fontsize=14, pad=1)
    for ax in list(axs.flat)[len(png_files):]:
        ax.set_visible(False)
    plt.tight_layout(pad=0.1)
    fig.savefig(output_path, bbox_inches="tight", dpi=dpi)
    plt.close(fig)
    print(f"Assembled: {output_path}")
