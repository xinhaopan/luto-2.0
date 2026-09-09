"""
31_Pairwise.py -- Fig. 5

Pairwise comparison of 2050 outcomes across the four agricultural futures.
A 2 x 2 panel of scatter plots, four points each (one per scenario):

    a  Agri-food production      vs  Biodiversity contribution-weighted score
    b  Agri-food production      vs  Net GHG emissions from land
    c  Net economic returns      vs  Biodiversity contribution-weighted score
    d  Net economic returns      vs  Net GHG emissions from land

Answers Referee 3 ("the text claims a gain in one indicator costs another, but
no figure shows a pairwise comparison") and Referee 2 ("the figures do not show
the trade-off").

Source
    EXCEL_DIR/04_trade_off_percent_threshold.xlsx, sheet 'summary'  -- ONLY this
    file.  05_scenario_synthesis_2050.xlsx holds a different, retired definition
    of the same quantities and must not be used here.

With four scenarios these panels support comparison between scenarios; they do
NOT support a fitted relationship, so no trend, regression or frontier line is
drawn -- by design, not by omission.

Output
    OUTPUT_DIR/31_Pairwise.svg   (svg.fonttype='none', editable in Inkscape)
"""

import _path_setup  # noqa: F401

import os

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from adjustText import adjust_text

from tools.parameters import EXCEL_DIR, OUTPUT_DIR
from tools.unit_text import mixed_xlabel, mixed_ylabel

# Relative paths in tools.parameters are anchored to this directory.
os.chdir(os.path.dirname(os.path.abspath(__file__)))

WORKBOOK = '04_trade_off_percent_threshold.xlsx'
SHEET = 'summary'
OUTPUT_NAME = '31_Pairwise.svg'

MM = 1.0 / 25.4
FIG_W_MM = 183.0        # Nature double-column width
FIG_H_MM = 164.0

# Marker geometry, shared by the scatter call and the label-gap calculation.
MARKER_SIZE = 26.0            # matplotlib `s`, i.e. area in points squared
MARKER_EDGE_WIDTH = 0.6
LABEL_CLEARANCE_PT = 2.5      # clear air between marker edge and label

# A second, non-colour cue for scenario identity.  The palette is fixed by the
# rest of the paper and cannot change, but colour alone fails the accessibility
# rule (the Climate Survival / Landscape Stewardship pair is orange vs green,
# and the figure has to survive greyscale), so shape carries identity too.
# Areas are tuned per shape so all four read as the same visual weight.
SCENARIO_MARKERS = {
    'Regional Ag Capitals':  ('o', 1.00),
    'Landscape Stewardship': ('s', 0.86),
    'Climate Survival':      ('^', 1.18),
    'System Decline':        ('D', 0.86),
}

# Taken from 04_Trade_off_percent_threshold.py (Fig. 4). Fixed for the whole
# paper -- do not re-pick.
SCENARIO_COLORS = {
    'Regional Ag Capitals':  '#2D688F',
    'Landscape Stewardship': '#2F8F5B',
    'Climate Survival':      '#D9872C',
    'System Decline':        '#B84A4A',
}

FOOD_LABEL = 'Agri-food production (Mt yr⁻¹)'
NER_LABEL = 'Net economic returns (billion AU$ yr⁻¹)'
BIO_LABEL = 'Biodiversity contribution-weighted score (Mha)'
GHG_LABEL = 'Net GHG emissions from land (Mt CO₂e yr⁻¹)'

# (letter, x column, y column, x label, y label, y-axis 2010 baseline column)
# The 2010 baseline is drawn only where it is common to all four scenarios:
# GHG (68.10) and biodiversity (90.13).  Food and NER differ by scenario, so no
# baseline line is drawn on those axes.
PANELS = [
    ('a', 'food_2050_mt', 'biodiversity_2050_mha', FOOD_LABEL, BIO_LABEL, 'biodiversity_2010_mha'),
    ('b', 'food_2050_mt', 'ghg_2050_mtco2e',       FOOD_LABEL, GHG_LABEL, 'ghg_2010_mtco2e'),
    ('c', 'ner_2050_baud', 'biodiversity_2050_mha', NER_LABEL, BIO_LABEL, 'biodiversity_2010_mha'),
    ('d', 'ner_2050_baud', 'ghg_2050_mtco2e',       NER_LABEL, GHG_LABEL, 'ghg_2010_mtco2e'),
]

REQUIRED_COLUMNS = [
    'scenario_label',
    'food_2050_mt', 'ner_2050_baud', 'ghg_2050_mtco2e', 'biodiversity_2050_mha',
    'food_2010_mt', 'ner_2010_baud', 'ghg_2010_mtco2e', 'biodiversity_2010_mha',
]

# Values the summary sheet is expected to carry, from the manuscript. Checked on
# load so a silently regenerated workbook cannot change the figure unnoticed.
EXPECTED_2050 = {
    'Regional Ag Capitals':  dict(food=240.36, ner=43.36, ghg=67.12, bio=89.39),
    'Landscape Stewardship': dict(food=213.57, ner=40.72, ghg=-73.72, bio=97.87),
    'Climate Survival':      dict(food=195.84, ner=37.96, ghg=101.47, bio=88.47),
    'System Decline':        dict(food=152.23, ner=31.23, ghg=100.51, bio=88.84),
}


def load_summary() -> pd.DataFrame:
    path = os.path.join(EXCEL_DIR, WORKBOOK)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f'{path} not found. Run 04_Trade_off_percent_threshold.py first.'
        )
    data = pd.read_excel(path, sheet_name=SHEET)

    missing = [c for c in REQUIRED_COLUMNS if c not in data.columns]
    if missing:
        raise ValueError(f'{WORKBOOK}[{SHEET}] is missing columns: {missing}')

    unknown = set(data['scenario_label']) - set(SCENARIO_COLORS)
    if unknown:
        raise ValueError(f'Unrecognised scenario labels: {sorted(unknown)}')

    _verify(data)
    return data


def _verify(data: pd.DataFrame) -> None:
    """Fail loudly if the workbook no longer holds the published numbers."""
    col = {'food': 'food_2050_mt', 'ner': 'ner_2050_baud',
           'ghg': 'ghg_2050_mtco2e', 'bio': 'biodiversity_2050_mha'}
    problems = []
    for _, row in data.iterrows():
        expected = EXPECTED_2050.get(row['scenario_label'])
        if expected is None:
            continue
        for key, want in expected.items():
            got = float(row[col[key]])
            if abs(got - want) > 0.01:
                problems.append(f"  {row['scenario_label']:<24} {key:<4} "
                                f"expected {want:>9.2f}, workbook has {got:>9.2f}")
    if problems:
        raise ValueError('Summary sheet does not match the published values:\n'
                         + '\n'.join(problems))

    # The 2010 baselines drawn as lines must be common to all four scenarios.
    for column in ('ghg_2010_mtco2e', 'biodiversity_2010_mha'):
        if data[column].round(6).nunique() != 1:
            raise ValueError(f'{column} is not common to all scenarios; '
                             'the baseline line would be misleading.')
    print('  summary sheet matches the published 2050 values')


def _label_sides(y: np.ndarray, ax) -> list:
    """Put each label directly above or below its own point.

    Every label is horizontally centred on its marker, so the only choice left
    is the vertical side: away from the other three points, which is what keeps
    a label from landing inside the cloud.  adjustText then resolves any
    residual label-on-label overlap, vertically only.
    """
    y0, y1 = ax.get_ylim()
    fy = (y - y0) / (y1 - y0)
    return [1.0 if fy[i] >= fy[[j for j in range(len(y)) if j != i]].mean() else -1.0
            for i in range(len(y))]


def draw_panel(ax, data: pd.DataFrame, letter, x_col, y_col, x_label, y_label,
               baseline_col) -> None:
    # Panel letters were dropped on request; the caption refers to the panels by
    # position instead.  `letter` is kept in PANELS so the reading order stays
    # documented in one place.
    del baseline_col, letter
    x = data[x_col].to_numpy(dtype=float)
    y = data[y_col].to_numpy(dtype=float)
    labels = data['scenario_label'].tolist()
    colors = [SCENARIO_COLORS[s] for s in labels]

    # Room for the direct labels.
    ax.set_xlim(*_padded(x, 0.34))
    ax.set_ylim(*_padded(y, 0.24))

    # No reference lines: the panels carry the four points only.
    # NO fitted line, trend line or frontier either -- four points cannot
    # support one.
    for xi, yi, label, color in zip(x, y, labels, colors):
        marker, area_scale = SCENARIO_MARKERS[label]
        ax.scatter(xi, yi, s=MARKER_SIZE * area_scale, c=color, marker=marker,
                   edgecolors='white', linewidths=MARKER_EDGE_WIDTH,
                   zorder=3, clip_on=False)

    # Direct labels instead of a legend: centred straight above or below the
    # marker, kept tight against it.
    #
    # The gap is computed in POINTS, from the marker's own radius, and only then
    # converted to data units.  Expressing it as a fraction of the y range (the
    # obvious shortcut) is wrong: it has no relation to how big the marker
    # actually is, so on a panel with a narrow y range the label lands on top of
    # its own point -- which is exactly what happened in panels c and d.
    y0, y1 = ax.get_ylim()
    largest = MARKER_SIZE * max(scale for _m, scale in SCENARIO_MARKERS.values())
    marker_radius_pt = (largest / np.pi) ** 0.5 + MARKER_EDGE_WIDTH
    gap_pt = marker_radius_pt + LABEL_CLEARANCE_PT
    axes_height_pt = ax.get_window_extent().height * 72.0 / ax.figure.dpi
    y_gap = gap_pt * (y1 - y0) / axes_height_pt

    texts = []
    for side, xi, yi, label, color in zip(
            _label_sides(y, ax), x, y, labels, colors):
        texts.append(ax.text(
            xi, yi + side * y_gap, label,
            fontsize=6, color=color, ha='center',
            va='bottom' if side > 0 else 'top', zorder=4,
        ))

    # Vertical-only nudging: a label must stay centred over its own point.
    adjust_text(
        texts,
        x=list(x), y=list(y), ax=ax,
        avoid_self=True,
        only_move={'text': 'y', 'static': 'y', 'explode': 'y', 'pull': 'y'},
        expand=(1.05, 1.6),
        force_text=(0.0, 0.7),      # label <-> label
        force_static=(0.0, 1.3),    # label <-> data point: must dominate
        force_pull=(0.0, 0.002),    # near-zero: the default pull drags labels
                                    # back onto their own marker
        min_arrow_len=1e9,          # never draw leader arrows
        time_lim=3.0,
    )

    # 4-6 whole-number ticks per axis: no decimals, no crowding.
    ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=5, integer=True, min_n_ticks=4))
    ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=5, integer=True, min_n_ticks=4))
    ax.xaxis.set_major_formatter(mticker.StrMethodFormatter('{x:.0f}'))
    ax.yaxis.set_major_formatter(mticker.StrMethodFormatter('{x:.0f}'))

    mixed_xlabel(ax, x_label, fontsize=7, pad=15.0)
    mixed_ylabel(ax, y_label, fontsize=7, pad=25.0)
    ax.tick_params(axis='both', labelsize=7, width=0.8, length=2.5, pad=1.5)
    # Spec: no top or right spine.
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_linewidth(0.8)
        ax.spines[side].set_color('#4D4D4D')



def _padded(values: np.ndarray, fraction: float):
    lo, hi = float(np.min(values)), float(np.max(values))
    span = hi - lo
    pad = span * fraction if span > 0 else max(abs(hi), 1.0) * fraction
    return lo - pad, hi + pad


def main() -> None:
    data = load_summary()

    # Straight from the Nature figure spec (static/fragments/backend/python.md).
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans', 'sans-serif'],
        'svg.fonttype': 'none',
        'pdf.fonttype': 42,
        'font.size': 7,
        'axes.spines.right': False,
        'axes.spines.top': False,
        'axes.linewidth': 0.8,
        'legend.frameon': False,
        'xtick.major.width': 0.8,
        'ytick.major.width': 0.8,
    })

    fig, axes = plt.subplots(
        2, 2, figsize=(FIG_W_MM * MM, FIG_H_MM * MM),
    )
    for ax, panel in zip(axes.ravel(), PANELS):
        draw_panel(ax, data, *panel)

    fig.subplots_adjust(left=0.105, right=0.985, bottom=0.075, top=0.965,
                        wspace=0.42, hspace=0.34)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, OUTPUT_NAME)
    fig.savefig(out_path, format='svg', facecolor='white')
    plt.close(fig)
    print(f'  wrote {out_path}')


if __name__ == '__main__':
    main()
