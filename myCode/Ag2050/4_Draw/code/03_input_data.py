"""03_input_data.py
Scenario-specific input parameter line plots for Ag2050 Methods section.

Panels (2 rows × 5 cols):
  Row 0 (no year labels, shared y-axis):
    Productivity – Beef | Productivity – Crop | Productivity – Dairy
    | Productivity – Sheep | Labour cost multiplier
  Row 1 (year labels, shared y-axis):
    Feedlot revenue ratio | Feedlot cost ratio | Feedlot GHG ratio
    | Feedlot water ratio | Legend
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
sns.set_theme(style="darkgrid")

sys.path.insert(0, os.path.dirname(__file__))
from _path_setup import *
from tools.parameters import OUTPUT_DIR

# ── Constants ──────────────────────────────────────────────────────────────────
INPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../input'))
CATTLE_CSV = os.path.abspath(os.path.join(
    os.path.dirname(__file__),
    '../../2_processed_data/cattle_percent_scenarios.csv'))

YEARS = list(range(2010, 2051))
# Years to show markers on the lines
MARK_YEARS = list(range(2010, 2051, 5))

# Scenario → scenario level mapping
SCENARIOS = {
    'AgS1': {'prod': 'high',      'ac': 'high',      'flc': 'FLC_multiplier_high',
             'cattle': 'High',    'feedlot': 'AgS1',
             'color': '#1f77b4',  'label': 'Regional Ag capitals'},
    'AgS2': {'prod': 'very_high', 'ac': 'very_high', 'flc': 'FLC_multiplier_very_high',
             'cattle': 'Very_High','feedlot': 'AgS2',
             'color': '#2ca02c',  'label': 'Landscape stewardship'},
    'AgS3': {'prod': 'medium',    'ac': 'medium',    'flc': 'FLC_multiplier_medium',
             'cattle': 'Medium',  'feedlot': 'AgS3',
             'color': '#ff7f0e',  'label': 'Climate survival'},
    'AgS4': {'prod': 'low',       'ac': 'low',       'flc': 'FLC_multiplier_low',
             'cattle': 'Low',     'feedlot': 'AgS4',
             'color': '#d62728',  'label': 'System decline'},
}

FONT_SIZE  = 11
TITLE_SIZE = 11
LINE_WIDTH = 1.5
MARKER_SIZE = 5

# ── ABARES productivity constants ───────────────────────────────────────────────
ABARES_FORECAST_FILE = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '../../2_processed_data/ABARES_productivity_forecast.xlsx'))
LAST_HIST_YEAR = 2023   # last year where all scenarios are identical

# ── Labour cost constants ────────────────────────────────────────────────────────
LABOUR_COST_FORECAST = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '../../2_processed_data/labour_cost_forecast.xlsx'))
LC_HIST_END  = 2024          # last real historical year in labour_cost_forecast.xlsx
LC_DEEP_BLUE = '#0b3d91'
# Level → colour (from tools.py)
LC_COLORS = {
    'Low':       '#ff7f0e',
    'Medium':    LC_DEEP_BLUE,
    'High':      '#d62728',
    'Very High': '#9467bd',
}
# Scenario → Plot_* column in ABARES_productivity_forecast.xlsx
ABARES_PLOT_COL   = {'AgS1': 'Plot_High', 'AgS2': 'Plot_Very_High', 'AgS3': 'Plot_Medium', 'AgS4': 'Plot_Low'}
# Productivity legend labels (level names, not scenario names)
ABARES_PROD_LABEL = {'AgS1': 'High', 'AgS2': 'Very High', 'AgS3': 'Medium', 'AgS4': 'Low'}

# Category label → ABARES Variable name
ABARES_VARS = {
    'Beef':  'Beef Productivity',
    'Crop':  'Crop Productivity',
    'Dairy': 'Dairy Productivity',
    'Sheep': 'Sheep Productivity',
}


# ── Data loaders ───────────────────────────────────────────────────────────────
def _load_productivity_abares():
    """Load ABARES productivity data (1988-2050) for Beef, Crop, Dairy, Sheep.

    Returns
    -------
    hist  : {cat: Series(year)}              historical observations 1988-LAST_HIST_YEAR (2010=1)
    fut   : {cat: {scen: Series(year)}}      scenario lines LAST_HIST_YEAR-2050 (2010=1)
    trend : {cat: Series(year)}              OLS linear trend 1988-2050 (2010=1)
    """
    df = pd.read_excel(ABARES_FORECAST_FILE, index_col=0)
    hist, fut, trend = {}, {}, {}
    for cat, var in ABARES_VARS.items():
        sub = df[df['Variable'] == var]
        # Historical scatter: Historical column up to LAST_HIST_YEAR
        hist[cat] = sub.loc[sub.index <= LAST_HIST_YEAR, 'Historical']
        # Scenario lines: Plot_* columns from LAST_HIST_YEAR to 2050
        fut_sub = sub.loc[sub.index >= LAST_HIST_YEAR]
        fut[cat] = {scen: fut_sub[col] for scen, col in ABARES_PLOT_COL.items()}
        # OLS fit line: Fitted column up to LAST_HIST_YEAR (same as tools.py)
        trend[cat] = sub.loc[sub.index <= LAST_HIST_YEAR, 'Fitted']
    return hist, fut, trend


def _load_labour_cost():
    """Load pre-computed labour cost forecast.

    Returns (hist, fut, trend) in the same format as _load_productivity_abares:
      hist  : Series(year)            historical scatter up to LC_HIST_END
      fut   : {scen: Series(year)}    Plot_* scenario lines from LC_HIST_END
      trend : Series(year)            OLS fitted line up to LC_HIST_END
    """
    df = pd.read_excel(LABOUR_COST_FORECAST, index_col=0)
    hist  = df.loc[df.index <= LC_HIST_END, 'Historical']
    trend = df.loc[df.index <= LC_HIST_END, 'Fitted']
    fut   = {
        scen: df.loc[df.index >= LC_HIST_END, col]
        for scen, col in ABARES_PLOT_COL.items()
    }
    return hist, fut, trend


FEEDLOT_HIST_END = 2024   # last year where all feedlot scenarios are identical

def _load_feedlots():
    """Returns dict of {label: (hist_s, fut_dict)}.

    hist_s   : Series(year)            common values 2010-FEEDLOT_HIST_END (black line)
    fut_dict : {scen: Series(year)}    scenario lines FEEDLOT_HIST_END-2050
    """
    files = {
        'Revenue ratio': 'Feedlots_revenue_ratio_from_ag2050.csv',
        'Cost ratio':    'Feedlots_cost_ratio_from_ag2050.csv',
        'GHG ratio':     'Feedlots_ghg_ratio_from_ag2050.csv',
        'Water ratio':   'Feedlots_water_ratio_from_ag2050.csv',
    }
    result = {}
    for label, fname in files.items():
        df = pd.read_csv(os.path.join(INPUT_DIR, fname), index_col='Year')
        df = df.loc[df.index.isin(YEARS)]
        # Historical: any column works (all identical up to FEEDLOT_HIST_END)
        hist_s = df.loc[df.index <= FEEDLOT_HIST_END, 'AgS1']
        fut_dict = {
            scen: df.loc[df.index >= FEEDLOT_HIST_END, info['feedlot']]
            for scen, info in SCENARIOS.items()
        }
        result[label] = (hist_s, fut_dict)
    return result


# ── Plotting helpers ───────────────────────────────────────────────────────────
def _style_ax(ax, title, xlim):
    ax.set_title(title, fontsize=TITLE_SIZE, pad=4)
    ax.tick_params(axis='both', labelsize=FONT_SIZE - 1, direction='out')
    ax.set_xlim(*xlim)
    for spine in ax.spines.values():
        spine.set_color('black')
        spine.set_linewidth(1.2)
    ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(5))


_AGS34_LABEL = (f"{SCENARIOS['AgS3']['label']} /\n"
                f"{SCENARIOS['AgS4']['label']}")

def _plot_feedlot(ax, hist_s, fut_dict, title=''):
    """Feedlot panel: black history + coloured scenario futures.
    AgS3 and AgS4 are identical → drawn as one combined line.
    """
    ax.plot(hist_s.index.values, hist_s.values,
            color='black', linewidth=LINE_WIDTH, zorder=3)
    for scen, s in fut_dict.items():
        if scen == 'AgS4':
            continue   # identical to AgS3, skip
        info  = SCENARIOS[scen]
        label = _AGS34_LABEL if scen == 'AgS3' else info['label']
        ax.plot(s.index.values, s.values,
                color=info['color'], linewidth=LINE_WIDTH,
                label=label, zorder=4)
    _style_ax(ax, title, (2010, 2050))


def _plot_productivity(ax, hist_s, fut_dict, trend_s, title):
    """ABARES-style productivity panel: scatter + OLS trend + scenario dashed lines.

    Parameters
    ----------
    hist_s   : Series(year)              historical observations 1988-LAST_HIST_YEAR
    fut_dict : {scen: Series(year)}      scenario lines LAST_HIST_YEAR-2050
    trend_s  : Series(year)              OLS linear trend 1988-2050
    """
    # OLS trend line (black, solid)
    ax.plot(trend_s.index, trend_s.values,
            color='black', linewidth=1.5, zorder=2)

    # Scenario lines from LAST_HIST_YEAR to 2050 (solid)
    for scen, s in fut_dict.items():
        info = SCENARIOS[scen]
        ax.plot(s.index.values, s.values, color=info['color'],
                linewidth=LINE_WIDTH, linestyle='-',
                label=ABARES_PROD_LABEL[scen], zorder=3)

    # Historical scatter dots (black)
    ax.scatter(hist_s.index, hist_s.values, s=8, c='black', zorder=5)

    ax.set_title(title, fontsize=TITLE_SIZE, pad=4)
    ax.tick_params(axis='both', labelsize=FONT_SIZE - 1, direction='out')
    ax.set_xlim(1988, 2050)
    for spine in ax.spines.values():
        spine.set_color('black')
        spine.set_linewidth(1.2)
    ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(5))



# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load data
    abares_hist, abares_fut, abares_trend = _load_productivity_abares()
    lc_hist, lc_fut, lc_trend             = _load_labour_cost()
    feedlots                              = _load_feedlots()

    # ── Figure: 2 rows × 6 cols (col 4 = narrow spacer before labour cost) ─────
    fig = plt.figure(figsize=(17, 7))
    gs  = gridspec.GridSpec(
        2, 6,
        figure=fig,
        width_ratios=[1, 1, 1, 1, 0.08, 1],
        hspace=0.22, wspace=0.12,
        top=0.93, bottom=0.10, left=0.06, right=0.98,
    )

    # Row 0: cols 0-3 = productivity (shared y), col 5 = labour cost (independent)
    ax_r0 = [fig.add_subplot(gs[0, 0])]
    for c in range(1, 4):
        ax_r0.append(fig.add_subplot(gs[0, c], sharey=ax_r0[0]))
    ax_r0.append(fig.add_subplot(gs[0, 5]))   # labour cost: independent

    # Row 1: cols 0-3 = feedlot panels (shared y), cols 4-5 = legend
    ax_r1 = [fig.add_subplot(gs[1, 0])]
    for c in range(1, 4):
        ax_r1.append(fig.add_subplot(gs[1, c], sharey=ax_r1[0]))

    # ── Row 0: plot ───────────────────────────────────────────────────────────
    prod_panels = [
        ('Beef',  '(a) Productivity and area cost – Beef'),
        ('Crop',  '(b) Productivity and area cost – Crop'),
        ('Dairy', '(c) Productivity and area cost – Dairy'),
        ('Sheep', '(d) Productivity and area cost – Sheep'),
    ]
    for ax, (cat, title) in zip(ax_r0[:4], prod_panels):
        _plot_productivity(ax, abares_hist[cat], abares_fut[cat],
                           abares_trend[cat], title)

    _plot_productivity(ax_r0[4], lc_hist, lc_fut, lc_trend, '(e) Labour cost')

    # Row 0 axis formatting
    ax_r0[0].set_ylim(bottom=0, top=ax_r0[0].get_ylim()[1] * 1.15)
    for ax in ax_r0[:4]:
        ax.set_xticks(range(1990, 2051, 10))
        ax.tick_params(labelleft=False)
        ax.set_xlabel('')
        ax.set_ylabel('')
    ax_r0[0].tick_params(labelleft=True)
    for ax in ax_r0[1:4]:
        ax.set_xticklabels(['', '2000', '2010', '2020', '2030', '2040', '2050'])
    ax_r0[4].set_xlim(2010, 2050)
    ax_r0[4].set_xticks(range(2010, 2051, 10))
    ax_r0[4].set_xlabel('')
    ax_r0[4].tick_params(labelleft=True)
    ax_r0[4].set_ylabel('')

    # Row 0 legend on labour cost panel (upper left)
    leg0_handles = [
        matplotlib.lines.Line2D([0], [0], color='black', linewidth=0,
                                 marker='o', markersize=MARKER_SIZE,
                                 label='Historical'),
        matplotlib.lines.Line2D([0], [0], color='black', linewidth=1.5,
                                 label='ETS mean'),
    ] + [
        matplotlib.lines.Line2D([0], [0], color=SCENARIOS[scen]['color'],
                                 linewidth=LINE_WIDTH, label=label)
        for scen, label in ABARES_PROD_LABEL.items()
    ]
    ax_r0[4].legend(handles=leg0_handles, loc='upper left',
                    fontsize=FONT_SIZE - 1, frameon=False, ncol=1)

    # ── Row 1: plot ───────────────────────────────────────────────────────────
    fl_labels  = list(feedlots.keys())
    # Strip "ratio" from display titles
    fl_display = [lbl.replace(' ratio', '') for lbl in fl_labels]
    panel_ids  = ['(f)', '(g)', '(h)', '(i)']
    for ax, raw_lbl, disp_lbl, pid in zip(ax_r1, fl_labels, fl_display, panel_ids):
        hist_s, fut_dict = feedlots[raw_lbl]
        _plot_feedlot(ax, hist_s, fut_dict,
                      title=f'{pid} Feedlot {disp_lbl.lower()}')

    # Row 1 axis formatting
    ax_r1[0].set_xticks(range(2010, 2051, 10))
    ax_r1[0].set_xticklabels(['2010', '2020', '2030', '2040', '2050'])
    ax_r1[0].set_xlabel('')
    ax_r1[0].tick_params(labelleft=True)
    for ax in ax_r1[1:]:
        ax.set_xticks(range(2010, 2051, 10))
        ax.set_xticklabels(['', '2020', '2030', '2040', '2050'])
        ax.set_xlabel('')
        ax.tick_params(labelleft=False)
        ax.set_ylabel('')

    # ── Shared y-axis label ────────────────────────────────────────────────────
    fig.supylabel('Multiplier', fontsize=FONT_SIZE, x=0.01)

    # ── Row 1 legend spanning cols 4-5 ────────────────────────────────────────
    ax_leg = fig.add_subplot(gs[1, 4:])
    ax_leg.axis('off')
    leg1_handles = [
        matplotlib.lines.Line2D([0], [0], color='black',
                                 linewidth=LINE_WIDTH, label='Historical'),
        matplotlib.lines.Line2D([0], [0], color=SCENARIOS['AgS1']['color'],
                                 linewidth=LINE_WIDTH, label=SCENARIOS['AgS1']['label']),
        matplotlib.lines.Line2D([0], [0], color=SCENARIOS['AgS2']['color'],
                                 linewidth=LINE_WIDTH, label=SCENARIOS['AgS2']['label']),
        matplotlib.lines.Line2D([0], [0], color=SCENARIOS['AgS3']['color'],
                                 linewidth=LINE_WIDTH, label=_AGS34_LABEL),
    ]
    ax_leg.legend(handles=leg1_handles, loc='center', fontsize=FONT_SIZE,
                  frameon=False, ncol=1)

    # ── Even y-ticks that naturally include 1.0 ───────────────────────────────
    fig.canvas.draw()
    for ax in ax_r0 + ax_r1:
        lo, hi = ax.get_ylim()
        for step in [0.05, 0.1, 0.2, 0.25, 0.5, 1.0]:
            n = round((np.floor(hi / step) - np.ceil(lo / step))) + 1
            if 4 <= n <= 6:
                ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(step))
                break

    # ── Save ──────────────────────────────────────────────────────────────────
    out = os.path.join(OUTPUT_DIR, '03_input_data.png')
    fig.savefig(out, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {out}')


if __name__ == '__main__':
    main()
