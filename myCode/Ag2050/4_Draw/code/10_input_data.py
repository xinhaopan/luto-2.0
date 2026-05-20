"""03_input_data.py
Scenario-specific input parameter plots for the Ag2050 Methods section.
"""

import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from statsmodels.tsa.holtwinters import ExponentialSmoothing

sns.set_theme(style="darkgrid")
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['svg.fonttype'] = 'none'

sys.path.insert(0, os.path.dirname(__file__))
from _path_setup import *  # noqa: F401,F403
from tools.parameters import OUTPUT_DIR


INPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../input'))

ABARES_FORECAST_FILE = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '../../2_processed_data/ABARES_productivity_forecast.xlsx'
))
LABOUR_COST_FORECAST = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '../../2_processed_data/labour_cost_forecast.xlsx'
))

YEARS = list(range(2010, 2051))
LAST_HIST_YEAR = 2023
LC_HIST_END = 2024
FEEDLOT_HIST_END = 2024

SCENARIOS = {
    'AgS1': {
        'prod': 'high',
        'ac': 'high',
        'flc': 'FLC_multiplier_high',
        'cattle': 'High',
        'feedlot': 'AgS1',
        'color': '#1f77b4',
        'label': 'Regional Ag capitals',
    },
    'AgS2': {
        'prod': 'very_high',
        'ac': 'very_high',
        'flc': 'FLC_multiplier_very_high',
        'cattle': 'Very_High',
        'feedlot': 'AgS2',
        'color': '#2ca02c',
        'label': 'Landscape stewardship',
    },
    'AgS3': {
        'prod': 'medium',
        'ac': 'medium',
        'flc': 'FLC_multiplier_medium',
        'cattle': 'Medium',
        'feedlot': 'AgS3',
        'color': '#ff7f0e',
        'label': 'Climate survival',
    },
    'AgS4': {
        'prod': 'low',
        'ac': 'low',
        'flc': 'FLC_multiplier_low',
        'cattle': 'Low',
        'feedlot': 'AgS4',
        'color': '#d62728',
        'label': 'System decline',
    },
}

ABARES_PLOT_COL = {
    'AgS1': 'Plot_High',
    'AgS2': 'Plot_Very_High',
    'AgS3': 'Plot_Medium',
    'AgS4': 'Plot_Low',
}
ABARES_PROD_LABEL = {
    'AgS1': 'High',
    'AgS2': 'Very High',
    'AgS3': 'Medium',
    'AgS4': 'Low',
}
ABARES_VARS = {
    'Beef': 'Beef Productivity',
    'Crop': 'Crop Productivity',
    'Dairy': 'Dairy Productivity',
    'Sheep': 'Sheep Productivity',
}
FEEDLOT_LEVELS = {
    'High': {'source': 'AgS1', 'color': SCENARIOS['AgS1']['color']},
    'Medium': {'source': 'AgS2', 'color': SCENARIOS['AgS3']['color']},
    'Low': {'source': 'AgS3', 'color': SCENARIOS['AgS4']['color']},
}

TEXT_SIZE = 25
LINE_WIDTH = 1.8
MARKER_SIZE = 8
HIST_SCATTER_SIZE = 26


def _fit_ets_series(hist_s: pd.Series) -> pd.Series:
    """Fit an ETS line to the historical series and return fitted values."""
    hist_s = hist_s.dropna().astype(float)
    try:
        work_s = pd.Series(hist_s.to_numpy(), index=pd.RangeIndex(len(hist_s)), dtype=float)
        fitted = ExponentialSmoothing(
            work_s,
            trend='add',
            damped_trend=True,
            seasonal=None,
            initialization_method='estimated',
        ).fit(optimized=True).fittedvalues
        return pd.Series(fitted.to_numpy(), index=hist_s.index)
    except Exception:
        return hist_s.ewm(span=min(5, max(2, len(hist_s) // 3)), adjust=False).mean()


def _load_productivity_abares():
    """Load ABARES productivity data for Beef, Crop, Dairy, and Sheep."""
    df = pd.read_excel(ABARES_FORECAST_FILE, sheet_name='fitted_scenarios', index_col=0)
    hist, fut, trend = {}, {}, {}
    for cat, var in ABARES_VARS.items():
        sub = df[df['Variable'] == var]
        hist[cat] = sub.loc[sub.index <= LAST_HIST_YEAR, 'Historical']
        fut_sub = sub.loc[sub.index >= LAST_HIST_YEAR]
        fut[cat] = {scen: fut_sub[col] for scen, col in ABARES_PLOT_COL.items()}
        trend[cat] = sub.loc[sub.index <= LAST_HIST_YEAR, 'Fitted']
    return hist, fut, trend


def _load_labour_cost():
    """Load labour cost history, forecast scenarios, and fitted line."""
    df = pd.read_excel(LABOUR_COST_FORECAST, sheet_name='fitted_scenarios', index_col=0)
    hist = df.loc[df.index <= LC_HIST_END, 'Historical']
    trend = df.loc[df.index <= LC_HIST_END, 'Fitted']
    fut = {
        scen: df.loc[df.index >= LC_HIST_END, col]
        for scen, col in ABARES_PLOT_COL.items()
    }
    return hist, fut, trend


def _load_feedlots():
    """Load feedlot ratios with historical series, ETS fit, and three futures."""
    files = {
        'Revenue ratio': 'Feedlots_revenue_ratio_from_ag2050.csv',
        'Cost ratio': 'Feedlots_cost_ratio_from_ag2050.csv',
        'GHG ratio': 'Feedlots_ghg_ratio_from_ag2050.csv',
        'Water ratio': 'Feedlots_water_ratio_from_ag2050.csv',
    }
    result = {}
    for label, fname in files.items():
        df = pd.read_csv(os.path.join(INPUT_DIR, fname), index_col='Year')
        df = df.loc[df.index.isin(YEARS)]
        hist_s = df.loc[df.index <= FEEDLOT_HIST_END, 'AgS1']
        fut_dict = {
            level: df.loc[df.index >= FEEDLOT_HIST_END, info['source']]
            for level, info in FEEDLOT_LEVELS.items()
        }
        trend_s = _fit_ets_series(hist_s)
        result[label] = (hist_s, fut_dict, trend_s)
    return result


def _style_ax(ax, title, xlim):
    ax.set_title(title, fontsize=TEXT_SIZE, pad=8)
    ax.tick_params(axis='both', labelsize=TEXT_SIZE, direction='out')
    ax.set_xlim(*xlim)
    for spine in ax.spines.values():
        spine.set_color('black')
        spine.set_linewidth(1.2)
    ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(5))


def _plot_productivity(ax, hist_s, fut_dict, trend_s, title, xlim):
    ax.plot(trend_s.index, trend_s.values, color='black', linewidth=LINE_WIDTH, zorder=2)
    for scen, s in fut_dict.items():
        ax.plot(
            s.index.values,
            s.values,
            color=SCENARIOS[scen]['color'],
            linewidth=LINE_WIDTH,
            linestyle='-',
            label=ABARES_PROD_LABEL[scen],
            zorder=3,
        )
    ax.scatter(hist_s.index, hist_s.values, s=HIST_SCATTER_SIZE, c='black', zorder=5)
    _style_ax(ax, title, xlim)


def _plot_feedlot(ax, hist_s, fut_dict, trend_s, title):
    ax.plot(trend_s.index.values, trend_s.values, color='black', linewidth=LINE_WIDTH, zorder=2)
    for level, s in fut_dict.items():
        ax.plot(
            s.index.values,
            s.values,
            color=FEEDLOT_LEVELS[level]['color'],
            linewidth=LINE_WIDTH,
            linestyle='-',
            label=level,
            zorder=3,
        )
    ax.scatter(hist_s.index, hist_s.values, s=HIST_SCATTER_SIZE, c='black', zorder=5)
    _style_ax(ax, title, (2010, 2050))


def _set_prod_ticks(ax):
    ax.set_xticks(range(1990, 2051, 10))
    ax.set_xticklabels(['1990', '2000', '2010', '2020', '2030', '2040', '2050'])


def _set_recent_ticks(ax):
    ax.set_xticks(range(2010, 2051, 10))
    ax.set_xticklabels(['2010', '2020', '2030', '2040', '2050'])


def _apply_row_shared_y(ax_left, ax_right):
    ax_left.tick_params(labelleft=True)
    ax_right.tick_params(labelleft=False)
    ax_left.set_ylabel('')
    ax_right.set_ylabel('')


def _legend_handles():
    return [
        matplotlib.lines.Line2D(
            [0], [0], color='black', linewidth=0,
            marker='o', markersize=MARKER_SIZE, label='Historical'
        ),
        matplotlib.lines.Line2D(
            [0], [0], color='black', linewidth=LINE_WIDTH, label='ETS mean'
        ),
        matplotlib.lines.Line2D(
            [0], [0], color=SCENARIOS['AgS1']['color'], linewidth=LINE_WIDTH, label='High'
        ),
        matplotlib.lines.Line2D(
            [0], [0], color=SCENARIOS['AgS2']['color'], linewidth=LINE_WIDTH, label='Very High'
        ),
        matplotlib.lines.Line2D(
            [0], [0], color=SCENARIOS['AgS3']['color'], linewidth=LINE_WIDTH, label='Medium'
        ),
        matplotlib.lines.Line2D(
            [0], [0], color=SCENARIOS['AgS4']['color'], linewidth=LINE_WIDTH, label='Low'
        ),
    ]


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    abares_hist, abares_fut, abares_trend = _load_productivity_abares()
    lc_hist, lc_fut, lc_trend = _load_labour_cost()
    feedlots = _load_feedlots()

    fig = plt.figure(figsize=(18, 30))
    gs = gridspec.GridSpec(
        5, 2,
        figure=fig,
        hspace=0.42,
        wspace=0.22,
        top=0.97,
        bottom=0.05,
        left=0.08,
        right=0.98,
    )

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1], sharey=ax_a)
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1], sharey=ax_c)
    ax_e = fig.add_subplot(gs[2, 0])
    ax_leg = fig.add_subplot(gs[2, 1])
    ax_f = fig.add_subplot(gs[3, 0])
    ax_g = fig.add_subplot(gs[3, 1], sharey=ax_f)
    ax_h = fig.add_subplot(gs[4, 0])
    ax_i = fig.add_subplot(gs[4, 1], sharey=ax_h)

    _plot_productivity(
        ax_a, abares_hist['Beef'], abares_fut['Beef'], abares_trend['Beef'],
        '(a) Productivity and area cost – Beef', (1988, 2050)
    )
    _plot_productivity(
        ax_b, abares_hist['Crop'], abares_fut['Crop'], abares_trend['Crop'],
        '(b) Productivity and area cost – Crop', (1988, 2050)
    )
    _plot_productivity(
        ax_c, abares_hist['Dairy'], abares_fut['Dairy'], abares_trend['Dairy'],
        '(c) Productivity and area cost – Dairy', (1988, 2050)
    )
    _plot_productivity(
        ax_d, abares_hist['Sheep'], abares_fut['Sheep'], abares_trend['Sheep'],
        '(d) Productivity and area cost – Sheep', (1988, 2050)
    )
    _plot_productivity(ax_e, lc_hist, lc_fut, lc_trend, '(e) Labour cost', (2010, 2050))

    _apply_row_shared_y(ax_a, ax_b)
    _apply_row_shared_y(ax_c, ax_d)

    row0_top = max(ax_a.get_ylim()[1], ax_b.get_ylim()[1]) * 1.10
    ax_a.set_ylim(bottom=0, top=row0_top)
    row1_top = max(ax_c.get_ylim()[1], ax_d.get_ylim()[1]) * 1.10
    ax_c.set_ylim(bottom=0, top=row1_top)
    ax_e.set_ylim(bottom=0, top=ax_e.get_ylim()[1] * 1.10)

    for ax in (ax_a, ax_b, ax_c, ax_d):
        _set_prod_ticks(ax)
    _set_recent_ticks(ax_e)

    ax_leg.axis('off')
    ax_leg.legend(
        handles=_legend_handles(),
        loc='center left',
        fontsize=TEXT_SIZE,
        frameon=False,
        ncol=1,
        markerscale=1.15,
        handlelength=2.0,
        labelspacing=0.65,
        borderaxespad=0.0,
    )

    feedlot_panels = [
        (ax_f, 'Revenue ratio', '(f) Feedlot revenue'),
        (ax_g, 'Cost ratio', '(g) Feedlot cost'),
        (ax_h, 'GHG ratio', '(h) Feedlot GHG'),
        (ax_i, 'Water ratio', '(i) Feedlot water'),
    ]
    for ax, raw_label, title in feedlot_panels:
        hist_s, fut_dict, trend_s = feedlots[raw_label]
        _plot_feedlot(ax, hist_s, fut_dict, trend_s, title)
        _set_recent_ticks(ax)

    _apply_row_shared_y(ax_f, ax_g)
    _apply_row_shared_y(ax_h, ax_i)

    fig.supylabel('Multiplier', fontsize=TEXT_SIZE, x=0.03)

    fig.canvas.draw()
    for ax in [ax_a, ax_b, ax_c, ax_d, ax_e, ax_f, ax_g, ax_h, ax_i]:
        lo, hi = ax.get_ylim()
        for step in [0.05, 0.1, 0.2, 0.25, 0.5, 1.0]:
            n = round((np.floor(hi / step) - np.ceil(lo / step))) + 1
            if 4 <= n <= 6:
                ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(step))
                break

    out = os.path.join(OUTPUT_DIR, '10_input_data.svg')
    fig.savefig(out, dpi=600, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {out}')


if __name__ == '__main__':
    main()
