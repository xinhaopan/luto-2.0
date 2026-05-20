"""Historical-only version of the Ag2050 input-data figure.

This reproduces the input-data figure without future scenario projections.
Each panel shows only historical observations and the fitted ETS line over the
historical period.
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

sns.set_theme(style='darkgrid')
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

LAST_HIST_YEAR = 2023
LC_HIST_END = 2024
FEEDLOT_HIST_END = 2024

ABARES_VARS = {
    'Beef': 'Beef Productivity',
    'Crop': 'Crop Productivity',
    'Dairy': 'Dairy Productivity',
    'Sheep': 'Sheep Productivity',
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
    """Load historical ABARES productivity and fitted values."""
    df = pd.read_excel(ABARES_FORECAST_FILE, sheet_name='fitted_scenarios', index_col=0)
    hist, trend = {}, {}
    for cat, var in ABARES_VARS.items():
        sub = df[df['Variable'] == var]
        hist[cat] = sub.loc[sub.index <= LAST_HIST_YEAR, 'Historical']
        trend[cat] = sub.loc[sub.index <= LAST_HIST_YEAR, 'Fitted']
    return hist, trend


def _load_labour_cost():
    """Load historical labour cost and fitted values."""
    df = pd.read_excel(LABOUR_COST_FORECAST, sheet_name='fitted_scenarios', index_col=0)
    hist = df.loc[df.index <= LC_HIST_END, 'Historical']
    trend = df.loc[df.index <= LC_HIST_END, 'Fitted']
    return hist, trend


def _load_feedlots():
    """Load historical feedlot ratios and fit an ETS line for each ratio."""
    files = {
        'Revenue ratio': 'Feedlots_revenue_ratio_from_ag2050.csv',
        'Cost ratio': 'Feedlots_cost_ratio_from_ag2050.csv',
        'GHG ratio': 'Feedlots_ghg_ratio_from_ag2050.csv',
        'Water ratio': 'Feedlots_water_ratio_from_ag2050.csv',
    }
    result = {}
    for label, fname in files.items():
        df = pd.read_csv(os.path.join(INPUT_DIR, fname), index_col='Year')
        hist_s = df.loc[df.index <= FEEDLOT_HIST_END, 'AgS1']
        trend_s = _fit_ets_series(hist_s)
        result[label] = (hist_s, trend_s)
    return result


def _style_ax(ax, title, xlim):
    ax.set_title(title, fontsize=TEXT_SIZE, pad=8)
    ax.tick_params(axis='both', labelsize=TEXT_SIZE, direction='out')
    ax.set_xlim(*xlim)
    for spine in ax.spines.values():
        spine.set_color('black')
        spine.set_linewidth(1.2)
    ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(5))


def _plot_historical(ax, hist_s, trend_s, title, xlim):
    ax.plot(trend_s.index, trend_s.values, color='black', linewidth=LINE_WIDTH, zorder=2)
    ax.scatter(hist_s.index, hist_s.values, s=HIST_SCATTER_SIZE, c='black', zorder=5)
    _style_ax(ax, title, xlim)


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
    ]


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    abares_hist, abares_trend = _load_productivity_abares()
    lc_hist, lc_trend = _load_labour_cost()
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

    _plot_historical(
        ax_a, abares_hist['Beef'], abares_trend['Beef'],
        '(a) Productivity and area cost - Beef', (1988, 2050)
    )
    _plot_historical(
        ax_b, abares_hist['Crop'], abares_trend['Crop'],
        '(b) Productivity and area cost - Crop', (1988, 2050)
    )
    _plot_historical(
        ax_c, abares_hist['Dairy'], abares_trend['Dairy'],
        '(c) Productivity and area cost - Dairy', (1988, 2050)
    )
    _plot_historical(
        ax_d, abares_hist['Sheep'], abares_trend['Sheep'],
        '(d) Productivity and area cost - Sheep', (1988, 2050)
    )
    _plot_historical(ax_e, lc_hist, lc_trend, '(e) Labour cost', (2010, 2050))

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
        hist_s, trend_s = feedlots[raw_label]
        _plot_historical(ax, hist_s, trend_s, title, (2010, 2050))
        _set_recent_ticks(ax)

    _apply_row_shared_y(ax_f, ax_g)
    _apply_row_shared_y(ax_h, ax_i)

    fig.supylabel('Multiplier', fontsize=TEXT_SIZE, x=0.015)

    fig.canvas.draw()
    for ax in [ax_a, ax_b, ax_c, ax_d, ax_e, ax_f, ax_g, ax_h, ax_i]:
        lo, hi = ax.get_ylim()
        for step in [0.05, 0.1, 0.2, 0.25, 0.5, 1.0]:
            n = round((np.floor(hi / step) - np.ceil(lo / step))) + 1
            if 4 <= n <= 6:
                ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(step))
                break

    out = os.path.join(OUTPUT_DIR, 'test_input_data_historical_only.png')
    fig.savefig(out, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {out}')


if __name__ == '__main__':
    main()
