"""
11_Demand.py
Agri-food demand stacked-area figure.
Uses the same CSV filter as LUTO data.py (AG2050_MODE):
  Domestic_diet='BAU', Global_diet='BAU', Convergence=2050,
  Imports='Static', Waste=1.0, Feed='BAU'
Layout: 4 horizontal scenario groups (AgS1–AgS4), each 13 rows × 2 cols = 26 on-land commodity subplots.
Group title placed above each group via fig.text() after canvas.draw().
"""
import _path_setup  # noqa: F401

import os
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator
import pandas as pd

from tools.parameters import OUTPUT_DIR, font_size, SCENARIO_LABELS
from tools.two_row_figure import _add_vertical_unit_label, export_long_tables

# ── Scenario groups ───────────────────────────────────────────────────────────
# Each entry: (scenario_key, panel_label)
# panel_label matches SCENARIO_LABELS in parameters.py
SCENARIO_GROUPS = [
    ('AgS1', 'Regional Ag Capitals'),
    ('AgS2', 'Landscape Stewardship'),
    ('AgS3', 'Climate Survival'),
    ('AgS4', 'System Decline'),
]

# ── Stack colors ──────────────────────────────────────────────────────────────
# Imports are shown as NEGATIVE (below zero) so the net positive area equals
# All_demand = domestic + exports + feed − imports, matching LUTO's production target.
COLORS = ['#66c2a5', '#fcbd62', '#bda0cb', '#fc8d62']
LABELS = ['Domestic', 'Exports', 'Feed', 'Imports']
OFF_LAND_COMMODITIES = {'aquaculture', 'chicken', 'eggs', 'pork'}

N_ROWS_G = 13
N_COLS_G = 2

# ── LUTO-consistent demand filter ────────────────────────────────────────────
DEMAND_FILTER = dict(
    Domestic_diet='BAU',
    Global_diet='BAU',
    Convergence=2050,
    Imports='Static',
    Waste=1.0,
    Feed='BAU',
)

CSV_PATH = os.path.join(
    os.path.dirname(__file__),
    '../../../../input/All_LUTO_demand_scenarios_with_convergences.csv',
)

YEARS = list(range(2010, 2051))


def _compact_tick_label(value, _):
    return f'{value:g}'


def load_demand():
    return pd.read_csv(CSV_PATH)


def get_commodities(df_raw):
    sub = df_raw[
        (df_raw['Scenario'] == 'AgS1') &
        (df_raw['Domestic_diet'] == DEMAND_FILTER['Domestic_diet']) &
        (df_raw['Global_diet'] == DEMAND_FILTER['Global_diet']) &
        (df_raw['Convergence'] == DEMAND_FILTER['Convergence']) &
        (df_raw['Imports'] == DEMAND_FILTER['Imports']) &
        (df_raw['Waste'] == DEMAND_FILTER['Waste']) &
        (df_raw['Feed'] == DEMAND_FILTER['Feed'])
    ]
    return sorted(
        c for c in sub['SPREAD_Commodity'].unique()
        if c not in OFF_LAND_COMMODITIES
    )


def prepare(df_raw, scenario):
    """Return wide DataFrame: index=commodity, columns=YEARS, with 4 component stacks."""
    sub = df_raw[
        (df_raw['Scenario'] == scenario) &
        (df_raw['Domestic_diet'] == DEMAND_FILTER['Domestic_diet']) &
        (df_raw['Global_diet'] == DEMAND_FILTER['Global_diet']) &
        (df_raw['Convergence'] == DEMAND_FILTER['Convergence']) &
        (df_raw['Imports'] == DEMAND_FILTER['Imports']) &
        (df_raw['Waste'] == DEMAND_FILTER['Waste']) &
        (df_raw['Feed'] == DEMAND_FILTER['Feed'])
    ]
    return sub  # groupby in plot loop


def draw_figure():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df_raw      = load_demand()
    commodities = get_commodities(df_raw)[:N_ROWS_G * N_COLS_G]

    # Export filtered long-format demand table to Excel
    long_rows = []
    for scen_key, scen_label in SCENARIO_GROUPS:
        sub = prepare(df_raw, scen_key)
        sub = sub[sub['SPREAD_Commodity'].isin(commodities)]
        for _, row in sub.iterrows():
            for component in ('domestic', 'exports', 'feed', 'imports'):
                long_rows.append({
                    'scenario': f'Run_{scen_key}',
                    'year':     int(row['Year']),
                    'commodity': row['SPREAD_Commodity'],
                    'category':  component,
                    'value':     float(row[component]) / 1e6,  # t → Mt
                })
            # All_demand = domestic + exports + feed − imports (LUTO's production target)
            all_d = sum(float(row[c]) for c in ('domestic', 'exports', 'feed')
                        if pd.notna(row[c])) - (float(row['imports']) if pd.notna(row['imports']) else 0)
            long_rows.append({
                'scenario': f'Run_{scen_key}',
                'year':     int(row['Year']),
                'commodity': row['SPREAD_Commodity'],
                'category':  'All_demand (net)',
                'value':     all_d / 1e6,
            })
    export_long_tables(
        '11_demand_long_tables.xlsx',
        demand=pd.DataFrame(long_rows),
    )

    plt.rcParams.update({
        'font.family':     'Arial',
        'font.sans-serif': ['Arial'],
        'font.size':        font_size,
        'svg.fonttype':     'none',
    })

    comm_fs = font_size * 0.60   # commodity subplot title
    tick_fs = font_size * 0.58   # tick labels
    hdr_fs  = font_size * 0.88   # scenario group header

    fig = plt.figure(figsize=(22, 36))

    # Outer GridSpec: one 13-row commodity panel per scenario, plus legend row.
    outer = gridspec.GridSpec(
        2, len(SCENARIO_GROUPS),
        figure=fig,
        height_ratios=[N_ROWS_G, 0.32],
        hspace=0.015,
        wspace=0.26,
        left=0.065, right=0.99,
        top=0.965, bottom=0.050,
    )

    first_row_edge  = []   # (ax_first_in_row, ax_last_in_row) for title placement
    bottom_row_axes = []   # bottom-row axes for x-label ha adjustment
    ref_axes        = {}   # c_idx -> ax from scen_idx==0, used for sharey

    for scen_idx, (scen_key, _) in enumerate(SCENARIO_GROUPS):
        sub_df = prepare(df_raw, scen_key)

        inner = gridspec.GridSpecFromSubplotSpec(
            N_ROWS_G, N_COLS_G,
            subplot_spec=outer[0, scen_idx],
            hspace=0.74,
            wspace=0.42,
        )

        ax_first = ax_last = None

        for c_idx, commodity in enumerate(commodities):
            r = c_idx // N_COLS_G
            c = c_idx % N_COLS_G

            # Share y-axis with the same commodity in scenario 0
            if scen_idx == 0:
                ax = fig.add_subplot(inner[r, c])
                ref_axes[c_idx] = ax
            else:
                ax = fig.add_subplot(inner[r, c], sharey=ref_axes[c_idx])
                ax.tick_params(axis='y', labelleft=False)

            if r == 0:
                if c == 0:
                    ax_first = ax
                if c == N_COLS_G - 1:
                    ax_last = ax

            # Filter this commodity's data and plot
            comm_sub = sub_df[sub_df['SPREAD_Commodity'] == commodity]
            if not comm_sub.empty:
                agg = (comm_sub.groupby('Year')[['domestic', 'exports', 'feed', 'imports']]
                       .sum()
                       .reindex(YEARS, fill_value=0)
                       .fillna(0)
                       / 1e6)                          # → Mt yr⁻¹
                # Positive stack: domestic + exports + feed (= what Australia must produce)
                ax.stackplot(
                    YEARS,
                    agg['domestic'].values,
                    agg['exports'].values,
                    agg['feed'].values,
                    colors=COLORS[:3],
                    alpha=0.85,
                )
                # Negative stack: imports shown below zero (offset against production requirement)
                ax.stackplot(
                    YEARS,
                    -agg['imports'].values,
                    colors=[COLORS[3]],
                    alpha=0.85,
                )
            ax.axhline(0, color='black', linewidth=0.4, zorder=5)

            ax.set_title(commodity.title(), fontsize=comm_fs, pad=5,
                         fontfamily='Arial', loc='center')

            # X-axis: 2010 and 2050, labels set centrally, ha adjusted after draw
            ax.set_xticks([2010, 2050])
            if r == N_ROWS_G - 1:
                ax.set_xticklabels(['2010', '2050'], fontsize=tick_fs, rotation=0)
                bottom_row_axes.append(ax)
            else:
                ax.set_xticklabels([])

            ax.tick_params(axis='x', length=2, pad=4)
            if scen_idx == 0:
                ax.tick_params(axis='y', labelsize=tick_fs, length=2, pad=2)
                ax.yaxis.set_major_locator(MaxNLocator(nbins=3, min_n_ticks=2))
                ax.yaxis.set_major_formatter(FuncFormatter(_compact_tick_label))

        first_row_edge.append((ax_first, ax_last))

    # ── Draw canvas so positions are available ────────────────────────────────
    fig.canvas.draw()

    # ── Nudge x-tick labels inward (2010 → left-aligned, 2050 → right-aligned) ─
    for ax in bottom_row_axes:
        lbls = ax.get_xticklabels()
        if len(lbls) >= 2:
            lbls[0].set_ha('left')    # 2010: text extends rightward (toward center)
            lbls[1].set_ha('right')   # 2050: text extends leftward  (toward center)

    # ── Group titles above each group ─────────────────────────────────────────
    for scen_idx, (_, label) in enumerate(SCENARIO_GROUPS):
        ax_l, ax_r = first_row_edge[scen_idx]
        if ax_l is None or ax_r is None:
            continue
        p_l = ax_l.get_position()
        p_r = ax_r.get_position()
        cx  = (p_l.x0 + p_r.x1) / 2
        ty  = p_l.y1 + 0.010
        fig.text(cx, ty, label,
                 ha='center', va='bottom',
                 fontsize=hdr_fs, fontweight='bold', fontfamily='Arial')

    _add_vertical_unit_label(fig, 0.018, 0.50, 'Agri-food demand (Mt yr⁻¹)', font_size)

    # ── Legend ────────────────────────────────────────────────────────────────
    ax_leg = fig.add_subplot(outer[1, :])
    ax_leg.set_axis_off()
    handles = [
        mpatches.Patch(facecolor=col, edgecolor='none', label=lbl, alpha=0.85)
        for col, lbl in zip(COLORS, LABELS)
    ]
    ax_leg.legend(
        handles, [h.get_label() for h in handles],
        loc='center',
        bbox_to_anchor=(0.5, 0.1),
        ncol=len(LABELS),
        fontsize=font_size,
        frameon=False,
        handlelength=0.9,
        handleheight=0.9,
        handletextpad=0.6,
        columnspacing=2.5,
    )

    out = os.path.join(OUTPUT_DIR, '11_demand.svg')
    fig.savefig(out, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {out}')


if __name__ == '__main__':
    draw_figure()
