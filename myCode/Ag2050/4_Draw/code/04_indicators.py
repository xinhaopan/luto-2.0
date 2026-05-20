"""
04_indicators.py
5-row × 4-column overview indicator figure.
Rows:    NER, GHG, Biodiversity, Agri-food, Water
Columns: 4 scenarios (AgS1–AgS4)
"""
import _path_setup  # noqa: F401

import os
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd

from tools.two_row_figure import (
    RENAME_AM_NON_AG,
    input_files,
    load_report_source_csv,
)
from tools.parameters import OUTPUT_DIR, SCENARIO_LABELS, font_size
from tools.plot_helper import calc_y_range, set_plot_style, stacked_area_pos_neg

# ── Colors ────────────────────────────────────────────────────────────────────

NER_COLORS = {
    'Ag cost':                         '#fab431',
    'Agmgt cost':                      '#ec7951',
    'Non-ag cost':                     '#cd4975',
    'Transition(ag->non-ag) cost':     '#6200ac',
    'Transition(ag->ag) cost':         '#9f0e9e',
    'Ag revenue':                      '#2d688f',
    'Agmgt revenue':                   '#19928e',
    'Non-ag revenue':                  '#35b876',
}
# Legend order: top of chart → bottom (positives reversed, then negatives in stacking order)
NER_LEGEND_ORDER = [
    'Non-ag revenue', 'Agmgt revenue', 'Ag revenue',
    'Ag cost', 'Agmgt cost', 'Non-ag cost',
    'Transition(ag->non-ag) cost', 'Transition(ag->ag) cost',
]

GHG_COLORS = {
    'Non-agricultural land-use':  '#6eabb1',
    'Agricultural management':    '#9A8AB3',
    'Agricultural land-use':      '#f39b8b',
    'Transition':                 '#eb9132',
}

BIO_COLORS = {
    'Non-agricultural land-use':  '#6eabb1',
    'Agricultural management':    '#9A8AB3',
    'Agricultural land-use':      '#f39b8b',
}

FOOD_COLORS = {
    'Non-agricultural land-use':  '#4eb5a0',
    'Agricultural management':    '#c97bb5',
    'Agricultural land-use':      '#e8a038',
}

WATER_COLORS = {
    'Non-agricultural land-use': '#3A7F4A',
    'Climate change impact':     '#d95f02',
    'Dryland agriculture':       '#92c5de',
    'Irrigated agriculture':     '#2166ac',
    'Agricultural management':   '#619b8a',
}

WATER_LEGEND_ORDER = [
    'Agricultural management',
    'Non-agricultural land-use',
    'Climate change impact',
    'Dryland agriculture',
    'Irrigated agriculture',
]

# (label, colors, unit_label_two_lines, legend_order_or_None)
ROW_CONFIG = [
    ('NER',          NER_COLORS,   'Net economic returns\n(Billion AU$ yr⁻¹)', NER_LEGEND_ORDER),
    ('GHG',          GHG_COLORS,   'GHG emissions\n(MtCO₂e yr⁻¹)',       None),
    ('Biodiversity', BIO_COLORS,   'Biodiversity contribution-\nweighted area (Mha yr⁻¹)', None),
    ('Agri-food',    FOOD_COLORS,  'Agri-food production\n(Mt yr⁻¹)',     None),
    ('Water',        WATER_COLORS, 'Change in water yield\nrelative to 2010 (GL yr⁻¹)', None),
]

# ── Data preparation ───────────────────────────────────────────────────────────

def prepare_ner():
    rows = []
    for scenario in input_files:
        revenue_ag = load_report_source_csv(scenario, 'economics_ag_revenue')
        cost_ag    = load_report_source_csv(scenario, 'economics_ag_cost')
        if not revenue_ag.empty:
            rev = revenue_ag.query(
                'region == "AUSTRALIA" and Water_supply != "ALL" and Type != "ALL"'
            ).copy()
            for _, row in rev.groupby('Year', as_index=False)['Value ($)'].sum().iterrows():
                rows.append({'year': int(row['Year']), 'scenario': scenario,
                             'category': 'Ag revenue', 'value': float(row['Value ($)']) / 1e9})
        if not cost_ag.empty:
            cost = cost_ag.query(
                'region == "AUSTRALIA" and Water_supply != "ALL" and Type != "ALL"'
            ).copy()
            for _, row in cost.groupby('Year', as_index=False)['Value ($)'].sum().iterrows():
                rows.append({'year': int(row['Year']), 'scenario': scenario,
                             'category': 'Ag cost', 'value': -float(row['Value ($)']) / 1e9})

        revenue_am = load_report_source_csv(scenario, 'economics_am_revenue')
        cost_am    = load_report_source_csv(scenario, 'economics_am_cost')
        if not revenue_am.empty:
            rev = revenue_am.query(
                'region == "AUSTRALIA" and Water_supply != "ALL" and `Land-use` != "ALL"'
            ).copy()
            for _, row in rev.groupby('Year', as_index=False)['Value ($)'].sum().iterrows():
                rows.append({'year': int(row['Year']), 'scenario': scenario,
                             'category': 'Agmgt revenue', 'value': float(row['Value ($)']) / 1e9})
        if not cost_am.empty:
            cost = cost_am.query(
                'region == "AUSTRALIA" and Water_supply != "ALL" and `Land-use` != "ALL"'
            ).copy()
            for _, row in cost.groupby('Year', as_index=False)['Value ($)'].sum().iterrows():
                rows.append({'year': int(row['Year']), 'scenario': scenario,
                             'category': 'Agmgt cost', 'value': -float(row['Value ($)']) / 1e9})

        revenue_non_ag = load_report_source_csv(scenario, 'economics_non_ag_revenue')
        cost_non_ag    = load_report_source_csv(scenario, 'economics_non_ag_cost')
        if not revenue_non_ag.empty:
            rev = revenue_non_ag.query('region == "AUSTRALIA"').copy()
            for _, row in rev.groupby('Year', as_index=False)['Value ($)'].sum().iterrows():
                rows.append({'year': int(row['Year']), 'scenario': scenario,
                             'category': 'Non-ag revenue', 'value': float(row['Value ($)']) / 1e9})
        if not cost_non_ag.empty:
            cost = cost_non_ag.query('region == "AUSTRALIA"').copy()
            for _, row in cost.groupby('Year', as_index=False)['Value ($)'].sum().iterrows():
                rows.append({'year': int(row['Year']), 'scenario': scenario,
                             'category': 'Non-ag cost', 'value': -float(row['Value ($)']) / 1e9})

        t_ag2ag = load_report_source_csv(scenario, 'transition_cost_ag2ag')
        if not t_ag2ag.empty:
            t = t_ag2ag.query(
                '`From-land-use` != "ALL" and `To-land-use` != "ALL" and Type != "ALL"'
            ).copy()
            for _, row in t.groupby('Year', as_index=False)['Cost ($)'].sum().iterrows():
                rows.append({'year': int(row['Year']), 'scenario': scenario,
                             'category': 'Transition(ag->ag) cost',
                             'value': -float(row['Cost ($)']) / 1e9})

        t_ag2non = load_report_source_csv(scenario, 'transition_cost_ag2non_ag')
        if not t_ag2non.empty:
            t = t_ag2non.query(
                '`From-land-use` != "ALL" and `To-land-use` != "ALL" and `Cost-type` != "ALL"'
            ).copy()
            for _, row in t.groupby('Year', as_index=False)['Cost ($)'].sum().iterrows():
                rows.append({'year': int(row['Year']), 'scenario': scenario,
                             'category': 'Transition(ag->non-ag) cost',
                             'value': -float(row['Cost ($)']) / 1e9})

    return pd.DataFrame(rows)


def prepare_ghg():
    rows = []
    for scenario in input_files:
        for csv, cat, vcol, filt in [
            ('GHG_emissions_separate_agricultural_landuse', 'Agricultural land-use',
             'Value (t CO2e)',
             'region == "AUSTRALIA" and Water_supply != "ALL" and Source != "ALL"'),
            ('GHG_emissions_separate_agricultural_management', 'Agricultural management',
             'Value (t CO2e)',
             'region == "AUSTRALIA" and Water_supply != "ALL" and `Land-use` != "ALL"'),
            ('GHG_emissions_separate_no_ag_reduction', 'Non-agricultural land-use',
             'Value (t CO2e)',
             'region == "AUSTRALIA" and `Land-use` != "ALL"'),
            ('GHG_emissions_separate_transition_penalty', 'Transition',
             'Value (t CO2e)',
             'region == "AUSTRALIA" and Type != "ALL" and Water_supply != "ALL"'),
        ]:
            df = load_report_source_csv(scenario, csv)
            if df.empty:
                continue
            df = df.query(filt).copy()
            for _, row in df.groupby('Year', as_index=False)[vcol].sum().iterrows():
                rows.append({'year': int(row['Year']), 'scenario': scenario,
                             'category': cat, 'value': float(row[vcol]) / 1e6})
    return pd.DataFrame(rows)


def prepare_bio():
    rows = []
    vcol = 'Area Weighted Score (ha)'
    cat_map = {
        'Agricultural land-use':     'Agricultural land-use',
        'Agricultural Management':   'Agricultural management',
        'Non-Agricultural Land-use': 'Non-agricultural land-use',
    }
    for scenario in input_files:
        bio = load_report_source_csv(scenario, 'biodiversity_GBF2_all_scores')
        if bio.empty:
            continue
        bio = bio.replace(RENAME_AM_NON_AG).query(
            'region == "AUSTRALIA" and Water_supply != "ALL" and Landuse != "ALL"'
            ' and abs(`Area Weighted Score (ha)`) > 1e-4'
        ).copy()
        for _, row in bio.groupby(['Year', 'Type'], as_index=False)[vcol].sum().iterrows():
            cat = cat_map.get(row['Type'])
            if cat is None:
                continue
            rows.append({'year': int(row['Year']), 'scenario': scenario,
                         'category': cat, 'value': float(row[vcol]) / 1e6})
    return pd.DataFrame(rows)


def prepare_food():
    rows = []
    vcol = 'Production (tonnes, KL)'
    cat_map = {
        'Agricultural Landuse':     'Agricultural land-use',
        'Agricultural Management':  'Agricultural management',
        'Non-agricultural Landuse': 'Non-agricultural land-use',
    }
    for scenario in input_files:
        food = load_report_source_csv(scenario, 'quantity_production_kt_separate')
        if food.empty:
            continue
        for raw, cat in cat_map.items():
            data = food.query('`Landuse Type` == @raw').groupby('Year', as_index=False)[vcol].sum()
            for _, row in data.iterrows():
                rows.append({'year': int(row['Year']), 'scenario': scenario,
                             'category': cat, 'value': float(row[vcol]) / 1e6})
    return pd.DataFrame(rows)


def _subtract_2010(df):
    if df.empty:
        return df
    baseline = df[df['year'] == 2010].set_index(['scenario', 'category'])['value']
    df = df.copy()
    df['value'] = df.apply(
        lambda r: r['value'] - baseline.get((r['scenario'], r['category']), 0.0), axis=1
    )
    return df


def prepare_water():
    rows = []
    vcol = 'Water Net Yield (ML)'
    for scenario in input_files:
        water = load_report_source_csv(scenario, 'water_yield_separate_watershed')
        if water.empty:
            continue
        water = (
            water.groupby(
                ['Water Supply', 'Landuse', 'Type', 'Agricultural Management', 'Year'],
                dropna=False, as_index=False,
            )[vcol].sum()
        )
        water = water.replace(RENAME_AM_NON_AG)
        water = water.query('`Water Supply` != "ALL" and Landuse != "ALL"').copy()

        cci_df = load_report_source_csv(scenario, 'water_yield_limits_and_public_land')
        cci_by_year = (
            {} if cci_df.empty
            else (cci_df.groupby('Year')['Climate Change Impact (ML)'].sum() / 1e3).to_dict()
        )

        irr    = water.query('Type == "Agricultural land-use" and `Water Supply` == "Irrigated"').groupby('Year', as_index=False)[vcol].sum()
        dry    = water.query('Type == "Agricultural land-use" and `Water Supply` == "Dryland"').groupby('Year', as_index=False)[vcol].sum()
        agmgt  = water.query('Type == "Agricultural Management"').groupby('Year', as_index=False)[vcol].sum()
        non_ag = water.query('Type == "Non-Agricultural Land-use"').groupby('Year', as_index=False)[vcol].sum()

        irr_d    = {} if irr.empty    else irr.set_index('Year')[vcol].to_dict()
        dry_d    = {} if dry.empty    else dry.set_index('Year')[vcol].to_dict()
        agmgt_d  = {} if agmgt.empty  else agmgt.set_index('Year')[vcol].to_dict()
        non_ag_d = {} if non_ag.empty else non_ag.set_index('Year')[vcol].to_dict()

        for year in sorted(set(irr_d) | set(dry_d) | set(agmgt_d) | set(non_ag_d) | set(cci_by_year)):
            rows.append({'year': int(year), 'scenario': scenario, 'category': 'Irrigated agriculture',     'value': irr_d.get(year, 0.0) / 1e3})
            rows.append({'year': int(year), 'scenario': scenario, 'category': 'Dryland agriculture',       'value': dry_d.get(year, 0.0) / 1e3})
            rows.append({'year': int(year), 'scenario': scenario, 'category': 'Climate change impact',     'value': cci_by_year.get(year, 0.0)})
            rows.append({'year': int(year), 'scenario': scenario, 'category': 'Agricultural management',   'value': agmgt_d.get(year, 0.0) / 1e3})
            rows.append({'year': int(year), 'scenario': scenario, 'category': 'Non-agricultural land-use', 'value': non_ag_d.get(year, 0.0) / 1e3})

    df = pd.DataFrame(rows)
    climate = df[df['category'] == 'Climate change impact'].copy()
    rest    = _subtract_2010(df[df['category'] != 'Climate change impact'].copy())
    return pd.concat([rest, climate], ignore_index=True)


# ── Figure ─────────────────────────────────────────────────────────────────────

def _add_row_unit_label(fig, x, y, text):
    fig.text(
        x, y, text,
        ha='center', va='center', rotation=90,
        fontsize=font_size, fontfamily='Arial',
        multialignment='center',
    )


def _draw_indicators_row(fig, gs, row_idx, df_long, colors, y_range):
    years = list(range(2010, 2051))
    cats  = list(colors.keys())
    axes  = []
    for col_idx, scenario in enumerate(input_files):
        ax = (fig.add_subplot(gs[row_idx, col_idx]) if col_idx == 0
              else fig.add_subplot(gs[row_idx, col_idx], sharey=axes[0]))
        axes.append(ax)

        df_s = df_long[df_long['scenario'] == scenario]
        df_wide = (
            df_s.pivot_table(index='year', columns='category', values='value', aggfunc='sum')
            .reindex(index=years)
            .reindex(columns=cats, fill_value=0)
        )

        stacked_area_pos_neg(
            ax, df_wide, colors=colors, alpha=0.60,
            title_name='', ylabel='', y_ticks_all=y_range, show_legend=False,
        )
        ax.plot([0, 1], [0, 0], transform=ax.transAxes,
                color='black', linewidth=1.2, zorder=50, clip_on=False)

        is_last_row = (row_idx == len(ROW_CONFIG) - 1)
        ax.set_xticks([y for y in range(2010, 2051, 10)])
        ax.tick_params(axis='x', labelrotation=45, labelbottom=is_last_row)
        if col_idx != 0:
            ax.tick_params(axis='y', labelleft=False)

    return axes


def save_indicators_figure(all_dfs):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    set_plot_style(font_size=font_size)

    n_rows = len(ROW_CONFIG)
    fig = plt.figure(figsize=(21.5, 5.2 * n_rows))
    gs  = gridspec.GridSpec(
        n_rows, 5, figure=fig,
        width_ratios=[1, 1, 1, 1, 0.72],
        hspace=0.22, wspace=0.10,
    )
    fig.subplots_adjust(left=0.13, right=0.95, top=0.94, bottom=0.07)

    all_row_axes = []
    for row_idx, (_, colors, _, legend_order) in enumerate(ROW_CONFIG):
        df = all_dfs[row_idx]
        df = df[df['category'].isin(colors)].copy()
        y_range = calc_y_range(df, 5)
        axes = _draw_indicators_row(fig, gs, row_idx, df, colors, y_range)
        all_row_axes.append(axes)

        # Legend — order matches visual top-to-bottom in chart
        legend_ax = fig.add_subplot(gs[row_idx, 4])
        if legend_order is not None:
            ordered_cats = [c for c in legend_order if c in colors]
        else:
            ordered_cats = list(reversed(colors.keys()))
        handles = [
            mpatches.Patch(facecolor=colors[cat], edgecolor='none', label=cat)
            for cat in ordered_cats
        ]
        legend_ax.axis('off')
        legend_ax.legend(handles, [h.get_label() for h in handles],
                         loc='center left', ncol=1, frameon=False,
                         handlelength=0.7, handleheight=0.7,
                         handletextpad=0.4, labelspacing=0.5)

    # Draw to get axes positions
    fig.canvas.draw()

    # Hide 2010 x-tick label for columns 1, 2, 3 (last row only, which has x labels)
    last_row_axes = all_row_axes[-1]
    for ax in last_row_axes[1:]:
        xlabels = ax.get_xticklabels()
        if xlabels:
            xlabels[0].set_visible(False)

    # Per-row y-axis unit label. Keep each row as one label so old split labels
    # cannot stack over each other on the left margin.
    for row_idx, (_, _, unit_label, _) in enumerate(ROW_CONFIG):
        pos  = all_row_axes[row_idx][0].get_position()
        y_mid = (pos.y0 + pos.y1) / 2
        _add_row_unit_label(fig, 0.035, y_mid, unit_label)

    # Scenario column headers above the first row
    header_y = all_row_axes[0][0].get_position().y1 + 0.012
    for ax, scenario in zip(all_row_axes[0], input_files):
        pos = ax.get_position()
        cx  = (pos.x0 + pos.x1) / 2
        fig.text(cx, header_y, SCENARIO_LABELS.get(scenario, scenario),
                 ha='center', va='bottom', fontsize=font_size,
                 fontweight='bold', fontfamily='Arial')

    out = os.path.join(OUTPUT_DIR, '04_indicators.svg')
    fig.savefig(out, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {out}')


def main():
    dfs = [
        prepare_ner(),
        prepare_ghg(),
        prepare_bio(),
        prepare_food(),
        prepare_water(),
    ]
    save_indicators_figure(dfs)


if __name__ == '__main__':
    main()
