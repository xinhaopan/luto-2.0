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
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr

from tools.two_row_figure import (
    RENAME_AM_NON_AG,
    export_long_tables,
    input_files,
    load_report_source_csv,
)
from tools.data_helper import get_path
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
    'Meat & live animals':      '#9B4528',
    'Livestock products':       '#CA927E',
    'Grains & oilseeds':        '#EB8500',
    'All other crops':          '#F3BD8B',
    'Fruit & vegetables':       '#5A8529',
    'All other horticulture':   '#D1D9BF',
}

WATER_COLORS = {
    'Non-agricultural land-use': '#3A7F4A',
    'Climate change impact':     '#d95f02',
    'Dryland agriculture':       '#92c5de',
    'Irrigated agriculture':     '#2166ac',
    'Agricultural management':   '#B1A7C6',
}

WATER_LEGEND_ORDER = [
    'Agricultural management',
    'Dryland agriculture',
    'Irrigated agriculture',
    'Non-agricultural land-use',
    'Climate change impact',
]

FOOD_LEGEND_ORDER = [
    'Meat & live animals',
    'Livestock products',
    'Grains & oilseeds',
    'All other crops',
    'Fruit & vegetables',
    'All other horticulture',
]

# (label, colors, unit_label_two_lines, legend_order_or_None)
ROW_CONFIG = [
    ('NER',          NER_COLORS,   'Net economic returns\n(Billion AU$ yr⁻¹)', NER_LEGEND_ORDER),
    ('GHG',          GHG_COLORS,   'GHG emissions\n(MtCO₂e yr⁻¹)',       None),
    ('Biodiversity', BIO_COLORS,   'Biodiversity contribution-\nweighted area (Mha yr⁻¹)', None),
    ('Agri-food',    FOOD_COLORS,  'Agri-food production\n(Mt yr⁻¹)',     FOOD_LEGEND_ORDER),
    ('Water',        WATER_COLORS, 'Difference in water yield\nrelative to 2010 (GL yr⁻¹)', WATER_LEGEND_ORDER),
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
    landuse_to_food_group = {
        'Beef - modified land': 'Meat & live animals',
        'Beef - natural land': 'Meat & live animals',
        'Beef Agroforestry': 'Meat & live animals',
        'Beef Carbon Plantings (Belt)': 'Meat & live animals',
        'Sheep - modified land': 'Meat & live animals',
        'Sheep - natural land': 'Meat & live animals',
        'Sheep Agroforestry': 'Meat & live animals',
        'Sheep Carbon Plantings (Belt)': 'Meat & live animals',
        'Dairy - modified land': 'Livestock products',
        'Dairy - natural land': 'Livestock products',
        'Summer cereals': 'Grains & oilseeds',
        'Winter cereals': 'Grains & oilseeds',
        'Summer legumes': 'Grains & oilseeds',
        'Winter legumes': 'Grains & oilseeds',
        'Summer oilseeds': 'Grains & oilseeds',
        'Winter oilseeds': 'Grains & oilseeds',
        'Rice': 'Grains & oilseeds',
        'Cotton': 'All other crops',
        'Hay': 'All other crops',
        'Other non-cereal crops': 'All other crops',
        'Sugar': 'All other crops',
        'Apples': 'Fruit & vegetables',
        'Citrus': 'Fruit & vegetables',
        'Grapes': 'Fruit & vegetables',
        'Pears': 'Fruit & vegetables',
        'Plantation fruit': 'Fruit & vegetables',
        'Stone fruit': 'Fruit & vegetables',
        'Tropical stone fruit': 'Fruit & vegetables',
        'Vegetables': 'Fruit & vegetables',
        'Nuts': 'All other horticulture',
    }
    for scenario in input_files:
        food = load_report_source_csv(scenario, 'quantity_production_kt_separate')
        if food.empty:
            continue
        food = food.copy()
        food['category'] = food['Landuse'].map(landuse_to_food_group)
        food = food.dropna(subset=['category'])
        data = food.groupby(['Year', 'category'], as_index=False)[vcol].sum()
        for _, row in data.iterrows():
            rows.append({'year': int(row['Year']), 'scenario': scenario,
                         'category': row['category'], 'value': float(row[vcol]) / 1e6})
    return pd.DataFrame(rows)


def _load_water_correction(base_path):
    """
    Per-cell attribution correction for water yield.

    For each year, identifies cells that have non-ag land-use and returns their
    total 2010 ag water yield split into (dryland_ML, irrigated_ML).

    This correction re-attributes the baseline yield from "Dryland/Irrigated
    agriculture" to "Non-agricultural land-use" so the non-ag delta correctly
    shows the net effect of land conversion (typically negative for tree planting).
    """
    nc_ag10 = os.path.join(base_path, 'out_2010', 'xr_water_yield_ag_2010.nc')
    if not os.path.exists(nc_ag10):
        return {}
    try:
        ds = xr.open_dataset(nc_ag10)
        data = ds['data'].values   # (cell, layer)
        n_lu = len(ds.lu)          # 29: 'ALL' + 28 specific land-uses
        # layer = lm_idx * n_lu + lu_idx  (lm: 0=ALL, 1=dry, 2=irr; lu_idx=0 for ALL)
        ag10_dry = data[:, n_lu]       # lm=dry, lu=ALL
        ag10_irr = data[:, 2 * n_lu]   # lm=irr, lu=ALL
        ds.close()
    except Exception:
        return {}

    corrections = {}
    for entry in os.scandir(base_path):
        if not (entry.is_dir() and entry.name.startswith('out_')):
            continue
        try:
            year = int(entry.name[4:])
        except ValueError:
            continue
        if year == 2010:
            corrections[year] = (0.0, 0.0)
            continue
        nc_nag = os.path.join(entry.path, f'xr_water_yield_non_ag_{year}.nc')
        if not os.path.exists(nc_nag):
            corrections[year] = (0.0, 0.0)
            continue
        try:
            ds_nag = xr.open_dataset(nc_nag)
            mask = ds_nag['data'].values[:, 0] > 0   # lu=ALL layer; True where non-ag exists
            ds_nag.close()
            corrections[year] = (float(ag10_dry[mask].sum()), float(ag10_irr[mask].sum()))
        except Exception:
            corrections[year] = (0.0, 0.0)
    return corrections


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

        # 2010 baselines for delta calculation
        dry_2010   = dry_d.get(2010, 0.0)
        irr_2010   = irr_d.get(2010, 0.0)
        agmgt_2010 = agmgt_d.get(2010, 0.0)

        # Per-cell correction: for each year, get the 2010 ag yield of cells now non-ag.
        # Corrected non-ag delta  = (non_ag_yield_y) − (2010 ag yield of those cells)
        # Corrected dryland delta = (dry_yield_y − dry_2010) + dry_corr
        # Corrected irrigated delta = (irr_yield_y − irr_2010) + irr_corr
        # This ensures the total water change is conserved while removing the
        # artificial positive contribution from 2010-baseline = 0 for non-ag.
        try:
            corrections = _load_water_correction(get_path(scenario))
        except Exception:
            corrections = {}

        for year in sorted(set(irr_d) | set(dry_d) | set(agmgt_d) | set(non_ag_d) | set(cci_by_year)):
            dry_corr, irr_corr = corrections.get(int(year), (0.0, 0.0))
            rows.append({'year': int(year), 'scenario': scenario,
                         'category': 'Irrigated agriculture',
                         'value': (irr_d.get(year, 0.0) - irr_2010 + irr_corr) / 1e3})
            rows.append({'year': int(year), 'scenario': scenario,
                         'category': 'Dryland agriculture',
                         'value': (dry_d.get(year, 0.0) - dry_2010 + dry_corr) / 1e3})
            rows.append({'year': int(year), 'scenario': scenario,
                         'category': 'Agricultural management',
                         'value': (agmgt_d.get(year, 0.0) - agmgt_2010) / 1e3})
            rows.append({'year': int(year), 'scenario': scenario,
                         'category': 'Non-agricultural land-use',
                         'value': (non_ag_d.get(year, 0.0) - dry_corr - irr_corr) / 1e3})
            rows.append({'year': int(year), 'scenario': scenario,
                         'category': 'Climate change impact',
                         'value': cci_by_year.get(year, 0.0)})

    return pd.DataFrame(rows)


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
    cats  = sorted(colors.keys(), reverse=True)  # reverse-alpha: Z=bottom, A=top of positive stack
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
        total = df_wide.fillna(0).sum(axis=1)
        ax.plot(
            years,
            total.values,
            color='black',
            linewidth=1.5,
            zorder=60,
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
    for row_idx, (_, colors, _, _legend_order) in enumerate(ROW_CONFIG):
        df = all_dfs[row_idx]
        df = df[df['category'].isin(colors)].copy()
        y_range = calc_y_range(df, 5)
        axes = _draw_indicators_row(fig, gs, row_idx, df, colors, y_range)
        all_row_axes.append(axes)

        # Legend — use _legend_order if provided, otherwise A-Z alphabetical
        legend_ax = fig.add_subplot(gs[row_idx, 4])
        patch_map = {cat: mpatches.Patch(facecolor=colors[cat], edgecolor='none', label=cat)
                     for cat in colors}
        if _legend_order:
            ordered = [patch_map[k] for k in _legend_order if k in patch_map]
            ordered += [patch_map[k] for k in sorted(patch_map) if k not in set(_legend_order)]
            handles = ordered
        else:
            handles = sorted(patch_map.values(), key=lambda h: h.get_label())
        handles.append(mlines.Line2D([], [], color='black', linewidth=1.5, label='Sum'))
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
    ner   = prepare_ner()
    ghg   = prepare_ghg()
    bio   = prepare_bio()
    food  = prepare_food()
    water = prepare_water()
    export_long_tables(
        '04_indicators_long_tables.xlsx',
        net_economic_return=ner,
        ghg=ghg,
        biodiversity=bio,
        food=food,
        water=water,
    )
    save_indicators_figure([ner, ghg, bio, food, water])


if __name__ == '__main__':
    main()
