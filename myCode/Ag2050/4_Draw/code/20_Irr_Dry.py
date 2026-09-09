"""
20_Irr_Dry.py
4-row × 4-column figure comparing dryland vs irrigated agriculture.
Rows:    1 Dryland land-use area      2 Irrigated land-use area
         3 Dryland agri-food output   4 Irrigated agri-food output
Columns: 4 scenarios (AgS1–AgS4)

Land-use area uses the same land-use classification as 19_Water.py row 2 (LU_COLORS),
split by water supply. Agri-food output uses the same food-group classification as
04_indicators.py row 4 (FOOD_COLORS), split by water supply.
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

from tools.two_row_figure import (
    COMMODITY_TO_FOOD_GROUP,
    CROPLAND_LUS,
    FOOD_PRODUCTION_CSV,
    FOOD_PRODUCTION_VALUE_COL,
    MODIFIED_PASTURE_LUS,
    NATIVE_PASTURE_LUS,
    UNALLOCATED_LUS,
    LU_COLORS,
    export_long_tables,
    filter_area_ag_rows,
    filter_food_detail_rows,
    input_files,
    load_long_tables,
    load_report_source_csv,
)
from tools.parameters import OUTPUT_DIR, SCENARIO_LABELS, font_size, GENERATE_TABLES
from tools.plot_helper import calc_y_range, set_plot_style, stacked_area_pos_neg

# ── Colors / classification ─────────────────────────────────────────────────────

FOOD_COLORS = {
    'Meat & live animals':      '#9B4528',
    'Livestock products':       '#CA927E',
    'Grains & oilseeds':        '#EB8500',
    'All other crops':          '#F3BD8B',
    'Fruit & vegetables':       '#5A8529',
    'All other horticulture':   '#D1D9BF',
}
FOOD_LEGEND_ORDER = [
    'Meat & live animals',
    'Livestock products',
    'Grains & oilseeds',
    'All other crops',
    'Fruit & vegetables',
    'All other horticulture',
]

def _classify_area(name, water_supply):
    """Same land-use grouping as 19_Water.py row 2 (LU_COLORS), given a water supply."""
    prefix = 'Dryland' if water_supply == 'Dryland' else 'Irrigated'
    if name in CROPLAND_LUS:
        return f'{prefix} cropland and horticulture'
    if name in MODIFIED_PASTURE_LUS:
        return f'{prefix} grazing (modified pastures)'
    if name in NATIVE_PASTURE_LUS:
        return 'Grazing (native vegetation)'
    if name in UNALLOCATED_LUS:
        return 'Unallocated land'
    return None


# ── Data preparation ────────────────────────────────────────────────────────────

def prepare_area(water_supply):
    """Agricultural land-use area (Mha) for one water supply ('Dryland'/'Irrigated')."""
    rows = []
    for scenario in input_files:
        area = filter_area_ag_rows(
            load_report_source_csv(scenario, 'area_agricultural_landuse')
        )
        if area.empty:
            continue
        area = area.query('Water_supply == @water_supply').copy()
        area['category'] = area.apply(
            lambda r: _classify_area(r['Land-use'], r['Water_supply']), axis=1
        )
        area = area.dropna(subset=['category'])
        g = area.groupby(['Year', 'category'], as_index=False)['Area (ha)'].sum()
        for _, r in g.iterrows():
            rows.append({'year': int(r['Year']), 'scenario': scenario,
                         'category': r['category'], 'value': float(r['Area (ha)']) / 1e6})
    return pd.DataFrame(rows)


def prepare_production(water_supply):
    """Agri-food output (Mt) for one water supply, grouped by commodity."""
    rows = []
    for scenario in input_files:
        food = filter_food_detail_rows(
            load_report_source_csv(scenario, FOOD_PRODUCTION_CSV)
        )
        if food.empty:
            continue
        food = food.query('Water_supply == @water_supply').copy()
        food['category'] = food['Commodity'].map(COMMODITY_TO_FOOD_GROUP)
        food = food.dropna(subset=['category'])
        g = food.groupby(['Year', 'category'], as_index=False)[FOOD_PRODUCTION_VALUE_COL].sum()
        for _, r in g.iterrows():
            rows.append({'year': int(r['Year']), 'scenario': scenario,
                         'category': r['category'],
                         'value': float(r[FOOD_PRODUCTION_VALUE_COL]) / 1e6})
    return pd.DataFrame(rows)


# ── Figure ──────────────────────────────────────────────────────────────────────

def _add_row_unit_label(fig, x, y, text):
    fig.text(x, y, text, ha='center', va='center', rotation=90,
             fontsize=font_size, fontfamily='Arial', multialignment='center')


def _draw_row(fig, gs, row_idx, n_rows, df_long, colors, y_range):
    years = list(range(2010, 2051))
    cats = sorted(colors.keys(), reverse=True)  # reverse-alpha: Z=bottom, A=top of positive stack
    axes = []
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
        ax.plot(years, total.values, color='black', linewidth=1.5, zorder=60)
        ax.plot([0, 1], [0, 0], transform=ax.transAxes,
                color='black', linewidth=1.2, zorder=50, clip_on=False)

        is_last_row = (row_idx == n_rows - 1)
        ax.set_xticks([y for y in range(2010, 2051, 10)])
        ax.tick_params(axis='x', labelrotation=45, labelbottom=is_last_row)
        if col_idx != 0:
            ax.tick_params(axis='y', labelleft=False)

    return axes


def save_figure(row_config):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    set_plot_style(font_size=font_size)

    n_rows = len(row_config)
    fig = plt.figure(figsize=(21.5, 5.2 * n_rows))
    gs = gridspec.GridSpec(
        n_rows, 5, figure=fig,
        width_ratios=[1, 1, 1, 1, 0.72],
        hspace=0.22, wspace=0.10,
    )
    fig.subplots_adjust(left=0.13, right=0.95, top=0.94, bottom=0.06)

    all_row_axes = []
    for row_idx, (_, colors, _unit, legend_order, df) in enumerate(row_config):
        df = df[df['category'].isin(colors)].copy()
        # Per-row colors limited to categories actually present (keeps legend honest)
        present = [c for c in colors if c in set(df['category'])]
        row_colors = {c: colors[c] for c in present}
        y_range = calc_y_range(df, 5)
        axes = _draw_row(fig, gs, row_idx, n_rows, df, row_colors, y_range)
        all_row_axes.append(axes)

        # Legend
        legend_ax = fig.add_subplot(gs[row_idx, 4])
        patch_map = {cat: mpatches.Patch(facecolor=row_colors[cat], edgecolor='none', label=cat)
                     for cat in row_colors}
        if legend_order:
            handles = [patch_map[k] for k in legend_order if k in patch_map]
            handles += [patch_map[k] for k in sorted(patch_map) if k not in set(legend_order)]
        else:
            handles = sorted(patch_map.values(), key=lambda h: h.get_label())
        handles.append(mlines.Line2D([], [], color='black', linewidth=1.5, label='Sum'))
        legend_ax.axis('off')
        legend_ax.legend(handles, [h.get_label() for h in handles],
                         loc='center left', ncol=1, frameon=False,
                         handlelength=0.7, handleheight=0.7,
                         handletextpad=0.4, labelspacing=0.5)

    fig.canvas.draw()

    # Hide 2010 x-tick label for columns 1-3 (last row only, which shows x labels)
    for ax in all_row_axes[-1][1:]:
        xlabels = ax.get_xticklabels()
        if xlabels:
            xlabels[0].set_visible(False)

    # Per-row y-axis unit label
    for row_idx, (_, _, unit_label, _, _) in enumerate(row_config):
        pos = all_row_axes[row_idx][0].get_position()
        _add_row_unit_label(fig, 0.035, (pos.y0 + pos.y1) / 2, unit_label)

    # Scenario column headers above the first row
    header_y = all_row_axes[0][0].get_position().y1 + 0.012
    for ax, scenario in zip(all_row_axes[0], input_files):
        pos = ax.get_position()
        cx = (pos.x0 + pos.x1) / 2
        fig.text(cx, header_y, SCENARIO_LABELS.get(scenario, scenario),
                 ha='center', va='bottom', fontsize=font_size,
                 fontweight='bold', fontfamily='Arial')

    out = os.path.join(OUTPUT_DIR, '20_irr_dry.svg')
    fig.savefig(out, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {out}')


def main():
    workbook = '20_irr_dry_long_tables.xlsx'
    if GENERATE_TABLES:
        export_long_tables(
            workbook,
            dryland_area=prepare_area('Dryland'),
            irrigated_area=prepare_area('Irrigated'),
            dryland_production=prepare_production('Dryland'),
            irrigated_production=prepare_production('Irrigated'),
        )
    tables = load_long_tables(
        workbook,
        'dryland_area',
        'irrigated_area',
        'dryland_production',
        'irrigated_production',
    )
    dry_area = tables['dryland_area']
    irr_area = tables['irrigated_area']
    dry_food = tables['dryland_production']
    irr_food = tables['irrigated_production']

    row_config = [
        ('Dryland area',        LU_COLORS,   'Dryland area\n(Mha)',        None,              dry_area),
        ('Irrigated area',      LU_COLORS,   'Irrigated area\n(Mha)',      None,              irr_area),
        ('Dryland agri-food',   FOOD_COLORS, 'Dryland agri-food\n(Mt yr⁻¹)',    FOOD_LEGEND_ORDER, dry_food),
        ('Irrigated agri-food', FOOD_COLORS, 'Irrigated agri-food\n(Mt yr⁻¹)',  FOOD_LEGEND_ORDER, irr_food),
    ]
    save_figure(row_config)


if __name__ == '__main__':
    main()
