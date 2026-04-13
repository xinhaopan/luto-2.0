"""
05_Area.py
Combined stacked-area figure for land use and agricultural management area.
Layout: 2 rows x 4 columns (Land-use, AM; AgS1-AgS4).
"""
import _path_setup  # noqa: F401
import os

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import pandas as pd

from tools.data_helper import (
    get_zip_info,
    get_path,
    _list_years,
    _list_years_zip,
    _read_csv_from_zip,
)
from tools.plot_helper import (
    get_colors,
    calc_y_range,
    set_plot_style,
    stacked_area_pos_neg,
)
from tools.parameters import input_files, font_size, OUTPUT_DIR, SCENARIO_LABELS
from tools.two_row_figure import export_long_tables

RENAME_AM = {
    "Asparagopsis taxiformis": "Methane reduction (livestock)",
    "Precision Agriculture": "Agricultural technology (fertiliser)",
    "Ecological Grazing": "Regenerative agriculture (livestock)",
    "Savanna Burning": "Early dry-season savanna burning",
    "AgTech EI": "Agricultural technology (energy)",
    "Biochar": "Biochar (soil amendment)",
    "HIR - Beef": "Human-induced regeneration (Beef)",
    "HIR - Sheep": "Human-induced regeneration (Sheep)",
    "Utility Solar PV": "Utility Solar PV",
    "Onshore Wind": "Onshore wind",
}

RENAME_NON_AG = {
    "Environmental Plantings": "Environmental plantings (mixed species)",
    "Riparian Plantings": "Riparian buffer restoration (mixed species)",
    "Sheep Agroforestry": "Agroforestry (mixed species + sheep)",
    "Beef Agroforestry": "Agroforestry (mixed species + beef)",
    "Carbon Plantings (Block)": "Carbon plantings (monoculture)",
    "Sheep Carbon Plantings (Belt)": "Farm forestry (hardwood timber + sheep)",
    "Beef Carbon Plantings (Belt)": "Farm forestry (hardwood timber + beef)",
    "BECCS": "BECCS (Bioenergy with Carbon Capture and Storage)",
    "Destocked - natural land": "Destocked - natural land",
}

RENAME_AM_NON_AG = {**RENAME_AM, **RENAME_NON_AG}

LU_COLORS = {
    "Cropland and horticulture": "#AECB75",
    "Grazing (modified pastures)": "#762400",
    "Grazing (native vegetation)": "#C4996B",
    "Non-agricultural land": "#3A7F4A",
}

CROPLAND_LUS = {
    "Apples", "Citrus", "Cotton", "Grapes", "Hay", "Nuts",
    "Other non-cereal crops", "Pears", "Plantation fruit", "Rice",
    "Stone fruit", "Sugar", "Summer cereals", "Summer legumes",
    "Summer oilseeds", "Tropical stone fruit", "Vegetables",
    "Winter cereals", "Winter legumes", "Winter oilseeds",
}
MODIFIED_PASTURE_LUS = {
    "Beef - modified land", "Dairy - modified land", "Sheep - modified land",
}
NATIVE_PASTURE_LUS = {
    "Beef - natural land", "Dairy - natural land", "Sheep - natural land",
}
COLORS_FILE = 'tools/land use colors.xlsx'


def classify_land_use(name):
    if name in CROPLAND_LUS:
        return "Cropland and horticulture"
    if name in MODIFIED_PASTURE_LUS:
        return "Grazing (modified pastures)"
    if name in NATIVE_PASTURE_LUS:
        return "Grazing (native vegetation)"
    return None


def load_report_source_csv(scenario, csv_name):
    info = get_zip_info(scenario)
    frames = []
    if info is not None:
        zip_path, prefix = info
        years = _list_years_zip(zip_path, prefix)
        for year in years:
            df = _read_csv_from_zip(zip_path, prefix, year, csv_name)
            if df is not None and not df.empty:
                frames.append(df)
    else:
        base = get_path(scenario)
        for year in _list_years(base):
            path = os.path.join(base, f'out_{year}', f'{csv_name}_{year}.csv')
            if os.path.exists(path):
                df = pd.read_csv(path)
                if not df.empty:
                    frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def prepare_land_use():
    rows = []

    for scenario in input_files:
        area_ag = load_report_source_csv(scenario, 'area_agricultural_landuse')
        if not area_ag.empty:
            area_ag = area_ag.query('region == "AUSTRALIA" and Water_supply != "ALL"').copy()
            area_ag['category'] = area_ag['Land-use'].map(classify_land_use)
            area_ag = area_ag.dropna(subset=['category'])
            for _, row in area_ag.iterrows():
                rows.append({
                    'year': int(row['Year']),
                    'scenario': scenario,
                    'category': row['category'],
                    'value': float(row['Area (ha)']) / 1e6,
                })

        area_non_ag = load_report_source_csv(scenario, 'area_non_agricultural_landuse')
        if not area_non_ag.empty:
            area_non_ag = area_non_ag.copy()
            area_non_ag['Land-use'] = area_non_ag['Land-use'].replace(RENAME_NON_AG)
            area_non_ag = area_non_ag.query('region == "AUSTRALIA"').copy()
            for _, row in area_non_ag.iterrows():
                rows.append({
                    'year': int(row['Year']),
                    'scenario': scenario,
                    'category': 'Non-agricultural land',
                    'value': float(row['Area (ha)']) / 1e6,
                })

    df = pd.DataFrame(rows)
    return df.groupby(['year', 'scenario', 'category'], as_index=False)['value'].sum()


def prepare_am():
    rows = []

    for scenario in input_files:
        area_am = load_report_source_csv(scenario, 'area_agricultural_management')
        if area_am.empty:
            continue
        area_am = area_am.copy()
        area_am['Type'] = area_am['Type'].replace(RENAME_AM_NON_AG)
        area_am = area_am.query('region == "AUSTRALIA" and Water_supply != "ALL" and `Land-use` != "ALL"').copy()
        for _, row in area_am.iterrows():
            rows.append({
                'year': int(row['Year']),
                'scenario': scenario,
                'category': row['Type'],
                'value': float(row['Area (ha)']) / 1e6,
            })

    return pd.DataFrame(rows)


def draw_row(fig, gs, row_idx, df_long, colors, y_range, row_label, show_titles):
    years = list(range(2025, 2051))
    cats = list(colors.keys())
    clrs = list(colors.values())
    axes = []

    for col_idx, scenario in enumerate(input_files):
        if col_idx == 0:
            ax = fig.add_subplot(gs[row_idx, col_idx])
        else:
            ax = fig.add_subplot(gs[row_idx, col_idx], sharey=axes[0])
        axes.append(ax)

        df_s = df_long[df_long['scenario'] == scenario]
        df_wide = (
            df_s.pivot_table(index='year', columns='category', values='value', aggfunc='sum')
            .reindex(index=years)
            .reindex(columns=cats, fill_value=0)
        )

        title = SCENARIO_LABELS.get(scenario, scenario).split('\n')[0] if show_titles else ''
        ylabel = ''
        stacked_area_pos_neg(
            ax,
            df_wide,
            colors=dict(zip(cats, clrs)),
            alpha=0.60,
            title_name=title,
            ylabel=ylabel,
            y_ticks_all=y_range,
            show_legend=False,
        )
        ax.plot([0, 1], [0, 0], transform=ax.transAxes,
                color='black', linewidth=1.2, zorder=50, clip_on=False)

        tick_pos = [y for y in range(2025, 2051, 5)]
        ax.set_xticks(tick_pos)
        ax.tick_params(axis='x', labelrotation=45, labelbottom=True)
        if row_idx == 0:
            ax.tick_params(axis='x', labelbottom=False)

        if col_idx != 0:
            ax.tick_params(axis='y', labelleft=False)

        if row_idx == 1 and col_idx != 0:
            fig.canvas.draw()
            xlabs = ax.get_xticklabels()
            if xlabs:
                xlabs[0].set_visible(False)

    return axes


def add_patch_legend(ax, colors, ncol):
    handles = [
        mpatches.Patch(facecolor=color, edgecolor='none', label=label)
        for label, color in colors.items()
    ]
    ax.axis('off')
    ax.legend(
        handles,
        [h.get_label() for h in handles],
        loc='center left',
        ncol=ncol,
        frameon=False,
        handlelength=0.7,
        handleheight=0.7,
        handletextpad=0.4,
        labelspacing=0.6,
        columnspacing=1.0,
    )


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    set_plot_style(font_size=font_size)

    area_lu = prepare_land_use()
    area_am = prepare_am()
    export_long_tables(
        '05_area_long_tables.xlsx',
        land_use=area_lu,
        agricultural_management=area_am,
    )

    am_colors_all = get_colors(COLORS_FILE, 'am')
    am_colors = {
        label: color
        for label, color in am_colors_all.items()
        if label not in {
            'Public and indigenous land, urban land, plantation forestry, and water bodies',
            'No agricultural management',
        }
    }

    area_lu = area_lu[area_lu['category'].isin(LU_COLORS)].copy()
    area_am = area_am[area_am['category'].isin(am_colors)].copy()

    y_range_lu = calc_y_range(area_lu, 5)
    y_range_am = calc_y_range(area_am, 5)

    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.sans-serif'] = ['Arial']

    fig = plt.figure(figsize=(21.5, 9.2))
    gs = gridspec.GridSpec(
        2, 5, figure=fig,
        width_ratios=[1, 1, 1, 1, 0.78],
        hspace=0.26, wspace=0.10
    )
    fig.subplots_adjust(left=0.08, right=0.95, top=0.90, bottom=0.12)

    axes_top = draw_row(fig, gs, 0, area_lu, LU_COLORS, y_range_lu, 'Land-use', show_titles=True)
    axes_bot = draw_row(fig, gs, 1, area_am, am_colors, y_range_am, 'Agricultural management', show_titles=False)

    for ax in axes_top:
        ax.set_xlabel('')
    for ax in axes_bot:
        ax.set_xlabel('')

    fig.text(0.43, 0.94, 'Land-use', ha='center', va='bottom',
             fontsize=font_size, fontfamily='Arial', fontweight='normal')
    fig.text(0.43, 0.47, 'Agricultural management', ha='center', va='bottom',
             fontsize=font_size, fontfamily='Arial', fontweight='normal')
    fig.text(0.038, 0.50, 'Area (Mha yr\u207b\u00b9)', ha='center', va='center', rotation=90,
             fontsize=font_size, fontfamily='Arial', fontweight='normal')

    ax_leg_lu = fig.add_subplot(gs[0, 4])
    ax_leg_am = fig.add_subplot(gs[1, 4])
    add_patch_legend(ax_leg_lu, LU_COLORS, ncol=1)
    add_patch_legend(ax_leg_am, am_colors, ncol=1)

    out = os.path.join(OUTPUT_DIR, '05_area.svg')
    fig.savefig(out, dpi=600, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == '__main__':
    main()
