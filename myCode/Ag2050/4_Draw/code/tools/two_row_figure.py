import os

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.gridspec as gridspec
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd

from tools.data_helper import (
    _list_years,
    _list_years_zip,
    _read_csv_from_zip,
    get_path,
    get_zip_info,
)
from tools.parameters import EXCEL_DIR, OUTPUT_DIR, SCENARIO_LABELS, font_size, input_files
from tools.plot_helper import calc_y_range, get_colors, set_plot_style, stacked_area_pos_neg

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
                if 'Year' not in df.columns:
                    df = df.copy()
                    df['Year'] = int(year)
                frames.append(df)
    else:
        base = get_path(scenario)
        for year in _list_years(base):
            path = os.path.join(base, f'out_{year}', f'{csv_name}_{year}.csv')
            if os.path.exists(path):
                df = pd.read_csv(path)
                if not df.empty:
                    if 'Year' not in df.columns:
                        df = df.copy()
                        df['Year'] = int(year)
                    frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def get_am_colors():
    am_colors_all = get_colors(COLORS_FILE, 'am')
    return {
        label: color
        for label, color in am_colors_all.items()
        if label not in {
            'Public and indigenous land, urban land, plantation forestry, and water bodies',
            'No agricultural management',
        }
    }


def _draw_row(fig, gs, row_idx, df_long, colors, y_range, show_titles):
    years = list(range(2025, 2051))
    cats = list(colors.keys())
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
        stacked_area_pos_neg(
            ax,
            df_wide,
            colors=colors,
            alpha=0.60,
            title_name=title,
            ylabel='',
            y_ticks_all=y_range,
            show_legend=False,
        )
        ax.plot(
            [0, 1], [0, 0],
            transform=ax.transAxes,
            color='black',
            linewidth=1.2,
            zorder=50,
            clip_on=False,
        )

        ax.set_xticks([year for year in range(2025, 2051, 5)])
        ax.tick_params(axis='x', labelrotation=45, labelbottom=True)
        if row_idx == 0:
            ax.tick_params(axis='x', labelbottom=False)

        if col_idx != 0:
            ax.tick_params(axis='y', labelleft=False)

        if row_idx != 0 and col_idx != 0:
            fig.canvas.draw()
            labels = ax.get_xticklabels()
            if labels:
                labels[0].set_visible(False)

    return axes


def _add_patch_legend(ax, colors):
    handles = [
        mpatches.Patch(facecolor=color, edgecolor='none', label=label)
        for label, color in colors.items()
    ]
    ax.axis('off')
    ax.legend(
        handles,
        [handle.get_label() for handle in handles],
        loc='center left',
        ncol=1,
        frameon=False,
        handlelength=0.7,
        handleheight=0.7,
        handletextpad=0.4,
        labelspacing=0.6,
        columnspacing=1.0,
    )


def _add_mixed_legend(ax, patch_colors, line_label=None, line_color='black'):
    handles = [
        mpatches.Patch(facecolor=color, edgecolor='none', label=label)
        for label, color in patch_colors.items()
    ]
    if line_label:
        handles.append(mlines.Line2D([], [], color=line_color, linewidth=1.5, label=line_label))
    ax.axis('off')
    ax.legend(
        handles,
        [handle.get_label() for handle in handles],
        loc='center left',
        ncol=1,
        frameon=False,
        handlelength=0.7,
        handleheight=0.7,
        handletextpad=0.4,
        labelspacing=0.6,
        columnspacing=1.0,
    )


def save_two_row_figure(df_top, df_bottom, top_colors, bottom_colors, y_label, output_name):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    set_plot_style(font_size=font_size)
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.sans-serif'] = ['Arial']

    df_top = df_top[df_top['category'].isin(top_colors)].copy()
    df_bottom = df_bottom[df_bottom['category'].isin(bottom_colors)].copy()
    y_range_top = calc_y_range(df_top, 5)
    y_range_bottom = calc_y_range(df_bottom, 5)

    fig = plt.figure(figsize=(21.5, 9.2))
    gs = gridspec.GridSpec(
        2, 5, figure=fig,
        width_ratios=[1, 1, 1, 1, 0.78],
        hspace=0.26, wspace=0.10,
    )
    fig.subplots_adjust(left=0.08, right=0.95, top=0.90, bottom=0.12)

    axes_top = _draw_row(fig, gs, 0, df_top, top_colors, y_range_top, show_titles=True)
    axes_bottom = _draw_row(fig, gs, 1, df_bottom, bottom_colors, y_range_bottom, show_titles=False)

    for ax in axes_top + axes_bottom:
        ax.set_xlabel('')

    fig.text(
        0.43, 0.915, 'Land-use',
        ha='center', va='bottom',
        fontsize=font_size, fontfamily='Arial', fontweight='normal',
    )
    fig.text(
        0.43, 0.47, 'Agricultural management',
        ha='center', va='bottom',
        fontsize=font_size, fontfamily='Arial', fontweight='normal',
    )
    fig.text(
        0.038, 0.50, y_label,
        ha='center', va='center', rotation=90,
        fontsize=font_size, fontfamily='Arial', fontweight='normal',
    )

    _add_patch_legend(fig.add_subplot(gs[0, 4]), top_colors)
    _add_patch_legend(fig.add_subplot(gs[1, 4]), bottom_colors)

    out = os.path.join(OUTPUT_DIR, output_name)
    fig.savefig(out, dpi=600, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out}")


def save_three_row_figure(
    df_overview,
    df_top,
    df_bottom,
    overview_colors,
    top_colors,
    bottom_colors,
    y_label,
    output_name,
    total_legend_label='Sum',
    y_label_x=0.038,
):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    set_plot_style(font_size=font_size)
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.sans-serif'] = ['Arial']

    df_overview = df_overview[df_overview['category'].isin(overview_colors)].copy()
    df_top = df_top[df_top['category'].isin(top_colors)].copy()
    df_bottom = df_bottom[df_bottom['category'].isin(bottom_colors)].copy()

    y_range_overview = calc_y_range(df_overview, 5)
    y_range_top = calc_y_range(df_top, 5)
    y_range_bottom = calc_y_range(df_bottom, 5)

    fig = plt.figure(figsize=(21.5, 12.8))
    gs = gridspec.GridSpec(
        3, 5, figure=fig,
        width_ratios=[1, 1, 1, 1, 0.82],
        hspace=0.24, wspace=0.10,
    )
    fig.subplots_adjust(left=0.08, right=0.95, top=0.90, bottom=0.10)

    axes_overview = _draw_row(fig, gs, 0, df_overview, overview_colors, y_range_overview, show_titles=True)
    axes_top = _draw_row(fig, gs, 1, df_top, top_colors, y_range_top, show_titles=False)
    axes_bottom = _draw_row(fig, gs, 2, df_bottom, bottom_colors, y_range_bottom, show_titles=False)

    years = list(range(2025, 2051))
    for ax, scenario in zip(axes_overview, input_files):
        df_s = df_overview[df_overview['scenario'] == scenario]
        total = (
            df_s.groupby('year', as_index=False)['value']
            .sum()
            .set_index('year')
            .reindex(years, fill_value=0)['value']
        )
        ax.plot(years, total.values, color='black', linewidth=1.5, zorder=60)
        ax.tick_params(axis='x', labelbottom=False)

    for ax in axes_top:
        ax.tick_params(axis='x', labelbottom=False)

    for ax in axes_overview + axes_top + axes_bottom:
        ax.set_xlabel('')

    fig.text(
        0.43, 0.948, 'Total',
        ha='center', va='bottom',
        fontsize=font_size, fontfamily='Arial', fontweight='normal',
    )
    fig.text(
        0.43, 0.620, 'Land-use',
        ha='center', va='bottom',
        fontsize=font_size, fontfamily='Arial', fontweight='normal',
    )
    fig.text(
        0.43, 0.332, 'Agricultural management',
        ha='center', va='bottom',
        fontsize=font_size, fontfamily='Arial', fontweight='normal',
    )
    fig.text(
        y_label_x, 0.50, y_label,
        ha='center', va='center', rotation=90,
        fontsize=font_size, fontfamily='Arial', fontweight='normal',
    )

    _add_mixed_legend(fig.add_subplot(gs[0, 4]), overview_colors, line_label=total_legend_label)
    _add_patch_legend(fig.add_subplot(gs[1, 4]), top_colors)
    _add_patch_legend(fig.add_subplot(gs[2, 4]), bottom_colors)

    out = os.path.join(OUTPUT_DIR, output_name)
    fig.savefig(out, dpi=600, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out}")


def export_long_tables(workbook_name, **tables):
    os.makedirs(EXCEL_DIR, exist_ok=True)
    out = os.path.join(EXCEL_DIR, workbook_name)
    with pd.ExcelWriter(out, engine='openpyxl') as writer:
        for sheet_name, df in tables.items():
            export_df = df.copy()
            if not export_df.empty:
                if 'scenario' in export_df.columns:
                    export_df['scenario_label'] = export_df['scenario'].map(SCENARIO_LABELS).fillna(export_df['scenario'])
                sort_cols = [col for col in ['scenario', 'year', 'category'] if col in export_df.columns]
                if sort_cols:
                    export_df = export_df.sort_values(sort_cols).reset_index(drop=True)
            else:
                export_df = pd.DataFrame({'note': ['No data']})
            export_df.to_excel(writer, sheet_name=sheet_name[:31], index=False)
    print(f"Saved: {out}")
