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
UNIT_DEJAVU_CHARS = {'₂', '⁻', '¹'}


def _split_unit_font_runs(text):
    runs = []
    current_text = []
    current_family = None
    for char in text:
        family = 'DejaVu Sans' if char in UNIT_DEJAVU_CHARS else 'Arial'
        if family != current_family and current_text:
            runs.append((''.join(current_text), current_family))
            current_text = []
        current_text.append(char)
        current_family = family
    if current_text:
        runs.append((''.join(current_text), current_family))
    return runs


def _measure_text_runs(fig, runs, fontsize, fontweight):
    temp_artists = [
        fig.text(
            0, 0, text,
            fontsize=fontsize,
            fontfamily=family,
            fontweight=fontweight,
            alpha=0,
        )
        for text, family in runs
    ]
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    widths = [
        artist.get_window_extent(renderer=renderer).width
        for artist in temp_artists
    ]
    for artist in temp_artists:
        artist.remove()
    return widths


def _add_vertical_unit_label(fig, x, y, text, fontsize, fontweight='normal'):
    runs = _split_unit_font_runs(text)
    if len(runs) == 1:
        text_run, family = runs[0]
        fig.text(
            x, y, text_run,
            ha='center', va='center', rotation=90,
            fontsize=fontsize, fontfamily=family, fontweight=fontweight,
        )
        return

    widths = _measure_text_runs(fig, runs, fontsize, fontweight)
    total_width = sum(widths)
    cursor = -total_width / 2
    fig_height_px = fig.bbox.height
    for (text_run, family), width in zip(runs, widths):
        offset_px = cursor + width / 2
        fig.text(
            x, y + offset_px / fig_height_px, text_run,
            ha='center', va='center', rotation=90,
            fontsize=fontsize, fontfamily=family, fontweight=fontweight,
        )
        cursor += width


def classify_land_use(name):
    if name in CROPLAND_LUS:
        return "Cropland and horticulture"
    if name in MODIFIED_PASTURE_LUS:
        return "Grazing (modified pastures)"
    if name in NATIVE_PASTURE_LUS:
        return "Grazing (native vegetation)"
    return None


def _filter_region_level(df: pd.DataFrame, level: str = 'region_NRM') -> pd.DataFrame:
    """Keep only one region_level to avoid double-counting with dual-region output.

    Backward-compatible: returns df unchanged if 'region_level' column is absent
    (old single-region format).
    """
    if 'region_level' not in df.columns:
        return df
    return df[df['region_level'] == level].drop(columns='region_level').reset_index(drop=True)


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
                frames.append(_filter_region_level(df))
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
                    frames.append(_filter_region_level(df))
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
    years = list(range(2010, 2051))
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

        ax.set_xticks([year for year in range(2010, 2051, 10)])
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
    handles = handles[::-1]  # top-of-stack first, matching visual order top→bottom
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
    handles = handles[::-1]  # top-of-stack first, matching visual order top→bottom
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

    axes_top = _draw_row(fig, gs, 0, df_top, top_colors, y_range_top, show_titles=False)
    axes_bottom = _draw_row(fig, gs, 1, df_bottom, bottom_colors, y_range_bottom, show_titles=False)

    for ax in axes_top + axes_bottom:
        ax.set_xlabel('')

    # top=0.90, bottom=0.12, hspace=0.26, 2 rows:
    #   row_h = 0.78/2.26 ≈ 0.345,  gap_h = 0.090
    #   Row 0: 0.555–0.900,  gap: 0.465–0.555,  Row 1: 0.120–0.465
    fig.text(
        0.43, 0.935, 'Land-use',
        ha='center', va='bottom',
        fontsize=font_size, fontfamily='Arial', fontweight='normal',
    )
    fig.text(
        0.43, 0.503, 'Agricultural management',
        ha='center', va='bottom',
        fontsize=font_size, fontfamily='Arial', fontweight='normal',
    )
    _add_vertical_unit_label(fig, 0.038, 0.50, y_label, font_size)

    # Bold scenario column headers — placed just above the section label
    fig.canvas.draw()
    for ax, scenario in zip(axes_top, input_files):
        pos = ax.get_position()
        cx = (pos.x0 + pos.x1) / 2
        fig.text(cx, 0.968, SCENARIO_LABELS.get(scenario, scenario),
                 ha='center', va='bottom',
                 fontsize=font_size, fontweight='bold', fontfamily='Arial')

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
    fig.subplots_adjust(left=0.08, right=0.95, top=0.84, bottom=0.10)

    axes_overview = _draw_row(fig, gs, 0, df_overview, overview_colors, y_range_overview, show_titles=False)
    axes_top = _draw_row(fig, gs, 1, df_top, top_colors, y_range_top, show_titles=False)
    axes_bottom = _draw_row(fig, gs, 2, df_bottom, bottom_colors, y_range_bottom, show_titles=False)

    years = list(range(2010, 2051))
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

    # Use actual axes positions so labels sit just above each row without overlap
    fig.canvas.draw()
    pos_ov = axes_overview[0].get_position()
    pos_tp = axes_top[0].get_position()
    pos_bt = axes_bottom[0].get_position()

    _section_gap = 0.008   # gap between row top edge and section-label bottom
    _header_gap  = 0.038   # additional gap from section label to column-header bottom

    fig.text(
        0.43, pos_ov.y1 + _section_gap, 'Total',
        ha='center', va='bottom',
        fontsize=font_size, fontfamily='Arial', fontweight='normal',
    )
    fig.text(
        0.43, pos_tp.y1 + _section_gap, 'Land-use',
        ha='center', va='bottom',
        fontsize=font_size, fontfamily='Arial', fontweight='normal',
    )
    fig.text(
        0.43, pos_bt.y1 + _section_gap, 'Agricultural management',
        ha='center', va='bottom',
        fontsize=font_size, fontfamily='Arial', fontweight='normal',
    )
    _add_vertical_unit_label(fig, y_label_x, (pos_bt.y0 + pos_ov.y1) / 2, y_label, font_size)

    # Bold scenario column headers just above the Total section label
    header_y = pos_ov.y1 + _section_gap + _header_gap
    for ax, scenario in zip(axes_overview, input_files):
        pos = ax.get_position()
        cx = (pos.x0 + pos.x1) / 2
        fig.text(cx, header_y, SCENARIO_LABELS.get(scenario, scenario),
                 ha='center', va='bottom',
                 fontsize=font_size, fontweight='bold', fontfamily='Arial')

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
