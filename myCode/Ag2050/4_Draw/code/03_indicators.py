"""
04_indicators.py
5-row × 4-column overview indicator figure.
Rows:    NER, GHG, Biodiversity, Agri-food, Water
Columns: 4 scenarios (AgS1–AgS4)
"""
import _path_setup  # noqa: F401

import os
import shutil
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.gridspec as gridspec
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd

from tools.two_row_figure import (
    COMMODITY_TO_FOOD_GROUP,
    FOOD_PRODUCTION_CSV,
    FOOD_PRODUCTION_VALUE_COL,
    RENAME_AM_NON_AG,
    export_long_tables,
    filter_food_detail_rows,
    filter_water_detail_rows,
    input_files,
    load_long_tables,
    load_report_source_csv,
    prepare_biodiversity_overview,
    prepare_ghg_overview,
    prepare_net_economic_return_overview,
)
from tools.data_helper import extract_nc_layer_as_tiff, list_output_years, load_output_dataset
from tools.parameters import EXCEL_DIR, OUTPUT_DIR, SCENARIO_LABELS, TIF_DIR, font_size, GENERATE_TABLES
from tools.plot_helper import calc_y_range, set_plot_style, stacked_area_pos_neg

# ── Colors ────────────────────────────────────────────────────────────────────

NER_COLORS = {
    'Ag cost':                         '#fab431',
    'Agmgt cost':                      '#ec7951',
    'Non-ag cost':                     '#cd4975',
    'Transition(ag->non-ag) cost':     '#6200ac',
    'Transition(non-ag->ag) cost':     '#7b4ab4',
    'Transition(ag->ag) cost':         '#9f0e9e',
    'Ag revenue':                      '#2d688f',
    'Agmgt revenue':                   '#19928e',
    'Non-ag revenue':                  '#35b876',
}
# Legend order: top of chart → bottom (positives reversed, then negatives in stacking order)
NER_LEGEND_ORDER = [
    'Non-ag revenue', 'Agmgt revenue', 'Ag revenue',
    'Ag cost', 'Agmgt cost', 'Non-ag cost',
    'Transition(ag->non-ag) cost', 'Transition(non-ag->ag) cost',
    'Transition(ag->ag) cost',
]

GHG_COLORS = {
    'Non-agricultural land-use':  '#6eabb1',
    'Agricultural management':    '#9A8AB3',
    'Agricultural land-use':      '#f39b8b',
    'Transition':                 '#eb9132',
    'Off-land commodities':       '#7f7f7f',
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
    ('Biodiversity', BIO_COLORS,   'Biodiversity contribution-\nweighted area (Mha)', None),
    ('Agri-food',    FOOD_COLORS,  'Agri-food production\n(Mt yr⁻¹)',     FOOD_LEGEND_ORDER),
    ('Water',        WATER_COLORS, 'Difference in water yield\nrelative to 2010 (GL yr⁻¹)', WATER_LEGEND_ORDER),
]

# ── Data preparation ───────────────────────────────────────────────────────────

def prepare_ner():
    return prepare_net_economic_return_overview()


def prepare_ghg():
    return prepare_ghg_overview()


def prepare_bio():
    return prepare_biodiversity_overview()


def prepare_food():
    rows = []
    for scenario in input_files:
        food = filter_food_detail_rows(
            load_report_source_csv(scenario, FOOD_PRODUCTION_CSV)
        )
        if food.empty:
            continue
        food['category'] = food['Commodity'].map(COMMODITY_TO_FOOD_GROUP)
        food = food.dropna(subset=['category'])
        data = food.groupby(['Year', 'category'], as_index=False)[FOOD_PRODUCTION_VALUE_COL].sum()
        for _, row in data.iterrows():
            rows.append({'year': int(row['Year']), 'scenario': scenario,
                         'category': row['category'],
                         'value': float(row[FOOD_PRODUCTION_VALUE_COL]) / 1e6})
    return pd.DataFrame(rows)


def _load_water_correction(scenario):
    """
    Per-cell attribution correction for water yield.

    For each year, identifies cells that have non-ag land-use and returns their
    total 2010 ag water yield split into (dryland_ML, irrigated_ML).

    This correction re-attributes the baseline yield from "Dryland/Irrigated
    agriculture" to "Non-agricultural land-use" so the non-ag delta correctly
    shows the net effect of land conversion (typically negative for tree planting).
    """
    ds = load_output_dataset(scenario, 2010, 'xr_water_yield_ag_2010.nc')
    if ds is None:
        return {}
    try:
        data = ds['data'].values   # (cell, layer)
        n_lu = len(ds.lu)          # 29: 'ALL' + 28 specific land-uses
        # layer = lm_idx * n_lu + lu_idx  (lm: 0=ALL, 1=dry, 2=irr; lu_idx=0 for ALL)
        ag10_dry = data[:, n_lu]       # lm=dry, lu=ALL
        ag10_irr = data[:, 2 * n_lu]   # lm=irr, lu=ALL
    except Exception:
        return {}
    finally:
        ds.close()

    corrections = {}
    for year in list_output_years(scenario):
        if year == 2010:
            corrections[year] = (0.0, 0.0)
            continue
        ds_nag = load_output_dataset(scenario, year, f'xr_water_yield_non_ag_{year}.nc')
        if ds_nag is None:
            corrections[year] = (0.0, 0.0)
            continue
        try:
            mask = ds_nag['data'].values[:, 0] > 0   # lu=ALL layer; True where non-ag exists
            corrections[year] = (float(ag10_dry[mask].sum()), float(ag10_irr[mask].sum()))
        except Exception:
            corrections[year] = (0.0, 0.0)
        finally:
            ds_nag.close()
    return corrections


_CLIMATE_WATER_CACHE = None
CLIMATE_WATER_CSV = '03_water_climate_impact_split.csv'


def _compute_climate_water_impact():
    """Compute the pure dry/irr climate impact on water yield (GL) per (scenario, year).

    HEAVY: loads each run's raw Data_RES*.lz4 object (the joblib dump of the full
    luto.data.Data), which holds the per-cell water-yield rate matrices that are absent
    from every report CSV. Uses water YIELD only (not net yield), holding land use fixed
    at the 2010 agricultural dvar, so the result is the true scenario-invariant climate
    signal (≈ −7,668 GL yr⁻¹ by 2050); the livestock water-requirement change that
    write.py folds into its reported CCI is deliberately excluded. Writes the result to
    EXCEL_DIR/03_water_climate_impact_split.csv and returns the DataFrame.

    Requires the `luto` package importable (xpluto env) and the Data_RES*.lz4 files.
    Only invoked when the cache CSV is missing (one-time, ~2 min).
    """
    import gc
    import sys
    import zipfile
    import joblib
    from tools.data_helper import get_zip_info

    repo = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..'))
    if repo not in sys.path:
        sys.path.insert(0, repo)
    import luto.settings as settings
    settings.AG2050_MODE = True   # Paper3 runs are Ag2050 mode
    import luto.data  # noqa: F401  (register classes for unpickling)
    import luto.economics.agricultural.water as ag_water

    os.makedirs(EXCEL_DIR, exist_ok=True)
    tmp = os.path.join(EXCEL_DIR, '_tmp_Data.lz4')
    rows = []
    for scenario in input_files:
        info = get_zip_info(scenario)
        if info is None:
            continue
        zip_path, _prefix = info
        with zipfile.ZipFile(zip_path) as z:
            member = next((n for n in z.namelist() if n.endswith('.lz4') and 'Data_RES' in n), None)
            if member is None:
                continue
            with z.open(member) as src, open(tmp, 'wb') as dst:
                while True:
                    chunk = src.read(1 << 24)
                    if not chunk:
                        break
                    dst.write(chunk)
        data = joblib.load(tmp)
        base_wy = ag_water.get_wyield_matrices(data, 0)   # 2010 climate-driven yield rates × area
        ag_2010 = data.AG_L_MRJ                            # fixed 2010 agricultural dvar
        for year in sorted(data.ag_dvars):
            yr_idx = year - data.YR_CAL_BASE
            wy = ag_water.get_wyield_matrices(data, yr_idx)
            clim = (wy - base_wy) * ag_2010                # pure climate, land use fixed at 2010
            rows.append({'scenario': scenario, 'year': int(year),
                         'dry_climate_GL': float(clim[0].sum()) / 1e3,   # m=0 dryland
                         'irr_climate_GL': float(clim[1].sum()) / 1e3})  # m=1 irrigated
        del data
        gc.collect()
    if os.path.exists(tmp):
        os.remove(tmp)

    df = pd.DataFrame(rows).sort_values(['scenario', 'year']).reset_index(drop=True)
    df.to_csv(os.path.join(EXCEL_DIR, CLIMATE_WATER_CSV), index=False)
    return df


def _load_climate_water_impact(force_regenerate=False):
    """Pure dryland/irrigated climate impact on water yield (GL), per (scenario, year).

    Reads the per-task-root cache EXCEL_DIR/03_water_climate_impact_split.csv; if it is
    missing (e.g. first run on a new TASK_ROOT), computes it once from the raw
    Data_RES*.lz4 objects via _compute_climate_water_impact(). Cached at module level.
    """
    global _CLIMATE_WATER_CACHE
    if force_regenerate or _CLIMATE_WATER_CACHE is None:
        path = os.path.join(EXCEL_DIR, CLIMATE_WATER_CSV)
        if force_regenerate:
            print(
                f'Regenerating {CLIMATE_WATER_CSV} from current Run_Archive.zip data...',
                flush=True,
            )
            df = _compute_climate_water_impact()
        elif os.path.exists(path):
            df = pd.read_csv(path)
        else:
            print(f'{CLIMATE_WATER_CSV} not found — computing from raw Data_RES*.lz4 '
                  '(one-time, ~2 min; needs xpluto env)...', flush=True)
            df = _compute_climate_water_impact()
        _CLIMATE_WATER_CACHE = {
            (r['scenario'], int(r['year'])): (float(r['dry_climate_GL']), float(r['irr_climate_GL']))
            for _, r in df.iterrows()
        }
    return _CLIMATE_WATER_CACHE


def prepare_water():
    rows = []
    vcol = 'Water Net Yield (ML)'
    climate = _load_climate_water_impact()
    for scenario in input_files:
        water = load_report_source_csv(scenario, 'water_yield_separate_watershed')
        if water.empty:
            continue
        water = filter_water_detail_rows(water).replace(RENAME_AM_NON_AG)

        irr    = water.query('Type == "Agricultural Land-use" and `Water Supply` == "Irrigated"').groupby('Year', as_index=False)[vcol].sum()
        dry    = water.query('Type == "Agricultural Land-use" and `Water Supply` == "Dryland"').groupby('Year', as_index=False)[vcol].sum()
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
        corrections = _load_water_correction(scenario)

        # The raw "Water Net Yield" deltas above already have the climate effect baked in.
        # We carve out only the PURE climate signal (water-yield change, land use held at
        # the 2010 agricultural dvar), split dry/irr, from _load_climate_water_impact(). This signal
        # is scenario-invariant (≈ −7,668 GL yr⁻¹ by 2050). The livestock water-requirement
        # change that write.py folds into its reported CCI is deliberately NOT carved out —
        # it is a management/stocking effect, not climate, so it stays inside Dryland/
        # Irrigated agriculture. A proportional split is avoided because irrigated
        # agriculture's 2010 net yield is negative, which would wrong-sign the irrigated share.
        years = sorted(set(irr_d) | set(dry_d) | set(agmgt_d) | set(non_ag_d)
                       | {y for (s, y) in climate if s == scenario})
        for year in years:
            dry_corr, irr_corr = corrections.get(int(year), (0.0, 0.0))
            dry_clim, irr_clim = climate.get((scenario, int(year)), (0.0, 0.0))  # GL, pure climate
            rows.append({'year': int(year), 'scenario': scenario,
                         'category': 'Irrigated agriculture',
                         'value': (irr_d.get(year, 0.0) - irr_2010 + irr_corr) / 1e3 - irr_clim})
            rows.append({'year': int(year), 'scenario': scenario,
                         'category': 'Dryland agriculture',
                         'value': (dry_d.get(year, 0.0) - dry_2010 + dry_corr) / 1e3 - dry_clim})
            rows.append({'year': int(year), 'scenario': scenario,
                         'category': 'Agricultural management',
                         'value': (agmgt_d.get(year, 0.0) - agmgt_2010) / 1e3})
            rows.append({'year': int(year), 'scenario': scenario,
                         'category': 'Non-agricultural land-use',
                         'value': (non_ag_d.get(year, 0.0) - dry_corr - irr_corr) / 1e3})
            rows.append({'year': int(year), 'scenario': scenario,
                         'category': 'Climate change impact',
                         'value': dry_clim + irr_clim})

    return pd.DataFrame(rows)


# ── Land-use GeoTIFF export ─────────────────────────────────────────────────────

def export_landuse_tifs():
    """Export the land-use map (lm='ALL') as a GeoTIFF, written directly into TIF_DIR
    (no intermediate cache folder).

    2010 is the fixed historical base year (identical across scenarios), so it is
    exported once. 2050 is scenario-specific and exported per scenario.
    """
    os.makedirs(TIF_DIR, exist_ok=True)

    def _extract_and_rename(scenario, year, dst_name):
        src = extract_nc_layer_as_tiff(scenario, 'map_lumap', {'lm': 'ALL'}, year, output_dir=TIF_DIR)
        if src is None:
            print(f'Land-use tif not found: {scenario} {year}')
            return
        dst = os.path.join(TIF_DIR, dst_name)
        if os.path.abspath(src) != os.path.abspath(dst):
            shutil.move(src, dst)
        print(f'Saved: {dst}')

    _extract_and_rename(input_files[0], 2010, 'landuse_2010.tif')
    for scenario in input_files:
        _extract_and_rename(scenario, 2050, f'landuse_{scenario}_2050.tif')


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

    out = os.path.join(OUTPUT_DIR, '03_indicators.svg')
    fig.savefig(out, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {out}')


def main():
    workbook = '03_indicators_long_tables.xlsx'
    if GENERATE_TABLES:
        _load_climate_water_impact(force_regenerate=True)
        export_long_tables(
            workbook,
            net_economic_return=prepare_ner(),
            ghg=prepare_ghg(),
            biodiversity=prepare_bio(),
            food=prepare_food(),
            water=prepare_water(),
        )
    tables = load_long_tables(
        workbook,
        'net_economic_return',
        'ghg',
        'biodiversity',
        'food',
        'water',
    )
    ner   = tables['net_economic_return']
    ghg   = tables['ghg']
    bio   = tables['biodiversity']
    food  = tables['food']
    water = tables['water']
    save_indicators_figure([ner, ghg, bio, food, water])
    if GENERATE_TABLES:
        export_landuse_tifs()


if __name__ == '__main__':
    main()
