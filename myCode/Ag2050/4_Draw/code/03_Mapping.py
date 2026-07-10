"""
03_Mapping.py — Combined SVG figure with three panels (a), (b), (c).
  (a) 2×2 grid (all 4 scenarios), 8-category LU with dryland/irrigated split
  (b) 1×2 grid (AgS1+AgS2 only), AG management types
  (c) single map (AgS2), non-ag land-use sub-types with side legend
"""
import os
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams.update({
    'font.family': 'Arial',
    'font.sans-serif': ['Arial'],
    'font.size': 9,
    'svg.fonttype': 'none',
})
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import rasterio

from tools.parameters import TIFF_DIR, OUTPUT_DIR, input_files, SCENARIO_LABELS, GENERATE_TABLES
from tools.data_helper import get_zip_info, extract_nc_layer_as_tiff
from tools.two_row_figure import export_long_tables, load_long_tables, missing_table_error

# ── Geography constants (EPSG:4283 / GDA94) ──────────────────────────────────
TIFF_LEFT, TIFF_BOTTOM, TIFF_RIGHT, TIFF_TOP = 112.925, -43.665, 153.625, -10.015
TIFF_CRS_EPSG = 4283
EXTENT = [TIFF_LEFT, TIFF_RIGHT, TIFF_BOTTOM, TIFF_TOP]

_HERE   = os.path.dirname(os.path.abspath(__file__))
ASSETS  = os.path.normpath(os.path.join(_HERE, '../../../draw_all/code/Assets'))
STE_SHP = os.path.join(ASSETS, 'AUS_adm/STE11aAust_mercator_simplified.shp')
WORKBOOK = '03_mapping_tiff_manifest.xlsx'
_TIFF_MANIFEST = None

# ── 8-category LU scheme ──────────────────────────────────────────────────────
CROPLAND = frozenset([0, 3, 4, 7, 8, 9, 10, 11, 12, 13,
                      16, 17, 18, 19, 20, 21, 24, 25, 26, 27])
MOD_PAST = frozenset([1, 5, 14])
NAT_PAST = frozenset([2, 6, 15])
UNALLOC  = frozenset([22, 23])

LU_CAT_8 = {
    1: ("Dryland cropland and horticulture",                              "#aecb75"),
    2: ("Irrigated cropland and horticulture",                            "#83b5ff"),
    3: ("Dryland grazing (modified pastures)",                            "#762400"),
    4: ("Irrigated grazing (modified pastures)",                          "#c4669b"),
    5: ("Grazing (native vegetation)",                                    "#c4996b"),
    6: ("Unallocated land",                                               "#e5d8a8"),
    7: ("Non-agricultural land-use",                                      "#3A7F4A"),
    8: ("Public and indigenous land, urban land, plantation forestry,\nand water bodies", "#bfbfbf"),
}

# Non-ag sub-type info (codes 100–108, from land use colors xlsx 'non_ag' sheet)
NON_AG_INFO = {
    100: ("Environmental plantings (mixed local native species)", "#267300"),
    101: ("Riparian buffer restoration (mixed species)",          "#005ce6"),
    102: ("Agroforestry (mixed species + sheep)",                 "#c500ff"),
    103: ("Agroforestry (mixed species + beef)",                  "#ff0000"),
    104: ("Carbon plantings (monoculture)",                       "#F2A361"),
    105: ("Farm forestry (hardwood timber + sheep)",              "#20B2AA"),
    106: ("Farm forestry (hardwood timber + beef)",               "#A0522D"),
    107: ("BECCS (Bioenergy with carbon capture and storage)",    "#FFFF00"),
    108: ("Destocked (natural land)",                             "#abcd66"),
}

NO_AM_COLOR = '#E0E0E0'

# ── Helpers ───────────────────────────────────────────────────────────────────
def _hex_rgb(h):
    return int(h[1:3], 16)/255, int(h[3:5], 16)/255, int(h[5:7], 16)/255


def _to_rgba(cat_arr, cat_info):
    rgba = np.zeros((*cat_arr.shape, 4), dtype='float32')
    for cid, (_, hexc) in cat_info.items():
        r, g, b = _hex_rgb(hexc)
        rgba[cat_arr == cid] = [r, g, b, 1.0]
    return rgba


def read_tiff(path):
    with rasterio.open(path) as src:
        arr = src.read(1).astype('float32')
        nd  = src.nodata if src.nodata is not None else -9999
        arr[arr == nd] = np.nan
    arr[arr < -9000] = np.nan
    return arr


def _style(ax):
    ax.set_xticks([]);  ax.set_yticks([])
    ax.set_xlim(TIFF_LEFT, TIFF_RIGHT)
    ax.set_ylim(TIFF_BOTTOM, TIFF_TOP)
    ax.set_aspect('equal', adjustable='box')
    for sp in ax.spines.values():
        sp.set_visible(False)


# ── Geography helpers ─────────────────────────────────────────────────────────
def load_states():
    if not os.path.exists(STE_SHP):
        print(f'State shp not found: {STE_SHP}')
        return None
    return gpd.read_file(STE_SHP).to_crs(epsg=TIFF_CRS_EPSG)


def fill_states(ax, gdf, color='#BFBFBF'):
    if gdf is not None:
        gdf.plot(ax=ax, color=color, zorder=0)


def add_states(ax, gdf, lw=0.4, color='#555555'):
    if gdf is not None:
        gdf.boundary.plot(ax=ax, color=color, linewidth=lw, zorder=4)


# ── Tiff extraction helpers ───────────────────────────────────────────────────
def _extract_or_get_tiff(scen, layer_name, sel, year, cache_key):
    cached = os.path.join(TIFF_DIR, f'_extracted_{scen}_{layer_name}_{cache_key}.tiff')
    if os.path.exists(cached):
        return cached
    return extract_nc_layer_as_tiff(scen, layer_name, sel, year)


def build_tiff_manifest():
    rows = []
    requests = []
    for scen in input_files:
        requests.extend([
            (scen, 'map_lumap', {'lm': 'ALL'}, 2050, 'lmALL'),
            (scen, 'map_lumap', {'lm': 'dry'}, 2050, 'lmDRY'),
            (scen, 'dvar_am', {'am': 'ALL', 'lm': 'ALL', 'lu': 'ALL'}, 2050, 'dvarAM'),
        ])

    for scen, layer_name, sel, year, cache_key in requests:
        path = _extract_or_get_tiff(scen, layer_name, sel, year, cache_key)
        rows.append({
            'scenario': scen,
            'year': int(year),
            'layer_name': layer_name,
            'cache_key': cache_key,
            'selector': repr(sel),
            'path': path or '',
        })
    return pd.DataFrame(rows)


def load_tiff_manifest():
    global _TIFF_MANIFEST
    _TIFF_MANIFEST = load_long_tables(WORKBOOK, 'tiffs')['tiffs']
    return _TIFF_MANIFEST


def _lookup_manifest_tiff(scen, layer_name, cache_key):
    if _TIFF_MANIFEST is None:
        return None
    match = _TIFF_MANIFEST[
        (_TIFF_MANIFEST['scenario'] == scen) &
        (_TIFF_MANIFEST['layer_name'] == layer_name) &
        (_TIFF_MANIFEST['cache_key'] == cache_key)
    ]
    if match.empty:
        raise missing_table_error(WORKBOOK, f'{scen} {layer_name} {cache_key}')
    path = str(match.iloc[0]['path'])
    if not path or path == 'nan' or not os.path.exists(path):
        raise FileNotFoundError(
            f'Missing TIFF listed in table cache: {path}\n'
            'Set GENERATE_TABLES = True in tools/parameters.py and run 03_Mapping.py once '
            'before plotting with GENERATE_TABLES = False.'
        )
    return path


def _get_tiff(scen, layer_name, sel, year, cache_key):
    manifest_path = _lookup_manifest_tiff(scen, layer_name, cache_key)
    if manifest_path is not None:
        return manifest_path
    return _extract_or_get_tiff(scen, layer_name, sel, year, cache_key)


def get_lu_all(scen):  return _get_tiff(scen, 'map_lumap', {'lm': 'ALL'}, 2050, 'lmALL')
def get_lu_dry(scen):  return _get_tiff(scen, 'map_lumap', {'lm': 'dry'}, 2050, 'lmDRY')
def get_am_tiff(scen): return _get_tiff(scen, 'dvar_am',   {'am': 'ALL', 'lm': 'ALL', 'lu': 'ALL'}, 2050, 'dvarAM')


# ── Classification helpers ────────────────────────────────────────────────────
def lumap_to_rgba_8cat(all_arr, dry_arr):
    """
    all_arr: lu codes for all LUTO research-area cells (lm='ALL')
    dry_arr: lu codes only for dryland-managed cells (lm='dry'); NaN for irrigated/non-ag.
    Dryland vs irrigated distinction: cell has code in dry_arr ↔ dryland management.
    """
    cat   = np.full(all_arr.shape, np.nan, dtype='float32')
    valid  = ~np.isnan(all_arr)
    is_dry = ~np.isnan(dry_arr)

    for code in CROPLAND:
        at = all_arr == code
        cat[at & is_dry]  = 1   # dryland cropland
        cat[at & ~is_dry] = 2   # irrigated cropland
    for code in MOD_PAST:
        at = all_arr == code
        cat[at & is_dry]  = 3   # dryland grazing (modified)
        cat[at & ~is_dry] = 4   # irrigated grazing (modified)
    for code in NAT_PAST:
        cat[all_arr == code] = 5
    for code in UNALLOC:
        cat[all_arr == code] = 6

    cat[valid & (all_arr >= 100)] = 7   # non-agricultural land-use

    cat[valid & np.isnan(cat)] = 8      # public/indigenous/urban/plantation/water

    return _to_rgba(cat, LU_CAT_8)


def nonag_to_rgba(all_arr):
    """Show non-ag sub-types (100–108); agricultural cells in #DADADA."""
    rgba = np.zeros((*all_arr.shape, 4), dtype='float32')
    ag_r, ag_g, ag_b = _hex_rgb('#DADADA')
    for code in (CROPLAND | MOD_PAST | NAT_PAST | UNALLOC):
        rgba[all_arr == code] = [ag_r, ag_g, ag_b, 1.0]
    for code, (_, hexc) in NON_AG_INFO.items():
        r, g, b = _hex_rgb(hexc)
        rgba[all_arr == code] = [r, g, b, 1.0]
    return rgba


def _load_am_info():
    df  = pd.read_excel('tools/land use colors.xlsx', sheet_name='am')
    col = 'desc_new' if 'desc_new' in df.columns else 'desc'
    rename = {
        "Human-Induced Regeneration (beef)":  "Managed regeneration (beef)",
        "Human-Induced Regeneration (sheep)": "Managed regeneration (sheep)",
    }
    return {
        int(r['code']): (rename.get(r[col], r[col]), r['color'])
        for _, r in df.iterrows()
    }


def ammap_to_rgba(arr, am_info):
    arr  = arr.copy()
    valid = ~np.isnan(arr)
    arr[valid] += 1          # 0-based index → 1-based am code
    cat  = np.full(arr.shape, np.nan, dtype='float32')
    for code in am_info:
        cat[arr == code] = float(code)
    cat[np.isnan(arr)] = np.nan
    return _to_rgba(cat, am_info)


# ── Figure A: 2×2 land-use maps ───────────────────────────────────────────────
def save_lu_maps(gdf_states):
    asp    = (TIFF_TOP - TIFF_BOTTOM) / (TIFF_RIGHT - TIFF_LEFT)  # ≈ 0.827
    MAP_W  = 4.4
    MAP_H  = MAP_W * asp
    HSPACE = 0.06
    WSPACE = 0.06
    LEG_H  = 1.6          # height reserved below maps for the horizontal legend

    ncols, nrows = 2, 2
    fig_w = ncols * MAP_W + (ncols - 1) * WSPACE
    fig_h = nrows * MAP_H + (nrows - 1) * HSPACE + LEG_H

    fig = plt.figure(figsize=(fig_w, fig_h), facecolor='white')
    gs  = gridspec.GridSpec(
        nrows, ncols, figure=fig,
        left=0, right=1,
        top=1,
        bottom=LEG_H / fig_h,
        wspace=WSPACE / MAP_W,
        hspace=HSPACE / MAP_H,
    )

    for idx, scen in enumerate(input_files):
        row, col = divmod(idx, ncols)
        ax = fig.add_subplot(gs[row, col])
        fill_states(ax, gdf_states)

        tp_all = get_lu_all(scen)
        tp_dry = get_lu_dry(scen)
        ok_all = tp_all and os.path.exists(tp_all)
        ok_dry = tp_dry and os.path.exists(tp_dry)
        if ok_all:
            all_arr = read_tiff(tp_all)
            dry_arr = read_tiff(tp_dry) if ok_dry else np.full_like(all_arr, np.nan)
            ax.imshow(lumap_to_rgba_8cat(all_arr, dry_arr),
                      extent=EXTENT, origin='upper', interpolation='nearest', zorder=1)

        add_states(ax, gdf_states)
        _style(ax)
        label = SCENARIO_LABELS.get(scen, scen).split('\n')[0]
        ax.set_title(label, fontsize=9, fontweight='bold', pad=3)

    # Horizontal legend: 2 rows × 4 cols for 8 categories
    handles = [mpatches.Patch(facecolor=hexc, label=lbl, edgecolor='none')
               for _, (lbl, hexc) in LU_CAT_8.items()]
    fig.legend(handles, [h.get_label() for h in handles],
               loc='upper center',
               bbox_to_anchor=(0.5, LEG_H / fig_h),
               bbox_transform=fig.transFigure,
               ncol=4, fontsize=8.5, frameon=False,
               handlelength=1.0, handleheight=1.0,
               handletextpad=0.4, columnspacing=1.0, borderpad=0)

    out = os.path.join(OUTPUT_DIR, '03a_landuse_maps.svg')
    fig.savefig(out, dpi=600, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'Saved: {out}')


# ── Figure B: 1×2 AG management maps (AgS1 + AgS2 only) ─────────────────────
def save_agmgt_maps(gdf_states):
    am_info  = _load_am_info()
    am_plot  = {k: v for k, v in am_info.items() if 1 <= k <= 8}

    asp    = (TIFF_TOP - TIFF_BOTTOM) / (TIFF_RIGHT - TIFF_LEFT)
    MAP_W  = 4.4
    MAP_H  = MAP_W * asp
    WSPACE = 0.06
    LEG_H  = 2.0

    scens_2 = input_files[:2]   # AgS1, AgS2
    fig_w   = 2 * MAP_W + WSPACE
    fig_h   = MAP_H + LEG_H

    fig = plt.figure(figsize=(fig_w, fig_h), facecolor='white')
    gs  = gridspec.GridSpec(
        1, 2, figure=fig,
        left=0, right=1,
        top=1, bottom=LEG_H / fig_h,
        wspace=WSPACE / MAP_W,
    )

    all_ag_codes = list(CROPLAND | MOD_PAST | NAT_PAST | UNALLOC)

    for col, scen in enumerate(scens_2):
        ax = fig.add_subplot(gs[0, col])
        fill_states(ax, gdf_states)

        tp_all = get_lu_all(scen)
        if tp_all and os.path.exists(tp_all):
            lu_arr   = read_tiff(tp_all)
            research = np.zeros(lu_arr.shape, dtype=bool)
            for code in all_ag_codes:
                research[lu_arr == code] = True
            no_am_rgba = np.zeros((*lu_arr.shape, 4), dtype='float32')
            r, g, b = _hex_rgb(NO_AM_COLOR)
            no_am_rgba[research] = [r, g, b, 1.0]
            ax.imshow(no_am_rgba, extent=EXTENT, origin='upper',
                      interpolation='nearest', zorder=1)

        tp_am = get_am_tiff(scen)
        if tp_am and os.path.exists(tp_am):
            ax.imshow(ammap_to_rgba(read_tiff(tp_am), am_plot),
                      extent=EXTENT, origin='upper', interpolation='nearest', zorder=2)

        add_states(ax, gdf_states)
        _style(ax)
        label = SCENARIO_LABELS.get(scen, scen).split('\n')[0]
        ax.set_title(label, fontsize=9, fontweight='bold', pad=3)

    handles = [
        *[mpatches.Patch(facecolor=hexc, label=lbl, edgecolor='none')
          for _, (lbl, hexc) in sorted(am_plot.items())],
        mpatches.Patch(facecolor=NO_AM_COLOR, edgecolor='none',
                       label='No agricultural management'),
    ]
    fig.legend(handles, [h.get_label() for h in handles],
               loc='upper center',
               bbox_to_anchor=(0.5, LEG_H / fig_h),
               bbox_transform=fig.transFigure,
               ncol=3, fontsize=8.5, frameon=False,
               handlelength=1.0, handleheight=1.0,
               handletextpad=0.4, columnspacing=1.0, borderpad=0)

    out = os.path.join(OUTPUT_DIR, '03b_agmgt_maps.svg')
    fig.savefig(out, dpi=600, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'Saved: {out}')


# ── Figure C: single non-ag map (AgS2) with side legend ──────────────────────
def save_nonag_map(gdf_states):
    asp    = (TIFF_TOP - TIFF_BOTTOM) / (TIFF_RIGHT - TIFF_LEFT)
    MAP_W  = 5.0
    MAP_H  = MAP_W * asp
    LEG_W  = 3.5

    scen  = input_files[1]   # AgS2: tree-planting scenario
    fig_w = MAP_W + LEG_W
    fig_h = MAP_H

    fig = plt.figure(figsize=(fig_w, fig_h), facecolor='white')
    gs  = gridspec.GridSpec(
        1, 2, figure=fig,
        left=0, right=1, top=1, bottom=0,
        width_ratios=[MAP_W, LEG_W],
        wspace=0.05,
    )

    ax_map = fig.add_subplot(gs[0, 0])
    fill_states(ax_map, gdf_states)

    tp = get_lu_all(scen)
    if tp and os.path.exists(tp):
        arr  = read_tiff(tp)
        rgba = nonag_to_rgba(arr)
        ax_map.imshow(rgba, extent=EXTENT, origin='upper',
                      interpolation='nearest', zorder=1)

    add_states(ax_map, gdf_states)
    _style(ax_map)
    label = SCENARIO_LABELS.get(scen, scen).split('\n')[0]
    ax_map.set_title(label, fontsize=9, fontweight='bold', pad=3)

    # Legend in the right panel (vertical / columnar)
    ax_leg = fig.add_subplot(gs[0, 1])
    ax_leg.axis('off')
    handles = [mpatches.Patch(facecolor=hexc, label=lbl, edgecolor='none')
               for _, (lbl, hexc) in NON_AG_INFO.items()]
    ax_leg.legend(handles, [h.get_label() for h in handles],
                  loc='center left', fontsize=8.5, frameon=False,
                  handlelength=1.0, handleheight=1.0,
                  handletextpad=0.4, labelspacing=0.6, borderpad=0)

    out = os.path.join(OUTPUT_DIR, '03c_nonag_map.svg')
    fig.savefig(out, dpi=600, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'Saved: {out}')


# ── Combined figure: panels (a), (b), (c) in one SVG ────────────────────────
def save_combined_maps(gdf_states):
    FONT_SIZE = 9
    asp = (TIFF_TOP - TIFF_BOTTOM) / (TIFF_RIGHT - TIFF_LEFT)  # ≈ 0.8268

    MAP_H  = 3.5
    MAP_W  = MAP_H / asp      # ≈ 4.23"
    WGAP   = 0.08             # horizontal gap between map columns
    HGAP   = 0.08             # vertical gap between map rows in panel (a)
    LEG_H_A = 1.05            # panel (a) legend height
    LEG_H_B = 1.25            # panel (b) legend height
    GAP_H  = 0.02             # blank gap between panels
    LEG_W_C = 3.8             # panel (c) side-legend width

    FIG_W = 2 * MAP_W + WGAP
    hr    = [2 * MAP_H + HGAP, LEG_H_A, GAP_H, MAP_H, LEG_H_B, GAP_H, MAP_H]
    FIG_H = sum(hr)

    fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor='white')
    outer_gs = gridspec.GridSpec(
        7, 1, figure=fig,
        left=0.0, right=1.0, top=1.0, bottom=0.0,
        height_ratios=hr, hspace=0.0,
    )

    # ── Panel (a): 2×2 LU maps ────────────────────────────────────────────────
    gs_a = gridspec.GridSpecFromSubplotSpec(
        2, 2, subplot_spec=outer_gs[0],
        hspace=HGAP / MAP_H, wspace=WGAP / MAP_W,
    )
    first_a = None
    for idx, scen in enumerate(input_files):
        row, col = divmod(idx, 2)
        ax = fig.add_subplot(gs_a[row, col])
        if first_a is None:
            first_a = ax
        fill_states(ax, gdf_states)
        tp_all = get_lu_all(scen)
        tp_dry = get_lu_dry(scen)
        ok_all = tp_all and os.path.exists(tp_all)
        ok_dry = tp_dry and os.path.exists(tp_dry)
        if ok_all:
            all_arr = read_tiff(tp_all)
            dry_arr = read_tiff(tp_dry) if ok_dry else np.full_like(all_arr, np.nan)
            ax.imshow(lumap_to_rgba_8cat(all_arr, dry_arr),
                      extent=EXTENT, origin='upper', interpolation='nearest', zorder=1)
        add_states(ax, gdf_states)
        _style(ax)
        label = SCENARIO_LABELS.get(scen, scen).split('\n')[0]
        ax.set_title(label, fontsize=FONT_SIZE, fontweight='bold', pad=3)

    ax_leg_a = fig.add_subplot(outer_gs[1])
    ax_leg_a.axis('off')
    handles_a = [mpatches.Patch(facecolor=hexc, label=lbl, edgecolor='none')
                 for k, (lbl, hexc) in LU_CAT_8.items() if k != 8]
    ax_leg_a.legend(handles_a, [h.get_label() for h in handles_a],
                    loc='upper center', ncol=4, fontsize=FONT_SIZE, frameon=False,
                    handlelength=1.0, handleheight=1.0,
                    handletextpad=0.4, columnspacing=1.0, borderpad=0)

    # ── Panel (b): 1×2 AM maps ────────────────────────────────────────────────
    am_info = _load_am_info()
    am_plot = {k: v for k, v in am_info.items() if 1 <= k <= 8}
    all_ag_codes = list(CROPLAND | MOD_PAST | NAT_PAST | UNALLOC)

    gs_b = gridspec.GridSpecFromSubplotSpec(
        1, 2, subplot_spec=outer_gs[3],
        wspace=WGAP / MAP_W,
    )
    first_b = None
    for col, scen in enumerate(input_files[:2]):
        ax = fig.add_subplot(gs_b[0, col])
        if first_b is None:
            first_b = ax
        fill_states(ax, gdf_states)
        tp_all = get_lu_all(scen)
        if tp_all and os.path.exists(tp_all):
            lu_arr = read_tiff(tp_all)
            research = np.zeros(lu_arr.shape, dtype=bool)
            for code in all_ag_codes:
                research[lu_arr == code] = True
            no_am_rgba = np.zeros((*lu_arr.shape, 4), dtype='float32')
            rc, gc, bc = _hex_rgb(NO_AM_COLOR)
            no_am_rgba[research] = [rc, gc, bc, 1.0]
            ax.imshow(no_am_rgba, extent=EXTENT, origin='upper',
                      interpolation='nearest', zorder=1)
        tp_am = get_am_tiff(scen)
        if tp_am and os.path.exists(tp_am):
            ax.imshow(ammap_to_rgba(read_tiff(tp_am), am_plot),
                      extent=EXTENT, origin='upper', interpolation='nearest', zorder=2)
        add_states(ax, gdf_states)
        _style(ax)
        label = SCENARIO_LABELS.get(scen, scen).split('\n')[0]
        ax.set_title(label, fontsize=FONT_SIZE, fontweight='bold', pad=3)

    ax_leg_b = fig.add_subplot(outer_gs[4])
    ax_leg_b.axis('off')
    handles_b = [
        *[mpatches.Patch(facecolor=hexc, label=lbl, edgecolor='none')
          for _, (lbl, hexc) in sorted(am_plot.items())],
        mpatches.Patch(facecolor=NO_AM_COLOR, edgecolor='none',
                       label='No agricultural management'),
    ]
    ax_leg_b.legend(handles_b, [h.get_label() for h in handles_b],
                    loc='upper center', ncol=3, fontsize=FONT_SIZE, frameon=False,
                    handlelength=1.0, handleheight=1.0,
                    handletextpad=0.4, columnspacing=1.0, borderpad=0)

    # ── Panel (c): single non-ag map (AgS2) + side legend ────────────────────
    gs_c = gridspec.GridSpecFromSubplotSpec(
        1, 2, subplot_spec=outer_gs[6],
        width_ratios=[FIG_W - LEG_W_C - WGAP, LEG_W_C],
        wspace=WGAP / MAP_W,
    )
    scen_c = input_files[1]   # AgS2
    ax_c_map = fig.add_subplot(gs_c[0, 0])
    fill_states(ax_c_map, gdf_states, color='#B7B7B7')
    tp_c = get_lu_all(scen_c)
    if tp_c and os.path.exists(tp_c):
        arr  = read_tiff(tp_c)
        rgba = nonag_to_rgba(arr)
        ax_c_map.imshow(rgba, extent=EXTENT, origin='upper',
                        interpolation='nearest', zorder=1)
    add_states(ax_c_map, gdf_states)
    _style(ax_c_map)
    label_c = SCENARIO_LABELS.get(scen_c, scen_c).split('\n')[0]
    ax_c_map.set_title(label_c, fontsize=FONT_SIZE, fontweight='bold', pad=3)
    ax_c_leg = fig.add_subplot(gs_c[0, 1])
    ax_c_leg.axis('off')
    handles_c = [mpatches.Patch(facecolor=hexc, label=lbl, edgecolor='none')
                 for _, (lbl, hexc) in NON_AG_INFO.items()]
    handles_c.append(mpatches.Patch(facecolor='#DADADA', label='Agricultural land-use',
                                    edgecolor='none'))
    handles_c.append(mpatches.Patch(
        facecolor='#B7B7B7',
        label='Public and indigenous land, urban land,\nplantation forestry, and water bodies',
        edgecolor='none'))
    ax_c_leg.legend(handles_c, [h.get_label() for h in handles_c],
                    loc='center left', fontsize=FONT_SIZE, frameon=False,
                    handlelength=1.0, handleheight=1.0,
                    handletextpad=0.4, labelspacing=0.6, borderpad=0)

    fig.canvas.draw()
    for ax, title in [(first_a, 'Agricultural land-use'),
                      (first_b, 'Agricultural management'),
                      (ax_c_map, 'Non-agricultural land-use')]:
        pos = ax.get_position()
        fig.text(0.5, pos.y1 + 0.012, title,
                 ha='center', va='bottom',
                 fontsize=FONT_SIZE + 4, fontfamily='Arial', fontweight='bold')

    out = os.path.join(OUTPUT_DIR, '03_maps.svg')
    fig.savefig(out, dpi=600, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'Saved: {out}')


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(TIFF_DIR, exist_ok=True)
    if GENERATE_TABLES:
        export_long_tables(WORKBOOK, tiffs=build_tiff_manifest())
    load_tiff_manifest()
    gdf_states = load_states()
    save_combined_maps(gdf_states)


if __name__ == '__main__':
    main()
