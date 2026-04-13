"""
04_Mapping.py
4-row x 2-col figure: land use (col 1) and agricultural management (col 2)
AM data: xr_dvar_am_2050.nc (values are 0-based am indices; +1 converts to am codes 1-8).
State boundaries overlaid on every panel.
"""
import os
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams.update({
    'font.family': 'Arial',
    'font.size':   9,
})
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import rasterio

from tools.parameters import TIFF_DIR, OUTPUT_DIR, input_files, SCENARIO_LABELS
from tools.data_helper import get_zip_info, extract_nc_layer_as_tiff

NO_AM_COLOR = '#E0E0E0'   # lighter grey: research area cells with no AM applied

# ── Geography constants (from tiff metadata, EPSG:4283) ───────────────────────
TIFF_LEFT, TIFF_BOTTOM, TIFF_RIGHT, TIFF_TOP = 112.925, -43.665, 153.625, -10.015
TIFF_CRS_EPSG = 4283
IMG_H, IMG_W   = 673, 814
EXTENT = [TIFF_LEFT, TIFF_RIGHT, TIFF_BOTTOM, TIFF_TOP]  # imshow extent

# ── Assets dir (state-boundary shapefile) ────────────────────────────────────
_HERE   = os.path.dirname(os.path.abspath(__file__))         # code/
ASSETS  = os.path.normpath(os.path.join(_HERE, '../../../draw_all/code/Assets'))
STE_SHP = os.path.join(ASSETS, 'AUS_adm/STE11aAust_mercator_simplified.shp')

# ── Land-use category mapping ─────────────────────────────────────────────────
LU_CAT_MAP = {
    **{k: 1 for k in [0, 3, 4, 7, 8, 9, 10, 11, 12, 13,
                       16, 17, 18, 19, 20, 21, 24, 25, 26, 27]},
    **{k: 2 for k in [1, 5, 14]},
    **{k: 3 for k in [2, 6, 15]},
    **{k: 4 for k in [22, 23]},
    **{k: 5 for k in range(100, 120)},
}
LU_CAT_INFO = {
    1: ("Cropland and horticulture",     "#AECB75"),
    2: ("Grazing (modified pastures)",   "#762400"),
    3: ("Grazing (native vegetation)",   "#C4996B"),
    4: ("Unallocated land",              "#E5D8A8"),
    5: ("Non-agricultural land",         "#3A7F4A"),
    6: ("Public and indigenous land, urban land, plantation forestry, and water bodies", "#BFBFBF"),
}


def _load_am_info():
    df  = pd.read_excel('tools/land use colors.xlsx', sheet_name='am')
    col = 'desc_new' if 'desc_new' in df.columns else 'desc'
    return {int(r['code']): (r[col], r['color']) for _, r in df.iterrows()}


def _hex_rgb(h):
    return int(h[1:3], 16)/255, int(h[3:5], 16)/255, int(h[5:7], 16)/255


# ── Raster helpers ────────────────────────────────────────────────────────────

def read_tiff(path):
    with rasterio.open(path) as src:
        arr = src.read(1).astype('float32')
        nd  = src.nodata if src.nodata is not None else -9999
        arr[arr == nd] = np.nan
    arr[arr < -9000] = np.nan       # catches float32 overflow from int NaN
    return arr


def _to_rgba(cat_arr, cat_info):
    rgba = np.zeros((*cat_arr.shape, 4), dtype='float32')
    for cid, (_, hexc) in cat_info.items():
        r, g, b = _hex_rgb(hexc)
        rgba[cat_arr == cid] = [r, g, b, 1.0]
    return rgba


def lumap_to_rgba(arr):
    cat = np.full(arr.shape, np.nan, dtype='float32')
    for code, cid in LU_CAT_MAP.items():
        cat[arr == code] = cid
    cat[(~np.isnan(arr)) & np.isnan(cat)] = 6
    cat[np.isnan(arr)] = np.nan
    return _to_rgba(cat, LU_CAT_INFO)


def ammap_to_rgba(arr, am_info):
    arr = arr.copy()
    valid = ~np.isnan(arr)
    # xr_dvar_am stores 0-based am index; add 1 to convert to actual am codes (1-8)
    arr[valid] += 1
    cat = np.full(arr.shape, np.nan, dtype='float32')
    for code in am_info:
        cat[arr == code] = float(code)
    cat[np.isnan(arr)] = np.nan
    return _to_rgba(cat, am_info)


# ── Tiff path helpers ─────────────────────────────────────────────────────────

def get_lu_tiff(scen):
    cached = os.path.join(TIFF_DIR, f'_extracted_{scen}_map_lumap_lmALL.tiff')
    if os.path.exists(cached):
        return cached
    from tools.data_helper import extract_nc_layer_as_tiff
    return extract_nc_layer_as_tiff(scen, 'map_lumap', {'lm': 'ALL'}, 2050)


def get_am_tiff(scen):
    """Use xr_dvar_am_2050.nc (values are 0-based am indices; +1 in ammap_to_rgba)."""
    return extract_nc_layer_as_tiff(scen, 'dvar_am',
                                    {'am': 'ALL', 'lm': 'ALL', 'lu': 'ALL'}, 2050)


# ── State boundaries ──────────────────────────────────────────────────────────

def load_states():
    if not os.path.exists(STE_SHP):
        print(f'State shp not found: {STE_SHP}')
        return None
    return gpd.read_file(STE_SHP).to_crs(epsg=TIFF_CRS_EPSG)


def fill_states(ax, gdf, color='#BFBFBF'):
    """Fill state polygons with grey as background (shows non-LUTO land areas)."""
    if gdf is not None:
        gdf.plot(ax=ax, color=color, zorder=0)


def add_states(ax, gdf, lw=0.4, color='#555555'):
    if gdf is not None:
        gdf.boundary.plot(ax=ax, color=color, linewidth=lw, zorder=4)


# ── Figure ────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(TIFF_DIR, exist_ok=True)

    am_info    = _load_am_info()
    gdf_states = load_states()

    # ── Step 1: extract all 8 tiffs first ────────────────────────────────────
    tiff_lu = {}
    tiff_am = {}
    for scen in input_files:
        tiff_lu[scen] = get_lu_tiff(scen)
        tiff_am[scen] = get_am_tiff(scen)
        lu_ok = tiff_lu[scen] and os.path.exists(tiff_lu[scen])
        am_ok = tiff_am[scen] and os.path.exists(tiff_am[scen])
        print(f'{scen}: lu={tiff_lu[scen] if lu_ok else "MISSING"}, '
              f'am={tiff_am[scen] if am_ok else "MISSING"}')

    n           = len(input_files)
    asp         = (TIFF_TOP - TIFF_BOTTOM) / (TIFF_RIGHT - TIFF_LEFT)  # ≈ 0.827

    MAP_W_IN   = 4.6
    MAP_H_IN   = MAP_W_IN * asp
    LABEL_W_IN = 0.24
    WSPACE_IN  = 0.06
    HSPACE_IN  = 0.06
    TOP_IN     = 0.25
    BOT_IN     = 1.95

    fig_w = LABEL_W_IN + MAP_W_IN + WSPACE_IN + MAP_W_IN + 0.15
    fig_h = TOP_IN + n * MAP_H_IN + (n - 1) * HSPACE_IN + BOT_IN

    fig = plt.figure(figsize=(fig_w, fig_h), facecolor='white')

    gs = gridspec.GridSpec(
        n, 3,
        figure=fig,
        width_ratios=[LABEL_W_IN / MAP_W_IN, 1.0, 1.0],
        left   = 0.0,
        right  = 1.0,
        top    = 1.0 - TOP_IN  / fig_h,
        bottom = BOT_IN / fig_h,
        wspace = WSPACE_IN / MAP_W_IN,
        hspace = HSPACE_IN / MAP_H_IN,
    )

    for row, scen in enumerate(input_files):
        short = SCENARIO_LABELS.get(scen, scen).split('\n')[0]

        # ── Row label ────────────────────────────────────────────────────────
        ax_lbl = fig.add_subplot(gs[row, 0])
        ax_lbl.set_axis_off()
        ax_lbl.text(0.68, 0.5, short,
                    ha='center', va='center',
                    transform=ax_lbl.transAxes,
                    fontsize=9, rotation=90,
                    clip_on=False)

        def _style(ax):
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_xlim(TIFF_LEFT, TIFF_RIGHT)
            ax.set_ylim(TIFF_BOTTOM, TIFF_TOP)
            ax.set_aspect('equal', adjustable='box')
            for sp in ax.spines.values():
                sp.set_visible(True)
                sp.set_linewidth(0.7)
                sp.set_edgecolor('#444444')

        # ── Land-use column ──────────────────────────────────────────────────
        ax_lu = fig.add_subplot(gs[row, 1])
        fill_states(ax_lu, gdf_states)           # grey background for non-LUTO land
        tp = tiff_lu[scen]
        if tp and os.path.exists(tp):
            ax_lu.imshow(lumap_to_rgba(read_tiff(tp)),
                         extent=EXTENT, origin='upper', interpolation='nearest',
                         zorder=1)
        add_states(ax_lu, gdf_states)
        _style(ax_lu)

        # ── AM column ────────────────────────────────────────────────────────
        ax_am = fig.add_subplot(gs[row, 2])
        # zorder=0: #BFBFBF fill for public/urban/water (all Australian land)
        fill_states(ax_am, gdf_states)
        # zorder=1: lighter grey for research area (LU cats 1-5) = "no AM used"
        lu_tp = tiff_lu[scen]
        if lu_tp and os.path.exists(lu_tp):
            lu_arr = read_tiff(lu_tp)
            research = np.zeros(lu_arr.shape, dtype=bool)
            for code, cid in LU_CAT_MAP.items():
                if cid in (1, 2, 3, 4, 5):
                    research[lu_arr == code] = True
            no_am_rgba = np.zeros((*lu_arr.shape, 4), dtype='float32')
            r, g, b = _hex_rgb(NO_AM_COLOR)
            no_am_rgba[research] = [r, g, b, 1.0]
            ax_am.imshow(no_am_rgba, extent=EXTENT, origin='upper',
                         interpolation='nearest', zorder=1)
        # zorder=2: actual AM colors where AM is applied
        tp = tiff_am[scen]
        if tp and os.path.exists(tp):
            ax_am.imshow(ammap_to_rgba(read_tiff(tp), am_info),
                         extent=EXTENT, origin='upper', interpolation='nearest',
                         zorder=2)
        add_states(ax_am, gdf_states)
        _style(ax_am)

    # ── Column titles ─────────────────────────────────────────────────────────
    total = LABEL_W_IN / MAP_W_IN + 2.0
    cf    = np.array([LABEL_W_IN / MAP_W_IN, 1.0, 1.0]) / total
    x_lu  = cf[0] + cf[1] / 2
    x_am  = cf[0] + cf[1] + cf[2] / 2
    ty    = 1.0 - TOP_IN / fig_h  # just above gridspec top

    fig.text(x_lu, ty, 'Land-use',
             ha='center', va='bottom', fontsize=9)
    fig.text(x_am, ty, 'Agricultural management',
             ha='center', va='bottom', fontsize=9)

    # ── Legends (no titles, equal spacing, AM = 8 types only) ────────────────
    lu_handles = [
        mpatches.Patch(facecolor=hexc, label=lbl, edgecolor='none')
        for _, (lbl, hexc) in LU_CAT_INFO.items()
    ]
    am_handles = [
        *[mpatches.Patch(facecolor=hexc, label=lbl, edgecolor='none')
          for code, (lbl, hexc) in sorted(am_info.items())
          if 1 <= code <= 8],
        mpatches.Patch(facecolor=NO_AM_COLOR, edgecolor='none',
                       label='No agricultural management'),
    ]

    leg_y = BOT_IN / fig_h  # = gridspec bottom; legends anchor at top and extend down
    fig.legend(lu_handles, [h.get_label() for h in lu_handles],
               loc='upper center',
               bbox_to_anchor=(x_lu, leg_y), bbox_transform=fig.transFigure,
               ncol=1, fontsize=9, frameon=False,
               handlelength=1.2, handleheight=0.9, borderpad=0)
    fig.legend(am_handles, [h.get_label() for h in am_handles],
               loc='upper center',
               bbox_to_anchor=(x_am, leg_y), bbox_transform=fig.transFigure,
               ncol=1, fontsize=9, frameon=False,
               handlelength=1.2, handleheight=0.9, borderpad=0)

    out = os.path.join(OUTPUT_DIR, '04_land use and ag management map.svg')
    fig.savefig(out, dpi=600, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'Saved: {out}')


if __name__ == '__main__':
    main()
