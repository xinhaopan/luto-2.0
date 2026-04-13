# ==============================================================================
# Figure: Spatial distribution of non-ag land use (2050, carbon price = 356.92)
# 2×3 panel, one map per biodiversity target scenario
# ==============================================================================

import io
import math
import os
import sys
import zipfile

import cf_xarray as cfxr
import geopandas as gpd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tools.config as config

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
TASK_NAME  = config.TASK_NAME
TASK_ROOT  = f'../../../output/{TASK_NAME}'
OUT_DIR    = f'../../../output/{TASK_NAME}/paper4/figures'
os.makedirs(OUT_DIR, exist_ok=True)

COLOR_FILE = 'tools/land use colors.xlsx'
SHP_FILE   = '../../paper4/Map/AUS_line1.shp'    # state boundaries

YEAR = 2050
CP   = 356.92    # carbon price to map

SCENARIO_GROUPS = [
    ('No biodiversity target', 'GBF2_off_CUT_50_CarbonPrice_{cp}'),
] + [
    (f'Restore top {cut}%', f'GBF2_high_CUT_{cut}_CarbonPrice_{{cp}}')
    for cut in config.BIO_CUTS
]

# Geographic extent matching LUTO template (EPSG:4283, geographic)
EXTENT = [113.0, 153.6, -43.64, -10.04]   # [left, right, bottom, top]

FS = 10
plt.rcParams.update({
    'font.family':      'sans-serif',
    'font.sans-serif':  ['Arial'],
    'font.size':        FS,
    'axes.titlesize':   FS,
    'axes.labelsize':   FS,
    'xtick.labelsize':  FS,
    'ytick.labelsize':  FS,
    'legend.fontsize':  FS,
    'figure.titlesize': FS,
    'mathtext.fontset': 'stixsans',
})

# ---------------------------------------------------------------------------
# Load non-ag color mapping
# ---------------------------------------------------------------------------
color_df = pd.read_excel(COLOR_FILE, sheet_name='non_ag')
# Exclude summary rows
color_df = color_df[~color_df['desc'].isin(['Agricultural land-use', 'Other land-use'])]

# Assign integer codes 1..N to each non-ag land use
LU_CODES  = {row['desc']: idx + 1 for idx, row in color_df.reset_index(drop=True).iterrows()}
CODE_COLOR = {idx + 1: row['color'] for idx, row in color_df.reset_index(drop=True).iterrows()}
CODE_LABEL = {idx + 1: row['desc'] for idx, row in color_df.reset_index(drop=True).iterrows()}

BACKGROUND_COLOR = '#BFBFBF'   # non-LUTO / ocean fill
NODATA_COLOR     = '#E8E8E8'   # LUTO cells with no non-ag land use


def _hex_to_rgb(h):
    return int(h[1:3], 16)/255, int(h[3:5], 16)/255, int(h[5:7], 16)/255


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def find_run_dir(task_root, suffix):
    for name in os.listdir(task_root):
        if name.endswith(suffix):
            return os.path.join(task_root, name)
    return None


def get_lu_code(lu_name):
    """Return integer code for a non-ag land use (case-insensitive strip)."""
    lu_norm = lu_name.strip().lower().replace('-', '').replace(' ', '')
    for desc, code in LU_CODES.items():
        if desc.strip().lower().replace('-', '').replace(' ', '') == lu_norm:
            return code
    return 0


def build_nonag_map(zip_path, year):
    """
    Returns:
      cat_2d  : 2D float32 array (H×W) with integer non-ag codes (0 = no non-ag,
                NaN = outside model, -1 = non-LUTO Australian land)
      tmpl_ds : xr.Dataset with template (for extent check)
    """
    area_fname = f'xr_area_non_agricultural_landuse_{year}.nc'
    tmpl_fname = f'xr_map_template_{year}.nc'

    with zipfile.ZipFile(zip_path) as z:
        # ---- area file ----
        area_matches = [n for n in z.namelist() if n.endswith(area_fname)]
        if not area_matches:
            return None, None
        with z.open(area_matches[0]) as f:
            ds_area = xr.open_dataset(io.BytesIO(f.read()), engine='h5netcdf')
        if 'layer' in ds_area.dims and 'compress' in ds_area['layer'].attrs:
            ds_area = cfxr.decode_compress_to_multi_index(ds_area, 'layer')
        da = list(ds_area.data_vars.values())[0]   # (cell, layer) with lu coord

        # ---- template ----
        tmpl_matches = [n for n in z.namelist() if n.endswith(tmpl_fname)]
        if not tmpl_matches:
            return None, None
        with z.open(tmpl_matches[0]) as f:
            ds_tmpl = xr.open_dataset(io.BytesIO(f.read()), engine='h5netcdf')

    # Build 1D dominant-lu array (186648 cells)
    lu_vals = [v for v in da.coords['lu'].values if v != 'ALL']
    n_cells = da.sizes['cell']
    dom_code = np.zeros(n_cells, dtype='float32')   # 0 = no non-ag

    # Stack area per lu into 2D array (n_cells × n_lu)
    areas = np.stack([da.sel(lu=lu).values for lu in lu_vals], axis=1)
    # For each cell, pick lu with maximum area
    max_area = areas.max(axis=1)
    max_idx  = areas.argmax(axis=1)
    has_nonag = max_area > 0
    for i, lu in enumerate(lu_vals):
        mask = has_nonag & (max_idx == i)
        dom_code[mask] = get_lu_code(lu)

    # Reconstruct 2D map using template
    tmpl_2d = ds_tmpl['layer'].values.astype('float32')   # (H, W), >=0 model cells, -1 non-LUTO, NaN ocean
    h, w = tmpl_2d.shape
    cat_2d = np.full((h, w), np.nan, dtype='float32')

    valid_mask = tmpl_2d >= 0     # model cells
    cat_2d[valid_mask] = dom_code   # 1D values → 2D positions (in order)
    cat_2d[tmpl_2d == -1] = -1      # non-LUTO Australian land

    return cat_2d, ds_tmpl


def cat_to_rgba(cat_2d):
    """Convert integer categorical 2D array to RGBA image."""
    h, w = cat_2d.shape
    rgba = np.zeros((h, w, 4), dtype='float32')

    # Background (non-LUTO Australian land)
    mask_bg = cat_2d == -1
    r, g, b = _hex_to_rgb(BACKGROUND_COLOR)
    rgba[mask_bg] = [r, g, b, 1.0]

    # Model cells with no non-ag
    mask_nodata = cat_2d == 0
    r, g, b = _hex_to_rgb(NODATA_COLOR)
    rgba[mask_nodata] = [r, g, b, 1.0]

    # Non-ag land use types
    for code, hexc in CODE_COLOR.items():
        r, g, b = _hex_to_rgb(hexc)
        rgba[cat_2d == code] = [r, g, b, 1.0]

    # Ocean / outside bounding box stays transparent (alpha=0)
    return rgba


# ---------------------------------------------------------------------------
# Load state boundary shapefile
# ---------------------------------------------------------------------------
def load_states():
    shp = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), SHP_FILE))
    if not os.path.exists(shp):
        print(f'Shapefile not found: {shp}')
        return None
    gdf = gpd.read_file(shp)
    # Reproject to EPSG:4283 (GDA94 geographic, matches template coords)
    try:
        gdf = gdf.to_crs(epsg=4283)
    except Exception:
        pass
    return gdf


# ---------------------------------------------------------------------------
# Collect maps
# ---------------------------------------------------------------------------
print(f"Building non-ag maps for carbon price = {CP}...")
maps = {}

for label, suffix_tmpl in SCENARIO_GROUPS:
    suffix  = suffix_tmpl.format(cp=CP)
    run_dir = find_run_dir(TASK_ROOT, suffix)
    if run_dir is None:
        print(f"  {label}: run dir not found")
        maps[label] = None
        continue
    zip_path = os.path.join(run_dir, 'Run_Archive.zip')
    if not os.path.isfile(zip_path):
        print(f"  {label}: MISSING zip")
        maps[label] = None
        continue
    print(f"  {label}: reading...")
    cat_2d, _ = build_nonag_map(zip_path, YEAR)
    maps[label] = cat_2d
    if cat_2d is not None:
        n_noag = int((cat_2d == 0).sum())
        n_nonag = int((cat_2d > 0).sum())
        print(f"    non-ag cells: {n_nonag}, no non-ag: {n_noag}")


# ---------------------------------------------------------------------------
# Plot: 2×3 figure
# ---------------------------------------------------------------------------
gdf_states = load_states()

asp      = (EXTENT[3] - EXTENT[2]) / (EXTENT[1] - EXTENT[0])   # ≈ 0.827
MAP_W    = 3.5
MAP_H    = MAP_W * asp
HSPACE   = 0.35   # larger gap between rows
WSPACE   = 0.06
TOP_PAD  = 0.25
BOT_PAD  = 1.10   # legend space

# Padding around Australia's bounding envelope (in degrees)
MAP_PAD  = 0.8
XLIM = (EXTENT[0] - MAP_PAD, EXTENT[1] + MAP_PAD)
YLIM = (EXTENT[2] - MAP_PAD, EXTENT[3] + MAP_PAD)

nrows, ncols = 2, 3
fig_w = ncols * MAP_W + (ncols - 1) * WSPACE + 0.10
fig_h = TOP_PAD + nrows * MAP_H + (nrows - 1) * HSPACE + BOT_PAD

fig = plt.figure(figsize=(fig_w, fig_h), facecolor='white')

gs = gridspec.GridSpec(
    nrows, ncols,
    figure=fig,
    left   = 0.01,
    right  = 0.99,
    top    = 1.0 - TOP_PAD / fig_h,
    bottom = BOT_PAD / fig_h,
    wspace = WSPACE / MAP_W,
    hspace = HSPACE / MAP_H,
)


def _style_ax(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(*XLIM)
    ax.set_ylim(*YLIM)
    ax.set_aspect('equal', adjustable='box')
    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_linewidth(0.6)
        sp.set_edgecolor('#444444')


for idx, (label, _) in enumerate(SCENARIO_GROUPS):
    row, col = divmod(idx, ncols)
    ax = fig.add_subplot(gs[row, col])

    cat_2d = maps[label]
    if cat_2d is not None:
        rgba = cat_to_rgba(cat_2d)
        img_extent = [EXTENT[0], EXTENT[1], EXTENT[2], EXTENT[3]]
        ax.imshow(rgba, extent=img_extent, origin='upper',
                  interpolation='nearest', zorder=1, aspect='auto')

    # State/territory boundaries (LineString shapefile — use .plot() directly, NOT .boundary)
    if gdf_states is not None:
        gdf_states.plot(ax=ax, edgecolor='#555555', linewidth=0.4,
                        facecolor='none', zorder=3)

    ax.set_title(label, fontsize=FS, pad=3)
    _style_ax(ax)

# ---------------------------------------------------------------------------
# Legend: matplotlib fills column-major (top→bottom per col).
# Reorder handles so column-major filling gives row-major visual layout:
#   row 0-1: 8 non-ag types (4 per row), row 2: 2 bg items
# ---------------------------------------------------------------------------
leg_handles = [
    mpatches.Patch(facecolor=CODE_COLOR[code], edgecolor='none', label=CODE_LABEL[code])
    for code in sorted(CODE_LABEL)
] + [
    mpatches.Patch(facecolor=NODATA_COLOR,     edgecolor='none', label='Agricultural land'),
    mpatches.Patch(facecolor=BACKGROUND_COLOR, edgecolor='none',
                   label='Public and indigenous land, urban land, plantation forestry, and water bodies'),
]

# ---------------------------------------------------------------------------
# Legend row 1-2: 8 non-ag items, 4 per row
# Reorder for column-major filling so visual order is row-major
# ---------------------------------------------------------------------------
row_h_frac = (FS * 1.8) / (fig_h * 72)   # ~1 row height in figure fraction

# Items 1-9 go into leg1 (4 per row; row 3 has only item 9 at col 0)
# Item 10 (long bg label) drawn manually at x=col1, y=row3
items_9  = leg_handles[:9]   # h0..h8
item_10  = leg_handles[9]    # long background label
empty    = mpatches.Patch(facecolor='none', edgecolor='none', label='')

# Reorder for column-major filling with ncols=4, nrows=3:
# Desired visual layout:
#   row0: h0 h1 h2 h3
#   row1: h4 h5 h6 h7
#   row2: h8 -- -- --
# column-major positions: col0=[h0,h4,h8], col1=[h1,h5,--], col2=[h2,h6,--], col3=[h3,h7,--]
ncols1, nrows1 = 4, 3
padded = items_9 + [empty] * (ncols1 * nrows1 - len(items_9))
reordered1 = [empty] * (ncols1 * nrows1)
for vrow in range(nrows1):
    for vcol in range(ncols1):
        reordered1[vcol * nrows1 + vrow] = padded[vrow * ncols1 + vcol]

leg1 = fig.legend(
    handles=reordered1,
    loc='lower center',
    bbox_to_anchor=(0.5, row_h_frac * 0.2),
    ncol=ncols1,
    frameon=False,
    fontsize=FS,
    handlelength=1.0,
    handleheight=1.0,
    handletextpad=0.5,
    columnspacing=1.0,
)

# ---------------------------------------------------------------------------
# Item 10: drawn manually.
# x = left edge of col 1 handle (item 6 = h5 is at reordered position 4)
# y = vertical centre of row 3  (item 9 = h8 is at reordered position 2)
# ---------------------------------------------------------------------------
fig.canvas.draw()
renderer = fig.canvas.get_renderer()

def _hbb(legend, idx):
    """Bounding box of handle[idx] in figure-fraction coords."""
    return legend.legend_handles[idx].get_window_extent(renderer)\
               .transformed(fig.transFigure.inverted())

bb_col0_row0 = _hbb(leg1, 0)   # h0: col 0 — for handle width
bb_col1_row1 = _hbb(leg1, 4)   # h5: col 1 — x alignment (item 6)
bb_col0_row2 = _hbb(leg1, 2)   # h8: col 0, row 3 — y alignment (item 9)

handle_w = bb_col0_row0.x1 - bb_col0_row0.x0
handle_h = bb_col0_row0.y1 - bb_col0_row0.y0
text_gap = (FS * 0.5) / (fig_w * 72)

x0    = bb_col1_row1.x0                               # left-align with col 1
y_ctr = (bb_col0_row2.y0 + bb_col0_row2.y1) / 2      # vertically align with item 9

rect = mpatches.Rectangle(
    (x0, y_ctr - handle_h / 2), handle_w, handle_h,
    facecolor=item_10.get_facecolor(), edgecolor='none',
    transform=fig.transFigure, clip_on=False, zorder=10,
)
fig.add_artist(rect)
fig.text(
    x0 + handle_w + text_gap, y_ctr,
    item_10.get_label(),
    va='center', ha='left', fontsize=FS,
    transform=fig.transFigure,
)

out_path = os.path.join(OUT_DIR, f'NonAg_Map_CP{CP}_{YEAR}.png')
fig.savefig(out_path, dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print(f"\nSaved: {out_path}")
