"""
Land-use and agricultural-management change animation, 2010-2050 (Paper 1 outreach).

Usage (working directory = myCode/draw_all/code):
    python 15_video_landuse_2010_2050.py preview   # area check + 2010/2030/2050 stills + side-by-side
    python 15_video_landuse_2010_2050.py full      # every frame + MP4 + 2050 thumbnail

Maps follow the paper map pipeline (tools/map_helper: basemap -> reproject -> colour -> overlay -> crop -> GCCSA lines);
areas are read the same way as 02_Area.py (tools/data_helper.get_dict_data).
"""
import os
import sys
import shutil
import argparse
import subprocess
from datetime import date

os.environ.setdefault('GDAL_DATA', '')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import rasterio
from PIL import Image
from joblib import Parallel, delayed

import tools.data_helper as data_helper
from tools.data_helper import get_dict_data, aggregate_by_mapping
from tools.map_helper import generate_basemap, reproject_overlay, overlay_map, crop_map, add_shp

# ============================================================================
# Configuration
# ============================================================================
SCENARIO = 'Run_9_GHG_high_BIO_high'
SCENARIO_TAG = 'high_high'                      # used in the output file name
# folder holding the Run_* directories (env VIDEO_DATA_ROOT overrides; it is inherited by the joblib workers)
DATA_ROOT = os.environ.get('VIDEO_DATA_ROOT', 'N:/LUF-Modelling/LUTO2_XH/output1/20250705_Paper1_Results')

WIDTH, HEIGHT = 1920, 1080
FPS = 25
SECONDS_PER_YEAR = 0.6
LEAD_S = 1.0           # hold on 2010
HOLD_S = 4.0           # hold on 2050
N_JOBS = 8

YEARS = list(range(2010, 2051))

OUT_DIR = '../output/video'
MAP_CACHE_DIR = f'{OUT_DIR}/map_cache_{SCENARIO}'
FRAME_DIR = f'{OUT_DIR}/frames_{SCENARIO}'
PREVIEW_DIR = f'{OUT_DIR}/preview_{SCENARIO}'
VIDEO_DIR = OUT_DIR
VIDEO_NAME = f'{date.today():%Y%m%d}_LUTO2_landuse_2010_2050_{SCENARIO_TAG}.mp4'
FFMPEG = shutil.which('ffmpeg') or 'ffmpeg'

COLORS_XLSX = 'tools/land use colors.xlsx'
GROUP_XLSX = 'tools/land use group.xlsx'
BASEMAP_TIF = 'Assets/basemap.tif'
SHP_PATH = 'Assets/shp/GCCSA_line.shp'

# Text sizes in pixels (figure dpi = 100, so pt = px * 0.72)
PX_TITLE, PX_PANEL, PX_AXIS, PX_YEAR = 34, 22, 18, 72
BG = '#FFFFFF'
TEXT = '#1A1A1A'
YEAR_RED = '#D7191C'

# Short legend labels for the video (paper desc_new without the long parentheticals)
NON_AG_LABELS = {
    100: 'Environmental plantings', 101: 'Riparian buffer restoration',
    102: 'Agroforestry (sheep)', 103: 'Agroforestry (beef)', 104: 'Carbon plantings',
    105: 'Farm forestry (sheep)', 106: 'Farm forestry (beef)', 107: 'Bioenergy with carbon capture',
    108: 'Destocked natural land',
}
AM_LABELS = {
    1: 'Methane reduction (livestock)', 2: 'Agricultural technology (fertiliser)',
    3: 'Regenerative agriculture (livestock)', 4: 'Early dry-season savanna burning',
    5: 'Agricultural technology (energy)', 6: 'Biochar',
    7: 'Human-induced regeneration (beef)', 8: 'Human-induced regeneration (sheep)',
    0: 'No agricultural management', -1: 'Other land',
}

DPI = 100
plt.rcParams.update({'font.family': 'Arial', 'axes.unicode_minus': True})


def pt(px):
    return px * 72 / DPI


def get_path(path_name):
    base = f'{DATA_ROOT}/{path_name}/output'
    sub = [s for s in os.listdir(base) if '2010-2050' in s][0]
    return os.path.join(base, sub)


data_helper.get_path = get_path      # get_dict_data reads from DATA_ROOT


# ============================================================================
# Colour tables (code -> (label, colour)), from the paper map sheets
# ============================================================================
def lu_table():
    """lumap codes: ag_group_map for 0-27 and -1, non_ag sheet for the non-agricultural codes 100-108."""
    ag = pd.read_excel(COLORS_XLSX, sheet_name='ag_group_map')
    nonag = pd.read_excel(COLORS_XLSX, sheet_name='non_ag')
    table = {int(r.code): (r.desc, r.color) for r in ag.itertuples()}
    for r in nonag.itertuples():
        if r.code >= 100:
            table[int(r.code)] = (NON_AG_LABELS[int(r.code)], r.color)
    table[-1] = ('Other land', table[-1][1])
    return table


def am_table():
    am = pd.read_excel(COLORS_XLSX, sheet_name='am')
    return {int(r.code): (AM_LABELS[int(r.code)], r.color) for r in am.itertuples()}


def shift_am_codes(data):
    """Same remap as 00_am_tif.py (raw ammap 3..7 -> 4..8; code 3 is not used in these runs)."""
    new = data.copy()
    for old, nw in ((3, 4), (4, 5), (5, 6), (6, 7), (7, 8)):
        new[data == old] = nw
    return new


LAYERS = {'lu': 'lumap', 'am': 'ammap'}    # layer -> GeoTIFF prefix


def layer_table(layer):
    return lu_table() if layer == 'lu' else am_table()


def layer_shift(layer):
    return shift_am_codes if layer == 'am' else None


# ============================================================================
# Areas (Mha), as 02_Area.py
# ============================================================================
def load_areas():
    S = [SCENARIO]
    kw = dict(use_parallel=False)
    lt = lu_table()

    ag = get_dict_data(S, 'area_agricultural_landuse', 'Area (ha)', 'Land-use', **kw)
    ag = aggregate_by_mapping(ag, GROUP_XLSX, 'desc', 'ag_group')[SCENARIO].sort_index()
    lu = pd.DataFrame(index=ag.index)
    for code in (0, 1, 22, 23):
        lu[code] = ag[lt[code][0]]

    nonag = get_dict_data(S, 'area_non_agricultural_landuse', 'Area (ha)', 'Land-use', **kw)[SCENARIO].sort_index()
    sheet = pd.read_excel(COLORS_XLSX, sheet_name='non_ag')
    norm = lambda s: str(s).replace('-', '').replace(' ', '').lower()
    name2code = {norm(r.desc): int(r.code) for r in sheet.itertuples() if r.code >= 100}
    for col in nonag.columns:
        lu[name2code[norm(col)]] = nonag[col]

    am = get_dict_data(S, 'area_agricultural_management', 'Area (ha)', 'Type', **kw)[SCENARIO].sort_index()
    sheet = pd.read_excel(COLORS_XLSX, sheet_name='am')
    name2code = {norm(r.desc): int(r.code) for r in sheet.itertuples() if r.code >= 1}
    amdf = pd.DataFrame(index=am.index)
    for col in am.columns:
        amdf[name2code[norm(col)]] = am[col]
    return {'lu': lu.fillna(0), 'am': amdf.reindex(lu.index).fillna(0)}


# ============================================================================
# Maps: one cached RGB PNG per year and layer, following the paper map pipeline
# ============================================================================
def _color_overlay(geo_tif, rgb_tif, table, shift=None, background_value=-9999):
    """assign_colors_to_overlay with an in-memory code table (unmapped codes -> -2 grey, as the paper)."""
    with rasterio.open(geo_tif) as src:
        data = src.read(1)
        transform, crs = src.transform, src.crs
    if shift is not None:
        data = shift(data)
    unmapped = set(np.unique(data)) - set(table) - {background_value}
    for v in unmapped:
        data[data == v] = -2
    rgb = np.zeros((*data.shape, 3), dtype=np.uint8)
    for code, (_, hexc) in table.items():
        rgb[data == code] = [int(hexc[i:i + 2], 16) for i in (1, 3, 5)]
    with rasterio.open(rgb_tif, 'w', driver='GTiff', height=rgb.shape[0], width=rgb.shape[1], count=3,
                       dtype='uint8', crs=crs, transform=transform) as dst:
        for i in range(3):
            dst.write(rgb[:, :, i], i + 1)


def render_year_map(layer, year, basemap_rgb, force=False):
    png = f'{MAP_CACHE_DIR}/{layer}_{year}.png'
    if os.path.exists(png) and not force:
        return png
    tmp = f'{MAP_CACHE_DIR}/tmp_{layer}_{year}'
    src = os.path.join(get_path(SCENARIO), f'out_{year}', f'{LAYERS[layer]}_{year}.tiff')
    reproject_overlay(src, BASEMAP_TIF, f'{tmp}_reproj.tif')
    _color_overlay(f'{tmp}_reproj.tif', f'{tmp}_rgb.tif', layer_table(layer), layer_shift(layer))
    overlay_map(basemap_rgb, f'{tmp}_rgb.tif', f'{tmp}_overlay.tif')
    crop_map(f'{tmp}_overlay.tif', src, f'{tmp}_crop.tif')
    add_shp(f'{tmp}_crop.tif', SHP_PATH, f'{tmp}_shp.tif', shapefile_color='#8F8F8F', linewidth=1)
    with rasterio.open(f'{tmp}_shp.tif') as s:
        rgb = np.dstack([s.read(i) for i in (1, 2, 3)])
    Image.fromarray(rgb).save(png)
    for suffix in ('reproj', 'rgb', 'overlay', 'crop', 'shp'):
        if os.path.exists(f'{tmp}_{suffix}.tif'):
            os.remove(f'{tmp}_{suffix}.tif')
    return png


def build_map_cache(years, force=False):
    os.makedirs(MAP_CACHE_DIR, exist_ok=True)
    basemap_rgb = f'{MAP_CACHE_DIR}/basemap_rgb.tif'
    if not os.path.exists(basemap_rgb):
        generate_basemap(BASEMAP_TIF, basemap_rgb)
    jobs = [(layer, y) for layer in LAYERS for y in years]
    Parallel(n_jobs=min(N_JOBS, len(jobs)))(delayed(render_year_map)(l, y, basemap_rgb, force) for l, y in jobs)


def codes_present(layer):
    codes, shift = set(), layer_shift(layer)
    for y in YEARS:
        with rasterio.open(os.path.join(get_path(SCENARIO), f'out_{y}', f'{LAYERS[layer]}_{y}.tiff')) as s:
            a = s.read(1)
        if shift is not None:
            a = shift(a)
        codes |= set(np.unique(a).astype(int).tolist())
    codes.discard(-9999)
    return codes


_MAP_CACHE = {}     # plain dict (an lru_cache wrapper does not pickle into the joblib workers)


def _map_array(layer, year):
    key = (layer, year)
    if key not in _MAP_CACHE:
        if len(_MAP_CACHE) > 8:
            _MAP_CACHE.clear()
        _MAP_CACHE[key] = np.asarray(Image.open(f'{MAP_CACHE_DIR}/{layer}_{year}.png').convert('RGB'),
                                     dtype=np.float32)
    return _MAP_CACHE[key]


def blended_map(layer, t_year):
    """Hold each year, cross-fade to the next centred on the year switch (frac 0.25 -> 0.75)."""
    y0 = int(np.floor(t_year))
    if y0 >= YEARS[-1]:
        return _map_array(layer, YEARS[-1]).astype(np.uint8)
    w = np.clip((t_year - y0 - 0.25) / 0.5, 0, 1)
    w = w * w * (3 - 2 * w)
    if w == 0:
        return _map_array(layer, y0).astype(np.uint8)
    return ((1 - w) * _map_array(layer, y0) + w * _map_array(layer, y0 + 1)).astype(np.uint8)


# ============================================================================
# Frame
# ============================================================================
def px_axes(fig, x, y, w, h):
    """Axes from a pixel box (x, y = top-left)."""
    return fig.add_axes([x / WIDTH, 1 - (y + h) / HEIGHT, w / WIDTH, h / HEIGHT])


def fig_text(fig, x, y, s, px, **kw):
    return fig.text(x / WIDTH, 1 - y / HEIGHT, s, fontsize=pt(px), color=kw.pop('color', TEXT), **kw)


def upto(df, t_year):
    """Rows for years <= t_year, plus a linearly interpolated row at t_year."""
    keep = df[df.index <= t_year]
    last = keep.index.max()
    if t_year > last and (df.index > last).any():
        nxt = df.index[df.index > last].min()
        w = (t_year - last) / (nxt - last)
        row = df.loc[last] * (1 - w) + df.loc[nxt] * w
        keep = pd.concat([keep, pd.DataFrame([row.values], index=[t_year], columns=df.columns)])
    return keep


def nice_range(hi, n=3):
    raw = hi / n
    mag = 10 ** np.floor(np.log10(raw))
    step = min(s * mag for s in (1, 2, 2.5, 5, 10) if s * mag >= raw)
    top = np.ceil(hi / step) * step
    return top, np.arange(0, top + step / 2, step)


class Layout:
    def __init__(self, areas):
        self.areas = areas
        self.tables = {k: layer_table(k) for k in LAYERS}
        self.ranges = {k: nice_range(v.sum(axis=1).max()) for k, v in areas.items()}
        order = {'lu': [0, 1, 22, 23] + list(range(100, 109)) + [-1],
                 'am': list(range(1, 9)) + [0, -1]}
        self.legend = {}
        for layer in LAYERS:
            present = codes_present(layer)
            present |= {c for c in areas[layer].columns if areas[layer][c].max() > 0}   # also classes in the chart
            if layer == 'lu':                                # ag_group_map: many lumap codes -> 4 groups
                grp = {c: lu_table()[c][0] for c in range(0, 28)}
                present |= {code for code in (0, 1, 22, 23) if any(grp[c] == lu_table()[code][0] for c in present if 0 <= c <= 27)}
            seen, items = set(), []
            for c in order[layer]:
                if c in present:
                    label, col = self.tables[layer][c]
                    if label not in seen:
                        seen.add(label)
                        items.append((label, col))
            self.legend[layer] = items
        h, w = _map_array('lu', YEARS[0]).shape[:2]
        self.map_aspect = h / w


def draw_panel(fig, L, layer, title, x0, t_year):
    # title
    fig_text(fig, x0, 62, title, PX_TITLE, weight='bold', va='baseline')
    # map
    mw = 860
    mh = int(mw * L.map_aspect)
    axm = px_axes(fig, x0, 88, mw, mh)
    axm.imshow(blended_map(layer, t_year), interpolation='antialiased')
    axm.set_axis_off()
    # area chart (only classes drawn on the map, same colours)
    y_chart = 88 + mh + 34
    fig_text(fig, x0, y_chart - 10, 'Area (million hectares)', PX_PANEL, va='baseline')
    ax = px_axes(fig, x0 + 52, y_chart + 6, 340, HEIGHT - y_chart - 6 - 40)
    df = upto(L.areas[layer], t_year)
    if len(df) >= 2:
        cols = [c for c in df.columns if c in L.tables[layer]]
        ax.stackplot(df.index.values.astype(float), np.array([df[c].values for c in cols]),
                     colors=[L.tables[layer][c][1] for c in cols], lw=0)
    top, ticks = L.ranges[layer]
    ax.set_xlim(2010, 2050)
    ax.set_xticks([2010, 2030, 2050])
    ax.set_ylim(0, top)
    ax.set_yticks(ticks)
    ax.tick_params(labelsize=pt(PX_AXIS), length=3, pad=2, colors=TEXT)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    # legend
    handles = [mpatches.Patch(facecolor=c, edgecolor='#999999', lw=0.4, label=l) for l, c in L.legend[layer]]
    fig.legend(handles=handles, loc='upper left', bbox_to_anchor=((x0 + 425) / WIDTH, 1 - (y_chart - 22) / HEIGHT),
               ncol=2 if layer == 'lu' else 1, frameon=False, fontsize=pt(PX_AXIS * 0.9), handlelength=1.1, handleheight=0.9,
               columnspacing=1.0, handletextpad=0.4, labelspacing=0.35)


def draw_frame(L, t_year, out_png):
    fig = plt.figure(figsize=(WIDTH / DPI, HEIGHT / DPI), dpi=DPI, facecolor=BG)
    draw_panel(fig, L, 'lu', 'Land use', 60, t_year)
    draw_panel(fig, L, 'am', 'Agricultural management', 1000, t_year)
    fig_text(fig, WIDTH - 40, 26, 'Simulation year', PX_PANEL, ha='right', va='baseline', color='#444444')
    fig_text(fig, WIDTH - 40, 86, f'{int(round(t_year))}', PX_YEAR, ha='right', va='baseline',
             color=YEAR_RED, weight='bold')
    fig.savefig(out_png, dpi=DPI, facecolor=BG)
    plt.close(fig)
    return out_png


# ============================================================================
# Preview and full render
# ============================================================================
def print_areas(L):
    yrs = [2010, 2030, 2050]
    for layer, name in (('lu', 'Land use'), ('am', 'Agricultural management')):
        df = L.areas[layer].reindex(yrs).copy()
        df.columns = [L.tables[layer][c][0] for c in df.columns]
        df = df.T.groupby(level=0, sort=False).sum().T
        df['Total'] = df.sum(axis=1)
        print(f'\n[{name}] area (Mha):\n', df.round(2).T.to_string())


def preview():
    os.makedirs(PREVIEW_DIR, exist_ok=True)
    build_map_cache([2010, 2011, 2029, 2030, 2031, 2049, 2050])
    L = Layout(load_areas())
    print_areas(L)
    stills = [draw_frame(L, y, f'{PREVIEW_DIR}/still_{y}.png') for y in (2010, 2030, 2050)]
    ims = [Image.open(s).convert('RGB').resize((WIDTH // 2, HEIGHT // 2), Image.LANCZOS) for s in stills]
    grid = Image.new('RGB', (WIDTH, HEIGHT), 'white')
    for i, im in enumerate(ims):
        grid.paste(im, ((i % 2) * WIDTH // 2, (i // 2) * HEIGHT // 2))
    grid.save(f'{PREVIEW_DIR}/compare_2010_2030_2050.png')
    print(f'\nPreview written to {PREVIEW_DIR}')


def timeline():
    n_grow = int(round((YEARS[-1] - YEARS[0]) * SECONDS_PER_YEAR * FPS))
    return [YEARS[0] + (YEARS[-1] - YEARS[0]) * i / n_grow for i in range(n_grow + 1)]


def render_part(part, nparts):
    """Render every nparts-th growth frame starting at `part` (run as a separate process; loky workers crash here)."""
    L = Layout(load_areas())
    for i, t in enumerate(timeline()):
        out = f'{FRAME_DIR}/main_{i:05d}.png'
        if i % nparts == part and not os.path.exists(out):
            draw_frame(L, t, out)


def full():
    os.makedirs(FRAME_DIR, exist_ok=True)
    build_map_cache(YEARS)
    L = Layout(load_areas())
    print_areas(L)

    t_years = timeline()
    procs = [subprocess.Popen([sys.executable, __file__, 'part', '--part', str(k), '--nparts', str(N_JOBS)])
             for k in range(N_JOBS)]
    if any(p.wait() for p in procs):
        raise RuntimeError('a frame-rendering process failed')
    pngs = [f'{FRAME_DIR}/main_{i:05d}.png' for i in range(len(t_years))]
    seq = [pngs[0]] * int(round(LEAD_S * FPS)) + pngs + [pngs[-1]] * int(round(HOLD_S * FPS))

    lst = f'{FRAME_DIR}/frames.txt'
    with open(lst, 'w') as fh:
        for p in seq:
            fh.write(f"file '{os.path.basename(p)}'\nduration {1 / FPS:.6f}\n")   # old ffmpeg: names relative to the list
        fh.write(f"file '{os.path.basename(seq[-1])}'\n")
    mp4 = os.path.join(VIDEO_DIR, VIDEO_NAME)
    if os.path.exists(mp4):
        raise FileExistsError(f'{mp4} exists; move it to an archive folder first')
    subprocess.run([FFMPEG, '-y', '-f', 'concat', '-safe', '0', '-i', lst, '-r', str(FPS),
                    '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '18', '-preset', 'slow',
                    '-movflags', '+faststart', mp4], check=True)
    thumb = mp4.replace('.mp4', '_thumbnail_2050.png')
    shutil.copy(pngs[-1], thumb)
    print(f'\nVideo: {mp4}\nThumbnail: {thumb}\nFrames: {len(seq)} ({len(seq) / FPS:.1f} s)')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['preview', 'full', 'part'])
    ap.add_argument('--part', type=int, default=0)
    ap.add_argument('--nparts', type=int, default=1)
    args = ap.parse_args()
    if args.mode == 'preview':
        preview()
    elif args.mode == 'full':
        full()
    else:
        render_part(args.part, args.nparts)
