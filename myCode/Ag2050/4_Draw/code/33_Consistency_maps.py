"""
33_Consistency_maps.py -- Extended Data figure + companion table

Twelve Same/Different maps: where do the scenarios agree on the 2050 dominant
land use, and where do they diverge?

Answers Referee 2, who said the maps in Fig. 2 are too small to show where the
scenarios actually differ and asked for agreement and divergence to be mapped
instead of four separate end-state maps.

The first eleven panels compare the dominant agricultural land-use category; the
last compares the dominant agricultural management option, and only between
Regional Ag Capitals and Landscape Stewardship, because the other two scenarios
adopt almost no management options.

Where the numbers come from
    NOT from the plotting rasters.  Those carry one already-rounded integer per
    cell and say nothing about how much of the cell that class occupies, so
    every cell would count the same regardless of its size or composition.
    Instead the per-cell area of every land use, of every management option and
    of the cell itself is read from each run archive and cached by
    tools/make_cell_area_cache.py.  The dominant class is then the largest area
    in the cell, and every total in the table is an area in hectares.

    Two checks are built into the cache and worth trusting: categories 1-7 sum
    to 464.797 Mha, the same study area as land_area_2010_mha in the trade-off
    workbook, and the residual category 8 (public, indigenous, urban,
    plantation, water) comes out identical for all four scenarios, as
    non-modelled land must.

The geography -- the 8-category scheme, the state boundaries, the extent and the
CRS -- is imported from 02_Mapping.py, so this figure cannot drift away from
Fig. 2.  A raster is read only to recover the 2-D position of each model cell.

Outputs
    OUTPUT_DIR/33_Consistency_maps.svg        (svg.fonttype='none')
    EXCEL_DIR/33_consistency_agreement.xlsx   the map shows where the scenarios
        differ, the table how much: per-panel areas, and pairwise matrices for
        land use and for management.
"""

import _path_setup  # noqa: F401

import importlib
import itertools
import os

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tools.parameters import EXCEL_DIR, OUTPUT_DIR, SCENARIO_LABELS, input_files
from tools.text_layout import balanced_wrap

# Relative paths in tools.parameters are anchored to this directory.
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Single source of truth for the geography and the land-use categories.
_map = importlib.import_module('02_Mapping')

OUTPUT_NAME = '33_Consistency_maps.svg'
TABLE_NAME = '33_consistency_agreement.xlsx'
AREA_CACHE = '33_cell_areas.npz'

MM = 1.0 / 25.4
FIG_W_MM = 183.0        # Nature double-column width
FIG_H_MM = 232.0

SAME_COLOR = '#D9D9D9'
DIFF_COLOR = '#B84A4A'
STATE_LINE = '#555555'

N_ROWS, N_COLS = 4, 3

# A panel is about 57 mm wide; at 7 pt that is roughly 46 characters, so 40
# keeps a wrapped title clear of the panel edges and of its neighbour.
TITLE_MAX_CHARS = 40

CODES = ['AgS1', 'AgS2', 'AgS3', 'AgS4']
RUN_OF = {code: run for code, run in zip(CODES, input_files)}
NAME_OF = {code: SCENARIO_LABELS[RUN_OF[code]] for code in CODES}

# "No management option adopted here" -- a state to be compared, not missing data.
NO_MANAGEMENT = -1


def _panel_title(codes):
    """Name the comparison. 'A and B', 'A, B and C' -- never 'A vs B'."""
    names = [NAME_OF[c] for c in codes]
    if len(names) == 4:
        return 'All four scenarios'
    if len(names) == 2:
        return f'{names[0]} and {names[1]}'
    return ', '.join(names[:-1]) + f' and {names[-1]}'


# The six pairs, then the four triples, then all four, then management.
PAIRS = list(itertools.combinations(CODES, 2))
TRIPLES = list(itertools.combinations(CODES, 3))
PANELS = (
    [('landuse', c) for c in PAIRS]
    + [('landuse', c) for c in TRIPLES]
    + [('landuse', tuple(CODES))]
    + [('management', ('AgS1', 'AgS2'))]
)
LETTERS = 'abcdefghijkl'
assert len(PANELS) == 12 == len(LETTERS)


def load_cell_areas():
    """Per-cell areas from the run archives: real area, land use, management."""
    path = os.path.join(EXCEL_DIR, AREA_CACHE)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f'{path} not found. Build it once with the modelling environment:\n'
            '    <xpluto>/python.exe tools/make_cell_area_cache.py'
        )
    cache = np.load(path, allow_pickle=True)
    scenarios = [str(s) for s in cache['scenarios']]
    if scenarios != list(input_files):
        raise ValueError(f'{AREA_CACHE} holds {scenarios}, expected {list(input_files)}')
    return {
        'real_area': cache['real_area_ha'].astype('float64'),
        'cat8': cache['cat8_ha'].astype('float64'),      # (scenario, 8, cell)
        'am': cache['am_ha'].astype('float64'),          # (scenario, option, cell)
        'categories': [str(c) for c in cache['categories']],
        'am_names': [str(a) for a in cache['am_names']],
    }


def raster_mask():
    """2-D positions of the model cells, taken from any of the map rasters.

    extract_nc_layer_as_tiff paints the 1-D model cells into the valid template
    cells in row-major order, so `mask` in C order lines up with the cached 1-D
    arrays element for element.
    """
    path = os.path.join(
        _map.TIFF_DIR,
        f'_extracted_{RUN_OF[CODES[0]]}_map_lumap_lmALL_2050.tiff')
    if not os.path.exists(path):
        raise FileNotFoundError(
            f'{path} not found. Run 02_Mapping.py first -- it extracts the '
            'rasters from each run archive.')
    return ~np.isnan(_map.read_tiff(path))


def dominant(area_by_class, mark_empty=False):
    """Index of the largest area in each cell.

    With mark_empty, a cell where nothing is adopted becomes NO_MANAGEMENT
    rather than silently reporting class 0.
    """
    winner = np.argmax(area_by_class, axis=0)
    if mark_empty:
        winner = np.where(area_by_class.max(axis=0) > 0.0, winner, NO_MANAGEMENT)
    return winner


def same_across(layers):
    """True where every scenario in `layers` shares the same class."""
    same = np.ones(layers[0].shape, dtype=bool)
    for other in layers[1:]:
        same &= layers[0] == other
    return same


def shared_composition_ha(area_stack):
    """Area in the same category in every scenario, cell by cell.

    The dominant-class test is all-or-nothing: a cell counts as Different even
    when the scenarios agree on 95% of it.  This is the area-weighted
    counterpart -- for each class, the smallest area any scenario puts there --
    and it is what the per-cell proportions are for.
    """
    return np.minimum.reduce(area_stack).sum(axis=0)


def draw_panel(ax, same_2d, mask, states, letter, title):
    rgba = np.zeros((*mask.shape, 4), dtype='float32')
    for flag, hexc in ((same_2d & mask, SAME_COLOR), (~same_2d & mask, DIFF_COLOR)):
        r, g, b = _map._hex_rgb(hexc)
        rgba[flag] = [r, g, b, 1.0]
    # Everything outside the study area keeps alpha 0, i.e. plain white.

    ax.imshow(rgba, extent=_map.EXTENT, origin='upper',
              interpolation='nearest', zorder=1)
    if states is not None:
        states.boundary.plot(ax=ax, linewidth=0.3, edgecolor=STATE_LINE, zorder=2)
    _map._style(ax)

    # Wrapped to the panel width: an unwrapped title is wider than the panel and
    # runs straight into the neighbouring map.
    ax.set_title(balanced_wrap(title, TITLE_MAX_CHARS), fontsize=7, pad=3)
    # The letter sits inside the map's empty top-left corner (ocean), where it
    # cannot collide with the title however many lines the title takes.
    ax.text(0.02, 0.98, letter, transform=ax.transAxes,
            fontsize=8, fontweight='bold', ha='left', va='top')


def _pairwise(label, dominants, real_area, total_mha):
    """Square matrices of pairwise agreement, in Mha and in per cent."""
    names = [NAME_OF[c] for c in CODES]
    mha = pd.DataFrame(index=names, columns=names, dtype=float)
    pct = pd.DataFrame(index=names, columns=names, dtype=float)
    for a, b in itertools.product(CODES, CODES):
        if a == b:
            value = total_mha
        else:
            same = same_across([dominants[a], dominants[b]])
            value = real_area[same].sum() / 1e6
        mha.loc[NAME_OF[a], NAME_OF[b]] = round(value, 2)
        pct.loc[NAME_OF[a], NAME_OF[b]] = round(100.0 * value / total_mha, 2)
    mha.index.name = f'Same (Mha) — dominant {label}'
    pct.index.name = f'Same (% of study area) — dominant {label}'
    return mha, pct


def main() -> None:
    cache = load_cell_areas()
    mask = raster_mask()
    real_area = cache['real_area']
    if real_area.size != int(mask.sum()):
        raise ValueError(
            f'cache holds {real_area.size:,} cells but the rasters have '
            f'{int(mask.sum()):,}; the cache belongs to a different run.')

    total_mha = real_area.sum() / 1e6
    allocated_mha = cache['cat8'][0, :7].sum() / 1e6
    print(f'  {real_area.size:,} cells | real area {total_mha:.2f} Mha | '
          f'allocated (categories 1-7) {allocated_mha:.2f} Mha')

    # Dominant class per cell, from the areas rather than the plotting raster.
    dom_lu = {c: dominant(cache['cat8'][i]) for i, c in enumerate(CODES)}
    dom_am = {c: dominant(cache['am'][i], mark_empty=True)
              for i, c in enumerate(CODES)}
    area_lu = {c: cache['cat8'][i] for i, c in enumerate(CODES)}

    states = _map.load_states()
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans', 'sans-serif'],
        'svg.fonttype': 'none',
        'pdf.fonttype': 42,
        'font.size': 7,
        'axes.linewidth': 0.8,
        'legend.frameon': False,
    })

    fig, axes = plt.subplots(N_ROWS, N_COLS,
                             figsize=(FIG_W_MM * MM, FIG_H_MM * MM))
    rows = []
    for ax, letter, (kind, codes) in zip(axes.ravel(), LETTERS, PANELS):
        dominants = dom_lu if kind == 'landuse' else dom_am
        same_1d = same_across([dominants[c] for c in codes])

        same_2d = np.zeros(mask.shape, dtype=bool)
        same_2d[mask] = same_1d

        title = _panel_title(codes)
        if kind == 'management':
            title = f'Agricultural management: {title}'
        draw_panel(ax, same_2d, mask, states, letter, title)

        same_mha = real_area[same_1d].sum() / 1e6
        row = {
            'panel': letter,
            'comparison': title,
            'layer': 'Dominant land use' if kind == 'landuse'
                     else 'Dominant agricultural management',
            'scenarios_compared': len(codes),
            'study_area_Mha': round(total_mha, 2),
            'same_Mha': round(same_mha, 2),
            'different_Mha': round(total_mha - same_mha, 2),
            'same_pct': round(100.0 * same_mha / total_mha, 2),
            'different_pct': round(100.0 * (total_mha - same_mha) / total_mha, 2),
        }
        if kind == 'landuse':
            shared = shared_composition_ha([area_lu[c] for c in codes]).sum() / 1e6
            row['shared_composition_Mha'] = round(shared, 2)
            row['shared_composition_pct'] = round(100.0 * shared / total_mha, 2)
        rows.append(row)
        print(f'  {letter}  same {same_mha:7.2f} Mha ({row["same_pct"]:5.1f}%)   {title}')

    fig.subplots_adjust(left=0.012, right=0.988, bottom=0.055, top=0.965,
                        wspace=0.06, hspace=0.24)
    handles = [
        mpatches.Patch(facecolor=SAME_COLOR, edgecolor='none', label='Same'),
        mpatches.Patch(facecolor=DIFF_COLOR, edgecolor='none', label='Different'),
    ]
    fig.legend(handles=handles, loc='lower center', ncol=2, frameon=False,
               fontsize=7, handlelength=1.1, handleheight=1.1,
               handletextpad=0.5, columnspacing=2.0, bbox_to_anchor=(0.5, 0.008))

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, OUTPUT_NAME)
    fig.savefig(out_path, format='svg', facecolor='white')
    plt.close(fig)

    lu_mha, lu_pct = _pairwise('land use', dom_lu, real_area, total_mha)
    am_mha, am_pct = _pairwise('agricultural management', dom_am, real_area, total_mha)
    table_path = os.path.join(EXCEL_DIR, TABLE_NAME)
    with pd.ExcelWriter(table_path, engine='openpyxl') as writer:
        pd.DataFrame(rows).to_excel(writer, sheet_name='panel_agreement', index=False)
        lu_mha.to_excel(writer, sheet_name='landuse_pairwise_Mha')
        lu_pct.to_excel(writer, sheet_name='landuse_pairwise_pct')
        am_mha.to_excel(writer, sheet_name='management_pairwise_Mha')
        am_pct.to_excel(writer, sheet_name='management_pairwise_pct')

    print(f'  wrote {out_path}')
    print(f'  wrote {table_path}')


if __name__ == '__main__':
    main()
