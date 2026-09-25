"""
34_Transition_matrix.py -- which land uses can convert into which

A land-use by land-use grid: green where the model is allowed to move land from
the row's use into the column's use, pink where it is not. Rows are the origin,
columns the destination, and the diagonal is always allowed because staying put
is a permitted "transition" at zero cost.

Two separate rules decide what the model can actually do, and the figure has to
apply both or it contradicts itself.  T_MAT says which transitions exist and what
they cost.  NON_AG_LAND_USES_REVERSIBLE then locks the irreversible non-
agricultural land uses in place: once a cell fraction is committed to one, the
solver receives a lower bound equal to last year's value, so that land can never
leave.  In these runs every tree-based non-agricultural land use is irreversible
and only destocked natural land is reversible, which means the tree rows of
T_MAT are full of transitions the model is never able to make.  Drawing T_MAT
alone would show those as permitted, so they are drawn pink like any other
transition the model cannot make.  The two reasons are not distinguished by
colour; the caption states them.

Where the data comes from
    luto.data builds T_MAT, a from_lu x to_lu matrix of establishment costs per
    hectare, out of five files in input/ -- ag_tmatrix.npy, ag_to_ep_tmatrix.npy,
    ag_to_destock_tmatrix.npy, ep_to_ag_tmatrix.npy and
    transition_cost_clearing_forest.npz -- plus a set of rules: non-agricultural
    land cannot return to natural land, clearing non-agricultural land carries a
    specific cost, destocked natural land inherits the unallocated-natural costs
    for livestock, and so on.  A NaN entry means the transition is not permitted;
    a finite entry is its cost.  Only the pattern is drawn here.

    Rather than mirror those rules and risk drifting from the model, the matrix
    is read from the Data object inside a run archive and cached by
    tools/make_transition_cache.py.  GENERATE_TABLES controls which half runs.

Outputs
    OUTPUT_DIR/34_Transition_matrix.svg   (svg.fonttype='none')
    EXCEL_DIR/34_transition_matrix.csv    the cached matrix itself
"""

import _path_setup  # noqa: F401

import ast
import importlib
import importlib.util
import os

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tools.parameters import EXCEL_DIR, GENERATE_TABLES, OUTPUT_DIR

# Relative paths in tools.parameters are anchored to this directory.
os.chdir(os.path.dirname(os.path.abspath(__file__)))

OUTPUT_NAME = '34_Transition_matrix.svg'
CACHE_NAME = '34_transition_matrix.csv'

MM = 1.0 / 25.4
FIG_W_MM = 183.0        # Nature double-column width

# Sampled off the reference figure rather than picked by eye: the modal colour
# of the eroded interiors of its cells, so JPEG bleed at the white gridlines and
# around the labels is excluded.  The blocked colour is a neutral salmon with
# equal green and blue, not the rose it was mistaken for.
ALLOWED_COLOR = '#BCD6AE'    # green  -- the model may make this switch
BLOCKED_COLOR = '#F4D0D0'    # pink   -- it may not, for either reason
GRID_COLOR = 'white'
TEXT_COLOR = '#222222'
LABEL_FONTSIZE = 5.0
AXIS_TITLE_FONTSIZE = 6.5
ROW_LABEL_MM = 47.0     # width taken by the row names
COL_LABEL_MM = 47.0     # height taken by the rotated column names
AXIS_TITLE_MM = 6.0     # strip outside the names for the "From"/"To" axis titles
RIGHT_PAD_MM = 2.0
LEGEND_MM = 13.0

# Display names for the non-agricultural land uses, matching the rest of the
# Ag2050 figures.  Agricultural names are already readable as they stand.
from tools.two_row_figure import RENAME_NON_AG   # noqa: E402


def build_cache():
    """Extract T_MAT from a run archive into EXCEL_DIR.

    Needs joblib and the luto package to unpickle the Data object, which the
    xpluto environment provides; the drawing half needs nothing but pandas.
    """
    missing = [name for name in ('joblib', 'lz4')
               if importlib.util.find_spec(name) is None]
    if missing:
        raise ImportError(
            f'Extracting the transition matrix needs {" and ".join(missing)}. '
            'Run this script with the xpluto environment, or set '
            'GENERATE_TABLES = False in tools/parameters.py to draw from the '
            'cache already in EXCEL_DIR.'
        )
    from tools.make_transition_cache import main as extract
    extract()


def load_matrix() -> pd.DataFrame:
    path = os.path.join(EXCEL_DIR, CACHE_NAME)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f'{path} not found. Build it once with the xpluto environment:\n'
            '    <xpluto>/python.exe tools/make_transition_cache.py\n'
            'or set GENERATE_TABLES = True and run this script with it.'
        )
    frame = pd.read_csv(path, index_col='from_lu')
    frame.columns.name = 'to_lu'
    if list(frame.index) != list(frame.columns):
        raise ValueError(
            'The transition matrix is not square over the same land uses: '
            f'{len(frame.index)} rows, {len(frame.columns)} columns.'
        )
    if frame.index.duplicated().any():
        raise ValueError('Duplicate land uses in the transition matrix')
    return frame


def drop_unavailable(frame: pd.DataFrame) -> pd.DataFrame:
    """Leave out land uses no Paper3 scenario can use.

    T_MAT covers every land use the model knows about, including ones switched
    off for these runs.  BECCS is off in all four scenarios, so a BECCS row and
    column would show transitions that could never happen.  Which ones are off
    is read from the recorded run grid rather than hard-coded, so the figure
    follows the runs if the grid changes.
    """
    grid = os.path.normpath(os.path.join(EXCEL_DIR, '..', '..',
                                         'grid_search_template.csv'))
    if not os.path.exists(grid):
        print(f'  {grid} not found; keeping every land use')
        return frame

    row = pd.read_csv(grid).set_index('Name').loc['NON_AG_LAND_USES']
    settings_all, enabled = set(), set()
    for column in row.index:
        mapping = ast.literal_eval(row[column])
        settings_all |= set(mapping)
        enabled |= {name for name, on in mapping.items() if on}
    never = sorted(settings_all - enabled)
    if never:
        print('  off in every scenario, left out: ' + ', '.join(never))
    keep = [lu for lu in frame.index if lu not in never]
    return frame.loc[keep, keep]


def locked_sources() -> set:
    """Non-agricultural land uses that are held in place in every scenario.

    NON_AG_LAND_USES_REVERSIBLE == False means the solver is given a lower bound
    equal to the previous year's allocation (get_non_ag_lb_matrices in
    luto/economics/non_agricultural/transitions.py), so no land can leave that
    use.  Every transition out of it is then unreachable whatever T_MAT says.

    Read from the recorded run grid for the same reason drop_unavailable does,
    so the figure follows the runs.  Checked against the settings.py archived in
    each Run_Archive.zip: all four scenarios lock every non-agricultural land use
    except destocked natural land.  Names come back in display spelling.
    """
    grid = os.path.normpath(os.path.join(EXCEL_DIR, '..', '..',
                                         'grid_search_template.csv'))
    if not os.path.exists(grid):
        print(f'  {grid} not found; no land use marked as held in place')
        return set()

    row = pd.read_csv(grid).set_index('Name').loc['NON_AG_LAND_USES_REVERSIBLE']
    per_run = [{name for name, on in ast.literal_eval(row[column]).items()
                if not on}
               for column in row.index]
    names = set.intersection(*per_run) if per_run else set()
    locked = {str(RENAME_NON_AG.get(name, name)) for name in names}
    if locked:
        print('  held in place once established: ' + ', '.join(sorted(locked)))
    return locked


def tidy_names(frame: pd.DataFrame) -> pd.DataFrame:
    """Spell the non-agricultural land uses the way the other figures do."""
    return frame.rename(index=RENAME_NON_AG, columns=RENAME_NON_AG)


def draw(frame: pd.DataFrame) -> None:
    allowed = frame.notna().to_numpy()
    n = len(frame)

    # A cell is unreachable for either of two reasons, and the figure does not
    # distinguish them: T_MAT has no entry for it, or the row's land use is
    # locked in place so no land can leave it.  The diagonal is exempt from the
    # lock, since staying put is exactly what the lock forces.
    locked = locked_sources()
    locked_row = np.array([lu in locked for lu in frame.index])[:, None]
    held = allowed & locked_row & ~np.eye(n, dtype=bool)
    available = allowed & ~held

    # Sized from the space the labels actually need rather than from a guessed
    # fraction: the longest name is about 42 characters, which at this size is
    # roughly ROW_LABEL_MM either rotated above the grid or written beside it.
    # AXIS_TITLE_MM is the strip beyond the names that holds the From/To titles.
    left_mm = ROW_LABEL_MM + AXIS_TITLE_MM
    top_mm = COL_LABEL_MM + AXIS_TITLE_MM
    grid_mm = FIG_W_MM - left_mm - RIGHT_PAD_MM
    fig_h_mm = grid_mm + top_mm + LEGEND_MM
    fig, ax = plt.subplots(figsize=(FIG_W_MM * MM, fig_h_mm * MM))

    rgb = np.zeros((n, n, 3), dtype=float)
    for flag, hexc in ((available, ALLOWED_COLOR), (~available, BLOCKED_COLOR)):
        rgb[flag] = [int(hexc[i:i + 2], 16) / 255.0 for i in (1, 3, 5)]
    ax.imshow(rgb, interpolation='nearest', aspect='equal')

    # White lines between the cells rather than a border round each one: at this
    # size a stroked rectangle per cell turns the grid into a grey wash.
    ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax.grid(which='minor', color=GRID_COLOR, linewidth=0.6)
    ax.tick_params(which='minor', length=0)

    labels = list(frame.index)
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(labels, rotation=90, fontsize=LABEL_FONTSIZE,
                       color=TEXT_COLOR)
    ax.set_yticklabels(labels, fontsize=LABEL_FONTSIZE, color=TEXT_COLOR)
    ax.xaxis.set_label_position('top')
    ax.xaxis.tick_top()
    ax.tick_params(axis='both', which='major', length=0, pad=2)
    for side in ax.spines.values():
        side.set_visible(False)

    # Each direction is named on its own axis, centred and outside that axis's
    # band of land-use names.  The previous "From:" prefix on the first row
    # label read as part of a land-use name, and the corner "To" sat directly
    # above the row names, which is the axis it does not describe.
    ax.set_xlabel('To (destination land use)', fontsize=AXIS_TITLE_FONTSIZE,
                  fontweight='bold', color=TEXT_COLOR, labelpad=3.0)
    ax.set_ylabel('From (source land use)', fontsize=AXIS_TITLE_FONTSIZE,
                  fontweight='bold', color=TEXT_COLOR, labelpad=3.0)

    handles = [
        mpatches.Patch(facecolor=ALLOWED_COLOR, edgecolor='none',
                       label='Transition permitted'),
        mpatches.Patch(facecolor=BLOCKED_COLOR, edgecolor='none',
                       label='Transition not permitted'),
    ]
    fig.legend(handles=handles, loc='lower center', ncol=len(handles),
               frameon=False, fontsize=7, handlelength=1.1, handleheight=1.1,
               handletextpad=0.5, columnspacing=2.0, bbox_to_anchor=(0.5, 0.012))

    fig.subplots_adjust(
        left=left_mm / FIG_W_MM,
        right=1.0 - RIGHT_PAD_MM / FIG_W_MM,
        top=1.0 - top_mm / fig_h_mm,
        bottom=LEGEND_MM / fig_h_mm,
    )

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, OUTPUT_NAME)
    fig.savefig(out_path, format='svg', facecolor='white')
    plt.close(fig)
    print(f'  wrote {out_path}')


def main() -> None:
    # GENERATE_TABLES: pull the matrix out of a run archive into EXCEL_DIR
    # first; otherwise draw straight from what is already cached there.
    if GENERATE_TABLES:
        build_cache()
    frame = tidy_names(drop_unavailable(load_matrix()))

    n = frame.shape[0]
    total = frame.size
    in_tmat = int(frame.notna().sum().sum())
    locked = locked_sources()
    locked_row = np.array([lu in locked for lu in frame.index])[:, None]
    held = int((frame.notna().to_numpy() & locked_row
                & ~np.eye(n, dtype=bool)).sum())
    print(f'  {n} land uses | in T_MAT {in_tmat:,} of {total:,} '
          f'({100.0 * in_tmat / total:.1f}%) | of those {held:,} unreachable '
          f'because the source is held in place | actually available '
          f'{in_tmat - held:,} ({100.0 * (in_tmat - held) / total:.1f}%)')
    blocked_rows = frame.isna().sum(axis=1).sort_values(ascending=False)
    print('  most restricted origins in T_MAT: '
          + ', '.join(f'{lu} ({n_blocked})'
                      for lu, n_blocked in blocked_rows.head(3).items()))
    draw(frame)


if __name__ == '__main__':
    main()
