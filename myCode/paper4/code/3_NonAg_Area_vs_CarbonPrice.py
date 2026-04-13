# ==============================================================================
# Figure: Non-agricultural land area (2050) vs Carbon Price
# 2×3 panel, one subplot per biodiversity target, stacked bar chart
# ==============================================================================

import io
import os
import sys
import zipfile

import cf_xarray as cfxr
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tools.config as config

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
TASK_NAME     = config.TASK_NAME
TASK_ROOT     = f'../../../output/{TASK_NAME}'
OUT_DIR       = f'../../../output/{TASK_NAME}/paper4/figures'
os.makedirs(OUT_DIR, exist_ok=True)

COLOR_FILE    = 'tools/land use colors.xlsx'

CARBON_PRICES = config.CARBON_PRICES
BIO_CUTS      = config.BIO_CUTS
YEAR          = 2050

SCENARIO_GROUPS = [
    ('No biodiversity target', 'GBF2_off_CUT_50_CarbonPrice_{cp}'),
] + [
    (f'Restore top {cut}%', f'GBF2_high_CUT_{cut}_CarbonPrice_{{cp}}')
    for cut in BIO_CUTS
]

FS = 11
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
# Load color mapping from carbonprice Excel
# ---------------------------------------------------------------------------
color_df = pd.read_excel(COLOR_FILE, sheet_name='non_ag')
# Build case-insensitive name → hex color dict
COLOR_MAP = {
    row['desc'].strip().lower().replace('-', '').replace(' ', ''):
    row['color']
    for _, row in color_df.iterrows()
}

def get_lu_color(lu_name):
    """Return hex color for a non-ag land use name."""
    key = lu_name.strip().lower().replace('-', '').replace(' ', '')
    return COLOR_MAP.get(key, '#888888')


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def find_run_dir(task_root, suffix):
    for name in os.listdir(task_root):
        if name.endswith(suffix):
            return os.path.join(task_root, name)
    return None


def read_nonag_area_from_zip(zip_path: str, year: int) -> dict:
    """
    Returns {lu_name: area_Mha} for a given year.
    Excludes 'ALL' and zero-area types.
    """
    fname = f'xr_area_non_agricultural_landuse_{year}.nc'
    with zipfile.ZipFile(zip_path) as z:
        matches = [n for n in z.namelist() if n.endswith(fname)]
        if not matches:
            return {}
        with z.open(matches[0]) as zf:
            ds = xr.open_dataset(io.BytesIO(zf.read()), engine='h5netcdf')
    if 'layer' in ds.dims and 'compress' in ds['layer'].attrs:
        ds = cfxr.decode_compress_to_multi_index(ds, 'layer')
    da = list(ds.data_vars.values())[0]
    lu_vals = da.coords['lu'].values
    result = {}
    for lu in lu_vals:
        if lu == 'ALL':
            continue
        val = float(da.sel(lu=lu).sum()) / 1e6   # Mha
        if val > 0:
            result[lu] = val
    return result


# ---------------------------------------------------------------------------
# Collect data: series[label] = list of dicts {lu: Mha} per carbon price
# ---------------------------------------------------------------------------
print("Collecting non-ag area data...")
series = {}

for label, suffix_tmpl in SCENARIO_GROUPS:
    label_key = label.replace('\n', ' ')
    print(f"\n  Group: {label_key}")
    cp_data = []
    for cp in CARBON_PRICES:
        suffix  = suffix_tmpl.format(cp=cp)
        run_dir = find_run_dir(TASK_ROOT, suffix)
        if run_dir is None:
            print(f"    cp={cp}: run dir not found")
            cp_data.append(None)
            continue
        zip_path = os.path.join(run_dir, 'Run_Archive.zip')
        if not os.path.isfile(zip_path):
            print(f"    cp={cp}: MISSING zip")
            cp_data.append(None)
            continue
        areas = read_nonag_area_from_zip(zip_path, YEAR)
        total = sum(areas.values())
        print(f"    cp={cp}: total {total:.2f} Mha  ({list(areas.keys())})")
        cp_data.append(areas)
    series[label] = cp_data


# ---------------------------------------------------------------------------
# Determine all land use types present (union across all scenarios/prices)
# ---------------------------------------------------------------------------
all_lu = []
seen = set()
for cp_data in series.values():
    for d in cp_data:
        if d is None:
            continue
        for lu in d:
            if lu not in seen:
                all_lu.append(lu)
                seen.add(lu)

# Sort: keep a preferred order matching the color file
preferred_order = [r['desc'] for _, r in color_df.iterrows()
                   if r['desc'] not in ('Agricultural land-use', 'Other land-use')]
all_lu_sorted = [lu for lu in preferred_order if lu in seen] + \
                [lu for lu in all_lu if lu not in preferred_order]

print("\nLand use types found:", all_lu_sorted)

# ---------------------------------------------------------------------------
# Plot: 2 rows × 3 columns, shared y-axis
# ---------------------------------------------------------------------------
n_groups = len(SCENARIO_GROUPS)   # 6
nrows, ncols = 2, 3

fig, axes = plt.subplots(
    nrows, ncols,
    figsize=(12, 7),
    sharey=True,
    constrained_layout=False,
)
axes_flat = axes.flatten()

x      = np.arange(len(CARBON_PRICES))
width  = 0.75

tick_positions = list(range(len(CARBON_PRICES)))
tick_labels = [
    f'{cp:.2f}' if cp != int(cp) else str(int(cp))
    for cp in CARBON_PRICES
]

for idx, (ax, (label, _)) in enumerate(zip(axes_flat, SCENARIO_GROUPS)):
    cp_data = series[label]
    bottoms = np.zeros(len(CARBON_PRICES))

    for lu in all_lu_sorted:
        heights = np.array([
            (d[lu] if d is not None and lu in d else 0.0)
            for d in cp_data
        ])
        if heights.sum() == 0:
            continue
        color = get_lu_color(lu)
        ax.bar(x, heights, width, bottom=bottoms, color=color, label=lu)
        bottoms += heights

    ax.set_title(label, fontsize=FS)
    ax.set_xticks(tick_positions)

    # x-tick labels only on bottom row; hide on top row
    if idx >= (nrows - 1) * ncols:
        ax.set_xticklabels(tick_labels, rotation=90, ha='center')
    else:
        ax.set_xticklabels([])

    ax.set_facecolor('white')
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.8)
        spine.set_color('black')

# Single shared y-axis label centred between the two rows
fig.text(0.02, 0.55, 'Area (Mha)',
         va='center', ha='center', rotation='vertical', fontsize=FS)

# ---------------------------------------------------------------------------
# Shared legend below figure
# ---------------------------------------------------------------------------
handles = []
for lu in all_lu_sorted:
    color = get_lu_color(lu)
    handles.append(mpatches.Patch(facecolor=color, edgecolor='none', label=lu))

fig.legend(
    handles=handles,
    loc='lower center',
    bbox_to_anchor=(0.5, 0.0),
    ncol=4,
    frameon=False,
    fontsize=FS,
    handlelength=1.0,
    handleheight=1.0,
    handletextpad=0.5,
    columnspacing=1.0,
)

plt.tight_layout()
plt.subplots_adjust(left=0.07, bottom=0.22, right=0.99, top=0.96, hspace=0.35, wspace=0.08)

# Single shared x-axis label — placed just below the bottom row tick labels
fig.text(0.53, 0.12, r'Carbon price (AU\$/tCO$_2$e)',
         va='center', ha='center', fontsize=FS)

out_path = os.path.join(OUT_DIR, f'NonAg_Area_vs_CarbonPrice_{YEAR}.png')
fig.savefig(out_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"\nSaved: {out_path}")
