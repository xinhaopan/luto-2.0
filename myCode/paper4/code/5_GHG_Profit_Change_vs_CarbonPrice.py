# ==============================================================================
# Figure: GHG reduction and profit increase (2050 vs 2010) vs Carbon Price
# No biodiversity target only; dual y-axis
# ==============================================================================

import io
import os
import sys
import zipfile

import cf_xarray as cfxr
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
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

CARBON_PRICES = config.CARBON_PRICES
SUFFIX_TMPL   = 'GBF2_off_CUT_50_CarbonPrice_{cp}'

GHG_FILES = [
    'xr_GHG_ag',
    'xr_GHG_ag_management',
    'xr_GHG_non_ag',
    'xr_transition_GHG',
]
PROFIT_FILES = [
    'xr_economics_ag_profit',
    'xr_economics_am_profit',
    'xr_economics_non_ag_profit',
]

COLOR_GHG    = '#1d52a1'
COLOR_PROFIT = '#f3793b'

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
# Helpers
# ---------------------------------------------------------------------------
def find_run_dir(task_root, suffix):
    for name in os.listdir(task_root):
        if name.endswith(suffix):
            return os.path.join(task_root, name)
    return None


def read_sum_from_zip(zip_path, file_stems, year):
    """Sum all listed NetCDF file stems for a given year. Returns raw sum."""
    total = 0.0
    with zipfile.ZipFile(zip_path) as z:
        all_names = z.namelist()
        for fname in file_stems:
            target = f'{fname}_{year}.nc'
            matches = [n for n in all_names if n.endswith(target)]
            if not matches:
                continue
            with z.open(matches[0]) as zf:
                ds = xr.open_dataset(io.BytesIO(zf.read()), engine='h5netcdf')
            if 'layer' in ds.dims and 'compress' in ds['layer'].attrs:
                ds = cfxr.decode_compress_to_multi_index(ds, 'layer')
            da = list(ds.data_vars.values())[0]
            for dim in list(da.dims):
                if dim != 'cell' and 'ALL' in da.coords[dim].values:
                    da = da.sel({dim: 'ALL'})
            total += float(da.sum())
    return total


# ---------------------------------------------------------------------------
# Collect data
# ---------------------------------------------------------------------------
print("Collecting data...")
ghg_change    = []   # GHG(2050) - GHG(2010), Mt CO2e
profit_change = []   # Profit(2050) - Profit(2010), Billion AUD

for cp in CARBON_PRICES:
    suffix  = SUFFIX_TMPL.format(cp=cp)
    run_dir = find_run_dir(TASK_ROOT, suffix)
    if run_dir is None:
        print(f"  cp={cp}: run dir not found")
        ghg_change.append(np.nan)
        profit_change.append(np.nan)
        continue
    zip_path = os.path.join(run_dir, 'Run_Archive.zip')
    if not os.path.isfile(zip_path):
        print(f"  cp={cp}: MISSING zip")
        ghg_change.append(np.nan)
        profit_change.append(np.nan)
        continue

    ghg_2010   = read_sum_from_zip(zip_path, GHG_FILES,    2010) / 1e6
    ghg_2050   = read_sum_from_zip(zip_path, GHG_FILES,    2050) / 1e6
    profit_2010 = read_sum_from_zip(zip_path, PROFIT_FILES, 2010) / 1e9
    profit_2050 = read_sum_from_zip(zip_path, PROFIT_FILES, 2050) / 1e9

    dghg   = ghg_2050   - ghg_2010
    dprofit = profit_2050 - profit_2010

    ghg_change.append(dghg)
    profit_change.append(dprofit)
    print(f"  cp={cp}: dGHG={dghg:.1f} Mt CO2e,  dProfit={dprofit:.2f} B AUD")

ghg_change    = np.array(ghg_change)
profit_change = np.array(profit_change)
cp            = np.array(CARBON_PRICES)

# GHG abatement = reduction relative to 2010 (positive = less emissions)
ghg_abatement = -ghg_change   # Mt CO2e abated

# ---------------------------------------------------------------------------
# Plot: x = GHG abatement, y = carbon price
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6, 5.5))

mask = ~np.isnan(ghg_abatement)

ax.plot(ghg_abatement[mask], cp[mask],
        color=COLOR_GHG, marker='o', linestyle='-',
        linewidth=1.5, markersize=5)

ax.set_xlabel(r'GHG abatement (Mt CO$_2$e)')
ax.set_ylabel(r'Carbon price (AU\$/tCO$_2$e)')

ax.set_ylim(bottom=0)
ax.set_facecolor('white')
ax.grid(False)
for spine in ax.spines.values():
    spine.set_visible(True)
    spine.set_linewidth(1.0)
    spine.set_color('black')

plt.tight_layout()

out_path = os.path.join(OUT_DIR, 'GHG_Abatement_vs_CarbonPrice_2050.png')
fig.savefig(out_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"\nSaved: {out_path}")
