# ==============================================================================
# Figure: Net economic return (2050) vs Carbon Price
# One line per biodiversity target level
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
BIO_CUTS      = config.BIO_CUTS
YEAR          = 2050

PROFIT_FILES = [
    'xr_economics_ag_profit',
    'xr_economics_am_profit',
    'xr_economics_non_ag_profit',
]

SCENARIO_GROUPS = [
    ('No biodiversity target', 'GBF2_off_CUT_50_CarbonPrice_{cp}'),
] + [
    (f'Restore top {cut}%', f'GBF2_high_CUT_{cut}_CarbonPrice_{{cp}}')
    for cut in BIO_CUTS
]

COLORS  = ['#1d52a1', '#f3793b', '#f0e94b', '#716db2', '#65c8cc', '#72c15a']

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


def read_net_econ_from_zip(zip_path: str, year: int) -> float:
    """Sum all profit components for a given year. Returns B AUD."""
    total = 0.0
    with zipfile.ZipFile(zip_path) as z:
        all_names = z.namelist()
        for fname in PROFIT_FILES:
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
    return total / 1e9   # AUD → B AUD


# ---------------------------------------------------------------------------
# Collect data
# ---------------------------------------------------------------------------
print("Collecting net economic return data...")
series = {}

for label, suffix_tmpl in SCENARIO_GROUPS:
    print(f"\n  Group: {label}")
    val_list = []
    for cp in CARBON_PRICES:
        suffix  = suffix_tmpl.format(cp=cp)
        run_dir = find_run_dir(TASK_ROOT, suffix)
        if run_dir is None:
            print(f"    cp={cp}: run dir not found")
            val_list.append(np.nan)
            continue
        zip_path = os.path.join(run_dir, 'Run_Archive.zip')
        if not os.path.isfile(zip_path):
            print(f"    cp={cp}: MISSING zip")
            val_list.append(np.nan)
            continue
        val = read_net_econ_from_zip(zip_path, YEAR)
        val_list.append(val)
        print(f"    cp={cp}: {val:.2f} Billion AUD")
    series[label] = np.array(val_list)

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6, 5.5))

x = np.array(CARBON_PRICES)

for (label, _), color in zip(SCENARIO_GROUPS, COLORS):
    y    = series[label]
    mask = ~np.isnan(y)
    ax.plot(
        x[mask], y[mask],
        color=color, marker='o', linestyle='-',
        linewidth=1.5, markersize=5,
        markerfacecolor=color, markeredgecolor=color,
        label=label,
    )

ax.set_xlabel(r'Carbon price (AU\$/tCO$_2$e)')
ax.set_ylabel('Net economic return (Billion AU\$)')
# ax.set_title(f'Net economic return vs Carbon price ({YEAR})')
ax.legend(
    frameon=False, fontsize=FS,
    loc='upper center', bbox_to_anchor=(0.5, -0.12),
    ncol=3, handlelength=1.5, handletextpad=0.5, columnspacing=1.0,
)

ax.set_facecolor('white')
ax.grid(False)
for spine in ax.spines.values():
    spine.set_visible(True)
    spine.set_linewidth(1.0)
    spine.set_color('black')

ax.set_xlim(left=0)

plt.tight_layout()
fig.subplots_adjust(bottom=0.22)
out_path = os.path.join(OUT_DIR, f'NetEcon_vs_CarbonPrice_{YEAR}.png')
fig.savefig(out_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"\nSaved: {out_path}")
