# ==============================================================================
# Figure A: GHG & Net economic return vs Carbon price (BioPrice=0)
# Figure B: GHG & Net economic return vs Bio price    (CarbonPrice=0)
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
TASK_NAME = config.TASK_NAME
TASK_ROOT = f'../../../output/{TASK_NAME}'
OUT_DIR   = f'../../../output/{TASK_NAME}/paper4/figures'
os.makedirs(OUT_DIR, exist_ok=True)

YEAR = 2050

# Read all run dirs once
ALL_RUNS = [d for d in os.listdir(TASK_ROOT)
            if d.startswith('Run_') and os.path.isdir(os.path.join(TASK_ROOT, d))]

# Parse run names → (carbon_price, bio_price)
import re
def parse_prices(run_name):
    cp = re.search(r'CarbonPrice_([\d.]+)', run_name)
    bp = re.search(r'BioPrice_([\d.]+)', run_name)
    return (float(cp.group(1)) if cp else None,
            float(bp.group(1)) if bp else None)

RUN_MAP = {}   # (cp, bp) → zip_path
for run in ALL_RUNS:
    cp, bp = parse_prices(run)
    if cp is not None and bp is not None:
        zp = os.path.join(TASK_ROOT, run, 'Run_Archive.zip')
        if os.path.isfile(zp):
            RUN_MAP[(cp, bp)] = zp

print(f"Found {len(RUN_MAP)} runs with zip archives")

# Grid axes
CP_VALS = sorted(set(k[0] for k in RUN_MAP))
BP_VALS = sorted(set(k[1] for k in RUN_MAP))
print(f"Carbon prices: {CP_VALS}")
print(f"Bio prices:    {BP_VALS}")

GHG_FILES = [
    'xr_GHG_ag', 'xr_GHG_ag_management',
    'xr_GHG_non_ag', 'xr_transition_GHG',
]
PROFIT_FILES = [
    'xr_economics_ag_profit',
    'xr_economics_am_profit',
    'xr_economics_non_ag_profit',
]

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------
def read_sum(zip_path, file_stems, year):
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
# Collect data for two slices:
#   Slice A: BioPrice=0, CarbonPrice varies
#   Slice B: CarbonPrice=0, BioPrice varies
# ---------------------------------------------------------------------------
def collect_slice(fixed_key, fixed_val, vary_vals, vary_key):
    """fixed_key: 'cp' or 'bp'; returns arrays (vary_vals, ghg_Mt, profit_BilAUD)"""
    ghg_list, profit_list = [], []
    for v in vary_vals:
        key = (v, fixed_val) if vary_key == 'cp' else (fixed_val, v)
        zp  = RUN_MAP.get(key)
        if zp is None:
            print(f"  Missing: {key}")
            ghg_list.append(np.nan)
            profit_list.append(np.nan)
            continue
        ghg    = read_sum(zp, GHG_FILES,    YEAR) / 1e6
        profit = read_sum(zp, PROFIT_FILES, YEAR) / 1e9
        ghg_list.append(ghg)
        profit_list.append(profit)
        print(f"  {vary_key}={v}: GHG={ghg:.1f} Mt, Profit={profit:.1f} B AUD")
    return np.array(vary_vals), np.array(ghg_list), np.array(profit_list)


print("\n--- Slice A: BioPrice=0, CarbonPrice varies ---")
x_cp, ghg_cp, profit_cp = collect_slice('bp', 0.0, CP_VALS, 'cp')

print("\n--- Slice B: CarbonPrice=0, BioPrice varies ---")
x_bp, ghg_bp, profit_bp = collect_slice('cp', 0.0, BP_VALS, 'bp')


# ---------------------------------------------------------------------------
# Plot styling
# ---------------------------------------------------------------------------
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
    'mathtext.fontset': 'stixsans',
})

COLOR_GHG    = '#1d52a1'
COLOR_PROFIT = '#f3793b'


def make_dual_plot(x, ghg, profit, xlabel, out_path):
    mask = ~np.isnan(ghg)
    fig, ax1 = plt.subplots(figsize=(6, 5))
    ax2 = ax1.twinx()

    l1, = ax1.plot(x[mask], ghg[mask],
                   color=COLOR_GHG, marker='o', linestyle='-',
                   linewidth=1.5, markersize=5,
                   label=r'GHG emissions (Mt CO$_2$e)')
    l2, = ax2.plot(x[mask], profit[mask],
                   color=COLOR_PROFIT, marker='o', linestyle='-',
                   linewidth=1.5, markersize=5,
                   label='Net economic return (Billion AU$)')

    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(r'GHG emissions (Mt CO$_2$e)', color=COLOR_GHG)
    ax2.set_ylabel('Net economic return (Billion AU$)', color=COLOR_PROFIT)

    ax1.tick_params(axis='y', colors=COLOR_GHG)
    ax2.tick_params(axis='y', colors=COLOR_PROFIT)
    ax1.spines['left'].set_color(COLOR_GHG)
    ax1.spines['left'].set_linewidth(1.0)
    ax2.spines['right'].set_color(COLOR_PROFIT)
    ax2.spines['right'].set_linewidth(1.0)
    for s in ('top', 'bottom'):
        ax1.spines[s].set_visible(True)
        ax1.spines[s].set_linewidth(1.0)
        ax1.spines[s].set_color('black')

    ax1.set_xlim(left=0)
    ax1.set_facecolor('white')
    ax1.grid(False)

    lines  = [l1, l2]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels,
               frameon=False, fontsize=FS,
               loc='upper center', bbox_to_anchor=(0.5, -0.12),
               ncol=1, handlelength=1.5, handletextpad=0.5)

    plt.tight_layout()
    fig.subplots_adjust(bottom=0.28)
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")


# Figure A
make_dual_plot(
    x_cp, ghg_cp, profit_cp,
    xlabel=r'Carbon price (AU\$/tCO$_2$e)',
    out_path=os.path.join(OUT_DIR, f'GHG_NetEcon_vs_CarbonPrice_{YEAR}.png'),
)

# Figure B
make_dual_plot(
    x_bp, ghg_bp, profit_bp,
    xlabel=r'Biodiversity price (AU\$/ha)',
    out_path=os.path.join(OUT_DIR, f'GHG_NetEcon_vs_BioPrice_{YEAR}.png'),
)
