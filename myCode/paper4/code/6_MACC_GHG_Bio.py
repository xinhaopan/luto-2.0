# ==============================================================================
# Figure 1: GHG abatement vs Carbon price  (MACC-style)
#   x = GHG_baseline(2050) − GHG(2050)  [positive = abatement],  y = carbon price
#   Slice: BioPrice=0, CarbonPrice varies
#
# Figure 2: Biodiversity change vs Bio price  (MACC-style)
#   x = Bio(2050) − Bio_baseline(2050),   y = bio price
#   Slice: CarbonPrice=0, BioPrice varies
# ==============================================================================

import io
import os
import re
import sys
import zipfile

import cf_xarray as cfxr
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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
CACHE_PATH = os.path.join(OUT_DIR, f'MACC_raw_data_{YEAR}.xlsx')

GHG_FILES = [
    'xr_GHG_ag', 'xr_GHG_ag_management',
    'xr_GHG_non_ag', 'xr_transition_GHG',
]
BIO_FILES = [
    'xr_biodiversity_overall_priority_ag',
    'xr_biodiversity_overall_priority_ag_management',
    'xr_biodiversity_overall_priority_non_ag',
]

COLOR_GHG = '#1d52a1'
COLOR_BIO = '#72c15a'

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

# ---------------------------------------------------------------------------
# Build run map: (cp, bp) → zip_path
# ---------------------------------------------------------------------------
def parse_prices(run_name):
    cp = re.search(r'CarbonPrice_([\d.]+)', run_name)
    bp = re.search(r'BioPrice_([\d.]+)', run_name)
    return (float(cp.group(1)) if cp else None,
            float(bp.group(1)) if bp else None)

RUN_MAP = {}
for run in os.listdir(TASK_ROOT):
    if not run.startswith('Run_'):
        continue
    cp, bp = parse_prices(run)
    if cp is None or bp is None:
        continue
    zp = os.path.join(TASK_ROOT, run, 'Run_Archive.zip')
    if os.path.isfile(zp):
        RUN_MAP[(cp, bp)] = zp

CP_VALS = sorted(set(k[0] for k in RUN_MAP))
BP_VALS = sorted(set(k[1] for k in RUN_MAP))


# ---------------------------------------------------------------------------
# Helper: sum NetCDF files from zip, selecting 'ALL' on non-cell dims
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
# Load from cache or read from zip files, then save cache
# ---------------------------------------------------------------------------
if os.path.isfile(CACHE_PATH):
    print(f"Loading cached data from {CACHE_PATH}")
    df_ghg = pd.read_excel(CACHE_PATH, sheet_name='GHG_slice')
    df_bio = pd.read_excel(CACHE_PATH, sheet_name='Bio_slice')
    base_ghg = df_ghg.attrs.get('base_ghg', df_ghg['GHG_MtCO2e'].iloc[0] - df_ghg['dGHG_MtCO2e'].iloc[0])
    base_bio = df_bio.attrs.get('base_bio', df_bio['Bio'].iloc[0] - df_bio['dBio'].iloc[0])
    # Recover base values stored in first row (cp=0 / bp=0)
    base_ghg = df_ghg.loc[df_ghg['CarbonPrice'] == 0.0, 'GHG_MtCO2e'].values[0]
    base_bio = df_bio.loc[df_bio['BioPrice'] == 0.0, 'Bio'].values[0]
    print(f"Baseline GHG={base_ghg:.1f} Mt CO2e,  Bio={base_bio:.2f}")
else:
    # --- Baseline ---
    base_zp  = RUN_MAP[(0.0, 0.0)]
    base_ghg = read_sum(base_zp, GHG_FILES, YEAR) / 1e6
    base_bio = read_sum(base_zp, BIO_FILES, YEAR)
    print(f"Baseline GHG={base_ghg:.1f} Mt CO2e,  Bio={base_bio:.2f}")

    # --- Slice A: BioPrice=0, CarbonPrice varies ---
    print("\n--- GHG change vs Carbon price (BioPrice=0) ---")
    rows_ghg = []
    for cp in CP_VALS:
        zp = RUN_MAP.get((cp, 0.0))
        ghg = read_sum(zp, GHG_FILES, YEAR) / 1e6 if zp else np.nan
        dghg = ghg - base_ghg if not np.isnan(ghg) else np.nan
        rows_ghg.append({'CarbonPrice': cp, 'GHG_MtCO2e': ghg, 'dGHG_MtCO2e': dghg})
        print(f"  cp={cp}: GHG={ghg:.1f}, dGHG={dghg:.1f} Mt CO2e")
    df_ghg = pd.DataFrame(rows_ghg)

    # --- Slice B: CarbonPrice=0, BioPrice varies ---
    print("\n--- Bio change vs Bio price (CarbonPrice=0) ---")
    rows_bio = []
    for bp in BP_VALS:
        zp = RUN_MAP.get((0.0, bp))
        bio = read_sum(zp, BIO_FILES, YEAR) if zp else np.nan
        dbio = bio - base_bio if not np.isnan(bio) else np.nan
        rows_bio.append({'BioPrice': bp, 'Bio': bio, 'dBio': dbio})
        print(f"  bp={bp}: Bio={bio:.2f}, dBio={dbio:.2f}")
    df_bio = pd.DataFrame(rows_bio)

    # --- Save to Excel cache ---
    with pd.ExcelWriter(CACHE_PATH, engine='openpyxl') as writer:
        df_ghg.to_excel(writer, sheet_name='GHG_slice', index=False)
        df_bio.to_excel(writer, sheet_name='Bio_slice', index=False)
    print(f"\nCache saved: {CACHE_PATH}")

# ---------------------------------------------------------------------------
# Prepare plot arrays
# ---------------------------------------------------------------------------
x_ghg = -df_ghg['dGHG_MtCO2e'].to_numpy()   # abatement = positive reduction
y_cp  = df_ghg['CarbonPrice'].to_numpy()

x_bio = df_bio['dBio'].to_numpy()
y_bp  = df_bio['BioPrice'].to_numpy()


# ---------------------------------------------------------------------------
# Plot helper
# ---------------------------------------------------------------------------
def macc_plot(x, y, xlabel, ylabel, color, out_path, xlim_left=None):
    mask = ~np.isnan(x)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(x[mask], y[mask],
            color=color, marker='o', linestyle='-',
            linewidth=1.5, markersize=5)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if xlim_left is not None:
        ax.set_xlim(left=xlim_left)
    ax.set_ylim(bottom=0)
    ax.set_facecolor('white')
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.0)
        spine.set_color('black')

    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")


# Figure 1: GHG MACC  (abatement ≥ 0, start x-axis at 0)
macc_plot(
    x=x_ghg, y=y_cp,
    xlabel=r'GHG abatement (Mt CO$_2$e)',
    ylabel=r'Carbon price (AU\$/tCO$_2$e)',
    color=COLOR_GHG,
    xlim_left=0,
    out_path=os.path.join(OUT_DIR, f'MACC_GHG_vs_CarbonPrice_{YEAR}.png'),
)

# Figure 2: Biodiversity MACC  (x-axis auto, direction TBD)
macc_plot(
    x=x_bio, y=y_bp,
    xlabel='Biodiversity change (contribution-weighted area)',
    ylabel=r'Biodiversity price (AU\$/ha)',
    color=COLOR_BIO,
    out_path=os.path.join(OUT_DIR, f'MACC_Bio_vs_BioPrice_{YEAR}.png'),
)
