"""
12_Mammals_Restoration.py
Which mammal species occur where LUTO restores land (2010 agriculture -> 2050
non-agricultural), using the Australian species occurrence dataset.

This figure focuses on AgS2 (Landscape Stewardship). For each of the 178 mammal
species we count occurrence records (from each species' occur.csv) that fall within
AgS2's restored 5 km cells, rank species, and draw the top-15 as a horizontal bar
chart (EPBC-listed threatened species highlighted).

Inputs:
  - LUTO land-use GeoTIFFs in TIF_DIR: landuse_2010.tif, landuse_<AgS2>_2050.tif
  - Species occurrence points: SPECIES_DIR/mammals/models/<Species>/occur.csv
Outputs:
  - EXCEL_DIR/12_mammals_restoration_ranked.csv   (all 178 species, ranked)
  - OUTPUT_DIR/12_mammals_restoration.png          (top-15 bar chart)
"""
import _path_setup  # noqa: F401

import os
import numpy as np
import pandas as pd
import rasterio
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from tools.parameters import EXCEL_DIR, OUTPUT_DIR, TIF_DIR, input_files, GENERATE_TABLES
from tools.two_row_figure import missing_table_error

SPECIES_DIR = r'F:\Users\s222552331\Work\Species-occurance-points'
RESTORE_SCENARIO = input_files[1]   # Run_2_SCN_AgS2, the scenario assessed here

# Non-agricultural land-use codes (restoration targets)
NONAG_CODES = set(range(100, 109))

COMMON = {
    'Tachyglossus_aculeatus': 'Short-beaked echidna', 'Vombatus_ursinus': 'Common wombat',
    'Phascolarctos_cinereus': 'Koala', 'Petaurus_breviceps': 'Sugar glider',
    'Pseudocheirus_peregrinus': 'Common ringtail possum', 'Ornithorhynchus_anatinus': 'Platypus',
    'Macropus_robustus': 'Common wallaroo', 'Chalinolobus_gouldii': "Gould's wattled bat",
    'Dasyurus_maculatus': 'Spotted-tailed quoll', 'Nyctophilus_geoffroyi': 'Lesser long-eared bat',
    'Macropus_fuliginosus': 'Western grey kangaroo', 'Tadarida_australis': 'White-striped free-tailed bat',
    'Chalinolobus_morio': 'Chocolate wattled bat', 'Antechinus_stuartii': 'Brown antechinus',
    'Pteropus_poliocephalus': 'Grey-headed flying-fox', 'Petaurus_australis': 'Yellow-bellied glider',
    'Nyctophilus_gouldi': "Gould's long-eared bat", 'Perameles_nasuta': 'Long-nosed bandicoot',
    'Miniopterus_schreibersii': 'Southern bent-wing bat', 'Macropus_rufus': 'Red kangaroo',
    'Petrogale_purpureicollis': 'Purple-necked rock-wallaby', 'Petrogale_herberti': "Herbert's rock-wallaby",
    'Pseudomys_occidentalis': 'Western mouse', 'Petrogale_xanthopus': 'Yellow-footed rock-wallaby',
    'Bettongia_penicillata': 'Woylie (brush-tailed bettong)', 'Uromys_hadrourus': 'Masked white-tailed rat',
    'Pseudomys_pilligaensis': 'Pilliga mouse', 'Pseudomys_apodemoides': 'Silky mouse',
    'Acrobates_pygmaeus': 'Feathertail glider', 'Myotis_macropus': 'Southern myotis',
    'Miniopterus_australis': 'Little bent-wing bat',
}
# EPBC-listed threatened (well-established status)
EPBC = {
    'Phascolarctos_cinereus': 'Endangered', 'Dasyurus_maculatus': 'Endangered',
    'Pteropus_poliocephalus': 'Vulnerable', 'Petrogale_xanthopus': 'Vulnerable',
    'Bettongia_penicillata': 'Endangered', 'Pseudomys_pilligaensis': 'Vulnerable',
    'Petaurus_australis': 'Vulnerable', 'Miniopterus_schreibersii': 'Critically Endangered',
}


def build_restoration_mask():
    """Boolean 5 km mask: cells agricultural (0-27) in 2010 -> non-agricultural (>=100) in AgS2 2050."""
    with rasterio.open(os.path.join(TIF_DIR, 'landuse_2010.tif')) as ds:
        lu2010 = ds.read(1)
        transform = ds.transform
        shape = lu2010.shape
        crs = ds.crs
    with rasterio.open(os.path.join(TIF_DIR, f'landuse_{RESTORE_SCENARIO}_2050.tif')) as ds:
        lu2050 = ds.read(1)
        target_transform = ds.transform
        target_crs = ds.crs
    if lu2050.shape != shape or target_transform != transform or target_crs != crs:
        raise ValueError('2010 and 2050 land-use rasters are not spatially aligned')
    if crs is None or not crs.is_geographic:
        raise ValueError(
            f'Land-use raster CRS must use geographic lon/lat coordinates; found {crs}'
        )
    restored = np.isin(lu2050, list(NONAG_CODES)) & (lu2010 >= 0) & (lu2010 < 100)
    return restored, transform, shape


def rank_mammals(restored, transform, shape):
    H, W = shape
    inv = ~transform
    mdir = os.path.join(SPECIES_DIR, 'mammals', 'models')
    species = sorted(d for d in os.listdir(mdir) if os.path.isdir(os.path.join(mdir, d)))
    rows = []
    for sp in species:
        fp = os.path.join(mdir, sp, 'occur.csv')
        if not os.path.exists(fp):
            continue
        df = pd.read_csv(fp, usecols=lambda c: c in ('lon', 'lat'))
        if df.empty:
            continue
        lon = df['lon'].to_numpy(); lat = df['lat'].to_numpy()
        cc, rr = inv * (lon, lat)
        rr = np.floor(rr).astype(int); cc = np.floor(cc).astype(int)
        ok = (rr >= 0) & (rr < H) & (cc >= 0) & (cc < W)
        inr = np.zeros(len(df), bool)
        inr[ok] = restored[rr[ok], cc[ok]]
        rows.append({'species': sp, 'common': COMMON.get(sp, ''), 'EPBC': EPBC.get(sp, ''),
                     'n_records_AUS': int(ok.sum()), 'n_in_restored': int(inr.sum())})
    res = pd.DataFrame(rows)
    res['pct_in_restored'] = (100 * res['n_in_restored'] / res['n_records_AUS'].clip(lower=1)).round(1)
    return res.sort_values('n_in_restored', ascending=False).reset_index(drop=True)


def save_barchart(res):
    top = res.head(15).iloc[::-1].copy()
    top['label'] = top.apply(
        lambda r: r['common'] if isinstance(r['common'], str) and r['common']
        else r['species'].replace('_', ' '), axis=1)
    is_epbc = top['EPBC'].fillna('').astype(str).str.strip().ne('')

    plt.rcParams.update({'font.family': 'Arial', 'font.sans-serif': ['Arial'], 'svg.fonttype': 'none'})
    fig, ax = plt.subplots(figsize=(9.2, 7.0))
    y = np.arange(len(top))
    colors = ['#c0392b' if listed else '#2d688f' for listed in is_epbc]
    ax.barh(y, top['n_in_restored'], color=colors, alpha=0.9)
    ax.set_yticks(y)
    ax.set_yticklabels(top['label'], fontsize=11)
    for i, r in enumerate(top.itertuples()):
        ax.text(r.n_in_restored + 30, i, f"{r.pct_in_restored:.0f}%", va='center', fontsize=9, color='#444')
    ax.set_xlabel('Occurrence records within restored areas', fontsize=11)
    ax.set_title('Top 15 mammals recorded where AgS2 restores land\n'
                 '(% = share of the species’ Australian records that fall in restored areas)',
                 fontsize=12.5, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(handles=[mpatches.Patch(facecolor='#c0392b', label='EPBC-listed threatened'),
                       mpatches.Patch(facecolor='#2d688f', label='Not listed / secure')],
              loc='lower right', fontsize=9.5, frameon=False)

    out = os.path.join(OUTPUT_DIR, '12_mammals_restoration.png')
    fig.savefig(out, dpi=250, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'Saved: {out}')


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(EXCEL_DIR, exist_ok=True)
    csv_path = os.path.join(EXCEL_DIR, '12_mammals_restoration_ranked.csv')
    if GENERATE_TABLES:
        restored, transform, shape = build_restoration_mask()
        print(f'{RESTORE_SCENARIO}: {int(restored.sum())} restored 5 km cells')
        res = rank_mammals(restored, transform, shape)
        res.to_csv(csv_path, index=False)
        print(f'Saved: {csv_path}  ({len(res)} species; '
              f'{int((res["n_in_restored"] > 0).sum())} present in restored areas)')
    if not os.path.exists(csv_path):
        raise missing_table_error(csv_path)
    res = pd.read_csv(csv_path, keep_default_na=False)
    save_barchart(res)


if __name__ == '__main__':
    main()
