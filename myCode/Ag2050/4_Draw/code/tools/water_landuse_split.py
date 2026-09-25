"""
Land-use component of water-yield change, attributed cell by cell to the land-use
category that RECEIVES the land.  Feeds row 2 of Extended Data Fig. 9
(19_Water.py) and the water row of Fig. 3 (03_indicators.py).

Why the old row 2 was wrong
    It plotted, for each category, (yield of the cells in that category this year)
    minus (yield of the cells in that category in 2010).  Nothing netted the
    transfers between categories and nothing took the climate signal out, so a
    category that merely lost land (Unallocated land) plunged by the full yield of
    what it lost, and the seven categories summed to about three times the land-use
    impact in row 1.

What is computed instead
    Write n_t, the per-cell net-yield rate of every (land management, land use) at
    year t exactly as write.py uses it (get_water_net_yield_matrices), x_t the
    agricultural allocation and x_0 its 2010 value.  The land-use part of the change
    in agricultural plus non-agricultural net yield splits exactly into

        sum n_t (x_t - x_0) + nonag_t          land moving between uses, valued
                                               at this year's rates
      + sum [(n_t - n_0) - (wy_t - wy_0)] x_0  requirement drift on the 2010
                                               pattern (the livestock water
                                               requirement change that row 1 also
                                               keeps out of the climate component)

    and the climate component of row 1, sum (wy_t - wy_0) x_0, is the remainder.

    The first term is attributed within each cell.  Categories whose share of the
    cell grew are receivers, those whose share shrank are sources.  A receiver is
    credited with the yield of the land it gained, under its own use, minus the
    yield that land would have delivered this year under the uses it came from
    (the sources' contributions pooled and split among receivers in proportion to
    the land each gained).  Sources get zero.  A category whose share did not change
    keeps its own within-category reshuffle.  Non-agricultural land is always a
    receiver because it holds nothing in 2010.

    The second term is attributed to categories by their 2010 occupancy, which is
    the same pattern row 1's climate component is evaluated on.

    Summed over categories the result equals row 1's land-use impact minus the
    agricultural-management change in row 3, to rounding -- checked here and again
    in 19_Water.py.

Why not a new model output
    The instruction was to add a per-cell, per-land-use climate output to LUTO2 and
    re-run the write step.  Every quantity needed is already in the Data_RES*.lz4
    object each run archives, so it is read from there, as the climate cache in
    03_indicators.py already is.  No model code changes and no re-runs.

Needs the xpluto environment (luto importable, joblib, lz4).
"""
import gc
import os
import sys
import zipfile

import numpy as np
import pandas as pd

from tools.parameters import EXCEL_DIR
from tools.two_row_figure import classify_land_use, input_files
from tools.data_helper import get_zip_info

CATEGORIES = [
    'Dryland cropland and horticulture',
    'Dryland grazing (modified pastures)',
    'Grazing (native vegetation)',
    'Irrigated cropland and horticulture',
    'Irrigated grazing (modified pastures)',
    'Unallocated land',
    'Non-agricultural land-use',
]
N_AG_CATS = len(CATEGORIES) - 1
CACHE_CATS = '19_water_landuse_transfer.csv'
CACHE_TOTALS = '19_water_landuse_transfer_totals.csv'
EPS = 1e-9          # below this a change in cell share is treated as none


def _import_luto():
    repo = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../..'))
    if repo not in sys.path:
        sys.path.insert(0, repo)
    import luto.settings as settings
    settings.AG2050_MODE = True
    import luto.data  # noqa: F401  (register classes for unpickling)
    import luto.economics.agricultural.water as ag_water
    import luto.economics.non_agricultural.water as non_ag_water
    return ag_water, non_ag_water


def _onehot(data):
    """(2*N_AG_LUS, N_AG_CATS): column c marks the (m, j) pairs in category c."""
    n_lu = data.N_AG_LUS
    oh = np.zeros((2 * n_lu, N_AG_CATS), dtype=np.float64)
    for m, lm in enumerate(('dry', 'irr')):
        for j in range(n_lu):
            cat = classify_land_use(data.AGLU2DESC[j], lm)
            if cat is None:
                raise ValueError(f'{data.AGLU2DESC[j]} ({lm}) has no water category')
            oh[m * n_lu + j, CATEGORIES.index(cat)] = 1.0
    return oh


def _flat(a):
    """(m, r, j) -> (r, m*j), matching the row order of _onehot."""
    return np.asarray(a, dtype=np.float64).transpose(1, 0, 2).reshape(a.shape[1], -1)


# The report CSVs that row 1 is built from are written through tools.ag_mrj_to_xr
# and tools.non_ag_rk_to_xr, which zero every cell whose total (over land uses,
# and land managements for agriculture) is at most 0.01 in absolute value, after
# casting to float32.  write.py applies that to the dvars AND to the water
# matrices.  Without the same masks here the non-agricultural total runs a few
# GL high (riparian plantings are capped at the stream-buffer share of a cell and
# often sit below 1 %), and row 2 no longer adds up to row 1.
REPORT_THRESHOLD = 0.01


def _report_ag(a):
    a = np.asarray(a, dtype=np.float32)
    keep = np.abs(a.sum(axis=(0, 2))) > REPORT_THRESHOLD
    return np.where(keep[None, :, None], a, 0.0).astype(np.float64)


def _report_nonag(a):
    a = np.asarray(a, dtype=np.float32)
    keep = np.abs(a.sum(axis=1)) > REPORT_THRESHOLD
    return np.where(keep[:, None], a, 0.0).astype(np.float64)


def _attribute(v, g):
    """Credit each cell's transfer to the categories that gained land.

    v : (r, c) this-year net yield of the change in each category's land
    g : (r, c) change in each category's share of the cell
    Returns (r, c) with the row sums of v preserved exactly.
    """
    rec = g > EPS
    src = g < -EPS
    gained = np.where(rec, g, 0.0).sum(axis=1)
    pooled = np.where(src, v, 0.0).sum(axis=1)
    move = (gained > EPS) & src.any(axis=1)
    share = np.where(rec, g / np.where(gained > EPS, gained, 1.0)[:, None], 0.0)
    moved = np.where(rec, v + share * pooled[:, None], np.where(src, 0.0, v))
    return np.where(move[:, None], moved, v)


def _am_net(ag_water, data, yr_idx, year):
    """Agricultural-management net yield, for the bookkeeping totals only.

    The water matrices carry one column per land use the option applies to, the
    stored dvars all 28, so the dvars are cut down to the option's land uses.
    """
    import luto.settings as settings
    am_w = ag_water.get_agricultural_management_water_matrices(data, yr_idx)
    dvars = data.ag_man_dvars.get(year, {})
    total = 0.0
    for a, w in am_w.items():
        if a not in dvars:
            continue
        d = np.asarray(dvars[a])
        w = np.asarray(w)
        if d.shape[-1] != w.shape[-1]:
            idx = [data.DESC2AGLU[lu] for lu in settings.AG_MANAGEMENTS_TO_LAND_USES[a]]
            d = d[..., idx]
        total += float((w * d).sum())
    return total


def _one_scenario(scenario, ag_water, non_ag_water):
    info = get_zip_info(scenario)
    if info is None:
        raise FileNotFoundError(f'No Run_Archive.zip for {scenario}')
    zip_path, _prefix = info
    tmp = os.path.join(EXCEL_DIR, f'_tmp_Data_{scenario}.lz4')
    with zipfile.ZipFile(zip_path) as z:
        member = next(n for n in z.namelist() if n.endswith('.lz4') and 'Data_RES' in n)
        with z.open(member) as src, open(tmp, 'wb') as dst:
            while True:
                chunk = src.read(1 << 24)
                if not chunk:
                    break
                dst.write(chunk)
    import joblib
    data = joblib.load(tmp)
    os.remove(tmp)

    oh = _onehot(data)
    base = data.YR_CAL_BASE
    # Climate component exactly as row 1 takes it from 03_indicators: yield-only
    # rates on the unmasked 2010 allocation AG_L_MRJ.
    x_clim = np.asarray(data.AG_L_MRJ, dtype=np.float64)
    fx_clim = _flat(x_clim)
    wy0 = ag_water.get_wyield_matrices(data, 0)
    # Everything else in the report's own terms (masked, float32 first).
    x0 = _report_ag(data.ag_dvars[base])
    n0 = _report_ag(ag_water.get_water_net_yield_matrices(data, 0))
    ag_net_0 = float((n0 * x0).sum())
    am_net_0 = _am_net(ag_water, data, 0, base)
    fx0 = _flat(x0)

    cats, totals = [], []
    for year in sorted(data.ag_dvars):
        yi = year - base
        n_raw = ag_water.get_water_net_yield_matrices(data, yi)
        n_t = _report_ag(n_raw)
        wy_t = ag_water.get_wyield_matrices(data, yi)
        x_t = _report_ag(data.ag_dvars[year])
        dx = x_t - x0

        v_ag = _flat(n_t * dx) @ oh
        g_ag = _flat(dx) @ oh
        y_t = _report_nonag(data.non_ag_dvars[year])
        nn_t = _report_nonag(non_ag_water.get_w_net_yield_matrix(
            data, n_t.astype(np.float32), data.lumaps[year], yi))
        v_na = (nn_t * y_t).sum(axis=1)
        v = np.column_stack([v_ag, v_na])
        g = np.column_stack([g_ag, y_t.sum(axis=1)])

        transfer = _attribute(v, g).sum(axis=0)
        # Rate drift on the 2010 pattern minus the climate component, both by
        # category: what is left is the requirement drift row 1 keeps out of
        # the climate component.
        drift = ((_flat(n_t - n0) * fx0) @ oh).sum(axis=0) \
            - ((_flat(wy_t - wy0) * fx_clim) @ oh).sum(axis=0)
        values = transfer + np.append(drift, 0.0)

        clim = float(((wy_t - wy0) * x_clim).sum())
        ag_net = float((n_t * x_t).sum())
        nonag_net = float(v_na.sum())
        am_net = _am_net(ag_water, data, yi, year)

        target = (ag_net - ag_net_0) + nonag_net - clim
        if not np.isclose(values.sum(), target, rtol=0, atol=1.0):   # ML
            raise AssertionError(
                f'{scenario} {year}: categories sum {values.sum():,.1f} ML but the '
                f'land-use target is {target:,.1f} ML')

        for c, val in zip(CATEGORIES, values):
            cats.append({'scenario': scenario, 'year': int(year),
                         'category': c, 'value_GL': val / 1e3})
        totals.append({'scenario': scenario, 'year': int(year),
                       'ag_net_GL': ag_net / 1e3, 'nonag_net_GL': nonag_net / 1e3,
                       'am_net_GL': am_net / 1e3, 'climate_GL': clim / 1e3,
                       'ag_net_2010_GL': ag_net_0 / 1e3,
                       'am_net_2010_GL': am_net_0 / 1e3})
        print(f'  {scenario} {year}  land-use {values.sum() / 1e3:10.1f} GL  '
              f'climate {clim / 1e3:10.1f} GL', flush=True)

    del data
    gc.collect()
    return cats, totals


def compute():
    ag_water, non_ag_water = _import_luto()
    os.makedirs(EXCEL_DIR, exist_ok=True)
    cats, totals = [], []
    for scenario in input_files:
        print(f'{scenario}: loading Data object ...', flush=True)
        c, t = _one_scenario(scenario, ag_water, non_ag_water)
        cats += c
        totals += t
    cats = pd.DataFrame(cats)
    totals = pd.DataFrame(totals)
    cats.to_csv(os.path.join(EXCEL_DIR, CACHE_CATS), index=False)
    totals.to_csv(os.path.join(EXCEL_DIR, CACHE_TOTALS), index=False)
    return cats, totals


def load():
    """(categories, totals) from the cache; raises if it has not been built."""
    paths = [os.path.join(EXCEL_DIR, n) for n in (CACHE_CATS, CACHE_TOTALS)]
    missing = [p for p in paths if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(
            'Missing water land-use split cache:\n  ' + '\n  '.join(missing)
            + '\nBuild it with the xpluto environment:\n'
              '    python -c "import _path_setup; from tools.water_landuse_split import compute; compute()"')
    return pd.read_csv(paths[0]), pd.read_csv(paths[1])
