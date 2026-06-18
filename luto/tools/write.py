# Copyright 2025 Bryan, B.A., Williams, N., Archibald, C.L., de Haan, F., Wang, J., 
# van Schoten, N., Hadjikakou, M., Sanson, J.,  Zyngier, R., Marcos-Martinez, R.,  
# Navarro, J.,  Gao, L., Aghighi, H., Armstrong, T., Bohl, H., Jaffe, P., Khan, M.S., 
# Moallemi, E.A., Nazari, A., Pan, X., Steyl, D., and Thiruvady, D.R.
#
# This file is part of LUTO2 - Version 2 of the Australian Land-Use Trade-Offs model
#
# LUTO2 is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# LUTO2 is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with
# LUTO2. If not, see <https://www.gnu.org/licenses/>.


import os
import gc
import re
import glob
import json
import shutil
import threading
import numpy as np
import pandas as pd
import xarray as xr
import cf_xarray as cfxr
import rasterio.features
import geopandas as gpd

from shapely.geometry import shape
from joblib import Parallel, delayed
from collections import defaultdict

from luto import settings
from luto import tools
from luto.data import Data

from luto.tools.Manual_jupyter_books.helpers import arr_to_xr
from luto.tools.report.data_tools.parameters import GHG_NAMES
from luto.tools.report.create_report_layers import save_report_layer
from luto.tools.report.create_report_data import save_report_data

import luto.economics.agricultural.quantity as ag_quantity                  # ag production already calculated in solver, imported but skip here                 
import luto.economics.agricultural.revenue as ag_revenue
import luto.economics.agricultural.cost as ag_cost
import luto.economics.agricultural.transitions as ag_transitions
import luto.economics.agricultural.ghg as ag_ghg
import luto.economics.agricultural.water as ag_water
import luto.economics.agricultural.biodiversity as ag_biodiversity

import luto.economics.non_agricultural.quantity as non_ag_quantity          # non-ag production already calculated in solver, imported but skip here
import luto.economics.non_agricultural.revenue as non_ag_revenue
import luto.economics.non_agricultural.cost as non_ag_cost
import luto.economics.non_agricultural.transitions as non_ag_transitions
import luto.economics.non_agricultural.ghg as non_ag_ghg
import luto.economics.non_agricultural.water as non_ag_water
import luto.economics.non_agricultural.biodiversity as non_ag_biodiversity




# ── Per-function peak memory  ────────────
# Used by write_data to compute n_jobs = floor(WRITE_REPORT_MAX_MEM_MB / peak_delta_mb).
# (MB above data-object baseline, measured at RESFACTOR=5)
peak_mb_RES5 = {
    # Profiled 2026-05-30 using output/2026_05_29__16_37_37_RF5_2010-2050/Data_RES5.lz4, yr_cal=2050
    'write_dvar_and_mosaic_map':                    1_452,  #    44s
    'write_transition_nonag2ag':                    6_783,  #    52s
    'write_transition_ag2ag':                       6_506,  #   353s
    'write_biodiversity_quality_scores':            6_300,  #   192s
    'write_economics':                              4_627,  #   352s
    'write_ghg':                                    4_693,  #    77s
    'write_transition_ag2nonag':                    2_895,  #   210s
    'write_quantity':                               2_956,  #   127s
    'write_water':                                  2_575,  #    38s
    'write_dvar_area':                              1_672,  #    32s
    'write_biodiversity_GBF2_scores':               1_929,  #    28s
    'write_area_transition_start_end':              1_493,  #   289s
    'write_renewable_production':                     876,  #    22s
    'write_crosstab':                                  13,  #     1s
    'write_biodiversity_GBF4_ECNES_scores':        10_852,  #  2402s
    'write_biodiversity_GBF3_NVIS_scores':         15_147,  #   431s
    'write_biodiversity_GBF4_SNES_scores':          3_585,  #  4727s
    'write_biodiversity_GBF8_scores_groups':            1,  #     1s
    'write_biodiversity_GBF8_scores_species':           1,  #     1s
}

# Scale RES5 measurements to the actual run resolution.
# Cell count grows as (5 / RESFACTOR)^2, so memory scales proportionally.
WRITE_FUNC_PEAK_MB = {
    k: max(1, round(v * (5 / settings.RESFACTOR) ** 2))
    for k, v in peak_mb_RES5.items()
}


# ── Magnitude parameters ─────────────────────────────────────────────────────────
'''
Dictionary that holds lists of non-zero cell values for each output type/layer, to calculate quantiles 
for setting colorbar limits in the report.
'''

MAX_CELL_MAGNITUDE = {
    'area':                     {'ag': [], 'non_ag': [], 'am': []},
    'bio_quality':              {'ag': [], 'non_ag': [], 'am': [], 'all': []},
    'biodiversity_GBF2':        {'ag': [], 'non_ag': [], 'am': [], 'sum': []},
    'biodiversity_GBF3':        {'ag': [], 'non_ag': [], 'am': [], 'sum': []},
    'biodiversity_GBF4_SNES':   {'ag': [], 'non_ag': [], 'am': [], 'sum': []},
    'biodiversity_GBF4_ECNES':  {'ag': [], 'non_ag': [], 'am': [], 'sum': []},
    'biodiversity_GBF8':        {'ag': [], 'non_ag': [], 'am': []},
    'Economics_ag':             {'ag_revenue': [], 'ag_cost': [], 'ag2ag_cost': [], 'non_ag2ag_cost': [], 'profit_ag': []},
    'Economics_am':             {'am_revenue': [], 'am_cost': [], 'am_profit': []},
    'Economics_non_ag':         {'non_ag_revenue': [], 'non_ag_cost': [], 'nonag2nonag_cost': [], 'ag2nonag_cost': [], 'non_ag_profit': []},
    'Economics_sum':            {'sum_profit': []},
    'ghg_emission':             {'ag': [], 'non_ag': [], 'ag_man': [], 'transition': [], 'sum': []},
    'production':               defaultdict(list),  # commodity names are dynamic
    'water_yield':              {'ag': [], 'non_ag': [], 'am': [], 'sum': []},
    'renewable_energy':         [],
    'renewable_existing_dvar':  [],
    'transition_area':          {'ag2ag': [], 'ag2non_ag': []},
}

# Quantiles to get a robust estimate of the magnitude for setting colorbar limits in the report.
# This elinimates extreme values calculates using vanilla min/max.
MIN_P, MAX_P = 0.005, 0.995


# ── Shared helpers ────────────────────────────────────────────────────────────

def get_year_gap(data: Data, yr_cal: int) -> int:
    """Number of years between yr_cal and the previous simulated year.

    Used to annualise period-aggregate metrics (Production, Economics, GHG,
    Water, Transition area) so report outputs are independent of the
    simulation year-gap. The base year has no previous year and is treated
    as a single annual snapshot (gap = 1).
    """
    simulated_year_list = sorted(data.lumaps.keys())
    yr_idx_sim = simulated_year_list.index(yr_cal)
    if yr_idx_sim == 0:
        return 1
    return yr_cal - simulated_year_list[yr_idx_sim - 1]


def add_all(da, dims):
    """Prepend an ALL-aggregate slice along dim."""
    for dim in dims:
        ds = da.sum(dim=dim, keepdims=True).assign_coords({dim: ['ALL']})
        da = xr.concat([ds, da], dim=dim)
    return da


def get_mag(arr: xr.DataArray) -> list:
    """Return [MIN_P-quantile, MAX_P-quantile] of non-zero values via numpy (avoids MultiIndex quantile bug)."""
    vals = arr.where(arr != 0).compute().values.ravel()
    return [float(np.nanquantile(vals, MIN_P)), float(np.nanquantile(vals, MAX_P))]


def save2chunk(in_xr: xr.DataArray, chunks_dir: str, chunk_idx: int) -> list:
    """Save one species/group batch as a self-contained CF-encoded chunk NC.

    Identical format to save2nc / concat_tmp2nc output (layer × cell, MultiIndex
    CF-compressed) so create_report_layers can read it with
    cfxr.decode_compress_to_multi_index without any pre-processing.
    Returns [min, max] magnitude of the chunk (computed before writing to avoid disk reload).
    """
    os.makedirs(chunks_dir, exist_ok=True)
    n_cells = in_xr.sizes['cell']
    ds = cfxr.encode_multi_index_as_compress(in_xr.to_dataset(name='data'), 'layer')
    chunksizes = [n_cells if d == 'cell' else 1 for d in ds['data'].dims]
    enc = {'data': {'dtype': 'float32', 'zlib': True, 'complevel': 1, 'chunksizes': chunksizes}}
    ds.to_netcdf(os.path.join(chunks_dir, f'chunk_{chunk_idx:06d}.nc'), encoding=enc)
    return get_mag(in_xr)




def concat_tmp2nc(tmp_dir: str, save_path: str):
    """
    Glob all layer_*.nc files in tmp_dir, restore each one's MultiIndex coords
    from its matching *_coords.csv, concat along 'layer', CF-encode, and write
    the final (layer × cell) NetCDF to save_path.  Removes tmp_dir on success.

    Files are loaded eagerly (not lazily) so Windows releases file handles before
    the tmp dir is deleted.
    """
    nc_files = sorted(glob.glob(os.path.join(tmp_dir, 'layer_*.nc')))
    n_cells  = xr.open_dataarray(nc_files[0]).sizes['cell']

    das = []
    for nc_f in nc_files:
        with xr.open_dataarray(nc_f) as raw:
            da = raw.load()                          # eager — releases file handle on context exit
        csv_f = nc_f.replace('.nc', '_coords.csv')
        if os.path.exists(csv_f):
            coord_df = pd.read_csv(csv_f)
            mindex   = pd.MultiIndex.from_frame(coord_df)
            da = da.assign_coords(xr.Coordinates.from_pandas_multiindex(mindex, 'layer'))
        das.append(da)

    da_all      = xr.concat(das, dim='layer')
    mi_names    = [c for c in da_all.coords if c not in ('layer', 'cell') and 'layer' in da_all[c].dims]
    layer_midx  = pd.MultiIndex.from_arrays([da_all[n].values for n in mi_names], names=mi_names)
    midx_coords = xr.Coordinates.from_pandas_multiindex(layer_midx, 'layer')
    da_all      = da_all.drop_vars(mi_names, errors='ignore').assign_coords(midx_coords)
    ds          = cfxr.encode_multi_index_as_compress(da_all.to_dataset(name='data'), 'layer')
    chunksizes  = [n_cells if d == 'cell' else 1 for d in ds['data'].dims]
    enc         = {'data': {'dtype': 'float32', 'zlib': True, 'complevel': 1, 'chunksizes': chunksizes}}
    ds.to_netcdf(save_path, encoding=enc)

    shutil.rmtree(tmp_dir)


def save2nc(in_xr: xr.DataArray, save_path: str):
    """
    CF-encode and write a complete (layer × cell) DataArray to save_path.

    If in_xr is dask-backed it is computed once before writing — no repeated
    graph traversal. 
    """
    in_xr = in_xr.compute()
    n_cells  = in_xr.sizes['cell']
    mi_names = [c for c in in_xr.coords if c not in ('layer', 'cell') and 'layer' in in_xr[c].dims]
    
    if mi_names:
        layer_midx  = pd.MultiIndex.from_arrays(
            [in_xr[n].values for n in mi_names], names=mi_names
        )
        midx_coords = xr.Coordinates.from_pandas_multiindex(layer_midx, 'layer')
        in_xr = in_xr.drop_vars(mi_names, errors='ignore').assign_coords(midx_coords)
        ds         = cfxr.encode_multi_index_as_compress(in_xr.to_dataset(name='data'), 'layer')
        chunksizes = [n_cells if d == 'cell' else 1 for d in ds['data'].dims]
        enc        = {'data': {'dtype': 'float32', 'zlib': True, 'complevel': 1, 'chunksizes': chunksizes}}
        ds.to_netcdf(save_path, encoding=enc)
    else:
        chunksizes = [n_cells if d == 'cell' else 1 for d in in_xr.dims]
        enc        = {'data': {'dtype': 'float32', 'zlib': True, 'complevel': 1, 'chunksizes': chunksizes}}
        in_xr.rename('data').to_netcdf(save_path, encoding=enc)



def chunk_unify_size(da: xr.DataArray, target_mb: float = 10.0) -> xr.DataArray:
    """Re-chunk `da` so each chunk ≈ target_mb; all non-cell dims are kept whole.

    Computes cell_chunk = target_bytes / (product of non-cell dim sizes × itemsize),
    so intermediate chunks from multi-dim broadcasts (e.g. mrj × sr → smrj) stay
    near target_mb regardless of how many extra dimensions the array carries.
    """
    other = max(1, int(np.prod([s for d, s in da.sizes.items() if d != 'cell'])))
    n = max(1, int(target_mb * 1e6 / (other * np.dtype(da.dtype).itemsize)))
    return da.chunk({d: (min(n, s) if d == 'cell' else s) for d, s in da.sizes.items()})


def save_csv(df, rename_map, filepath):
    (df.rename(columns=rename_map)
       .infer_objects(copy=False)
       .replace({'dry': 'Dryland', 'irr': 'Irrigated'})
       .to_csv(filepath, index=False))


def to_region_and_aus_df(da: xr.DataArray, group_dims: list, yr_cal: int, region_levels: list):
    """Aggregate xarray to region-level DataFrame; return (AUS+region combined, AUS only).

    group_dims: non-region dims only (e.g. ['am', 'lm', 'lu']).
    region_levels: coordinate names on da's 'cell' dim (e.g. ['region_state', 'region_NRM']).
    Each level is aggregated separately and concatenated; a 'region_level' column
    records which level each row belongs to.  AUS-only return uses the first level
    (valid_layers extraction is region-independent).
    """
    aus_dims = group_dims + ['Year']
    frames = []
    aus_df = None
    for region in region_levels:
        region_group_dims = [region] + group_dims
        region_df = (
            da.groupby(region).sum(dim='cell')
            .to_dataframe('Value').reset_index()
            .groupby(region_group_dims)[['Value']].sum().reset_index()
            .assign(Year=yr_cal)
            .query('abs(Value) > 1')
            .rename(columns={region: 'region'})
            .assign(region_level=region)
        )
        aus = (
            region_df.drop(columns='region_level')
            .groupby(aus_dims).sum().reset_index()
            .assign(region='AUSTRALIA', region_level=region)
            .query('abs(Value) > 1')
        )
        frames.append(pd.concat([aus, region_df]))
        if aus_df is None:
            aus_df = aus
    return pd.concat(frames, ignore_index=True), aus_df



def bio_to_region_and_aus_df(da: xr.DataArray, group_dims: list, value_name: str, base_score: float, yr_cal: int, region_levels: list = None):
    """
    Aggregate xarray to region-level DataFrame; return (AUS+region combined, AUS only).
    group_dims must NOT include the region coord name (region is added internally).
    region_levels: list of coord names e.g. ['region_state', 'region_NRM'].
    """
    if region_levels is None:
        region_levels = ['region_NRM']
    aus_dims = group_dims + ['Year']
    frames = []
    aus_df = None
    for region in region_levels:
        region_group_dims = [region] + group_dims
        region_df = (
            da.groupby(region)
            .sum(dim='cell')
            .to_dataframe(value_name).reset_index()
            .groupby(region_group_dims)[[value_name]]
            .sum()
            .reset_index()
            .assign(Year=yr_cal)
            .eval(f'Relative_Contribution_Percentage = (`{value_name}` / {base_score}) * 100')
            .query(f'abs(`{value_name}`) > 1')
            .rename(columns={region: 'region'})
            .assign(region_level=region)
        )
        aus = (
            region_df.drop(columns='region_level')
            .groupby(aus_dims).sum().reset_index()
            .assign(region='AUSTRALIA', region_level=region)
            .query(f'abs(`{value_name}`) > 1')
        )
        frames.append(pd.concat([aus, region_df]))
        if aus_df is None:
            aus_df = aus
    return pd.concat(frames, ignore_index=True), aus_df


def process_chunks(trans_xr, data, yr_cal, chunk_size, groupby_cols, value_col, region_levels):
    """
    Process large xarray in chunks and aggregate to a region-level DataFrame.

    Keeps the chunk loop to cap memory (one materialised chunk at a time), but
    replaces the per-chunk to_dataframe + pandas groupby (which created a full
    Cartesian-product DataFrame per chunk — up to 31 M rows at RF5) with a BLAS
    matrix multiply that contracts the cell dimension directly into a small
    (*non_cell_dims, n_regions) accumulator.

    Args:
        trans_xr:      xarray DataArray with a 'cell' dim and region coords on it
        data:          Data object (needs NCELLS)
        yr_cal:        Calendar year (written into the output DataFrame)
        chunk_size:    Number of cells per chunk
        groupby_cols:  Non-region dimension names (used to build the output index)
        value_col:     Name of the value column in the output DataFrame
        region_levels: Cell-coord names to aggregate over (e.g. ['region_state', 'region_NRM'])

    Returns:
        DataFrame with columns ['region_level', 'region'] + groupby_cols + [value_col, 'Year']
    """
    non_cell_dims  = [d for d in trans_xr.dims if d != 'cell']
    dim_coords     = {d: trans_xr.coords[d].values for d in non_cell_dims}
    non_cell_shape = tuple(len(dim_coords[d]) for d in non_cell_dims)
    n_combos       = int(np.prod(non_cell_shape))

    # xarray broadcasts can leave 'cell' in the middle of the dim order.
    # Transpose once so cell is the final axis — required for reshape(n_combos, -1).
    trans_xr = trans_xr.transpose(*non_cell_dims, 'cell')

    level_frames = []
    for region_coord in region_levels:
        labels = trans_xr.coords[region_coord].values           # [NCELLS] str
        unique_regions, codes = np.unique(labels, return_inverse=True)
        n_regions = len(unique_regions)

        # Pre-allocated 2D accumulator — (n_combos, n_regions), stays tiny in memory
        accum = np.zeros((n_combos, n_regions), dtype=np.float64)

        for i in range(0, data.NCELLS, chunk_size):
            sl       = slice(i, min(i + chunk_size, data.NCELLS))
            chunk    = trans_xr.isel(cell=sl).compute().values  # (*non_cell_shape, chunk_cells)
            codes_sl = codes[sl]                                 # (chunk_cells,) int

            # One-hot region indicator: (chunk_cells, n_regions) — tiny
            onehot = np.eye(n_regions, dtype=np.float32)[codes_sl]

            # BLAS GEMM: (n_combos, chunk_cells) @ (chunk_cells, n_regions)
            # Contracts the cell dimension in one vectorised call — no Python loop,
            # no intermediate DataFrame.
            accum += chunk.reshape(n_combos, -1).astype(np.float64) @ onehot

        # Convert the small accumulator to DataFrame once, at the end
        idx = pd.MultiIndex.from_product(
            [dim_coords[d] for d in non_cell_dims], names=non_cell_dims
        )
        level_df = (
            pd.DataFrame(accum, index=idx, columns=unique_regions)
            .reset_index()
            .melt(id_vars=non_cell_dims, var_name='region', value_name=value_col)
            .query(f'abs(`{value_col}`) > 1')
            .assign(Year=yr_cal, region_level=region_coord)
        )
        level_frames.append(level_df)

    return pd.concat(level_frames, ignore_index=True)


# ── Config / Orchestration ────────────────────────────────────────────────────

def write_outputs(data: Data):
    """Write outputs using dynamic timestamp from read_timestamp."""
    timestamp = tools.read_timestamp()
    out_dir = f'{settings.OUTPUT_DIR}/{timestamp}_RF{settings.RESFACTOR}_{settings.SIM_YEARS[0]}-{settings.SIM_YEARS[-1]}'

    @tools.LogToFile(f"{out_dir}/LUTO_RUN_")
    def _write_outputs():
        stop_event = threading.Event()
        memory_thread = threading.Thread(target=tools.log_memory_usage, args=(out_dir, 'a', 1, stop_event))
        memory_thread.start()
        try:
            write_data(data)
            print("Data writing complete, now creating report...\n")
            create_report(data)
        except Exception as e:
            print(f"An error occurred while writing outputs: {e}")
            raise
        finally:
            stop_event.set()
            memory_thread.join()

    return _write_outputs()



def write_data(data: Data):
    years = [yr for yr in settings.SIM_YEARS if yr <= data.last_year]
    paths = [f"{data.path}/out_{yr}" for yr in years]
    write_settings(data.path)

    def get_n_jobs(peak_mb):
        return max(1, settings.WRITE_REPORT_MAX_MEM_MB // max(peak_mb, 1))

    # DVars must be written first as other outputs depend on them
    dvar_jobs = [delayed(write_dvar_and_mosaic_map)(data, yr, path_yr) for yr, path_yr in zip(years, paths)]
    for result in Parallel(n_jobs=get_n_jobs(WRITE_FUNC_PEAK_MB['write_dvar_and_mosaic_map']), return_as="generator")(dvar_jobs):
        print(result)

    # Collect all annotated tasks: (delayed_obj, peak_delta_mb)
    annotated_tasks = [(
        delayed(write_area_transition_start_end)(data, f'{data.path}/out_{years[-1]}', years[-1]),
        WRITE_FUNC_PEAK_MB['write_area_transition_start_end'],
    )]
    for yr, path_yr in zip(years, paths):
        annotated_tasks.extend(write_output_single_year(data, yr, path_yr))

    def process_write_task(result):
        if isinstance(result, tuple):
            msg, mag = result
            for top_key, sub in mag.items():
                target = MAX_CELL_MAGNITUDE[top_key]
                if isinstance(target, list):
                    target.extend(sub)
                else:
                    for sub_key, vals in sub.items():
                        target[sub_key].extend(vals)
            print(msg)
        else:
            print(result)

    # Dynamic scheduling: tasks that fit multiple times within budget run in parallel; heaviest run one at a time.
    groups = {}
    for task, peak_mb in annotated_tasks:
        n = get_n_jobs(peak_mb)
        if n not in groups:
            groups[n] = []
        groups[n].append(task)

    # Windows WaitForMultipleObjects limit: 63 handles total; loky uses ~2 internally, so cap at 61.
    max_workers = min(os.cpu_count(), 61) if os.name == 'nt' else os.cpu_count()
    # Run tiers from most memory-constrained (n_jobs=1) to least (n_jobs=max_workers)
    for n_workers in sorted(groups.keys()):
        tasks = groups[n_workers]
        valid_workers = min(n_workers, len(tasks), max_workers)
        for result in Parallel(n_jobs=valid_workers, return_as='generator_unordered')(tasks):
            process_write_task(result)

    clean = lambda lst: [0.0 if np.isnan(v) else float(v) for v in lst]
    with open(os.path.join(data.path, 'max_cell_magnitudes.json'), 'w') as f:
        json.dump(
            {k: (clean(v) if isinstance(v, list) else {sk: clean(sv) for sk, sv in v.items()})
             for k, v in MAX_CELL_MAGNITUDE.items()},
            f, indent=2
        )
  
  

def write_settings(path):
    pattern = re.compile(r"^(\s*[A-Z].*?)\s*=")
    all_settings = {k: getattr(settings, k) for k in dir(settings) if k.isupper()}
    with open('luto/settings.py') as f:
        order = [m[1].strip() for line in f if (m := pattern.match(line))]
        ordered = {k: all_settings[k] for k in order if k in all_settings}
    with open(os.path.join(path, 'model_run_settings.txt'), 'w') as f:
        f.writelines(f'{k}:{v}\n' for k, v in ordered.items())



def create_report(data: Data):
    out_dir = f"{settings.OUTPUT_DIR}/{tools.read_timestamp()}_RF{settings.RESFACTOR}_{settings.SIM_YEARS[0]}-{settings.SIM_YEARS[-1]}"

    @tools.LogToFile(f"{out_dir}/LUTO_RUN_", mode='a')
    def _create_report():
        print('Creating report...')
        print('├── Copying report template...')
        shutil.copytree('luto/tools/report/VUE_modules', f"{data.path}/DATA_REPORT", dirs_exist_ok=True)
        print('├── Creating chart data...')
        save_report_data(data.path)
        print('├── Creating map data...')
        save_report_layer(data.path)
        print('└── Report created successfully!')

    return _create_report()



def write_output_single_year(data: Data, yr_cal, path_yr):
    """Return list of (delayed_task, peak_delta_mb) for a single year.

    peak_delta_mb values come from WRITE_FUNC_PEAK_MB (derived from peak_mb_RES5
    profiled at RESFACTOR=5, then scaled by (5/RESFACTOR)^2 for the actual run resolution).
    write_data uses them to compute n_jobs = floor(WRITE_MEM_BUDGET_MB / peak_delta_mb),
    so tasks that fit multiple times within the budget run in parallel automatically.
    """
    if not os.path.isdir(path_yr):
        os.mkdir(path_yr)

    P = WRITE_FUNC_PEAK_MB
    return [
        (delayed(write_dvar_area)(data, yr_cal, path_yr),                           P['write_dvar_area']),
        (delayed(write_crosstab)(data, yr_cal, path_yr),                            P['write_crosstab']),
        (delayed(write_ghg)(data, yr_cal, path_yr),                                 P['write_ghg']),
        (delayed(write_water)(data, yr_cal, path_yr),                               P['write_water']),
        (delayed(write_quantity)(data, yr_cal, path_yr),                            P['write_quantity']),
        (delayed(write_economics)(data, yr_cal, path_yr),                           P['write_economics']),
        (delayed(write_transition_nonag2ag)(data, yr_cal, path_yr),                 P['write_transition_nonag2ag']),
        (delayed(write_transition_ag2ag)(data, yr_cal, path_yr),                    P['write_transition_ag2ag']),
        (delayed(write_transition_ag2nonag)(data, yr_cal, path_yr),                 P['write_transition_ag2nonag']),
        (delayed(write_renewable_production)(data, yr_cal, path_yr),                P['write_renewable_production']),
        (delayed(write_biodiversity_quality_scores)(data, yr_cal, path_yr),         P['write_biodiversity_quality_scores']),
        (delayed(write_biodiversity_GBF2_scores)(data, yr_cal, path_yr),            P['write_biodiversity_GBF2_scores']),
        (delayed(write_biodiversity_GBF3_NVIS_scores)(data, yr_cal, path_yr),       P['write_biodiversity_GBF3_NVIS_scores']),
        (delayed(write_biodiversity_GBF4_SNES_scores)(data, yr_cal, path_yr),       P['write_biodiversity_GBF4_SNES_scores']),
        (delayed(write_biodiversity_GBF4_ECNES_scores)(data, yr_cal, path_yr),      P['write_biodiversity_GBF4_ECNES_scores']),
        (delayed(write_biodiversity_GBF8_scores_groups)(data, yr_cal, path_yr),     P['write_biodiversity_GBF8_scores_groups']),
        (delayed(write_biodiversity_GBF8_scores_species)(data, yr_cal, path_yr),    P['write_biodiversity_GBF8_scores_species']),
    ]



# ── DVAR ─────────────────────────────────────────────────────────────────────

def write_dvar_and_mosaic_map(data: Data, yr_cal, path):
    """No annualisation needed: decision variables are a point-in-time snapshot (stock),
    not a flow accumulated over the period since the previous simulated year."""

    ag_map = chunk_unify_size(tools.ag_mrj_to_xr(data, data.ag_dvars[yr_cal]))
    non_ag_map = chunk_unify_size(tools.non_ag_rk_to_xr(data, data.non_ag_dvars[yr_cal]))
    am_map = chunk_unify_size(tools.am_mrj_to_xr(data, data.ag_man_dvars[yr_cal]))
    
    ag_mask = ag_map.sum(['lm','lu']) > 0.001
    am_mask = am_map.sum(['am','lm', 'lu']) > 0.001
    non_ag_mask = non_ag_map.sum('lu') > 0.001

    # ── Inject existing renewable capacity as lu='Existing Capacity' ──────────
    # Same pattern as write_dvar_area: lm='dry' carries the real dvar fraction,
    # lm='irr' is zeros, reindexed to all am types so the lu dimension stays
    # Cartesian. add_all then handles lm='ALL' and lu='ALL' for free.
    # am_mask is computed above from optimised dvars only — intentional, since
    # the argmax mosaic reflects LP allocation, not existing installations.
    if any(settings.RENEWABLES_OPTIONS.values()):
        solar_exist_r = ag_quantity.get_existing_renewable_dvar_fraction(data, 'Utility Solar PV', yr_cal)
        wind_exist_r  = ag_quantity.get_existing_renewable_dvar_fraction(data, 'Onshore Wind',     yr_cal)

        exist_re_dry = xr.DataArray(
            np.stack([solar_exist_r, wind_exist_r], axis=0),
            dims=['am', 'cell'],
            coords={'am': ['Utility Solar PV', 'Onshore Wind'], 'cell': range(data.NCELLS)},
        ).expand_dims(lm=['dry'], lu=['Existing Capacity'])

        exist_re_irr  = xr.zeros_like(exist_re_dry).assign_coords(lm=['irr'])
        exist_re_full = (
            xr.concat([exist_re_dry, exist_re_irr], dim='lm')
            .reindex(am=am_map.am.values, fill_value=0.0)
        )
        am_map = xr.concat([am_map, exist_re_full], dim='lu')

    ag_map = add_all(ag_map, ['lm'])
    am_map = add_all(am_map, ['lm', 'lu'])

    lm_map = data.lmmaps[yr_cal].astype(bool) # has to be boolean for the '~' operator to work
    lu_map = data.lumaps[yr_cal]
    
    ag_map_argmax_ALL = ag_map.sum('lm').argmax(dim='lu').expand_dims(lm=['ALL']).astype(np.float32)
    ag_map_argmax_dry = ag_map_argmax_ALL.where(~lm_map).drop_vars('lm').assign_coords(lm=['dry']).astype(np.float32)
    ag_map_argmax_irr = ag_map_argmax_ALL.where(lm_map).drop_vars('lm').assign_coords(lm=['irr']).astype(np.float32)
    ag_map_argmax = xr.concat([ag_map_argmax_ALL, ag_map_argmax_dry, ag_map_argmax_irr], dim='lm').astype(np.float32)
    ag_map_argmax = ag_map_argmax.expand_dims(lu=['ALL'])
    ag_map_argmax = xr.where(ag_mask.values[None, None, :], ag_map_argmax, np.nan)
    
    am_argmax_ALL = am_map.sum(['lm','lu']).argmax(dim='am').expand_dims(lm=['ALL']).astype(np.float32)
    am_argmax_dry = am_argmax_ALL.where(~lm_map).drop_vars('lm').assign_coords(lm=['dry']).astype(np.float32)
    am_argmax_irr = am_argmax_ALL.where(lm_map).drop_vars('lm').assign_coords(lm=['irr']).astype(np.float32)
    am_argmax_lm = xr.concat([am_argmax_ALL, am_argmax_dry, am_argmax_irr], dim='lm')
    am_argmax_lu = xr.concat([
        am_argmax_lm.where(lu_map == lu_code).expand_dims(lu=[lu_desc])
        for lu_code, lu_desc in data.AGLU2DESC.items()
        if lu_code != -1 # Exclude NoData (cells outside LUTO study area)
    ], dim='lu')
    am_argmax = xr.concat([am_argmax_lu.sum('lu', skipna=True).expand_dims(lu=['ALL']), am_argmax_lu], dim='lu')
    am_argmax = am_argmax.expand_dims(am=['ALL'])
    am_argmax = xr.where(am_mask.values[None,None,None,:], am_argmax, np.nan)
    

    non_ag_map_argmax = non_ag_map.argmax(dim='lu') + settings.NON_AGRICULTURAL_LU_BASE_CODE
    non_ag_map_argmax = xr.where(non_ag_mask, non_ag_map_argmax, np.nan)
    non_ag_map_argmax = non_ag_map_argmax.expand_dims(lu=['ALL']).astype(np.float32)
    
    ag_map_cat = xr.concat([ag_map_argmax, ag_map], dim='lu')
    non_ag_map_cat = xr.concat([non_ag_map_argmax, non_ag_map], dim='lu')
    am_map_cat = xr.concat([am_argmax, am_map], dim='am')
    
    ag_map_stack = ag_map_cat.stack(layer=['lm','lu'])
    non_ag_map_stack = non_ag_map_cat.stack(layer=['lu'])
    am_map_stack = am_map_cat.stack(layer=['am','lm','lu'])
    
    valid_layers_ag = (ag_map_stack.sum('cell') > 0.001).to_dataframe('valid').query('valid == True').index
    valid_layers_non_ag = (non_ag_map_stack.sum('cell') > 0.001).to_dataframe('valid').query('valid == True').index
    valid_layers_am = (am_map_stack.sum('cell') > 0.001).to_dataframe('valid').query('valid == True').index

    save2nc(ag_map_stack.sel(layer=valid_layers_ag), os.path.join(path, f'xr_dvar_ag_{yr_cal}.nc'))

    save2nc(non_ag_map_stack.sel(layer=valid_layers_non_ag), os.path.join(path, f'xr_dvar_non_ag_{yr_cal}.nc'))
    save2nc(am_map_stack.sel(layer=valid_layers_am), os.path.join(path, f'xr_dvar_am_{yr_cal}.nc'))

    lumap_xr_ALL= xr.DataArray(data.lumaps[yr_cal].astype(np.float32), dims=['cell'], coords={'cell': range(data.NCELLS)})
    lumap_xr_dry = xr.DataArray(np.where(~lm_map, lumap_xr_ALL, np.nan).astype(np.float32), dims=['cell'], coords={'cell': range(data.NCELLS)})
    lumap_xr_irr = xr.DataArray(np.where(lm_map, lumap_xr_ALL, np.nan).astype(np.float32), dims=['cell'], coords={'cell': range(data.NCELLS)})
    
    lumap_xr = xr.concat([
        lumap_xr_ALL.expand_dims(lm=['ALL']),
        lumap_xr_dry.expand_dims(lm=['dry']),
        lumap_xr_irr.expand_dims(lm=['irr'])
    ], dim='lm').astype(np.float32)
        
    save2nc(lumap_xr.stack(layer=['lm']), os.path.join(path, f'xr_map_lumap_{yr_cal}.nc'))
        
    
    
    xr.Dataset({
        'layer':arr_to_xr(data, lumap_xr_ALL.astype(np.float32))
    }).to_netcdf(os.path.join(path, f'xr_map_template_{yr_cal}.nc'))

    return f"Mosaic maps written for year {yr_cal}"

def write_dvar_area(data: Data, yr_cal, path):
    """No annualisation needed: land-use area is a point-in-time snapshot (stock),
    not a flow accumulated over the period since the previous simulated year."""

    ag_dvar_mrj = chunk_unify_size(tools.ag_mrj_to_xr(data, data.ag_dvars[yr_cal])
        ).assign_coords({'region_state': ('cell', data.REGION_STATE_NAME), 'region_NRM': ('cell', data.REGION_NRM_NAME)})
    non_ag_rj = chunk_unify_size(tools.non_ag_rk_to_xr(data, data.non_ag_dvars[yr_cal])
        ).assign_coords({'region_state': ('cell', data.REGION_STATE_NAME), 'region_NRM': ('cell', data.REGION_NRM_NAME)})
    am_dvar_mrj = chunk_unify_size(tools.am_mrj_to_xr(data, data.ag_man_dvars[yr_cal])
        ).assign_coords({'region_state': ('cell', data.REGION_STATE_NAME), 'region_NRM': ('cell', data.REGION_NRM_NAME)})


    real_area_r = xr.DataArray(data.REAL_AREA.astype(np.float32), dims=['cell'], coords={'cell': range(data.NCELLS)})

    area_ag = (ag_dvar_mrj * real_area_r)
    area_non_ag = (non_ag_rj * real_area_r)
    area_am = (am_dvar_mrj * real_area_r)

    # ── Existing renewable capacity: inject as lu='Existing Capacity' before add_all ──
    # Design: all existing capacity is attributed to lm='dry' (irr layer is zero) so that
    # downstream groupby sums (over lm) do not double-count existing area.
    # Reindex to all am types so the lu dimension is Cartesian with the rest of area_am.
    # Non-renewable am types get fill_value=0, preserving the Cartesian hierarchy without
    # inflating their totals. add_all then handles lm='ALL' and lu='ALL' for free.
    if any(settings.RENEWABLES_OPTIONS.values()):
        solar_exist_r = ag_quantity.get_existing_renewable_dvar_fraction(data, 'Utility Solar PV', yr_cal)
        wind_exist_r  = ag_quantity.get_existing_renewable_dvar_fraction(data, 'Onshore Wind',     yr_cal)

        exist_re_dry = xr.DataArray(
            np.stack([solar_exist_r * data.REAL_AREA, wind_exist_r * data.REAL_AREA], axis=0),
            dims=['am', 'cell'],
            coords={
                'am': ['Utility Solar PV', 'Onshore Wind'],
                'cell': range(data.NCELLS),
                'region_state': ('cell', data.REGION_STATE_NAME),
                'region_NRM': ('cell', data.REGION_NRM_NAME),
            },
        ).expand_dims(lm=['dry'], lu=['Existing Capacity']
        ).pipe(chunk_unify_size)

        exist_re_irr = xr.zeros_like(exist_re_dry).assign_coords(lm=['irr'])  # zero — avoids double-counting

        exist_re_full = (
            xr.concat([exist_re_dry, exist_re_irr], dim='lm')
            .reindex(am=area_am['am'].values, fill_value=0.0)  # broadcast to all am types
        )

        area_am = xr.concat([area_am, exist_re_full], dim='lu')

    area_ag = add_all(area_ag, ['lm'])
    area_am = add_all(area_am, ['lm', 'lu'])

    REGION_LEVELS = ['region_state', 'region_NRM']
    ag_area_df, ag_area_df_AUS = to_region_and_aus_df(area_ag, ['lm', 'lu'], yr_cal, region_levels=REGION_LEVELS)
    non_ag_area_df, non_ag_area_df_AUS = to_region_and_aus_df(area_non_ag, ['lu'], yr_cal, region_levels=REGION_LEVELS)
    am_area_df, am_area_df_AUS = to_region_and_aus_df(area_am, ['am', 'lm', 'lu'], yr_cal, region_levels=REGION_LEVELS)

    (ag_area_df
        .rename(columns={'lu': 'Land-use', 'lm': 'Water_supply', 'Value': 'Area (ha)'})
        .infer_objects(copy=False)
        .replace({'dry': 'Dryland', 'irr': 'Irrigated'})
        .to_csv(os.path.join(path, f'area_agricultural_landuse_{yr_cal}.csv'), index=False))
    (non_ag_area_df
        .rename(columns={'lu': 'Land-use', 'Value': 'Area (ha)'})
        .to_csv(os.path.join(path, f'area_non_agricultural_landuse_{yr_cal}.csv'), index=False))
    (am_area_df
        .rename(columns={'lu': 'Land-use', 'lm': 'Water_supply', 'am': 'Type', 'Value': 'Area (ha)'})
        .infer_objects(copy=False)
        .replace({'dry': 'Dryland', 'irr': 'Irrigated'})
        .to_csv(os.path.join(path, f'area_agricultural_management_{yr_cal}.csv'), index=False))


    # ==================== Agricultural Area ====================
    # Get valid data layers
    valid_ag_layers = pd.MultiIndex.from_frame(ag_area_df_AUS[['lm', 'lu']]).sort_values()
    area_ag_valid_layers = area_ag.stack(layer=['lm', 'lu']).sel(layer=valid_ag_layers)

    # Get mosaic and filter
    ag_mosaic = cfxr.decode_compress_to_multi_index(
            xr.load_dataset(os.path.join(path, f'xr_dvar_ag_{yr_cal}.nc')), 'layer'
        )['data'].sel(lu='ALL', lm='ALL')
    ag_mosaic_area = ag_mosaic.where(
            area_ag.sum(dim='lu').transpose('cell', ...)
        ).expand_dims(lu=['ALL'])

    # Stack mosaic and filter by valid lm (NOT lu since mosaic has lu='ALL' only)
    ag_mosaic_area_stack = ag_mosaic_area.stack(layer=['lm', 'lu'])
    ag_mosaic_area_stack = ag_mosaic_area_stack.sel(
        layer=ag_mosaic_area_stack['layer']['lm'].isin(valid_ag_layers.get_level_values('lm'))
    )

    # Combine valid layers from data and mosaic
    area_ag_cat = xr.concat([ag_mosaic_area_stack, area_ag_valid_layers], dim='layer').drop_vars(['region_state', 'region_NRM'], errors='ignore')


    # ==================== Non-Agricultural Area ====================
    # Get valid data layers (NonAg: lu dimension only)
    valid_non_ag_layers = pd.MultiIndex.from_frame(non_ag_area_df_AUS[['lu']]).sort_values()

    if non_ag_area_df_AUS['Value'].abs().sum() < 1e-3:
        area_non_ag_cat = xr.DataArray(
            np.zeros((1, data.NCELLS), dtype=np.float32),
            dims=['lu', 'cell'],
            coords={'lu': ['ALL'], 'cell': range(data.NCELLS)}
        ).stack(layer=['lu'])

    else:
        # Stack and select valid data layers
        area_non_ag_valid_layers = area_non_ag.stack(layer=['lu']).sel(layer=valid_non_ag_layers)

        # Get valid mosaic layers
        non_ag_mosaic = cfxr.decode_compress_to_multi_index(
            xr.load_dataset(os.path.join(path, f'xr_dvar_non_ag_{yr_cal}.nc')), 'layer'
        )['data']

        non_ag_mosaic = non_ag_mosaic.sel(lu='ALL').expand_dims('lu').stack(layer=['lu'])

        # Combine valid layers from dvar and mosaic
        area_non_ag_cat = xr.concat([non_ag_mosaic, area_non_ag_valid_layers], dim='layer').drop_vars(['region_state', 'region_NRM'], errors='ignore')


    # ==================== Agricultural Management Area ====================
    valid_am_layers = pd.MultiIndex.from_frame(am_area_df_AUS[['am', 'lm', 'lu']]).sort_values()

    if yr_cal == data.YR_CAL_BASE:
        # Base year: no dvar file exists, so build from existing capacity layers only (no mosaic)
        area_am_cat = area_am.stack(layer=['am', 'lm', 'lu']).sel(layer=valid_am_layers).drop_vars(['region_state', 'region_NRM'], errors='ignore')

    else:
        # Stack and select valid data layers (includes 'Existing Capacity' lu naturally)
        area_am_valid_layers = area_am.stack(layer=['am', 'lm', 'lu']).sel(layer=valid_am_layers)

        # Get mosaic and filter.
        am_mosaic = cfxr.decode_compress_to_multi_index(
            xr.load_dataset(os.path.join(path, f'xr_dvar_am_{yr_cal}.nc')), 'layer'
        )['data'].sel(am='ALL', lm='ALL', lu='ALL')

        # Filter mosaic where data exists, then expand am dimension
        am_mosaic_area = am_mosaic.where(
            area_am.sum('am').transpose('cell', ...)
        ).expand_dims('am')

        # Stack mosaic and filter by lm and lu (NOT am since mosaic has am='ALL' only).
        # Exclude 'Existing Capacity' from lu filter — float layer, not a categorical mosaic entry.
        valid_am_lu_mosaic = valid_am_layers.get_level_values('lu').difference(['Existing Capacity'])
        am_mosaic_area_stack = am_mosaic_area.stack(layer=['am', 'lm', 'lu'])
        am_mosaic_area_stack = am_mosaic_area_stack.sel(
            layer=(
                am_mosaic_area_stack['layer']['lm'].isin(valid_am_layers.get_level_values('lm')) &
                am_mosaic_area_stack['layer']['lu'].isin(valid_am_lu_mosaic)
            )
        )
        area_am_cat = xr.concat([area_am_valid_layers, am_mosaic_area_stack], dim='layer').drop_vars(['region_state', 'region_NRM'], errors='ignore')


    # Save to netcdf with valid layers
    save2nc(area_ag_cat, os.path.join(path, f'xr_area_agricultural_landuse_{yr_cal}.nc'))
    save2nc(area_non_ag_cat, os.path.join(path, f'xr_area_non_agricultural_landuse_{yr_cal}.nc'))
    save2nc(area_am_cat, os.path.join(path, f'xr_area_agricultural_management_{yr_cal}.nc'))
    
    # Save REAL_AREA for calculating val/ha in report layers
    save2nc(real_area_r.expand_dims({'lu': ['ALL']}).stack(layer=['lu']), os.path.join(path, f'xr_area_real_area_ha_{yr_cal}.nc'))
    
    
    # Records cell magnitudes
    area_magnitudes = {
        'area': {
            'ag':     get_mag(area_ag_cat),
            'non_ag': get_mag(area_non_ag_cat),
            'am':     get_mag(area_am_cat),
        }
    }
    
    return (f"Decision variable areas written for year {yr_cal}", area_magnitudes)




# ── Quantity ─────────────────────────────────────────────────────────────────

def write_quantity(data: Data, yr_cal: int, path: str) -> np.ndarray:
    """No annualisation needed: `get_actual_production_lyr()` returns an annual
    production rate (tonnes/KL per year) for `yr_cal`, independent of the gap
    to the previous simulated year — dividing by `gap` would double-annualise it.

    Write commodity production quantities for a specific year.

    Covers: quantity comparison summary CSV, and per-category spatial NetCDF/CSV outputs.
    'yr_cal' is calendar year. Includes impacts of land-use change, productivity
    increases, and climate change on yield.
    """

    # ==================== Total / Comparison Summary ====================

    simulated_year_list = sorted(list(data.lumaps.keys()))
    yr_idx = yr_cal - data.YR_CAL_BASE
    yr_idx_sim = simulated_year_list.index(yr_cal)
    yr_cal_sim_pre = simulated_year_list[yr_idx_sim - 1]

    if yr_cal > data.YR_CAL_BASE:
        assert data.YR_CAL_BASE <= yr_cal_sim_pre < yr_cal, f"yr_cal_sim_pre ({yr_cal_sim_pre}) must be >= {data.YR_CAL_BASE} and < {yr_cal}"
        prod_base = np.array(data.prod_data[yr_cal_sim_pre]['Production'])
        prod_targ = np.array(data.prod_data[yr_cal]['Production'])
        demands = data.D_CY[yr_idx]
        pd.DataFrame({
            'Commodity': [i[0].capitalize() + i[1:] for i in data.COMMODITIES],
            'Prod_base_year (tonnes, KL)': prod_base,
            'Prod_targ_year (tonnes, KL)': prod_targ,
            'Demand (tonnes, KL)': demands,
            'Abs_diff (tonnes, KL)': prod_targ - demands,
            'Prop_diff (%)': (prod_targ / demands) * 100,
            'Year': yr_cal,
        }).to_csv(os.path.join(path, f'quantity_comparison_{yr_cal}.csv'), index=False)

    # ==================== Separate Spatial Outputs ====================

    # Get the commodity quantity dataarrays (sptial layers, (tonnes/KL)/(cell))
    ag_q_mrc, non_ag_p_rc, am_p_amrc = data.get_actual_production_lyr(yr_cal)

    ag_q_mrc  = add_all(ag_q_mrc,  ['lm'])
    am_p_amrc = add_all(am_p_amrc, ['lm', 'Commodity'])

    ag_q_mrc  = ag_q_mrc.assign_coords(region_state=('cell', data.REGION_STATE_NAME), region_NRM=('cell', data.REGION_NRM_NAME))
    non_ag_p_rc = non_ag_p_rc.assign_coords(region_state=('cell', data.REGION_STATE_NAME), region_NRM=('cell', data.REGION_NRM_NAME))
    am_p_amrc = am_p_amrc.assign_coords(region_state=('cell', data.REGION_STATE_NAME), region_NRM=('cell', data.REGION_NRM_NAME))

    # ==================== Region Level Aggregation ====================

    REGION_LEVELS = ['region_state', 'region_NRM']
    ag_q_df, ag_q_df_AUS = to_region_and_aus_df(ag_q_mrc, ['lm', 'Commodity'], yr_cal, region_levels=REGION_LEVELS)
    non_ag_q_df, non_ag_q_df_AUS = to_region_and_aus_df(non_ag_p_rc, ['Commodity'], yr_cal, region_levels=REGION_LEVELS)
    am_q_df, am_q_df_AUS = to_region_and_aus_df(am_p_amrc, ['am', 'lm', 'Commodity'], yr_cal, region_levels=REGION_LEVELS)

    ag_q_df   = ag_q_df.assign(Type='Agricultural').rename(columns={'Value': 'Production (t/KL)'})
    non_ag_q_df = non_ag_q_df.assign(Type='Non_Agricultural').rename(columns={'Value': 'Production (t/KL)'})
    am_q_df   = am_q_df.assign(Type='Agricultural_Management').rename(columns={'Value': 'Production (t/KL)'})

    pd.concat([ag_q_df, non_ag_q_df, am_q_df]
        ).rename(columns={'lm': 'Water_supply'}
        ).infer_objects(copy=False
        ).replace({'dry': 'Dryland', 'irr': 'Irrigated'}
        ).to_csv(os.path.join(path, f'quantity_production_t_separate_{yr_cal}.csv'), index=False)
    
    
    # ==================== Agricultural: Stack, Mosaic, Save ====================
    valid_ag_layers = pd.MultiIndex.from_frame(ag_q_df_AUS[['lm', 'Commodity']]).sort_values()
    ag_q_mrc_stack = ag_q_mrc.stack(layer=['lm','Commodity']).sel(layer=valid_ag_layers)

    ag_mosaic = cfxr.decode_compress_to_multi_index(
        xr.load_dataset(os.path.join(path, f'xr_dvar_ag_{yr_cal}.nc')), 'layer')['data'
        ].sel(lu='ALL', lm='ALL').rename({'lu':'Commodity'})
    ag_mosaic_valid = ag_mosaic.where(ag_q_mrc.sum('Commodity').transpose('cell', ...)).expand_dims('Commodity')
    ag_mosaic_stack = ag_mosaic_valid.stack(layer=['lm','Commodity'])
    ag_mosaic_stack = ag_mosaic_stack.sel(
        layer=ag_mosaic_stack['layer']['lm'].isin(valid_ag_layers.get_level_values('lm'))
    )

    ag_q_mrc_cat_stack = xr.concat([ag_mosaic_stack, ag_q_mrc_stack], dim='layer').drop_vars(['region_state', 'region_NRM'], errors='ignore').compute()
    
    
    # ==================== Non-Agricultural: Stack, Mosaic, Save ====================
    valid_non_ag_layers = pd.MultiIndex.from_frame(non_ag_q_df_AUS[['Commodity']]).sort_values()

    if non_ag_q_df_AUS['Value'].abs().sum() < 1e-3:
        non_ag_p_rc_cat_stack = xr.DataArray(
            np.zeros((1, data.NCELLS), dtype=np.float32),
            dims=['Commodity', 'cell'],
            coords={'Commodity': ['ALL'], 'cell': range(data.NCELLS)}
        ).stack(layer=['Commodity'])
    else:
        non_ag_p_rc_stack = non_ag_p_rc.stack(layer=['Commodity']).sel(layer=valid_non_ag_layers)

        non_ag_mosaic = cfxr.decode_compress_to_multi_index(
            xr.load_dataset(os.path.join(path, f'xr_dvar_non_ag_{yr_cal}.nc')), 'layer')['data'
            ].sel(lu='ALL').rename({'lu':'Commodity'})
        non_ag_mosaic_stack = non_ag_mosaic.expand_dims('Commodity').stack(layer=['Commodity'])

        non_ag_p_rc_cat_stack = xr.concat([non_ag_mosaic_stack, non_ag_p_rc_stack], dim='layer').drop_vars(['region_state', 'region_NRM'], errors='ignore').compute()


    # ==================== Agricultural Management: Stack, Mosaic, Save ====================
    valid_am_layers = pd.MultiIndex.from_frame(am_q_df_AUS[['am', 'lm', 'Commodity']]).sort_values()

    if am_q_df_AUS['Value'].abs().sum() < 1e-3:
        am_p_amrc_cat_stack = xr.DataArray(
            np.zeros((1, 1, 1, data.NCELLS), dtype=np.float32),
            dims=['am', 'lm', 'Commodity', 'cell'],
            coords={'am': ['ALL'], 'lm': ['ALL'], 'Commodity': ['ALL'], 'cell': range(data.NCELLS)}
        ).stack(layer=['am','lm','Commodity'])

    else:
        am_p_amrc_stack = am_p_amrc.stack(layer=['am','lm','Commodity']).sel(layer=valid_am_layers)

        am_mosaic = cfxr.decode_compress_to_multi_index(
            xr.load_dataset(os.path.join(path, f'xr_dvar_am_{yr_cal}.nc')), 'layer')['data'
            ].sel(am='ALL', lm='ALL').sel(lu='ALL').rename({'lu':'Commodity'})
        am_mosaic_valid = am_mosaic.where(am_p_amrc.sum('am').transpose('cell', ...)).expand_dims('am')
        am_mosaic_stack = am_mosaic_valid.stack(layer=['am','lm','Commodity'])
        am_mosaic_stack = am_mosaic_stack.sel(
            layer=(
                am_mosaic_stack['layer']['Commodity'].isin(valid_am_layers.get_level_values('Commodity')) &
                am_mosaic_stack['layer']['lm'].isin(valid_am_layers.get_level_values('lm'))
            )
        )

        am_p_amrc_cat_stack = xr.concat([am_mosaic_stack, am_p_amrc_stack], dim='layer').drop_vars(['region_state', 'region_NRM'], errors='ignore').compute()

    save2nc(ag_q_mrc_cat_stack, os.path.join(path, f'xr_quantities_agricultural_{yr_cal}.nc'))

    save2nc(non_ag_p_rc_cat_stack, os.path.join(path, f'xr_quantities_non_agricultural_{yr_cal}.nc'))
    save2nc(am_p_amrc_cat_stack, os.path.join(path, f'xr_quantities_agricultural_management_{yr_cal}.nc'))


    # ==================== Sum (Ag + Am + NonAg): Stack and Save ====================
    # Sum ag + am (summed over 'am'); non_ag assigned to lm='dry' to avoid double counting
    am_sum_mrc = am_p_amrc.sel(lm=['dry', 'irr'], Commodity=[c for c in am_p_amrc.coords['Commodity'].values if c != 'ALL']).sum('am')
    non_ag_as_dry = non_ag_p_rc.expand_dims('lm').assign_coords(lm=['dry']).reindex(lm=['dry', 'irr'], fill_value=0)

    sum_dry_irr = (ag_q_mrc.sel(lm=['dry', 'irr'])
                   + am_sum_mrc.reindex_like(ag_q_mrc.sel(lm=['dry', 'irr']), fill_value=0)
                   + non_ag_as_dry.reindex_like(ag_q_mrc.sel(lm=['dry', 'irr']), fill_value=0))

    sum_all = sum_dry_irr.sum('lm', keepdims=True).assign_coords(lm=['ALL'])
    sum_mrc = xr.concat([sum_all, sum_dry_irr], dim='lm')

    # Float layers: per-commodity production sums
    sum_mrc_stack = sum_mrc.stack(layer=['lm', 'Commodity'])

    # Mosaic layers: use xr_map_lumap for Commodity='ALL' (categorical land-use map with lm splits)
    lumap_mosaic = cfxr.decode_compress_to_multi_index(
        xr.load_dataset(os.path.join(path, f'xr_map_lumap_{yr_cal}.nc')), 'layer')['data'].unstack('layer')
    sum_mosaic = lumap_mosaic.expand_dims('Commodity').assign_coords(Commodity=['ALL'])
    sum_mosaic_stack = sum_mosaic.stack(layer=['lm', 'Commodity'])

    sum_cat_stack = xr.concat([sum_mosaic_stack, sum_mrc_stack], dim='layer').drop_vars(['region_state', 'region_NRM'], errors='ignore').compute()
    save2nc(sum_cat_stack, os.path.join(path, f'xr_quantities_sum_{yr_cal}.nc'))


    # Record max cell value for report generation later (e.g., for setting colorbar limits)
    prod_magnitudes = {
        'ag':     {cm: get_mag(ag_q_mrc.sel(Commodity=cm)) for cm in data.COMMODITIES},
        'non_ag': {cm: get_mag(non_ag_p_rc.sel(Commodity=cm)) for cm in data.COMMODITIES},
        'am':     {cm: get_mag(am_p_amrc.sel(Commodity=cm)) for cm in data.COMMODITIES},
        'sum':    {cm: get_mag(sum_mrc.sel(Commodity=cm)) for cm in data.COMMODITIES},
    }
    
    commodity_magnitudes = {'production': {}}
    for cm in data.COMMODITIES:
        vals = [*prod_magnitudes['ag'][cm], *prod_magnitudes['non_ag'][cm], *prod_magnitudes['am'][cm], *prod_magnitudes['sum'][cm]]
        commodity_magnitudes['production'][cm] = [i for i in vals if not np.isnan(i)]  # Filter out None values (in case some categories don't produce certain commodities)
    
    return (
        f"Separate quantity production written for year {yr_cal}", 
        commodity_magnitudes
    )



# ── Economics ────────────────────────────────────────────────────────────────

def write_economics(data: Data, yr_cal, path):
    """Mixed annualisation: revenue/cost matrices (`get_rev_matrices`, `get_cost_matrices`,
    existing-capacity OPEX/revenue) are already annual rates and are NOT divided by `gap`.
    Transition-related terms (`ag2ag_mrj` establishment cost, existing-capacity CAPEX delta,
    `nonag2nonag_mat`, `ag2nonag_mat`) represent a one-off cost incurred over the period since
    the previous simulated year and ARE divided by `gap` to annualise."""
    yr_idx = yr_cal - data.YR_CAL_BASE
    gap = get_year_gap(data, yr_cal)  # annualise: divide period value matrices by this

    if yr_idx == 0:
        yr_cal_sim_pre = None; yr_idx_pre = None
    else:
        sy = sorted(data.lumaps.keys())
        yr_cal_sim_pre = sy[sy.index(yr_cal) - 1]
        yr_idx_pre = yr_cal_sim_pre - data.YR_CAL_BASE


    # ==================== Agricultural Economics ====================

    ag_dvar_mrj = (
        chunk_unify_size(tools.ag_mrj_to_xr(data, data.ag_dvars[yr_cal]))
        .assign_coords(region_state=('cell', data.REGION_STATE_NAME),
                       region_NRM=('cell', data.REGION_NRM_NAME))
    )

    # Delta dvar: only net increases in allocation pay transition costs.
    # When BLENDED_AG_TRANSITION_COSTS=True, use the solver's D vars directly — they are
    # exactly max(0, X_new - x_old) at optimality and match what entered the objective.
    # Otherwise fall back to clipping the dvar difference.
    if settings.BLENDED_AG_TRANSITION_COSTS and data.ag_delta_dvars.get(yr_cal) is not None:
        ag_dvar_mrj_delta = chunk_unify_size(
            tools.ag_mrj_to_xr(data, data.ag_delta_dvars[yr_cal])
        ).assign_coords(region_state=('cell', data.REGION_STATE_NAME),
                        region_NRM=('cell', data.REGION_NRM_NAME))
    elif yr_cal_sim_pre is not None:
        ag_dvar_mrj_delta = chunk_unify_size(
            tools.ag_mrj_to_xr(
                data,
                np.clip(data.ag_dvars[yr_cal] - data.ag_dvars[yr_cal_sim_pre], 0, None)
            )
        ).assign_coords(region_state=('cell', data.REGION_STATE_NAME),
                        region_NRM=('cell', data.REGION_NRM_NAME))
    else:
        ag_dvar_mrj_delta = ag_dvar_mrj  # base year: ag2ag_mrj is zeros, so value is immaterial

    ag_rev_df = ag_revenue.get_rev_matrices(data, yr_idx, aggregate=False)
    ag_cost_df = ag_cost.get_cost_matrices(data, yr_idx, aggregate=False)
    ag_rev_df.columns.names = ag_cost_df.columns.names = ['lu', 'lm', 'source']

    if yr_cal_sim_pre is not None:
        ag2ag_mrj   = ag_transitions.get_transition_matrices_ag2ag_from_base_year(
            data, yr_idx, yr_cal_sim_pre, separate=True)
        nonag2ag_mrj = non_ag_transitions.get_transition_matrix_nonag2ag(
            data, yr_idx, data.lumaps[yr_cal_sim_pre], data.lmmaps[yr_cal_sim_pre], separate=True, base_year=yr_cal_sim_pre)
    else:
        ag2ag_mrj    = {'Establishment cost': np.zeros((data.NLMS, data.NCELLS, data.N_AG_LUS), dtype=np.float32)}
        nonag2ag_mrj = {}
    del nonag2ag_mrj  # intentionally unused in economics — see below
    ag2ag_mrj = {k: v / gap for k, v in ag2ag_mrj.items()}

    # ── profit_ag: sum over source from DataFrames (never builds 4D array) ──
    rev_sum_df  = ag_rev_df.T.groupby(level=['lu', 'lm']).sum().T.astype(np.float32)
    cost_sum_df = ag_cost_df.T.groupby(level=['lu', 'lm']).sum().T.astype(np.float32)

    rev_sum_xr = chunk_unify_size(xr.DataArray(
        rev_sum_df.values, dims=['cell', 'layer'],
        coords={'cell': range(data.NCELLS), 'layer': rev_sum_df.columns}
    ).unstack('layer'))   # (cell, lu, lm) — 3D, no source

    cost_sum_xr = chunk_unify_size(xr.DataArray(
        cost_sum_df.values, dims=['cell', 'layer'],
        coords={'cell': range(data.NCELLS), 'layer': cost_sum_df.columns}
    ).unstack('layer'))

    ag2ag_sum_arr = np.sum(np.stack(list(ag2ag_mrj.values())), axis=0)  # (NLMS, NCELLS, N_AG_LUS)
    ag2ag_sum_xr  = chunk_unify_size(xr.DataArray(
        ag2ag_sum_arr,
        dims=['lm', 'cell', 'lu'],
        coords={'lm': data.LANDMANS, 'cell': range(data.NCELLS), 'lu': data.AGRICULTURAL_LANDUSES}
    ))
    del ag2ag_sum_arr

    # NonAg→Ag transition costs are intentionally excluded from profit_ag here.
    # The original write_economics() always used np.zeros() for nonag2ag_smrj
    # (the actual values from nonag2ag_mrj were never filled in). These costs are
    # handled separately by write_transition_nonag2ag(). Matching that behaviour.
    # Transition cost is charged on the delta dvar (net increases only) so heterogeneous
    # base-year cells at RESFACTOR>1 do not accumulate phantom cross-LU transition costs.
    xr_profit_ag = ag_dvar_mrj * (rev_sum_xr - cost_sum_xr) - ag_dvar_mrj_delta * ag2ag_sum_xr
    del rev_sum_xr, cost_sum_xr, ag2ag_sum_xr

    # raw_profit_ag for Sum Profit (computed now so ag_dvar_mrj can be freed)
    raw_profit_ag = xr_profit_ag.drop_vars(['region_state', 'region_NRM'], errors='ignore')

    # ── per-source products via loop (never builds full 4D unstack) ──────────
    parts = []
    for src in ag_rev_df.columns.get_level_values('source').unique():
        src_df = ag_rev_df.xs(src, level='source', axis=1)
        src_df.columns.names = ['lu', 'lm']
        src_xr = xr.DataArray(src_df.values.astype(np.float32), dims=['cell', 'layer'],
            coords={'cell': range(data.NCELLS), 'layer': src_df.columns}).unstack('layer')
        parts.append((ag_dvar_mrj * src_xr).expand_dims({'source': [src]}).compute())
        del src_xr
        gc.collect()
    xr_ag_rev = xr.concat(parts, dim='source')
    del parts

    parts = []
    for src in ag_cost_df.columns.get_level_values('source').unique():
        src_df = ag_cost_df.xs(src, level='source', axis=1)
        src_df.columns.names = ['lu', 'lm']
        src_xr = xr.DataArray(src_df.values.astype(np.float32), dims=['cell', 'layer'],
            coords={'cell': range(data.NCELLS), 'layer': src_df.columns}).unstack('layer')
        parts.append((ag_dvar_mrj * src_xr).expand_dims({'source': [src]}).compute())
        del src_xr
        gc.collect()
    xr_ag_cost = xr.concat(parts, dim='source')
    del parts

    # ag2ag: loop over transition types (small, but keeps same pattern)
    ag2ag_parts = []
    for src_name, src_arr in ag2ag_mrj.items():
        src_xr = chunk_unify_size(xr.DataArray(
            src_arr, dims=['lm', 'cell', 'lu'],
            coords={'lm': data.LANDMANS, 'cell': range(data.NCELLS),
                    'lu': data.AGRICULTURAL_LANDUSES}
        ))
        ag2ag_parts.append((ag_dvar_mrj_delta * src_xr).expand_dims({'source': [src_name]}).compute())
        del src_xr
    xr_ag2ag_cost = xr.concat(ag2ag_parts, dim='source')
    del ag2ag_parts, ag2ag_mrj

    # ── nonag2ag: output zeros, matching original write_economics() design ────
    # nonag2ag_mrj contains real cost arrays but they represent potential costs
    # for cells CURRENTLY under NonAg land use. write_economics() has always
    # excluded these from the Ag profit calculation (nonag2ag_smrj = np.zeros()
    # in the original). The actual transition outputs live in write_transition_nonag2ag().
    # Because we set valid_nonag2ag_cost_layers directly (never via to_region_and_aus_df),
    # the original `if len(valid_nonag2ag_cost_layers) == 0` sentinel guard is not needed
    # here. If this zero-bypass is ever removed, restore that guard.
    nonag2ag_lu = ['ALL'] + list(data.AGRICULTURAL_LANDUSES)
    valid_nonag2ag_cost_layers = pd.MultiIndex.from_tuples(
        [('ALL', 'ALL', 'ALL')], names=['lm', 'source', 'from_lu'])
    nonag2ag_cost_jms = pd.DataFrame({
        'from_lu': ['ALL'], 'lm': ['ALL'], 'source': ['ALL'],
        'Year': [yr_cal], 'region': ['AUSTRALIA'], 'Value': [0.0],
    })
    nonag2ag_cost_jms_AUS = nonag2ag_cost_jms.copy()
    nonag2ag_cost_valid_layers = xr.DataArray(
        np.zeros((data.NCELLS, len(nonag2ag_lu), 1), dtype=np.float32),
        dims=['cell', 'lu', 'layer'],
        coords={'cell': range(data.NCELLS), 'lu': nonag2ag_lu,
                'layer': valid_nonag2ag_cost_layers},
    )

    del ag_dvar_mrj, ag_dvar_mrj_delta
    gc.collect()

    # ── add_all, aggregate, save ──────────────────────────────────────────────
    xr_ag_rev        = add_all(xr_ag_rev.assign_coords(
        region_state=('cell', data.REGION_STATE_NAME), region_NRM=('cell', data.REGION_NRM_NAME)),
        ['lu', 'lm', 'source'])
    xr_ag_cost       = add_all(xr_ag_cost.assign_coords(
        region_state=('cell', data.REGION_STATE_NAME), region_NRM=('cell', data.REGION_NRM_NAME)),
        ['lu', 'lm', 'source'])
    xr_ag2ag_cost    = add_all(xr_ag2ag_cost.assign_coords(
        region_state=('cell', data.REGION_STATE_NAME), region_NRM=('cell', data.REGION_NRM_NAME)),
        ['lu', 'lm', 'source'])
    xr_profit_ag     = add_all(xr_profit_ag.assign_coords(
        region_state=('cell', data.REGION_STATE_NAME), region_NRM=('cell', data.REGION_NRM_NAME)),
        ['lm', 'lu'])

    ag_rev_jms,     ag_rev_jms_AUS     = to_region_and_aus_df(xr_ag_rev,     ['lu', 'lm', 'source'], yr_cal, region_levels=['region_state', 'region_NRM'])
    ag_cost_jms,    ag_cost_jms_AUS    = to_region_and_aus_df(xr_ag_cost,    ['lu', 'lm', 'source'], yr_cal, region_levels=['region_state', 'region_NRM'])
    ag2ag_cost_jms, ag2ag_cost_jms_AUS = to_region_and_aus_df(xr_ag2ag_cost, ['lu', 'lm', 'source'], yr_cal, region_levels=['region_state', 'region_NRM'])
    profit_ag_jms,  profit_ag_jms_AUS  = to_region_and_aus_df(xr_profit_ag,  ['lu', 'lm'],           yr_cal, region_levels=['region_state', 'region_NRM'])

    save_csv(ag_rev_jms,        {'lu': 'Land-use', 'lm': 'Water_supply', 'source': 'Type', 'Value': 'Value ($)'}, os.path.join(path, f'economics_ag_revenue_{yr_cal}.csv'))
    save_csv(ag_cost_jms,       {'lu': 'Land-use', 'lm': 'Water_supply', 'source': 'Type', 'Value': 'Value ($)'}, os.path.join(path, f'economics_ag_cost_{yr_cal}.csv'))
    save_csv(ag2ag_cost_jms,    {'lu': 'To_Land-use', 'lm': 'Water_supply', 'source': 'Type', 'Value': 'Value ($)'}, os.path.join(path, f'economics_ag_transition_Ag2Ag_{yr_cal}.csv'))
    save_csv(nonag2ag_cost_jms, {'from_lu': 'From_Land-use', 'lm': 'Water_supply', 'source': 'Type', 'Value': 'Value ($)'}, os.path.join(path, f'economics_ag_transition_NonAg2Ag_{yr_cal}.csv'))
    save_csv(profit_ag_jms,     {'lu': 'Land-use', 'lm': 'Water_supply', 'source': 'Type', 'Value': 'Value ($)'}, os.path.join(path, f'economics_ag_profit_{yr_cal}.csv'))

    valid_rev_layers        = pd.MultiIndex.from_frame(ag_rev_jms_AUS[['lm', 'source', 'lu']]).sort_values()
    valid_cost_layers       = pd.MultiIndex.from_frame(ag_cost_jms_AUS[['lm', 'source', 'lu']]).sort_values()
    valid_ag2ag_cost_layers = pd.MultiIndex.from_frame(ag2ag_cost_jms_AUS[['lm', 'source', 'lu']]).sort_values()
    valid_profit_ag_layers  = pd.MultiIndex.from_frame(profit_ag_jms_AUS[['lm', 'lu']]).sort_values()
    ag_rev_valid_layers     = xr_ag_rev.stack(layer=['lm', 'source', 'lu']).sel(layer=valid_rev_layers).drop_vars(['region_state', 'region_NRM'], errors='ignore')
    ag_cost_valid_layers    = xr_ag_cost.stack(layer=['lm', 'source', 'lu']).sel(layer=valid_cost_layers).drop_vars(['region_state', 'region_NRM'], errors='ignore')
    ag2ag_cost_valid_layers = xr_ag2ag_cost.stack(layer=['lm', 'source', 'lu']).sel(layer=valid_ag2ag_cost_layers).drop_vars(['region_state', 'region_NRM'], errors='ignore')
    profit_ag_valid_layers  = xr_profit_ag.stack(layer=['lm', 'lu']).sel(layer=valid_profit_ag_layers).drop_vars(['region_state', 'region_NRM'], errors='ignore')

    save2nc(ag_rev_valid_layers, os.path.join(path, f'xr_economics_ag_revenue_{yr_cal}.nc'))

    save2nc(ag_cost_valid_layers, os.path.join(path, f'xr_economics_ag_cost_{yr_cal}.nc'))
    save2nc(ag2ag_cost_valid_layers, os.path.join(path, f'xr_economics_ag_transition_Ag2Ag_{yr_cal}.nc'))
    save2nc(nonag2ag_cost_valid_layers, os.path.join(path, f'xr_economics_ag_transition_NonAg2Ag_{yr_cal}.nc'))
    save2nc(profit_ag_valid_layers, os.path.join(path, f'xr_economics_ag_profit_{yr_cal}.nc'))

    del xr_ag_rev, xr_ag_cost, xr_ag2ag_cost, xr_profit_ag
    gc.collect()


    # ==================== Agricultural Management Economics ====================

    am_dvar_mrj = (
        chunk_unify_size(tools.am_mrj_to_xr(data, data.ag_man_dvars[yr_cal]))
        .assign_coords(region_state=('cell', data.REGION_STATE_NAME),region_NRM=('cell', data.REGION_NRM_NAME))
    )

    ag_rev_mrj  = ag_revenue.get_rev_matrices(data, yr_idx)
    ag_cost_mrj = ag_cost.get_cost_matrices(data, yr_idx)

    am_revenue_mat = chunk_unify_size(tools.am_mrj_to_xr(data, ag_revenue.get_agricultural_management_revenue_matrices(data, ag_rev_mrj, yr_idx)))
    am_cost_mat    = chunk_unify_size(tools.am_mrj_to_xr(data, ag_cost.get_agricultural_management_cost_matrices(data, ag_cost_mrj, yr_cal)))

    renewable_ams = [am for am, enabled in settings.RENEWABLES_OPTIONS.items() if enabled]
    if renewable_ams:
        re_mask = am_revenue_mat.am.isin(renewable_ams)
        am_revenue_mat = am_revenue_mat.where(~re_mask, other=0.0)
        am_cost_mat    = am_cost_mat.where(~re_mask, other=0.0)

    raw_profit_am_pre = (am_dvar_mrj * (am_revenue_mat - am_cost_mat))

    xr_revenue_am = am_dvar_mrj * am_revenue_mat
    xr_cost_am    = (am_dvar_mrj * am_cost_mat).expand_dims(Cost_type=['Operating Cost'])

    if renewable_ams:
        solar_cost_opt = ag_cost.get_utility_solar_pv_effect_c_mrj(data, ag_cost_mrj, yr_idx, aggregate=False)
        wind_cost_opt  = ag_cost.get_onshore_wind_effect_c_mrj(data, ag_cost_mrj, yr_idx, aggregate=False)

        solar_lu = settings.AG_MANAGEMENTS_TO_LAND_USES['Utility Solar PV']
        wind_lu  = settings.AG_MANAGEMENTS_TO_LAND_USES['Onshore Wind']

        solar_opex_xr  = xr.DataArray(solar_cost_opt['opex'],  dims=['lm', 'cell', 'lu'], coords={'lu': solar_lu})
        solar_capex_xr = xr.DataArray(solar_cost_opt['capex'], dims=['lm', 'cell', 'lu'], coords={'lu': solar_lu})
        wind_opex_xr   = xr.DataArray(wind_cost_opt['opex'],   dims=['lm', 'cell', 'lu'], coords={'lu': wind_lu})
        wind_capex_xr  = xr.DataArray(wind_cost_opt['capex'],  dims=['lm', 'cell', 'lu'], coords={'lu': wind_lu})

        solar_dvar_now = am_dvar_mrj.sel(am='Utility Solar PV')
        wind_dvar_now  = am_dvar_mrj.sel(am='Onshore Wind')

        if yr_cal_sim_pre is None:
            solar_dvar_delta = solar_dvar_now; wind_dvar_delta = wind_dvar_now
        else:
            am_dvar_xr_pre   = tools.am_mrj_to_xr(data, data.ag_man_dvars[yr_cal_sim_pre])
            solar_dvar_delta = solar_dvar_now - am_dvar_xr_pre.sel(am='Utility Solar PV')
            wind_dvar_delta  = wind_dvar_now  - am_dvar_xr_pre.sel(am='Onshore Wind')

        solar_potential = xr.concat([
            (solar_dvar_now * solar_opex_xr).reindex(lu=xr_cost_am.lu.values, fill_value=0.0).expand_dims(am=['Utility Solar PV']).expand_dims(Cost_type=['Operating Cost']),
            (solar_dvar_delta * solar_capex_xr / gap).reindex(lu=xr_cost_am.lu.values, fill_value=0.0).expand_dims(am=['Utility Solar PV']).expand_dims(Cost_type=['Capital expenditure']),
        ], dim='Cost_type')
        wind_potential = xr.concat([
            (wind_dvar_now * wind_opex_xr).reindex(lu=xr_cost_am.lu.values, fill_value=0.0).expand_dims(am=['Onshore Wind']).expand_dims(Cost_type=['Operating Cost']),
            (wind_dvar_delta * wind_capex_xr / gap).reindex(lu=xr_cost_am.lu.values, fill_value=0.0).expand_dims(am=['Onshore Wind']).expand_dims(Cost_type=['Capital expenditure']),
        ], dim='Cost_type')

        re_reindexed = xr.concat([solar_potential, wind_potential], dim='am').reindex(am=xr_cost_am.am.values, lu=xr_cost_am.lu.values, fill_value=0.0)
        xr_cost_am = xr.concat([
            (xr_cost_am.sel(Cost_type='Operating Cost') + re_reindexed.sel(Cost_type='Operating Cost')).expand_dims(Cost_type=['Operating Cost']),
            re_reindexed.sel(Cost_type='Capital expenditure').expand_dims(Cost_type=['Capital expenditure']),
        ], dim='Cost_type')

        solar_rev_opt = ag_revenue.get_utility_solar_pv_effect_r_mrj(data, ag_rev_mrj, yr_idx)
        wind_rev_opt  = ag_revenue.get_onshore_wind_effect_r_mrj(data, ag_rev_mrj, yr_idx)
        re_rev = xr.concat([
            (solar_dvar_now * xr.DataArray(solar_rev_opt, dims=['lm', 'cell', 'lu'], coords={'lu': solar_lu})).reindex(lu=xr_revenue_am.lu.values, fill_value=0.0).expand_dims(am=['Utility Solar PV']),
            (wind_dvar_now  * xr.DataArray(wind_rev_opt,  dims=['lm', 'cell', 'lu'], coords={'lu': wind_lu})).reindex(lu=xr_revenue_am.lu.values, fill_value=0.0).expand_dims(am=['Onshore Wind']),
        ], dim='am')
        xr_revenue_am = xr_revenue_am + re_rev.reindex_like(xr_revenue_am, fill_value=0.0)

        solar_cells_now = ag_cost.get_utility_solar_pv_existing_cost_by_region(data, yr_idx, return_cells=True)
        wind_cells_now  = ag_cost.get_onshore_wind_existing_cost_by_region(data, yr_idx, return_cells=True)
        solar_capex_pre = (
            0.0
            if yr_cal_sim_pre is None
            else ag_cost.get_utility_solar_pv_existing_cost_by_region(data, yr_idx_pre, return_cells=True)['capex_r'].values
        )
        wind_capex_pre = (
            0.0
            if yr_cal_sim_pre is None
            else ag_cost.get_onshore_wind_existing_cost_by_region(data, yr_idx_pre, return_cells=True)['capex_r'].values
        )

        # Split existing capacity cost into OPEX (all cumulative capacity) and CAPEX delta
        # (new installations this period). Previously these were lumped into 'Operating Cost',
        # causing apparent OPEX to decrease in years with few new installations.
        solar_exist_opex  = solar_cells_now['opex_r'].values
        wind_exist_opex   = wind_cells_now['opex_r'].values
        solar_exist_capex = (solar_cells_now['capex_r'].values - solar_capex_pre) / gap
        wind_exist_capex  = (wind_cells_now['capex_r'].values  - wind_capex_pre) / gap

        def _make_exist_cost_da(solar_vals, wind_vals):
            return xr.DataArray(
                np.stack([solar_vals, wind_vals], axis=0),
                dims=['am', 'cell'],
                coords={'am': ['Utility Solar PV', 'Onshore Wind'], 'cell': np.arange(data.NCELLS),
                        'region_state': ('cell', data.REGION_STATE_NAME),
                        'region_NRM': ('cell', data.REGION_NRM_NAME)},
            ).expand_dims(lm=['dry'], lu=['Existing Capacity'])

        exist_opex_da  = _make_exist_cost_da(solar_exist_opex,  wind_exist_opex)
        exist_capex_da = _make_exist_cost_da(solar_exist_capex, wind_exist_capex)

        def _expand_exist(da, cost_type_label):
            return (
                xr.concat([da, xr.zeros_like(da).assign_coords(lm=['irr'])], dim='lm')
                .reindex(am=xr_cost_am.am.values, fill_value=0.0)
                .expand_dims(Cost_type=[cost_type_label])
            )

        exist_cost_full = xr.concat([
            _expand_exist(exist_opex_da,  'Operating Cost'),
            _expand_exist(exist_capex_da, 'Capital expenditure'),
        ], dim='Cost_type')
        xr_cost_am = xr.concat([xr_cost_am, exist_cost_full], dim='lu')

        solar_prices = {data.REGION_STATE_NAME2CODE[k]: v for k, v in data.SOLAR_PRICES.query('Year==@yr_cal').set_index('State')['Price_AUD_per_MWh'].items()}
        wind_prices  = {data.REGION_STATE_NAME2CODE[k]: v for k, v in data.WIND_PRICES.query('Year==@yr_cal').set_index('State')['Price_AUD_per_MWh'].items()}
        solar_exist_rev = ag_quantity.get_exist_renewable_capacity(data, 'Utility Solar PV', yr_cal).values * np.vectorize(solar_prices.get, otypes=[np.float32])(data.REGION_STATE_CODE)
        wind_exist_rev  = ag_quantity.get_exist_renewable_capacity(data, 'Onshore Wind',     yr_cal).values * np.vectorize(wind_prices.get,  otypes=[np.float32])(data.REGION_STATE_CODE)
        exist_rev_da  = xr.DataArray(
            np.stack([solar_exist_rev, wind_exist_rev], axis=0),
            dims=['am', 'cell'],
            coords={'am': ['Utility Solar PV', 'Onshore Wind'], 'cell': np.arange(data.NCELLS),
                    'region_state': ('cell', data.REGION_STATE_NAME),
                    'region_NRM': ('cell', data.REGION_NRM_NAME)},
        ).expand_dims(lm=['dry'], lu=['Existing Capacity'])
        exist_rev_full = xr.concat([exist_rev_da, xr.zeros_like(exist_rev_da).assign_coords(lm=['irr'])], dim='lm').reindex(am=xr_revenue_am.am.values, fill_value=0.0)
        xr_revenue_am  = xr.concat([xr_revenue_am, exist_rev_full], dim='lu')

    del am_dvar_mrj, am_revenue_mat, am_cost_mat
    gc.collect()

    xr_profit_am  = xr_revenue_am - xr_cost_am.sum('Cost_type')
    xr_revenue_am = add_all(xr_revenue_am, ['lm', 'lu', 'am'])
    xr_cost_am    = add_all(xr_cost_am,    ['lm', 'lu', 'am', 'Cost_type'])
    xr_profit_am  = add_all(xr_profit_am,  ['lm', 'lu', 'am'])

    revenue_am_df, revenue_am_df_AUS = to_region_and_aus_df(xr_revenue_am, ['am', 'lm', 'lu'],            yr_cal, region_levels=['region_state', 'region_NRM'])
    cost_am_df,    cost_am_df_AUS    = to_region_and_aus_df(xr_cost_am,    ['am', 'lm', 'lu', 'Cost_type'], yr_cal, region_levels=['region_state', 'region_NRM'])
    profit_am_df,  profit_am_df_AUS  = to_region_and_aus_df(xr_profit_am,  ['am', 'lm', 'lu'],            yr_cal, region_levels=['region_state', 'region_NRM'])

    rename_am = {'lu': 'Land-use', 'lm': 'Water_supply', 'am': 'Management Type', 'Value': 'Value ($)'}
    save_csv(revenue_am_df, rename_am, os.path.join(path, f'economics_am_revenue_{yr_cal}.csv'))
    save_csv(cost_am_df,    rename_am, os.path.join(path, f'economics_am_cost_{yr_cal}.csv'))
    save_csv(profit_am_df,  rename_am, os.path.join(path, f'economics_am_profit_{yr_cal}.csv'))

    vl_rev_am = (
        pd.MultiIndex.from_tuples([('ALL','ALL','ALL')], names=['am','lm','lu'])
        if revenue_am_df_AUS.empty
        else pd.MultiIndex.from_frame(revenue_am_df_AUS[['am','lm','lu']]).sort_values()
    )
    vl_cost_am = (
        pd.MultiIndex.from_tuples([('ALL','ALL','ALL','Operating Cost')], names=['am','lm','lu','Cost_type'])
        if cost_am_df_AUS.empty
        else pd.MultiIndex.from_frame(cost_am_df_AUS[['am','lm','lu','Cost_type']]).sort_values()
    )
    vl_profit_am = (
        pd.MultiIndex.from_tuples([('ALL','ALL','ALL')], names=['am','lm','lu'])
        if profit_am_df_AUS.empty
        else pd.MultiIndex.from_frame(profit_am_df_AUS[['am','lm','lu']]).sort_values()
    )

    valid_layers_stack_rev_am    = xr_revenue_am.stack(layer=['am','lm','lu']).sel(layer=vl_rev_am).drop_vars(['region_state','region_NRM'], errors='ignore')
    valid_layers_stack_cost_am   = xr_cost_am.stack(layer=['am','lm','lu','Cost_type']).sel(layer=vl_cost_am).drop_vars(['region_state','region_NRM'], errors='ignore')
    valid_layers_stack_profit_am = xr_profit_am.stack(layer=['am','lm','lu']).sel(layer=vl_profit_am).drop_vars(['region_state','region_NRM'], errors='ignore')

    save2nc(valid_layers_stack_rev_am, os.path.join(path, f'xr_economics_am_revenue_{yr_cal}.nc'))

    save2nc(valid_layers_stack_cost_am, os.path.join(path, f'xr_economics_am_cost_{yr_cal}.nc'))
    save2nc(valid_layers_stack_profit_am, os.path.join(path, f'xr_economics_am_profit_{yr_cal}.nc'))

    del xr_revenue_am, xr_cost_am, xr_profit_am
    gc.collect()


    # ==================== Non-Agricultural Economics ====================

    non_ag_dvar = (
        chunk_unify_size(tools.non_ag_rk_to_xr(data, data.non_ag_dvars[yr_cal]))
        .assign_coords(region_state=('cell', data.REGION_STATE_NAME),
                       region_NRM=('cell', data.REGION_NRM_NAME))
    )

    non_ag_rev_mat  = tools.non_ag_rk_to_xr(data, non_ag_revenue.get_rev_matrix(data, yr_cal, ag_rev_mrj, data.lumaps[yr_cal]))
    non_ag_cost_mat = tools.non_ag_rk_to_xr(data, non_ag_cost.get_cost_matrix(data, ag_cost_mrj, data.lumaps[yr_cal], yr_cal))
    nonag2nonag_mat = tools.non_ag_rk_to_xr(data, non_ag_transitions.get_non_ag_to_non_ag_transition_matrix(data)) / gap

    if yr_cal_sim_pre is None:
        ag2nonag_mat = xr.DataArray(np.zeros((data.NCELLS, len(data.NON_AGRICULTURAL_LANDUSES)), dtype=np.float32),
            dims=['cell', 'lu'], coords={'cell': range(data.NCELLS), 'lu': data.NON_AGRICULTURAL_LANDUSES})
    else:
        ag2nonag_mat = tools.non_ag_rk_to_xr(data, non_ag_transitions.get_transition_matrix_ag2nonag(
            data, yr_idx, data.lumaps[yr_cal_sim_pre], data.lmmaps[yr_cal_sim_pre],
            base_year=yr_cal_sim_pre,
            ).astype(np.float32)) / gap

    non_ag_profit_mat = non_ag_rev_mat - (non_ag_cost_mat + nonag2nonag_mat + ag2nonag_mat)
    raw_profit_nonag = (non_ag_dvar * non_ag_profit_mat).drop_vars(['region_state', 'region_NRM'], errors='ignore')

    xr_revenue_non_ag = non_ag_dvar * non_ag_rev_mat
    xr_cost_non_ag    = non_ag_dvar * non_ag_cost_mat
    xr_nonag2nonag    = non_ag_dvar * nonag2nonag_mat
    xr_ag2nonag       = non_ag_dvar * ag2nonag_mat
    xr_non_ag_profit  = non_ag_dvar * non_ag_profit_mat

    del non_ag_rev_mat, non_ag_cost_mat, nonag2nonag_mat, ag2nonag_mat, non_ag_profit_mat, non_ag_dvar
    del ag_rev_mrj, ag_cost_mrj
    gc.collect()

    xr_revenue_non_ag = add_all(xr_revenue_non_ag, ['lu'])
    xr_cost_non_ag    = add_all(xr_cost_non_ag,    ['lu'])
    xr_nonag2nonag    = add_all(xr_nonag2nonag,    ['lu'])
    xr_ag2nonag       = add_all(xr_ag2nonag,       ['lu'])
    xr_non_ag_profit  = add_all(xr_non_ag_profit,  ['lu'])

    revenue_na_df,    revenue_na_df_AUS    = to_region_and_aus_df(xr_revenue_non_ag, ['lu'], yr_cal, region_levels=['region_state', 'region_NRM'])
    cost_na_df,       cost_na_df_AUS       = to_region_and_aus_df(xr_cost_non_ag,    ['lu'], yr_cal, region_levels=['region_state', 'region_NRM'])
    t_nonag2nonag_df, t_nonag2nonag_df_AUS = to_region_and_aus_df(xr_nonag2nonag,    ['lu'], yr_cal, region_levels=['region_state', 'region_NRM'])
    t_ag2nonag_df,    t_ag2nonag_df_AUS    = to_region_and_aus_df(xr_ag2nonag,       ['lu'], yr_cal, region_levels=['region_state', 'region_NRM'])
    profit_na_df,     profit_na_df_AUS     = to_region_and_aus_df(xr_non_ag_profit,  ['lu'], yr_cal, region_levels=['region_state', 'region_NRM'])

    rename_na = {'lu': 'Land-use', 'Value': 'Value ($)'}
    save_csv(revenue_na_df,    rename_na, os.path.join(path, f'economics_non_ag_revenue_{yr_cal}.csv'))
    save_csv(cost_na_df,       rename_na, os.path.join(path, f'economics_non_ag_cost_{yr_cal}.csv'))
    save_csv(t_nonag2nonag_df, rename_na, os.path.join(path, f'economics_non_ag_transition_NonAg2NonAg_{yr_cal}.csv'))
    save_csv(t_ag2nonag_df,    rename_na, os.path.join(path, f'economics_non_ag_transition_Ag2NonAg_{yr_cal}.csv'))
    save_csv(profit_na_df,     rename_na, os.path.join(path, f'economics_non_ag_profit_{yr_cal}.csv'))

    vl_rev_na = (
        pd.MultiIndex.from_tuples([('ALL',)], names=['lu'])
        if revenue_na_df_AUS.empty
        else pd.MultiIndex.from_frame(revenue_na_df_AUS[['lu']]).sort_values()
    )
    vl_cost_na = (
        pd.MultiIndex.from_tuples([('ALL',)], names=['lu'])
        if cost_na_df_AUS.empty
        else pd.MultiIndex.from_frame(cost_na_df_AUS[['lu']]).sort_values()
    )
    vl_t_nonag2nonag = (
        pd.MultiIndex.from_tuples([('ALL',)], names=['lu'])
        if t_nonag2nonag_df_AUS.empty
        else pd.MultiIndex.from_frame(t_nonag2nonag_df_AUS[['lu']]).sort_values()
    )
    vl_t_ag2nonag = (
        pd.MultiIndex.from_tuples([('ALL',)], names=['lu'])
        if t_ag2nonag_df_AUS.empty
        else pd.MultiIndex.from_frame(t_ag2nonag_df_AUS[['lu']]).sort_values()
    )
    vl_profit_na = (
        pd.MultiIndex.from_tuples([('ALL',)], names=['lu'])
        if profit_na_df_AUS.empty
        else pd.MultiIndex.from_frame(profit_na_df_AUS[['lu']]).sort_values()
    )

    vl_stack_rev_na        = xr_revenue_non_ag.stack(layer=['lu']).sel(layer=vl_rev_na).drop_vars(['region_state','region_NRM'], errors='ignore').compute()
    vl_stack_cost_na       = xr_cost_non_ag.stack(layer=['lu']).sel(layer=vl_cost_na).drop_vars(['region_state','region_NRM'], errors='ignore').compute()
    vl_stack_t_nonag2nonag = xr_nonag2nonag.stack(layer=['lu']).sel(layer=vl_t_nonag2nonag).drop_vars(['region_state','region_NRM'], errors='ignore').compute()
    vl_stack_t_ag2nonag    = xr_ag2nonag.stack(layer=['lu']).sel(layer=vl_t_ag2nonag).drop_vars(['region_state','region_NRM'], errors='ignore').compute()
    vl_stack_profit_na     = xr_non_ag_profit.stack(layer=['lu']).sel(layer=vl_profit_na).drop_vars(['region_state','region_NRM'], errors='ignore').compute()

    save2nc(vl_stack_rev_na, os.path.join(path, f'xr_economics_non_ag_revenue_{yr_cal}.nc'))

    save2nc(vl_stack_cost_na, os.path.join(path, f'xr_economics_non_ag_cost_{yr_cal}.nc'))
    save2nc(vl_stack_t_nonag2nonag, os.path.join(path, f'xr_economics_non_ag_transition_NonAg2NonAg_{yr_cal}.nc'))
    save2nc(vl_stack_t_ag2nonag, os.path.join(path, f'xr_economics_non_ag_transition_Ag2NonAg_{yr_cal}.nc'))
    save2nc(vl_stack_profit_na, os.path.join(path, f'xr_economics_non_ag_profit_{yr_cal}.nc'))

    del xr_revenue_non_ag, xr_cost_non_ag, xr_nonag2nonag, xr_ag2nonag, xr_non_ag_profit
    gc.collect()


    # ==================== Sum Profit (Ag + Am + NonAg) ====================

    ag_lus        = list(data.AGRICULTURAL_LANDUSES)
    nonag_lus     = list(data.NON_AGRICULTURAL_LANDUSES)
    am_types_list = [a for a in raw_profit_am_pre.coords['am'].values if a != 'ALL']

    sum_type_ag = raw_profit_ag.sel(lm=['dry','irr'], lu=ag_lus).sum(['lm','lu']).expand_dims({'Type': ['ag']})
    sum_type_am = (
        raw_profit_am_pre.sel(am=am_types_list, lm=['dry','irr'], lu=ag_lus).sum(['am','lm','lu']).expand_dims({'Type': ['ag-man']})
        if am_types_list
        else xr.zeros_like(sum_type_ag).assign_coords(Type=['ag-man'])
    )
    sum_type_nonag = raw_profit_nonag.sel(lu=nonag_lus).sum('lu').expand_dims({'Type': ['non-ag']})
    del raw_profit_ag, raw_profit_am_pre, raw_profit_nonag

    sum_profit       = add_all(xr.concat([sum_type_ag, sum_type_am, sum_type_nonag], dim='Type'), dims=['Type'])
    sum_profit_stack = sum_profit.stack(layer=['Type']).compute()
    del sum_profit, sum_type_ag, sum_type_am, sum_type_nonag
    save2nc(sum_profit_stack, os.path.join(path, f'xr_economics_sum_profit_{yr_cal}.nc'))


    # ==================== Record Cell Magnitudes ====================

    magnitudes = {
        'Economics_ag': {
            'ag_revenue':       get_mag(ag_rev_valid_layers),
            'ag_cost':          get_mag(ag_cost_valid_layers),
            'ag2ag_cost':       get_mag(ag2ag_cost_valid_layers),
            'non_ag2ag_cost':   get_mag(nonag2ag_cost_valid_layers),
            'profit_ag':        get_mag(profit_ag_valid_layers),
        },
        'Economics_am': {
            'am_revenue':       get_mag(valid_layers_stack_rev_am),
            'am_cost':          get_mag(valid_layers_stack_cost_am),
            'am_profit':        get_mag(valid_layers_stack_profit_am),
        },
        'Economics_non_ag': {
            'non_ag_revenue':   get_mag(vl_stack_rev_na),
            'non_ag_cost':      get_mag(vl_stack_cost_na),
            'nonag2nonag_cost': get_mag(vl_stack_t_nonag2nonag),
            'ag2nonag_cost':    get_mag(vl_stack_t_ag2nonag),
            'non_ag_profit':    get_mag(vl_stack_profit_na),
        },
        'Economics_sum': {'sum_profit': get_mag(sum_profit_stack)},
    }
    return (f"Economics (Ag+Am+NonAg+Sum) written for year {yr_cal}", magnitudes)



# ── Renewable energy ────────────────────────────────────────────────────────────────

def write_renewable_production(data: Data, yr_cal, path):
    """No annualisation needed: `get_quantity_renewable()` and `get_exist_renewable_capacity()`
    both return MWh/year (the `x 8760 hours` factor is baked in), independent of the gap
    to the previous simulated year — dividing by `gap` would double-annualise it."""

    yr_idx = yr_cal - data.YR_CAL_BASE
    re_types = list(settings.RENEWABLES_OPTIONS.keys())

    # Get decision variable for renewable energy land-use
    am_dvar_mrj_base = chunk_unify_size(tools.am_mrj_to_xr(data, data.ag_man_dvars[yr_cal])
        ).assign_coords({
            'region_state': ('cell', data.REGION_STATE_NAME),
            'region_NRM':   ('cell', data.REGION_NRM_NAME),
        })

    am_dvar_mrj = xr.concat(
        [
            am_dvar_mrj_base.sel(am=re_type) if settings.RENEWABLES_OPTIONS[re_type]
            else xr.zeros_like(am_dvar_mrj_base.isel(am=0)).assign_coords(am=re_type)
            for re_type in re_types
        ],
        dim='am',
    )

    # Get potential renewable energy production (MWh) for each renewable type — zeros if disabled
    renewable_potentials = xr.DataArray(
        [
            ag_quantity.get_quantity_renewable(data, re_type, yr_idx) if settings.RENEWABLES_OPTIONS[re_type]
            else np.zeros(data.NCELLS, dtype=np.float32)
            for re_type in re_types
        ],
        dims=['am', 'cell'],
        coords={'am': re_types, 'cell': range(data.NCELLS)},
    )
    
    
    # Get renewable energy by dvar * potential
    renewable_energy = am_dvar_mrj * renewable_potentials

    # ── Inject existing capacity as lu='Existing Capacity' before add_all ────
    # Per-cell MWh already available from get_exist_renewable_capacity.
    # Follows the same injection pattern as write_dvar_area and write_economics:
    # lm='dry' carries real values, lm='irr' is zeros (avoids double-counting),
    # and add_all then handles lm='ALL' and lu='ALL' for free.
    if any(settings.RENEWABLES_OPTIONS.values()):
        solar_exist_mwh_r = ag_quantity.get_exist_renewable_capacity(data, 'Utility Solar PV', yr_cal)
        wind_exist_mwh_r  = ag_quantity.get_exist_renewable_capacity(data, 'Onshore Wind',     yr_cal)

        exist_re_dry = xr.DataArray(
            np.stack([solar_exist_mwh_r.values, wind_exist_mwh_r.values], axis=0),
            dims=['am', 'cell'],
            coords={
                'am':          re_types,
                'cell':        range(data.NCELLS),
                'region_state': ('cell', data.REGION_STATE_NAME),
                'region_NRM':  ('cell', data.REGION_NRM_NAME),
            },
        ).expand_dims(lm=['dry'], lu=['Existing Capacity'])

        exist_re_irr  = xr.zeros_like(exist_re_dry).assign_coords(lm=['irr'])
        exist_re_full = xr.concat([exist_re_dry, exist_re_irr], dim='lm')
        renewable_energy = xr.concat([renewable_energy, exist_re_full], dim='lu')

    renewable_energy = add_all(renewable_energy, ['am', 'lu', 'lm'])

    # Regionally aggregate renewable energy for reporting — both state and NRM levels
    renewable_energy_df, renewable_energy_df_AUS = to_region_and_aus_df(
        renewable_energy, ['am', 'lm', 'lu'], yr_cal,
        region_levels=['region_state', 'region_NRM'],
    )

    rename_map_re = {'Value': 'Value (MWh)'}
    save_csv(renewable_energy_df, rename_map_re, os.path.join(path, f'renewable_energy_with_existing_{yr_cal}.csv'))
    
    # Save renewable targets (empty when renewables are off)
    if any(settings.RENEWABLES_OPTIONS.values()):
        re_targets = (
            data.RENEWABLE_TARGETS
            .query('Year == @yr_cal')
            .rename(columns={'state': 'region', 'Renewable_Target_MWh': 'Value (MWh)'})
            .replace({'Utility Solar': 'Utility Solar PV', 'Wind': 'Onshore Wind'})
            [['tech', 'region', 'Year', 'Value (MWh)']]
        )
        re_targets = pd.concat([re_targets, re_targets.groupby(['Year','region'], as_index=False)['Value (MWh)'].sum().assign(tech='ALL')])
        re_targets = pd.concat([re_targets, re_targets.groupby(['Year', 'tech'], as_index=False)['Value (MWh)'].sum().assign(region='AUSTRALIA')])
        re_targets = re_targets.sort_values(['region', 'tech']).rename(columns={'tech': 'am'}).assign(lm='ALL')
    else:
        re_targets = pd.DataFrame(columns=['am', 'region', 'Year', 'Value (MWh)', 'lm'])
    re_targets.to_csv(os.path.join(path, f'renewable_energy_targets_{yr_cal}.csv'), index=False)

    # Stack and save to netcdf for later use in report (e.g., for setting colorbar limits)
    valid_layers = pd.MultiIndex.from_frame(renewable_energy_df_AUS[['am', 'lm', 'lu']]).sort_values()
    renewable_energy_stack = renewable_energy.stack(layer=['am', 'lm', 'lu']).sel(layer=valid_layers).drop_vars(['region_state', 'region_NRM'], errors='ignore').compute()
    save2nc(renewable_energy_stack, os.path.join(path, f'xr_renewable_energy_{yr_cal}.nc'))

    magnitudes = {'renewable_energy': get_mag(renewable_energy_stack)}

    # ── Existing dvar fraction spatial layer ──────────────────────────────────
    # Build a (am, lm, lu, cell) DataArray representing the fraction of each cell
    # already occupied by real-world existing renewable installations.
    # lm='ALL' and lu='Existing Capacity' since the fraction is not lm/lu specific.
    solar_exist_r = ag_quantity.get_existing_renewable_dvar_fraction(data, 'Utility Solar PV', yr_cal)
    wind_exist_r  = ag_quantity.get_existing_renewable_dvar_fraction(data, 'Onshore Wind',     yr_cal)

    # Only the 'am' dimension is needed: in the Vue report the existing layer is shown
    # unconditionally whenever the user selects a renewable AM type, independent of lm/lu.
    exist_dvar = xr.DataArray(
        np.stack([solar_exist_r, wind_exist_r], axis=0),
        dims=['am', 'cell'],
        coords={'am': re_types, 'cell': range(data.NCELLS)},
    )
    exist_dvar = add_all(exist_dvar, ['am'])   # prepend ALL (sum of solar + wind fractions)

    # Stack am into layer MultiIndex so save2nc / get_map2json can handle it
    # consistently with other NetCDF layers (cfxr encode/decode pattern).
    exist_dvar_stack = exist_dvar.stack(layer=['am'])
    save2nc(exist_dvar_stack, os.path.join(path, f'xr_renewable_existing_dvar_{yr_cal}.nc'))

    magnitudes['renewable_existing_dvar'] = get_mag(exist_dvar_stack)
    return (f"Renewable energy written for year {yr_cal}", magnitudes)





# ── Transitions ──────────────────────────────────────────────────────────────

def write_transition_ag2ag(data: Data, yr_cal, path, yr_cal_sim_pre=None):
    """Annualised by `gap`: transition area/cost/GHG/water matrices represent a one-off
    change incurred over the period since the previous simulated year, so are divided
    by `gap` to express as an annual rate."""

    # Set up
    simulated_year_list = sorted(list(data.lumaps.keys()))
    yr_idx = yr_cal - data.YR_CAL_BASE
    chunk_size = min(settings.WRITE_CHUNK_SIZE, data.NCELLS)
    gap = get_year_gap(data, yr_cal)  # annualise: divide period value matrices by this

    # Get index of yr_cal in simulated_year_list (e.g., if yr_cal is 2050 then yr_idx_sim = 2 if snapshot)
    yr_idx_sim = simulated_year_list.index(yr_cal)
    yr_cal_sim_pre = simulated_year_list[yr_idx_sim - 1] if yr_cal_sim_pre is None else yr_cal_sim_pre

    if yr_cal == data.YR_CAL_BASE:
        base_lumap = base_lmmap = l_mrj = l_mrj_not = x_mrj = None
    else:
        base_lumap = data.lumaps[yr_cal_sim_pre]
        base_lmmap = data.lmmaps[yr_cal_sim_pre]
        l_mrj = tools.lumap2ag_l_mrj(base_lumap, base_lmmap)
        l_mrj_not = np.logical_not(l_mrj)
        x_mrj = ag_transitions.get_to_ag_exclude_matrices(data, base_lumap)

    # Get the decision variables for agricultural land-use
    ag_dvar_mrj_target = tools.ag_mrj_to_xr(
        data, 
        data.ag_dvars[yr_cal]
    ).assign_coords(region_state=('cell', data.REGION_STATE_NAME), region_NRM=('cell', data.REGION_NRM_NAME))
    
    ag_dvar_mrj_base = tools.ag_mrj_to_xr(
        data, 
        (tools.lumap2ag_l_mrj(data.lumaps[yr_cal_sim_pre], data.lmmaps[yr_cal_sim_pre]))
    )

    ag_dvar_mrj_target = ag_dvar_mrj_target.rename({'lm': 'To-water-supply', 'lu': 'To-land-use'}
        ).assign_coords(region_state=('cell', data.REGION_STATE_NAME), region_NRM=('cell', data.REGION_NRM_NAME)
        ).chunk({'cell': chunk_size})

    ag_dvar_mrj_base = ag_dvar_mrj_base.rename({'lm': 'From-water-supply', 'lu': 'From-land-use'}
        ).assign_coords(region_state=('cell', data.REGION_STATE_NAME), region_NRM=('cell', data.REGION_NRM_NAME)
        ).chunk({'cell': chunk_size})

    # Delta dvars: only pay transition cost for INCREASES in allocation.
    # When BLENDED_AG_TRANSITION_COSTS=True, use solver D vars directly — they are
    # exactly max(0, X_new - x_old) at optimality and match what entered the objective.
    # Otherwise fall back to clipping the dvar difference.
    if settings.BLENDED_AG_TRANSITION_COSTS and data.ag_delta_dvars.get(yr_cal) is not None:
        ag_dvar_mrj_delta = tools.ag_mrj_to_xr(
            data, data.ag_delta_dvars[yr_cal]
        ).rename({'lm': 'To-water-supply', 'lu': 'To-land-use'}
        ).assign_coords(region_state=('cell', data.REGION_STATE_NAME), region_NRM=('cell', data.REGION_NRM_NAME)
        ).chunk({'cell': chunk_size})
    elif yr_cal != data.YR_CAL_BASE:
        ag_dvar_mrj_old = tools.ag_mrj_to_xr(
            data, data.ag_dvars[yr_cal_sim_pre]
        ).rename({'lm': 'To-water-supply', 'lu': 'To-land-use'}
        ).assign_coords(region_state=('cell', data.REGION_STATE_NAME), region_NRM=('cell', data.REGION_NRM_NAME)
        ).chunk({'cell': chunk_size})
        ag_dvar_mrj_delta = (ag_dvar_mrj_target - ag_dvar_mrj_old).clip(min=0)
    else:
        ag_dvar_mrj_delta = ag_dvar_mrj_target


    # ==================== Transitions - Area ====================

    if yr_idx == 0:
        # If it's the first year, we assume no transition cost (i.e., all land remains the same)
        ag_trans_mat = xr.DataArray(
            np.zeros_like(data.AG_L_MRJ).astype(np.float32),
            coords={
                'To-water-supply': data.LANDMANS,
                'cell': range(data.NCELLS),
                'To-land-use': data.AGRICULTURAL_LANDUSES,
            }
        )
    else:
        # Get the transition area matrices for agricultural land-uses
        ag_trans_mat = xr.DataArray(
            np.einsum('r,mrj,mrj->mrj', data.REAL_AREA, l_mrj_not, x_mrj).astype(np.float32),
            coords={
                'To-water-supply': data.LANDMANS,
                'cell': range(data.NCELLS),
                'To-land-use': data.AGRICULTURAL_LANDUSES,
            }
        )
        
    ag_trans_mat = ag_trans_mat / gap

    xr_ag_trans_area = ag_dvar_mrj_base * ag_dvar_mrj_target * ag_trans_mat

    xr_ag_trans_area = add_all(xr_ag_trans_area, dims=['From-land-use', 'To-land-use', 'From-water-supply', 'To-water-supply'])

    # Calculate total transition area by region and land-use (for report generation later)
    transition_area_region = process_chunks(
        xr_ag_trans_area, data, yr_cal, chunk_size,
        groupby_cols=['From-land-use', 'To-land-use', 'From-water-supply', 'To-water-supply'],
        value_col='Transition Area (ha)',
        region_levels=['region_state', 'region_NRM'],
    )
    transition_area_AUS = transition_area_region.groupby(['region_level', 'From-land-use', 'To-land-use', 'From-water-supply', 'To-water-supply']
        )['Transition Area (ha)'
        ].sum(
        ).reset_index(
        ).assign(region='AUSTRALIA', Year=yr_cal
        ).query('`Transition Area (ha)` > 1') # Skip transitions under 1 ha at national level
        
    # Stack array and save to netcdf for later use in report (e.g., for setting colorbar limits)
    valid_trans_area_layers = pd.MultiIndex.from_frame(
        transition_area_AUS[['From-water-supply', 'To-water-supply', 'From-land-use', 'To-land-use']].drop_duplicates()
    ).sort_values()

    # Save transition area to csv
    pd.concat([transition_area_region, transition_area_AUS]
        ).replace({'dry': 'Dryland', 'irr': 'Irrigated'}
        ).to_csv(os.path.join(path, f'transition_ag2ag_area_{yr_cal}.csv'), index=False)

    
    
    transition_area_stacked = xr_ag_trans_area.stack({'layer': ['From-water-supply', 'To-water-supply', 'From-land-use', 'To-land-use']}
        ).sel(layer=valid_trans_area_layers
        ).drop_vars(['region_state', 'region_NRM']
        ).compute()
        
    save2nc(transition_area_stacked, os.path.join(path, f'xr_transition_ag2ag_area_{yr_cal}.nc'))
        


    # ==================== Transitions - Cost ====================
    if yr_idx == 0:
        ag_transitions_cost_mat = {'Establishment cost': np.zeros((data.NLMS, data.NCELLS, data.N_AG_LUS)).astype(np.float32)}
    else:
        # Get the transition cost matrices for agricultural land-use
        #   l_mrj_not and x_mrj are already considered in the `get_transition_matrices_ag2ag_from_base_year`
        ag_transitions_cost_mat = ag_transitions.get_transition_matrices_ag2ag_from_base_year(data, yr_idx, yr_cal_sim_pre, separate=True)

    ag_transitions_cost_mat = xr.DataArray(
        np.stack(list(ag_transitions_cost_mat.values())).astype(np.float32),
        coords={
            'Cost-type': list(ag_transitions_cost_mat.keys()),
            'To-water-supply': data.LANDMANS,
            'cell': range(data.NCELLS),
            'To-land-use': data.AGRICULTURAL_LANDUSES
        }
    )
    ag_transitions_cost_mat = ag_transitions_cost_mat / gap

    # Use xr.dot() to contract To-water-supply without broadcasting:
    #   Plain `*` would first expand ag_dvar_mrj_target × ag_transitions_cost_mat into a full
    #   [To-water-supply, To-land-use, Cost-type, cell] intermediate array, then sum — allocating that
    #   array even though it is immediately discarded. xr.dot() fuses the multiply-and-sum into
    #   one pass so the large intermediate never exists in memory.
    #   Similarly, From-water-supply is summed on ag_dvar_mrj_base before the multiply to keep
    #   the operand size small.
    cost_xr = (
        ag_dvar_mrj_base.sum(dim='From-water-supply')
        * xr.dot(ag_dvar_mrj_delta, ag_transitions_cost_mat, dims=['To-water-supply'])
    )

    cost_xr = add_all(cost_xr, ['From-land-use', 'To-land-use', 'Cost-type'])

    # Get transition cost by region and land-use; This is for report generation later (e.g., for setting colorbar limits)
    cost_df_region = process_chunks(
        cost_xr, data, yr_cal, chunk_size,
        groupby_cols=['Cost-type', 'From-land-use', 'To-land-use'],
        value_col='Cost ($)',
        region_levels=['region_state', 'region_NRM'],
    )
    cost_df_AUS = cost_df_region.groupby(['region_level', 'From-land-use', 'To-land-use', 'Cost-type', 'Year']
        )['Cost ($)'
        ].sum(
        ).reset_index(
        ).assign(region='AUSTRALIA'
        ).query('abs(`Cost ($)`) > 1e4') # Skip transitions under $10,000 at national level

    # Get valid data layers (before renaming/replacing)
    valid_layers_transition = pd.MultiIndex.from_frame(
        cost_df_AUS[['From-land-use', 'To-land-use', 'Cost-type']].drop_duplicates()
    ).sort_values()

    # Write to csv
    pd.concat([cost_df_region, cost_df_AUS]
        ).replace({'dry': 'Dryland', 'irr': 'Irrigated'}
        ).to_csv(os.path.join(path, f'transition_ag2ag_cost_{yr_cal}.csv'), index=False)

    cost_xr_stacked = cost_xr.stack({'layer': ['From-land-use', 'To-land-use', 'Cost-type']}
        ).drop_vars(['region_state', 'region_NRM']
        ).sel(layer=valid_layers_transition
        ).compute()


    # Save the compact filtered array
    save2nc(cost_xr_stacked, os.path.join(path, f'xr_transition_ag2ag_cost_{yr_cal}.nc'))
    
    
    
    # ==================== Transitions - GHG ====================

    if yr_cal == data.YR_CAL_BASE:
        ghg_t_smrj_values = np.zeros_like(data.AG_L_MRJ).astype(np.float32)[np.newaxis]
        ghg_t_types = ['Unallocated natural to modified']
    else:
        ghg_t_dict = ag_ghg.get_ghg_transition_emissions(data, data.lumaps[yr_cal_sim_pre], separate=True)
        ghg_t_smrj_values = (np.stack(list(ghg_t_dict.values()), axis=0) * l_mrj_not * x_mrj).astype(np.float32)
        ghg_t_types = list(ghg_t_dict.keys())

    ghg_t_smrj = xr.DataArray(
        ghg_t_smrj_values,
        dims=['GHG-type', 'To-water-supply', 'cell', 'To-land-use'],
        coords={
            'GHG-type': ghg_t_types,
            'To-water-supply': data.LANDMANS,
            'cell': range(data.NCELLS),
            'To-land-use': data.AGRICULTURAL_LANDUSES
        }
    )
    ghg_t_smrj = ghg_t_smrj / gap

    # Calculate GHG emissions for transition penalties (collapse water dims via xr.dot, same rationale as cost)
    xr_ghg_transition = (
        ag_dvar_mrj_base.sum(dim='From-water-supply')
        * xr.dot(ag_dvar_mrj_target, ghg_t_smrj, dims=['To-water-supply'])
    )

    xr_ghg_transition = add_all(xr_ghg_transition, ['From-land-use', 'To-land-use', 'GHG-type'])

    # Get transition GHG emissions by region and land-use; This is for report generation later (e.g., for setting colorbar limits)
    ghg_df_region = process_chunks(
        xr_ghg_transition, data, yr_cal, chunk_size,
        groupby_cols=['GHG-type', 'From-land-use', 'To-land-use'],
        value_col='Value (t CO2e)',
        region_levels=['region_state', 'region_NRM'],
    )
    ghg_df_AUS = ghg_df_region.groupby(['region_level', 'From-land-use', 'To-land-use', 'GHG-type', 'Year']
        )['Value (t CO2e)'
        ].sum(
        ).reset_index(
        ).assign(region='AUSTRALIA'
        ).query('abs(`Value (t CO2e)`) > 1e-3') # Skip transitions under 1 t CO2e at national level

    # Get valid data layers (before renaming/replacing)
    valid_transition_layers = pd.MultiIndex.from_frame(ghg_df_AUS[['From-land-use', 'To-land-use', 'GHG-type']].drop_duplicates()).sort_values()

    # Write to csv
    pd.concat([ghg_df_AUS, ghg_df_region]
        ).infer_objects(copy=False
        ).replace({'dry': 'Dryland', 'irr': 'Irrigated'}
        ).to_csv(os.path.join(path, f'transition_ag2ag_ghg_{yr_cal}.csv'), index=False)
    transition_valid_layers = xr_ghg_transition.stack(layer=['From-land-use', 'To-land-use', 'GHG-type']
        ).sel(layer=valid_transition_layers
        ).drop_vars(['region_state', 'region_NRM'])
    save2nc(transition_valid_layers, os.path.join(path, f'xr_transition_ag2ag_ghg_{yr_cal}.nc'))



    # ==================== Transitions - Water ====================
    if yr_cal == data.YR_CAL_BASE:
        w_delta_mrj = xr.DataArray(
            np.zeros_like(data.AG_L_MRJ).astype(np.float32),
            dims=['To-water-supply', 'cell', 'To-land-use'],
            coords={
                'To-water-supply': data.LANDMANS,
                'cell': range(data.NCELLS),
                'To-land-use': data.AGRICULTURAL_LANDUSES
            }
        )
    else:
        w_mrj = ag_water.get_wreq_matrices(data, yr_idx)                                                 
        w_delta_mrj = tools.get_ag_to_ag_water_delta_matrix(w_mrj, l_mrj, data, yr_idx)
        w_delta_mrj = xr.DataArray(
            np.einsum('mrj,mrj,mrj->mrj', w_delta_mrj, x_mrj, l_mrj_not).astype(np.float32),
            dims=['To-water-supply', 'cell', 'To-land-use'],
            coords={
                'To-water-supply': data.LANDMANS,
                'cell': range(data.NCELLS),
                'To-land-use': data.AGRICULTURAL_LANDUSES
            }
        )

    w_delta_mrj = w_delta_mrj / gap

    # Calculate water requirement changes for transition penalties
    xr_water_transition = (
        ag_dvar_mrj_base
        * ag_dvar_mrj_target
        * w_delta_mrj
    )
    
    xr_water_transition = add_all(xr_water_transition, ['From-land-use', 'To-land-use', 'From-water-supply', 'To-water-supply'])
    
    # Get transition water requirement changes by region and land-use; This is for report generation later (e.g., for setting colorbar limits)
    # Flip the water requirement to water yield change for easier
    #   interpolation in report (i.e., requirement decrease = yield increase).
    water_df_region = process_chunks(
        xr_water_transition, data, yr_cal, chunk_size,
        groupby_cols=['From-land-use', 'To-land-use', 'From-water-supply', 'To-water-supply'],
        value_col='Water Requirement Change (ML)',
        region_levels=['region_state', 'region_NRM'],
    )
    water_df_region['Water Yield Change (ML)'] = -water_df_region['Water Requirement Change (ML)']
    water_df_region = water_df_region.drop(columns='Water Requirement Change (ML)')
    water_df_AUS = water_df_region.groupby(['region_level', 'From-land-use', 'To-land-use', 'Year', 'From-water-supply', 'To-water-supply']
        )['Water Yield Change (ML)'
        ].sum(
        ).reset_index(
        ).assign(region='AUSTRALIA')
    water_df_AUS = water_df_AUS.loc[water_df_AUS['Water Yield Change (ML)'].abs() > 1e3] # Skip transitions under 1,000 ML at national level
        
    # Get valid data layers (before renaming/replacing)
    valid_water_transition_layers = pd.MultiIndex.from_frame(water_df_AUS[['From-land-use', 'To-land-use', 'From-water-supply', 'To-water-supply']].drop_duplicates()).sort_values()

    # Save to csv 
    pd.concat([water_df_region, water_df_AUS]
        ).replace({'dry': 'Dryland', 'irr': 'Irrigated'}
        ).to_csv(os.path.join(path, f'transition_ag2ag_water_{yr_cal}.csv'), index=False)
    water_transition_valid_layers = (
        xr_water_transition
        .stack(layer=['From-land-use', 'To-land-use', 'From-water-supply', 'To-water-supply'])
        .sel(layer=valid_water_transition_layers)
        .drop_vars(['region_state', 'region_NRM'])
    )
    save2nc(water_transition_valid_layers, os.path.join(path, f'xr_transition_ag2ag_water_{yr_cal}.nc'))
    
    
    
    # ==================== Transitions - Bio ====================
    '''
    Only consider GBF2 for now. Will add more if requested.
    '''
    # TODO: complete bio transition after introducing the bio transition matrices module.
    
    
    return f"Agricultural to agricultural transition changes written for year {yr_cal}"



def write_transition_ag2nonag(data: Data, yr_cal, path, yr_cal_sim_pre=None):
    """Annualised by `gap`: transition area/cost/GHG/water matrices represent a one-off
    change incurred over the period since the previous simulated year, so are divided
    by `gap` to express as an annual rate."""

    # Set up
    chunk_size = min(settings.WRITE_CHUNK_SIZE, data.NCELLS)
    simulated_year_list = sorted(list(data.lumaps.keys()))
    yr_idx = yr_cal - data.YR_CAL_BASE
    gap = get_year_gap(data, yr_cal)  # annualise: divide period value matrices by this

    # Get index of yr_cal in simulated_year_list (e.g., if yr_cal is 2050 then yr_idx_sim = 2 if snapshot)
    yr_idx_sim = simulated_year_list.index(yr_cal)
    yr_cal_sim_pre = simulated_year_list[yr_idx_sim - 1] if yr_cal_sim_pre is None else yr_cal_sim_pre

    if yr_cal == data.YR_CAL_BASE:
        base_lumap = l_rk = l_rk_not = x_rk = None
    else:
        base_lumap = data.lumaps[yr_cal_sim_pre]
        l_rk = tools.lumap2non_ag_l_mk(base_lumap, data.N_NON_AG_LUS)
        l_rk_not = np.logical_not(l_rk)
        x_rk = non_ag_transitions.get_non_ag_ub_matrices(data, base_lumap)

    # Get the non-agricultural decision variable
    ag_dvar_base = tools.ag_mrj_to_xr(data, (tools.lumap2ag_l_mrj(data.lumaps[yr_cal_sim_pre], data.lmmaps[yr_cal_sim_pre]))
        ).assign_coords(region_state=('cell', data.REGION_STATE_NAME), region_NRM=('cell', data.REGION_NRM_NAME)
        ).rename({'lm': 'From-water-supply', 'lu': 'From-land-use'}
        ).chunk({'cell': chunk_size})
    non_ag_dvar_target = tools.non_ag_rk_to_xr(data, tools.non_ag_rk_to_xr(data, data.non_ag_dvars[yr_cal])
        ).assign_coords(region_state=('cell', data.REGION_STATE_NAME), region_NRM=('cell', data.REGION_NRM_NAME)
        ).rename({'lu': 'To-land-use'}
        ).chunk({'cell': chunk_size})

    # Delta dvars: only pay transition cost for INCREASES in non-ag allocation.
    # Same rationale as ag2ag: using the full dvar_new charges cells already at the
    # target non-ag LU and adds phantom reverse-direction costs for shrinking LUs.
    if yr_cal != data.YR_CAL_BASE:
        non_ag_dvar_old = tools.non_ag_rk_to_xr(data, tools.non_ag_rk_to_xr(data, data.non_ag_dvars[yr_cal_sim_pre])
            ).assign_coords(region_state=('cell', data.REGION_STATE_NAME), region_NRM=('cell', data.REGION_NRM_NAME)
            ).rename({'lu': 'To-land-use'}
            ).chunk({'cell': chunk_size})
        non_ag_dvar_delta = (non_ag_dvar_target - non_ag_dvar_old).clip(min=0)
    else:
        non_ag_dvar_delta = non_ag_dvar_target


    
    # ==================== Transitions - Area ====================
    if yr_idx == 0:
        non_ag_transitions_area_mat = xr.DataArray(
            np.zeros((data.NCELLS, data.N_NON_AG_LUS), dtype=np.float32),
            coords={
                'cell': range(data.NCELLS),
                'To-land-use': data.NON_AGRICULTURAL_LANDUSES
            }
        )
    else:
        non_ag_transitions_area_mat = xr.DataArray(
            np.einsum('r,rk,rk->rk', data.REAL_AREA, l_rk_not, x_rk).astype(np.float32),
            coords={
                'cell': range(data.NCELLS),
                'To-land-use': data.NON_AGRICULTURAL_LANDUSES
            }
        )

    non_ag_transitions_area_mat = non_ag_transitions_area_mat / gap

    # Calculate transition area; expand To-water-supply='dry' (area only, not cost/GHG)
    non_ag_transitions_area = ag_dvar_base * non_ag_transitions_area_mat * non_ag_dvar_target.expand_dims({'To-water-supply': ['dry']})
    non_ag_transitions_area = add_all(non_ag_transitions_area, ['From-water-supply', 'To-water-supply', 'From-land-use', 'To-land-use'])
    
    # Get transition area by region and land-use; This is for report generation later (e.g., for setting colorbar limits)
    area_df_region = process_chunks(
        non_ag_transitions_area, data, yr_cal, chunk_size,
        groupby_cols=['From-water-supply', 'To-water-supply', 'From-land-use', 'To-land-use'],
        value_col='Transition Area (ha)',
        region_levels=['region_state', 'region_NRM'],
    )
    area_df_AUS = area_df_region.groupby(['region_level', 'From-water-supply', 'To-water-supply', 'From-land-use', 'To-land-use', 'Year']
        )['Transition Area (ha)'
        ].sum(
        ).reset_index(
        ).assign(region='AUSTRALIA'
        ).query('`Transition Area (ha)` > 1') # Skip transitions under 1 ha at national level
        
        
    # Get valid data layers (before renaming/replacing)
    valid_layers_transition = pd.MultiIndex.from_frame(
        area_df_AUS[['From-water-supply', 'To-water-supply', 'From-land-use', 'To-land-use']].drop_duplicates()
    ).sort_values()
    
    # Save to csv
    pd.concat([area_df_region, area_df_AUS]
        ).replace({'dry': 'Dryland', 'irr': 'Irrigated'}
        ).to_csv(os.path.join(path, f'transition_ag2nonag_area_{yr_cal}.csv'), index=False)
    
    valid_layers_stack_area = non_ag_transitions_area.stack({'layer': ['From-water-supply', 'To-water-supply', 'From-land-use', 'To-land-use']}
        ).sel(layer=valid_layers_transition).drop_vars(['region_state', 'region_NRM']).compute()
    
    save2nc(valid_layers_stack_area, os.path.join(path, f'xr_transition_ag2nonag_area_{yr_cal}.nc'))
    
        
    

    # ==================== Transitions - Cost ====================
    if yr_idx == 0:
        non_ag_transitions_cost_mat = {
            k:{'Transition cost':np.zeros(data.NCELLS).astype(np.float32)}
            for k in settings.NON_AG_LAND_USES.keys()
        }
    else:
        non_ag_transitions_cost_mat = non_ag_transitions.get_transition_matrix_ag2nonag(
            data, yr_idx, data.lumaps[yr_cal_sim_pre], data.lmmaps[yr_cal_sim_pre],
            base_year=yr_cal_sim_pre,
            separate=True,
        )

    non_ag_transitions_flat = {}
    for lu, sub_dict in non_ag_transitions_cost_mat.items():
        for source, arr in sub_dict.items():
            non_ag_transitions_flat[(lu, source)] = arr
            
    non_ag_transitions_flat = xr.DataArray(
        np.stack(list(non_ag_transitions_flat.values())).astype(np.float32),
        coords={
            'lu_source': pd.MultiIndex.from_tuples(list(non_ag_transitions_flat.keys()), names= ('To-land-use', 'Cost-type')),
            'cell': range(data.NCELLS),
        }
    )
    non_ag_transitions_flat = non_ag_transitions_flat / gap

    # Compute in chunks and aggregate to DataFrame; This is to reduce memory usage
    cost_xr = (
        ag_dvar_base.sum(dim='From-water-supply')       # Colapse water supply dimension to reduce mem
        * non_ag_transitions_flat.unstack('lu_source')
        * non_ag_dvar_delta
    )
    
    cost_xr = add_all(cost_xr, ['From-land-use', 'To-land-use', 'Cost-type'])


    # Get transition cost by region and land-use
    cost_df_region = process_chunks(
        cost_xr, data, yr_cal, chunk_size,
        groupby_cols=['From-land-use', 'To-land-use', 'Cost-type'],
        value_col='Cost ($)',
        region_levels=['region_state', 'region_NRM'],
    )
    cost_df_AUS = cost_df_region.groupby(['region_level', 'From-land-use', 'To-land-use', 'Cost-type', 'Year']
        )['Cost ($)'
        ].sum(
        ).reset_index(
        ).assign(region='AUSTRALIA'
        ).query('abs(`Cost ($)`) > 1000') # Skip transitions under $1,000 at national level
        
    # Get valid data layers (before renaming/replacing)
    valid_layers_transition = pd.MultiIndex.from_frame(
        cost_df_AUS[['From-land-use', 'To-land-use', 'Cost-type']].drop_duplicates()
    ).sort_values()

    pd.concat([cost_df_AUS, cost_df_region]
        ).infer_objects(copy=False
        ).replace({'dry':'Dryland', 'irr':'Irrigated'}
        ).to_csv(os.path.join(path, f'transition_ag2nonag_cost_{yr_cal}.csv'), index=False)
    
    valid_layers_stack_cost = cost_xr.stack({'layer': ['From-land-use', 'To-land-use', 'Cost-type']}
        ).sel(layer=valid_layers_transition).drop_vars(['region_state', 'region_NRM']).compute()
    
    save2nc(valid_layers_stack_cost, os.path.join(path, f'xr_transition_ag2nonag_cost_{yr_cal}.nc'))
    
    
    
    

    # ==================== Transitions - GHG ====================
    if yr_idx == 0:
        g_rk = xr.DataArray(
            np.zeros((data.NCELLS, data.N_NON_AG_LUS), dtype=np.float32),
            coords={'cell': range(data.NCELLS), 'To-land-use': data.NON_AGRICULTURAL_LANDUSES}
        )
    else:
        ag_g_mrj = ag_ghg.get_ghg_matrices(data, yr_idx, aggregate=True)
        g_rk_raw = non_ag_ghg.get_ghg_matrix(data, ag_g_mrj, data.lumaps[yr_cal_sim_pre]).astype(np.float32)
        g_rk_eligible = np.einsum('rk,rk,rk->rk', g_rk_raw, l_rk_not, x_rk).astype(np.float32)
        g_rk = xr.DataArray(
            g_rk_eligible,
            coords={'cell': range(data.NCELLS), 'To-land-use': data.NON_AGRICULTURAL_LANDUSES}
        )

    g_rk = g_rk / gap

    xr_ghg_transition = (
        ag_dvar_base.sum(dim='From-water-supply')
        * non_ag_dvar_target
        * g_rk
    )
    xr_ghg_transition = add_all(xr_ghg_transition, ['From-land-use', 'To-land-use'])

    ghg_df_region = process_chunks(
        xr_ghg_transition, data, yr_cal, chunk_size,
        groupby_cols=['From-land-use', 'To-land-use'],
        value_col='Value (t CO2e)',
        region_levels=['region_state', 'region_NRM'],
    )
    ghg_df_AUS = ghg_df_region.groupby(['region_level', 'From-land-use', 'To-land-use', 'Year']
        )['Value (t CO2e)'
        ].sum(
        ).reset_index(
        ).assign(region='AUSTRALIA'
        ).query('abs(`Value (t CO2e)`) > 1e-3')

    valid_ghg_layers = pd.MultiIndex.from_frame(ghg_df_AUS[['From-land-use', 'To-land-use']].drop_duplicates()).sort_values()

    pd.concat([ghg_df_AUS, ghg_df_region]
        ).infer_objects(copy=False
        ).replace({'dry': 'Dryland', 'irr': 'Irrigated'}
        ).to_csv(os.path.join(path, f'transition_ag2nonag_ghg_{yr_cal}.csv'), index=False)

    ghg_transition_valid = xr_ghg_transition.stack(layer=['From-land-use', 'To-land-use']
        ).sel(layer=valid_ghg_layers
        ).drop_vars(['region_state', 'region_NRM'])
    save2nc(ghg_transition_valid, os.path.join(path, f'xr_transition_ag2nonag_ghg_{yr_cal}.nc'))



    # ==================== Transitions - Water ====================
    if yr_idx == 0:
        w_rk = xr.DataArray(
            np.zeros((data.NCELLS, data.N_NON_AG_LUS), dtype=np.float32),
            coords={'cell': range(data.NCELLS), 'To-land-use': data.NON_AGRICULTURAL_LANDUSES}
        )
    else:
        ag_w_mrj = ag_water.get_wreq_matrices(data, yr_idx)
        w_rk_raw = non_ag_water.get_w_net_yield_matrix(data, ag_w_mrj, data.lumaps[yr_cal_sim_pre], yr_idx).astype(np.float32)
        w_rk_eligible = np.einsum('rk,rk,rk->rk', w_rk_raw, l_rk_not, x_rk).astype(np.float32)
        w_rk = xr.DataArray(
            w_rk_eligible,
            coords={'cell': range(data.NCELLS), 'To-land-use': data.NON_AGRICULTURAL_LANDUSES}
        )

    w_rk = w_rk / gap

    xr_water_transition = (
        ag_dvar_base.sum(dim='From-water-supply')
        * non_ag_dvar_target
        * w_rk
    )
    xr_water_transition = add_all(xr_water_transition, ['From-land-use', 'To-land-use'])

    water_df_region = process_chunks(
        xr_water_transition, data, yr_cal, chunk_size,
        groupby_cols=['From-land-use', 'To-land-use'],
        value_col='Water Yield Change (ML)',
        region_levels=['region_state', 'region_NRM'],
    )
    water_df_AUS = water_df_region.groupby(['region_level', 'From-land-use', 'To-land-use', 'Year']
        )['Water Yield Change (ML)'
        ].sum(
        ).reset_index(
        ).assign(region='AUSTRALIA')
    water_df_AUS = water_df_AUS.loc[water_df_AUS['Water Yield Change (ML)'].abs() > 1e3]

    valid_water_layers = pd.MultiIndex.from_frame(water_df_AUS[['From-land-use', 'To-land-use']].drop_duplicates()).sort_values()

    pd.concat([water_df_region, water_df_AUS]
        ).replace({'dry': 'Dryland', 'irr': 'Irrigated'}
        ).to_csv(os.path.join(path, f'transition_ag2nonag_water_{yr_cal}.csv'), index=False)

    water_transition_valid = xr_water_transition.stack(layer=['From-land-use', 'To-land-use']
        ).sel(layer=valid_water_layers
        ).drop_vars(['region_state', 'region_NRM'])
    save2nc(water_transition_valid, os.path.join(path, f'xr_transition_ag2nonag_water_{yr_cal}.nc'))



    return f"Agricultural to non-agricultural transition written for year {yr_cal}"





def write_transition_nonag2ag(data: Data, yr_cal, path, yr_cal_sim_pre=None):
    """Annualised by `gap`: transition cost matrix represents a one-off change incurred
    over the period since the previous simulated year, so is divided by `gap` to
    express as an annual rate."""

    # Set up
    simulated_year_list = sorted(list(data.lumaps.keys()))
    yr_idx = yr_cal - data.YR_CAL_BASE
    gap = get_year_gap(data, yr_cal)  # annualise: divide period value matrices by this

    # Get index of yr_cal in simulated_year_list (e.g., if yr_cal is 2050 then yr_idx_sim = 2 if snapshot)
    yr_idx_sim = simulated_year_list.index(yr_cal)
    yr_cal_sim_pre = simulated_year_list[yr_idx_sim - 1] if yr_cal_sim_pre is None else yr_cal_sim_pre

    # Get the decision variables for agricultural land-use
    nonag_dvar = tools.non_ag_rk_to_xr(data, data.non_ag_dvars[yr_cal]
        ).rename({'lu': 'From-land-use'}
        ).assign_coords(region=('cell', data.REGION_NRM_NAME)
        ).chunk({'cell': min(settings.WRITE_CHUNK_SIZE, data.NCELLS)})
    ag_dvar = tools.ag_mrj_to_xr(data, data.ag_dvars[yr_cal]
        ).rename({'lm': 'To-water-supply', 'lu': 'To-land-use'}
        ).assign_coords(region=('cell', data.REGION_NRM_NAME))
        
    # ==================== Transitions - Area ====================
    '''
    NonAg to Ag transition is currently prohibited in the model, so the transition area is zero.
    We skip the area calculation and directly create a zero array for the cost calculation below.
    We keep the code here as a placeholder for future when the transition is allowed.
    '''
    
    area_df = pd.DataFrame({
        'region': ['AUSTRALIA', 'ACT'],
        'From-water-supply': ['ALL', 'ALL'],
        'From-land-use': ['ALL', 'ALL'],
        'To-land-use': ['ALL', 'ALL'],
        'Transition Area (ha)': [0, 0],
        'Year': [yr_cal, yr_cal]
    })
    
    area_df.to_csv(os.path.join(path, f'transition_nonag2ag_area_{yr_cal}.csv'), index=False)
    
    area_xr = xr.DataArray(
        np.zeros((1, 1, data.NCELLS, 1)).astype(np.float32),
        coords={
            'From-water-supply': ['ALL'],
            'From-land-use': ['ALL'],
            'cell': range(data.NCELLS),
            'To-land-use': ['ALL']
        }
    )
    area_xr_stack = area_xr.stack({'layer': ['From-water-supply', 'From-land-use', 'To-land-use']})
    save2nc(area_xr_stack, os.path.join(path, f'xr_transition_nonag2ag_area_{yr_cal}.nc'))


    # ==================== Transitions - Cost ====================
    if yr_idx == 0:
        non_ag_transitions_cost_mat = {
            k:{'Transition cost (Non-Ag2Ag)':np.zeros((data.NLMS, data.NCELLS, data.N_AG_LUS)).astype(np.float32)}
            for k in settings.NON_AG_LAND_USES.keys()
        }
    else:
        non_ag_transitions_cost_mat = non_ag_transitions.get_transition_matrix_nonag2ag(
            data,
            yr_idx,
            data.lumaps[yr_cal_sim_pre],
            data.lmmaps[yr_cal_sim_pre],
            separate=True,
            base_year=yr_cal_sim_pre,
        )


    non_ag_transitions_flat = {}
    for lu, sub_dict in non_ag_transitions_cost_mat.items():
        for source, arr in sub_dict.items():
            non_ag_transitions_flat[(lu, source)] = arr.sum(0) # Sum over `lm` dimension
            
    non_ag_transitions_flat = xr.DataArray(
        np.stack(list(non_ag_transitions_flat.values())).astype(np.float32),
        coords={
            'lu_source': pd.MultiIndex.from_tuples(
                list(non_ag_transitions_flat.keys()),
                names= ('From-land-use', 'Cost-type')
            ),
            'cell': range(data.NCELLS),
            'To-land-use': data.AGRICULTURAL_LANDUSES
        }
    ).unstack('lu_source').chunk({'cell': min(settings.WRITE_CHUNK_SIZE, data.NCELLS)})

    non_ag_transitions_flat = non_ag_transitions_flat / gap

    # Compute transition cost
    cost_xr = xr.dot(ag_dvar, nonag_dvar, dim=['To-water-supply']) * non_ag_transitions_flat 
    
    cost_xr = add_all(cost_xr, ['From-land-use', 'To-land-use', 'Cost-type'])
    
    
    
    #   !!! cost_xr is zero for now
    #   !!! so only selecting a chunk to get the stats
    chunk_size = min(settings.WRITE_CHUNK_SIZE, data.NCELLS)
    cost_df_region = cost_xr.isel(cell=slice(0, chunk_size)
        ).groupby('region'
        ).sum(dim='cell'
        ).to_dataframe('Cost ($)'
        ).reset_index(
        ).groupby(['region', 'From-land-use', 'To-land-use', 'Cost-type'], dropna=False
        )['Cost ($)'].sum(
        ).reset_index(
        ).assign(Year=yr_cal)
        
    cost_df_AUS = cost_df_region.groupby(['From-land-use', 'To-land-use', 'Cost-type'],
        )['Cost ($)'
        ].sum(
        ).reset_index(
        ).assign(region='AUSTRALIA', Year=yr_cal)
        
    '''
    NoAg to Ag are currently all zeros, so we skip below calculation.
    '''
    # Get valid data layers (before renaming/replacing)
    valid_layers_transition = pd.MultiIndex.from_frame(
        cost_df_AUS[['From-land-use', 'Cost-type']].iloc[:1] # Only take the first row to avoid empty layers, since all costs are zero for now
    ).sort_values()
    
    # Save to csv
    pd.concat([cost_df_AUS, cost_df_region]
        ).infer_objects(copy=False
        ).replace({'dry':'Dryland', 'irr':'Irrigated'}
        ).to_csv(os.path.join(path, f'transition_nonag2ag_cost_{yr_cal}.csv'), index=False)
    
    cost_xr_stacked = cost_xr.stack({
        'layer': ['From-land-use', 'Cost-type']
    }).drop_vars('region').sel(layer=valid_layers_transition).compute()
    
    # Save valid layers 
    save2nc(cost_xr_stacked, os.path.join(path, f'xr_transition_nonag2ag_cost_{yr_cal}.nc'))

    return f"Non-agricultural to agricultural transition written for year {yr_cal}"



def write_area_transition_start_end(data: Data, path, yr_cal_end):
    """No annualisation: this is a cumulative area transition over the whole simulation
    horizon (YR_CAL_BASE to `yr_cal_end`), not a per-period flow, so `gap` does not apply."""

    yr_cal_start = data.YR_CAL_BASE
    real_area_r = xr.DataArray(data.REAL_AREA.astype(np.float32), dims=['cell'], coords={'cell': range(data.NCELLS)})

    # Get the decision variables for the start year
    ag_dvar_base_mrj = tools.ag_mrj_to_xr(data, tools.lumap2ag_l_mrj(data.lumaps[yr_cal_start], data.lmmaps[yr_cal_start])
        ).assign_coords({'region': ('cell', data.REGION_NRM_NAME)}
        ).rename({'lu':'From-land-use', 'lm':'From-water-supply'}
        ).chunk({'cell': min(settings.WRITE_CHUNK_SIZE, data.NCELLS)})

    ag_dvar_target_mrj = tools.ag_mrj_to_xr(
        data, tools.lumap2ag_l_mrj(data.lumaps[yr_cal_end], data.lmmaps[yr_cal_end])
        ).rename({'lu':'To-land-use', 'lm':'To-water-supply'}
        ).chunk({'cell': min(settings.WRITE_CHUNK_SIZE, data.NCELLS)})

    non_ag_dvar_target_rk = tools.non_ag_rk_to_xr(
        data, data.non_ag_dvars[yr_cal_end]
        ).rename({'lu':'To-land-use'}
        ).chunk({'cell': min(settings.WRITE_CHUNK_SIZE, data.NCELLS)})

    xr_ag2ag = ag_dvar_base_mrj * ag_dvar_target_mrj * real_area_r
    xr_ag2non_ag = ag_dvar_base_mrj * non_ag_dvar_target_rk * real_area_r
    
    # Assign dry to water supply dimension for non-ag
    xr_ag2non_ag = xr_ag2non_ag.expand_dims({'To-water-supply': ['dry']})

    xr_ag2ag     = add_all(xr_ag2ag,     ['From-water-supply', 'From-land-use', 'To-water-supply', 'To-land-use'])
    xr_ag2non_ag = add_all(xr_ag2non_ag, ['From-water-supply', 'From-land-use', 'To-water-supply', 'To-land-use'])


    # ==================== Chunk Level Aggregation ====================
    '''
    Process both ag2ag and ag2non_ag transitions in a single loop to reduce memory usage.
        This is because the `xr_ag2ag` and `xr_ag2non_ag` are huge intermediate arrays that consume a lot of memory.
        By manually selecting each chunk, we can limit the size of the intermediate array.

    '''
    chunk_size = min(settings.WRITE_CHUNK_SIZE, data.NCELLS)
    transition_ag2ag_dfs = []
    transition_ag2non_ag_dfs = []

    for i in range(0, data.NCELLS, chunk_size):
        end_idx = min(i + chunk_size, data.NCELLS)
        cell_slice = slice(i, end_idx)

        # Process ag2ag chunk
        chunk_arr_ag2ag = xr_ag2ag.isel(cell=cell_slice).compute()
        transition_df_ag2ag = chunk_arr_ag2ag.groupby('region'
            ).sum(dim='cell'
            ).to_dataframe('Area (ha)'
            ).reset_index(
            ).groupby(['region', 'From-water-supply', 'From-land-use', 'To-water-supply', 'To-land-use']
            )['Area (ha)'
            ].sum(
            ).reset_index(
            ).query('abs(`Area (ha)`) > 0.01'
            ).assign(chunk_idx=i//chunk_size)
        transition_ag2ag_dfs.append(transition_df_ag2ag)

        # Process ag2non_ag chunk
        chunk_arr_ag2non_ag = xr_ag2non_ag.isel(cell=cell_slice).compute()
        transition_df_ag2non_ag = chunk_arr_ag2non_ag.groupby('region'
            ).sum(dim='cell'
            ).to_dataframe('Area (ha)'
            ).reset_index(
            ).groupby(['region', 'From-water-supply', 'From-land-use', 'To-water-supply', 'To-land-use']
            )['Area (ha)'
            ].sum(
            ).reset_index(
            ).query('abs(`Area (ha)`) > 0.01'
            ).assign(chunk_idx=i//chunk_size)
        transition_ag2non_ag_dfs.append(transition_df_ag2non_ag)

    # Combine all chunks df for ag2ag
    transition_ag2ag = pd.concat(transition_ag2ag_dfs, ignore_index=True
        ).groupby(['region', 'From-water-supply', 'From-land-use', 'To-water-supply', 'To-land-use']
        )['Area (ha)'
        ].sum(
        ).reset_index()

    transition_ag2ag_AUS = transition_ag2ag.groupby(['From-water-supply', 'From-land-use', 'To-water-supply', 'To-land-use']
        )['Area (ha)'
        ].sum(
        ).reset_index(
        ).assign(region='AUSTRALIA'
        ).query('abs(`Area (ha)`) > 1')  # Skip transitions under 1 ha at national level

    # Combine all chunks df for ag2non_ag
    # To-water-supply is already set ('dry' and 'ALL') from xr_ag2non_ag via add_all.
    transition_ag2non_ag = pd.concat(transition_ag2non_ag_dfs, ignore_index=True
        ).groupby(['region', 'From-water-supply', 'From-land-use', 'To-water-supply', 'To-land-use']
        )['Area (ha)'
        ].sum(
        ).reset_index()

    transition_ag2non_ag_AUS = transition_ag2non_ag.groupby(['From-water-supply', 'From-land-use', 'To-land-use', 'To-water-supply']
        )['Area (ha)'
        ].sum(
        ).reset_index(
        ).assign(region='AUSTRALIA'
        ).query('abs(`Area (ha)`) > 1')  # Skip transitions under 1 ha at national level

    # Write the unified transition matrix (ag2ag + ag2non_ag)
    pd.concat([transition_ag2ag, transition_ag2ag_AUS,
               transition_ag2non_ag, transition_ag2non_ag_AUS]
        ).infer_objects(copy=False
        ).replace({'dry':'Dryland', 'irr':'Irrigated'}
        ).to_csv(os.path.join(path, f'transition_matrix_start_end.csv'), index=False)


    # ==================== Stack Array, Get Valid Layers for ag2ag ====================
    '''
    We manually loop through chunks to save stacked array to reduce memory usage.
        The materializing of stacked arrays requires a lot of memory.
    '''

    # Get valid data layers for ag2ag
    valid_layers_ag2ag = pd.MultiIndex.from_frame(
        transition_ag2ag_AUS[['From-water-supply', 'From-land-use', 'To-water-supply', 'To-land-use']]
    ).sort_values()

    xr_ag2ag_stacked = xr_ag2ag.stack({
        'layer': ['From-water-supply', 'From-land-use', 'To-water-supply', 'To-land-use']
    }).sel(layer=valid_layers_ag2ag)

    # Materialize the filtered array by looping through chunks
    xr_ag2ag_filtered_array = xr.DataArray(
        np.zeros((data.NCELLS, len(valid_layers_ag2ag)), dtype=np.float32),
        coords={
            'cell': range(data.NCELLS),
            'layer': valid_layers_ag2ag
        }
    )

    for i in range(0, data.NCELLS, chunk_size):
        end_idx = min(i + chunk_size, data.NCELLS)
        cell_slice = slice(i, end_idx)
        xr_ag2ag_filtered_array[cell_slice, :] = xr_ag2ag_stacked.isel(cell=cell_slice)

    # Save the compact filtered array
    save2nc(xr_ag2ag_filtered_array, os.path.join(path, f'xr_transition_ag2ag_area_start_end.nc'))


    # ==================== Stack Array, Get Valid Layers for ag2non_ag ====================

    # Get valid data layers for ag2non_ag (spatial nc only needs the 'dry' slice;
    # 'ALL' is the same values since non-ag has no irrigation dimension)
    valid_layers_ag2non_ag = pd.MultiIndex.from_frame(
        transition_ag2non_ag_AUS.query('`To-water-supply` == "dry"')[['From-water-supply', 'From-land-use', 'To-land-use']]
    ).sort_values()

    xr_ag2non_ag_stacked = xr_ag2non_ag.sel({'To-water-supply': 'dry'}).stack({
        'layer': ['From-water-supply', 'From-land-use', 'To-land-use']
    }).sel(layer=valid_layers_ag2non_ag)

    # Materialize the filtered array by looping through chunks
    xr_ag2non_ag_filtered_array = xr.DataArray(
        np.zeros((data.NCELLS, len(valid_layers_ag2non_ag)), dtype=np.float32),
        coords={
            'cell': range(data.NCELLS),
            'layer': valid_layers_ag2non_ag
        }
    )

    for i in range(0, data.NCELLS, chunk_size):
        end_idx = min(i + chunk_size, data.NCELLS)
        cell_slice = slice(i, end_idx)
        xr_ag2non_ag_filtered_array[cell_slice, :] = xr_ag2non_ag_stacked.isel(cell=cell_slice)

    # Save the compact filtered array
    save2nc(xr_ag2non_ag_filtered_array, os.path.join(path, f'xr_transition_area_ag2non_ag_start_end.nc'))
    
    # Record maximum cell magnitude for this transition period for later use in scaling the transition area in the visualization
    def _minmax(arr):
        return (arr.min().item(), arr.max().item()) if arr.size > 0 else (0.0, 0.0)

    return (f"Area transition matrix written from year {data.YR_CAL_BASE} to {yr_cal_end}", {
        'transition_area': {
            'ag2ag':     _minmax(xr_ag2ag_filtered_array),
            'ag2non_ag': _minmax(xr_ag2non_ag_filtered_array),
        }
    })



def write_crosstab(data: Data, yr_cal, path):
    """No annualisation needed: this is a land-use crosstab between two point-in-time
    lumap snapshots (stock-to-stock), not a flow accumulated over `gap` years.

    Write out land-use and production data"""

    if yr_cal == data.YR_CAL_BASE:
        return "Skip land-use transition calculation for the base year."

    simulated_year_list = sorted(list(data.lumaps.keys()))
    yr_idx_sim = simulated_year_list.index(yr_cal)
    yr_cal_sim_pre = simulated_year_list[yr_idx_sim - 1]
    
    # Check if yr_cal_sim_pre meets the requirement
    assert yr_cal_sim_pre >= data.YR_CAL_BASE and yr_cal_sim_pre < yr_cal,\
        f"yr_cal_sim_pre ({yr_cal_sim_pre}) must be >= {data.YR_CAL_BASE} and < {yr_cal}"

    lumap_pre = data.lumaps[yr_cal_sim_pre]
    lumap = data.lumaps[yr_cal]
    
    crosstab_region = pd.crosstab(lumap_pre,  [lumap, data.REGION_NRM_NAME], values=data.REAL_AREA, aggfunc=lambda x:x.sum(), margins = False
        ).unstack(
        ).reset_index(
        ).rename(
            columns={
                'row_0': 'From-land-use', 
                'NRM_NAME': 'region', 
                'col_0':'To-land-use', 
                'col_1': 'region',
                0: 'Area (ha)'
            }
        ).dropna(
        ).infer_objects(copy=False
        ).replace({'From-land-use': data.ALLLU2DESC, 'To-land-use': data.ALLLU2DESC})
    crosstab_AUS = crosstab_region.groupby(['From-land-use', 'To-land-use']        
        )['Area (ha)'
        ].sum(
        ).reset_index(
        ).assign(region='AUSTRALIA')
        
    crosstab = pd.concat([crosstab_AUS, crosstab_region], ignore_index=True)
        
        
    switches = (crosstab.groupby(['region', 'From-land-use'])['Area (ha)'].sum() - crosstab.groupby(['region', 'To-land-use'])['Area (ha)'].sum()
        ).reset_index(
        ).rename(columns={'index':'Landuse'}
        ).query('abs(`Area (ha)`) > 100') # Skip switches under 100 ha
    
    
    crosstab['Year'] = yr_cal
    switches['Year'] = yr_cal

    crosstab.to_csv(os.path.join(path, f'crosstab-lumap_{yr_cal}.csv'), index=False)
    switches.to_csv(os.path.join(path, f'switches-lumap_{yr_cal}.csv'), index=False)

    return f"Land-use cross-tabulation and switches written for year {yr_cal}"




# ── GHG ──────────────────────────────────────────────────────────────────────

def write_ghg(data: Data, yr_cal: int, path: str):
    """Mixed annualisation: ag/non-ag/ag-man GHG matrices (`ag_g_rsmj`, `non_ag_g_rk`,
    `ag_man_g_mrj` from `get_ghg_matrices`/`get_agricultural_management_ghg_matrices`)
    are already annual emission rates and are NOT divided by `gap`. 
    
    The transition penalty matrix (`ghg_t_smrj`) represents a one-off emission incurred 
    over the period since the previous simulated year and IS divided by `gap`.

    Write all GHG emissions outputs to NetCDF and CSV files.

    Covers: total/limit summary, off-land commodity, agricultural land-use,
    non-agricultural land-use, agricultural management, land-use transition
    penalties, and cross-category sum.
    """
    yr_idx = yr_cal - data.YR_CAL_BASE
    gap = get_year_gap(data, yr_cal)  # annualise: divide period value matrices by this

    # ==================== Total / Limit Summary ====================

    ghg_limits = 0 if settings.GHG_EMISSIONS_LIMITS == 'off' else data.GHG_TARGETS[yr_cal]
    if yr_cal >= data.YR_CAL_BASE + 1:
        ghg_emissions = data.prod_data[yr_cal]['GHG']
    else:
        ghg_emissions = (ag_ghg.get_ghg_matrices(data, yr_idx, aggregate=True) * data.ag_dvars[settings.SIM_YEARS[0]]).sum()
    pd.DataFrame({
        'Variable': ['GHG_EMISSIONS_LIMIT_TCO2e', 'GHG_EMISSIONS_TCO2e'],
        'Emissions (t CO2e)': [ghg_limits, ghg_emissions],
        'Year': yr_cal,
    }).to_csv(os.path.join(path, f'GHG_emissions_{yr_cal}.csv'), index=False)

    # ==================== Off-land Commodity ====================

    offland_ghg = data.OFF_LAND_GHG_EMISSION.query(f'YEAR == {yr_cal}').rename(columns={'YEAR': 'Year'})
    offland_ghg.to_csv(os.path.join(path, f'GHG_emissions_offland_commodity_{yr_cal}.csv'), index=False)

    # ==================== Agricultural Land-use ====================

    ag_g_xr = xr.Dataset(ag_ghg.get_ghg_matrices(data, yr_idx, aggregate=False)
        ).rename({'dim_0': 'cell'})
    ag_dvar_mrj = chunk_unify_size(tools.ag_mrj_to_xr(data, data.ag_dvars[yr_cal])
        ).assign_coords(region_state=('cell', data.REGION_STATE_NAME), region_NRM=('cell', data.REGION_NRM_NAME))

    mindex = pd.MultiIndex.from_tuples(ag_g_xr.data_vars.keys(), names=['GHG_source', 'lm', 'lu'])
    mindex_coords = xr.Coordinates.from_pandas_multiindex(mindex, 'variable')
    ag_g_rsmj = chunk_unify_size(ag_g_xr.to_dataarray().assign_coords(mindex_coords).unstack())
    ag_g_rsmj['GHG_source'] = ag_g_rsmj['GHG_source'].to_series().infer_objects(copy=False).replace(GHG_NAMES)

    ghg_e = ag_g_rsmj * ag_dvar_mrj
    ghg_e = add_all(ghg_e, ['lm', 'GHG_source', 'lu'])

    ghg_ag_df, ghg_ag_df_AUS = to_region_and_aus_df(ghg_e, ['lm', 'GHG_source', 'lu'], yr_cal, region_levels=['region_state', 'region_NRM'])
    (ghg_ag_df
        .rename(columns={'Value': 'Value (t CO2e)', 'lu': 'Land-use', 'lm': 'Water_supply', 'GHG_source': 'Source'})
        .assign(Type='Agricultural Land-use')
        .infer_objects(copy=False)
        .replace({'dry': 'Dryland', 'irr': 'Irrigated'})
        .query('abs(`Value (t CO2e)`) > 1e-3')
        .to_csv(os.path.join(path, f'GHG_emissions_separate_agricultural_landuse_{yr_cal}.csv'), index=False))
    ghg_df_AUS = ghg_ag_df_AUS

    valid_ghg_layers = pd.MultiIndex.from_frame(ghg_df_AUS[['lm', 'GHG_source', 'lu']]).sort_values()
    valid_layers_stack_ghg = ghg_e.stack(layer=['lm', 'GHG_source', 'lu']).sel(layer=valid_ghg_layers).drop_vars(['region_state', 'region_NRM'], errors='ignore').compute()
    save2nc(valid_layers_stack_ghg, os.path.join(path, f'xr_GHG_ag_{yr_cal}.nc'))

    # ==================== Non-Agricultural Land-use ====================

    non_ag_dvar_rk = chunk_unify_size(tools.non_ag_rk_to_xr(data, data.non_ag_dvars[yr_cal])
        ).assign_coords(region_state=('cell', data.REGION_STATE_NAME), region_NRM=('cell', data.REGION_NRM_NAME))
    non_ag_g_rk = tools.non_ag_rk_to_xr(
        data,
        non_ag_ghg.get_ghg_matrix(data, ag_ghg.get_ghg_matrices(data, yr_idx, aggregate=True), data.lumaps[yr_cal])
    )

    xr_ghg_non_ag = non_ag_dvar_rk * non_ag_g_rk
    xr_ghg_non_ag = add_all(xr_ghg_non_ag, ['lu'])

    ghg_non_ag_df, ghg_non_ag_df_AUS = to_region_and_aus_df(xr_ghg_non_ag, ['lu'], yr_cal, region_levels=['region_state', 'region_NRM'])
    (ghg_non_ag_df
        .rename(columns={'Value': 'Value (t CO2e)', 'lu': 'Land-use'})
        .assign(Type='Non-Agricultural Land-use')
        .query('abs(`Value (t CO2e)`) > 1e-3')
        .to_csv(os.path.join(path, f'GHG_emissions_separate_no_ag_reduction_{yr_cal}.csv'), index=False))

    valid_non_ag_ghg_layers = pd.MultiIndex.from_frame(ghg_non_ag_df_AUS[['lu']]).sort_values()
    if ghg_non_ag_df_AUS['Value'].abs().sum() < 1e-3:
        xr_ghg_non_ag_cat = xr.DataArray(
            np.zeros((1, data.NCELLS), dtype=np.float32),
            dims=['lu', 'cell'],
            coords={'lu': ['ALL'], 'cell': range(data.NCELLS)}
        ).stack(layer=['lu'])
    else:
        xr_ghg_non_ag_cat = xr_ghg_non_ag.stack(layer=['lu']).sel(layer=valid_non_ag_ghg_layers).drop_vars(['region_state', 'region_NRM'], errors='ignore').compute()
    save2nc(xr_ghg_non_ag_cat, os.path.join(path, f'xr_GHG_non_ag_{yr_cal}.nc'))

    # ==================== Agricultural Management ====================

    ag_man_dvar_mrj = chunk_unify_size(tools.am_mrj_to_xr(data, data.ag_man_dvars[yr_cal])
        ).assign_coords(region_state=('cell', data.REGION_STATE_NAME), region_NRM=('cell', data.REGION_NRM_NAME))
    ag_man_g_mrj = tools.am_mrj_to_xr(data, ag_ghg.get_agricultural_management_ghg_matrices(data, yr_idx))

    xr_ghg_ag_man = ag_man_dvar_mrj * ag_man_g_mrj
    xr_ghg_ag_man = add_all(xr_ghg_ag_man, ['lm', 'lu', 'am'])

    ghg_am_df, ghg_am_df_AUS = to_region_and_aus_df(xr_ghg_ag_man, ['am', 'lm', 'lu'], yr_cal, region_levels=['region_state', 'region_NRM'])
    (ghg_am_df
        .rename(columns={'Value': 'Value (t CO2e)', 'lu': 'Land-use', 'lm': 'Water_supply', 'am': 'Agricultural Management Type'})
        .assign(Type='Agricultural Management')
        .infer_objects(copy=False)
        .replace({'dry': 'Dryland', 'irr': 'Irrigated'})
        .query('abs(`Value (t CO2e)`) > 1e-3')
        .to_csv(os.path.join(path, f'GHG_emissions_separate_agricultural_management_{yr_cal}.csv'), index=False))

    valid_am_ghg_layers = pd.MultiIndex.from_frame(ghg_am_df_AUS[['am', 'lm', 'lu']]).sort_values()
    if ghg_am_df_AUS['Value'].abs().sum() < 1e-3:
        valid_layers_stack_am_ghg = xr.DataArray(
            np.zeros((1, 1, 1, data.NCELLS), dtype=np.float32),
            dims=['am', 'lm', 'lu', 'cell'],
            coords={'am': ['ALL'], 'lm': ['ALL'], 'lu': ['ALL'], 'cell': range(data.NCELLS)}
        ).stack(layer=['am', 'lm', 'lu'])
    else:
        valid_layers_stack_am_ghg = xr_ghg_ag_man.stack(layer=['am', 'lm', 'lu']).sel(layer=valid_am_ghg_layers).drop_vars(['region_state', 'region_NRM'], errors='ignore').compute()
    save2nc(valid_layers_stack_am_ghg, os.path.join(path, f'xr_GHG_ag_management_{yr_cal}.nc'))

    # ==================== Transition Penalty ====================

    transition_magnitudes = []
    if yr_cal != data.YR_CAL_BASE:
        simulated_year_list = sorted(list(data.lumaps.keys()))
        yr_idx_sim = simulated_year_list.index(yr_cal)
        yr_cal_sim_pre = simulated_year_list[yr_idx_sim - 1]

        ghg_t_dict = ag_ghg.get_ghg_transition_emissions(data, data.lumaps[yr_cal_sim_pre], separate=True)
        ghg_t_smrj = xr.DataArray(
            np.stack(list(ghg_t_dict.values()), axis=0).astype(np.float32),
            dims=['Type', 'lm', 'cell', 'lu'],
            coords={
                'Type': list(ghg_t_dict.keys()),
                'lm': data.LANDMANS,
                'cell': range(data.NCELLS),
                'lu': data.AGRICULTURAL_LANDUSES
            }
        )
        ghg_t_smrj = ghg_t_smrj / gap

        xr_ghg_transition = ghg_t_smrj * ag_dvar_mrj
        xr_ghg_transition = add_all(xr_ghg_transition, ['lm', 'Type'])

        ghg_trans_df, ghg_trans_df_AUS = to_region_and_aus_df(xr_ghg_transition, ['Type', 'lm', 'lu'], yr_cal, region_levels=['region_state', 'region_NRM'])
        (ghg_trans_df
            .rename(columns={'Value': 'Value (t CO2e)', 'lu': 'Land-use', 'lm': 'Water_supply'})
            .infer_objects(copy=False)
            .replace({'dry': 'Dryland', 'irr': 'Irrigated'})
            .query('abs(`Value (t CO2e)`) > 1e-3')
            .to_csv(os.path.join(path, f'GHG_emissions_separate_transition_penalty_{yr_cal}.csv'), index=False))

        valid_transition_layers = pd.MultiIndex.from_frame(ghg_trans_df_AUS[['Type', 'lm', 'lu']]).sort_values()
        transition_valid_layers = xr_ghg_transition.stack(layer=['Type', 'lm', 'lu']).sel(layer=valid_transition_layers).drop_vars(['region_state', 'region_NRM'], errors='ignore').compute()
        save2nc(transition_valid_layers, os.path.join(path, f'xr_transition_GHG_{yr_cal}.nc'))
        transition_magnitudes = get_mag(transition_valid_layers)

    # ==================== Sum (Ag + Am + NonAg + Transition) ====================

    # Ag: sum over GHG_source to remove source dim → (lm=['dry','irr'], lu, cell)
    ag_lus = [l for l in ghg_e.coords['lu'].values if l != 'ALL']
    ghg_sources = [s for s in ghg_e.coords['GHG_source'].values if s != 'ALL']
    ghg_pre_ag = ghg_e.sel(lm=['dry', 'irr'], GHG_source=ghg_sources, lu=ag_lus).sum('GHG_source')

    # Am: sum over am dim → (lm=['dry','irr'], lu, cell); same lu coords as Ag
    am_types = [a for a in xr_ghg_ag_man.coords['am'].values if a != 'ALL']
    ghg_pre_am = (xr_ghg_ag_man.sel(am=am_types, lm=['dry', 'irr'], lu=ag_lus).sum('am')
                  if am_types else xr.zeros_like(ghg_pre_ag))

    # NonAg: no water dim — assign lm='dry', fill lm='irr' with 0 → (lm=['dry','irr'], lu, cell)
    non_ag_lus = [l for l in xr_ghg_non_ag.coords['lu'].values if l != 'ALL']
    ghg_pre_nonag = (xr_ghg_non_ag.sel(lu=non_ag_lus)
                     .expand_dims('lm').assign_coords(lm=['dry'])
                     .reindex(lm=['dry', 'irr'], fill_value=0))

    # Transition: dims are (Type, lm, cell, lu) — lm/lu are already destination water/lu.
    # Sum over Type (other dim) → (lm=['dry','irr'], lu, cell). Zero for base year.
    if yr_cal != data.YR_CAL_BASE:
        trans_types = [t for t in xr_ghg_transition.coords['Type'].values if t != 'ALL']
        ghg_pre_transition = xr_ghg_transition.sel(Type=trans_types, lm=['dry', 'irr'], lu=ag_lus).sum('Type')
    else:
        ghg_pre_transition = xr.zeros_like(ghg_pre_ag)

    # Build by Type (ag / ag-man / non-ag), fold transition into ag, then add ALL
    sum_ghg_ag = (ghg_pre_ag + ghg_pre_transition).sum(['lm', 'lu']).expand_dims({'Type': ['ag']})
    sum_ghg_am = ghg_pre_am.sum(['lm', 'lu']).expand_dims({'Type': ['ag-man']})
    sum_ghg_nonag = ghg_pre_nonag.sum(['lm', 'lu']).expand_dims({'Type': ['non-ag']})

    sum_ghg = xr.concat([sum_ghg_ag, sum_ghg_am, sum_ghg_nonag], dim='Type')
    sum_ghg = add_all(sum_ghg, dims=['Type'])

    sum_ghg_stack = sum_ghg.stack(layer=['Type']).drop_vars(['region_state', 'region_NRM'], errors='ignore').compute()
    save2nc(sum_ghg_stack, os.path.join(path, f'xr_GHG_sum_{yr_cal}.nc'))

    magnitudes = {
        'ghg_emission': {
            'ag':         get_mag(valid_layers_stack_ghg),
            'non_ag':     get_mag(xr_ghg_non_ag_cat),
            'ag_man':     get_mag(valid_layers_stack_am_ghg),
            'transition': transition_magnitudes,
            'sum':        get_mag(sum_ghg_stack),
        }
    }
    return (f"GHG emissions written for year {yr_cal}", magnitudes)





# ── Water ────────────────────────────────────────────────────────────────────

def write_water(data: Data, yr_cal, path):
    """No annualisation needed: `get_water_net_yield_matrices()` and the non-ag/ag-man
    water matrices are already annual yield rates (ML/year) for `yr_cal`, independent
    of the gap to the previous simulated year — dividing by `gap` would double-annualise.

    Water yield is written to disk no matter if `WATER_LIMITS` is on or off. """

    yr_idx = yr_cal - data.YR_CAL_BASE
    region2code = {v: k for k, v in data.WATER_REGION_NAMES.items()}

    # Get the decision variables
    ag_dvar_mrj = chunk_unify_size(tools.ag_mrj_to_xr(data, data.ag_dvars[yr_cal])
        ).assign_coords(region_water=('cell', data.WATER_REGION_ID), region_NRM=('cell', data.REGION_NRM_NAME))
    non_ag_dvar_rj = chunk_unify_size(tools.non_ag_rk_to_xr(data, data.non_ag_dvars[yr_cal])
        ).assign_coords(region_water=('cell', data.WATER_REGION_ID), region_NRM=('cell', data.REGION_NRM_NAME))
    am_dvar_mrj = chunk_unify_size(tools.am_mrj_to_xr(data, data.ag_man_dvars[yr_cal])
        ).assign_coords(region_water=('cell', data.WATER_REGION_ID), region_NRM=('cell', data.REGION_NRM_NAME))
        
    # Get water target and domestic use
    w_limit_inside_luto = xr.DataArray(
        np.array(list(data.WATER_YIELD_TARGETS.values()), dtype=np.float32),
        dims=['region_water'],
        coords={'region_water': list(data.WATER_YIELD_TARGETS.keys())}
    )
    domestic_water_use = xr.DataArray(
        np.array(list(data.WATER_USE_DOMESTIC.values()), dtype=np.float32),
        dims=['region_water'],
        coords={'region_water': list(data.WATER_USE_DOMESTIC.keys())}
    )

    # ==================== Get Water Yield without CCI ====================

    # Get water yield matrix
    if settings.WATER_CLIMATE_CHANGE_IMPACT == 'on':
        ag_w_mrj = tools.ag_mrj_to_xr(
            data,
            ag_water.get_water_net_yield_matrices(data, yr_idx)
        )
        non_ag_w_rk = tools.non_ag_rk_to_xr(
            data,
            non_ag_water.get_w_net_yield_matrix(data, ag_w_mrj.values, data.lumaps[yr_cal], yr_idx)
        )
        ag_man_w_mrj = tools.am_mrj_to_xr(  # Ag-man water yield only related to water requirement, that not affected by climate change
            data,
            ag_water.get_agricultural_management_water_matrices(data, yr_idx)
        )
    elif settings.WATER_CLIMATE_CHANGE_IMPACT == 'off':
        ag_w_mrj = tools.ag_mrj_to_xr(
            data,
            ag_water.get_water_net_yield_matrices(data, yr_idx, data.WATER_YIELD_HIST_DR, data.WATER_YIELD_HIST_SR)
        )
        non_ag_w_rk = tools.non_ag_rk_to_xr(
            data,
            non_ag_water.get_w_net_yield_matrix(data, ag_w_mrj.values, data.lumaps[yr_cal], yr_idx, data.WATER_YIELD_HIST_DR, data.WATER_YIELD_HIST_SR)
        )
        ag_man_w_mrj = tools.am_mrj_to_xr(  # Ag-man water yield only related to water requirement, that not affected by climate change
            data,
            ag_water.get_agricultural_management_water_matrices(data, yr_idx)
        )
    else:
        raise ValueError("Invalid setting for WATER_CLIMATE_CHANGE_IMPACT, only 'on' or 'off' allowed.")

    # Calculate water net yield inside LUTO study region
    xr_ag_wny = ag_dvar_mrj * ag_w_mrj
    xr_non_ag_wny = non_ag_dvar_rj * non_ag_w_rk
    xr_am_wny = ag_man_w_mrj * am_dvar_mrj

    xr_ag_wny     = add_all(xr_ag_wny,     ['lm', 'lu'])
    xr_non_ag_wny = add_all(xr_non_ag_wny, ['lu'])
    xr_am_wny     = add_all(xr_am_wny,     ['lm', 'lu', 'am'])

    ag_wny = xr_ag_wny.groupby('region_water'
        ).sum(['cell']
        ).to_dataframe('Water Net Yield (ML)'
        ).reset_index(
        ).assign(Type='Agricultural Land-use'
        ).infer_objects(copy=False
        ).replace({'region_water': data.WATER_REGION_NAMES})
    non_ag_wny = xr_non_ag_wny.groupby('region_water'
        ).sum(['cell']
        ).to_dataframe('Water Net Yield (ML)'
        ).reset_index(
        ).assign(Type='Non-Agricultural Land-use', lm='dry'
        ).infer_objects(copy=False
        ).replace({'region_water': data.WATER_REGION_NAMES})
    am_wny = xr_am_wny.groupby('region_water'
        ).sum(['cell']
        ).to_dataframe('Water Net Yield (ML)'
        ).reset_index(
        ).assign(Type='Agricultural Management'
        ).infer_objects(copy=False
        ).replace({'region_water': data.WATER_REGION_NAMES})
    wny_inside_luto = pd.concat([ag_wny, non_ag_wny, am_wny], ignore_index=True
        ).assign(Year=yr_cal
        ).rename(columns={
            'region_water': 'Region',
            'lu':'Landuse',
            'am':'Agricultural Management',
            'lm':'Water Supply'}
        ).infer_objects(copy=False
        ).replace({'dry':'Dryland', 'irr':'Irrigated'}
        ).dropna(axis=0, how='all')

    wny_inside_luto.to_csv(os.path.join(path, f'water_yield_separate_watershed_{yr_cal}.csv'), index=False)


    # ==================== Get Water Yield outside LUTO Study Region ====================
    wny_outside_luto_study_area = xr.DataArray(
        np.array(list(data.WATER_OUTSIDE_LUTO_BY_CCI.loc[data.YR_CAL_BASE].to_dict().values()), dtype=np.float32),
        dims=['region_water'],
        coords={'region_water': list(data.WATER_REGION_INDEX_R.keys())},
    )


    # ==================== Get Water Yield Change (delta) under CCI ====================

    # Get CCI matrix
    if settings.WATER_CLIMATE_CHANGE_IMPACT == 'on':
        ag_w_mrj_base = tools.ag_mrj_to_xr(data, ag_water.get_water_net_yield_matrices(data, 0))
        ag_w_mrj_base = add_all(ag_w_mrj_base, ['lm'])
        wny_outside_luto_study_area_base = np.array(list(data.WATER_OUTSIDE_LUTO_BY_CCI.loc[data.YR_CAL_BASE].to_dict().values()))
    elif settings.WATER_CLIMATE_CHANGE_IMPACT == 'off':
        ag_w_mrj_base = tools.ag_mrj_to_xr(data, ag_water.get_water_net_yield_matrices(data, 0, data.WATER_YIELD_HIST_DR, data.WATER_YIELD_HIST_SR))
        ag_w_mrj_base = add_all(ag_w_mrj_base, ['lm'])
        wny_outside_luto_study_area_base = np.array(list(data.WATER_OUTSIDE_LUTO_HIST.values()))

    ag_w_mrj_CCI = ag_w_mrj - ag_w_mrj_base
    wny_outside_luto_study_area_CCI = wny_outside_luto_study_area - wny_outside_luto_study_area_base



    # Calculate water net yield (delta) under CCI; 
    #   we use BASE_YEAR (2010) dvar_mrj to calculate CCI, 
    #   because the CCI calculated with base year (previouse year) 
    #   dvar_mrj includes wny From-land-use change
    xr_ag_dvar_BASE = tools.ag_mrj_to_xr(data, data.AG_L_MRJ).assign_coords(region_water=('cell', data.WATER_REGION_ID), region_NRM=('cell', data.REGION_NRM_NAME))
    xr_ag_dvar_BASE = add_all(xr_ag_dvar_BASE, ['lm'])

    xr_ag_wny_CCI = xr_ag_dvar_BASE * ag_w_mrj_CCI


    # Get the CCI impact (delta)
    CCI_impact = (
        xr_ag_wny_CCI.groupby('region_water').sum(['cell','lm', 'lu']) 
        + wny_outside_luto_study_area_CCI
    )

    # ==================== Organise Water Yield Components ====================

    # Water net yield for watershed regions
    wny_inside_luto_sum = wny_inside_luto\
        .query('`Water Supply` != "ALL" and `Agricultural Management` != "ALL"')\
        .groupby('Region')[['Water Net Yield (ML)']]\
        .sum()
    wny_inside_luto_sum = xr.DataArray(
        wny_inside_luto_sum['Water Net Yield (ML)'].values.astype(np.float32),
        dims=['region_water'],
        coords={'region_water': [region2code[i] for i in wny_inside_luto_sum.index.values]}
    )
    wny_watershed_sum = wny_inside_luto_sum + wny_outside_luto_study_area - domestic_water_use  # CCI delta already include in the wny_inside_luto_sum

    w_limit_region = w_limit_inside_luto + wny_outside_luto_study_area - domestic_water_use     # CCI delta already include in the w_limit_inside_luto

    water_other_records = xr.Dataset(
            {   
                'Water yield inside LUTO (ML)': wny_inside_luto_sum,
                'Water yield outside LUTO (ML)': wny_outside_luto_study_area,
                'Climate Change Impact (ML)': CCI_impact,
                'Domestic Water Use (ML)': domestic_water_use,
                'Water Net Yield (ML)': wny_watershed_sum,
                'Water Yield Limit (ML)': w_limit_region,
            },
        ).to_dataframe(
        ).reset_index(
        ).rename(columns={'region_water': 'Region'}
        ).infer_objects(copy=False
        ).replace({'Region': data.WATER_REGION_NAMES}
        ).assign(Year=yr_cal)
        
    water_other_records.to_csv(os.path.join(path, f'water_yield_limits_and_public_land_{yr_cal}.csv'), index=False)

    # Water yield for NRM region (use add_all'd arrays so Water Supply="ALL" is included)
    ag_wny = xr_ag_wny.groupby('region_NRM'
        ).sum(['cell']
        ).to_dataframe('Water Net Yield (ML)'
        ).reset_index(
        ).assign(Type='Agricultural Land-use'
        ).infer_objects(copy=False
        ).replace({'region_NRM': data.WATER_REGION_NAMES})
    non_ag_wny = xr_non_ag_wny.groupby('region_NRM'
        ).sum(['cell']
        ).to_dataframe('Water Net Yield (ML)'
        ).reset_index(
        ).assign(Type='Non-Agricultural Land-use', lm='dry'
        ).infer_objects(copy=False
        ).replace({'region_NRM': data.WATER_REGION_NAMES})
    am_wny = xr_am_wny.groupby('region_NRM'
        ).sum(['cell']
        ).to_dataframe('Water Net Yield (ML)'
        ).reset_index(
        ).assign(Type='Agricultural Management'
        ).infer_objects(copy=False
        ).replace({'region_NRM': data.WATER_REGION_NAMES})
    wny_NRM = pd.concat([ag_wny, non_ag_wny, am_wny], ignore_index=True
        ).assign(Year=yr_cal
        ).rename(columns={
            'region_water': 'Region',
            'lu':'Landuse',
            'am':'Agricultural Management',
            'lm':'Water Supply'}
        ).infer_objects(copy=False
        ).replace({'dry':'Dryland', 'irr':'Irrigated'}
        ).dropna(axis=0, how='all')

    wny_NRM.to_csv(os.path.join(path, f'water_yield_separate_NRM_{yr_cal}.csv'), index=False)


    xr_ag_wny_cat = xr_ag_wny.stack(layer=['lm', 'lu']).drop_vars(['region_water', 'region_NRM']).compute()
    xr_non_ag_wny_cat = xr_non_ag_wny.stack(layer=['lu']).drop_vars(['region_water', 'region_NRM']).compute()
    xr_am_wny_cat = xr_am_wny.stack(layer=['am', 'lm', 'lu']).drop_vars(['region_water', 'region_NRM']).compute()

    save2nc(xr_ag_wny_cat, os.path.join(path, f'xr_water_yield_ag_{yr_cal}.nc'))

    save2nc(xr_non_ag_wny_cat, os.path.join(path, f'xr_water_yield_non_ag_{yr_cal}.nc'))
    save2nc(xr_am_wny_cat, os.path.join(path, f'xr_water_yield_ag_management_{yr_cal}.nc'))

    # --- Sum water yield (Ag + Am + NonAg) ---
    ag_lus = [lu for lu in xr_ag_wny.coords['lu'].values if lu != 'ALL']
    nonag_lus = [lu for lu in xr_non_ag_wny.coords['lu'].values if lu != 'ALL']
    am_non_all = [am for am in xr_am_wny.coords['am'].values if am != 'ALL']

    raw_wny_ag = xr_ag_wny.sel(lm=['dry', 'irr'], lu=ag_lus)
    am_sum_wny = xr_am_wny.sel(am=am_non_all, lm=['dry', 'irr']).sum('am').sel(lu=ag_lus)
    nonag_as_dry = (
        xr_non_ag_wny.sel(lu=nonag_lus)
        .expand_dims('lm').assign_coords(lm=['dry'])
        .reindex(lm=['dry', 'irr'], fill_value=0)
    )

    # Build by Type (ag / ag-man / non-ag) then add ALL
    sum_wny_ag = raw_wny_ag.sum(['lm', 'lu']).expand_dims({'Type': ['ag']})
    sum_wny_am = am_sum_wny.sum(['lm', 'lu']).expand_dims({'Type': ['ag-man']})
    sum_wny_nonag = xr_non_ag_wny.sel(lu=nonag_lus).sum('lu').expand_dims({'Type': ['non-ag']})

    sum_wny = xr.concat([sum_wny_ag, sum_wny_am, sum_wny_nonag], dim='Type')
    sum_wny = add_all(sum_wny, dims=['Type'])

    xr_sum_wny_cat = sum_wny.stack(layer=['Type']).drop_vars(['region_water', 'region_NRM']).compute()
    save2nc(xr_sum_wny_cat, os.path.join(path, f'xr_water_yield_sum_{yr_cal}.nc'))


    # ==================== Write Original Targets for Relaxed Watershed Regions ====================
    water_relaxed_region_raw_targets = pd.DataFrame(
        [[k, v, data.WATER_REGION_NAMES[k]] for k, v in data.WATER_RELAXED_REGION_RAW_TARGETS.items()], 
        columns=['Region Id', 'Target', 'Region Name']
    )
    water_relaxed_region_raw_targets['Year'] = yr_cal
    water_relaxed_region_raw_targets.to_csv(os.path.join(path, f'water_yield_relaxed_region_raw_{yr_cal}.csv'), index=False)

    return (
        f"Water yield data written for year {yr_cal}",
        {
            'water_yield': {
                'ag': (xr_ag_wny_cat.min().item(), xr_ag_wny_cat.max().item()),
                'non_ag': (xr_non_ag_wny_cat.min().item(), xr_non_ag_wny_cat.max().item()),
                'am': (xr_am_wny_cat.min().item(), xr_am_wny_cat.max().item()),
                'sum': (xr_sum_wny_cat.min().item(), xr_sum_wny_cat.max().item()),
            }
        }
    )



# ── Biodiversity ─────────────────────────────────────────────────────────────

def write_biodiversity_quality_scores(data: Data, yr_cal, path):
    ''' No annualisation needed: biodiversity quality scores are a point-in-time
    snapshot (stock) derived from the `yr_cal` lumap, not a flow over `gap` years.

    Biodiversity overall quality scores — computed for every BIO_QUALITY_LAYER backend. '''

    # Invariant setup
    yr_idx_previous = sorted(data.lumaps.keys()).index(yr_cal) - 1
    yr_cal_previous = sorted(data.lumaps.keys())[yr_idx_previous]
    yr_idx = yr_cal - data.YR_CAL_BASE

    # Decision variables are the same for every backend layer — load once
    ag_dvar_mrj = chunk_unify_size(tools.ag_mrj_to_xr(data, data.ag_dvars[yr_cal])
        ).assign_coords(region_state=('cell', data.REGION_STATE_NAME), region_NRM=('cell', data.REGION_NRM_NAME))
    ag_mam_dvar_mrj = chunk_unify_size(tools.am_mrj_to_xr(data, data.ag_man_dvars[yr_cal])
        ).assign_coords(region_state=('cell', data.REGION_STATE_NAME), region_NRM=('cell', data.REGION_NRM_NAME))
    non_ag_dvar_rk = chunk_unify_size(tools.non_ag_rk_to_xr(data, data.non_ag_dvars[yr_cal])
        ).assign_coords(region_state=('cell', data.REGION_STATE_NAME), region_NRM=('cell', data.REGION_NRM_NAME))

    # Per-backend accumulators
    all_ag_dfs, all_non_ag_dfs, all_am_dfs, all_priority_all_dfs = [], [], [], []
    all_ag_dfs_AUS, all_non_ag_dfs_AUS, all_am_dfs_AUS, all_priority_all_dfs_AUS = [], [], [], []
    xr_per_backend_ag, xr_per_backend_non_ag, xr_per_backend_am, xr_per_backend_all = [], [], [], []

    for backend_layer in settings.BIO_QUALITY_LAYERS:
        # Load raw biodiversity arrays for this backend
        bio_quality_raw, bio_quality_lds = data.compute_bio_quality_arrays(backend_layer)

        # Compute biodiversity score matrices
        bio_ag_priority_mrj = tools.ag_mrj_to_xr(
            data, ag_biodiversity.get_bio_quality_score_mrj(data, bio_quality_raw, bio_quality_lds)
        )
        bio_am_priority_amrj = tools.am_mrj_to_xr(
            data, ag_biodiversity.get_ag_mgt_biodiversity_matrices(
                data, bio_ag_priority_mrj.values, yr_idx, bio_quality_raw
            )
        )
        bio_non_ag_priority_rk = tools.non_ag_rk_to_xr(
            data, non_ag_biodiversity.get_breq_matrix(
                data, bio_ag_priority_mrj.values, data.lumaps[yr_cal_previous], bio_quality_raw
            )
        )

        if yr_cal == data.YR_CAL_BASE:
            bio_am_priority_amrj *= 0.0
            bio_non_ag_priority_rk *= 0.0

        # Base-year reference score normalises percentages for this backend
        base_yr_score = bio_ag_priority_mrj.sel(lu='Unallocated - natural land', lm='dry').sum().item()

        # Weighted biodiversity scores
        xr_priority_ag = ag_dvar_mrj * bio_ag_priority_mrj
        xr_priority_non_ag = non_ag_dvar_rk * bio_non_ag_priority_rk
        xr_priority_am = ag_mam_dvar_mrj * bio_am_priority_amrj
        xr_priority_all = xr.concat(
            [
                xr_priority_ag.sum(dim=['lm', 'lu']).expand_dims({'Type': ['ag']}),
                xr_priority_non_ag.sum(dim=['lu']).expand_dims({'Type': ['non-ag']}),
                xr_priority_am.sum(dim=['am', 'lu', 'lm']).expand_dims({'Type': ['ag-man']}),
            ],
            dim='Type',
        )

        xr_priority_ag = add_all(xr_priority_ag, dims=['lm', 'lu'])
        xr_priority_non_ag = add_all(xr_priority_non_ag, dims=['lu'])
        xr_priority_am = add_all(xr_priority_am, dims=['am', 'lm', 'lu'])
        xr_priority_all = add_all(xr_priority_all, dims=['Type'])

        # Aggregate to DataFrames
        ag_df, ag_df_AUS = bio_to_region_and_aus_df(
            xr_priority_ag, ['lm', 'lu'], 'Area Weighted Score (ha)', base_yr_score, yr_cal, region_levels=['region_state', 'region_NRM']
        )
        non_ag_df, non_ag_df_AUS = bio_to_region_and_aus_df(
            xr_priority_non_ag, ['lu'], 'Area Weighted Score (ha)', base_yr_score, yr_cal, region_levels=['region_state', 'region_NRM']
        )
        am_df, am_df_AUS = bio_to_region_and_aus_df(
            xr_priority_am, ['am', 'lm', 'lu'], 'Area Weighted Score (ha)', base_yr_score, yr_cal, region_levels=['region_state', 'region_NRM']
        )
        priority_all_df, priority_all_df_AUS = bio_to_region_and_aus_df(
            xr_priority_all, ['Type'], 'Area Weighted Score (ha)', base_yr_score, yr_cal, region_levels=['region_state', 'region_NRM']
        )

        # Fill empty DataFrames
        if ag_df.empty:
            ag_df = pd.DataFrame({
                'region': ['AUSTRALIA'], 'Year': [yr_cal], 'lm': ['dry'],
                'lu': ['Unallocated - natural land'],
                'Area Weighted Score (ha)': [0.0], 'Relative_Contribution_Percentage': [0.0],
            })
        if non_ag_df.empty:
            non_ag_df = pd.DataFrame({
                'region': ['AUSTRALIA'], 'Year': [yr_cal], 'lu': ['Environmental Plantings'],
                'lm': ['dry'], 'Area Weighted Score (ha)': [0.0], 'Relative_Contribution_Percentage': [0.0],
            })
        if am_df.empty:
            am_df = pd.DataFrame({
                'region': ['AUSTRALIA', 'AUSTRALIA'], 'Year': [yr_cal, yr_cal],
                'am': ['ALL', 'Savanna Burning'], 'lm': ['dry', 'dry'], 'lu': ['Apples', 'Apples'],
                'Area Weighted Score (ha)': [0.0, 0.0], 'Relative_Contribution_Percentage': [0.0, 0.0],
            })

        # Tag backend
        for df in [ag_df, non_ag_df, am_df, priority_all_df]:
            df['backend'] = backend_layer

        all_ag_dfs.append(ag_df)
        all_non_ag_dfs.append(non_ag_df)
        all_am_dfs.append(am_df)
        all_priority_all_dfs.append(priority_all_df)
        all_ag_dfs_AUS.append(ag_df_AUS.assign(backend=backend_layer))
        all_non_ag_dfs_AUS.append(non_ag_df_AUS.assign(backend=backend_layer))
        all_am_dfs_AUS.append(am_df_AUS.assign(backend=backend_layer))
        all_priority_all_dfs_AUS.append(priority_all_df_AUS.assign(backend=backend_layer))

        # Collect per-backend xr arrays for NC output
        xr_per_backend_ag.append(xr_priority_ag.expand_dims({'backend': [backend_layer]}))
        xr_per_backend_non_ag.append(xr_priority_non_ag.expand_dims({'backend': [backend_layer]}))
        xr_per_backend_am.append(xr_priority_am.expand_dims({'backend': [backend_layer]}))
        xr_per_backend_all.append(xr_priority_all.expand_dims({'backend': [backend_layer]}))

    # ── Combine DataFrames and write CSVs ────────────────────────────────────
    combined_ag_df     = pd.concat(all_ag_dfs, axis=0)
    combined_non_ag_df = pd.concat(all_non_ag_dfs, axis=0)
    combined_am_df     = pd.concat(all_am_dfs, axis=0)
    combined_all_df    = pd.concat(all_priority_all_dfs, axis=0)

    pd.concat([
        combined_ag_df.assign(Type='Agricultural Land-use'),
        combined_non_ag_df.assign(Type='Non-Agricultural Land-use'),
        combined_am_df.assign(Type='Agricultural Management'),
    ], axis=0
    ).rename(columns={
        'lu': 'Landuse',
        'lm': 'Water_supply',
        'am': 'Agricultural Management',
        'Relative_Contribution_Percentage': 'Contribution Relative to Base Year Level (%)',
    }).reset_index(drop=True
    ).infer_objects(copy=False
    ).replace({'dry': 'Dryland', 'irr': 'Irrigated'}
    ).to_csv(os.path.join(path, f'biodiversity_overall_priority_scores_{yr_cal}.csv'), index=False)

    combined_all_df.rename(columns={
        'lu': 'Landuse',
        'lm': 'Water_supply',
        'am': 'Agricultural Management',
        'Relative_Contribution_Percentage': 'Contribution Relative to Base Year Level (%)',
    }).reset_index(drop=True
    ).infer_objects(copy=False
    ).replace({'dry': 'Dryland', 'irr': 'Irrigated'}
    ).to_csv(os.path.join(path, f'biodiversity_overall_priority_scores_all_{yr_cal}.csv'), index=False)

    # ── Build combined xr arrays (backend dim) and write NC ──────────────────
    combined_ag_df_AUS     = pd.concat(all_ag_dfs_AUS, axis=0)
    combined_non_ag_df_AUS = pd.concat(all_non_ag_dfs_AUS, axis=0)
    combined_am_df_AUS     = pd.concat(all_am_dfs_AUS, axis=0)
    combined_all_df_AUS    = pd.concat(all_priority_all_dfs_AUS, axis=0)

    xr_combined_ag     = xr.concat(xr_per_backend_ag, dim='backend')
    xr_combined_non_ag = xr.concat(xr_per_backend_non_ag, dim='backend')
    xr_combined_am     = xr.concat(xr_per_backend_am, dim='backend')
    xr_combined_all    = xr.concat(xr_per_backend_all, dim='backend')

    # ==================== Ag Valid Layers ====================
    valid_ag_layers = pd.MultiIndex.from_frame(combined_ag_df_AUS[['backend', 'lm', 'lu']]).sort_values()
    valid_layers_stack_ag = (
        xr_combined_ag.stack(layer=['backend', 'lm', 'lu'])
        .sel(layer=valid_ag_layers)
        .drop_vars(['region_state', 'region_NRM'], errors='ignore')
        .compute()
    )

    # ==================== Non-Ag Valid Layers ====================
    valid_non_ag_layers = pd.MultiIndex.from_frame(combined_non_ag_df_AUS[['backend', 'lu']]).sort_values()
    if combined_non_ag_df_AUS['Area Weighted Score (ha)'].abs().sum() < 1:
        valid_layers_stack_non_ag = xr.DataArray(
            np.zeros((1, 1, data.NCELLS), dtype=np.float32),
            dims=['backend', 'lu', 'cell'],
            coords={
                'backend': [settings.BIO_QUALITY_LAYERS[0]],
                'lu': ['ALL'],
                'cell': range(data.NCELLS),
            },
        ).stack(layer=['backend', 'lu'])
    else:
        valid_layers_stack_non_ag = (
            xr_combined_non_ag.stack(layer=['backend', 'lu'])
            .sel(layer=valid_non_ag_layers)
            .drop_vars(['region_state', 'region_NRM'], errors='ignore')
            .compute()
        )

    # ==================== Ag Management Valid Layers ====================
    valid_am_layers = pd.MultiIndex.from_frame(combined_am_df_AUS[['backend', 'am', 'lm', 'lu']]).sort_values()
    if combined_am_df_AUS['Area Weighted Score (ha)'].abs().sum() < 1:
        valid_layers_stack_am = xr.DataArray(
            np.zeros((1, 1, 1, 1, data.NCELLS), dtype=np.float32),
            dims=['backend', 'am', 'lm', 'lu', 'cell'],
            coords={
                'backend': [settings.BIO_QUALITY_LAYERS[0]],
                'am': ['ALL'],
                'lm': ['ALL'],
                'lu': ['ALL'],
                'cell': range(data.NCELLS),
            },
        ).stack(layer=['backend', 'am', 'lm', 'lu'])
    else:
        valid_layers_stack_am = (
            xr_combined_am.stack(layer=['backend', 'am', 'lm', 'lu'])
            .sel(layer=valid_am_layers)
            .drop_vars(['region_state', 'region_NRM'], errors='ignore')
            .compute()
        )

    # ==================== All-Type Valid Layers ====================
    valid_all_layers = pd.MultiIndex.from_frame(combined_all_df_AUS[['backend', 'Type']]).sort_values()
    valid_layers_stack_all = (
        xr_combined_all.stack(layer=['backend', 'Type'])
        .sel(layer=valid_all_layers)
        .drop_vars(['region_state', 'region_NRM'], errors='ignore')
        .compute()
    )

    save2nc(valid_layers_stack_ag, os.path.join(path, f'xr_biodiversity_overall_priority_ag_{yr_cal}.nc'))

    save2nc(valid_layers_stack_non_ag, os.path.join(path, f'xr_biodiversity_overall_priority_non_ag_{yr_cal}.nc'))
    save2nc(valid_layers_stack_am, os.path.join(path, f'xr_biodiversity_overall_priority_ag_management_{yr_cal}.nc'))
    save2nc(valid_layers_stack_all, os.path.join(path, f'xr_biodiversity_overall_priority_all_{yr_cal}.nc'))

    magnitudes = {
        'bio_quality': {
            'ag':     get_mag(valid_layers_stack_ag),
            'non_ag': get_mag(valid_layers_stack_non_ag),
            'am':     get_mag(valid_layers_stack_am),
            'all':    get_mag(valid_layers_stack_all),
        }
    }
    return (f"Biodiversity overall priority scores written for year {yr_cal}", magnitudes)



def write_biodiversity_GBF2_scores(data: Data, yr_cal, path):
    ''' No annualisation needed: GBF2 scores are a point-in-time snapshot (stock)
    derived from the `yr_cal` lumap, not a flow over `gap` years.

    Biodiversity GBF2 only being written to disk when `GBF2_TARGET` is not 'off' '''

    # Do nothing if biodiversity limits are off and no need to report
    if settings.GBF2_TARGET == 'off':
        return 'Skipped: Biodiversity GBF2 scores not written as `GBF2_TARGET` is set to "off"'

        
    # Unpack the ag managements and land uses
    am_lu_unpack = [(am, l) for am, lus in data.AG_MAN_LU_DESC.items() for l in lus]

    # Get decision variables for the year
    ag_dvar_mrj = chunk_unify_size(
        tools.ag_mrj_to_xr(data, data.ag_dvars[yr_cal])
    ).assign_coords(region_state=('cell', data.REGION_STATE_NAME), region_NRM=('cell', data.REGION_NRM_NAME))
    non_ag_dvar_rk = chunk_unify_size(
        tools.non_ag_rk_to_xr(data, data.non_ag_dvars[yr_cal])
    ).assign_coords(region_state=('cell', data.REGION_STATE_NAME), region_NRM=('cell', data.REGION_NRM_NAME))
    am_dvar_amrj = chunk_unify_size(
        tools.am_mrj_to_xr(data, data.ag_man_dvars[yr_cal])
    ).assign_coords(region_state=('cell', data.REGION_STATE_NAME), region_NRM=('cell', data.REGION_NRM_NAME))

    # Get the priority degraded areas 
    GBF2_MASK_area_ha = ag_biodiversity.get_GBF2_MASK_area(data)
    
    # Get the priority degraded areas score
    priority_degraded_area_score_r = xr.DataArray(
        GBF2_MASK_area_ha.astype(np.float32),
        dims=['cell'],
        coords={'cell':range(data.NCELLS)}
    )

    # Get the impacts of each ag/non-ag/am to vegetation matrices
    ag_impact_j = xr.DataArray(
        ag_biodiversity.get_ag_biodiversity_contribution(data).astype(np.float32),
        dims=['lu'],
        coords={'lu':data.AGRICULTURAL_LANDUSES}
    )
    non_ag_impact_k = xr.DataArray(
        np.array(list(non_ag_biodiversity.get_non_ag_lu_biodiv_contribution(data).values()), dtype=np.float32),
        dims=['lu'],
        coords={'lu':data.NON_AGRICULTURAL_LANDUSES}
    )
    am_impact_ajr = xr.DataArray(
        np.stack([
            arr.astype(np.float32)
            for v in ag_biodiversity.get_ag_management_biodiversity_contribution(data, yr_cal).values()
            for arr in v.values()
        ]),
        dims=['idx', 'cell'],
        coords={
            'idx': pd.MultiIndex.from_tuples(am_lu_unpack, names=['am', 'lu']),
            'cell': range(data.NCELLS)}
    ).unstack()

    # Masked base-year ag dvar (consistent masking with ag_dvar_mrj)
    BASEYEAR_dvar = chunk_unify_size(
        tools.ag_mrj_to_xr(data, data.AG_L_MRJ)
    ).assign_coords(region_state=('cell', data.REGION_STATE_NAME), region_NRM=('cell', data.REGION_NRM_NAME))

    # Denominator: total degraded area score at base year, consistent with solver target formula.
    # = (BIO_GBF2_MASK * REAL_AREA * AG_MASK_PROPORTION_R).sum() - BIO_GBF2_BASE_YR.sum()
    degreded_area_weighted_bio_contr = (
        float((data.BIO_GBF2_MASK * data.REAL_AREA * data.AG_MASK_PROPORTION_R).sum())
        - float(data.BIO_GBF2_BASE_YR.sum())
    )

    # Savanna-burning correction: LDS-treated cells have a lower effective base-year bio contribution,
    # so their restoration contribution must be increased by (1 - BIO_CONTRIBUTION_LDS) * base_dvar.
    savburn_eligible_r = xr.DataArray(
        data.SAVBURN_ELIGIBLE.astype(np.float32),
        dims=['cell'],
        coords={'cell': range(data.NCELLS)}
    ).assign_coords(region_NRM=('cell', data.REGION_NRM_NAME))

    # Calculate xarray biodiversity GBF2 scores;
    #   ag: (current - base) relative to the solver-consistent base year, including savburn correction.
    #   non-ag and am: absolute score (zero in base year 2010).
    xr_gbf2_ag      = priority_degraded_area_score_r * (
        ag_impact_j * (ag_dvar_mrj - BASEYEAR_dvar)
        + savburn_eligible_r * (1 - settings.BIO_CONTRIBUTION_LDS) * BASEYEAR_dvar
    )
    xr_gbf2_non_ag  = priority_degraded_area_score_r * non_ag_impact_k * non_ag_dvar_rk
    xr_gbf2_am      = priority_degraded_area_score_r * am_impact_ajr * am_dvar_amrj

    xr_gbf2_ag      = add_all(xr_gbf2_ag,     ['lm', 'lu'])
    xr_gbf2_non_ag  = add_all(xr_gbf2_non_ag, ['lu'])
    xr_gbf2_am      = add_all(xr_gbf2_am,     ['lm', 'lu', 'am'])

    GBF2_score_ag, GBF2_score_ag_AUS = bio_to_region_and_aus_df(
        xr_gbf2_ag, group_dims=['lm', 'lu'],
        value_name='Area Weighted Score (ha)', base_score=degreded_area_weighted_bio_contr, yr_cal=yr_cal, region_levels=['region_state', 'region_NRM'])
    GBF2_score_non_ag, GBF2_score_non_ag_AUS = bio_to_region_and_aus_df(
        xr_gbf2_non_ag, group_dims=['lu'],
        value_name='Area Weighted Score (ha)', base_score=degreded_area_weighted_bio_contr, yr_cal=yr_cal, region_levels=['region_state', 'region_NRM'])
    GBF2_score_am, GBF2_score_am_AUS = bio_to_region_and_aus_df(
        xr_gbf2_am, group_dims=['am', 'lm', 'lu'],
        value_name='Area Weighted Score (ha)', base_score=degreded_area_weighted_bio_contr, yr_cal=yr_cal, region_levels=['region_state', 'region_NRM'])

    GBF2_score_ag       = GBF2_score_ag.assign(Type='Agricultural Land-use')
    GBF2_score_non_ag   = GBF2_score_non_ag.assign(Type='Non-Agricultural Land-use')
    GBF2_score_am       = GBF2_score_am.assign(Type='Agricultural Management')
        
    # Fill nan to empty dataframes
    if GBF2_score_ag.empty:
        GBF2_score_ag = pd.DataFrame(
            {
                'region': ['AUSTRALIA', 'AUSTRALIA'], 
                'Year': [yr_cal, yr_cal], 
                'lm': ['dry', 'dry'], 
                'lu': ['ALL', 'Apples'], 
                'Area Weighted Score (ha)': [0.0, 0.0], 
                'Relative_Contribution_Percentage': [0.0, 0.0], 
                'Type': ['Agricultural Land-use', 'Agricultural Land-use']
            }
        )

    if GBF2_score_non_ag.empty:
        GBF2_score_non_ag = pd.DataFrame(
            {
                'region': ['AUSTRALIA', 'AUSTRALIA'], 
                'Year': [yr_cal, yr_cal], 
                'lm': ['dry', 'dry'], 
                'lu': ['ALL', 'Environmental Plantings'], 
                'Area Weighted Score (ha)': [0.0, 0.0], 
                'Relative_Contribution_Percentage': [0.0, 0.0], 
                'Type': ['Non-Agricultural Land-use', 'Non-Agricultural Land-use']
            }
        )
    if GBF2_score_am.empty:
        GBF2_score_am = pd.DataFrame(
            {
                'region': ['AUSTRALIA', 'AUSTRALIA'], 
                'Year': [yr_cal, yr_cal], 
                'am': ['ALL', 'Savanna Burning'],
                'lm': ['dry', 'dry'], 
                'lu': ['ALL', 'Apples'], 
                'Area Weighted Score (ha)': [0.0, 0.0], 
                'Relative_Contribution_Percentage': [0.0, 0.0], 
                'Type': ['Agricultural Management', 'Agricultural Management']
            }
        )
        
    # Save to disk  
    df = pd.concat([
            GBF2_score_ag,
            GBF2_score_non_ag,
            GBF2_score_am], axis=0
        ).rename(columns={
            'lu':'Landuse',
            'lm':'Water_supply',
            'am':'Agricultural Management',
            'Relative_Contribution_Percentage':'Contribution Relative to Pre-1750 Level (%)',
            'Priority_Target':'Priority Target (%)'}
        ).reset_index(drop=True
        ).infer_objects(copy=False
        ).replace({'dry':'Dryland', 'irr':'Irrigated'}
        )
    df.to_csv(os.path.join(path, f'biodiversity_GBF2_priority_scores_{yr_cal}.csv'), index=False)


    # ------------------------- Vectorize GBF2 mask to GeoJSON -------------------------
    
    geojson_js_path = f'{data.path}/DATA_REPORT/data/geo/biodiversity_GBF2_mask.js'
    os.makedirs(os.path.dirname(geojson_js_path), exist_ok=True)
    if not os.path.exists(geojson_js_path):
        mask_2d_da = arr_to_xr(data, data.BIO_GBF2_MASK)
        mask_2d_np = np.where(np.isnan(mask_2d_da.values), 0, mask_2d_da.values).astype(np.uint8)

        # Vectorize using rasterio.features.shapes with the model's CRS and transform
        transform = data.GEO_META['transform']
        crs = data.GEO_META['crs']
        pixel_area = abs(transform.a * transform.e)
        min_area = 10 * pixel_area  # drop isolated patches smaller than 10 pixels
        polygons = [
            shape(geom)
            for geom, val in rasterio.features.shapes(mask_2d_np, transform=transform)
            if val == 1 and shape(geom).area >= min_area
        ]

        # Dissolve, smooth in projected CRS (EPSG:3577 Australian Albers, metres), then simplify
        # smooth_d in metres: approx 2× the pixel width converted from degrees to metres
        smooth_d = abs(transform.a) * 111_000 * 2  # 1 degree ≈ 111 km
        gdf = gpd.GeoDataFrame(geometry=polygons, crs=crs).dissolve().to_crs('EPSG:3577')
        gdf['geometry'] = gdf.buffer(smooth_d).buffer(-smooth_d)
        gdf = gdf.to_crs('EPSG:4326')
        gdf['geometry'] = gdf.simplify(tolerance=0.05, preserve_topology=True)

        # Write as a JS window variable directly to VUE_modules
        geojson_dict = json.loads(gdf.to_json())
        with open(geojson_js_path, 'w', encoding='utf-8') as f:
            f.write(f'window.BIO_GBF2_MASK = {json.dumps(geojson_dict)};\n')



    # ------------------------- Stack array, get valid layers -------------------------

    # ---- Ag valid layers ----
    if GBF2_score_ag_AUS.empty:
        valid_layers_stack_ag = xr.DataArray(
            np.zeros((1, 1, data.NCELLS), dtype=np.float32),
            dims=['lm', 'lu', 'cell'],
            coords={'lm': ['ALL'], 'lu': ['ALL'], 'cell': range(data.NCELLS)}
        ).stack(layer=['lm', 'lu'])

    else:
        valid_ag_layers = pd.MultiIndex.from_frame(GBF2_score_ag_AUS[['lm', 'lu']]).sort_values()
        valid_layers_stack_ag = xr_gbf2_ag.stack(layer=['lm', 'lu']).sel(layer=valid_ag_layers).drop_vars(['region_state', 'region_NRM'], errors='ignore').compute()

    # ---- Non-ag valid layers ----
    valid_non_ag_layers = pd.MultiIndex.from_frame(GBF2_score_non_ag_AUS[['lu']]).sort_values()

    if GBF2_score_non_ag_AUS.empty:
        valid_layers_stack_non_ag = xr.DataArray(
            np.zeros((1, data.NCELLS), dtype=np.float32),
            dims=['lu', 'cell'],
            coords={'lu': ['ALL'], 'cell': range(data.NCELLS)}
        ).stack(layer=['lu'])

    else:
        valid_layers_stack_non_ag = xr_gbf2_non_ag.stack(layer=['lu']).sel(layer=valid_non_ag_layers).drop_vars(['region_state', 'region_NRM'], errors='ignore').compute()

    # ---- Ag management valid layers ----
    valid_am_layers = pd.MultiIndex.from_frame(GBF2_score_am_AUS[['am', 'lm', 'lu']]).sort_values()

    if GBF2_score_am_AUS.empty:
        valid_layers_stack_am = xr.DataArray(
            np.zeros((1, 1, 1, data.NCELLS), dtype=np.float32),
            dims=['am', 'lm', 'lu', 'cell'],
            coords={'am': ['ALL'], 'lm': ['ALL'], 'lu': ['ALL'], 'cell': range(data.NCELLS)}
        ).stack(layer=['am', 'lm', 'lu'])

    else:
        valid_layers_stack_am = xr_gbf2_am.stack(layer=['am', 'lm', 'lu']).sel(layer=valid_am_layers).drop_vars(['region_state', 'region_NRM'], errors='ignore').compute()

    # min/max should calculated using array without appending mosaic layers
    save2nc(valid_layers_stack_ag, os.path.join(path, f'xr_biodiversity_GBF2_priority_ag_{yr_cal}.nc'))
    save2nc(valid_layers_stack_non_ag, os.path.join(path, f'xr_biodiversity_GBF2_priority_non_ag_{yr_cal}.nc'))
    save2nc(valid_layers_stack_am, os.path.join(path, f'xr_biodiversity_GBF2_priority_ag_management_{yr_cal}.nc'))

    # --- Sum GBF2 (Ag + Am + NonAg) — Type dimension matches Quality Sum pattern ---
    xr_gbf2_all = xr.concat(
        [   xr_gbf2_ag.sum(dim=['lm', 'lu']).expand_dims({'Type': ['ag']}),
            xr_gbf2_non_ag.sum(dim=['lu']).expand_dims({'Type': ['non-ag']}),
            xr_gbf2_am.sum(dim=['am', 'lm', 'lu']).expand_dims({'Type': ['ag-man']})
        ], dim='Type'
    )
    xr_gbf2_all = add_all(xr_gbf2_all, dims=['Type'])
    xr_sum_gbf2_cat = xr_gbf2_all.stack(layer=['Type']).drop_vars(['region_state', 'region_NRM'], errors='ignore').compute()
    save2nc(xr_sum_gbf2_cat, os.path.join(path, f'xr_biodiversity_GBF2_priority_sum_{yr_cal}.nc'))

    magnitudes = {
        'biodiversity_GBF2': {
            'ag':     get_mag(valid_layers_stack_ag),
            'non_ag': get_mag(valid_layers_stack_non_ag),
            'am':     get_mag(valid_layers_stack_am),
            'sum':    get_mag(xr_sum_gbf2_cat),
        }
    }
    return (f"Biodiversity GBF2 priority scores written for year {yr_cal}", magnitudes)



def write_biodiversity_GBF3_NVIS_scores(data: Data, yr_cal: int, path) -> None:
    ''' No annualisation needed: GBF3 NVIS scores are a point-in-time snapshot (stock)
    derived from the `yr_cal` lumap, not a flow over `gap` years.

    Biodiversity GBF3 (NVIS) scores are written regardless of GBF3_NVIS_TARGET, but only when `WRITE_GBF3_NVIS` is not 'off'. '''

    if settings.WRITE_GBF3_NVIS == 'off':
        return f"Skipping Biodiversity GBF3 NVIS scores for year {yr_cal} as `WRITE_GBF3_NVIS` is set to 'off'"

    # All region-group pairs — used for baseline ALL_HA table.
    nvis_all_targets_df = (
        pd.read_csv(settings.INPUT_DIR + "/BIODIVERSITY_GBF3_NVIS_SCORES_AND_TARGETS.csv")
        .rename(columns={'species': 'group'})
        .query(f'region_level in ["NRM", "STATE"]')
        .query(f"sheet_name == '{settings.GBF3_NVIS_TARGET_CLASS}'")
        .query(f"resfactor == {settings.RESFACTOR}")
    )
    
    all_groups = sorted(nvis_all_targets_df['group'].unique())

    # Unpack the agricultural management land-use
    am_lu_unpack = [(am, l) for am, lus in data.AG_MAN_LU_DESC.items() for l in lus]

    # Decision variables for the year — in-memory (no chunking).
    ag_dvar_mrj = chunk_unify_size(tools.ag_mrj_to_xr(data, data.ag_dvars[yr_cal]).assign_coords(
        {'region_state': ('cell', data.REGION_STATE_NAME), 'region_NRM': ('cell', data.REGION_NRM_NAME)}
    ))
    non_ag_dvar_rk = chunk_unify_size(tools.non_ag_rk_to_xr(data, data.non_ag_dvars[yr_cal]).assign_coords(
        {'region_state': ('cell', data.REGION_STATE_NAME), 'region_NRM': ('cell', data.REGION_NRM_NAME)}
    ))
    am_dvar_amrj = chunk_unify_size(tools.am_mrj_to_xr(data, data.ag_man_dvars[yr_cal]).assign_coords(
        {'region_state': ('cell', data.REGION_STATE_NAME), 'region_NRM': ('cell', data.REGION_NRM_NAME)}
    ))

    # Impact arrays — in-memory (no chunking).
    ag_impact_j = xr.DataArray(
        ag_biodiversity.get_ag_biodiversity_contribution(data),
        dims=['lu'],
        coords={'lu': data.AGRICULTURAL_LANDUSES}
    )
    non_ag_impact_k = xr.DataArray(
        np.array(list(non_ag_biodiversity.get_non_ag_lu_biodiv_contribution(data).values()), dtype=np.float32),
        dims=['lu'],
        coords={'lu': data.NON_AGRICULTURAL_LANDUSES}
    )
    am_impact_amr = xr.DataArray(
        np.stack([
            arr.astype(np.float32)
            for v in ag_biodiversity.get_ag_management_biodiversity_contribution(data, yr_cal).values()
            for arr in v.values()
        ]),
        dims=['idx', 'cell'],
        coords={
            'idx': pd.MultiIndex.from_tuples(am_lu_unpack, names=['am', 'lu']),
            'cell': range(data.NCELLS),
        }
    ).unstack()

    REGION_LEVELS = ['region_NRM', 'region_state']
    ag_frames, am_frames, non_ag_frames, sum_frames_raw = [], [], [], []

    # Chunk dirs — each group batch is saved as a self-contained CF-encoded NC chunk.
    chunks_ag     = os.path.join(path, f'xr_biodiversity_GBF3_NVIS_ag_{yr_cal}_chunks')
    chunks_non_ag = os.path.join(path, f'xr_biodiversity_GBF3_NVIS_non_ag_{yr_cal}_chunks')
    chunks_am     = os.path.join(path, f'xr_biodiversity_GBF3_NVIS_ag_management_{yr_cal}_chunks')
    chunks_sum    = os.path.join(path, f'xr_biodiversity_GBF3_NVIS_sum_{yr_cal}_chunks')
    mags_ag, mags_non_ag, mags_am, mags_sum = [], [], [], []

    # ── Loop over each vegetation group ──────────────────────────────────────────
    group_slice_indices = np.arange(10, len(all_groups), 10)
    
    for group_idx,group in enumerate(np.array_split(all_groups, group_slice_indices)):
        
        nvis_layers_arr = xr.DataArray(
            np.array(
                [
                    data.get_resfactored_average_fraction(data.GBF3_NVIS_LAYERS_ALL.sel(group=g).data.todense().astype(np.float32), use_valid_cell_count=False)
                    for g in group
                ],
                dtype=np.float32,
            ),
            dims=['group', 'cell'],
            coords={'group': group, 'cell': np.arange(data.NCELLS)},
        )
                
        nvis_layers_arr = xr.where(
            data.SAVBURN_ELIGIBLE, nvis_layers_arr * settings.BIO_CONTRIBUTION_LDS, nvis_layers_arr
        ).astype(np.float32)

        veg_score_r = xr.DataArray(
            nvis_layers_arr * data.REAL_AREA[None, :],
            dims=['group', 'cell'],
            coords={
                'group': group,
                'cell': np.arange(data.NCELLS),
                'region_state': ('cell', data.REGION_STATE_NAME),
                'region_NRM':   ('cell', data.REGION_NRM_NAME),
            }
        )

        # ── Step 1 (per group): veg_result = dvar × veg_score × impact ──────────
        xr_gbf3_ag_g     = veg_score_r * ag_impact_j     * ag_dvar_mrj
        xr_gbf3_am_g     = veg_score_r * am_impact_amr   * am_dvar_amrj
        xr_gbf3_non_ag_g = veg_score_r * non_ag_impact_k * non_ag_dvar_rk

        # Pre-add_all sum for the 'sum' CSV (Type × cell).
        xr_gbf3_all_g = xr.concat(
            [
                xr_gbf3_ag_g.sum(dim=['lm', 'lu']).expand_dims({'Type': ['ag']}),
                xr_gbf3_non_ag_g.sum(dim=['lu']).expand_dims({'Type': ['non-ag']}),
                xr_gbf3_am_g.sum(dim=['am', 'lm', 'lu']).expand_dims({'Type': ['ag-man']}),
            ], dim='Type'
        )

        # Apply add_all inside the loop.
        xr_gbf3_ag_g     = add_all(xr_gbf3_ag_g,     ['lm', 'lu'])
        xr_gbf3_non_ag_g = add_all(xr_gbf3_non_ag_g, ['lu'])
        xr_gbf3_am_g     = add_all(xr_gbf3_am_g,     ['lm', 'lu', 'am'])
        xr_gbf3_all_g    = add_all(xr_gbf3_all_g,    ['Type'])

        # ── Step 2: groupby → inside scores ───────────────────────
        for rl in REGION_LEVELS:
            ag_df_region = (
                xr_gbf3_ag_g.groupby(rl).sum('cell')
                    .to_dataframe('Area Weighted Score (ha)').reset_index()
                    .rename(columns={rl: 'region'})
                    .assign(Type='Agricultural Land-use', Year=yr_cal, region_level=rl)
            )
            ag_df_AUS = (
                xr_gbf3_ag_g.sum('cell')
                    .to_dataframe('Area Weighted Score (ha)').reset_index()
                    .assign(region='AUSTRALIA', Type='Agricultural Land-use', Year=yr_cal, region_level=rl)
            )
            ag_frames.extend([ag_df_region.query('`Area Weighted Score (ha)` != 0'), ag_df_AUS.query('`Area Weighted Score (ha)` != 0')])

            am_df_region = (
                xr_gbf3_am_g.groupby(rl).sum('cell')
                    .to_dataframe('Area Weighted Score (ha)').reset_index(allow_duplicates=True)
                    .rename(columns={rl: 'region'})
                    .assign(Type='Agricultural Management', Year=yr_cal, region_level=rl)
            )
            am_df_AUS = (
                xr_gbf3_am_g.sum('cell')
                    .to_dataframe('Area Weighted Score (ha)').reset_index(allow_duplicates=True)
                    .assign(region='AUSTRALIA', Type='Agricultural Management', Year=yr_cal, region_level=rl)
            )
            am_frames.extend([am_df_region.query('`Area Weighted Score (ha)` != 0'), am_df_AUS.query('`Area Weighted Score (ha)` != 0')])

            non_ag_df_region = (
                xr_gbf3_non_ag_g.groupby(rl).sum('cell')
                    .to_dataframe('Area Weighted Score (ha)').reset_index()
                    .rename(columns={rl: 'region'})
                    .assign(Type='Non-Agricultural Land-use', Year=yr_cal, region_level=rl)
            )
            non_ag_df_AUS = (
                xr_gbf3_non_ag_g.sum('cell')
                    .to_dataframe('Area Weighted Score (ha)').reset_index()
                    .assign(region='AUSTRALIA', Type='Non-Agricultural Land-use', Year=yr_cal, region_level=rl)
            )
            non_ag_frames.extend([non_ag_df_region.query('`Area Weighted Score (ha)` != 0'), non_ag_df_AUS.query('`Area Weighted Score (ha)` != 0')])

            sum_df_region = (
                xr_gbf3_all_g.groupby(rl).sum('cell')
                    .to_dataframe('Area Weighted Score (ha)').reset_index()
                    .rename(columns={rl: 'region'})
                    .assign(Year=yr_cal, region_level=rl)
            )
            sum_df_AUS = (
                xr_gbf3_all_g.sum('cell')
                    .to_dataframe('Area Weighted Score (ha)').reset_index()
                    .assign(region='AUSTRALIA', Year=yr_cal, region_level=rl)
            )
            sum_frames_raw.extend([sum_df_region.query('`Area Weighted Score (ha)` != 0'), sum_df_AUS.query('`Area Weighted Score (ha)` != 0')])

        # ── Per-group valid-layer NC write (incremental, avoids full concat in RAM) ──
        # AG: save layers whose AUS score exceeds threshold (reuse ag_df_AUS from loop).
        ag_g_stacked = (
            xr_gbf3_ag_g
            .stack(layer=['group', 'lm', 'lu'])
            .drop_vars(['region_NRM', 'region_state'], errors='ignore')
        )
        valid_ag_g = ag_g_stacked.sel(
            layer=pd.MultiIndex.from_frame(ag_df_AUS.query('`Area Weighted Score (ha)` > 1')[['group', 'lm', 'lu']])
        )
        if valid_ag_g.sizes['layer'] == 0:
            fallback_midx = pd.MultiIndex.from_tuples(
                [(group[0], 'ALL', 'ALL')], names=['group', 'lm', 'lu']
            )
            valid_ag_g = ag_g_stacked.sel(layer=fallback_midx)
        mags_ag.extend(save2chunk(valid_ag_g, chunks_ag, group_idx))

        # Non-AG: same pattern.
        non_ag_g_stacked = (
            xr_gbf3_non_ag_g
            .stack(layer=['group', 'lu'])
            .drop_vars(['region_NRM', 'region_state'], errors='ignore')
        )
        valid_non_ag_g = non_ag_g_stacked.sel(
            layer=pd.MultiIndex.from_frame(non_ag_df_AUS.query('`Area Weighted Score (ha)` > 1')[['group', 'lu']])
        )
        if valid_non_ag_g.sizes['layer'] == 0:
            fallback_midx = pd.MultiIndex.from_tuples(
                [(group[0], 'ALL')], names=['group', 'lu']
            )
            valid_non_ag_g = non_ag_g_stacked.sel(layer=fallback_midx)
        mags_non_ag.extend(save2chunk(valid_non_ag_g, chunks_non_ag, group_idx))

        # AM: save valid layers; fall back to ALL/ALL/ALL so chunks dir is never empty.
        am_g_stacked = (
            xr_gbf3_am_g
            .stack(layer=['group', 'am', 'lm', 'lu'])
            .drop_vars(['region_NRM', 'region_state'], errors='ignore')
        )
        valid_am_g = am_g_stacked.sel(
            layer=pd.MultiIndex.from_frame(am_df_AUS.query('`Area Weighted Score (ha)` > 1')[['group', 'am', 'lm', 'lu']])
        )
        if valid_am_g.sizes['layer'] == 0:
            fallback_midx = pd.MultiIndex.from_tuples(
                [(group[0], 'ALL', 'ALL', 'ALL')], names=['group', 'am', 'lm', 'lu']
            )
            valid_am_g = am_g_stacked.sel(layer=fallback_midx)
        mags_am.extend(save2chunk(valid_am_g, chunks_am, group_idx))

        # Sum: save non-ALL Type layers (ALL is aggregate, excluded per convention).
        sum_g_stacked = (
            xr_gbf3_all_g
            .stack(layer=['Type', 'group'])
            .drop_vars(['region_NRM', 'region_state'], errors='ignore')
        )
        non_all_mask = np.array([t != 'ALL' for t in sum_g_stacked.coords['Type'].values])
        valid_sum_g = sum_g_stacked.isel(layer=non_all_mask)
        if valid_sum_g.sizes['layer'] > 0:
            mags_sum.extend(save2chunk(valid_sum_g, chunks_sum, group_idx))


    # ── Concat frames from loop ───────────────────────────────────────────────────
    GBF3_score_ag     = pd.concat(ag_frames,     ignore_index=True)
    GBF3_score_am     = pd.concat(am_frames,     ignore_index=True)
    GBF3_score_non_ag = pd.concat(non_ag_frames, ignore_index=True)

    # ── Step 3: baseline tables → append ALL_HA + OUTSIDE, compute percentages ──
    all_ha_df_nrm = (
        nvis_all_targets_df.query("region_level == 'NRM'")
        .groupby(['region', 'group'], sort=True)[['NATURAL_OUT_LUTO_HA', 'ALL_HA']]
        .sum().reset_index()
        .rename(columns={'NATURAL_OUT_LUTO_HA': 'BASE_OUTSIDE_SCORE', 'ALL_HA': 'BASE_TOTAL_SCORE'})
        .assign(region_level='region_NRM')
    )
    all_ha_df_state = (
        nvis_all_targets_df.query("region_level == 'STATE'")
        .groupby(['region', 'group'], sort=True)[['NATURAL_OUT_LUTO_HA', 'ALL_HA']]
        .sum().reset_index()
        .rename(columns={'NATURAL_OUT_LUTO_HA': 'BASE_OUTSIDE_SCORE', 'ALL_HA': 'BASE_TOTAL_SCORE'})
        .assign(region_level='region_state')
    )
    # AUSTRALIA rows: NRM-only sum prevents NRM+STATE double-counting.
    aus_base = all_ha_df_nrm.groupby('group', as_index=False)[['BASE_OUTSIDE_SCORE', 'BASE_TOTAL_SCORE']].sum()
    baseline_df = pd.concat([
        all_ha_df_nrm,
        all_ha_df_state,
        aus_base.assign(region='AUSTRALIA', region_level='region_NRM'),
        aus_base.assign(region='AUSTRALIA', region_level='region_state'),
    ], ignore_index=True)

    # ALL_HA lookup: (region, region_level) → total pre-1750 area across all groups.
    all_ha_lookup = (
        baseline_df.groupby(['region', 'region_level'], as_index=False)['BASE_TOTAL_SCORE']
        .sum().rename(columns={'BASE_TOTAL_SCORE': 'ALL_HA'})
    )

    # Merge baseline into score dfs and compute percent representation.
    BASELINE_COLS = ['region', 'group', 'region_level', 'BASE_OUTSIDE_SCORE', 'BASE_TOTAL_SCORE']
    GBF3_score_ag, GBF3_score_am, GBF3_score_non_ag = [
        df.merge(baseline_df[BASELINE_COLS], on=['region', 'group', 'region_level'], how='inner')
        .astype({'Area Weighted Score (ha)': float, 'BASE_TOTAL_SCORE': float})
        .eval('Relative_Contribution_Percentage = `Area Weighted Score (ha)` / BASE_TOTAL_SCORE * 100')
        for df in [GBF3_score_ag, GBF3_score_am, GBF3_score_non_ag]
    ]
    for df in [GBF3_score_ag, GBF3_score_am, GBF3_score_non_ag]:
        df['ALL_HA'] = df.merge(all_ha_lookup, on=['region', 'region_level'], how='left')['ALL_HA'].values

    # ── Step 4: targets for selected region-species pairs only ──────────────────
    target_nrm = (
        data.get_GBF3_NVIS_limit_score_inside_LUTO_by_yr(yr_cal)
        .to_series().groupby(['region', 'group']).sum()
        .reset_index(name='TARGET_INSIDE_SCORE')
    )
    nrm_to_state = (
        pd.DataFrame({'region': data.REGION_NRM_NAME, 'region_state': data.REGION_STATE_NAME})
        .drop_duplicates()
    )
    target_state = (
        target_nrm.merge(nrm_to_state, on='region')
        .groupby(['region_state', 'group'], as_index=False)['TARGET_INSIDE_SCORE'].sum()
        .rename(columns={'region_state': 'region'})
    )
    target_aus = target_nrm.groupby('group', as_index=False)['TARGET_INSIDE_SCORE'].sum()
    all_targets = pd.concat([
        target_nrm.assign(region_level='region_NRM'),
        target_state.assign(region_level='region_state'),
        target_aus.assign(region='AUSTRALIA', region_level='region_NRM'),
        target_aus.assign(region='AUSTRALIA', region_level='region_state'),
    ], ignore_index=True)

    for df in [GBF3_score_ag, GBF3_score_am, GBF3_score_non_ag]:
        df['TARGET_INSIDE_SCORE'] = (
            df.drop(columns=['TARGET_INSIDE_SCORE', 'Target_by_Percent'], errors='ignore')
            .merge(all_targets, on=['region', 'group', 'region_level'], how='left')
            ['TARGET_INSIDE_SCORE'].fillna(0).values
        )
        df['Target_by_Percent'] = np.where(
            df['TARGET_INSIDE_SCORE'] > 0,
            (df['TARGET_INSIDE_SCORE'] + df['BASE_OUTSIDE_SCORE']) / df['BASE_TOTAL_SCORE'] * 100,
            np.nan,
        )

    # Outside-LUTO rows: derived from baseline_df (AUSTRALIA rows already included).
    outside_per_region = baseline_df.assign(
        **{'Area Weighted Score (ha)': baseline_df['BASE_OUTSIDE_SCORE']},
        Relative_Contribution_Percentage=baseline_df['BASE_OUTSIDE_SCORE'] / baseline_df['BASE_TOTAL_SCORE'] * 100,
        Type='Outside LUTO study area', Year=yr_cal, lu='Outside LUTO study area',
    )
    outside_per_region['ALL_HA'] = (
        outside_per_region.merge(all_ha_lookup, on=['region', 'region_level'], how='left')['ALL_HA'].values
    )
    outside_per_region['TARGET_INSIDE_SCORE'] = (
        outside_per_region.merge(all_targets, on=['region', 'group', 'region_level'], how='left')['TARGET_INSIDE_SCORE'].fillna(0).values
    )
    outside_per_region['Target_by_Percent'] = np.where(
        outside_per_region['TARGET_INSIDE_SCORE'] > 0,
        (outside_per_region['TARGET_INSIDE_SCORE'] + outside_per_region['BASE_OUTSIDE_SCORE']) / outside_per_region['BASE_TOTAL_SCORE'] * 100,
        np.nan,
    )
    am_full = ['ALL'] + sorted(data.AG_MAN_LU_DESC.keys())
    lm_full = ['ALL', 'dry', 'irr']
    outside_expanded = outside_per_region.merge(
        pd.MultiIndex.from_product([am_full, lm_full], names=['am', 'lm']).to_frame(index=False), how='cross'
    )

    pd.concat([
        GBF3_score_ag,
        GBF3_score_am,
        GBF3_score_non_ag,
        outside_expanded], axis=0
        ).rename(columns={
            'lu':'Landuse',
            'lm':'Water_supply',
            'am':'Agricultural Management',
            'group':'Vegetation Group',
            'Relative_Contribution_Percentage':'Contribution Relative to Pre-1750 Level (%)'}
        ).reset_index(drop=True
        ).infer_objects(copy=False
        ).replace({'dry':'Dryland', 'irr':'Irrigated'}
        ).query('abs(`Area Weighted Score (ha)`) > 0'
        ).to_csv(os.path.join(path, f'biodiversity_GBF3_NVIS_scores_{yr_cal}.csv'), index=False)

    # ── Sum CSV ───────────────────────────────────────────────────────────────────
    # Merge baseline into raw sum frames (already have both region and AUSTRALIA rows).
    all_sum_raw = pd.concat(sum_frames_raw, ignore_index=True)
    sum_df = (
        all_sum_raw
        .merge(
            baseline_df[['region', 'group', 'region_level', 'BASE_OUTSIDE_SCORE', 'BASE_TOTAL_SCORE']],
            on=['region', 'group', 'region_level'], how='inner'
        )
    )
    sum_df['ALL_HA'] = sum_df.merge(all_ha_lookup, on=['region', 'region_level'], how='left')['ALL_HA'].values
    sum_df['Relative_Contribution_Percentage'] = sum_df['Area Weighted Score (ha)'] / sum_df['ALL_HA'] * 100

    # Outside rows for sum CSV — from baseline_df (no separate loop needed).
    outside_sum = baseline_df.assign(
        **{'Area Weighted Score (ha)': baseline_df['BASE_OUTSIDE_SCORE']},
        Year=yr_cal, Type='Outside LUTO study area',
    )
    outside_sum['ALL_HA'] = outside_sum.merge(all_ha_lookup, on=['region', 'region_level'], how='left')['ALL_HA'].values
    outside_sum['Relative_Contribution_Percentage'] = outside_sum['Area Weighted Score (ha)'] / outside_sum['ALL_HA'] * 100

    pd.concat([sum_df, outside_sum], axis=0, ignore_index=True
        ).rename(columns={'group': 'Vegetation Group'}
        ).reset_index(drop=True
        ).query('abs(`Area Weighted Score (ha)`) > 0'
        ).to_csv(os.path.join(path, f'biodiversity_GBF3_NVIS_sum_scores_{yr_cal}.csv'), index=False)

    # ── Write manifest (group name per chunk — used by create_report_layers for pagination) ──
    nvis_batches = list(enumerate(np.array_split(all_groups, group_slice_indices)))
    manifest = {str(idx): list(batch) for idx, batch in nvis_batches}
    for cdir in [chunks_ag, chunks_non_ag, chunks_am, chunks_sum]:
        if os.path.isdir(cdir):
            with open(os.path.join(cdir, 'manifest.json'), 'w') as f:
                json.dump(manifest, f)

    magnitudes = {
        'biodiversity_GBF3': {
            'ag':     mags_ag,
            'non_ag': mags_non_ag,
            'am':     mags_am,
            'sum':    mags_sum,
        }
    }
    return (f"Biodiversity GBF3 scores written for year {yr_cal}", magnitudes)



def write_biodiversity_GBF4_SNES_scores(data: Data, yr_cal: int, path) -> None:
    ''' No annualisation needed: GBF4 SNES scores are a point-in-time snapshot (stock)
    derived from the `yr_cal` lumap, not a flow over `gap` years.

    Biodiversity GBF4 SNES only being written to disk when `GBF4_TARGET_SNES` is not 'off' '''
    
    if settings.WRITE_GBF4_SNES == 'off':
        return f"Skipping Biodiversity GBF4 SNES scores for year {yr_cal} as `WRITE_GBF4_SNES` is set to 'off'"

    # 1. Load all species target df and get species list.
    snes_all_targets_df = data.get_SNES_targets_df(include_all=True)
    all_species = sorted(snes_all_targets_df['SCIENTIFIC_NAME'].unique())

    # 2. Unify chunk ag/agmgt/nonag decision variables.
    am_lu_unpack = [(am, l) for am, lus in data.AG_MAN_LU_DESC.items() for l in lus]
    ag_dvar_mrj = tools.ag_mrj_to_xr(data, data.ag_dvars[yr_cal]).assign_coords(
        {'region_state': ('cell', data.REGION_STATE_NAME), 'region_NRM': ('cell', data.REGION_NRM_NAME)}
    )
    non_ag_dvar_rk = tools.non_ag_rk_to_xr(data, data.non_ag_dvars[yr_cal]).assign_coords(
        {'region_state': ('cell', data.REGION_STATE_NAME), 'region_NRM': ('cell', data.REGION_NRM_NAME)}
    )
    am_dvar_amrj = tools.am_mrj_to_xr(data, data.ag_man_dvars[yr_cal]).assign_coords(
        {'region_state': ('cell', data.REGION_STATE_NAME), 'region_NRM': ('cell', data.REGION_NRM_NAME)}
    )

    # 3. Load ag/agmgt/nonag impact arrays without chunking.
    ag_impact_j = xr.DataArray(
        ag_biodiversity.get_ag_biodiversity_contribution(data),
        dims=['lu'],
        coords={'lu': data.AGRICULTURAL_LANDUSES}
    )
    non_ag_impact_k = xr.DataArray(
        np.array(list(non_ag_biodiversity.get_non_ag_lu_biodiv_contribution(data).values()), dtype=np.float32),
        dims=['lu'],
        coords={'lu': data.NON_AGRICULTURAL_LANDUSES}
    )
    am_impact_amr = xr.DataArray(
        np.stack([
            arr.astype(np.float32)
            for v in ag_biodiversity.get_ag_management_biodiversity_contribution(data, yr_cal).values()
            for arr in v.values()
        ]),
        dims=['idx', 'cell'],
        coords={
            'idx': pd.MultiIndex.from_tuples(am_lu_unpack, names=['am', 'lu']),
            'cell': range(data.NCELLS),
        }
    ).unstack()

    # 4. Set up region levels and tmp dirs.
    REGION_LEVELS = ['region_NRM', 'region_state']
    GBF4_score_ag = pd.DataFrame()
    GBF4_score_am = pd.DataFrame()
    GBF4_score_non_ag = pd.DataFrame()
    GBF4_score_sum = pd.DataFrame()
    chunks_ag     = os.path.join(path, f'xr_biodiversity_GBF4_SNES_ag_{yr_cal}_chunks')
    chunks_non_ag = os.path.join(path, f'xr_biodiversity_GBF4_SNES_non_ag_{yr_cal}_chunks')
    chunks_am     = os.path.join(path, f'xr_biodiversity_GBF4_SNES_ag_management_{yr_cal}_chunks')
    chunks_sum    = os.path.join(path, f'xr_biodiversity_GBF4_SNES_sum_{yr_cal}_chunks')
    mags_ag, mags_non_ag, mags_am, mags_sum = [], [], [], []

    am_impact_amr = (
        am_impact_amr
        .reindex(
            am=am_dvar_amrj.coords['am'].values,
            lu=ag_dvar_mrj.coords['lu'].values,
            fill_value=0,
        )
        .fillna(0)
    )

    def _load_sparse_veg_scores(species_batch):
        snes_layers_arr = np.array(
            [
                data.get_resfactored_average_fraction(
                    data.GBF4_SNES_LAYERS_ALL.sel(
                        species=sp, presence=settings.GBF4_SNES_PRESENCE_CLASS
                    ).data.todense().astype(np.float32),
                    use_valid_cell_count=False,
                )
                for sp in species_batch
            ],
            dtype=np.float32,
        )
        snes_layers_arr = np.where(
            data.SAVBURN_ELIGIBLE.astype(bool)[None, :],
            snes_layers_arr * settings.BIO_CONTRIBUTION_LDS,
            snes_layers_arr,
        ).astype(np.float32) * data.REAL_AREA[None, :]

        out_arr = np.empty(0)
        species_names = np.empty(0, dtype=object)
        nonzero_idx = np.empty(0, dtype=np.int32)
        for idx, sp in enumerate(species_batch):
            np_idx          = np.nonzero(snes_layers_arr[idx, :])[0]
            out_arr         = np.concatenate([out_arr, snes_layers_arr[idx, np_idx]])
            species_names   = np.concatenate([species_names, np.full(len(np_idx), sp, dtype=object)])
            nonzero_idx     = np.concatenate([nonzero_idx, np_idx])

        return xr.DataArray(
                out_arr,
                dims=['cell'],
                coords={'cell': np.arange(len(out_arr))},
            ).assign_coords({
                'species': ('cell', np.array(species_names)),
                'nonzero_idx': ('cell', np.array(nonzero_idx)),
                }
            )

    def _select_species_cells(da, species_batch, species_nz_map, veg_nz):
        selected = xr.concat(
            [da.sel(cell=species_nz_map[sp]) for sp in species_batch],
            dim='cell',
        )
        return selected.assign_coords({
            'cell': veg_nz['cell'].values,
            'species': ('cell', veg_nz['species'].values),
        })

    def _score_to_df(score, group_dims, type_str):
        out = []
        for species in pd.unique(score['species'].values):
            sub = score.sel(cell=score['species'] == species)
            for rl in REGION_LEVELS:
                region_df = (
                    sub.groupby(rl)
                    .sum(dim='cell')
                    .to_dataframe('Area Weighted Score (ha)').reset_index()
                    .assign(species=species, Year=yr_cal, region_level=rl)
                    .query('`Area Weighted Score (ha)` != 0')
                    .rename(columns={rl: 'region'})
                )
                aus_df = (
                    sub.sum(dim='cell')
                    .to_dataframe('Area Weighted Score (ha)').reset_index()
                    .assign(species=species, Year=yr_cal, region='AUSTRALIA', region_level=rl)
                    .query('`Area Weighted Score (ha)` != 0')
                )
                out.extend([region_df, aus_df])

        if out:
            out_df = pd.concat(out, ignore_index=True)
        else:
            out_df = pd.DataFrame(
                columns=group_dims + ['species', 'region', 'Year', 'region_level', 'Area Weighted Score (ha)']
            )
        if type_str is not None:
            out_df = out_df.assign(Type=type_str)
        return out_df

    def _valid_layers_from_aus(score_df, layer_cols):
        aus_df = score_df.query("region == 'AUSTRALIA' and region_level == 'region_NRM' and `Area Weighted Score (ha)` > 1")
        if aus_df.empty:
            return pd.MultiIndex.from_tuples([], names=layer_cols)
        return pd.MultiIndex.from_frame(aus_df[layer_cols].drop_duplicates()).sort_values()

    def _save_fullsize_layers(score, species_batch, species_nz_map, valid_layers, layer_dims, tmp_dir, sp_idx):
        score_layer_dims = [d for d in layer_dims if d in score.dims]
        species_pos = layer_dims.index('species')

        if len(valid_layers) == 0:
            fallback = tuple(
                score.coords[d].values[0] if d in score.dims else species_batch[0]
                for d in layer_dims
            )
            valid_layers = pd.MultiIndex.from_tuples([fallback], names=layer_dims)

        # Build output only for the (usually small) set of valid layers.
        # Peak memory: n_valid_layers × NCELLS × float32 — avoids the dense
        # [n_am × n_lm × n_lu × batch × NCELLS] intermediate that caused ~200 GB usage.
        n_valid = len(valid_layers)
        out_data = np.zeros((n_valid, data.NCELLS), dtype=np.float32)

        for i, layer_tuple in enumerate(valid_layers):
            species_val = layer_tuple[species_pos]
            nz_idx = species_nz_map.get(species_val, np.empty(0, dtype=np.int32))
            if len(nz_idx) == 0:
                continue
            sel_coords = {d: layer_tuple[layer_dims.index(d)] for d in score_layer_dims}
            layer_slice = score.sel(cell=score['species'] == species_val, **sel_coords)
            out_data[i, nz_idx] = layer_slice.values

        valid_arr = xr.DataArray(
            out_data,
            dims=['layer', 'cell'],
            coords={'layer': np.arange(n_valid), 'cell': np.arange(data.NCELLS)},
        ).assign_coords(xr.Coordinates.from_pandas_multiindex(valid_layers, 'layer'))

        return save2chunk(valid_arr, tmp_dir, sp_idx)


    # 5-6. Loop through each species in batches of 10.
    species_slice_indices = np.arange(10, len(all_species), 10)
    for sp_idx, species_batch in enumerate(np.array_split(all_species, species_slice_indices)):

        veg_nz = _load_sparse_veg_scores(species_batch)

        species_nz_map = {
            sp: veg_nz['nonzero_idx'].values[veg_nz['species'].values == sp]
            for sp in species_batch
        }

        dvar_species_nz = xr.concat(
            [ag_dvar_mrj.sel(cell=species_nz_map[sp]) for sp in species_batch], dim='cell'
        ).assign_coords({
            'cell': veg_nz['cell'].values,
            'species': ('cell', veg_nz['species'].values)
        })

        score_ag = add_all(
            (dvar_species_nz * ag_impact_j * veg_nz).astype(np.float32),
            ['lm', 'lu'],
        )

        non_ag_dvar_species_nz = _select_species_cells(
            non_ag_dvar_rk, species_batch, species_nz_map, veg_nz
        )
        score_non_ag = add_all(
            (non_ag_dvar_species_nz * non_ag_impact_k * veg_nz).astype(np.float32),
            ['lu'],
        )

        am_dvar_species_nz = _select_species_cells(
            am_dvar_amrj, species_batch, species_nz_map, veg_nz
        )
        am_impact_species_nz = _select_species_cells(
            am_impact_amr, species_batch, species_nz_map, veg_nz
        )
        score_am = add_all(
            (am_dvar_species_nz * am_impact_species_nz * veg_nz).astype(np.float32),
            ['am', 'lm', 'lu'],
        )

        score_sum = add_all(xr.concat([
            score_ag.sel(lm='ALL', lu='ALL', drop=True).expand_dims({'Type': ['ag']}),
            score_non_ag.sel(lu='ALL', drop=True).expand_dims({'Type': ['non-ag']}),
            score_am.sel(am='ALL', lm='ALL', lu='ALL', drop=True).expand_dims({'Type': ['ag-man']}),
        ], dim='Type'), ['Type'])

        score_ag_df = _score_to_df(score_ag, ['lm', 'lu'], 'Agricultural Land-use')
        score_non_ag_df = _score_to_df(score_non_ag, ['lu'], 'Non-Agricultural Land-use')
        score_am_df = _score_to_df(score_am, ['am', 'lm', 'lu'], 'Agricultural Management')
        score_sum_df = _score_to_df(score_sum, ['Type'], None)

        GBF4_score_ag = pd.concat([GBF4_score_ag, score_ag_df], ignore_index=True, copy=False)
        GBF4_score_non_ag = pd.concat([GBF4_score_non_ag, score_non_ag_df], ignore_index=True, copy=False)
        GBF4_score_am = pd.concat([GBF4_score_am, score_am_df], ignore_index=True, copy=False)
        GBF4_score_sum = pd.concat([GBF4_score_sum, score_sum_df], ignore_index=True, copy=False)

        mags_ag.extend(_save_fullsize_layers(
            score_ag,
            species_batch,
            species_nz_map,
            _valid_layers_from_aus(score_ag_df, ['lm', 'lu', 'species']),
            ['lm', 'lu', 'species'],
            chunks_ag,
            sp_idx,
        ))
        mags_non_ag.extend(_save_fullsize_layers(
            score_non_ag,
            species_batch,
            species_nz_map,
            _valid_layers_from_aus(score_non_ag_df, ['lu', 'species']),
            ['lu', 'species'],
            chunks_non_ag,
            sp_idx,
        ))
        mags_am.extend(_save_fullsize_layers(
            score_am,
            species_batch,
            species_nz_map,
            _valid_layers_from_aus(score_am_df, ['am', 'lm', 'lu', 'species']),
            ['am', 'lm', 'lu', 'species'],
            chunks_am,
            sp_idx,
        ))
        mags_sum.extend(_save_fullsize_layers(
            score_sum,
            species_batch,
            species_nz_map,
            _valid_layers_from_aus(score_sum_df, ['Type', 'species']),
            ['Type', 'species'],
            chunks_sum,
            sp_idx,
        ))
        del (
            veg_nz,
            dvar_species_nz,
            non_ag_dvar_species_nz,
            am_dvar_species_nz,
            am_impact_species_nz,
            score_ag,
            score_non_ag,
            score_am,
            score_sum,
            score_ag_df,
            score_non_ag_df,
            score_am_df,
            score_sum_df,
        )
        gc.collect()


    # 7. Concat frames from loop, combine with baseline df, compute percentages.
    all_ha_df_nrm = (
        snes_all_targets_df.query("region_level == 'NRM'")
        .groupby(['region', 'SCIENTIFIC_NAME'], sort=True)[['NATURAL_OUT_LUTO_HA', 'ALL_HA']]
        .sum().reset_index()
        .rename(columns={'SCIENTIFIC_NAME': 'species', 'NATURAL_OUT_LUTO_HA': 'BASE_OUTSIDE_SCORE', 'ALL_HA': 'BASE_TOTAL_SCORE'})
        .assign(region_level='region_NRM')
    )
    all_ha_df_state = (
        snes_all_targets_df.query("region_level == 'STATE'")
        .groupby(['region', 'SCIENTIFIC_NAME'], sort=True)[['NATURAL_OUT_LUTO_HA', 'ALL_HA']]
        .sum().reset_index()
        .rename(columns={'SCIENTIFIC_NAME': 'species', 'NATURAL_OUT_LUTO_HA': 'BASE_OUTSIDE_SCORE', 'ALL_HA': 'BASE_TOTAL_SCORE'})
        .assign(region_level='region_state')
    )
    aus_base = all_ha_df_nrm.groupby('species', as_index=False)[['BASE_OUTSIDE_SCORE', 'BASE_TOTAL_SCORE']].sum()
    baseline_df = pd.concat([
        all_ha_df_nrm,
        all_ha_df_state,
        aus_base.assign(region='AUSTRALIA', region_level='region_NRM'),
        aus_base.assign(region='AUSTRALIA', region_level='region_state'),
    ], ignore_index=True)

    all_ha_lookup = (
        baseline_df.groupby(['region', 'region_level'], as_index=False)['BASE_TOTAL_SCORE']
        .sum().rename(columns={'BASE_TOTAL_SCORE': 'ALL_HA'})
    )

    BASELINE_COLS = ['region', 'species', 'region_level', 'BASE_OUTSIDE_SCORE', 'BASE_TOTAL_SCORE']
    merged_score_dfs = []
    for df in [GBF4_score_ag, GBF4_score_am, GBF4_score_non_ag]:
        merged_df = df.merge(baseline_df[BASELINE_COLS], on=['region', 'species', 'region_level'], how='inner')
        merged_df = merged_df.astype({'Area Weighted Score (ha)': float, 'BASE_TOTAL_SCORE': float})
        merged_df['Relative_Contribution_Percentage'] = (
            merged_df['Area Weighted Score (ha)'] / merged_df['BASE_TOTAL_SCORE'] * 100
        )
        merged_score_dfs.append(merged_df)
    GBF4_score_ag, GBF4_score_am, GBF4_score_non_ag = merged_score_dfs
    for df in [GBF4_score_ag, GBF4_score_am, GBF4_score_non_ag]:
        df['ALL_HA'] = df.merge(all_ha_lookup, on=['region', 'region_level'], how='left')['ALL_HA'].values

    # 8. Get target scores for selected region-species pairs, compute target percentages.
    target_nrm = (
        data.get_GBF4_SNES_target_inside_LUTO_by_year(yr_cal)
        .to_series().groupby(['region', 'species']).sum()
        .reset_index(name='TARGET_INSIDE_SCORE')
    )
    nrm_to_state = (
        pd.DataFrame({'region': data.REGION_NRM_NAME, 'region_state': data.REGION_STATE_NAME})
        .drop_duplicates()
    )
    target_state = (
        target_nrm.merge(nrm_to_state, on='region')
        .groupby(['region_state', 'species'], as_index=False)['TARGET_INSIDE_SCORE'].sum()
        .rename(columns={'region_state': 'region'})
    )
    target_aus = target_nrm.groupby('species', as_index=False)['TARGET_INSIDE_SCORE'].sum()
    # Strip any 'AUSTRALIA' from nrm/state rows — in AUSTRALIA mode target_nrm carries
    # region='AUSTRALIA' which would duplicate the explicit Australia aggregates below.
    all_targets = pd.concat([
        target_nrm[target_nrm['region'] != 'AUSTRALIA'].assign(region_level='region_NRM'),
        target_state[target_state['region'] != 'AUSTRALIA'].assign(region_level='region_state'),
        target_aus.assign(region='AUSTRALIA', region_level='region_NRM'),
        target_aus.assign(region='AUSTRALIA', region_level='region_state'),
    ], ignore_index=True)

    for df in [GBF4_score_ag, GBF4_score_am, GBF4_score_non_ag]:
        df['TARGET_INSIDE_SCORE'] = (
            df.drop(columns=['TARGET_INSIDE_SCORE', 'Target_by_Percent'], errors='ignore')
            .merge(all_targets, on=['region', 'species', 'region_level'], how='left')
            ['TARGET_INSIDE_SCORE'].fillna(0).values
        )
        df['Target_by_Percent'] = np.where(
            df['TARGET_INSIDE_SCORE'] > 0,
            (df['TARGET_INSIDE_SCORE'] + df['BASE_OUTSIDE_SCORE']) / df['BASE_TOTAL_SCORE'] * 100,
            np.nan,
        )

    outside_per_region = baseline_df.assign(
        **{'Area Weighted Score (ha)': baseline_df['BASE_OUTSIDE_SCORE']},
        Relative_Contribution_Percentage=baseline_df['BASE_OUTSIDE_SCORE'] / baseline_df['BASE_TOTAL_SCORE'] * 100,
        Type='Outside LUTO study area', Year=yr_cal, lu='Outside LUTO study area',
    )
    outside_per_region['ALL_HA'] = (
        outside_per_region.merge(all_ha_lookup, on=['region', 'region_level'], how='left')['ALL_HA'].values
    )
    outside_per_region['TARGET_INSIDE_SCORE'] = (
        outside_per_region.merge(all_targets, on=['region', 'species', 'region_level'], how='left')
        ['TARGET_INSIDE_SCORE'].fillna(0).values
    )
    outside_per_region['Target_by_Percent'] = np.where(
        outside_per_region['TARGET_INSIDE_SCORE'] > 0,
        (outside_per_region['TARGET_INSIDE_SCORE'] + outside_per_region['BASE_OUTSIDE_SCORE'])
        / outside_per_region['BASE_TOTAL_SCORE'] * 100,
        np.nan,
    )
    am_full = ['ALL'] + sorted(data.AG_MAN_LU_DESC.keys())
    lm_full = ['ALL', 'dry', 'irr']
    outside_expanded = outside_per_region.merge(
        pd.MultiIndex.from_product([am_full, lm_full], names=['am', 'lm']).to_frame(index=False), how='cross'
    )

    scores_out = (pd.concat([GBF4_score_ag, GBF4_score_am, GBF4_score_non_ag, outside_expanded], axis=0)
        .rename(columns={
            'lu': 'Landuse', 'lm': 'Water_supply', 'am': 'Agricultural Management',
            'Relative_Contribution_Percentage': 'Contribution Relative to Pre-1750 Level (%)',
            'Target_by_Percent': 'Target by Percent (%)'}
        )
        .reset_index(drop=True)
        .infer_objects(copy=False)
        .replace({'dry': 'Dryland', 'irr': 'Irrigated'})
    )
    scores_out = scores_out[scores_out['Area Weighted Score (ha)'].abs() > 0]
    scores_out.to_csv(os.path.join(path, f'biodiversity_GBF4_SNES_scores_{yr_cal}.csv'), index=False)

    # Sum CSV.
    all_sum_raw = GBF4_score_sum
    sum_df = all_sum_raw.merge(
        baseline_df[['region', 'species', 'region_level', 'BASE_OUTSIDE_SCORE', 'BASE_TOTAL_SCORE']],
        on=['region', 'species', 'region_level'], how='inner'
    )
    sum_df['ALL_HA'] = sum_df.merge(all_ha_lookup, on=['region', 'region_level'], how='left')['ALL_HA'].values
    sum_df['Relative_Contribution_Percentage'] = sum_df['Area Weighted Score (ha)'] / sum_df['ALL_HA'] * 100

    outside_sum = baseline_df.assign(
        **{'Area Weighted Score (ha)': baseline_df['BASE_OUTSIDE_SCORE']},
        Year=yr_cal, Type='Outside LUTO study area',
    )
    outside_sum['ALL_HA'] = outside_sum.merge(all_ha_lookup, on=['region', 'region_level'], how='left')['ALL_HA'].values
    outside_sum['Relative_Contribution_Percentage'] = outside_sum['Area Weighted Score (ha)'] / outside_sum['ALL_HA'] * 100

    sum_out = pd.concat([sum_df, outside_sum], axis=0, ignore_index=True).reset_index(drop=True)
    sum_out = sum_out[sum_out['Area Weighted Score (ha)'].abs() > 0]
    sum_out.to_csv(os.path.join(path, f'biodiversity_GBF4_SNES_sum_scores_{yr_cal}.csv'), index=False)

    # 9. Write manifest (species per chunk) and compute magnitudes from chunk dirs.
    snes_batches = list(enumerate(np.array_split(all_species, species_slice_indices)))
    manifest = {str(idx): list(batch) for idx, batch in snes_batches}
    for cdir in [chunks_ag, chunks_non_ag, chunks_am, chunks_sum]:
        if os.path.isdir(cdir):
            with open(os.path.join(cdir, 'manifest.json'), 'w') as f:
                json.dump(manifest, f)

    magnitudes = {
        'biodiversity_GBF4_SNES': {
            'ag':     mags_ag,
            'non_ag': mags_non_ag,
            'am':     mags_am,
            'sum':    mags_sum,
        }
    }
    return (f"Biodiversity GBF4 SNES scores written for year {yr_cal}", magnitudes)



def write_biodiversity_GBF4_ECNES_scores(data: Data, yr_cal: int, path) -> None:
    ''' No annualisation needed: GBF4 ECNES scores are a point-in-time snapshot (stock)
    derived from the `yr_cal` lumap, not a flow over `gap` years.

    Biodiversity GBF4 ECNES is written regardless of `GBF4_TARGET_SNES`, but only when `WRITE_GBF4_ECNES` is not 'off'. '''

    if settings.WRITE_GBF4_ECNES == 'off':
        return f"Skipping Biodiversity GBF4 ECNES scores for year {yr_cal} as `WRITE_GBF4_ECNES` is set to 'off'"

    # 1. Load all community target df and get species list.
    ecnes_all_targets_df = data.get_ECNES_targets_df(include_all=True)
    all_species = sorted(ecnes_all_targets_df['COMMUNITY'].unique())

    # 2. Unify chunk ag/agmgt/nonag decision variables.
    am_lu_unpack = [(am, l) for am, lus in data.AG_MAN_LU_DESC.items() for l in lus]
    ag_dvar_mrj = chunk_unify_size(tools.ag_mrj_to_xr(data, data.ag_dvars[yr_cal]).assign_coords(
        {'region_state': ('cell', data.REGION_STATE_NAME), 'region_NRM': ('cell', data.REGION_NRM_NAME)}
    ))
    non_ag_dvar_rk = chunk_unify_size(tools.non_ag_rk_to_xr(data, data.non_ag_dvars[yr_cal]).assign_coords(
        {'region_state': ('cell', data.REGION_STATE_NAME), 'region_NRM': ('cell', data.REGION_NRM_NAME)}
    ))
    am_dvar_amrj = chunk_unify_size(tools.am_mrj_to_xr(data, data.ag_man_dvars[yr_cal]).assign_coords(
        {'region_state': ('cell', data.REGION_STATE_NAME), 'region_NRM': ('cell', data.REGION_NRM_NAME)}
    ))

    # 3. Load ag/agmgt/nonag impact arrays without chunking.
    ag_impact_j = xr.DataArray(
        ag_biodiversity.get_ag_biodiversity_contribution(data),
        dims=['lu'],
        coords={'lu': data.AGRICULTURAL_LANDUSES}
    )
    non_ag_impact_k = xr.DataArray(
        np.array(list(non_ag_biodiversity.get_non_ag_lu_biodiv_contribution(data).values()), dtype=np.float32),
        dims=['lu'],
        coords={'lu': data.NON_AGRICULTURAL_LANDUSES}
    )
    am_impact_amr = xr.DataArray(
        np.stack([
            arr.astype(np.float32)
            for v in ag_biodiversity.get_ag_management_biodiversity_contribution(data, yr_cal).values()
            for arr in v.values()
        ]),
        dims=['idx', 'cell'],
        coords={
            'idx': pd.MultiIndex.from_tuples(am_lu_unpack, names=['am', 'lu']),
            'cell': range(data.NCELLS),
        }
    ).unstack().fillna(0)

    # 4. Set up region levels and tmp dirs.
    REGION_LEVELS = ['region_NRM', 'region_state']
    ag_frames, am_frames, non_ag_frames, sum_frames_raw = [], [], [], []
    chunks_ag     = os.path.join(path, f'xr_biodiversity_GBF4_ECNES_ag_{yr_cal}_chunks')
    chunks_non_ag = os.path.join(path, f'xr_biodiversity_GBF4_ECNES_non_ag_{yr_cal}_chunks')
    chunks_am     = os.path.join(path, f'xr_biodiversity_GBF4_ECNES_ag_management_{yr_cal}_chunks')
    chunks_sum    = os.path.join(path, f'xr_biodiversity_GBF4_ECNES_sum_{yr_cal}_chunks')
    mags_ag, mags_non_ag, mags_am, mags_sum = [], [], [], []


    type_raw = ['ag', 'non-ag', 'ag-man']

    def _load_veg_scores(species_batch):
        rows = np.array([
            data.get_resfactored_average_fraction(
                data.GBF4_ECNES_LAYERS_ALL.sel(
                    species=sp, presence=settings.GBF4_ECNES_PRESENCE_CLASS
                ).data.todense().astype(np.float32),
                use_valid_cell_count=False,
            )
            for sp in species_batch
        ], dtype=np.float32)
        rows = np.where(
            data.SAVBURN_ELIGIBLE.astype(bool)[None, :],
            rows * settings.BIO_CONTRIBUTION_LDS, rows,
        ).astype(np.float32)
        rows = (rows * data.REAL_AREA[None, :]).astype(np.float32)
        return xr.DataArray(
            rows,
            dims=['species', 'cell'],
            coords={
                'species': species_batch,
                'cell': np.arange(data.NCELLS),
                'region_NRM': ('cell', data.REGION_NRM_NAME),
                'region_state': ('cell', data.REGION_STATE_NAME),
            },
        )

    # 5-6. Loop through each community in batches of 10.
    species_slice_indices = np.arange(10, len(all_species), 10)
    for sp_idx, species_batch in enumerate(np.array_split(all_species, species_slice_indices)):

        veg_xr = _load_veg_scores(species_batch)

        score_ag = add_all(
            (ag_dvar_mrj * ag_impact_j * veg_xr).astype(np.float32),
            ['lm', 'lu'],
        )
        score_am = add_all(
            (am_dvar_amrj * am_impact_amr * veg_xr).astype(np.float32),
            ['am', 'lm', 'lu'],
        )
        score_non_ag = add_all(
            (non_ag_dvar_rk * non_ag_impact_k * veg_xr).astype(np.float32),
            ['lu'],
        )

        score_by_type = add_all(xr.concat([
            score_ag.sel(lm='ALL', lu='ALL', drop=True).expand_dims({'Type': ['ag']}),
            score_non_ag.sel(lu='ALL', drop=True).expand_dims({'Type': ['non-ag']}),
            score_am.sel(am='ALL', lm='ALL', lu='ALL', drop=True).expand_dims({'Type': ['ag-man']}),
        ], dim='Type'), ['Type'])

        def _score_df(score, group_dims, type_str, rl):
            region_group_dims = [rl, 'species'] + group_dims
            region_df = (
                score.groupby(rl)
                .sum(dim='cell')
                .to_dataframe('Area Weighted Score (ha)').reset_index()
                .groupby(region_group_dims)[['Area Weighted Score (ha)']]
                .sum().reset_index()
                .assign(Year=yr_cal)
                .query('`Area Weighted Score (ha)` != 0')
                .rename(columns={rl: 'region'})
                .assign(region_level=rl)
            )
            aus_df = (
                score.sum(dim='cell')
                .to_dataframe('Area Weighted Score (ha)').reset_index()
                .groupby(['species'] + group_dims)[['Area Weighted Score (ha)']]
                .sum().reset_index()
                .assign(region='AUSTRALIA', Year=yr_cal, region_level=rl)
                .query('`Area Weighted Score (ha)` != 0')
            )
            if type_str:
                region_df = region_df.assign(Type=type_str)
                aus_df = aus_df.assign(Type=type_str)
            return pd.concat([region_df, aus_df], ignore_index=True)

        for rl in REGION_LEVELS:
            ag_frames.append(_score_df(score_ag, ['lm', 'lu'], 'Agricultural Land-use', rl))
            am_frames.append(_score_df(score_am, ['am', 'lm', 'lu'], 'Agricultural Management', rl))
            non_ag_frames.append(_score_df(score_non_ag, ['lu'], 'Non-Agricultural Land-use', rl))
            sum_frames_raw.append(_score_df(score_by_type, ['Type'], None, rl))

        def _valid_layers(score, layer_dims):
            aus_df = (
                score.sum(dim='cell')
                .to_dataframe('Area Weighted Score (ha)').reset_index()
                .query('`Area Weighted Score (ha)` > 1')
            )
            if aus_df.empty:
                return pd.MultiIndex.from_tuples(
                    [tuple(score.coords[d].values[0] for d in layer_dims)],
                    names=layer_dims,
                )
            return pd.MultiIndex.from_frame(aus_df[layer_dims]).sort_values()

        valid_ag_layers = _valid_layers(score_ag, ['species', 'lm', 'lu'])
        valid_am_layers = _valid_layers(score_am, ['species', 'am', 'lm', 'lu'])
        valid_non_ag_layers = _valid_layers(score_non_ag, ['species', 'lu'])

        valid_ag_s = (
            score_ag.stack(layer=['species', 'lm', 'lu'])
            .sel(layer=valid_ag_layers)
            .drop_vars(['region_state', 'region_NRM'], errors='ignore')
            .transpose('layer', 'cell')
            .compute()
        )
        mags_ag.extend(save2chunk(valid_ag_s, chunks_ag, sp_idx))

        valid_non_ag_s = (
            score_non_ag.stack(layer=['species', 'lu'])
            .sel(layer=valid_non_ag_layers)
            .drop_vars(['region_state', 'region_NRM'], errors='ignore')
            .transpose('layer', 'cell')
            .compute()
        )
        mags_non_ag.extend(save2chunk(valid_non_ag_s, chunks_non_ag, sp_idx))

        valid_am_s = (
            score_am.stack(layer=['species', 'am', 'lm', 'lu'])
            .sel(layer=valid_am_layers)
            .drop_vars(['region_state', 'region_NRM'], errors='ignore')
            .transpose('layer', 'cell')
            .compute()
        )
        mags_am.extend(save2chunk(valid_am_s, chunks_am, sp_idx))

        valid_sum_s = (
            score_by_type.sel(Type=type_raw)
            .stack(layer=['Type', 'species'])
            .drop_vars(['region_state', 'region_NRM'], errors='ignore')
            .transpose('layer', 'cell')
            .compute()
        )
        if valid_sum_s.sizes['layer'] > 0:
            mags_sum.extend(save2chunk(valid_sum_s, chunks_sum, sp_idx))

    # 7. Concat frames from loop, combine with baseline df, compute percentages.
    GBF4_score_ag     = pd.concat(ag_frames, ignore_index=True)
    GBF4_score_am     = pd.concat(am_frames, ignore_index=True)
    GBF4_score_non_ag = pd.concat(non_ag_frames, ignore_index=True)

    all_ha_df_nrm = (
        ecnes_all_targets_df.query("region_level == 'NRM'")
        .groupby(['region', 'COMMUNITY'], sort=True)[['NATURAL_OUT_LUTO_HA', 'ALL_HA']]
        .sum().reset_index()
        .rename(columns={'COMMUNITY': 'species', 'NATURAL_OUT_LUTO_HA': 'BASE_OUTSIDE_SCORE', 'ALL_HA': 'BASE_TOTAL_SCORE'})
        .assign(region_level='region_NRM')
    )
    all_ha_df_state = (
        ecnes_all_targets_df.query("region_level == 'STATE'")
        .groupby(['region', 'COMMUNITY'], sort=True)[['NATURAL_OUT_LUTO_HA', 'ALL_HA']]
        .sum().reset_index()
        .rename(columns={'COMMUNITY': 'species', 'NATURAL_OUT_LUTO_HA': 'BASE_OUTSIDE_SCORE', 'ALL_HA': 'BASE_TOTAL_SCORE'})
        .assign(region_level='region_state')
    )
    aus_base = all_ha_df_nrm.groupby('species', as_index=False)[['BASE_OUTSIDE_SCORE', 'BASE_TOTAL_SCORE']].sum()
    baseline_df = pd.concat([
        all_ha_df_nrm,
        all_ha_df_state,
        aus_base.assign(region='AUSTRALIA', region_level='region_NRM'),
        aus_base.assign(region='AUSTRALIA', region_level='region_state'),
    ], ignore_index=True)

    all_ha_lookup = (
        baseline_df.groupby(['region', 'region_level'], as_index=False)['BASE_TOTAL_SCORE']
        .sum().rename(columns={'BASE_TOTAL_SCORE': 'ALL_HA'})
    )

    BASELINE_COLS = ['region', 'species', 'region_level', 'BASE_OUTSIDE_SCORE', 'BASE_TOTAL_SCORE']
    GBF4_score_ag, GBF4_score_am, GBF4_score_non_ag = [
        df.merge(baseline_df[BASELINE_COLS], on=['region', 'species', 'region_level'], how='inner')
        .astype({'Area Weighted Score (ha)': float, 'BASE_TOTAL_SCORE': float})
        .eval('Relative_Contribution_Percentage = `Area Weighted Score (ha)` / BASE_TOTAL_SCORE * 100')
        for df in [GBF4_score_ag, GBF4_score_am, GBF4_score_non_ag]
    ]
    for df in [GBF4_score_ag, GBF4_score_am, GBF4_score_non_ag]:
        df['ALL_HA'] = df.merge(all_ha_lookup, on=['region', 'region_level'], how='left')['ALL_HA'].values

    # 8. Get target scores for selected region-community pairs, compute target percentages.
    target_nrm = (
        data.get_GBF4_ECNES_target_inside_LUTO_by_year(yr_cal)
        .to_series().groupby(['region', 'species']).sum()
        .reset_index(name='TARGET_INSIDE_SCORE')
    )
    nrm_to_state = (
        pd.DataFrame({'region': data.REGION_NRM_NAME, 'region_state': data.REGION_STATE_NAME})
        .drop_duplicates()
    )
    target_state = (
        target_nrm.merge(nrm_to_state, on='region')
        .groupby(['region_state', 'species'], as_index=False)['TARGET_INSIDE_SCORE'].sum()
        .rename(columns={'region_state': 'region'})
    )
    target_aus = target_nrm.groupby('species', as_index=False)['TARGET_INSIDE_SCORE'].sum()
    # Strip any 'AUSTRALIA' from nrm/state rows — in AUSTRALIA mode target_nrm carries
    # region='AUSTRALIA' which would duplicate the explicit Australia aggregates below.
    all_targets = pd.concat([
        target_nrm[target_nrm['region'] != 'AUSTRALIA'].assign(region_level='region_NRM'),
        target_state[target_state['region'] != 'AUSTRALIA'].assign(region_level='region_state'),
        target_aus.assign(region='AUSTRALIA', region_level='region_NRM'),
        target_aus.assign(region='AUSTRALIA', region_level='region_state'),
    ], ignore_index=True)

    for df in [GBF4_score_ag, GBF4_score_am, GBF4_score_non_ag]:
        df['TARGET_INSIDE_SCORE'] = (
            df.drop(columns=['TARGET_INSIDE_SCORE', 'Target_by_Percent'], errors='ignore')
            .merge(all_targets, on=['region', 'species', 'region_level'], how='left')
            ['TARGET_INSIDE_SCORE'].fillna(0).values
        )
        df['Target_by_Percent'] = np.where(
            df['TARGET_INSIDE_SCORE'] > 0,
            (df['TARGET_INSIDE_SCORE'] + df['BASE_OUTSIDE_SCORE']) / df['BASE_TOTAL_SCORE'] * 100,
            np.nan,
        )

    outside_per_region = baseline_df.assign(
        **{'Area Weighted Score (ha)': baseline_df['BASE_OUTSIDE_SCORE']},
        Relative_Contribution_Percentage=baseline_df['BASE_OUTSIDE_SCORE'] / baseline_df['BASE_TOTAL_SCORE'] * 100,
        Type='Outside LUTO study area', Year=yr_cal, lu='Outside LUTO study area',
    )
    outside_per_region['ALL_HA'] = (
        outside_per_region.merge(all_ha_lookup, on=['region', 'region_level'], how='left')['ALL_HA'].values
    )
    outside_per_region['TARGET_INSIDE_SCORE'] = (
        outside_per_region.merge(all_targets, on=['region', 'species', 'region_level'], how='left')
        ['TARGET_INSIDE_SCORE'].fillna(0).values
    )
    outside_per_region['Target_by_Percent'] = np.where(
        outside_per_region['TARGET_INSIDE_SCORE'] > 0,
        (outside_per_region['TARGET_INSIDE_SCORE'] + outside_per_region['BASE_OUTSIDE_SCORE'])
        / outside_per_region['BASE_TOTAL_SCORE'] * 100,
        np.nan,
    )
    am_full = ['ALL'] + sorted(data.AG_MAN_LU_DESC.keys())
    lm_full = ['ALL', 'dry', 'irr']
    outside_expanded = outside_per_region.merge(
        pd.MultiIndex.from_product([am_full, lm_full], names=['am', 'lm']).to_frame(index=False), how='cross'
    )

    pd.concat([GBF4_score_ag, GBF4_score_am, GBF4_score_non_ag, outside_expanded], axis=0
        ).rename(columns={
            'lu': 'Landuse', 'lm': 'Water_supply', 'am': 'Agricultural Management',
            'Relative_Contribution_Percentage': 'Contribution Relative to Pre-1750 Level (%)',
            'Target_by_Percent': 'Target by Percent (%)'}
        ).reset_index(drop=True
        ).infer_objects(copy=False
        ).replace({'dry': 'Dryland', 'irr': 'Irrigated'}
        ).query('abs(`Area Weighted Score (ha)`) > 0'
        ).to_csv(os.path.join(path, f'biodiversity_GBF4_ECNES_scores_{yr_cal}.csv'), index=False)

    # Sum CSV.
    all_sum_raw = pd.concat(sum_frames_raw, ignore_index=True)
    sum_df = (
        all_sum_raw
        .merge(
            baseline_df[['region', 'species', 'region_level', 'BASE_OUTSIDE_SCORE', 'BASE_TOTAL_SCORE']],
            on=['region', 'species', 'region_level'], how='inner'
        )
    )
    sum_df['ALL_HA'] = sum_df.merge(all_ha_lookup, on=['region', 'region_level'], how='left')['ALL_HA'].values
    sum_df['Relative_Contribution_Percentage'] = sum_df['Area Weighted Score (ha)'] / sum_df['ALL_HA'] * 100

    outside_sum = baseline_df.assign(
        **{'Area Weighted Score (ha)': baseline_df['BASE_OUTSIDE_SCORE']},
        Year=yr_cal, Type='Outside LUTO study area',
    )
    outside_sum['ALL_HA'] = outside_sum.merge(all_ha_lookup, on=['region', 'region_level'], how='left')['ALL_HA'].values
    outside_sum['Relative_Contribution_Percentage'] = outside_sum['Area Weighted Score (ha)'] / outside_sum['ALL_HA'] * 100

    pd.concat([sum_df, outside_sum], axis=0, ignore_index=True
        ).reset_index(drop=True
        ).query('abs(`Area Weighted Score (ha)`) > 0'
        ).to_csv(os.path.join(path, f'biodiversity_GBF4_ECNES_sum_scores_{yr_cal}.csv'), index=False)

    # 9. Write manifest and compute magnitudes from chunk dirs.
    ecnes_batches = list(enumerate(np.array_split(all_species, species_slice_indices)))
    manifest = {str(idx): list(batch) for idx, batch in ecnes_batches}
    for cdir in [chunks_ag, chunks_non_ag, chunks_am, chunks_sum]:
        if os.path.isdir(cdir):
            with open(os.path.join(cdir, 'manifest.json'), 'w') as f:
                json.dump(manifest, f)

    magnitudes = {
        'biodiversity_GBF4_ECNES': {
            'ag':     mags_ag,
            'non_ag': mags_non_ag,
            'am':     mags_am,
            'sum':    mags_sum,
        }
    }
    return (f"Biodiversity GBF4 ECNES scores written for year {yr_cal}", magnitudes)



def write_biodiversity_GBF8_scores_groups(data: Data, yr_cal, path):
    ''' No annualisation needed: GBF8 group scores are a point-in-time snapshot (stock)
    derived from the `yr_cal` lumap, not a flow over `gap` years.

    Biodiversity GBF8 groups only being written to disk when `GBF8_TARGET` is 'on' '''
    
    # Do nothing if biodiversity limits are off and no need to report
    if not settings.GBF8_TARGET == 'on':
        return "Skipped: Biodiversity GBF8 groups scores not written as `GBF8_TARGET` is set to 'off'"

        
    # Unpack the agricultural management land-use
    am_lu_unpack = [(am, l) for am, lus in data.AG_MAN_LU_DESC.items() for l in lus]

    # Get decision variables for the year
    ag_dvar_mrj = chunk_unify_size(
        tools.ag_mrj_to_xr(data, data.ag_dvars[yr_cal])
    ).assign_coords(region=('cell', data.REGION_NRM_NAME))
    non_ag_dvar_rk = chunk_unify_size(
        tools.non_ag_rk_to_xr(data, data.non_ag_dvars[yr_cal])
    ).assign_coords(region=('cell', data.REGION_NRM_NAME))
    am_dvar_amrj = chunk_unify_size(
        tools.am_mrj_to_xr(data, data.ag_man_dvars[yr_cal])
    ).assign_coords(region=('cell', data.REGION_NRM_NAME))

    # Get biodiversity scores for selected species
    bio_scores_sr = xr.DataArray(
        (data.get_GBF8_bio_layers_by_yr(yr_cal, level='group') * data.REAL_AREA[None,:]).astype(np.float32),
        dims=['group','cell'],
        coords={
            'group': data.BIO_GBF8_GROUPS_NAMES,
            'cell': np.arange(data.NCELLS)}
    ).chunk({'cell': data.NCELLS, 'group': 1})

    # Get the habitat contribution for ag/non-ag/am land-use to biodiversity scores
    ag_impact_j = xr.DataArray(
        ag_biodiversity.get_ag_biodiversity_contribution(data).astype(np.float32),
        dims=['lu'],
        coords={'lu':data.AGRICULTURAL_LANDUSES}
    )
    non_ag_impact_k = xr.DataArray(
        np.array(list(non_ag_biodiversity.get_non_ag_lu_biodiv_contribution(data).values()), dtype=np.float32),
        dims=['lu'],
        coords={'lu': data.NON_AGRICULTURAL_LANDUSES}
    )
    am_impact_amr = xr.DataArray(
        np.stack([
            arr.astype(np.float32)
            for v in ag_biodiversity.get_ag_management_biodiversity_contribution(data, yr_cal).values()
            for arr in v.values()
        ]),
        dims=['idx', 'cell'],
        coords={
            'idx': pd.MultiIndex.from_tuples(am_lu_unpack, names=['am', 'lu']),
            'cell': np.arange(data.NCELLS)}
    ).unstack()

    # Get the base year biodiversity scores
    base_yr_score = pd.DataFrame({
            'group': data.BIO_GBF8_GROUPS_NAMES,
            'BASE_OUTSIDE_SCORE': data.get_GBF8_score_outside_natural_LUTO_by_yr(yr_cal, level='group'),
            'BASE_TOTAL_SCORE': data.BIO_GBF8_BASELINE_SCORE_GROUPS['HABITAT_SUITABILITY_BASELINE_SCORE_ALL_AUSTRALIA']}
        ).eval('Relative_Contribution_Percentage = BASE_OUTSIDE_SCORE / BASE_TOTAL_SCORE * 100')

    # Calculate GBF8 scores for groups
    # Calculate xarray biodiversity GBF8 group scores
    xr_gbf8_groups_ag = bio_scores_sr * ag_impact_j * ag_dvar_mrj
    xr_gbf8_groups_am = am_dvar_amrj * bio_scores_sr * am_impact_amr
    xr_gbf8_groups_non_ag = non_ag_dvar_rk * bio_scores_sr * non_ag_impact_k

    xr_gbf8_groups_ag     = add_all(xr_gbf8_groups_ag,     ['lm', 'lu'])
    xr_gbf8_groups_non_ag = add_all(xr_gbf8_groups_non_ag, ['lu'])
    xr_gbf8_groups_am     = add_all(xr_gbf8_groups_am,     ['lm', 'lu', 'am'])

    GBF8_scores_groups_ag_region = xr_gbf8_groups_ag.groupby('region'
        ).sum('cell'
        ).to_dataframe('Area Weighted Score (ha)'
        ).reset_index(
        ).merge(base_yr_score
        ).eval('Relative_Contribution_Percentage = `Area Weighted Score (ha)` / BASE_TOTAL_SCORE * 100'
        ).assign(Type='Agricultural Land-use', Year=yr_cal)
        
    GBF8_scores_groups_am_region = xr_gbf8_groups_am.groupby('region'
        ).sum('cell'
        ).to_dataframe('Area Weighted Score (ha)'
        ).reset_index(allow_duplicates=True
        ).merge(base_yr_score
        ).astype({'Area Weighted Score (ha)': 'float', 'BASE_TOTAL_SCORE': 'float'}
        ).eval('Relative_Contribution_Percentage = `Area Weighted Score (ha)` / BASE_TOTAL_SCORE * 100'
        ).assign(Type='Agricultural Management', Year=yr_cal)
        
    GBF8_scores_groups_non_ag_region = xr_gbf8_groups_non_ag.groupby('region'
        ).sum(['cell']
        ).to_dataframe('Area Weighted Score (ha)'
        ).reset_index(
        ).merge(base_yr_score
        ).eval('Relative_Contribution_Percentage = `Area Weighted Score (ha)` / BASE_TOTAL_SCORE * 100'
        ).assign(Type='Non-Agricultural Land-use', Year=yr_cal)

    GBF8_scores_groups_ag_AUS = xr_gbf8_groups_ag.sum('cell'
        ).to_dataframe('Area Weighted Score (ha)'
        ).reset_index(
        ).merge(base_yr_score
        ).eval('Relative_Contribution_Percentage = `Area Weighted Score (ha)` / BASE_TOTAL_SCORE * 100'
        ).assign(Type='Agricultural Land-use', Year=yr_cal, region='AUSTRALIA')
        
    GBF8_scores_groups_am_AUS = xr_gbf8_groups_am.sum('cell'
        ).to_dataframe('Area Weighted Score (ha)'
        ).reset_index(allow_duplicates=True
        ).merge(base_yr_score
        ).astype({'Area Weighted Score (ha)': 'float', 'BASE_TOTAL_SCORE': 'float'}
        ).eval('Relative_Contribution_Percentage = `Area Weighted Score (ha)` / BASE_TOTAL_SCORE * 100'
        ).assign(Type='Agricultural Management', Year=yr_cal, region='AUSTRALIA')
        
    GBF8_scores_groups_non_ag_AUS = xr_gbf8_groups_non_ag.sum(['cell']
        ).to_dataframe('Area Weighted Score (ha)'
        ).reset_index(
        ).merge(base_yr_score
        ).eval('Relative_Contribution_Percentage = `Area Weighted Score (ha)` / BASE_TOTAL_SCORE * 100'
        ).assign(Type='Non-Agricultural Land-use', Year=yr_cal, region='AUSTRALIA')

    # Combine regional and Australia level data
    GBF8_scores_groups_ag = pd.concat([GBF8_scores_groups_ag_region, GBF8_scores_groups_ag_AUS], axis=0)
    GBF8_scores_groups_am = pd.concat([GBF8_scores_groups_am_region, GBF8_scores_groups_am_AUS], axis=0)
    GBF8_scores_groups_non_ag = pd.concat([GBF8_scores_groups_non_ag_region, GBF8_scores_groups_non_ag_AUS], axis=0)

    # Concatenate the dataframes, rename the columns, and reset the index, then save to a csv file
    base_yr_score = base_yr_score.assign(Type='Outside LUTO study area', Year=yr_cal) 

    pd.concat([
        GBF8_scores_groups_ag, 
        GBF8_scores_groups_am, 
        GBF8_scores_groups_non_ag,
        base_yr_score], axis=0
        ).rename(columns={
            'group': 'Group',
            'lu': 'Landuse',
            'lm': 'Water_supply',
            'am': 'Agricultural Management',
            'Relative_Contribution_Percentage': 'Contribution Relative to Pre-1750 Level (%)'}
        ).reset_index(drop=True
        ).infer_objects(copy=False
        ).replace({'dry':'Dryland', 'irr':'Irrigated'}
        ).query('abs(`Area Weighted Score (ha)`) > 0'
        ).to_csv(os.path.join(path, f'biodiversity_GBF8_groups_scores_{yr_cal}.csv'), index=False)

    

    # If a source has no non-trivial layers for this year, skip writing that NetCDF
    # entirely (avoids surfacing a phantom 'ALL' group in the map JSON).

    # ==================== Ag Valid Layers ====================
    if GBF8_scores_groups_ag_AUS['Area Weighted Score (ha)'].abs().sum() >= 1e-3:
        valid_ag_layers = pd.MultiIndex.from_frame(GBF8_scores_groups_ag_AUS.query('abs(`Area Weighted Score (ha)`) > 1')[['group', 'lm', 'lu']]).sort_values()
        valid_layers_stack_ag = xr_gbf8_groups_ag.stack(layer=['group', 'lm', 'lu']).sel(layer=valid_ag_layers).drop_vars('region').compute()
        save2nc(valid_layers_stack_ag, os.path.join(path, f'xr_biodiversity_GBF8_groups_ag_{yr_cal}.nc'))

    # ==================== Non-Ag Valid Layers ====================
    if GBF8_scores_groups_non_ag_AUS['Area Weighted Score (ha)'].abs().sum() >= 1e-3:
        valid_non_ag_layers = pd.MultiIndex.from_frame(GBF8_scores_groups_non_ag_AUS.query('abs(`Area Weighted Score (ha)`) > 1')[['group', 'lu']]).sort_values()
        valid_layers_stack_non_ag = xr_gbf8_groups_non_ag.stack(layer=['group', 'lu']).sel(layer=valid_non_ag_layers).drop_vars('region').compute()
        save2nc(valid_layers_stack_non_ag, os.path.join(path, f'xr_biodiversity_GBF8_groups_non_ag_{yr_cal}.nc'))

    # ==================== Ag Management Valid Layers ====================
    if GBF8_scores_groups_am_AUS['Area Weighted Score (ha)'].abs().sum() >= 1e-3:
        valid_am_layers = pd.MultiIndex.from_frame(GBF8_scores_groups_am_AUS.query('abs(`Area Weighted Score (ha)`) > 1')[['group', 'am', 'lm', 'lu']]).sort_values()
        valid_layers_stack_am = xr_gbf8_groups_am.stack(layer=['group', 'am', 'lm', 'lu']).sel(layer=valid_am_layers).drop_vars('region').compute()
    else:
        valid_layers_stack_am = (
            xr_gbf8_groups_am.sel(am='ALL', lm='ALL', lu='ALL')
            .expand_dims({'am': ['ALL'], 'lm': ['ALL'], 'lu': ['ALL']})
            .stack(layer=['group', 'am', 'lm', 'lu'])
            .drop_vars('region').compute()
        )
    save2nc(valid_layers_stack_am, os.path.join(path, f'xr_biodiversity_GBF8_groups_ag_management_{yr_cal}.nc'))

    magnitudes = {
        'biodiversity_GBF8': {
            'ag':     get_mag(valid_layers_stack_ag) if GBF8_scores_groups_ag_AUS['Area Weighted Score (ha)'].abs().sum() >= 1e-3 else [],
            'non_ag': get_mag(valid_layers_stack_non_ag) if GBF8_scores_groups_non_ag_AUS['Area Weighted Score (ha)'].abs().sum() >= 1e-3 else [],
            'am':     get_mag(valid_layers_stack_am),
        }
    }
    return (f"Biodiversity GBF8 groups scores written for year {yr_cal}", magnitudes)




def write_biodiversity_GBF8_scores_species(data: Data, yr_cal, path):
    ''' No annualisation needed: GBF8 species scores are a point-in-time snapshot (stock)
    derived from the `yr_cal` lumap, not a flow over `gap` years.

    Biodiversity GBF8 species only being written to disk when `GBF8_TARGET` is 'on' and selected species are provided '''

    if settings.GBF8_TARGET != 'on':
        return "Skipped: Biodiversity GBF8 species scores not written as `GBF8_TARGET` is set to 'off'"
    if len(data.BIO_GBF8_SEL_SPECIES) == 0:
        return "Skipped: Biodiversity GBF8 species scores not written as no selected species provided in `BIO_GBF8_SEL_SPECIES`"

    # Unpack the agricultural management land-use
    am_lu_unpack = [(am, l) for am, lus in data.AG_MAN_LU_DESC.items() for l in lus]

    # Get decision variables for the year
    ag_dvar_mrj = chunk_unify_size(
        tools.ag_mrj_to_xr(data, data.ag_dvars[yr_cal])
    ).assign_coords(region=('cell', data.REGION_NRM_NAME))
    non_ag_dvar_rk = chunk_unify_size(
        tools.non_ag_rk_to_xr(data, data.non_ag_dvars[yr_cal])
    ).assign_coords(region=('cell', data.REGION_NRM_NAME))
    am_dvar_amrj = chunk_unify_size(
        tools.am_mrj_to_xr(data, data.ag_man_dvars[yr_cal])
    ).assign_coords(region=('cell', data.REGION_NRM_NAME))
    
    # Get biodiversity scores for selected species
    bio_scores_sr = xr.DataArray(
        (data.get_GBF8_bio_layers_by_yr(yr_cal, level='species') * data.REAL_AREA[None, :]).astype(np.float32),
        dims=['species', 'cell'],
        coords={
            'species': data.BIO_GBF8_SEL_SPECIES,
            'cell': np.arange(data.NCELLS)}
    ).chunk({'cell': data.NCELLS, 'species': 1})

    # Get the habitat contribution for ag/non-ag/am land-use to biodiversity scores
    ag_impact_j = xr.DataArray(
        ag_biodiversity.get_ag_biodiversity_contribution(data).astype(np.float32),
        dims=['lu'],
        coords={'lu':data.AGRICULTURAL_LANDUSES}
    )
    non_ag_impact_k = xr.DataArray(
        np.array(list(non_ag_biodiversity.get_non_ag_lu_biodiv_contribution(data).values()), dtype=np.float32),
        dims=['lu'],
        coords={'lu': data.NON_AGRICULTURAL_LANDUSES}
    )
    am_impact_amr = xr.DataArray(
        np.stack([
            arr.astype(np.float32)
            for v in ag_biodiversity.get_ag_management_biodiversity_contribution(data, yr_cal).values()
            for arr in v.values()
        ]),
        dims=['idx', 'cell'],
        coords={
            'idx': pd.MultiIndex.from_tuples(am_lu_unpack, names=['am', 'lu']),
            'cell': np.arange(data.NCELLS)}
    ).unstack()

    # Get the base year biodiversity scores.
    # get_GBF8_target_inside_LUTO_by_yr now returns xr.DataArray(layer=(region,species));
    # GBF8 is always Australia mode so there is one region, but extract values consistently
    # with the other GBF functions.
    gbf8_target = (
        data.get_GBF8_target_inside_LUTO_by_yr(yr_cal)
        .to_series().groupby('species').sum()
        .reindex(data.BIO_GBF8_SEL_SPECIES).to_numpy()
    )
    base_yr_score = pd.DataFrame({
            'species':            data.BIO_GBF8_SEL_SPECIES,
            'BASE_OUTSIDE_SCORE': data.get_GBF8_score_outside_natural_LUTO_by_yr(yr_cal),
            'BASE_TOTAL_SCORE':   data.BIO_GBF8_BASELINE_SCORE_AND_TARGET_PERCENT_SPECIES['HABITAT_SUITABILITY_BASELINE_SCORE_ALL_AUSTRALIA'],
            'TARGET_INSIDE_SCORE': gbf8_target,}
        ).eval('Target_by_Percent = (TARGET_INSIDE_SCORE + BASE_OUTSIDE_SCORE) / BASE_TOTAL_SCORE * 100')

    # Calculate GBF8 scores for species
    # Calculate xarray biodiversity GBF8 species scores
    xr_gbf8_species_ag = bio_scores_sr * ag_impact_j * ag_dvar_mrj
    xr_gbf8_species_am = am_dvar_amrj * bio_scores_sr * am_impact_amr
    xr_gbf8_species_non_ag = non_ag_dvar_rk * bio_scores_sr * non_ag_impact_k

    xr_gbf8_species_ag     = add_all(xr_gbf8_species_ag,     ['lm', 'lu'])
    xr_gbf8_species_non_ag = add_all(xr_gbf8_species_non_ag, ['lu'])
    xr_gbf8_species_am     = add_all(xr_gbf8_species_am,     ['lm', 'lu', 'am'])

    GBF8_scores_species_ag_region = xr_gbf8_species_ag.groupby('region'
        ).sum('cell'
        ).to_dataframe('Area Weighted Score (ha)'
        ).reset_index(
        ).merge(base_yr_score
        ).eval('Relative_Contribution_Percentage = `Area Weighted Score (ha)` / BASE_TOTAL_SCORE * 100'
        ).assign(Type='Agricultural Land-use', Year=yr_cal)

    GBF8_scores_species_am_region = xr_gbf8_species_am.groupby('region'
        ).sum('cell'
        ).to_dataframe('Area Weighted Score (ha)'
        ).reset_index(allow_duplicates=True
        ).merge(base_yr_score
        ).astype({'Area Weighted Score (ha)': 'float', 'BASE_TOTAL_SCORE': 'float'}
        ).eval('Relative_Contribution_Percentage = `Area Weighted Score (ha)` / BASE_TOTAL_SCORE * 100'
        ).assign(Type='Agricultural Management', Year=yr_cal)

    GBF8_scores_species_non_ag_region = xr_gbf8_species_non_ag.groupby('region'
        ).sum(['cell']
        ).to_dataframe('Area Weighted Score (ha)'
        ).reset_index(
        ).merge(base_yr_score
        ).eval('Relative_Contribution_Percentage = `Area Weighted Score (ha)` / BASE_TOTAL_SCORE * 100'
        ).assign(Type='Non-Agricultural Land-use', Year=yr_cal)

    GBF8_scores_species_ag_AUS = xr_gbf8_species_ag.sum('cell'
        ).to_dataframe('Area Weighted Score (ha)'
        ).reset_index(
        ).merge(base_yr_score
        ).eval('Relative_Contribution_Percentage = `Area Weighted Score (ha)` / BASE_TOTAL_SCORE * 100'
        ).assign(Type='Agricultural Land-use', Year=yr_cal, region='AUSTRALIA')

    GBF8_scores_species_am_AUS = xr_gbf8_species_am.sum('cell'
        ).to_dataframe('Area Weighted Score (ha)'
        ).reset_index(allow_duplicates=True
        ).merge(base_yr_score
        ).astype({'Area Weighted Score (ha)': 'float', 'BASE_TOTAL_SCORE': 'float'}
        ).eval('Relative_Contribution_Percentage = `Area Weighted Score (ha)` / BASE_TOTAL_SCORE * 100'
        ).assign(Type='Agricultural Management', Year=yr_cal, region='AUSTRALIA')

    GBF8_scores_species_non_ag_AUS = xr_gbf8_species_non_ag.sum(['cell']
        ).to_dataframe('Area Weighted Score (ha)'
        ).reset_index(
        ).merge(base_yr_score
        ).eval('Relative_Contribution_Percentage = `Area Weighted Score (ha)` / BASE_TOTAL_SCORE * 100'
        ).assign(Type='Non-Agricultural Land-use', Year=yr_cal, region='AUSTRALIA')

    # Combine regional and Australia level data
    GBF8_scores_species_ag = pd.concat([GBF8_scores_species_ag_region, GBF8_scores_species_ag_AUS], axis=0)
    GBF8_scores_species_am = pd.concat([GBF8_scores_species_am_region, GBF8_scores_species_am_AUS], axis=0)
    GBF8_scores_species_non_ag = pd.concat([GBF8_scores_species_non_ag_region, GBF8_scores_species_non_ag_AUS], axis=0)

    # Concatenate the dataframes, rename the columns, and reset the index, then save to a csv file
    base_yr_score = base_yr_score.assign(Type='Outside LUTO study area', Year=yr_cal, lu='Outside LUTO study area'
        ).eval('Relative_Contribution_Percentage = BASE_OUTSIDE_SCORE / BASE_TOTAL_SCORE * 100')

    pd.concat([
        GBF8_scores_species_ag,
        GBF8_scores_species_am,
        GBF8_scores_species_non_ag,
        base_yr_score], axis=0
        ).rename(columns={
            'species': 'Species',
            'lu': 'Landuse',
            'lm': 'Water_supply',
            'am': 'Agricultural Management',
            'Relative_Contribution_Percentage': 'Contribution Relative to Pre-1750 Level (%)'}
        ).reset_index(drop=True
        ).infer_objects(copy=False
        ).replace({'dry':'Dryland', 'irr':'Irrigated'}
        ).query('abs(`Area Weighted Score (ha)`) > 0'
        ).to_csv(os.path.join(path, f'biodiversity_GBF8_species_scores_{yr_cal}.csv'), index=False)

    

    # If a source has no non-trivial layers for this year, skip writing that NetCDF
    # entirely (avoids surfacing a phantom 'ALL' species in the map JSON).

    # ==================== Ag Valid Layers ====================
    if GBF8_scores_species_ag_AUS['Area Weighted Score (ha)'].abs().sum() >= 1e-3:
        valid_ag_layers = pd.MultiIndex.from_frame(GBF8_scores_species_ag_AUS.query('abs(`Area Weighted Score (ha)`) > 1')[['species', 'lm', 'lu']]).sort_values()
        valid_layers_stack_ag = xr_gbf8_species_ag.stack(layer=['species', 'lm', 'lu']).sel(layer=valid_ag_layers).drop_vars('region').compute()
        save2nc(valid_layers_stack_ag, os.path.join(path, f'xr_biodiversity_GBF8_species_ag_{yr_cal}.nc'))

    # ==================== Non-Ag Valid Layers ====================
    if GBF8_scores_species_non_ag_AUS['Area Weighted Score (ha)'].abs().sum() >= 1e-3:
        valid_non_ag_layers = pd.MultiIndex.from_frame(GBF8_scores_species_non_ag_AUS.query('abs(`Area Weighted Score (ha)`) > 1')[['species', 'lu']]).sort_values()
        valid_layers_stack_non_ag = xr_gbf8_species_non_ag.stack(layer=['species', 'lu']).sel(layer=valid_non_ag_layers).drop_vars('region').compute()
        save2nc(valid_layers_stack_non_ag, os.path.join(path, f'xr_biodiversity_GBF8_species_non_ag_{yr_cal}.nc'))

    # ==================== Ag Management Valid Layers ====================
    if GBF8_scores_species_am_AUS['Area Weighted Score (ha)'].abs().sum() >= 1e-3:
        valid_am_layers = pd.MultiIndex.from_frame(GBF8_scores_species_am_AUS.query('abs(`Area Weighted Score (ha)`) > 1')[['species', 'am', 'lm', 'lu']]).sort_values()
        valid_layers_stack_am = xr_gbf8_species_am.stack(layer=['species', 'am', 'lm', 'lu']).sel(layer=valid_am_layers).drop_vars('region').compute()
    else:
        valid_layers_stack_am = (
            xr_gbf8_species_am.sel(am='ALL', lm='ALL', lu='ALL')
            .expand_dims({'am': ['ALL'], 'lm': ['ALL'], 'lu': ['ALL']})
            .stack(layer=['species', 'am', 'lm', 'lu'])
            .drop_vars('region').compute()
        )
    save2nc(valid_layers_stack_am, os.path.join(path, f'xr_biodiversity_GBF8_species_ag_management_{yr_cal}.nc'))

    magnitudes = {
        'biodiversity_GBF8': {
            'ag':     get_mag(valid_layers_stack_ag) if GBF8_scores_species_ag_AUS['Area Weighted Score (ha)'].abs().sum() >= 1e-3 else [],
            'non_ag': get_mag(valid_layers_stack_non_ag) if GBF8_scores_species_non_ag_AUS['Area Weighted Score (ha)'].abs().sum() >= 1e-3 else [],
            'am':     get_mag(valid_layers_stack_am),
        }
    }
    return (f"Biodiversity GBF8 species scores written for year {yr_cal}", magnitudes)







