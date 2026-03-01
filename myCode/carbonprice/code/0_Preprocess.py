# ==============================================================================
# Standard library
# ==============================================================================
import io
import math
import os
import shutil
import threading
import time
import zipfile
from datetime import datetime
from typing import Dict, Iterable, Optional, Sequence, Tuple, Union

# ==============================================================================
# Third-party
# ==============================================================================
import cf_xarray as cfxr
import geopandas as gpd
import numpy as np
import numpy_financial as npf
import pandas as pd
import rasterio
import xarray as xr
from joblib import Parallel, delayed
from rasterio.features import rasterize

import sys
sys.path.insert(0, "../../../")

# ==============================================================================
# Local packages
# ==============================================================================
import tools.config as config
from tools import LogToFile, log_memory_usage
from tools.helper_data import (
    build_profit_and_cost_nc,
    build_sol_profit_and_cost_nc,
    create_profit_for_cost,
    create_summary,
    make_prices_nc,
    make_sol_prices_nc,
    summarize_netcdf_to_excel,
    summarize_to_category,
    summarize_to_type,
)
from tools.tools import filter_all_from_dims, get_data_RES, get_path, nc_to_tif, save2nc


# ==============================================================================
# Utilities
# ==============================================================================

def tprint(*args, **kwargs):
    """Print with a timestamp prefix (YYYY-MM-DD HH:MM:SS)."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}]", *args, **kwargs)


def get_main_data_variable_name(ds: xr.Dataset) -> str:
    """Return the single data variable name from a Dataset, or raise."""
    data_vars_list = list(ds.data_vars)
    if len(data_vars_list) == 1:
        return data_vars_list[0]
    if len(data_vars_list) == 0:
        raise ValueError("Dataset contains no data variables.")
    raise ValueError(f"Dataset contains multiple data variables: {data_vars_list}")


def _run_or_parallel(njobs: int, func, arg_tuples) -> None:
    """Execute func(*args) for each args in arg_tuples, serially or in parallel.

    Parameters
    ----------
    njobs : int
        0 for serial execution; any positive integer for parallel (n_jobs=njobs).
    func : callable
    arg_tuples : iterable of tuples
        Each element is unpacked as positional arguments to func.
    """
    arg_list = list(arg_tuples)
    if njobs == 0:
        for args in arg_list:
            func(*args)
    else:
        Parallel(n_jobs=njobs)(delayed(func)(*args) for args in arg_list)


# ==============================================================================
# NetCDF Processing Helpers
# ==============================================================================

def sum_dims_if_exist(
        nc_path: str,
        vars: Optional[Sequence[str]] = None,
        dims=('lm', 'source', 'Type', 'GHG_source', 'Cost type',
              'From water-supply', 'To water-supply'),
        engine: Optional[str] = "h5netcdf",
        chunks="auto",
        keep_attrs: bool = True,
        finalize: str = "compute",
        save_inplace: bool = True,
):
    """Open a NetCDF file and sum over the given dimensions where they exist.

    Parameters
    ----------
    save_inplace : bool
        If True, overwrite the source file; otherwise return the Dataset.
    """
    if isinstance(dims, str):
        dims = [dims]

    ds = xr.open_dataset(nc_path, engine=engine, chunks=chunks)

    def _reduce(da: xr.DataArray) -> xr.DataArray:
        present = [d for d in dims if d in da.dims]
        return da.sum(dim=present, keep_attrs=keep_attrs, skipna=True) if present else da

    if vars is None:
        out = ds.map(_reduce)
    else:
        missing = [v for v in vars if v not in ds.data_vars]
        if missing:
            raise KeyError(f"Variables not found: {missing}")
        out = ds.copy()
        for v in vars:
            out[v] = _reduce(ds[v])

    if finalize == "compute":
        out = out.compute()
    elif finalize == "persist":
        out = out.persist()

    ds.close()

    if save_inplace:
        temp_path = nc_path + ".tmp"
        out.to_netcdf(temp_path, engine=engine)
        shutil.move(temp_path, nc_path)
        return nc_path

    return out


def reduce_layered_da(
    da: xr.DataArray,
    dims_to_sum: Sequence[str],
    layer_dim: str = "layer",
    keep_attrs: bool = True,
    skipna: bool = True,
    layer_level_order: Optional[Iterable[str]] = None,
) -> xr.DataArray:
    """Sum over dims_to_sum while preserving any MultiIndex on layer_dim.

    Input dims are typically ('cell', 'layer') where layer is a MultiIndex.
    Output retains the same shape with layer as a MultiIndex.
    """

    def _ensure_layer_multiindex(
            da: xr.DataArray,
            layer_dim: str = "layer",
            level_order: Optional[Iterable[str]] = None,
    ) -> xr.DataArray:
        """Ensure layer_dim is a MultiIndex, rebuilding from level coords if needed."""
        if layer_dim not in da.dims:
            return da

        idx = da.indexes.get(layer_dim, None)
        if isinstance(idx, pd.MultiIndex):
            return da

        layer_level_vars = [
            c for c in da.coords
            if c != layer_dim and da.coords[c].dims == (layer_dim,)
        ]
        if not layer_level_vars:
            raise ValueError(
                f"'{layer_dim}' is not a MultiIndex and no layer-level coords exist to rebuild it. "
                f"Available coords with dims=('layer',): {layer_level_vars}"
            )

        if level_order is not None:
            level_order = list(level_order)
            layer_level_vars = (
                [c for c in level_order if c in layer_level_vars]
                + [c for c in layer_level_vars if c not in level_order]
            )

        return da.set_index({layer_dim: layer_level_vars})

    if not np.issubdtype(da.dtype, np.number):
        return da

    if layer_dim not in da.dims:
        present = [d for d in dims_to_sum if d in da.dims]
        return da.sum(dim=present, keep_attrs=keep_attrs, skipna=skipna) if present else da

    # 1. Ensure layer is a MultiIndex
    da = _ensure_layer_multiindex(da, layer_dim=layer_dim, level_order=layer_level_order)

    # 2. Unstack: layer → individual dims
    da_u = da.unstack(layer_dim)

    # 3. Sum requested dims
    present = [d for d in dims_to_sum if d in da_u.dims]
    if present:
        da_u = da_u.sum(dim=present, keep_attrs=keep_attrs, skipna=skipna)

    # 4. Stack all non-cell dims back into layer
    stack_levels = [d for d in da_u.dims if d != "cell"]
    if layer_level_order is not None:
        layer_level_order = list(layer_level_order)
        stack_levels = (
            [d for d in layer_level_order if d in stack_levels]
            + [d for d in stack_levels if d not in layer_level_order]
        )

    if stack_levels:
        return da_u.stack({layer_dim: stack_levels})

    return da_u


# ==============================================================================
# File I/O
# ==============================================================================

def extract_files_from_zip(
        zip_path: str,
        copy_files: list,
        years: list,
        allow_missing_2010: bool = True,
) -> Dict[Tuple[str, int], bytes]:
    """Extract required files from a zip archive or directory into memory.

    Parameters
    ----------
    zip_path : str
        Path to a .zip file or an already-extracted directory.
    copy_files : list
        Variable prefix strings to look for.
    years : list
        Year integers.
    allow_missing_2010 : bool
        If True, silently skip missing files for year 2010.

    Returns
    -------
    dict
        Mapping of (var_prefix, year) → file bytes (or None if skipped/missing).
    """
    files_dict = {}

    if os.path.isfile(zip_path) and zipfile.is_zipfile(zip_path):
        tprint(f"Extracting files from {os.path.basename(zip_path)}...")
        with zipfile.ZipFile(zip_path, 'r') as zf:
            all_names = set(zf.namelist())
            for var_prefix in copy_files:
                for year in years:
                    src = f"out_{year}/{var_prefix}_{year}.nc"
                    if src in all_names:
                        files_dict[(var_prefix, year)] = zf.read(src)
                    elif allow_missing_2010 and year == 2010:
                        files_dict[(var_prefix, year)] = None
                    else:
                        tprint(f"  Warning: {src} not found")
                        files_dict[(var_prefix, year)] = None

    elif os.path.isdir(zip_path):
        tprint(f"Reading files from directory {os.path.basename(zip_path)}...")
        for var_prefix in copy_files:
            for year in years:
                src_path = os.path.join(zip_path, f"out_{year}", f"{var_prefix}_{year}.nc")
                if os.path.exists(src_path):
                    with open(src_path, 'rb') as f:
                        files_dict[(var_prefix, year)] = f.read()
                elif allow_missing_2010 and year == 2010:
                    files_dict[(var_prefix, year)] = None
                else:
                    tprint(f"  Warning: out_{year}/{var_prefix}_{year}.nc not found")
                    files_dict[(var_prefix, year)] = None

    else:
        raise ValueError(f"Invalid path: {zip_path} is neither a zip file nor a directory.")

    tprint(f"Extraction complete: {sum(1 for v in files_dict.values() if v is not None)} files loaded")
    return files_dict


def process_single_file_from_memory(
    file_bytes: bytes,
    var_prefix: str,
    year: int,
    target_path_name: str,
    dims_to_sum=("lm", "source", "Type", "GHG_source", "Cost type",
                 "From water-supply", "To water-supply"),
    engine: str = "h5netcdf",
    chunks="auto",
    layer_level_order=None,
) -> str:
    """Process an in-memory NetCDF file: decode MultiIndex, fill NaN, sum dims, save."""
    if file_bytes is None:
        return f"Skipped: {var_prefix}_{year}.nc"

    target_year_path = os.path.join(target_path_name, str(year))
    os.makedirs(target_year_path, exist_ok=True)
    dst_file = os.path.join(target_year_path, f"{var_prefix}_{year}.nc")

    with io.BytesIO(file_bytes) as bio:
        with xr.open_dataset(bio, engine=engine, chunks=chunks) as ds:
            ds = cfxr.decode_compress_to_multi_index(ds, "layer")
            ds = ds.fillna(0)

            out_vars = {}
            for v in ds.data_vars:
                da = ds[v]
                da = filter_all_from_dims(
                    da,
                    layer_dim="layer",
                    strict_layer_multiindex=True,
                    layer_level_order=layer_level_order,
                )
                da = reduce_layered_da(
                    da,
                    dims_to_sum=dims_to_sum,
                    layer_dim="layer",
                    keep_attrs=True,
                    skipna=True,
                    layer_level_order=layer_level_order,
                )
                out_vars[v] = da

            out = xr.Dataset(out_vars, attrs=ds.attrs).load()
            if 'layer' in out.dims:
                idx = out.indexes.get('layer')
                if isinstance(idx, pd.MultiIndex):
                    out = out.unstack('layer')

    # Apply the same dim reduction as sum_dims_if_exist, but in memory to avoid
    # a redundant write→reopen→reduce→write cycle (data is already loaded).
    _EXTRA_DIMS = ('lm', 'source', 'Type', 'GHG_source', 'Cost type',
                   'From water-supply', 'To water-supply')
    final_da = out["data"] if "data" in out.data_vars else out[list(out.data_vars)[0]]
    present = [d for d in _EXTRA_DIMS if d in final_da.dims]
    if present:
        final_da = final_da.sum(dim=present, keep_attrs=True, skipna=True)
    save2nc(final_da, dst_file)
    return f"Processed: {var_prefix}_{year}.nc"


def copy_single_file(
        origin_path_name: str,
        target_path_name: str,
        var_prefix: str,
        year: int,
        dims_to_sum=('lm', 'source', 'Type', 'GHG_source', 'Cost type',
                     'From water-supply', 'To water-supply'),
        engine: str = "h5netcdf",
        chunks="auto",
        allow_missing_2010: bool = True,
) -> str:
    """Copy and process a single NetCDF file (reduce specified dims, fill NaN)."""
    year_path = os.path.join(origin_path_name, f"out_{year}")
    target_year_path = os.path.join(target_path_name, str(year))
    os.makedirs(target_year_path, exist_ok=True)

    src_file = os.path.join(year_path, f"{var_prefix}_{year}.nc")
    dst_file = os.path.join(target_year_path, f"{var_prefix}_{year}.nc")

    if not os.path.exists(src_file):
        if allow_missing_2010 and year == 2010:
            return
        raise FileNotFoundError(src_file)

    def _reduce_one(da: xr.DataArray) -> xr.DataArray:
        if not np.issubdtype(da.dtype, np.number):
            return da
        present_dims = [d for d in dims_to_sum if d in da.dims]
        return da.sum(dim=present_dims, keep_attrs=True, skipna=True) if present_dims else da

    with xr.open_dataset(src_file, engine=engine, chunks=chunks) as ds:
        ds = filter_all_from_dims(ds)
        out = ds.fillna(0).map(_reduce_one).load()
        save2nc(out, dst_file)

    return f"Copied: {os.path.basename(src_file)} -> {dst_file}"


# ==============================================================================
# Cost Computation
# ==============================================================================

def amortize_costs(data_path_name, amortize_file, years, njobs=0, rate=0.07, horizon=60):
    """Compute amortized costs and write one output file per year.

    Loads upfront costs for all years, converts each to an annual payment
    via npf.pmt, then accumulates the horizon-year payment stream into a
    per-year matrix.  Output files are written per affected year.
    """
    tprint(f"Computing amortized costs for '{data_path_name}'...")

    # 1. Load all cost files
    file_paths = [
        os.path.join(data_path_name, f'{year}', f'{amortize_file}_{year}.nc')
        for year in years
    ]
    existing_files = [p for p in file_paths if os.path.exists(p)]
    if not existing_files:
        raise FileNotFoundError(
            f"No files matching '{amortize_file}' found under {data_path_name}."
        )
    valid_years = sorted([int(p.split('_')[-1].split('.')[0]) for p in existing_files])

    all_costs_ds = xr.open_mfdataset(
        existing_files,
        engine="h5netcdf",
        combine="nested",
        concat_dim="year",
        parallel=False,
        chunks={"cell": 'auto', "year": 1},  # one chunk per year → isel loads only 1 file at a time
    ).assign_coords(year=valid_years)

    cost_variable_name = get_main_data_variable_name(all_costs_ds)
    pv_values_all_years = all_costs_ds[cost_variable_name]

    # 2. Convert PV to annual payments
    annual_payments = xr.apply_ufunc(
        lambda x: -1 * npf.pmt(rate, horizon, pv=x.astype(np.float64), fv=0, when='begin'),
        pv_values_all_years,
        dask="parallelized",
        output_dtypes=[np.float32],
    ).astype('float32')

    all_years = annual_payments.year.values
    n_years = len(all_years)
    base_shape = annual_payments.isel(year=0).drop_vars('year').shape

    # 3. Accumulate payments into a per-affected-year matrix.
    # With year:1 chunks, each isel(year=s_idx) only reads 1 file (not all 41).
    # The inner offset loop is replaced by a numpy slice-add: same result at C speed.
    # Memory peak: amortized_matrix (n_years×cells) + one payment slice — unchanged.
    amortized_matrix = np.zeros((n_years,) + base_shape, dtype=np.float32)
    for s_idx in range(n_years):
        payment = np.nan_to_num(
            annual_payments.isel(year=s_idx).drop_vars('year').values, nan=0.0
        )
        amortized_matrix[s_idx : min(s_idx + horizon, n_years)] += payment

    coords = {k: v for k, v in annual_payments.coords.items() if k != 'year'}
    coords['year'] = all_years
    dims = ('year',) + tuple(d for d in annual_payments.dims if d != 'year')
    amortized_by_affect_year = xr.DataArray(
        data=amortized_matrix, dims=dims, coords=coords, name='data',
    )

    all_costs_ds.close()

    # 4. Save one file per year
    def _save_one_year(y: int):
        out_dir = os.path.join(data_path_name, f"{y}")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{amortize_file}_amortised_{y}.nc")
        ds_y = xr.Dataset({'data': amortized_by_affect_year.sel(year=y)})
        save2nc(ds_y, out_path)

    if njobs > 0:
        Parallel(n_jobs=njobs, backend="threading")(
            delayed(_save_one_year)(y) for y in all_years
        )
    else:
        for y in all_years:
            _save_one_year(y)


def calculate_and_save_single_diff(diff_file, year, data_path_name):
    """Compute and save the year-over-year diff for a single file pair."""
    # 1. Build source file paths
    src_file_0 = os.path.join(data_path_name, str(year), f"{diff_file}_{year}.nc")
    src_file_1 = os.path.join(data_path_name, str(year - 1), f"{diff_file}_{year - 1}.nc")

    # 2. Open and align the two datasets
    with xr.open_dataset(src_file_0) as ds_0, xr.open_dataset(src_file_1) as ds_1:
        ds_0, ds_1 = xr.align(ds_0, ds_1, join='outer', fill_value=0)
        ds_res = ds_0 - ds_1

    # 3. Build output path and save
    variable_name = diff_file.replace('.nc', '')
    dst_filename = f"{variable_name}_diff_{year}.nc"
    dst_file = os.path.join(data_path_name, str(year), dst_filename)
    save2nc(ds_res, dst_file)

    return f"Calculated and saved diff: {dst_filename}"


# ==============================================================================
# Differential & Aggregation Computation
# ==============================================================================

def calculate_diff_two_scenarios(input_all_names, output_names, output_path, year,
                                  env_file_basename, output_part_name):
    """Compute pairwise differences between two scenario groups and save results."""
    for i, (run_name_0, run_name_1) in enumerate(zip(input_all_names[0], input_all_names[1])):
        output_subdir = output_names[i]
        run0_path = os.path.join(output_path, run_name_0, str(year), env_file_basename)
        run1_path = os.path.join(output_path, run_name_1, str(year), env_file_basename)

        for run_name, f in [(run_name_0, run0_path), (run_name_1, run1_path)]:
            if not os.path.exists(f):
                raise FileNotFoundError(
                    f"Scenario '{run_name}' | Year {year} | Missing: {os.path.basename(f)}\n"
                    f"  Full path: {f}"
                )

        with xr.open_dataset(run0_path, chunks='auto') as ds_0, \
                xr.open_dataset(run1_path, chunks='auto') as ds_1:
            ds_0 = filter_all_from_dims(ds_0)
            ds_1 = filter_all_from_dims(ds_1)
            ds_0, ds_1 = xr.align(ds_0, ds_1, join='outer', fill_value=0)
            # GHG: higher is worse, so diff is ds_0 - ds_1; costs: ds_1 - ds_0
            env_diff = (ds_0 - ds_1) if 'GHG' in env_file_basename else (ds_1 - ds_0)
            env_diff = env_diff.compute()

        output_dir = os.path.join(output_path, output_subdir, str(year))
        os.makedirs(output_dir, exist_ok=True)
        output_filename = f"{output_part_name}_{output_subdir}_{year}.nc"
        save2nc(env_diff, os.path.join(output_dir, output_filename))


def calculate_env_diff(year, output_path, input_all_names, env_file, output_all_names):
    """Compute environment (carbon/bio) diffs for all three scenario pairs."""
    env_file_basename = f"{env_file}_{year}.nc"
    calculate_diff_two_scenarios(
        [input_all_names[0], input_all_names[1]], output_all_names[0:10],
        output_path, year, env_file_basename, env_file,
    )
    calculate_diff_two_scenarios(
        [input_all_names[1], input_all_names[2]], output_all_names[10:20],
        output_path, year, env_file_basename, env_file,
    )
    calculate_diff_two_scenarios(
        [input_all_names[0], input_all_names[2]], output_all_names[20:30],
        output_path, year, env_file_basename, env_file,
    )


def calculate_profit_for_run(year, out_path, run_name, cost_basename, revenue_basename,
                              profit_category=None):
    """Calculate profit (revenue − cost) for a single run and year, then save.

    profit_category : str, optional
        Category label used in the output filename (e.g. 'ag', 'agricultural_management',
        'non_ag').  When omitted the label is derived from cost_basename by stripping the
        'xr_cost_' prefix (legacy behaviour).
    """
    cost_file = os.path.join(out_path, run_name, str(year), f'{cost_basename}_{year}.nc')
    revenue_file = os.path.join(out_path, run_name, str(year), f'{revenue_basename}_{year}.nc')

    for f in [cost_file, revenue_file]:
        if not os.path.exists(f):
            raise FileNotFoundError(
                f"Scenario '{run_name}' | Year {year} | Missing: {os.path.basename(f)}\n"
                f"  Full path: {f}"
            )

    with xr.open_dataset(cost_file, chunks='auto') as ds_cost, \
            xr.open_dataset(revenue_file, chunks='auto') as ds_revenue:
        ds_revenue = filter_all_from_dims(ds_revenue).fillna(0)
        ds_cost = filter_all_from_dims(ds_cost).fillna(0)

        total_revenue = ds_revenue.sum(dim='source') if 'source' in ds_revenue.dims else ds_revenue
        total_cost = ds_cost.sum(dim='source') if 'source' in ds_cost.dims else ds_cost

        total_revenue, total_cost = xr.align(total_revenue, total_cost, join='outer', fill_value=0)
        profit = total_revenue - total_cost

    profit_out_path = os.path.join(out_path, run_name, str(year))
    os.makedirs(profit_out_path, exist_ok=True)
    if profit_category is None:
        profit_category = cost_basename.replace("xr_cost_", "")
    profit_filename = f'xr_profit_{profit_category}_{year}.nc'
    save2nc(profit, os.path.join(profit_out_path, profit_filename))
    return f"Profit calculated: {os.path.basename(out_path)}/{profit_filename}"


def calculate_policy_cost(year, output_path, run_all_names, cost_category, policy_type, cost_names):
    """Calculate policy cost (Carbon or Bio) based on profit differences."""
    profit_file_basename = f'xr_profit_{cost_category}_{year}.nc'
    output_part_name = f"xr_cost_{cost_category}"

    policy_pairs = {
        'carbon':  [run_all_names[1], run_all_names[0]],
        'bio':     [run_all_names[2], run_all_names[1]],
        'counter': [run_all_names[2], run_all_names[0]],
    }
    if policy_type not in policy_pairs:
        raise ValueError(f"Unknown policy_type '{policy_type}'. Use 'carbon', 'bio', or 'counter'.")

    calculate_diff_two_scenarios(
        policy_pairs[policy_type], cost_names,
        output_path, year, profit_file_basename, output_part_name,
    )


def calculate_transition_cost_diff(year, output_path, run_all_names, tran_cost_file,
                                    policy_type, cost_names):
    """Compute transition cost file diffs between two scenario groups."""
    tran_file_basename = f"{tran_cost_file}_{year}.nc"
    output_part_name = f"{tran_cost_file}_diff"

    policy_pairs = {
        'carbon':  [run_all_names[0], run_all_names[1]],
        'bio':     [run_all_names[1], run_all_names[2]],
        'counter': [run_all_names[0], run_all_names[2]],
    }
    if policy_type not in policy_pairs:
        raise ValueError(f"Unknown policy_type '{policy_type}'. Use 'carbon', 'bio', or 'counter'.")

    calculate_diff_two_scenarios(
        policy_pairs[policy_type], cost_names,
        output_path, year, tran_file_basename, output_part_name,
    )


def aggregate_and_save_cost(year, output_path, cost_names):
    """Aggregate cost components per scenario name and year; save original and amortised totals."""
    base_names = [
        'xr_cost_ag',
        'xr_cost_agricultural_management',
        'xr_cost_non_ag',
        'xr_transition_cost_ag2ag_diff',
    ]
    add_variants = [
        'xr_transition_cost_ag2non_ag_amortised_diff',
        'xr_transition_cost_ag2non_ag_diff',
    ]

    for cost_name in cost_names:
        file_dir = os.path.join(output_path, cost_name, str(year))

        for add_name in add_variants:
            data_type_names_all = base_names + [add_name]
            full_paths = [
                os.path.join(file_dir, f'{basename}_{cost_name}_{year}.nc')
                for basename in data_type_names_all
            ]

            total_sum_ds = None
            for file_path in full_paths:
                if not os.path.exists(file_path):
                    raise FileNotFoundError(
                        f"Scenario '{cost_name}' | Year {year} | Missing: {os.path.basename(file_path)}\n"
                        f"  Full path: {file_path}"
                    )
                with xr.open_dataset(file_path, chunks='auto') as ds:
                    ds = filter_all_from_dims(ds)
                    sum_dims = [d for d in ds.dims if d != 'cell']
                    summed = ds.sum(dim=sum_dims) if sum_dims else ds

                    if total_sum_ds is None:
                        total_sum_ds = summed
                    else:
                        total_sum_ds, summed = xr.align(
                            total_sum_ds, summed, join='outer', fill_value=0
                        )
                        total_sum_ds = total_sum_ds + summed

            am_type = 'amortised' if 'amortised' in add_name else 'original'
            final_path = os.path.join(file_dir, f'xr_total_cost_{cost_name}_{am_type}_{year}.nc')
            save2nc(total_sum_ds, final_path)


def aggregate_and_save_cost_sol(year, output_path, cost_names):
    """Aggregate solution-specific cost components (non_ag + amortised transition)."""
    base_names = [
        'xr_cost_non_ag',
        'xr_transition_cost_ag2non_ag_amortised_diff',
    ]

    for cost_name in cost_names:
        file_dir = os.path.join(output_path, cost_name, str(year))
        full_paths = [
            os.path.join(file_dir, f'{basename}_{cost_name}_{year}.nc')
            for basename in base_names
        ]
        final_path = os.path.join(file_dir, f'xr_total_sol_cost_{cost_name}_{year}.nc')

        total_sum_ds = None
        for file_path in full_paths:
            if not os.path.exists(file_path):
                raise FileNotFoundError(
                    f"Scenario '{cost_name}' | Year {year} | Missing: {os.path.basename(file_path)}\n"
                    f"  Full path: {file_path}"
                )
            with xr.open_dataset(file_path, chunks='auto') as ds:
                ds = filter_all_from_dims(ds)
                sum_dims = [d for d in ds.dims if d != 'cell']
                summed = ds.sum(dim=sum_dims) if sum_dims else ds

                if total_sum_ds is None:
                    total_sum_ds = summed
                else:
                    total_sum_ds, summed = xr.align(
                        total_sum_ds, summed, join='outer', fill_value=0
                    )
                    total_sum_ds = total_sum_ds + summed

        save2nc(total_sum_ds, final_path)


def aggregate_and_save_summary(year, output_path, data_type_names, input_files_names, type):
    """Sum all data_type files per scenario name and save a combined summary."""
    for input_files_name in input_files_names:
        file_dir = os.path.join(output_path, input_files_name, str(year))
        os.makedirs(file_dir, exist_ok=True)

        total_sum_ds = None
        for basename in data_type_names:
            file_path = os.path.join(file_dir, f'{basename}_{input_files_name}_{year}.nc')
            if not os.path.exists(file_path):
                raise FileNotFoundError(
                    f"Scenario '{input_files_name}' | Year {year} | Missing: {os.path.basename(file_path)}\n"
                    f"  Full path: {file_path}"
                )
            with xr.open_dataset(file_path, chunks='auto') as ds:
                filtered = filter_all_from_dims(ds)
                summed = filtered.sum(dim=[d for d in filtered.dims if d != 'cell'])

                if total_sum_ds is None:
                    total_sum_ds = summed
                else:
                    total_sum_ds, summed = xr.align(
                        total_sum_ds, summed, join='outer', fill_value=0
                    )
                    total_sum_ds += summed

        final_path = os.path.join(file_dir, f'xr_total_{type}_{input_files_name}_{year}.nc')
        save2nc(total_sum_ds, final_path)


def calculate_cell_price(input_file, year, base_dir, type, chunks='auto'):
    """Compute per-cell price as total_cost / total_benefit (amortised cost)."""
    output_path = os.path.join(base_dir, input_file, str(year),
                               f"xr_{type}_price_{input_file}_{year}.nc")
    cost_path = os.path.join(base_dir, input_file, str(year),
                             f"xr_total_cost_{input_file}_amortised_{year}.nc")
    env_path = os.path.join(base_dir, input_file, str(year),
                            f"xr_total_{type}_{input_file}_{year}.nc")

    for f in [cost_path, env_path]:
        if not os.path.exists(f):
            raise FileNotFoundError(
                f"Scenario '{input_file}' | Year {year} | Missing: {os.path.basename(f)}\n"
                f"  Full path: {f}"
            )

    with xr.open_dataarray(cost_path, chunks=chunks) as cost_da, \
            xr.open_dataarray(env_path, chunks=chunks) as env_da:
        mask_da = (cost_da >= 1) & (env_da >= 1)
        price_da = (cost_da / env_da).where(mask_da, np.nan)
        save2nc(price_da, output_path)


def calculate_cell_price_sol(input_file, year, base_dir, type, chunks='auto'):
    """Compute per-cell solution price as total_sol_cost / total_benefit."""
    output_path = os.path.join(base_dir, input_file, str(year),
                               f"xr_{type}_price_{input_file}_{year}.nc")
    cost_path = os.path.join(base_dir, input_file, str(year),
                             f"xr_total_sol_cost_{input_file}_{year}.nc")
    env_path = os.path.join(base_dir, input_file, str(year),
                            f"xr_total_{type}_{input_file}_{year}.nc")

    for f in [cost_path, env_path]:
        if not os.path.exists(f):
            raise FileNotFoundError(
                f"Scenario '{input_file}' | Year {year} | Missing: {os.path.basename(f)}\n"
                f"  Full path: {f}"
            )

    with xr.open_dataarray(cost_path, chunks=chunks) as cost_da, \
            xr.open_dataarray(env_path, chunks=chunks) as env_da:
        mask_da = (cost_da >= 1) & (env_da >= 1)
        price_da = (cost_da / env_da).where(mask_da, np.nan)
        save2nc(price_da, output_path)


# ==============================================================================
# Raster / GIS Functions
# ==============================================================================

def xarrays_to_tifs(env_cat, file_part, base_dir, tif_dir, data,
                    remove_negative=True, per_ha=True):
    """Convert a 2050 NetCDF to a GeoTIFF (optionally per-ha and/or clipped to ≥0)."""
    print(f"Processing {env_cat} - {file_part}")

    if file_part == 'total_cost':
        input_path = f"{base_dir}/{env_cat}/2050/xr_{file_part}_{env_cat}_amortised_2050.nc"
    else:
        input_path = f"{base_dir}/{env_cat}/2050/xr_{file_part}_{env_cat}_2050.nc"

    da = xr.open_dataarray(input_path)
    da = da.sum(dim=[d for d in da.dims if d != 'cell'])

    if remove_negative:
        da = da.where(da >= 0, np.nan)
    if per_ha:
        da = da / data.REAL_AREA
        out_tif = f"{tif_dir}/{env_cat}/xr_{file_part}_ha_{env_cat}_2050.tif"
    else:
        out_tif = f"{tif_dir}/{env_cat}/xr_{file_part}_cell_{env_cat}_2050.tif"

    os.makedirs(os.path.dirname(out_tif), exist_ok=True)
    nc_to_tif(data, da, out_tif)
    return out_tif


def xarrays_to_tifs_by_type(env_cat, file_part, base_dir, tif_dir, data,
                              sum_dim, remove_negative=False, per_ha=True):
    """Output one total GeoTIFF plus one per coordinate value along sum_dim."""
    print(f"Processing {env_cat} - {file_part} by {sum_dim}")
    input_path = f"{base_dir}/{env_cat}/2050/xr_{file_part}_2050.nc"
    da = xr.open_dataarray(input_path)

    if sum_dim not in da.dims:
        raise ValueError(f"{sum_dim} not in data dims {da.dims}.")

    # 1. Total (sum all non-cell dims)
    da_total = da.sum(dim=[d for d in da.dims if d != 'cell'])
    if per_ha:
        da_total = da_total / data.REAL_AREA
    if remove_negative:
        da_total = da_total.where(da_total >= 0, np.nan)
    out_total_tif = f"{tif_dir}/{env_cat}/xr_total_{file_part}_{env_cat}_2050.tif"
    os.makedirs(os.path.dirname(out_total_tif), exist_ok=True)
    nc_to_tif(data, da_total, out_total_tif)

    # 2. Per coordinate value along sum_dim
    results = [out_total_tif]
    for coord_val in da[sum_dim].values:
        da_slice = da.sel({sum_dim: coord_val})
        da_out = da_slice.sum(dim=[d for d in da_slice.dims if d != 'cell'])
        if per_ha:
            da_out = da_out / data.REAL_AREA
        if remove_negative:
            da_out = da_out.where(da_out >= 0, np.nan)
        out_tif = f"{tif_dir}/{env_cat}/xr_{file_part}_{env_cat}_{coord_val}_2050.tif"
        os.makedirs(os.path.dirname(out_tif), exist_ok=True)
        nc_to_tif(data, da_out, out_tif)
        results.append(out_tif)

    return results


def subtract_tifs(a_path, b_path, out_path):
    """Compute A − B pixel-wise; pixels ≤0 or fully NaN are set to nodata."""
    with rasterio.open(a_path) as A, rasterio.open(b_path) as B:
        if (A.width, A.height) != (B.width, B.height) \
                or A.transform != B.transform or A.crs != B.crs:
            raise ValueError("Input rasters differ in size/transform/CRS.")

        arr_a = A.read(1, masked=True).filled(np.nan).astype(np.float32)
        arr_b = B.read(1, masked=True).filled(np.nan).astype(np.float32)
        arr_a[arr_a < 0] = np.nan
        arr_b[arr_b < 0] = np.nan

        all_nan_mask = np.isnan(arr_a) & np.isnan(arr_b)
        out = np.nan_to_num(arr_a, nan=0.0) - np.nan_to_num(arr_b, nan=0.0)
        out[all_nan_mask] = np.nan
        out[out <= 0] = np.nan

        nodata_value = -9999
        profile = A.profile.copy()
        profile.update(dtype="float32", compress="lzw", nodata=nodata_value)
        out = np.where(np.isnan(out), nodata_value, out)

        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(out, 1)


def plus_tifs(base_dir, env_cat, cost_names, outpath_part, remove_negative=True):
    """Sum a list of per-cell and per-ha GeoTIFFs and write the combined result."""
    for unit in ['cell', 'ha']:
        cost_arrs = []
        cost_profile = None
        for fname_part in cost_names:
            fname = f"{base_dir}/{env_cat}/xr_{fname_part}_{unit}_{env_cat}_2050.tif"
            with rasterio.open(fname) as src:
                arr = src.read(1, masked=True).filled(np.nan).astype(np.float32)
                cost_arrs.append(arr)
                if cost_profile is None:
                    cost_profile = src.profile.copy()

        cost_stack = np.stack(cost_arrs, axis=0)
        all_nan_mask = np.all(np.isnan(cost_stack), axis=0)
        cost_sum = np.sum(np.nan_to_num(cost_stack, nan=0.0), axis=0)
        cost_sum[all_nan_mask] = np.nan

        nodata_value = -9999
        cost_sum[np.isnan(cost_sum)] = nodata_value
        if remove_negative:
            cost_sum[cost_sum < 1] = nodata_value

        profile = cost_profile.copy()
        profile.update(dtype="float32", compress="lzw", nodata=nodata_value)
        out_path = f"{base_dir}/{env_cat}/xr_{outpath_part}_{unit}_{env_cat}_2050.tif"
        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(cost_sum, 1)


def divide_tifs(base_dir, env_cat, cost_name, benefit_name, outpath_part):
    """Divide a cost GeoTIFF by a benefit GeoTIFF; mask pixels < 1."""
    cost_path = f"{base_dir}/{env_cat}/xr_{cost_name}_cell_{env_cat}_2050.tif"
    benefit_path = f"{base_dir}/{env_cat}/xr_{benefit_name}_cell_{env_cat}_2050.tif"

    with rasterio.open(cost_path) as src:
        cost_arr = src.read(1, masked=True).filled(np.nan).astype(np.float32)
        cost_arr[cost_arr < 1] = np.nan
        cost_profile = src.profile.copy()

    with rasterio.open(benefit_path) as src:
        benefit_arr = src.read(1, masked=True).filled(np.nan).astype(np.float32)
        benefit_arr[cost_arr < 1] = np.nan

    out = cost_arr / benefit_arr

    nodata_value = -9999
    out[np.isnan(out)] = nodata_value
    out[benefit_arr < 1] = nodata_value
    out[cost_arr < 1] = nodata_value

    profile = cost_profile.copy()
    profile.update(dtype="float32", compress="lzw", nodata=nodata_value)
    out_path = f"{base_dir}/{env_cat}/xr_{outpath_part}_{env_cat}_2050.tif"
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(out, 1)


def create_shp(env_cat, shp_name, file_parts, tif_dir):
    """Run zonal statistics for each file_part TIF and save as shapefile."""
    for file_part in file_parts:
        tif_env_dir = os.path.join(tif_dir, env_cat)
        input_tif_name = f'xr_{file_part}_{env_cat}_2050.tif'
        out_shp = os.path.join(tif_env_dir, shp_name,
                               f'{shp_name}_{file_part}_{env_cat}_2050.shp')
        os.makedirs(os.path.dirname(out_shp), exist_ok=True)
        shp_path = f"../Map/{shp_name}.shp"
        zonal_stats_rasterized(tif_env_dir, input_tif_name, shp_path, out_shp)


def zonal_stats_rasterized(input_tif_dir, input_tif_name, shp_path, out_shp,
                            extra_nodata_vals=(-9999.0,), drop_allnan=True):
    """Compute sum and mean zonal statistics by rasterizing vector polygons."""
    gdf = gpd.read_file(shp_path)
    input_tif = os.path.join(input_tif_dir, input_tif_name)

    with rasterio.open(input_tif) as src:
        img_m = src.read(1, masked=True)
        transform = src.transform
        shape = (src.height, src.width)
        if gdf.crs is not None and src.crs is not None and gdf.crs != src.crs:
            gdf = gdf.to_crs(src.crs)

    n_shapes = len(gdf)
    arr = img_m.filled(np.nan).astype('float64', copy=False)
    for nd in (extra_nodata_vals or ()):
        arr[np.isclose(arr, nd)] = np.nan

    shapes = ((geom, i + 1) for i, geom in enumerate(gdf.geometry))
    id_arr = rasterize(shapes, out_shape=shape, transform=transform, fill=0, dtype="int32")

    valid_mask = (id_arr > 0) & np.isfinite(arr)
    if not np.any(valid_mask):
        if drop_allnan:
            print("All polygons have no valid pixels; nothing written.")
            return
        gdf["sum"] = np.nan
        gdf["mean"] = np.nan
        gdf.to_file(out_shp)
        print(f"Saved {out_shp} (all NaN)")
        return

    vals = arr[valid_mask]
    ids = id_arr[valid_mask]

    sum_per_id = np.bincount(ids, weights=vals, minlength=n_shapes + 1)
    cnt_per_id = np.bincount(ids, minlength=n_shapes + 1)
    sum_stat = sum_per_id[1:]
    cnt_stat = cnt_per_id[1:]

    mean_stat = np.full_like(sum_stat, np.nan, dtype="float64")
    np.divide(sum_stat, cnt_stat, out=mean_stat, where=cnt_stat > 0)

    if 'total_carbon' in input_tif_name:
        sum_stat = sum_stat / 1e6
        mean_stat = mean_stat / 1e6

    gdf["sum"] = sum_stat
    gdf["mean"] = mean_stat
    gdf["count"] = cnt_stat

    if drop_allnan:
        before = len(gdf)
        gdf = gdf[gdf["count"] > 0].copy()
        print(f"Removed {before - len(gdf)} all-NaN polygons.")
        if gdf.empty:
            print("No features remain after filtering; nothing written.")
            return

    gdf.to_file(out_shp)
    print(f"Saved {out_shp} ({len(gdf)} features)")


# ==============================================================================
# Pipeline
# ==============================================================================

def main(task_dir, njobs):
    output_path = f'{task_dir}/carbon_price/0_base_data'
    os.makedirs(output_path, exist_ok=True)
    tprint(f"Task directory: {task_dir}")

    # --- File lists ---
    area_files    = ['xr_area_agricultural_landuse', 'xr_area_agricultural_management',
                     'xr_area_non_agricultural_landuse']
    cost_files    = ['xr_economics_ag_cost', 'xr_economics_am_cost', 'xr_economics_non_ag_cost',
                     'xr_transition_cost_ag2ag', 'xr_transition_cost_ag2non_ag']
    revenue_files = ['xr_economics_ag_revenue', 'xr_economics_am_revenue', 'xr_economics_non_ag_revenue']
    carbon_files  = ['xr_GHG_ag', 'xr_GHG_ag_management', 'xr_GHG_non_ag', 'xr_transition_GHG']
    bio_files     = ['xr_biodiversity_GBF2_priority_ag', 'xr_biodiversity_GBF2_priority_ag_management',
                     'xr_biodiversity_GBF2_priority_non_ag']
    carbon_sol_files = ['xr_GHG_ag', 'xr_GHG_non_ag']
    bio_sol_files    = ['xr_biodiversity_GBF2_priority_ag', 'xr_biodiversity_GBF2_priority_non_ag']
    amortize_files   = ['xr_transition_cost_ag2non_ag']
    economic_files   = config.economic_files
    economic_sol_files = ['xr_economics_non_ag_cost', 'xr_transition_cost_ag2non_ag_amortised',
                          'xr_economics_non_ag_revenue']
    env_files = carbon_files + bio_files

    # --- Scenario names ---
    input_files_0 = config.input_files_0
    input_files_1 = config.input_files_1
    input_files_2 = config.input_files_2
    input_files = input_files_0 + input_files_1 + input_files_2
    input_all_names = [input_files_0, input_files_1, input_files_2]

    carbon_names              = config.carbon_names
    carbon_bio_names          = config.carbon_bio_names
    counter_carbon_bio_names  = config.counter_carbon_bio_names
    output_all_names = carbon_names + carbon_bio_names + counter_carbon_bio_names

    # Policy config: (policy_type, output_names)
    policy_config = [
        ('carbon',  carbon_names),
        ('bio',     carbon_bio_names),
        ('counter', counter_carbon_bio_names),
    ]

    years = list(range(2010, 2051))
    unique_input_files = list(dict.fromkeys(input_files))
    copy_files = cost_files + revenue_files + carbon_files + bio_files + area_files

    start_time = time.time()

    # ==========================================================================
    # Stage 1: File copy and amortisation
    # ==========================================================================
    tprint("\n--- Stage 1: File copy ---")
    for input_file in unique_input_files:
        origin_path_name = get_path(config.TASK_NAME, input_file)
        target_path_name = os.path.join(output_path, input_file)
        tprint(f"  -> Copying: {origin_path_name}")

        files_in_memory = extract_files_from_zip(
            origin_path_name, copy_files, years, allow_missing_2010=True
        )
        _run_or_parallel(
            njobs, process_single_file_from_memory,
            [(fb, vp, y, target_path_name, ('source',))
             for (vp, y), fb in files_in_memory.items()],
        )
    tprint("File copy complete.")

    tprint("\n--- Stage 1b: Amortise transition costs ---")
    if njobs == 0:
        for run_name in unique_input_files:
            amortize_costs(os.path.join(output_path, run_name), amortize_files[0], years, njobs=0)
    else:
        Parallel(n_jobs=5, backend="loky")(
            delayed(amortize_costs)(
                os.path.join(output_path, run_name), amortize_files[0], years, njobs=1
            )
            for run_name in unique_input_files
        )
    tprint("Amortisation complete.")

    # ==========================================================================
    # Stage 2: Environment (carbon & bio) diff and summary
    # ==========================================================================
    tprint("\n--- Stage 2a: Carbon & bio diff calculation ---")
    for env_file in env_files:
        _run_or_parallel(
            njobs, calculate_env_diff,
            [(y, output_path, input_all_names, env_file, output_all_names) for y in years[1:]],
        )

    tprint("\n--- Stage 2b: Aggregate carbon & bio summaries ---")
    for summary_type, file_list in [('carbon', carbon_files), ('bio', bio_files),
                                    ('sol_carbon', carbon_sol_files), ('sol_bio', bio_sol_files)]:
        _run_or_parallel(
            njobs, aggregate_and_save_summary,
            [(y, output_path, file_list, output_all_names, summary_type) for y in years[1:]],
        )
    tprint("Stage 2 complete.")

    # ==========================================================================
    # Stage 3: Profit calculation
    # ==========================================================================
    tprint("\n--- Stage 3: Profit calculation ---")
    all_run_names = [r for run_names in input_all_names for r in run_names]
    # Explicit (cost_file, revenue_file, profit_category) triples so the
    # profit file names stay as xr_profit_{ag|agricultural_management|non_ag}_YEAR.nc
    # even though the source files were renamed.
    profit_triples = [
        ('xr_economics_ag_cost',      'xr_economics_ag_revenue',      'ag'),
        ('xr_economics_am_cost',      'xr_economics_am_revenue',      'agricultural_management'),
        ('xr_economics_non_ag_cost',  'xr_economics_non_ag_revenue',  'non_ag'),
    ]
    for cost_base, rev_base, profit_cat in profit_triples:
        for run_name in all_run_names:
            _run_or_parallel(
                njobs, calculate_profit_for_run,
                [(y, output_path, run_name, cost_base, rev_base, profit_cat) for y in years],
            )
    tprint("Stage 3 complete.")

    # ==========================================================================
    # Stage 4: Policy cost calculation
    # ==========================================================================
    tprint("\n--- Stage 4: Policy cost calculation ---")
    for category in ['agricultural_management', 'ag', 'non_ag']:
        for policy_type, cost_names in policy_config:
            _run_or_parallel(
                njobs, calculate_policy_cost,
                [(y, output_path, input_all_names, category, policy_type, cost_names)
                 for y in years[1:]],
            )
    tprint("Stage 4 complete.")

    # ==========================================================================
    # Stage 5: Transition cost diff calculation
    # ==========================================================================
    tprint("\n--- Stage 5: Transition cost diff calculation ---")
    independent_tran_files = [
        'xr_transition_cost_ag2ag',
        'xr_transition_cost_ag2non_ag',
        'xr_transition_cost_ag2non_ag_amortised',
    ]
    tran_njobs = math.ceil(njobs / 2)
    for tran_file in independent_tran_files:
        tprint(f"  Processing: {tran_file}")
        for policy_type, cost_names in policy_config:
            _run_or_parallel(
                tran_njobs, calculate_transition_cost_diff,
                [(y, output_path, input_all_names, tran_file, policy_type, cost_names)
                 for y in years[1:]],
            )
    tprint("Stage 5 complete.")

    # ==========================================================================
    # Stage 6: Cost aggregation
    # ==========================================================================
    tprint("\n--- Stage 6: Cost aggregation ---")
    output_name_groups = [carbon_names, carbon_bio_names, counter_carbon_bio_names]
    for fn in [aggregate_and_save_cost, aggregate_and_save_cost_sol]:
        for grp in output_name_groups:
            _run_or_parallel(njobs, fn, [(y, output_path, grp) for y in years[1:]])
    tprint("Stage 6 complete.")

    # ==========================================================================
    # Stage 7: Price calculation
    # ==========================================================================
    tprint("\n--- Stage 7: Price calculation ---")
    for fn in [calculate_cell_price, calculate_cell_price_sol]:
        for ptype in ['carbon', 'bio']:
            _run_or_parallel(
                njobs, fn,
                [(f, y, output_path, ptype) for f in output_all_names for y in years[1:]],
            )
    tprint("Stage 7 complete.")

    # ==========================================================================
    # Excel summaries
    # ==========================================================================
    excel_path = f"../../../output/{config.TASK_NAME}/carbon_price/1_excel"
    os.makedirs(excel_path, exist_ok=True)

    tprint("\n--- Excel: carbon summaries ---")
    for input_file in unique_input_files:
        tprint(f"  {input_file}")
        summarize_netcdf_to_excel(input_file, years[1:], carbon_files, njobs, 'carbon')
    tprint("\n--- Excel: biodiversity summaries ---")
    for input_file in unique_input_files:
        tprint(f"  {input_file}")
        summarize_netcdf_to_excel(input_file, years[1:], bio_files, njobs, 'biodiversity')
    tprint("\n--- Excel: economic summaries ---")
    for input_file in unique_input_files:
        tprint(f"  {input_file}")
        summarize_netcdf_to_excel(input_file, years[1:], economic_files,
                                  math.ceil(njobs / 2), 'economic')

    # --- Excel: profit-based costs per scenario pair ---
    profit_lists = [
        [create_profit_for_cost(excel_path, f) for f in files]
        for files in [input_files_0, input_files_1, input_files_2]
    ]
    profit_0_list, profit_1_list, profit_2_list = profit_lists

    for i in range(len(input_files_1)):
        for diff, name in [
            (profit_0_list[i] - profit_1_list[i], carbon_names[i]),
            (profit_1_list[i] - profit_2_list[i], carbon_bio_names[i]),
            (profit_0_list[i] - profit_2_list[i], counter_carbon_bio_names[i]),
        ]:
            diff.columns = diff.columns.str.replace('profit', '')
            diff['Total'] = diff.sum(axis=1)
            diff.to_excel(os.path.join(excel_path, f'1_Cost_{name}.xlsx'))

    # --- Excel: processed carbon and bio time-series ---
    for input_file in dict.fromkeys(input_files):
        df = pd.read_excel(
            os.path.join(excel_path, f'0_Origin_carbon_{input_file}.xlsx'), index_col=0
        )
        df.columns = df.columns.str.replace(' GHG', '')
        new_rows_list = []
        for i in range(1, len(df)):
            new_row = df.iloc[i].copy() * -1
            new_row.iloc[0] = -df.iloc[i, 0] + df.iloc[i - 1, 0]
            new_rows_list.append(new_row)
        new_df = pd.DataFrame(new_rows_list, index=df.index[1:])
        new_df['Total'] = new_df.sum(axis=1)
        new_df.to_excel(os.path.join(excel_path, f'1_Processed_carbon_{input_file}.xlsx'))

    for input_file in dict.fromkeys(input_files):
        df = pd.read_excel(
            os.path.join(excel_path, f'0_Origin_biodiversity_{input_file}.xlsx'), index_col=0
        )
        df.columns = df.columns.str.replace(' biodiversity', '')
        new_rows_list = []
        for i in range(1, len(df)):
            new_row = df.iloc[i].copy()
            new_row.iloc[0] = df.iloc[i, 0] - df.iloc[i - 1, 0]
            new_rows_list.append(new_row)
        new_df = pd.DataFrame(new_rows_list, index=df.index[1:])
        new_df['Total'] = new_df.sum(axis=1)
        new_df.to_excel(os.path.join(excel_path, f'1_Processed_bio_{input_file}.xlsx'))

    # --- Excel: cost/carbon/bio summary tables ---
    colnames = ["GHG benefits (Mt CO2e)", "Carbon cost (M AUD$)", "Average Carbon price (AUD$/t CO2e)"]
    _run_or_parallel(
        njobs, create_summary,
        [(name, years[1:], output_path, 'carbon', colnames) for name in output_all_names],
    )

    colnames = ["Biodiversity benefits (Mt CO2e)", "Biodiversity cost (M AUD$)",
                "Average Biodiversity price (AUD$/t CO2e)"]
    _run_or_parallel(
        njobs, create_summary,
        [(name, years[1:], output_path, 'bio', colnames) for name in output_all_names],
    )

    # --- NetCDF summaries ---
    summarize_to_category(output_all_names, years[1:], carbon_files, 'xr_total_carbon', n_jobs=41)
    summarize_to_category(output_all_names, years[1:], bio_files, 'xr_total_bio', n_jobs=41)
    summarize_to_category(unique_input_files, years[1:], carbon_files,
                          'xr_total_carbon_original', n_jobs=41, scenario_name=False)
    summarize_to_category(unique_input_files, years[1:], bio_files,
                          'xr_total_bio_original', n_jobs=41, scenario_name=False)

    profit_da = summarize_to_category(unique_input_files, years[1:], economic_files,
                                       'xr_cost_for_profit', n_jobs=41, scenario_name=False)
    build_profit_and_cost_nc(
        profit_da, list(dict.fromkeys(input_files_0)), input_files_1, input_files_2,
        carbon_names, carbon_bio_names, counter_carbon_bio_names,
    )
    make_prices_nc(output_all_names)

    summarize_to_category(output_all_names, years[1:], carbon_sol_files,
                          'xr_total_sol_carbon', n_jobs=41)
    summarize_to_category(output_all_names, years[1:], bio_sol_files,
                          'xr_total_sol_bio', n_jobs=41)

    profit_sol_da = summarize_to_category(unique_input_files, years[1:], economic_sol_files,
                                           'xr_sol_cost_for_profit', n_jobs=41, scenario_name=False)
    build_sol_profit_and_cost_nc(
        profit_sol_da, list(dict.fromkeys(input_files_0)), input_files_1, input_files_2,
        carbon_names, carbon_bio_names, counter_carbon_bio_names,
    )
    make_sol_prices_nc(output_all_names)

    # --- Dimension-level summaries ---
    diff_files    = ['xr_cost_agricultural_management', 'xr_cost_non_ag',
                     'xr_transition_cost_ag2non_ag_amortised_diff',
                     'xr_GHG_ag_management', 'xr_GHG_non_ag',
                     'xr_biodiversity_GBF2_priority_ag_management',
                     'xr_biodiversity_GBF2_priority_non_ag']
    diff_dim_names = ['am', 'lu', 'To-land-use', 'am', 'lu', 'am', 'lu']
    for file, dim_name in zip(diff_files, diff_dim_names):
        summarize_to_type(
            scenarios=output_all_names, years=years[1:], file=file, keep_dim=dim_name,
            output_file=file, var_name='data', scale=1e6, n_jobs=njobs, dtype='float32',
        )

    orig_files    = ['xr_area_agricultural_management', 'xr_area_non_agricultural_landuse',
                     'xr_biodiversity_GBF2_priority_ag_management',
                     'xr_biodiversity_GBF2_priority_non_ag',
                     'xr_GHG_ag_management', 'xr_GHG_non_ag',
                     'xr_economics_am_cost', 'xr_economics_non_ag_cost',
                     'xr_transition_cost_ag2non_ag_amortised']
    orig_dim_names = ['am', 'lu', 'am', 'lu', 'am', 'lu', 'am', 'lu', 'To-land-use']
    for file, dim_name in zip(orig_files, orig_dim_names):
        summarize_to_type(
            scenarios=unique_input_files, years=years[1:], file=file, keep_dim=dim_name,
            output_file=file, var_name='data', scale=1e6, n_jobs=njobs, dtype='float32',
            scenario_name=False,
        )

    # ==========================================================================
    # Stage 8: TIF export
    # ==========================================================================
    tif_dir = f"../../../output/{config.TASK_NAME}/carbon_price/4_tif"
    data = get_data_RES(config.TASK_NAME, input_files_0[0])

    cost_file_parts    = ['total_cost', 'cost_agricultural_management', 'cost_non_ag',
                          'transition_cost_ag2non_ag_amortised_diff']
    GHG_file_parts     = ['total_carbon', 'GHG_ag_management', 'GHG_non_ag']
    bio_file_parts     = ['total_bio', 'biodiversity_GBF2_priority_ag_management',
                          'biodiversity_GBF2_priority_non_ag']
    agmgt_file_parts   = ['area_agricultural_management']
    nonag_file_parts   = ['area_non_agricultural_landuse']

    cost_tasks  = [(c, fp) for c in output_all_names for fp in cost_file_parts]
    GHG_tasks   = [(c, fp) for c in output_all_names for fp in GHG_file_parts]
    bio_tasks   = [(c, fp) for c in output_all_names for fp in bio_file_parts]
    agmgt_tasks = [(c, fp) for c in input_files for fp in agmgt_file_parts]
    nonag_tasks = [(c, fp) for c in input_files for fp in nonag_file_parts]

    Parallel(n_jobs=njobs)(
        delayed(xarrays_to_tifs)(c, fp, output_path, tif_dir, data)
        for c, fp in cost_tasks
    )
    Parallel(n_jobs=njobs)(
        delayed(xarrays_to_tifs)(c, fp, output_path, tif_dir, data, remove_negative=False)
        for c, fp in GHG_tasks + bio_tasks
    )
    Parallel(n_jobs=njobs)(
        delayed(xarrays_to_tifs)(c, fp, output_path, tif_dir, data, per_ha=False)
        for c, fp in cost_tasks
    )
    Parallel(n_jobs=njobs)(
        delayed(xarrays_to_tifs)(c, fp, output_path, tif_dir, data,
                                 remove_negative=False, per_ha=False)
        for c, fp in GHG_tasks + bio_tasks
    )
    Parallel(n_jobs=njobs)(
        delayed(xarrays_to_tifs_by_type)(c, fp, output_path, tif_dir, data, sum_dim='am')
        for c, fp in agmgt_tasks
    )
    Parallel(n_jobs=njobs)(
        delayed(xarrays_to_tifs_by_type)(c, fp, output_path, tif_dir, data, sum_dim='lu')
        for c, fp in nonag_tasks
    )

    # --- Solution cost/benefit TIFs ---
    solution_cost_parts        = ['cost_non_ag', 'transition_cost_ag2non_ag_amortised_diff']
    solution_ghg_benefit_parts = ['GHG_non_ag']
    solution_bio_benefit_parts = ['biodiversity_GBF2_priority_non_ag']

    Parallel(n_jobs=njobs)(
        delayed(plus_tifs)(tif_dir, c, solution_cost_parts, "total_sol_cost")
        for c in output_all_names
    )
    Parallel(n_jobs=njobs)(
        delayed(plus_tifs)(tif_dir, c, solution_ghg_benefit_parts,
                           "total_sol_ghg_benefit", remove_negative=False)
        for c in output_all_names
    )
    Parallel(n_jobs=njobs)(
        delayed(divide_tifs)(tif_dir, c, 'total_sol_cost', 'total_sol_ghg_benefit',
                             "carbon_sol_price")
        for c in output_all_names
    )
    Parallel(n_jobs=njobs)(
        delayed(plus_tifs)(tif_dir, c, solution_bio_benefit_parts,
                           "total_sol_bio_benefit", remove_negative=False)
        for c in output_all_names
    )
    Parallel(n_jobs=njobs)(
        delayed(divide_tifs)(tif_dir, c, 'total_sol_cost', 'total_sol_bio_benefit',
                             "bio_sol_price")
        for c in output_all_names
    )

    # --- Diff: bio price = counterfactual − carbon-only ---
    tif_path_1 = os.path.join(tif_dir, 'carbon_high_50',
                               "xr_carbon_sol_price_carbon_high_50_2050.tif")
    tif_path_2 = os.path.join(tif_dir, 'Counterfactual_carbon_high_bio_50',
                               "xr_carbon_sol_price_Counterfactual_carbon_high_bio_50_2050.tif")
    tif_output = os.path.join(tif_dir, 'carbon_high_bio_50',
                               "xr_carbon_sol_price_carbon_high_bio_50_2050.tif")
    subtract_tifs(tif_path_2, tif_path_1, tif_output)

    # --- Summary ---
    total_time = time.time() - start_time
    tprint("\n" + "=" * 80)
    tprint("All tasks complete.")
    tprint(f"Total elapsed time: {total_time / 3600:.2f} hours")
    tprint("=" * 80)


def run(task_dir, njobs):
    save_dir = os.path.join(task_dir, 'carbon_price')
    log_path = os.path.join(save_dir, 'log_0_preprocess')

    @LogToFile(log_path)
    def _run():
        stop_event = threading.Event()
        memory_thread = threading.Thread(
            target=log_memory_usage, args=(save_dir, 'a', 1, stop_event)
        )
        memory_thread.start()
        try:
            print('\n')
            main(task_dir, njobs)
        except Exception as e:
            print(f"An error occurred: {e}")
            raise
        finally:
            stop_event.set()
            memory_thread.join()

    return _run()


if __name__ == "__main__":
    task_name = config.TASK_NAME
    njobs = math.ceil(41 / 1)
    task_dir = f'../../../output/{task_name}'
    run(task_dir, njobs)
