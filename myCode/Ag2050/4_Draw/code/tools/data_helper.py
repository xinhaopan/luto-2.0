"""
data_helper.py
Utilities for loading LUTO2 CSV outputs into long-format DataFrames.
Supports both unpacked directories and Run_Archive.zip archives.
"""
import io
import os
import re
import zipfile
import numpy as np
import pandas as pd
import rasterio
from pyproj import CRS, Transformer
from joblib import Parallel, delayed
from tools.parameters import TASK_ROOT, TIFF_DIR


# ── Path resolution ──────────────────────────────────────────────────────────

def _scenario_root(scenario):
    """Return base folder for a scenario under TASK_ROOT."""
    return f"../../../../output/{TASK_ROOT}/{scenario}"


def get_zip_info(scenario):
    """Return (zip_path, internal_prefix) if Run_Archive.zip exists, else None."""
    zip_path = os.path.join(_scenario_root(scenario), "Run_Archive.zip")
    if not os.path.exists(zip_path):
        return None
    with zipfile.ZipFile(zip_path) as z:
        subdir = next(
            n.split('/')[1]
            for n in z.namelist()
            if n.startswith('output/') and '2010-2050' in n
        )
    return zip_path, f"output/{subdir}"


def get_path(scenario):
    """Return timestamped output directory for an unpacked scenario run.
    Raises FileNotFoundError if the directory does not exist.
    """
    output_path = os.path.join(_scenario_root(scenario), "output")
    subdir = next(s for s in os.listdir(output_path) if "2010-2050" in s)
    return os.path.join(output_path, subdir)


def _list_years(base):
    """Sorted list of simulation years found under base/out_YYYY directories."""
    return sorted(
        int(m.group(1))
        for f in os.listdir(base)
        if (m := re.search(r"out_(\d+)", f))
    )


def _list_years_zip(zip_path, prefix):
    """Sorted list of simulation years from zip file."""
    with zipfile.ZipFile(zip_path) as z:
        names = z.namelist()
    years = sorted(set(
        int(m.group(1))
        for n in names
        if n.startswith(prefix + '/')
        if (m := re.search(r"out_(\d+)", n))
    ))
    return years


def extract_nc_layer_as_tiff(scenario, nc_stem, nc_sel, year, output_dir=None):
    """Read an xr_* NetCDF layer from zip, reconstruct 2D map, save as GeoTIFF.

    Parameters
    ----------
    scenario  : e.g. 'Run_1_SCN_AgS1'
    nc_stem   : stem inside out_{year}/, e.g. 'map_lumap', 'dvar_am', 'dvar_non_ag'
    nc_sel    : dict for .sel(), e.g. {'lm':'ALL'}, {'am':'ALL'}, {'lu':'ALL'}
    year      : output year, e.g. 2050
    output_dir: directory to write extracted tiff (defaults to TIFF_DIR)

    Returns path to the GeoTIFF, or None on failure.
    """
    if output_dir is None:
        output_dir = TIFF_DIR
    import xarray as xr
    import cf_xarray as cfxr

    info = get_zip_info(scenario)
    if info is None:
        return None
    zip_path, prefix = info

    nc_file    = f"{prefix}/out_{year}/xr_{nc_stem}_{year}.nc"
    tmpl_file  = f"{prefix}/out_{year}/xr_map_template_{year}.nc"
    sel_tag    = "_".join(f"{k}{v}" for k, v in nc_sel.items())
    out_path   = os.path.join(output_dir, f"_extracted_{scenario}_{nc_stem}_{sel_tag}.tiff")
    os.makedirs(output_dir, exist_ok=True)

    if os.path.exists(out_path):
        return out_path

    with zipfile.ZipFile(zip_path) as z:
        if nc_file not in z.namelist():
            print(f"Not found in zip: {nc_file}")
            return None
        with z.open(nc_file) as f:
            ds = xr.open_dataset(io.BytesIO(f.read()))
        xr_arr = cfxr.decode_compress_to_multi_index(ds, 'layer')['data']
        valid_layers = xr_arr['layer'].to_index().to_frame().to_dict(orient='records')
        if nc_sel not in valid_layers:
            print(f"Layer {nc_sel} not in {nc_file} — skipping (scenario may have no data for this category)")
            return None
        arr = xr_arr.sel(**nc_sel).squeeze()

        if tmpl_file not in z.namelist():
            print(f"Template not found in zip: {tmpl_file}")
            return None
        with z.open(tmpl_file) as f:
            ds_tmpl = xr.open_dataset(io.BytesIO(f.read()))

    # Reconstruct 2D map using template
    tmpl = ds_tmpl['layer'].values.astype('float32')
    valid = tmpl >= 0
    tmpl[valid] = arr.values
    tmpl[~valid] = -9999

    # Write GeoTIFF using template CRS and transform
    crs_wkt   = ds_tmpl['spatial_ref'].attrs['crs_wkt']
    transform = ds_tmpl['layer'].rio.transform()

    import rasterio
    from rasterio.transform import from_bounds
    from rasterio.crs import CRS as RioCRS
    with rasterio.open(
        out_path, 'w',
        driver='GTiff',
        height=tmpl.shape[0],
        width=tmpl.shape[1],
        count=1,
        dtype='float32',
        crs=RioCRS.from_wkt(crs_wkt),
        transform=transform,
        nodata=-9999,
    ) as dst:
        dst.write(tmpl, 1)

    return out_path


def extract_tiff_from_zip(scenario, tiff_name, output_dir):
    """Extract a single tiff from Run_Archive.zip to output_dir.

    Parameters
    ----------
    scenario  : scenario folder name, e.g. 'Run_1_SCN_AgS1'
    tiff_name : filename inside out_2050/, e.g. 'lumap_2050.tiff'
    output_dir: directory to write extracted file

    Returns path to extracted file, or None if not found.
    """
    info = get_zip_info(scenario)
    if info is None:
        return None
    zip_path, prefix = info
    internal = f"{prefix}/out_2050/{tiff_name}"
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"_extracted_{scenario}_{tiff_name}")
    if not os.path.exists(out_path):
        with zipfile.ZipFile(zip_path) as z:
            if internal not in z.namelist():
                print(f"Not found in zip: {internal}")
                return None
            with z.open(internal) as src, open(out_path, 'wb') as dst:
                dst.write(src.read())
    return out_path


# ── Core data loader ──────────────────────────────────────────────────────────

def _read_csv_from_zip(zip_path, prefix, year, csv_name):
    """Read a single year CSV from zip, return DataFrame or None."""
    internal = f"{prefix}/out_{year}/{csv_name}_{year}.csv"
    with zipfile.ZipFile(zip_path) as z:
        if internal not in z.namelist():
            return None
        with z.open(internal) as f:
            return pd.read_csv(io.BytesIO(f.read()))


def load_long(input_files, csv_name, value_col, category_col=None,
              condition_col=None, condition_val=None,
              unit_scale=1e6, unit_adopt=True):
    """Load CSVs from multiple scenarios/years into a single long DataFrame.
    Transparently handles both unpacked directories and Run_Archive.zip.

    Returns
    -------
    DataFrame with columns: year, scenario, category, value
    """
    scale = unit_scale if unit_adopt else 1.0
    rows = []

    for scenario in input_files:
        info = get_zip_info(scenario)
        if info is not None:
            # ── zip-based loading ──────────────────────────────────────────
            zip_path, prefix = info
            for year in _list_years_zip(zip_path, prefix):
                df = _read_csv_from_zip(zip_path, prefix, year, csv_name)
                if df is None:
                    continue
                rows.extend(_extract_rows(df, year, scenario, value_col,
                                          category_col, condition_col, condition_val, scale))
        else:
            # ── normal file-based loading ──────────────────────────────────
            base = get_path(scenario)
            for year in _list_years(base):
                fp = os.path.join(base, f"out_{year}", f"{csv_name}_{year}.csv")
                if not os.path.exists(fp):
                    continue
                df = pd.read_csv(fp)
                rows.extend(_extract_rows(df, year, scenario, value_col,
                                          category_col, condition_col, condition_val, scale))

    return pd.DataFrame(rows, columns=["year", "scenario", "category", "value"])


def _extract_rows(df, year, scenario, value_col, category_col,
                  condition_col, condition_val, scale):
    """Apply filters and aggregate one DataFrame into row dicts."""
    if condition_col is not None and condition_val is not None:
        cols = condition_col if isinstance(condition_col, list) else [condition_col]
        vals = condition_val if isinstance(condition_val, list) else [condition_val]
        for c, v in zip(cols, vals):
            df = df[df[c] == v]

    rows = []
    if category_col is None:
        rows.append({"year": year, "scenario": scenario,
                     "category": "Total",
                     "value": df[value_col].sum() / scale})
    else:
        for cat, g in df.groupby(category_col):
            rows.append({"year": year, "scenario": scenario,
                         "category": cat,
                         "value": g[value_col].sum() / scale})
    return rows


# ── Aggregation & manipulation ────────────────────────────────────────────────

def group_by_mapping(df, mapping_file, from_col, to_col, sheet=None):
    """Map categories via an Excel lookup table and aggregate values.

    Returns a long DataFrame with the same columns as the input,
    but with *category* values replaced by the mapped group names.
    """
    kw = {"sheet_name": sheet} if sheet else {}
    mapping = pd.read_excel(mapping_file, **kw)[[from_col, to_col]].dropna()
    return (
        df.merge(mapping, left_on="category", right_on=from_col, how="inner")
          .groupby(["year", "scenario", to_col])["value"].sum()
          .reset_index()
          .rename(columns={to_col: "category"})
    )


def filter_category(df, category_name=None, keep_name=None):
    """Keep only rows matching *category_name*, optionally renaming them."""
    if category_name is not None:
        if isinstance(category_name, list):
            df = df[df["category"].isin(category_name)]
        else:
            df = df[df["category"] == category_name]
    if keep_name is not None:
        df = df.copy()
        df["category"] = keep_name
    return df


# ── Specialised loaders ───────────────────────────────────────────────────────

def load_long_json(input_files, json_name, unit_scale=1e6):
    """Load a summary JSON from DATA_REPORT into long format."""
    rows = []
    for scenario in input_files:
        base = get_path(scenario)
        if base is None:
            continue
        fp = os.path.join(base, "DATA_REPORT", "data", f"{json_name}.json")
        df = pd.read_json(fp)
        df = df.explode("data")
        df[["year", "value"]] = pd.DataFrame(df["data"].tolist(), index=df.index)
        for _, row in df.iterrows():
            rows.append({"year": int(row["year"]), "scenario": scenario,
                         "category": row["name"],
                         "value": row["value"] / unit_scale})
    return pd.DataFrame(rows, columns=["year", "scenario", "category", "value"])


# ── Legacy helpers (keep for backward compatibility) ─────────────────────────

def get_lon_lat(tif_path):
    with rasterio.open(tif_path) as ds:
        data = ds.read(1)
        transform = ds.transform
    valid = data > 0
    values = data[valid]
    rows, cols = np.where(valid)
    lon, lat = rasterio.transform.xy(transform, rows, cols)
    lon, lat = np.array(lon), np.array(lat)
    return np.sum(lon * values) / np.sum(values), np.sum(lat * values) / np.sum(values)


def compute_land_use_change_metrics(input_file, use_parallel=True):
    path = get_path(input_file)
    pattern = re.compile(r'(?<!Non-)Ag_LU.*\.tif{1,2}$', re.IGNORECASE)
    years = list(range(2010, 2051, 5))
    folder_2050 = os.path.join(path, "out_2050", "lucc_separate")
    sample_files = [f for f in os.listdir(folder_2050) if pattern.match(f)]
    names = [f.split("_")[3] for f in sample_files]

    coord_cols = pd.MultiIndex.from_product([names, ['Centroid Lon', 'Centroid Lat']])
    coord_df = pd.DataFrame(index=years, columns=coord_cols)

    for year in years:
        folder = os.path.join(path, f"out_{year}", "lucc_separate")
        year_files = [f for f in os.listdir(folder) if pattern.match(f) and "mercator" not in f]
        fps = [os.path.join(folder, f) for f in year_files]
        centroids = (Parallel(n_jobs=30)(delayed(get_lon_lat)(fp) for fp in fps)
                     if use_parallel else [get_lon_lat(fp) for fp in fps])
        for fname, (clon, clat) in zip(year_files, centroids):
            lu = fname.split("_")[3]
            coord_df.loc[year, (lu, 'Centroid Lon')] = clon
            coord_df.loc[year, (lu, 'Centroid Lat')] = clat

    with rasterio.open(os.path.join(folder_2050, sample_files[0])) as ds:
        crs_from = ds.crs
    transformer = Transformer.from_crs(crs_from, CRS.from_epsg(3577), always_xy=True)

    metric_cols = pd.MultiIndex.from_product([names, ['Distance (km)', 'Angle (degrees)', 'Area', 'Area×Distance']])
    result_df = pd.DataFrame(index=years, columns=metric_cols)

    for year in years:
        area_file = os.path.join(path, f"out_{year}", f"area_agricultural_landuse_{year}.csv")
        area_dict = (pd.read_csv(area_file).groupby("Land-use")["Area (ha)"].sum().to_dict()
                     if os.path.exists(area_file) else {})
        for lu in names:
            ox, oy = (transformer.transform(*coord_df.loc[2010, (lu, c)].values[[0, 1]])
                      if not pd.isna(coord_df.loc[2010, (lu, 'Centroid Lon')]) else (np.nan, np.nan))
            cx, cy = (transformer.transform(*coord_df.loc[year, (lu, c)].values[[0, 1]])
                      if not pd.isna(coord_df.loc[year, (lu, 'Centroid Lon')]) else (np.nan, np.nan))
            if not any(np.isnan([ox, oy, cx, cy])):
                dist = np.sqrt((cx - ox)**2 + (cy - oy)**2) / 1000
                angle = (np.degrees(np.arctan2(cx - ox, cy - oy)) + 360) % 360
            else:
                dist = angle = np.nan
            area_ha = area_dict.get(lu, np.nan)
            area_conv = area_ha / 1e8 if not np.isnan(area_ha) else np.nan
            result_df.loc[year, (lu, 'Distance (km)')] = round(dist, 3) if not np.isnan(dist) else np.nan
            result_df.loc[year, (lu, 'Angle (degrees)')] = round(angle, 2) if not np.isnan(angle) else np.nan
            result_df.loc[year, (lu, 'Area')] = area_conv
            result_df.loc[year, (lu, 'Area×Distance')] = (round(area_conv * dist, 3)
                                                            if not np.isnan(area_conv) and not np.isnan(dist) else np.nan)

    flat = result_df.stack(level=[0, 1], future_stack=True).reset_index()
    flat.columns = ['Year', 'Land Use', 'Metric', 'Value']
    new_df = flat.pivot_table(index='Land Use', columns=['Year', 'Metric'], values='Value')
    excel_path = os.path.join("..", "output", "12_land_use_movement_all.xlsx")
    mode = 'a' if os.path.exists(excel_path) else 'w'
    kw = {"if_sheet_exists": "replace"} if mode == "a" else {}
    with pd.ExcelWriter(excel_path, engine='openpyxl', mode=mode, **kw) as writer:
        new_df.to_excel(writer, sheet_name=input_file)
