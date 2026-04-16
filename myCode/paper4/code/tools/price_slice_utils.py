import io
import re
import zipfile
from pathlib import Path

import cf_xarray as cfxr
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import xarray as xr

import tools.config as config


CODE_DIR = Path(__file__).resolve().parent.parent
TASK_ROOT = (CODE_DIR / ".." / ".." / ".." / "output" / config.TASK_NAME).resolve()
OUT_DIR = TASK_ROOT / "paper4" / "figures"
DATA_DIR = TASK_ROOT / "paper4" / "data"

OUT_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)


def get_paper4_paths():
    return TASK_ROOT, OUT_DIR, DATA_DIR


def parse_prices(run_name):
    cp = re.search(r"CarbonPrice_([\d.]+)", run_name)
    bp = re.search(r"BioPrice_([\d.]+)", run_name)
    return (
        float(cp.group(1)) if cp else None,
        float(bp.group(1)) if bp else None,
    )


def build_run_map(task_root=None):
    task_root = Path(task_root) if task_root is not None else TASK_ROOT
    run_map = {}

    for run_dir in task_root.iterdir():
        if not run_dir.is_dir() or not run_dir.name.startswith("Run_"):
            continue

        cp, bp = parse_prices(run_dir.name)
        if cp is None or bp is None:
            continue

        zip_path = run_dir / "Run_Archive.zip"
        if zip_path.is_file():
            run_map[(cp, bp)] = zip_path

    cp_vals = sorted({key[0] for key in run_map})
    bp_vals = sorted({key[1] for key in run_map})
    return run_map, cp_vals, bp_vals


def get_slice_key(varying_key, price_value):
    if varying_key == "cp":
        return price_value, 0.0
    if varying_key == "bp":
        return 0.0, price_value
    raise ValueError(f"Unsupported varying_key: {varying_key}")


def get_price_column(varying_key):
    if varying_key == "cp":
        return "CarbonPrice"
    if varying_key == "bp":
        return "BioPrice"
    raise ValueError(f"Unsupported varying_key: {varying_key}")


def get_price_axis_label(varying_key):
    if varying_key == "cp":
        return r"Carbon price (AU\$/tCO$_2$e)"
    if varying_key == "bp":
        return r"Biodiversity price (AU\$/ha)"
    raise ValueError(f"Unsupported varying_key: {varying_key}")


def format_thousands(value):
    if pd.isna(value):
        return ""

    value = float(value)
    if np.isclose(value, round(value)):
        return f"{int(round(value)):,}"
    return f"{value:,.2f}"


PRICE_TICK_FORMATTER = ticker.FuncFormatter(lambda value, _: format_thousands(value))


def apply_price_formatter(ax, axis="x"):
    if axis == "x":
        ax.xaxis.set_major_formatter(PRICE_TICK_FORMATTER)
    elif axis == "y":
        ax.yaxis.set_major_formatter(PRICE_TICK_FORMATTER)
    else:
        raise ValueError(f"Unsupported axis: {axis}")


def style_box_axis(ax, linewidth=1.0):
    ax.set_facecolor("white")
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(linewidth)
        spine.set_color("black")


def _select_all_coords(da):
    for coord_name in list(da.coords):
        if coord_name in {"cell", "layer"}:
            continue

        coord_values = da.coords[coord_name].values
        try:
            has_all = "ALL" in coord_values
        except TypeError:
            has_all = False

        if has_all:
            da = da.sel({coord_name: "ALL"})

    return da


def read_sum(zip_path, file_stems, year):
    total = 0.0
    with zipfile.ZipFile(zip_path) as archive:
        all_names = archive.namelist()
        for stem in file_stems:
            target = f"{stem}_{year}.nc"
            matches = [name for name in all_names if name.endswith(target)]
            if not matches:
                continue

            with archive.open(matches[0]) as file_obj:
                ds = xr.open_dataset(io.BytesIO(file_obj.read()), engine="h5netcdf")

            try:
                if "layer" in ds.dims and "compress" in ds["layer"].attrs:
                    ds = cfxr.decode_compress_to_multi_index(ds, "layer")

                da = next(iter(ds.data_vars.values()))
                da = _select_all_coords(da)
                total += float(da.sum())
            finally:
                ds.close()

    return total
