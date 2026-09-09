"""
make_cell_area_cache.py -- one-off: cache the per-cell area composition.

Why
    The Same/Different figure must not be built from the plotting raster.  That
    raster carries one integer per cell -- the dominant class, already rounded --
    and says nothing about how much of the cell that class actually occupies, so
    every cell would count the same regardless of size or composition.

    Each run archive already stores what is needed, per cell and in hectares:
        xr_area_real_area_ha_2050.nc          the cell's real area
        xr_area_agricultural_landuse_2050.nc  area by (water supply, land use)
        xr_area_non_agricultural_landuse_2050.nc  area by non-ag land use
        xr_area_agricultural_management_2050.nc   area by management option
    Reading them needs xarray and cf_xarray (the 'layer' dimension is a
    compressed multi-index), which the xpluto environment provides.  It runs once and
    leaves a compact .npz the figure script can open with numpy alone.

What it writes
    EXCEL_DIR/33_cell_areas.npz
        real_area_ha   (n_cells,)              LUTO's own per-cell area
        cat8_ha        (n_scen, 8, n_cells)    area in each of the 8 Fig. 2 categories
        am_ha          (n_scen, n_am, n_cells) area under each management option
        scenarios, categories, am_names        label arrays

Run it from the 4_Draw/code directory with the xpluto environment, which is
what the Ag2050 figures use throughout:
    <xpluto>/python.exe tools/make_cell_area_cache.py
"""

import io
import os
import sys
import zipfile

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
os.chdir(os.path.dirname(_HERE))

from tools.data_helper import get_zip_info                      # noqa: E402
from tools.parameters import EXCEL_DIR, input_files             # noqa: E402
from tools.two_row_figure import (                              # noqa: E402
    CROPLAND_LUS, MODIFIED_PASTURE_LUS, NATIVE_PASTURE_LUS, UNALLOCATED_LUS,
)

CACHE_NAME = '33_cell_areas.npz'
YEAR = 2050

# The eight categories of Fig. 2, in the order 02_Mapping.py numbers them.
CATEGORIES = [
    'Dryland cropland and horticulture',
    'Irrigated cropland and horticulture',
    'Dryland grazing (modified pastures)',
    'Irrigated grazing (modified pastures)',
    'Grazing (native vegetation)',
    'Unallocated land',
    'Non-agricultural land-use',
    'Other (public, indigenous, urban, plantation, water)',
]


def _layer_frame(z, prefix, stem):
    """Return (DataArray, MultiIndex) for one xr_* archive member."""
    import xarray as xr
    import cf_xarray as cfxr
    name = f'{prefix}/out_{YEAR}/xr_{stem}_{YEAR}.nc'
    with z.open(name) as f:
        ds = xr.open_dataset(io.BytesIO(f.read()))
    arr = cfxr.decode_compress_to_multi_index(ds, 'layer')['data']
    return arr, arr['layer'].to_index()


def _ag_area(arr, index, water_supply, land_uses):
    """Sum the (water supply, land use) layers that fall in one category."""
    wanted = [i for i, (lm, lu) in enumerate(index)
              if lm == water_supply and lu in land_uses]
    if not wanted:
        return np.zeros(arr.shape[0], dtype='float32')
    return np.nan_to_num(arr.values[:, wanted]).sum(axis=1).astype('float32')


def main() -> None:
    scenarios = list(input_files)
    cat8, am_stack, real_area = [], [], None
    am_names = None

    for scenario in scenarios:
        info = get_zip_info(scenario)
        if info is None:
            raise FileNotFoundError(f'No Run_Archive.zip for {scenario}')
        zip_path, prefix = info
        print(f'  {scenario}')
        with zipfile.ZipFile(zip_path) as z:
            if real_area is None:
                arr, _ = _layer_frame(z, prefix, 'area_real_area_ha')
                real_area = np.nan_to_num(arr.values[:, 0]).astype('float32')

            ag, ag_idx = _layer_frame(z, prefix, 'area_agricultural_landuse')
            nonag, nonag_idx = _layer_frame(z, prefix, 'area_non_agricultural_landuse')

            cats = np.zeros((8, ag.shape[0]), dtype='float32')
            cats[0] = _ag_area(ag, ag_idx, 'dry', CROPLAND_LUS)
            cats[1] = _ag_area(ag, ag_idx, 'irr', CROPLAND_LUS)
            cats[2] = _ag_area(ag, ag_idx, 'dry', MODIFIED_PASTURE_LUS)
            cats[3] = _ag_area(ag, ag_idx, 'irr', MODIFIED_PASTURE_LUS)
            # Native pasture and unallocated land are not split by water supply
            # in Fig. 2, so take the ALL layer.
            cats[4] = _ag_area(ag, ag_idx, 'ALL', NATIVE_PASTURE_LUS)
            cats[5] = _ag_area(ag, ag_idx, 'ALL', UNALLOCATED_LUS)
            # NOT the 'ALL' layer: in these files 'ALL' is not the sum over land
            # uses.  For AgS2 it reads 3.3 Mha while the eight individual
            # non-agricultural types add to 13.9 Mha, and using it silently lost
            # 10.6 Mha of restored land into the residual category.  Sum the
            # types.  (In the agricultural file, by contrast, lm='ALL' *is*
            # dry+irr for a given land use, which is verified and used above.)
            nonag_types = [i for i, (lu,) in enumerate(nonag_idx) if lu != 'ALL']
            # Cells with no non-agricultural land carry NaN, not zero.
            cats[6] = np.nan_to_num(nonag.values[:, nonag_types]).sum(axis=1).astype('float32')
            # Whatever the model does not allocate is public/indigenous/urban/
            # plantation/water -- category 8 in Fig. 2.
            cats[7] = np.clip(real_area - cats[:7].sum(axis=0), 0.0, None)
            cat8.append(cats)

            am, am_idx = _layer_frame(z, prefix, 'area_agricultural_management')
            # Scenarios do not carry the same set of options -- Climate Survival
            # and System Decline only run Ecological Grazing, Savanna Burning
            # and the two HIR options -- so collect per scenario and align on
            # the union afterwards, with zeros where an option is switched off.
            per_am = {}
            # 'ALL' is the across-option total, not an option.
            for option in sorted({a for a, lm, lu in am_idx} - {'ALL'}):
                cols = [i for i, (a, lm, lu) in enumerate(am_idx)
                        if a == option and lm == 'ALL' and lu == 'ALL']
                if cols:
                    per_am[option] = np.nan_to_num(
                        am.values[:, cols]).sum(axis=1).astype('float32')
            am_stack.append(per_am)

    cat8 = np.stack(cat8)
    am_names = sorted({option for per_am in am_stack for option in per_am})
    n_cells = real_area.size
    am_array = np.zeros((len(scenarios), len(am_names), n_cells), dtype='float32')
    for i, per_am in enumerate(am_stack):
        for k, option in enumerate(am_names):
            if option in per_am:
                am_array[i, k] = per_am[option]
    am_stack = am_array
    os.makedirs(EXCEL_DIR, exist_ok=True)
    out = os.path.join(EXCEL_DIR, CACHE_NAME)
    np.savez_compressed(
        out,
        real_area_ha=real_area,
        cat8_ha=cat8,
        am_ha=am_stack,
        scenarios=np.array(scenarios),
        categories=np.array(CATEGORIES),
        am_names=np.array(am_names),
    )
    print(f'  cells        : {real_area.size:,}')
    print(f'  real area    : {real_area.sum() / 1e6:.3f} Mha (all cells)')
    print(f'  allocated    : {(cat8[0, :7].sum()) / 1e6:.3f} Mha '
          '(categories 1-7; matches land_area_2010_mha in the trade-off workbook)')
    for i, scenario in enumerate(scenarios):
        allocated = cat8[i].sum() / 1e6
        print(f'  {scenario:<18} allocated {allocated:8.3f} Mha '
              f'(residual cat 8 = {cat8[i, 7].sum() / 1e6:7.3f} Mha)')
    print(f'  management   : {am_names}')
    print(f'  wrote {out}')


if __name__ == '__main__':
    main()
