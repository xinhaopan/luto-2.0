import io, zipfile
import importlib.util
from pathlib import Path

import cf_xarray as cfxr
import xarray as xr


CONFIG_PATH = Path(__file__).resolve().parent / 'tools' / 'config.py'
CONFIG_SPEC = importlib.util.spec_from_file_location('paper4_config', CONFIG_PATH)
config = importlib.util.module_from_spec(CONFIG_SPEC)
CONFIG_SPEC.loader.exec_module(config)

BIO_FILES = [
    'xr_biodiversity_overall_priority_ag',
    'xr_biodiversity_overall_priority_ag_management',
    'xr_biodiversity_overall_priority_non_ag',
]
YEAR = 2050

def read_sum(zip_path, file_stems, year):
    total = 0.0
    with zipfile.ZipFile(zip_path) as z:
        all_names = z.namelist()
        for fname in file_stems:
            target = f'{fname}_{year}.nc'
            matches = [n for n in all_names if n.endswith(target)]
            if not matches:
                print(f'  MISSING: {target}')
                continue
            with z.open(matches[0]) as zf:
                ds = xr.open_dataset(io.BytesIO(zf.read()), engine='h5netcdf')
            if 'layer' in ds.dims and 'compress' in ds['layer'].attrs:
                ds = cfxr.decode_compress_to_multi_index(ds, 'layer')
            da = list(ds.data_vars.values())[0]
            for coord_name in list(da.coords):
                if coord_name in {'cell', 'layer'}:
                    continue
                if 'ALL' in da.coords[coord_name].values:
                    da = da.sel({coord_name: 'ALL'})
            val = float(da.sum())
            short = fname.replace('xr_biodiversity_overall_priority_', '')
            print(f'  {short}: {val:.2f}')
            total += val
    return total

root = str(config.TASK_ROOT)
runs = [
    (
        run_name,
        config.SCENARIO_PRICE_LOOKUP[run_name]['CarbonPrice'],
        config.SCENARIO_PRICE_LOOKUP[run_name]['BioPrice'],
    )
    for run_name in config.biodiversity_price_scenarios
]
base = None
for run, cp, bp in runs:
    zp = f'{root}/{run}/Run_Archive.zip'
    print(f'\n=== cp={cp}, bp={bp} ===')
    total = read_sum(zp, BIO_FILES, YEAR)
    if base is None:
        base = total
    print(f'  TOTAL={total:.2f}, dBio={total - base:.2f}')
