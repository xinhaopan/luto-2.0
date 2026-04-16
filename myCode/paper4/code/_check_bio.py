import io, zipfile
import cf_xarray as cfxr
import xarray as xr

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

root = 'F:/Users/s222552331/Work/LUTO2_XH/luto-2.0/output/20260414_paper4_NCI'
runs = [
    ('Run_01_CarbonPrice_0_BioPrice_0',    0,    0),
    ('Run_02_CarbonPrice_0_BioPrice_500',  0,  500),
    ('Run_03_CarbonPrice_0_BioPrice_1000', 0, 1000),
    ('Run_05_CarbonPrice_0_BioPrice_2000', 0, 2000),
    ('Run_11_CarbonPrice_0_BioPrice_5000', 0, 5000),
]
base = None
for run, cp, bp in runs:
    zp = f'{root}/{run}/Run_Archive.zip'
    print(f'\n=== cp={cp}, bp={bp} ===')
    total = read_sum(zp, BIO_FILES, YEAR)
    if base is None:
        base = total
    print(f'  TOTAL={total:.2f}, dBio={total - base:.2f}')
