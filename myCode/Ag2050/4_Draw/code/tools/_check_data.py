import sys, zipfile
import xarray as xr
sys.path.insert(0, r'F:\Users\s222552331\Work\LUTO2_XH\luto-2.0\myCode\Ag2050\4_Draw\code')
import _path_setup
from tools.parameters import input_files
from tools.data_helper import get_zip_info, _list_years_zip

scenario = input_files[0]
info = get_zip_info(scenario)
zip_path, prefix = info

# List ALL nc files in year 2010
with zipfile.ZipFile(zip_path) as z:
    all_files = z.namelist()
    nc_2010 = sorted([f for f in all_files if '2010' in f and f.endswith('.nc')])
    print('NC files in 2010:')
    for f in nc_2010:
        fname = f.split('/')[-1]
        print(' ', fname)

print()
# Open water yield ag nc and print structure
with zipfile.ZipFile(zip_path) as z:
    with z.open(f'{prefix}/out_2010/xr_water_yield_ag_2010.nc') as f:
        import io
        data = f.read()
    ds = xr.open_dataset(io.BytesIO(data))
    print('=== xr_water_yield_ag_2010.nc ===')
    print(ds)

print()
# Check if there's a dvar or area NC file
dvar_files = [f for f in nc_2010 if 'dvar' in f.lower() or 'area' in f.lower()]
print('dvar/area NC files:', dvar_files)
