import tools.config as config
from tools.tools import save2nc
import xarray as xr
import os
import numpy as np
from joblib import Parallel, delayed



def create_mask_year(year, base_path, env_category, chunks='auto'):
    data_tmpl = f"{base_path}/{env_category}/{year}/xr_total_{env_category}_{year}.nc"
    cost_tml = f"{base_path}/{env_category}/{year}/xr_total_cost_{env_category}_amortised_{year}.nc"

    data_xr = xr.open_dataarray(data_tmpl, chunks=chunks)
    cost_xr = xr.open_dataarray(cost_tml, chunks=chunks)

    mask = (data_xr >= 1) & (cost_xr >= 1)
    # mask = abs(data_xr) >= 1
    return mask

def caculate_price(env_cat, year, base_dir,type,chunks='auto'):
    print(f"Processing {env_cat} for year {year}...")

    output_path = os.path.join(base_dir, env_cat, str(year), f"xr_{type}_price_{env_cat}_{year}.nc")
    cost_path = os.path.join(base_dir, env_cat, str(year), f"xr_total_cost_{env_cat}_amortised_{year}.nc")
    env_path = os.path.join(base_dir, env_cat, str(year), f"xr_{type}_total_{env_cat}_{year}.nc")

    data_tmpl = f"{base_dir}/{env_cat}/{year}/xr_total_{env_cat}_{year}.nc"
    cost_tml = f"{base_dir}/{env_cat}/{year}/xr_total_cost_{env_cat}_amortised_{year}.nc"

    data_xr = xr.open_dataarray(data_tmpl, chunks=chunks)
    cost_xr = xr.open_dataarray(cost_tml, chunks=chunks)

    mask_da = (data_xr >= 1) & (cost_xr >= 1)

    with xr.open_dataarray(cost_path, chunks=chunks) as cost_da, xr.open_dataarray(env_path, chunks=chunks) as env_da:
        price_da = cost_da / env_da
        price_da = price_da.where(mask_da, np.nan)
        save2nc(price_da, output_path)

base_dir = f"../../../output/{config.TASK_NAME}/carbon_price/0_base_data"
env_categorys = config.carbon_names + config.carbon_bio_names
# for env_cat in env_categorys:
#     for year in range(2011, 2051):
#         print(f"Processing {env_cat} for year {year}...")
#         output_path = os.path.join(base_dir, env_cat, str(year), f"xr_price_{env_cat}_{year}.nc")
#
#         cost_path = os.path.join(base_dir,env_cat,str(year), f"xr_total_cost_{env_cat}_amortised_{year}.nc")
#         env_path = os.path.join(base_dir,env_cat,str(year), f"xr_total_{env_cat}_{year}.nc")
#         mask_da = create_mask_year(year, base_dir, env_cat)
#         with xr.open_dataarray(cost_path) as cost_da, xr.open_dataarray(env_path) as env_da:
#             price_da = cost_da / env_da
#             price_da = price_da.where(mask_da, np.nan)
#             save2nc(price_da, output_path)

for env_cat in env_categorys:
        Parallel(n_jobs=41)(
            delayed(process_year)(env_cat, year, base_dir) for year in range(2011, 2051)
        )



