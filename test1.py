import xarray as xr

dc1 = xr.open_dataset(r"F:\xinhao\xr_transition_cost_ag2non_ag_amortised_2015.nc")
dc2 = xr.open_dataset(r"F:\xinhao\xr_transition_cost_ag2non_ag_amortised_2015_1.nc")

print(dc1.equals(dc2))