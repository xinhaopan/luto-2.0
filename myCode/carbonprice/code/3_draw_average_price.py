import xarray as xr
import pandas as pd
import numpy as np
import tools.config as config
from tools.helper_plot import set_plot_style, draw_12_price, draw_10_price



task_name = config.TASK_NAME
# task_name = '20250922_Paper2_Results_HPC_test'
input_dir = f'../../../output/{task_name}/carbon_price/1_draw_data'
output_dir = f"../../../output/{task_name}/carbon_price/3_Paper_figure"
carbon_price_da = xr.open_dataarray(f"{input_dir}/xr_carbon_price.nc")
bio_price_da = xr.open_dataarray(f"{input_dir}/xr_bio_price.nc")

df1 = pd.concat([carbon_price_da.to_pandas().T.iloc[:, :2],bio_price_da.to_pandas().T.iloc[:, 2:12]], axis=1, join='inner')
df1[df1 < 0] = 0
df1 = df1[df1.index >= config.START_YEAR].fillna(0)
df2 = carbon_price_da.to_pandas().T.iloc[:, 12:].fillna(0)
df2[df2 < 0] = 0
df2 = df2[df2.index >= config.START_YEAR]

arr1, arr2 = df1.to_numpy(), df2.to_numpy()

# 结果 = df2的10列 - df1的后10列
# 前5列：df2前5列 - df1第一列
# 后5列：df2后5列 - df1第二列
res = np.empty_like(arr2, dtype=float)
res[:, :5] = arr2[:, :5] - arr1[:, [0]]
res[:, 5:] = arr2[:, 5:] - arr1[:, [1]]

# 转回 DataFrame，列名用 df1 的后10列
df3 = pd.DataFrame(res, index=df1.index, columns=df1.columns[-10:]).fillna(0)
df3[df3 < 0] = 0
color = 'green'

set_plot_style(20)

draw_12_price(df1,config.PRICE_TITLE_MAP,color,f"{output_dir}/05_Carbon_Bio_price.png",desired_ticks=5,ci=None)
draw_10_price(df2,config.PRICE_TITLE_MAP,color,f"{output_dir}/05_Carbon_price_for_bio&carbon.png",desired_ticks=5,ylabel="Shadow carbon price under net-zero targets and nature-positive targets\n(AU\$ tCO$_2$e$^{-1}$ yr$^{-1}$)",ci=None)
draw_10_price(df3,config.PRICE_TITLE_MAP,color,f"{output_dir}/05_Carbon_price_for_bio.png",desired_ticks=5,ylabel="Shadow carbon price under nature-positive targets\n(AU\$ tCO$_2$e$^{-1}$ yr$^{-1}$)",ci=None)
