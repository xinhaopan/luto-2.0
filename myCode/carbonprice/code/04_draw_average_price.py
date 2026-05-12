import xarray as xr
import pandas as pd
import numpy as np
import tools.config as config
from tools.helper_plot import set_plot_style, draw_combined_carbon_bio_price

def recalculate_carbon_price(df):
    scenario_list = [
        'carbon_low_bio_10', 'carbon_low_bio_20', 'carbon_low_bio_30',
        'carbon_low_bio_40', 'carbon_low_bio_50',
        'carbon_high_bio_10', 'carbon_high_bio_20', 'carbon_high_bio_30',
        'carbon_high_bio_40', 'carbon_high_bio_50'
    ]

    for sc in scenario_list:
        # 构造相关scenario名字
        # 假定Counterfactual名字规则为："Counterfactual_" + scenario
        # 比如 carbon_low_bio_10 → Counterfactual_carbon_low_bio_10
        counter_sc = f'Counterfactual_{sc}'
        # 另一个名字规则，比如 carbon_low_bio_10 → carbon_low_10
        base_sc = sc.replace('_bio_', '_')

        # 找到所有年份
        for year in df.loc[df['scenario'] == sc, 'Year']:
            # 取同年份的Counterfactual和base值
            v_counter = df.loc[(df['scenario'] == counter_sc) & (df['Year'] == year), 'data']
            v_base = df.loc[(df['scenario'] == base_sc) & (df['Year'] == year), 'data']
            # 如果二者都存在，则更新
            if not v_counter.empty and not v_base.empty:
                diff = float(v_counter.values[0]) - float(v_base.values[0])
                df.loc[(df['scenario'] == sc) & (df['Year'] == year), 'data'] = np.float32(diff)
            else:
                print(f"Warning: missing data for year {year} in {counter_sc} or {base_sc}.")
    return df

task_name = config.TASK_NAME
# task_name = '20250922_Paper2_Results_HPC_test'
input_dir = f'../../../output/{task_name}/{config.CARBON_PRICE_DIR}/1_draw_data'
output_dir = f"../../../output/{task_name}/{config.CARBON_PRICE_DIR}/3_Paper_figure"
carbon_price_da = xr.open_dataarray(f"{input_dir}/xr_carbon_price.nc")
bio_price_da = xr.open_dataarray(f"{input_dir}/xr_bio_price.nc")

df_carbon_long = carbon_price_da.to_dataframe().reset_index()
df_carbon_long = recalculate_carbon_price(df_carbon_long )
df_bio_long = bio_price_da.to_dataframe().reset_index()
import os

os.makedirs(output_dir, exist_ok=True)

set_plot_style(30)
draw_combined_carbon_bio_price(
    df_carbon_long, df_bio_long,
    config.CP_TITLE_MAP, config.BP_TITLE_MAP,
    output_path=f"{output_dir}/05_Carbon_bio_price_combined.png",
    start_year=config.START_YEAR,
    desired_ticks=5, ci=95,
)