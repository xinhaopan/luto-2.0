"""Look at raw file structure for sheep economics."""
import pandas as pd, os

BASE_OLD = r'F:/Users/s222552331/Work/LUTO2_XH/luto-2.0/output/20260512_Paper3_aquila/Run_3_SCN_AgS3/output/2026_05_13__00_29_59_RF5_2010-2050'
yr = 2022

# Profit file
fp = os.path.join(BASE_OLD, f'out_{yr}', f'economics_ag_profit_{yr}.csv')
dp = pd.read_csv(fp)
print('=== Profit file columns:', dp.columns.tolist())
aus_sheep = dp[(dp['region'] == 'AUSTRALIA') & (dp['Land-use'] == 'Sheep - modified land')]
print(f'Rows for AUSTRALIA + Sheep-modified: {len(aus_sheep)}')
print(aus_sheep.to_string())

# Revenue file
fr = os.path.join(BASE_OLD, f'out_{yr}', f'economics_ag_revenue_{yr}.csv')
dr = pd.read_csv(fr)
print('\n=== Revenue file columns:', dr.columns.tolist())
aus_rev = dr[(dr['region'] == 'AUSTRALIA') & (dr['Land-use'] == 'Sheep - modified land')]
print(f'Rows for AUSTRALIA + Sheep-modified: {len(aus_rev)}')
print(aus_rev.to_string())

# Cost file
fc = os.path.join(BASE_OLD, f'out_{yr}', f'economics_ag_cost_{yr}.csv')
dc = pd.read_csv(fc)
print('\n=== Cost file columns:', dc.columns.tolist())
aus_cost = dc[(dc['region'] == 'AUSTRALIA') & (dc['Land-use'] == 'Sheep - modified land') & (dc['Water_supply'] == 'ALL')]
print(f'Rows for AUSTRALIA + Sheep-modified + Water=ALL: {len(aus_cost)}')
print(aus_cost.to_string())
