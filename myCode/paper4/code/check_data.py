import pandas as pd

path = r'F:\Users\s222552331\Work\LUTO2_XH\luto-2.0\output\20260427_paper4_NCI\paper4\data'

# Fig 2: bio score across prices
df2 = pd.read_excel(f'{path}/2_NetEcon_raw_data_2050.xlsx', sheet_name='BioPrice')
print('=== Fig2 BioPrice slice ===')
print(df2[['BioPrice','BioScore2050_ha_yr','BioPayment2050_BAUD','NetEconChangevs_Baseline_BAUD']].to_string())
print()

# Fig 3: Non-ag area by category for bio slice
df3 = pd.read_excel(f'{path}/3_Area_raw_data_2050.xlsx', sheet_name='BioPrice')
df3_nonag = df3[df3['AreaType'] == 'Non-ag'][['Price','Category','AreaChangevs_Baseline_Mha']]
print('=== Fig3 BioPrice Non-ag categories ===')
print(df3_nonag.pivot_table(index='Category', columns='Price', values='AreaChangevs_Baseline_Mha').to_string())
print()

# Fig 4: GHG bio slice Non-ag
df4 = pd.read_excel(f'{path}/4_Contribution_raw_data_2050.xlsx', sheet_name='BioPrice')
df4_nonag = df4[df4['AreaType'] == 'Non-ag'][['Price','Category','ContributionValue']]
print('=== Fig4 BioPrice Non-ag GHG/Bio contribution ===')
print(df4_nonag.pivot_table(index='Category', columns='Price', values='ContributionValue').to_string())
