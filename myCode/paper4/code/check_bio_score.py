import pandas as pd
import numpy as np

path = r'F:\Users\s222552331\Work\LUTO2_XH\luto-2.0\output\20260427_paper4_NCI\paper4\data'

# Fig 4: Bio slice - Ag land-use biodiversity contribution & area
df4 = pd.read_excel(f'{path}/4_Contribution_raw_data_2050.xlsx', sheet_name='BioPrice')
df3 = pd.read_excel(f'{path}/3_Area_raw_data_2050.xlsx', sheet_name='BioPrice')

print('=== Fig4 BioPrice Ag land-use biodiversity contribution (Mha yr) ===')
df4_ag = df4[df4['AreaType'] == 'Agricultural land-use'][['Price','Category','ContributionValue']]
pivot4_ag = df4_ag.pivot_table(index='Category', columns='Price', values='ContributionValue')
print(pivot4_ag.to_string())
print()

print('=== Fig3 BioPrice Ag land-use area change (Mha) ===')
df3_ag = df3[df3['AreaType'] == 'Agricultural land-use'][['Price','Category','AreaChangevs_Baseline_Mha']]
pivot3_ag = df3_ag.pivot_table(index='Category', columns='Price', values='AreaChangevs_Baseline_Mha')
print(pivot3_ag.to_string())
print()

# Compute per-ha biodiversity score: bio_contribution / area_change for non-zero cases
# Use the highest price point to get a meaningful signal
prices = [3850, 11000, 22000, 81950]
print('=== Per-ha biodiversity score (Contribution / Area) at each price ===')
print('(Mha yr / Mha = yr, represents relative bio score per ha)')
for cat in pivot4_ag.index:
    row = {}
    for p in prices:
        if p in pivot4_ag.columns and p in pivot3_ag.columns:
            bio = pivot4_ag.loc[cat, p] if cat in pivot4_ag.index else np.nan
            area = pivot3_ag.loc[cat, p] if cat in pivot3_ag.index else np.nan
            if not np.isnan(bio) and not np.isnan(area) and abs(area) > 0.001:
                row[p] = bio / area
    if row:
        print(f"  {cat}: {row}")

print()
# Same for Non-ag
print('=== Fig4 BioPrice Non-ag biodiversity contribution (Mha yr) ===')
df4_nonag = df4[df4['AreaType'] == 'Non-ag'][['Price','Category','ContributionValue']]
pivot4_nonag = df4_nonag.pivot_table(index='Category', columns='Price', values='ContributionValue')
print(pivot4_nonag.to_string())
print()

print('=== Per-ha score for Non-ag ===')
df3_nonag = df3[df3['AreaType'] == 'Non-ag'][['Price','Category','AreaChangevs_Baseline_Mha']]
pivot3_nonag = df3_nonag.pivot_table(index='Category', columns='Price', values='AreaChangevs_Baseline_Mha')
for cat in pivot4_nonag.index:
    row = {}
    for p in prices:
        if p in pivot4_nonag.columns and p in pivot3_nonag.columns:
            bio = pivot4_nonag.loc[cat, p] if cat in pivot4_nonag.index else np.nan
            area = pivot3_nonag.loc[cat, p] if cat in pivot3_nonag.index else np.nan
            if not np.isnan(bio) and not np.isnan(area) and abs(area) > 0.001:
                row[p] = bio / area
    if row:
        print(f"  {cat}: {row}")
