import pandas as pd

path = r'F:\Users\s222552331\Work\LUTO2_XH\luto-2.0\output\20260427_paper4_NCI\paper4\data'

# Fig 3: Ag land-use area by category for bio slice
df3 = pd.read_excel(f'{path}/3_Area_raw_data_2050.xlsx', sheet_name='BioPrice')

print('=== Fig3 BioPrice Ag land-use categories (area change vs baseline) ===')
df3_ag = df3[df3['AreaType'] == 'Agricultural land-use'][['Price','Category','AreaChangevs_Baseline_Mha']]
pivot_ag = df3_ag.pivot_table(index='Category', columns='Price', values='AreaChangevs_Baseline_Mha')
print(pivot_ag.to_string())
print()

print('=== Fig3 BioPrice Ag management categories ===')
df3_am = df3[df3['AreaType'] == 'Ag management'][['Price','Category','AreaChangevs_Baseline_Mha']]
pivot_am = df3_am.pivot_table(index='Category', columns='Price', values='AreaChangevs_Baseline_Mha')
print(pivot_am.to_string())
print()

# Check if "Unallocated" is in any category
all_cats = df3['Category'].unique()
nat_cats = [c for c in all_cats if 'natural' in str(c).lower() or 'unalloc' in str(c).lower()]
print(f'Natural/Unallocated categories in fig3: {nat_cats}')
