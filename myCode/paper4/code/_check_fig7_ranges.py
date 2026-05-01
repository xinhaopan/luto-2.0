import pandas as pd
f = r'F:\Users\s222552331\Work\LUTO2_XH\luto-2.0\output\20260429_paper4_NCI\paper4\data\7_Average_Net_Cost_raw_data_2025.xlsx'
df = pd.read_excel(f, sheet_name='CurveData')
for panel in ['Carbon', 'Biodiversity']:
    sub = df[(df['Panel']==panel) & (df['IncludeInCurve']==True)].copy()
    sub = sub.sort_values('AverageNetCost')
    print(f'=== {panel} ===')
    print(f'  AverageNetCost: min={sub.AverageNetCost.min():.1f}  max={sub.AverageNetCost.max():.1f}')
    print(f'  ContributionWidth: min={sub.ContributionWidth.min():.4f}  max={sub.ContributionWidth.max():.4f}  total={sub.ContributionWidth.sum():.2f}')
    print(sub[['Category','AverageNetCost','ContributionWidth']].to_string())
    print()
