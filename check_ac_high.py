"""Compare Area_cost high vs medium scenarios."""
import pandas as pd
xl = pd.ExcelFile('input/Area_cost.xlsx')
print('Sheets:', xl.sheet_names)
for sht in ['high', 'medium']:
    ac = xl.parse(sht)
    yr_col = [c for c in ac.columns if 'year' in str(c).lower() or c == 'Year']
    if not yr_col:
        print(f'{sht}: no Year column'); continue
    sub = ac[(ac[yr_col[0]] >= 2019) & (ac[yr_col[0]] <= 2026)]
    print(f'\nSheet: {sht}')
    print(sub[[yr_col[0]] + [c for c in ac.columns if c not in yr_col][:6]].to_string(index=False))
