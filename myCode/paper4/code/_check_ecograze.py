import pandas as pd
f = r'f:\Users\s222552331\Work\LUTO2_XH\luto-2.0\input\20231107_ECOGRAZE_Bundle.xlsx'
xl = pd.ExcelFile(f)
print('Sheets:', xl.sheet_names)
for sheet in xl.sheet_names[1:]:
    df = pd.read_excel(f, sheet_name=sheet, index_col='Year')
    print(f'\n--- {sheet} ---')
    print('Columns:', df.columns.tolist())
    row = df.loc[2025] if 2025 in df.index else df.iloc[0]
    print(f'2025 values:\n{row}')
