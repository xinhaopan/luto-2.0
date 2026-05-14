"""Demand satisfaction and full transition picture AgS2 2023."""
import pandas as pd, os

S2 = r'F:/Users/s222552331/Work/LUTO2_XH/luto-2.0/output/20260513_Paper3_aquila/Run_2_SCN_AgS2/output/2026_05_13__19_24_45_RF5_2010-2050'
S1 = r'F:/Users/s222552331/Work/LUTO2_XH/luto-2.0/output/20260513_Paper3_aquila/Run_1_SCN_AgS1/output/2026_05_13__19_24_46_RF5_2010-2050'

# Peek at quantity_comparison columns
f = os.path.join(S2, 'out_2021', 'quantity_comparison_2021.csv')
df0 = pd.read_csv(f)
print('quantity_comparison columns:', df0.columns.tolist())
print(df0.head(5).to_string())

# ── 1. Demand satisfaction AgS2 ─────────────────────────────────────────────
print('\n=== 1. Demand satisfaction AgS2 ===')
sup_col = [c for c in df0.columns if 'supply' in c.lower() or 'Supply' in c][0] if any('supply' in c.lower() for c in df0.columns) else None
dem_col = [c for c in df0.columns if 'demand' in c.lower() or 'Demand' in c][0] if any('demand' in c.lower() for c in df0.columns) else None
prop_col = [c for c in df0.columns if 'prop' in c.lower() or 'Prop' in c][0] if any('prop' in c.lower() for c in df0.columns) else None
print(f'  Using: sup={sup_col}, dem={dem_col}, prop={prop_col}')

for yr in [2021, 2022, 2023, 2024]:
    f = os.path.join(S2, f'out_{yr}', f'quantity_comparison_{yr}.csv')
    if not os.path.exists(f): continue
    dq = pd.read_csv(f)
    aus = dq[dq['region'] == 'AUSTRALIA'] if 'region' in dq.columns else dq
    com_col = [c for c in dq.columns if 'Commodity' in c or 'commodity' in c][0]
    print(f'  {yr}:')
    for com in ['Sheep meat', 'Beef meat', 'Winter cereals']:
        row = aus[aus[com_col] == com]
        if len(row):
            parts = []
            if prop_col: parts.append(f'{prop_col}={row[prop_col].values[0]:.2f}%')
            if dem_col: parts.append(f'dem={row[dem_col].values[0]/1e3:.0f}kt')
            if sup_col: parts.append(f'sup={row[sup_col].values[0]/1e3:.0f}kt')
            print(f'    {com:22s}: ' + '  '.join(parts))

# ── 2. AgMgt area AgS2 2021-2024 ─────────────────────────────────────────────
print('\n=== 2. AgMgt area AgS2 2021-2024 ===')
for yr in [2021, 2022, 2023, 2024]:
    f = os.path.join(S2, f'out_{yr}', f'area_agricultural_management_{yr}.csv')
    if not os.path.exists(f): continue
    da = pd.read_csv(f)
    aus = da[(da['region'] == 'AUSTRALIA') & (da['Water_supply'] == 'ALL') & (da['Land-use'] != 'ALL')]
    area_col = [c for c in da.columns if 'Area' in c][0]
    print(f'\n  {yr}:')
    totals = aus.groupby('Type')[area_col].sum().sort_values(ascending=False)
    for t, v in totals.items():
        if v > 1e5:
            print(f'    {t:<40}: {v/1e6:.3f} Mha')

# ── 3. Summary: total transition cost per year AgS2 ─────────────────────────
print('\n=== 3. Total transition cost AgS2 (ag2ag + ag2nonag, B AUD) ===')
for yr in range(2018, 2028):
    ag2ag_f = os.path.join(S2, f'out_{yr}', f'transition_cost_ag2ag_{yr}.csv')
    na_f    = os.path.join(S2, f'out_{yr}', f'transition_cost_ag2non_ag_{yr}.csv')
    if not os.path.exists(ag2ag_f): continue
    df1 = pd.read_csv(ag2ag_f)
    aus1 = df1[df1['region'] == 'AUSTRALIA']
    mask1 = (aus1['From-land-use']=='ALL') & (aus1['To-land-use']=='ALL') & (aus1['Type']=='ALL')
    ag2ag = aus1[mask1]['Cost ($)'].values[0]

    ag2na = 0
    if os.path.exists(na_f):
        df2 = pd.read_csv(na_f)
        aus2 = df2[df2['region'] == 'AUSTRALIA']
        mask2 = (aus2['From-land-use']=='ALL') & (aus2['To-land-use']=='ALL') & (aus2['Cost-type']=='ALL')
        ag2na = aus2[mask2]['Cost ($)'].values[0]

    total = (ag2ag + ag2na) / 1e9
    mark = ' <---' if total > 15 else ''
    print(f'  {yr}: ag2ag={ag2ag/1e9:.2f}B  ag2nonag={ag2na/1e9:.2f}B  total={total:.2f}B{mark}')

# ── 4. Sheep demand year-by-year to show why it spikes in 2023 ──────────────
print('\n=== 4. Sheep meat demand (kt) AgS1 vs AgS2 ===')
for yr in range(2020, 2030):
    row_out = [f'  {yr}:']
    for base, label in [(S1,'S1'), (S2,'S2')]:
        f = os.path.join(base, f'out_{yr}', f'quantity_comparison_{yr}.csv')
        if not os.path.exists(f): row_out.append(f'{label}=N/A'); continue
        dq = pd.read_csv(f)
        aus = dq[dq['region'] == 'AUSTRALIA'] if 'region' in dq.columns else dq
        com_col2 = [c for c in dq.columns if 'Commodity' in c][0]
        dem_col2 = [c for c in dq.columns if 'demand' in c.lower() or 'Demand' in c][0]
        r = aus[aus[com_col2] == 'Sheep meat']
        if len(r):
            row_out.append(f'{label}={r[dem_col2].values[0]/1e3:.0f}kt')
    print(' '.join(row_out))
