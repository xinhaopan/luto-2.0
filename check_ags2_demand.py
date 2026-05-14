"""Demand satisfaction and full transition picture AgS2 2023."""
import pandas as pd, os

S2 = r'F:/Users/s222552331/Work/LUTO2_XH/luto-2.0/output/20260513_Paper3_aquila/Run_2_SCN_AgS2/output/2026_05_13__19_24_45_RF5_2010-2050'
S1 = r'F:/Users/s222552331/Work/LUTO2_XH/luto-2.0/output/20260513_Paper3_aquila/Run_1_SCN_AgS1/output/2026_05_13__19_24_46_RF5_2010-2050'

# ── 1. All transition cost types 2023 AgS2 (using boolean index, no query) ───
print('=== 1. All transition cost types 2023 AgS2 (B AUD) ===')
for fname in ['transition_cost_ag2ag', 'transition_cost_ag2non_ag', 'transition_cost_non_ag2ag']:
    f = os.path.join(S2, 'out_2023', f'{fname}_2023.csv')
    if not os.path.exists(f):
        print(f'  {fname}: NOT FOUND'); continue
    df = pd.read_csv(f)
    aus = df[df['region'] == 'AUSTRALIA'] if 'region' in df.columns else df
    cost_col = 'Cost ($)'
    total_mask = (aus['From-land-use'] == 'ALL') & (aus['To-land-use'] == 'ALL')
    if 'Cost-type' in aus.columns:
        total_mask &= (aus['Cost-type'] == 'ALL')
    elif 'Type' in aus.columns:
        total_mask &= (aus['Type'] == 'ALL')
    total = aus[total_mask][cost_col].values[0]
    print(f'  {fname}: {total/1e9:.2f} B')

# ── 2. Demand satisfaction AgS2 2021-2024 ───────────────────────────────────
print('\n=== 2. Demand satisfaction AgS2 ===')
for yr in [2021, 2022, 2023, 2024]:
    f = os.path.join(S2, f'out_{yr}', f'quantity_comparison_{yr}.csv')
    if not os.path.exists(f): continue
    dq = pd.read_csv(f)
    aus = dq[dq['region'] == 'AUSTRALIA'] if 'region' in dq.columns else dq
    print(f'  {yr}:')
    for com in ['Sheep meat', 'Beef meat', 'Winter cereals']:
        row = aus[aus['Commodity'] == com]
        if len(row):
            prop = row['Prop_diff (%)'].values[0]
            dem = row['Demand (tonnes, KL)'].values[0]
            sup = row['Supply (tonnes, KL)'].values[0]
            print(f'    {com:22s}: {prop:.2f}%  demand={dem/1e3:.0f}kt  supply={sup/1e3:.0f}kt')

# ── 3. Demand comparison AgS1 vs AgS2 2023 ──────────────────────────────────
print('\n=== 3. Sheep+beef demand AgS1 vs AgS2 ===')
for yr in [2021, 2022, 2023, 2024]:
    row_out = [f'  {yr}:']
    for base, label in [(S1,'AgS1'), (S2,'AgS2')]:
        f = os.path.join(base, f'out_{yr}', f'quantity_comparison_{yr}.csv')
        if not os.path.exists(f): row_out.append(f'{label}=N/A'); continue
        dq = pd.read_csv(f)
        aus = dq[dq['region'] == 'AUSTRALIA'] if 'region' in dq.columns else dq
        for com in ['Sheep meat', 'Beef meat']:
            row = aus[aus['Commodity'] == com]
            if len(row):
                dem = row['Demand (tonnes, KL)'].values[0]
                row_out.append(f'{label} {com[:5]}dem={dem/1e3:.0f}kt')
    print(' '.join(row_out))

# ── 4. AgMgt area comparison: HIR-Sheep, Eco Grazing ─────────────────────────
print('\n=== 4. AgMgt area AgS2 2021-2024 ===')
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

# ── 5. Summary: total transition cost (ag2ag + ag2nonag) per year AgS2 ────────
print('\n=== 5. Total transition cost AgS2 (ag2ag + ag2nonag, B AUD) ===')
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
