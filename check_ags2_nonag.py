"""Check ag2non_ag transition costs and demand satisfaction in AgS2 2023."""
import pandas as pd, os

S2 = r'F:/Users/s222552331/Work/LUTO2_XH/luto-2.0/output/20260513_Paper3_aquila/Run_2_SCN_AgS2/output/2026_05_13__19_24_45_RF5_2010-2050'
S1 = r'F:/Users/s222552331/Work/LUTO2_XH/luto-2.0/output/20260513_Paper3_aquila/Run_1_SCN_AgS1/output/2026_05_13__19_24_46_RF5_2010-2050'

# ── 1. ag2non_ag transition cost 2023 ────────────────────────────────────────
print('=== 1. ag→non_ag transition cost 2023 (AgS2) ===')
f = os.path.join(S2, 'out_2023', 'transition_cost_ag2non_ag_2023.csv')
df = pd.read_csv(f)
print('Columns:', df.columns.tolist())
aus = df[df['region'] == 'AUSTRALIA'] if 'region' in df.columns else df
cost_col = 'Cost ($)'
# ALL/ALL/ALL row = total
total_row = aus.query("`From-land-use` == 'ALL' and `To-land-use` == 'ALL' and `Cost-type` == 'ALL'")
print(f'Total: {total_row[cost_col].values[0]/1e9:.2f} B')
# Top non-ag targets
top = aus.query("`From-land-use` == 'ALL' and `To-land-use` != 'ALL' and `Cost-type` == 'ALL'").nlargest(8, cost_col)[['To-land-use', cost_col]]
print(top.to_string())

# ── 2. Total transition cost all types 2023 AgS2 ─────────────────────────────
print('\n=== 2. All transition cost types 2023 AgS2 (B AUD) ===')
for fname in ['transition_cost_ag2ag', 'transition_cost_ag2non_ag', 'transition_cost_non_ag2ag']:
    f = os.path.join(S2, f'out_2023', f'{fname}_2023.csv')
    if not os.path.exists(f):
        print(f'  {fname}: NOT FOUND')
        continue
    df = pd.read_csv(f)
    aus = df[df['region'] == 'AUSTRALIA'] if 'region' in df.columns else df
    total_row = aus.query("`From-land-use` == 'ALL' and `To-land-use` == 'ALL' and `Cost-type` == 'ALL'")
    if len(total_row):
        total = total_row['Cost ($)'].values[0]
    else:
        total = aus.query("`From-land-use` == 'ALL' and `To-land-use` == 'ALL'")['Cost ($)'].values[0]
    print(f'  {fname}: {total/1e9:.2f} B')

# ── 3. Demand satisfaction AgS2 2021-2024 ───────────────────────────────────
print('\n=== 3. Demand satisfaction AgS2 ===')
for yr in [2021, 2022, 2023, 2024]:
    f = os.path.join(S2, f'out_{yr}', f'quantity_comparison_{yr}.csv')
    if not os.path.exists(f): continue
    dq = pd.read_csv(f)
    aus = dq[dq['region'] == 'AUSTRALIA'] if 'region' in dq.columns else dq
    print(f'  {yr}:')
    for com in ['Sheep meat', 'Beef meat', 'Winter cereals', 'Sheep live export']:
        row = aus[aus['Commodity'] == com]
        if len(row):
            prop = row['Prop_diff (%)'].values[0]
            dem = row['Demand (tonnes, KL)'].values[0]
            sup = row['Supply (tonnes, KL)'].values[0]
            print(f'    {com:22s}: {prop:.2f}%  demand={dem/1e3:.0f}kt  supply={sup/1e3:.0f}kt')

# ── 4. Demand comparison AgS1 vs AgS2 2023 ──────────────────────────────────
print('\n=== 4. Demand values AgS1 vs AgS2 2023 ===')
for base, label in [(S1,'AgS1'), (S2,'AgS2')]:
    f = os.path.join(base, 'out_2023', 'quantity_comparison_2023.csv')
    if not os.path.exists(f): continue
    dq = pd.read_csv(f)
    aus = dq[dq['region'] == 'AUSTRALIA'] if 'region' in dq.columns else dq
    print(f'  {label}:')
    for com in ['Sheep meat', 'Beef meat']:
        row = aus[aus['Commodity'] == com]
        if len(row):
            dem = row['Demand (tonnes, KL)'].values[0]
            sup = row['Supply (tonnes, KL)'].values[0]
            prop = row['Prop_diff (%)'].values[0]
            print(f'    {com:22s}: demand={dem/1e3:.0f}kt  supply={sup/1e3:.0f}kt  diff={prop:.2f}%')

# ── 5. GHG limit (actual targets from CSV if available) ──────────────────────
print('\n=== 5. AgS2 GHG limit year-by-year (from quantity_comparison) ===')
for yr in range(2020, 2030):
    f = os.path.join(S2, f'out_{yr}', f'GHG_emissions_separate_agricultural_landuse_{yr}.csv')
    if not os.path.exists(f): continue
    df = pd.read_csv(f)
    ag = df.query("region == 'AUSTRALIA' and Source == 'ALL' and Water_supply == 'ALL' and `Land-use` == 'ALL'")['Value (t CO2e)'].sum()
    print(f'  {yr}: Ag GHG = {ag/1e6:.1f} Mt')
