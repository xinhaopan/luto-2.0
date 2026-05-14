"""Find why sheep area suddenly jumps in 2023: check demand trajectory and GHG accounting."""
import pandas as pd, os, sys

BASE = r'F:/Users/s222552331/Work/LUTO2_XH/luto-2.0/output/20260513_Paper3_test/Run_1_SCN_AgS1/output/2026_05_13__22_14_05_RF5_2010-2050'
INPUT = r'F:/Users/s222552331/Work/LUTO2_XH/luto-2.0/input'

# ── 1. Demand trajectory: sheep meat, beef meat 2018-2028 ────────────────────
print('=== 1. Quantity demand target AUSTRALIA, sheep & beef ===')
for yr in [2020, 2021, 2022, 2023, 2024, 2025]:
    f = os.path.join(BASE, f'out_{yr}', f'quantity_comparison_{yr}.csv')
    if not os.path.exists(f): continue
    dq = pd.read_csv(f)
    aus = dq[dq['region'] == 'AUSTRALIA'] if 'region' in dq.columns else dq
    for com in ['Sheep meat', 'Sheep wool', 'Sheep lexp', 'Beef meat', 'Beef lexp']:
        row = aus[aus['Commodity'] == com]
        if len(row):
            prod = row['Prod_targ_year (tonnes, KL)'].values[0]
            dem  = row['Demand (tonnes, KL)'].values[0]
            prop = row['Prop_diff (%)'].values[0]
            print(f'  {yr} {com:15s}: prod={prod/1e3:.0f} kt, demand={dem/1e3:.0f} kt, prop={prop:.2f}%')

# ── 2. Demand change year-on-year ────────────────────────────────────────────
print('\n=== 2. Demand target CHANGE (kt) year-over-year for sheep/beef ===')
prev_demands = {}
for yr in range(2018, 2028):
    f = os.path.join(BASE, f'out_{yr}', f'quantity_comparison_{yr}.csv')
    if not os.path.exists(f): continue
    dq = pd.read_csv(f)
    aus = dq[dq['region'] == 'AUSTRALIA'] if 'region' in dq.columns else dq
    for com in ['Sheep meat', 'Beef meat']:
        row = aus[aus['Commodity'] == com]
        if not len(row): continue
        dem = row['Demand (tonnes, KL)'].values[0]
        if com in prev_demands:
            delta = (dem - prev_demands[com]) / 1e3
            print(f'  {yr-1}→{yr} {com:15s}: demand {prev_demands[com]/1e3:.0f} → {dem/1e3:.0f} kt (Δ={delta:+.1f} kt)')
        prev_demands[com] = dem

# ── 3. GHG emissions per sector 2021-2024 ───────────────────────────────────
print('\n=== 3. GHG breakdown by sector 2021-2024 ===')
for yr in [2020, 2021, 2022, 2023, 2024]:
    f = os.path.join(BASE, f'out_{yr}', f'GHG_emissions_{yr}.csv')
    if not os.path.exists(f): continue
    dg = pd.read_csv(f)
    print(f'  {yr}: {dg.to_string(index=False)}')
    print()

# ── 4. GHG per LU (beef vs sheep) to see emission intensity ─────────────────
print('=== 4. GHG emissions separate AUSTRALIA (Beef vs Sheep), ALL sources ===')
for yr in [2022, 2023]:
    f = os.path.join(BASE, f'out_{yr}', f'GHG_emissions_separate_agricultural_landuse_{yr}.csv')
    if not os.path.exists(f): continue
    dg = pd.read_csv(f)
    aus = dg.query("region == 'AUSTRALIA' and Water_supply == 'ALL' and Source == 'ALL'")
    for lu in ['Beef - modified land', 'Sheep - modified land']:
        row = aus[aus['Land-use'] == lu]
        v = row['Value (t CO2e)'].sum() if 'Value (t CO2e)' in row.columns else row.iloc[:, -1].sum()
        area_f = os.path.join(BASE, f'out_{yr}', f'area_agricultural_landuse_{yr}.csv')
        da = pd.read_csv(area_f)
        area_row = da.query("region == 'AUSTRALIA' and Water_supply == 'ALL' and `Land-use` == @lu")
        area = area_row['Area (ha)'].sum() if len(area_row) else 1
        print(f'  {yr} {lu:35s}: {v/1e9:.2f} Gt, {v/area:.1f} t/ha')

# ── 5. Check if settings have dietary change or demand multiplier in 2023 ────
print('\n=== 5. Demand-related settings from model_run_settings ===')
sf = os.path.join(BASE, 'model_run_settings.txt')
with open(sf) as fh:
    for line in fh:
        k = line.split(':')[0]
        if any(x in k for x in ('DIET', 'DEMAND', 'WASTE', 'FEED_EFF', 'IMPORT', 'CONVERGENCE',
                                  'APPLY_DEMAND', 'AG_YIELD', 'AG2050_SCENARIO')):
            print(' ', line.strip())

# ── 6. Input demand files - check if there's a step change ──────────────────
print('\n=== 6. Raw demand input file for sheep/beef 2020-2026 ===')
sys.path.insert(0, r'F:/Users/s222552331/Work/LUTO2_XH/luto-2.0')
import luto.settings as settings
settings.AG2050_MODE = True
settings.AG2050_SCENARIO = 'AgS1'

demand_dir = INPUT
# Find demand CSV
for fname in os.listdir(demand_dir):
    if 'demand' in fname.lower() and fname.endswith('.csv') and 'separate' not in fname.lower():
        fp = os.path.join(demand_dir, fname)
        try:
            dd = pd.read_csv(fp)
            if 'Year' in dd.columns:
                sub = dd[(dd['Year'] >= 2020) & (dd['Year'] <= 2026)]
                animal_cols = [c for c in dd.columns if any(x in c.lower() for x in
                               ['sheep', 'beef', 'lamb', 'mutton', 'live'])]
                if animal_cols and len(sub) > 0:
                    print(f'  File: {fname}')
                    print(sub[['Year'] + animal_cols[:6]].to_string(index=False))
                    break
        except:
            pass
