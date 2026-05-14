"""AgS1 2023 spike: area changes and GHG context."""
import pandas as pd, os

BASE = r'F:/Users/s222552331/Work/LUTO2_XH/luto-2.0/output/20260513_Paper3_aquila/Run_1_SCN_AgS1/output/2026_05_13__19_24_46_RF5_2010-2050'

# ── 1. Area changes 2021→2023 ────────────────────────────────────────────────
print('=== 1. Area changes (Mha) 2021 → 2022 → 2023 ===')
areas = {}
for yr in [2021, 2022, 2023, 2024]:
    fa = os.path.join(BASE, f'out_{yr}', f'area_agricultural_landuse_{yr}.csv')
    if not os.path.exists(fa): continue
    da = pd.read_csv(fa)
    area_col = [c for c in da.columns if 'Area' in c][0]
    aus = da[(da['region']=='AUSTRALIA') & (da['Water_supply']=='ALL') & (da['Land-use']!='ALL')]
    areas[yr] = aus.groupby('Land-use')[area_col].sum()

key_lu = ['Sheep - modified land', 'Sheep - natural land', 'Beef - modified land',
          'Beef - natural land', 'Winter cereals', 'Unallocated - modified land',
          'Unallocated - natural land', 'Dairy - modified land']
print(f'  {"Land use":<40} {"2021":>8} {"2022":>8} {"Δ22":>7} {"2023":>8} {"Δ23":>7} {"2024":>8}')
for lu in key_lu:
    row = []
    for yr in [2021, 2022, 2023, 2024]:
        a = areas.get(yr, {}).get(lu, 0)/1e6
        row.append(a)
    d22 = row[1]-row[0]; d23 = row[2]-row[1]
    print(f'  {lu:<40} {row[0]:>8.2f} {row[1]:>8.2f} {d22:>+7.2f} {row[2]:>8.2f} {d23:>+7.2f} {row[3]:>8.2f}')

# ── 2. GHG totals and limit 2021-2024 ────────────────────────────────────────
print('\n=== 2. GHG totals and limit ===')
for yr in [2021, 2022, 2023, 2024]:
    fg = os.path.join(BASE, f'out_{yr}', f'GHG_emissions_separate_agricultural_landuse_{yr}.csv')
    if not os.path.exists(fg): continue
    dg = pd.read_csv(fg)
    aus = dg[(dg['region']=='AUSTRALIA') & (dg['Source']=='ALL') & (dg['Water_supply']=='ALL') & (dg['Land-use']=='ALL')]
    ghg = aus['Value (t CO2e)'].sum()
    print(f'  {yr}: ag GHG = {ghg/1e6:.2f} Mt CO2e')

# ── 3. Sheep demand target vs actual production ──────────────────────────────
print('\n=== 3. Sheep production vs target ===')
for yr in [2021, 2022, 2023, 2024]:
    fq = os.path.join(BASE, f'out_{yr}', f'quantity_comparison_{yr}.csv')
    if not os.path.exists(fq): continue
    dq = pd.read_csv(fq)
    cols = dq.columns.tolist()
    print(f'  {yr} columns: {cols[:6]}')
    sheep = dq[dq['Commodity'].str.contains('Sheep', na=False)] if 'Commodity' in cols else dq.head(5)
    print(sheep.to_string(index=False))
    break

# ── 4. Check Area_cost scenario for AgS1 ─────────────────────────────────────
print('\n=== 4. AgS1 area cost and productivity settings ===')
sf = os.path.join(BASE, 'model_run_settings.txt')
with open(sf) as f:
    for line in f:
        if any(k in line for k in ['AC_MAP', 'PRODUCTIVITY', 'FLC', 'AREA_COST', 'AG2050']):
            print(' ', line.strip()[:130])
