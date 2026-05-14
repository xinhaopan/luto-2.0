"""Deep-dive: why is 2023 ag2ag cost so high in the new runs (and same for both smooth/flat FLC)."""
import pandas as pd, os

BASE = r'F:/Users/s222552331/Work/LUTO2_XH/luto-2.0/output/20260513_Paper3_test/Run_1_SCN_AgS1/output/2026_05_13__22_14_05_RF5_2010-2050'
BASE_OLD = r'F:/Users/s222552331/Work/LUTO2_XH/luto-2.0/output/20260512_Paper3_test/Run_1_SCN_AgS1/output/2026_05_12__21_13_13_RF15_2010-2050'

# ── 1. Check SIM_YEARS and key settings ───────────────────────────────────────
print('=== Key settings from model_run_settings.txt ===')
for label, base in [('NEW test_flat', BASE), ('OLD test RF15', BASE_OLD)]:
    sf = os.path.join(base, 'model_run_settings.txt')
    if not os.path.exists(sf):
        print(f'  {label}: settings NOT FOUND')
        continue
    print(f'\n  [{label}]')
    with open(sf) as f:
        for line in f:
            k = line.split(':')[0]
            if k in ('SIM_YEARS', 'RESFACTOR', 'GHG_EMISSIONS_LIMITS', 'GHG_CONSTRAINT_TYPE',
                     'DEMAND_CONSTRAINT_TYPE', 'AG2050_USE_FLC_SCENARIO', 'AG2050_SCENARIO'):
                print(f'    {line.strip()}')

# ── 2. Area by LU for years around the spike ─────────────────────────────────
print('\n=== Area (Mha) AUSTRALIA ALL water by key LU, new run ===')
for yr in [2021, 2022, 2023, 2024]:
    f = os.path.join(BASE, f'out_{yr}', f'area_{yr}.csv')
    if not os.path.exists(f):
        print(f'  {yr}: area file NOT found')
        continue
    da = pd.read_csv(f)
    cols = da.columns.tolist()
    area_col = [c for c in cols if 'area' in c.lower() or 'ha' in c.lower()][0]

    if 'Water_supply' in da.columns:
        aus = da.query("region == 'AUSTRALIA' and Water_supply == 'ALL'")
    else:
        aus = da.query("region == 'AUSTRALIA'") if 'region' in da.columns else da

    lu_col = [c for c in cols if 'land' in c.lower() or 'lu' in c.lower() or 'use' in c.lower()][0]

    for lu in ['Beef - modified land', 'Sheep - modified land', 'Winter cereals',
               'Unallocated - modified land', 'Dairy - modified land']:
        row = aus[aus[lu_col] == lu]
        v = row[area_col].sum() if len(row) else 0
        print(f'  {yr} {lu:35s}: {v/1e6:.3f} Mha')
    print()

# ── 3. Top transitions in 2022 vs 2023 for new run ───────────────────────────
print('=== 2022 vs 2023: ALL type transition costs (AUSTRALIA ALL→ALL), new run ===')
for yr in [2020, 2021, 2022, 2023, 2024]:
    f = os.path.join(BASE, f'out_{yr}', f'transition_cost_ag2ag_{yr}.csv')
    if not os.path.exists(f): continue
    df = pd.read_csv(f)
    aus_all = df.query("region == 'AUSTRALIA' and `From-land-use` == 'ALL' and `To-land-use` == 'ALL'")
    total = aus_all['Cost ($)'].sum()
    print(f'  {yr}: {total/1e9:.2f} B AUD')
    for _, row in aus_all.iterrows():
        print(f'      {row["Type"]:35s}: {row["Cost ($)"]/1e9:.2f} B')

# ── 4. What's the transition matrix (total areas switching TO/FROM sheep) ────
print('\n=== Switch TO Sheep (AUSTRALIA) by source, new run ===')
for yr in [2022, 2023]:
    f = os.path.join(BASE, f'out_{yr}', f'transition_cost_ag2ag_{yr}.csv')
    if not os.path.exists(f): continue
    df = pd.read_csv(f)
    sheep = df.query("region == 'AUSTRALIA' and `To-land-use` == 'Sheep - modified land' and `From-land-use` != 'ALL' and Type == 'Establishment cost'")
    sheep = sheep.sort_values('Cost ($)', ascending=False)
    print(f'\n  {yr} → Sheep (Establishment, top 8):')
    print(sheep.head(8)[['From-land-use', 'Cost ($)']].to_string(index=False))
    print(f'    TOTAL to Sheep: {sheep["Cost ($)"].sum()/1e9:.2f} B')

# ── 5. Economics transition cost for 2022 vs 2023 ────────────────────────────
print('\n=== Economics ag transition (AUSTRALIA, ALL) - 2022 vs 2023, new run ===')
for yr in [2022, 2023]:
    f = os.path.join(BASE, f'out_{yr}', f'economics_ag_transition_ag2ag_{yr}.csv')
    if not os.path.exists(f):
        print(f'  {yr}: econ transition file NOT found')
        continue
    de = pd.read_csv(f)
    val_col = [c for c in de.columns if 'value' in c.lower() or 'cost' in c.lower() or '$' in c][0]
    aus_all = de.query("region == 'AUSTRALIA' and `From_Land-use` == 'ALL' and `To_Land-use` == 'ALL'")
    print(f'\n  {yr} economics transition (AUSTRALIA ALL→ALL):')
    print(aus_all[['Type', val_col]].to_string(index=False))

# ── 6. GHG tightness and demand: check if demand for sheep products spikes ───
print('\n=== Check demand multiplier / GHG limits, new run ===')
sf = os.path.join(BASE, 'model_run_settings.txt')
if os.path.exists(sf):
    with open(sf) as f:
        for line in f:
            if any(k in line for k in ('GHG_LIMIT', 'SIM_YEARS', 'DEMAND', 'DIET', 'WASTE', 'FEED_EFF')):
                print(' ', line.strip())
