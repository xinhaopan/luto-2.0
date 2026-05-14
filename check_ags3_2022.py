"""Investigate why AgS3 (no GHG target) has high ag2ag cost in 2022 (old aquila run)."""
import pandas as pd, os

# OLD aquila run, AgS3
BASE_OLD = r'F:/Users/s222552331/Work/LUTO2_XH/luto-2.0/output/20260512_Paper3_aquila/Run_3_SCN_AgS3/output'
subdirs = [d for d in os.listdir(BASE_OLD) if os.path.isdir(os.path.join(BASE_OLD, d))]
BASE = os.path.join(BASE_OLD, subdirs[0])
print(f'Using: {subdirs[0]}')

# ── 1. Transition cost ALL years ────────────────────────────────────────────
print('\n=== 1. Ag2Ag transition cost ALL (B AUD), AUSTRALIA ===')
for yr in range(2015, 2030):
    f = os.path.join(BASE, f'out_{yr}', f'transition_cost_ag2ag_{yr}.csv')
    if not os.path.exists(f): continue
    df = pd.read_csv(f)
    aus_all = df.query("region == 'AUSTRALIA' and `From-land-use` == 'ALL' and `To-land-use` == 'ALL' and Type == 'ALL'")
    total = aus_all['Cost ($)'].sum()
    mark = ' <---' if total > 5e9 else ''
    print(f'  {yr}: {total/1e9:.2f} B{mark}')

# ── 2. Top transitions in 2022 for AgS3 ────────────────────────────────────
print('\n=== 2. AgS3 2022 top transitions (Establishment cost) ===')
for yr in [2021, 2022, 2023]:
    f = os.path.join(BASE, f'out_{yr}', f'transition_cost_ag2ag_{yr}.csv')
    if not os.path.exists(f): continue
    df = pd.read_csv(f)
    sub = df.query("region == 'AUSTRALIA' and `From-land-use` != 'ALL' and `To-land-use` != 'ALL' and Type == 'Establishment cost'")
    sub = sub.nlargest(6, 'Cost ($)')
    print(f'\n  {yr} top transitions:')
    print(sub[['From-land-use', 'To-land-use', 'Cost ($)']].to_string(index=False))

# ── 3. Area change 2021→2022→2023 for AgS3 ─────────────────────────────────
print('\n=== 3. Area (Mha) AUSTRALIA key LU, AgS3 ===')
for yr in [2019, 2020, 2021, 2022, 2023, 2024]:
    f = os.path.join(BASE, f'out_{yr}', f'area_agricultural_landuse_{yr}.csv')
    if not os.path.exists(f): continue
    da = pd.read_csv(f)
    aus = da.query("region == 'AUSTRALIA' and Water_supply == 'ALL'")
    area_col = [c for c in da.columns if 'Area' in c][0]
    vals = []
    for lu in ['Beef - modified land', 'Sheep - modified land', 'Winter cereals',
               'Unallocated - modified land', 'Dairy - modified land']:
        row = aus[aus['Land-use'] == lu]
        v = row[area_col].sum() if len(row) else 0
        vals.append(f'{lu[:20]:20s}:{v/1e6:.2f}')
    print(f'  {yr}:  ' + '  '.join(vals))

# ── 4. Profit/ha for key LU in AgS3 ────────────────────────────────────────
print('\n=== 4. Profit/ha ($/ha) for key LU, AgS3 ===')
for yr in [2020, 2021, 2022, 2023]:
    fp = os.path.join(BASE, f'out_{yr}', f'economics_ag_profit_{yr}.csv')
    fa = os.path.join(BASE, f'out_{yr}', f'area_agricultural_landuse_{yr}.csv')
    if not os.path.exists(fp): continue
    de = pd.read_csv(fp)
    da = pd.read_csv(fa)
    val_col = [c for c in de.columns if 'Value' in c or '$' in c][0]
    area_col = [c for c in da.columns if 'Area' in c][0]
    print(f'\n  {yr}:')
    for lu in ['Beef - modified land', 'Sheep - modified land', 'Winter cereals',
               'Unallocated - modified land', 'Dairy - modified land']:
        ep = de.query("region == 'AUSTRALIA' and Water_supply == 'ALL' and `Land-use` == @lu")
        ea = da.query("region == 'AUSTRALIA' and Water_supply == 'ALL' and `Land-use` == @lu")
        profit = ep[val_col].sum() if len(ep) else 0
        area = ea[area_col].sum() if len(ea) else 1
        print(f'    {lu:35s}: {profit/area:>6.0f} $/ha  ({profit/1e9:.2f} B total)')

# ── 5. Check FLC cost for AgS3 settings ─────────────────────────────────────
print('\n=== 5. AgS3 settings (GHG, demand, FLC) ===')
sf = os.path.join(BASE, 'model_run_settings.txt')
with open(sf) as fh:
    for line in fh:
        k = line.split(':')[0]
        if k in ('GHG_EMISSIONS_LIMITS', 'DEMAND_CONSTRAINT_TYPE', 'AG2050_FLC_MAP',
                 'AG2050_SCENARIO', 'AG2050_USE_FLC_SCENARIO', 'RESFACTOR',
                 'BIODIVERSITY_TARGET_GBF_2'):
            print(' ', line.strip()[:120])

# ── 6. Quantity demand met? (2021-2023) ──────────────────────────────────────
print('\n=== 6. Demand satisfaction AgS3 (sheep/beef) ===')
for yr in [2021, 2022, 2023]:
    f = os.path.join(BASE, f'out_{yr}', f'quantity_comparison_{yr}.csv')
    if not os.path.exists(f): continue
    dq = pd.read_csv(f)
    aus = dq[dq['region'] == 'AUSTRALIA'] if 'region' in dq.columns else dq
    print(f'  {yr}:')
    for com in ['Sheep meat', 'Beef meat', 'Winter cereals']:
        row = aus[aus['Commodity'] == com]
        if len(row):
            prop = row['Prop_diff (%)'].values[0]
            dem = row['Demand (tonnes, KL)'].values[0]
            print(f'    {com:20s}: {prop:.1f}%  (demand={dem/1e3:.0f} kt)')
