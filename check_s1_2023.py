"""Why does AgS1 make a large batch switch to sheep in 2023 specifically?"""
import pandas as pd, os

BASE = r'F:/Users/s222552331/Work/LUTO2_XH/luto-2.0/output/20260513_Paper3_aquila/Run_1_SCN_AgS1/output/2026_05_13__19_24_46_RF5_2010-2050'

# ── 1. Profit/ha for key land uses 2020-2024 ─────────────────────────────────
print('=== 1. Profit/ha ($/ha) for key land uses ===')
print(f'  {"Land-use":<40}', end='')
for yr in [2020, 2021, 2022, 2023, 2024]: print(f'  {yr}', end='')
print()
profit_by_lu = {}
for yr in [2020, 2021, 2022, 2023, 2024]:
    fp = os.path.join(BASE, f'out_{yr}', f'economics_ag_profit_{yr}.csv')
    fa = os.path.join(BASE, f'out_{yr}', f'area_agricultural_landuse_{yr}.csv')
    if not os.path.exists(fp): continue
    de = pd.read_csv(fp); da = pd.read_csv(fa)
    val_col = [c for c in de.columns if 'Value' in c or '$' in c][0]
    area_col = [c for c in da.columns if 'Area' in c][0]
    aus_p = de[(de['region']=='AUSTRALIA') & (de['Water_supply']=='ALL')]
    aus_a = da[(da['region']=='AUSTRALIA') & (da['Water_supply']=='ALL')]
    for lu in aus_p['Land-use'].unique():
        if lu == 'ALL': continue
        ep = aus_p[aus_p['Land-use']==lu]; ea = aus_a[aus_a['Land-use']==lu]
        profit = ep[val_col].sum(); area = ea[area_col].sum()
        if area > 1e5:
            profit_by_lu.setdefault(lu, {})[yr] = profit / area

key_lu = ['Sheep - modified land', 'Sheep - natural land', 'Beef - modified land',
          'Beef - natural land', 'Winter cereals', 'Unallocated - modified land',
          'Unallocated - natural land', 'Dairy - modified land', 'Winter oilseeds']
for lu in key_lu:
    d = profit_by_lu.get(lu, {})
    row = '  ' + f'{lu:<40}'
    for yr in [2020, 2021, 2022, 2023, 2024]:
        row += f'  {d.get(yr, 0):>6.0f}'
    print(row)

# ── 2. Demand vs production for sheep year by year ───────────────────────────
print('\n=== 2. Sheep demand vs production ===')
for yr in [2019, 2020, 2021, 2022, 2023, 2024]:
    fq = os.path.join(BASE, f'out_{yr}', f'quantity_comparison_{yr}.csv')
    if not os.path.exists(fq): continue
    dq = pd.read_csv(fq)
    sheep = dq[dq['Commodity'].str.contains('Sheep', na=False)]
    for _, row in sheep.iterrows():
        demand = row.get('Demand (tonnes, KL)', 0)
        prod = row.get('Prod_targ_year (tonnes, KL)', 0)
        pct = row.get('Prop_diff (%)', 0)
        print(f'  {yr} {row["Commodity"]:<15}: demand={demand/1e3:.0f}kt  prod={prod/1e3:.0f}kt  {pct:.1f}%')

# ── 3. GHG: how close to limit? ──────────────────────────────────────────────
print('\n=== 3. GHG total (ag+non_ag) vs limit ===')
for yr in [2020, 2021, 2022, 2023, 2024]:
    # Try net GHG summary file
    fn = os.path.join(BASE, f'out_{yr}', f'GHG_emissions_separate_agricultural_landuse_{yr}.csv')
    if not os.path.exists(fn): continue
    dg = pd.read_csv(fn)
    ag = dg[(dg['region']=='AUSTRALIA') & (dg['Source']=='ALL') & (dg['Water_supply']=='ALL') & (dg['Land-use']=='ALL')]['Value (t CO2e)'].sum()
    # Non-ag GHG
    fna = os.path.join(BASE, f'out_{yr}', f'GHG_emissions_separate_agricultural_management_{yr}.csv')
    am_ghg = 0
    if os.path.exists(fna):
        dm = pd.read_csv(fna)
        if 'Value (t CO2e)' in dm.columns:
            filt = (dm['region']=='AUSTRALIA') & (dm['Land-use']=='ALL')
            if 'Source' in dm.columns: filt &= (dm['Source']=='ALL')
            if 'Water_supply' in dm.columns: filt &= (dm['Water_supply']=='ALL')
            am_ghg = dm[filt]['Value (t CO2e)'].sum()
    print(f'  {yr}: ag={ag/1e6:.2f}Mt  am={am_ghg/1e6:.2f}Mt  net_ag={( ag+am_ghg)/1e6:.2f}Mt')

# ── 4. Profit without transition cost (operating profit only) ────────────────
print('\n=== 4. Revenue/ha and cost/ha (excl. transition) for sheep vs cereals ===')
for yr in [2021, 2022, 2023]:
    print(f'\n  {yr}:')
    for lu in ['Sheep - modified land', 'Winter cereals']:
        fr = os.path.join(BASE, f'out_{yr}', f'economics_ag_revenue_{yr}.csv')
        fc = os.path.join(BASE, f'out_{yr}', f'economics_ag_cost_{yr}.csv')
        fa = os.path.join(BASE, f'out_{yr}', f'area_agricultural_landuse_{yr}.csv')
        dr = pd.read_csv(fr); dc = pd.read_csv(fc); da = pd.read_csv(fa)
        rv_col = [c for c in dr.columns if 'Value' in c or '$' in c][0]
        cost_col = [c for c in dc.columns if 'Value' in c or '$' in c][0]
        area_col = [c for c in da.columns if 'Area' in c][0]
        aus_r = dr[(dr['region']=='AUSTRALIA') & (dr['Water_supply']=='ALL') & (dr['Land-use']==lu)]
        aus_c = dc[(dc['region']=='AUSTRALIA') & (dc['Water_supply']=='ALL') & (dc['Land-use']==lu)]
        aus_a = da[(da['region']=='AUSTRALIA') & (da['Water_supply']=='ALL') & (da['Land-use']==lu)]
        # Only ALL type rows
        rev = aus_r[aus_r['Type']=='ALL'][rv_col].sum() if 'Type' in dr.columns else aus_r[rv_col].sum()
        cost = aus_c[aus_c['Type']=='ALL'][cost_col].sum() if 'Type' in dc.columns else aus_c[cost_col].sum()
        area = aus_a[area_col].sum()
        if area > 0:
            print(f'    {lu:<40}: rev/ha={rev/area:>7.1f}  cost/ha={cost/area:>7.1f}  oper_profit/ha={( rev-cost)/area:>7.1f}')
