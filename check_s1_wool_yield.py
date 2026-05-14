"""Why does ag2ag cost drop after 2023 spike in AgS1?"""
import pandas as pd, os

BASE = r'F:/Users/s222552331/Work/LUTO2_XH/luto-2.0/output/20260513_Paper3_aquila/Run_1_SCN_AgS1/output/2026_05_13__19_24_46_RF5_2010-2050'

# ── 1. Wool/lexp yield per ha, and sheep area, across all years ──────────────
print('=== 1. Sheep area & wool/lexp yield per total sheep ha ===')
print(f'  {"Year":>4}  {"mod_ha":>8}  {"nat_ha":>8}  {"total_ha":>9}  {"wool_kt":>7}  {"wool/ha":>8}  {"lexp_kt":>7}  {"lexp/ha":>8}  {"ag2ag_B":>7}')
ag2ag_cost = {
    2019:7.47,2020:18.43,2021:17.64,2022:10.25,2023:33.59,2024:6.44,
    2025:5.88,2026:5.95,2027:6.51,2028:7.30,2029:7.65,2030:8.01
}
for yr in range(2019, 2031):
    fa = os.path.join(BASE, f'out_{yr}', f'area_agricultural_landuse_{yr}.csv')
    fq = os.path.join(BASE, f'out_{yr}', f'quantity_comparison_{yr}.csv')
    if not os.path.exists(fa) or not os.path.exists(fq): continue
    da = pd.read_csv(fa); dq = pd.read_csv(fq)
    area_col = [c for c in da.columns if 'Area' in c][0]
    aus = da[(da['region']=='AUSTRALIA') & (da['Water_supply']=='ALL') & (da['Land-use']!='ALL')]
    mod_ha = aus[aus['Land-use']=='Sheep - modified land'][area_col].sum()
    nat_ha = aus[aus['Land-use']=='Sheep - natural land'][area_col].sum()
    total_ha = mod_ha + nat_ha
    wool = dq[dq['Commodity']=='Sheep wool']['Prod_targ_year (tonnes, KL)'].values
    lexp = dq[dq['Commodity']=='Sheep lexp']['Prod_targ_year (tonnes, KL)'].values
    wool_kt = wool[0]/1e3 if len(wool) else 0
    lexp_kt = lexp[0]/1e3 if len(lexp) else 0
    cost = ag2ag_cost.get(yr, 0)
    print(f'  {yr:>4}  {mod_ha/1e6:>8.2f}  {nat_ha/1e6:>8.2f}  {total_ha/1e6:>9.2f}  {wool_kt:>7.1f}  {wool_kt/total_ha*1e6:>8.2f}  {lexp_kt:>7.1f}  {lexp_kt/total_ha*1e6:>8.4f}  {cost:>7.2f}')

# ── 2. How much sheep area is churned (switched in AND out) in each year ─────
print('\n=== 2. Sheep area switched IN vs OUT (from ag2ag transition file) ===')
for yr in range(2020, 2031):
    fp = os.path.join(BASE, f'out_{yr}', f'transition_cost_ag2ag_{yr}.csv')
    if not os.path.exists(fp): continue
    d = pd.read_csv(fp)
    aus = d[d['region']=='AUSTRALIA']
    non_all = aus[(aus['From-land-use']!='ALL') & (aus['To-land-use']!='ALL') & (aus['Type']=='ALL')]
    to_sheep = non_all[non_all['To-land-use'].str.contains('Sheep', na=False)]['Cost ($)'].sum()
    from_sheep = non_all[non_all['From-land-use'].str.contains('Sheep', na=False)]['Cost ($)'].sum()
    total = non_all['Cost ($)'].sum()
    print(f'  {yr}: →sheep={to_sheep/1e9:.2f}B  sheep→={from_sheep/1e9:.2f}B  total={total/1e9:.2f}B')

# ── 3. Wool demand schedule ───────────────────────────────────────────────────
print('\n=== 3. Wool & lexp demand targets ===')
for yr in range(2018, 2030):
    fq = os.path.join(BASE, f'out_{yr}', f'quantity_comparison_{yr}.csv')
    if not os.path.exists(fq): continue
    dq = pd.read_csv(fq)
    for comm in ['Sheep wool', 'Sheep lexp']:
        row = dq[dq['Commodity']==comm]
        if len(row):
            dem = row['Demand (tonnes, KL)'].values[0]
            print(f'  {yr} {comm:<14}: demand={dem/1e3:.1f}kt')
    break  # just show columns once then all data
for yr in range(2018, 2030):
    fq = os.path.join(BASE, f'out_{yr}', f'quantity_comparison_{yr}.csv')
    if not os.path.exists(fq): continue
    dq = pd.read_csv(fq)
    wrow = dq[dq['Commodity']=='Sheep wool']
    lrow = dq[dq['Commodity']=='Sheep lexp']
    wd = wrow['Demand (tonnes, KL)'].values[0]/1e3 if len(wrow) else 0
    ld = lrow['Demand (tonnes, KL)'].values[0]/1e3 if len(lrow) else 0
    print(f'  {yr}: wool_demand={wd:.1f}kt  lexp_demand={ld:.1f}kt')
