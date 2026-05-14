import pandas as pd, os

RUNS = {
    'AgS1': r'F:/Users/s222552331/Work/LUTO2_XH/luto-2.0/output/20260513_Paper3_aquila/Run_1_SCN_AgS1/output/2026_05_13__19_24_46_RF5_2010-2050',
    'AgS2': r'F:/Users/s222552331/Work/LUTO2_XH/luto-2.0/output/20260513_Paper3_aquila/Run_2_SCN_AgS2/output/2026_05_13__19_24_45_RF5_2010-2050',
    'AgS3': r'F:/Users/s222552331/Work/LUTO2_XH/luto-2.0/output/20260513_Paper3_aquila/Run_3_SCN_AgS3/output/2026_05_13__19_24_45_RF5_2010-2050',
    'AgS4': r'F:/Users/s222552331/Work/LUTO2_XH/luto-2.0/output/20260513_Paper3_aquila/Run_4_SCN_AgS4/output/2026_05_13__19_24_46_RF5_2010-2050',
}

print('=== 1. ALL demands in AgS4 2022 vs 2023 ===')
BASE4 = RUNS['AgS4']
for yr in [2022, 2023]:
    fq = os.path.join(BASE4, f'out_{yr}', f'quantity_comparison_{yr}.csv')
    dq = pd.read_csv(fq)
    print(f'\n  {yr}:')
    for _, row in dq.iterrows():
        dem = row.get('Demand (tonnes, KL)', 0)
        prod = row.get('Prod_targ_year (tonnes, KL)', 0)
        pct = row.get('Prop_diff (%)', 0)
        mark = ' BINDING' if abs(pct - 100.0) < 0.01 else f'  [{pct:.1f}%]'
        print(f'    {row["Commodity"]:<25}: dem={dem/1e3:>8.1f}kt  prod={prod/1e3:>8.1f}kt{mark}')

print('\n=== 2. Area_cost low and very_high ===')
xl = pd.ExcelFile('input/Area_cost.xlsx')
for sht in ['low', 'very_high']:
    ac = xl.parse(sht)
    yr_col = [c for c in ac.columns if 'year' in str(c).lower() or c == 'Year']
    if not yr_col: continue
    sub = ac[(ac[yr_col[0]] >= 2019) & (ac[yr_col[0]] <= 2026)]
    print(f'\nSheet: {sht}')
    print(sub[[yr_col[0]] + [c for c in ac.columns if c not in yr_col][:6]].to_string(index=False))

print('\n=== 3. Top transitions 2023 by scenario ===')
for scn, BASE in RUNS.items():
    ft = os.path.join(BASE, 'out_2023', 'transition_cost_ag2ag_2023.csv')
    if not os.path.exists(ft): continue
    d = pd.read_csv(ft)
    non_all = d[(d['region']=='AUSTRALIA') & (d['From-land-use']!='ALL') & (d['To-land-use']!='ALL') & (d['Type']=='ALL')]
    total = non_all['Cost ($)'].sum()
    sheep = non_all[non_all['To-land-use'].str.contains('Sheep',na=False) | non_all['From-land-use'].str.contains('Sheep',na=False)]['Cost ($)'].sum()
    print(f'\n  {scn} total={total/1e9:.2f}B  sheep={sheep/1e9:.2f}B ({sheep/total*100:.0f}%)')
    grp = non_all.groupby(['From-land-use','To-land-use'])['Cost ($)'].sum().sort_values(ascending=False)
    for (frm, to), val in grp.head(8).items():
        if abs(val) > 1.5e8:
            tag = '[S]' if ('Sheep' in frm or 'Sheep' in to) else '   '
            print(f'    {tag} {frm:<38} -> {to:<35}: {val/1e9:.3f}B')

print('\n=== 4. AgS4 operating profit/ha 2021-2024 ===')
for yr in [2021, 2022, 2023, 2024]:
    fp = os.path.join(RUNS['AgS4'], f'out_{yr}', f'economics_ag_revenue_{yr}.csv')
    fc = os.path.join(RUNS['AgS4'], f'out_{yr}', f'economics_ag_cost_{yr}.csv')
    fa = os.path.join(RUNS['AgS4'], f'out_{yr}', f'area_agricultural_landuse_{yr}.csv')
    if not os.path.exists(fp): continue
    dr=pd.read_csv(fp); dc=pd.read_csv(fc); da=pd.read_csv(fa)
    rv_col=[c for c in dr.columns if 'Value' in c or '$' in c][0]
    cost_col=[c for c in dc.columns if 'Value' in c or '$' in c][0]
    area_col=[c for c in da.columns if 'Area' in c][0]
    print(f'\n  {yr}:')
    for lu in ['Sheep - modified land','Beef - modified land','Winter cereals','Unallocated - modified land']:
        r = dr[(dr['region']=='AUSTRALIA')&(dr['Water_supply']=='ALL')&(dr['Land-use']==lu)&(dr.get('Type',pd.Series(['ALL']*len(dr)))=='ALL')][rv_col].sum()
        c = dc[(dc['region']=='AUSTRALIA')&(dc['Water_supply']=='ALL')&(dc['Land-use']==lu)&(dc.get('Type',pd.Series(['ALL']*len(dc)))=='ALL')][cost_col].sum()
        a = da[(da['region']=='AUSTRALIA')&(da['Water_supply']=='ALL')&(da['Land-use']==lu)][area_col].sum()
        if 'Type' not in dr.columns:
            r = dr[(dr['region']=='AUSTRALIA')&(dr['Water_supply']=='ALL')&(dr['Land-use']==lu)][rv_col].sum()
            c = dc[(dc['region']=='AUSTRALIA')&(dc['Water_supply']=='ALL')&(dc['Land-use']==lu)][cost_col].sum()
        if a>1e5: print(f'    {lu:<40}: oper_profit/ha={(r-c)/a:>7.0f}')
