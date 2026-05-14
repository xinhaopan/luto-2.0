"""Compare ag2ag costs vs sheep wool production across all 4 scenarios."""
import pandas as pd, os

RUNS = {
    'AgS1': r'F:/Users/s222552331/Work/LUTO2_XH/luto-2.0/output/20260513_Paper3_aquila/Run_1_SCN_AgS1/output/2026_05_13__19_24_46_RF5_2010-2050',
    'AgS2': r'F:/Users/s222552331/Work/LUTO2_XH/luto-2.0/output/20260513_Paper3_aquila/Run_2_SCN_AgS2/output/2026_05_13__19_24_45_RF5_2010-2050',
    'AgS3': r'F:/Users/s222552331/Work/LUTO2_XH/luto-2.0/output/20260513_Paper3_aquila/Run_3_SCN_AgS3/output/2026_05_13__19_24_45_RF5_2010-2050',
    'AgS4': r'F:/Users/s222552331/Work/LUTO2_XH/luto-2.0/output/20260513_Paper3_aquila/Run_4_SCN_AgS4/output/2026_05_13__19_24_46_RF5_2010-2050',
}

for scn, BASE in RUNS.items():
    print(f'\n{"="*70}')
    print(f'=== {scn} ===')
    print(f'  {"Year":>4}  {"ag2ag_B":>8}  {"sheep_mod":>9}  {"wool_kt":>7}  {"Δwool":>6}  {"wool/ha":>8}  {"Δwool/ha":>9}  {"→sheep_B":>9}')
    prev_wool = None; prev_yield = None
    for yr in range(2011, 2036):
        fp = os.path.join(BASE, f'out_{yr}', f'transition_cost_ag2ag_{yr}.csv')
        fa = os.path.join(BASE, f'out_{yr}', f'area_agricultural_landuse_{yr}.csv')
        fq = os.path.join(BASE, f'out_{yr}', f'quantity_comparison_{yr}.csv')
        if not os.path.exists(fp): continue

        # ag2ag cost
        d = pd.read_csv(fp)
        aus = d[d['region']=='AUSTRALIA']
        non_all = aus[(aus['From-land-use']!='ALL') & (aus['To-land-use']!='ALL') & (aus['Type']=='ALL')]
        ag2ag = non_all['Cost ($)'].sum()
        to_sheep = non_all[non_all['To-land-use'].str.contains('Sheep', na=False)]['Cost ($)'].sum()

        # sheep area
        sheep_mod = 0; sheep_nat = 0
        if os.path.exists(fa):
            da = pd.read_csv(fa)
            area_col = [c for c in da.columns if 'Area' in c][0]
            aus_a = da[(da['region']=='AUSTRALIA') & (da['Water_supply']=='ALL')]
            sheep_mod = aus_a[aus_a['Land-use']=='Sheep - modified land'][area_col].sum()
            sheep_nat = aus_a[aus_a['Land-use']=='Sheep - natural land'][area_col].sum()

        # wool production
        wool_kt = 0
        if os.path.exists(fq):
            dq = pd.read_csv(fq)
            row = dq[dq['Commodity']=='Sheep wool']
            if len(row): wool_kt = row['Prod_targ_year (tonnes, KL)'].values[0]/1e3

        total_sheep = (sheep_mod + sheep_nat) / 1e6
        wool_per_ha = wool_kt / total_sheep if total_sheep > 0 else 0
        d_wool = wool_kt - prev_wool if prev_wool is not None else 0
        d_yield = wool_per_ha - prev_yield if prev_yield is not None else 0

        # mark spike years
        mark = ' ***' if ag2ag > 20e9 else (' **' if ag2ag > 10e9 else '')
        print(f'  {yr:>4}  {ag2ag/1e9:>8.2f}  {sheep_mod/1e6:>9.2f}  {wool_kt:>7.1f}  {d_wool:>+6.1f}  {wool_per_ha:>8.2f}  {d_yield:>+9.3f}  {to_sheep/1e9:>9.2f}{mark}')
        prev_wool = wool_kt; prev_yield = wool_per_ha
