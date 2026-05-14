"""Verify HIR-Sheep GHG mechanism: natural land only, check per-ha values."""
import pandas as pd, numpy as np, os

BASE = r'F:/Users/s222552331/Work/LUTO2_XH/luto-2.0/output/20260513_Paper3_aquila/Run_1_SCN_AgS1/output/2026_05_13__19_24_46_RF5_2010-2050'

# ── 1. HIR area: modified vs natural land breakdown ─────────────────────────
print('=== 1. HIR-Sheep area: which land management types? ===')
for yr in [2022, 2023]:
    f = os.path.join(BASE, f'out_{yr}', f'area_agricultural_management_{yr}.csv')
    da = pd.read_csv(f)
    aus = da[(da['region']=='AUSTRALIA') & (da['Land-use'] != 'ALL')]
    hir = aus[aus['Type'].str.contains('HIR', na=False)]
    print(f'\n  {yr}:')
    # Show by land-use (natural vs modified)
    by_lu = hir.groupby(['Type', 'Land-use'])['Area (ha)'].sum().reset_index()
    nat_mod = by_lu.copy()
    nat_mod['land_type'] = nat_mod['Land-use'].apply(lambda x: 'natural' if 'natural' in x.lower() else 'modified')
    for _, row in nat_mod[nat_mod['Area (ha)'] > 1e4].iterrows():
        print(f'    {row["Type"]:<20} {row["Land-use"]:<35} {row["Area (ha)"]/1e6:.3f} Mha  [{row["land_type"]}]')

# ── 2. GHG per ha for sheep natural land: base vs with HIR ──────────────────
print('\n=== 2. GHG per ha: Sheep natural land base vs HIR abatement ===')
for yr in [2022, 2023]:
    ag_f = os.path.join(BASE, f'out_{yr}', f'GHG_emissions_separate_agricultural_landuse_{yr}.csv')
    am_f = os.path.join(BASE, f'out_{yr}', f'GHG_emissions_separate_agricultural_management_{yr}.csv')
    ar_f = os.path.join(BASE, f'out_{yr}', f'area_agricultural_landuse_{yr}.csv')

    dg = pd.read_csv(ag_f)
    dam = pd.read_csv(am_f)
    da = pd.read_csv(ar_f)
    area_col = [c for c in da.columns if 'Area' in c][0]

    # Sheep natural land base GHG (Source=ALL, Water=ALL)
    sheep_nat = dg[(dg['region']=='AUSTRALIA') & (dg['Source']=='ALL') &
                   (dg['Water_supply']=='ALL') & (dg['Land-use']=='Sheep - natural land')]
    sheep_mod = dg[(dg['region']=='AUSTRALIA') & (dg['Source']=='ALL') &
                   (dg['Water_supply']=='ALL') & (dg['Land-use']=='Sheep - modified land')]

    # Area for per-ha calc
    nat_area = da[(da['region']=='AUSTRALIA') & (da['Water_supply']=='ALL') &
                  (da['Land-use']=='Sheep - natural land')][area_col].sum()
    mod_area = da[(da['region']=='AUSTRALIA') & (da['Water_supply']=='ALL') &
                  (da['Land-use']=='Sheep - modified land')][area_col].sum()

    sheep_nat_ghg = sheep_nat['Value (t CO2e)'].sum()
    sheep_mod_ghg = sheep_mod['Value (t CO2e)'].sum()

    # HIR-Sheep abatement (on natural land)
    hir_sheep = dam[(dam['region']=='AUSTRALIA') & (dam['Water_supply']=='ALL') &
                    (dam['Agricultural Management Type']=='HIR - Sheep') &
                    (dam['Land-use']!='ALL')]
    hir_ghg = hir_sheep['Value (t CO2e)'].sum()
    hir_area = da[(da['region']=='AUSTRALIA') & (da['Water_supply']=='ALL')][area_col].sum()

    # Get HIR area
    hir_area_f = os.path.join(BASE, f'out_{yr}', f'area_agricultural_management_{yr}.csv')
    da2 = pd.read_csv(hir_area_f)
    hir_nat_area = da2[(da2['region']=='AUSTRALIA') & (da2['Water_supply']=='ALL') &
                       (da2['Type']=='HIR - Sheep') & (da2['Land-use']!='ALL')]['Area (ha)'].sum()

    print(f'\n  {yr}:')
    print(f'    Sheep NATURAL land: {nat_area/1e6:.2f} Mha, base GHG = {sheep_nat_ghg/1e6:.2f} Mt → {sheep_nat_ghg/nat_area:.3f} t/ha')
    print(f'    Sheep MODIFIED land:{mod_area/1e6:.2f} Mha, base GHG = {sheep_mod_ghg/1e6:.2f} Mt → {sheep_mod_ghg/mod_area:.3f} t/ha')
    print(f'    HIR-Sheep (natural): {hir_nat_area/1e6:.2f} Mha managed, abatement = {hir_ghg/1e6:.2f} Mt → {hir_ghg/hir_nat_area:.3f} t/ha managed')
    print(f'    Net natural sheep+HIR per ha = {(sheep_nat_ghg + hir_ghg)/nat_area:.3f} t/ha')

# ── 3. GHG per ha for all key LU for comparison ─────────────────────────────
print('\n=== 3. GHG per ha (t CO2e/ha) for key land uses 2023 ===')
yr = 2023
dg = pd.read_csv(os.path.join(BASE, f'out_{yr}', f'GHG_emissions_separate_agricultural_landuse_{yr}.csv'))
da = pd.read_csv(os.path.join(BASE, f'out_{yr}', f'area_agricultural_landuse_{yr}.csv'))
area_col = [c for c in da.columns if 'Area' in c][0]
aus_g = dg[(dg['region']=='AUSTRALIA') & (dg['Source']=='ALL') & (dg['Water_supply']=='ALL') & (dg['Land-use']!='ALL')]
aus_a = da[(da['region']=='AUSTRALIA') & (da['Water_supply']=='ALL') & (da['Land-use']!='ALL')]
ghg_dict = aus_g.set_index('Land-use')['Value (t CO2e)'].to_dict()
area_dict = aus_a.groupby('Land-use')[area_col].sum().to_dict()
for lu in ['Sheep - modified land', 'Sheep - natural land', 'Beef - modified land',
           'Beef - natural land', 'Winter cereals', 'Unallocated - modified land',
           'Unallocated - natural land']:
    g = ghg_dict.get(lu, 0)
    a = area_dict.get(lu, 1)
    if a > 1e5:
        print(f'  {lu:<40}: {g/a:.4f} t/ha  ({g/1e6:.2f} Mt, {a/1e6:.2f} Mha)')
