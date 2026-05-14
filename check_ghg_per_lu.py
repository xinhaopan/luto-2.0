"""Check GHG intensity per ha for each land use to verify why sheep is overproduced."""
import pandas as pd, os

BASE = r'F:/Users/s222552331/Work/LUTO2_XH/luto-2.0/output/20260513_Paper3_test/Run_1_SCN_AgS1/output/2026_05_13__22_14_05_RF5_2010-2050'

print('=== GHG intensity (t CO2e/ha) by land use, AUSTRALIA ALL water, ALL source ===')
print(f'{"Land-use":<40} {"2022 t/ha":>10} {"2023 t/ha":>10}')
print('-' * 65)

ghg_by_lu = {}
for yr in [2022, 2023]:
    fg = os.path.join(BASE, f'out_{yr}', f'GHG_emissions_separate_agricultural_landuse_{yr}.csv')
    fa = os.path.join(BASE, f'out_{yr}', f'area_agricultural_landuse_{yr}.csv')
    dg = pd.read_csv(fg)
    da = pd.read_csv(fa)

    # GHG: AUSTRALIA, ALL water, ALL source, specific land-use
    ghg_aus = dg.query("region == 'AUSTRALIA' and Water_supply == 'ALL' and Source == 'ALL' and `Land-use` != 'ALL'")
    area_aus = da.query("region == 'AUSTRALIA' and Water_supply == 'ALL' and `Land-use` != 'ALL'")

    ghg_lu = ghg_aus.groupby('Land-use')['Value (t CO2e)'].sum()
    area_lu = area_aus.groupby('Land-use')['Area (ha)'].sum()

    intensity = (ghg_lu / area_lu.clip(lower=1)).rename(yr)
    ghg_by_lu[yr] = intensity

df = pd.DataFrame(ghg_by_lu).fillna(0)
df = df.sort_values(2022, ascending=False)
for lu, row in df.iterrows():
    print(f'{lu:<40} {row[2022]:>10.2f} {row[2023]:>10.2f}')

# Also check unallocated separately
print('\n=== Total GHG (Mt CO2e) by land use (not per ha), AUSTRALIA 2022 vs 2023 ===')
for yr in [2022, 2023]:
    fg = os.path.join(BASE, f'out_{yr}', f'GHG_emissions_separate_agricultural_landuse_{yr}.csv')
    dg = pd.read_csv(fg)
    ghg_aus = dg.query("region == 'AUSTRALIA' and Water_supply == 'ALL' and Source == 'ALL' and `Land-use` != 'ALL'")
    ghg_lu = ghg_aus.groupby('Land-use')['Value (t CO2e)'].sum().sort_values(ascending=False)
    print(f'\n  {yr} top emitters:')
    for lu, v in ghg_lu.head(10).items():
        print(f'    {lu:<40}: {v/1e9:.3f} Gt')

# Check transition penalty GHG (if any)
print('\n=== Transition penalty GHG (2022 vs 2023) ===')
for yr in [2022, 2023]:
    ft = os.path.join(BASE, f'out_{yr}', f'GHG_emissions_separate_transition_penalty_{yr}.csv')
    if not os.path.exists(ft): continue
    dt = pd.read_csv(ft)
    aus = dt.query("region == 'AUSTRALIA'") if 'region' in dt.columns else dt
    val_col = [c for c in dt.columns if 'value' in c.lower() or 'co2' in c.lower() or 't CO2' in c][0]
    total = aus[val_col].sum()
    print(f'  {yr} transition penalty GHG: {total/1e6:.1f} Mt CO2e')
