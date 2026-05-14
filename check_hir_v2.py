"""Check AgMgt area/GHG correctly using right column names."""
import pandas as pd, os

BASE = r'F:/Users/s222552331/Work/LUTO2_XH/luto-2.0/output/20260513_Paper3_test/Run_1_SCN_AgS1/output/2026_05_13__22_14_05_RF5_2010-2050'

# ── 1. AgMgt area (column name = 'Type') ─────────────────────────────────────
print('=== 1. AgMgt area AUSTRALIA (Mha) by Type, ALL land-use ===')
for yr in [2021, 2022, 2023]:
    f = os.path.join(BASE, f'out_{yr}', f'area_agricultural_management_{yr}.csv')
    da = pd.read_csv(f)
    aus = da.query("region == 'AUSTRALIA' and Water_supply == 'ALL' and `Land-use` == 'ALL'")
    print(f'\n  {yr}:')
    for _, row in aus.iterrows():
        print(f'    {row["Type"]:<40}: {row["Area (ha)"]/1e6:.3f} Mha')

# ── 2. AgMgt GHG by Type and Land-use ────────────────────────────────────────
print('\n=== 2. AgMgt GHG AUSTRALIA (Mt CO2e), ALL water, Source==ALL ===')
for yr in [2021, 2022, 2023]:
    f = os.path.join(BASE, f'out_{yr}', f'GHG_emissions_separate_agricultural_management_{yr}.csv')
    dg = pd.read_csv(f)
    # AgMgt GHG file: columns = [Agricultural Management Type, Water_supply, Land-use, Value (t CO2e), Year, Type, region]
    aus = dg.query("region == 'AUSTRALIA' and Water_supply == 'ALL' and `Land-use` != 'ALL'")
    val_col = 'Value (t CO2e)'
    print(f'\n  {yr}:')
    totals = aus.groupby('Agricultural Management Type')[val_col].sum().sort_values()
    for am, v in totals.items():
        if abs(v) > 1e4:
            print(f'    {am:<40}: {v/1e6:.2f} Mt')
    total_all = dg.query("region == 'AUSTRALIA' and Water_supply == 'ALL' and `Land-use` == 'ALL'")[val_col].sum()
    print(f'    TOTAL: {total_all/1e6:.2f} Mt')

# ── 3. GHG budget using Source==ALL correctly ────────────────────────────────
print('\n=== 3. GHG budget (Source==ALL, Water==ALL, LU==ALL) ===')
for yr in [2021, 2022, 2023, 2024]:
    ag_f = os.path.join(BASE, f'out_{yr}', f'GHG_emissions_separate_agricultural_landuse_{yr}.csv')
    am_f = os.path.join(BASE, f'out_{yr}', f'GHG_emissions_separate_agricultural_management_{yr}.csv')
    tr_f = os.path.join(BASE, f'out_{yr}', f'GHG_emissions_separate_transition_penalty_{yr}.csv')
    ol_f = os.path.join(BASE, f'out_{yr}', f'GHG_emissions_offland_commodity_{yr}.csv')

    # Ag land: Source=='ALL', Water=='ALL', LU=='ALL'
    ag_df = pd.read_csv(ag_f)
    ag = ag_df.query("region == 'AUSTRALIA' and Source == 'ALL' and Water_supply == 'ALL' and `Land-use` == 'ALL'")['Value (t CO2e)'].sum()

    # AgMgt: LU=='ALL', Water=='ALL' (no Source column - it's 'Agricultural Management Type')
    am_df = pd.read_csv(am_f)
    am = am_df.query("region == 'AUSTRALIA' and Water_supply == 'ALL' and `Land-use` == 'ALL'")['Value (t CO2e)'].sum()

    # Transition penalty
    tr_df = pd.read_csv(tr_f)
    tr = tr_df.query("region == 'AUSTRALIA'")['Value (t CO2e)'].sum() if 'region' in tr_df.columns else tr_df['Value (t CO2e)'].sum()

    # Off-land commodity
    ol = 0
    if os.path.exists(ol_f):
        ol_df = pd.read_csv(ol_f)
        if 'region' in ol_df.columns:
            ol = ol_df.query("region == 'AUSTRALIA'").iloc[:, -2].sum()
        else:
            ol = ol_df.iloc[:, -2].sum()

    net = (ag + am + tr + ol) / 1e6
    print(f'  {yr}: Ag={ag/1e6:.1f}  AmMgt={am/1e6:.1f}  Trans={tr/1e6:.2f}  Offland={ol/1e6:.1f}  Net={net:.1f} Mt  (limit=65.58)')

# ── 4. Why sheep is unprofitable: detailed cost breakdown ──────────────────
print('\n=== 4. Sheep cost components (B AUD), AUSTRALIA, ALL water ===')
for yr in [2022, 2023]:
    f = os.path.join(BASE, f'out_{yr}', f'economics_ag_cost_{yr}.csv')
    de = pd.read_csv(f)
    cols = de.columns.tolist()
    val_col = [c for c in cols if 'Value' in c or '$' in c][0]
    aus = de.query("region == 'AUSTRALIA' and Water_supply == 'ALL' and `Land-use` == 'Sheep - modified land'")
    print(f'\n  {yr} Sheep cost by type:')
    if 'Type' in de.columns:
        by_type = aus.groupby('Type')[val_col].sum().sort_values(ascending=False)
        for t, v in by_type.items():
            print(f'    {t:<35}: {v/1e9:.2f} B')
    print(f'  Total cost: {aus[val_col].sum()/1e9:.2f} B')

    f2 = os.path.join(BASE, f'out_{yr}', f'economics_ag_revenue_{yr}.csv')
    de2 = pd.read_csv(f2)
    val2 = [c for c in de2.columns if 'Value' in c or '$' in c][0]
    aus2 = de2.query("region == 'AUSTRALIA' and Water_supply == 'ALL' and `Land-use` == 'Sheep - modified land'")
    print(f'  Total revenue: {aus2[val2].sum()/1e9:.2f} B')

# ── 5. HIR management area and GHG ───────────────────────────────────────────
print('\n=== 5. HIR management area and GHG 2020-2024 ===')
for yr in [2020, 2021, 2022, 2023, 2024]:
    fa = os.path.join(BASE, f'out_{yr}', f'area_agricultural_management_{yr}.csv')
    fg = os.path.join(BASE, f'out_{yr}', f'GHG_emissions_separate_agricultural_management_{yr}.csv')
    da = pd.read_csv(fa)
    dg = pd.read_csv(fg)
    aus_a = da.query("region == 'AUSTRALIA' and Water_supply == 'ALL'")
    aus_g = dg.query("region == 'AUSTRALIA' and Water_supply == 'ALL' and `Land-use` != 'ALL'")
    hir_area = aus_a[aus_a['Type'].str.contains('HIR', na=False)].groupby('Type')['Area (ha)'].sum()
    hir_ghg  = aus_g[aus_g['Agricultural Management Type'].str.contains('HIR', na=False)].groupby('Agricultural Management Type')['Value (t CO2e)'].sum()
    if len(hir_area):
        print(f'\n  {yr} HIR area:')
        for t, v in hir_area.items():
            print(f'    {t}: {v/1e6:.3f} Mha')
        print(f'  {yr} HIR GHG:')
        for t, v in hir_ghg.items():
            print(f'    {t}: {v/1e6:.2f} Mt')
