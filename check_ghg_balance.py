"""Check what compensates for GHG increase from sheep expansion: non-ag sequestration."""
import pandas as pd, os

BASE = r'F:/Users/s222552331/Work/LUTO2_XH/luto-2.0/output/20260513_Paper3_test/Run_1_SCN_AgS1/output/2026_05_13__22_14_05_RF5_2010-2050'

# ── 1. Non-ag area change 2022 vs 2023 ───────────────────────────────────────
print('=== 1. Non-ag area AUSTRALIA (Mha), key sequestration land uses ===')
for yr in [2020, 2021, 2022, 2023, 2024]:
    f = os.path.join(BASE, f'out_{yr}', f'area_non_agricultural_landuse_{yr}.csv')
    if not os.path.exists(f): continue
    da = pd.read_csv(f)
    aus = da.query("region == 'AUSTRALIA'")
    area_col = [c for c in da.columns if 'Area' in c][0]
    print(f'  {yr}:')
    for lu in ['Environmental Plantings', 'Riparian Plantings', 'Carbon Plantings (Block)',
               'Sheep Carbon Plantings (Belt)', 'Beef Carbon Plantings (Belt)',
               'Beef Agroforestry', 'Sheep Agroforestry', 'BECCS']:
        row = aus[aus['Land-use'] == lu]
        v = row[area_col].sum() if len(row) else 0
        if v > 0:
            print(f'    {lu:<40}: {v/1e6:.3f} Mha')
    total = aus[area_col].sum()
    print(f'    TOTAL non-ag: {total/1e6:.3f} Mha')

# ── 2. Non-ag GHG (carbon sequestration) 2021-2024 ──────────────────────────
print('\n=== 2. Non-ag GHG sequestration AUSTRALIA (Mt CO2e) ===')
for yr in [2020, 2021, 2022, 2023, 2024]:
    f = os.path.join(BASE, f'out_{yr}', f'GHG_emissions_separate_no_ag_reduction_{yr}.csv')
    if not os.path.exists(f): continue
    dg = pd.read_csv(f)
    aus = dg.query("region == 'AUSTRALIA'") if 'region' in dg.columns else dg
    val_col = [c for c in dg.columns if 'value' in c.lower() or 'co2' in c.lower() or 't CO2' in c][0]
    # Show by land use
    if 'Land-use' in dg.columns:
        totals = aus.groupby('Land-use')[val_col].sum().sort_values()
        print(f'\n  {yr}:')
        for lu, v in totals.items():
            if abs(v) > 1e5:
                print(f'    {lu:<40}: {v/1e6:.2f} Mt')
        print(f'    TOTAL non-ag GHG: {aus[val_col].sum()/1e6:.2f} Mt')
    else:
        total = aus[val_col].sum()
        print(f'  {yr} total non-ag GHG: {total/1e6:.2f} Mt')

# ── 3. Full GHG budget: ag + non-ag + transition = total ────────────────────
print('\n=== 3. Full GHG budget breakdown 2021-2024 (Mt CO2e) ===')
for yr in [2021, 2022, 2023, 2024]:
    ag_f = os.path.join(BASE, f'out_{yr}', f'GHG_emissions_separate_agricultural_landuse_{yr}.csv')
    am_f = os.path.join(BASE, f'out_{yr}', f'GHG_emissions_separate_agricultural_management_{yr}.csv')
    nonag_f = os.path.join(BASE, f'out_{yr}', f'GHG_emissions_separate_no_ag_reduction_{yr}.csv')
    tr_f = os.path.join(BASE, f'out_{yr}', f'GHG_emissions_separate_transition_penalty_{yr}.csv')

    def get_total(fpath):
        if not os.path.exists(fpath): return 0
        df = pd.read_csv(fpath)
        aus = df.query("region == 'AUSTRALIA'") if 'region' in df.columns else df
        val_col = [c for c in df.columns if 'value' in c.lower() or 't CO2' in c][0]
        # Get AUSTRALIA ALL row if available, else sum all
        if 'Land-use' in aus.columns:
            all_row = aus.query("`Land-use` == 'ALL' and Water_supply == 'ALL'") if 'Water_supply' in aus.columns else aus.query("`Land-use` == 'ALL'")
            if len(all_row):
                return all_row[val_col].sum()
        return aus[val_col].sum()

    ag_total = get_total(ag_f)
    am_total = get_total(am_f)
    nonag_total = get_total(nonag_f)
    tr_total = get_total(tr_f)
    net = (ag_total + am_total + nonag_total + tr_total) / 1e6
    print(f'  {yr}:')
    print(f'    Ag land GHG:      {ag_total/1e6:>8.2f} Mt')
    print(f'    Ag mgt GHG:       {am_total/1e6:>8.2f} Mt')
    print(f'    Non-ag GHG (seq): {nonag_total/1e6:>8.2f} Mt')
    print(f'    Transition GHG:   {tr_total/1e6:>8.2f} Mt')
    print(f'    NET total:        {net:>8.2f} Mt  (limit=65.58 Mt)')

# ── 4. Ag2non-ag transition costs 2022 vs 2023 ──────────────────────────────
print('\n=== 4. Ag→Non-ag transition cost (B AUD) AUSTRALIA 2022 vs 2023 ===')
for yr in [2022, 2023]:
    f = os.path.join(BASE, f'out_{yr}', f'transition_cost_ag2non_ag_{yr}.csv')
    if not os.path.exists(f): continue
    dt = pd.read_csv(f)
    aus_all = dt.query("region == 'AUSTRALIA' and `From-land-use` == 'ALL' and `To-land-use` == 'ALL'")
    total = aus_all['Cost ($)'].sum()
    print(f'  {yr}: {total/1e9:.2f} B AUD')
    # Top destinations
    detail = dt.query("region == 'AUSTRALIA' and `From-land-use` != 'ALL' and `To-land-use` != 'ALL'")
    detail = detail.groupby('To-land-use')['Cost ($)'].sum().sort_values(ascending=False)
    for lu, v in detail.head(5).items():
        print(f'    → {lu:<40}: {v/1e9:.2f} B')

# ── 5. Economics profit for sheep vs unallocated (why sheep chosen) ──────────
print('\n=== 5. Economics profit/ha: Sheep vs Unallocated vs Winter cereals ===')
for yr in [2022, 2023]:
    fp = os.path.join(BASE, f'out_{yr}', f'economics_ag_profit_{yr}.csv')
    fa = os.path.join(BASE, f'out_{yr}', f'area_agricultural_landuse_{yr}.csv')
    if not os.path.exists(fp): continue
    de = pd.read_csv(fp)
    da = pd.read_csv(fa)
    val_col = [c for c in de.columns if 'Value' in c or '$' in c][0]
    area_col = [c for c in da.columns if 'Area' in c][0]

    print(f'\n  {yr}:')
    for lu in ['Sheep - modified land', 'Beef - modified land', 'Winter cereals',
               'Unallocated - modified land']:
        ep = de.query("region == 'AUSTRALIA' and `Land-use` == @lu and Water_supply == 'ALL'")
        ea = da.query("region == 'AUSTRALIA' and `Land-use` == @lu and Water_supply == 'ALL'")
        profit = ep[val_col].sum() if len(ep) else 0
        area = ea[area_col].sum() if len(ea) else 1
        ppa = profit / max(area, 1)
        print(f'    {lu:<40}: {profit/1e9:.2f} B  ({ppa:.0f} $/ha)')
