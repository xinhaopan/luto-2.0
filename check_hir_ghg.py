"""Check HIR management area/GHG to understand why sheep overproduced despite losses."""
import pandas as pd, os

BASE = r'F:/Users/s222552331/Work/LUTO2_XH/luto-2.0/output/20260513_Paper3_test/Run_1_SCN_AgS1/output/2026_05_13__22_14_05_RF5_2010-2050'

# ── 1. AgMgt area by management type ────────────────────────────────────────
print('=== 1. Agricultural management area AUSTRALIA (Mha) ===')
for yr in [2021, 2022, 2023, 2024]:
    f = os.path.join(BASE, f'out_{yr}', f'area_agricultural_management_{yr}.csv')
    if not os.path.exists(f): continue
    da = pd.read_csv(f)
    aus = da.query("region == 'AUSTRALIA' and Water_supply == 'ALL'")
    area_col = [c for c in da.columns if 'Area' in c][0]
    am_col = [c for c in da.columns if 'Mgt' in c or 'management' in c.lower() or 'Ag_Mgt' in c][0]
    print(f'\n  {yr}:')
    totals = aus.groupby(am_col)[area_col].sum().sort_values(ascending=False)
    for am, v in totals.items():
        if v > 1e4:
            print(f'    {am:<40}: {v/1e6:.3f} Mha')

# ── 2. AgMgt GHG by management type ─────────────────────────────────────────
print('\n=== 2. AgMgt GHG reduction AUSTRALIA (Mt CO2e), Source==ALL ===')
for yr in [2021, 2022, 2023, 2024]:
    f = os.path.join(BASE, f'out_{yr}', f'GHG_emissions_separate_agricultural_management_{yr}.csv')
    if not os.path.exists(f): continue
    dg = pd.read_csv(f)
    cols = dg.columns.tolist()
    aus = dg.query("region == 'AUSTRALIA' and Source == 'ALL'")
    val_col = [c for c in cols if 'Value' in c or 't CO2' in c][0]
    am_col = [c for c in cols if 'Mgt' in c or 'management' in c.lower() or 'Am' in c][0]
    print(f'\n  {yr}:')
    # Group by management and land use
    if 'Land-use' in dg.columns:
        group_cols = [am_col, 'Land-use'] if 'Land-use' in dg.columns else [am_col]
        totals = aus.groupby(group_cols)[val_col].sum()
    else:
        totals = aus.groupby(am_col)[val_col].sum()
    totals = totals.sort_values()
    for k, v in totals.items():
        if abs(v) > 1e6:
            print(f'    {str(k):<50}: {v/1e6:.2f} Mt')
    print(f'    TOTAL AgMgt GHG: {aus[val_col].sum()/1e6:.2f} Mt')

# ── 3. GHG from Ag land use - just the Source==ALL and Water==ALL values ─────
print('\n=== 3. True GHG budget (Source==ALL, Water==ALL) ===')
for yr in [2021, 2022, 2023, 2024]:
    ag_f = os.path.join(BASE, f'out_{yr}', f'GHG_emissions_separate_agricultural_landuse_{yr}.csv')
    am_f = os.path.join(BASE, f'out_{yr}', f'GHG_emissions_separate_agricultural_management_{yr}.csv')
    tr_f = os.path.join(BASE, f'out_{yr}', f'GHG_emissions_separate_transition_penalty_{yr}.csv')

    def get_aus_all(fpath, extra_filter=''):
        if not os.path.exists(fpath): return 0
        df = pd.read_csv(fpath)
        q = "region == 'AUSTRALIA' and Source == 'ALL'"
        if 'Water_supply' in df.columns: q += " and Water_supply == 'ALL'"
        if 'Land-use' in df.columns: q += " and `Land-use` == 'ALL'"
        if extra_filter: q += f" and {extra_filter}"
        aus = df.query(q)
        val_col = [c for c in df.columns if 'Value' in c or 't CO2' in c][0]
        return aus[val_col].sum()

    ag = get_aus_all(ag_f)
    am = get_aus_all(am_f)
    tr = get_aus_all(tr_f) if os.path.exists(tr_f) else 0
    net = (ag + am + tr) / 1e6
    print(f'  {yr}: Ag={ag/1e6:.1f}  AmMgt={am/1e6:.1f}  Trans={tr/1e6:.2f}  Net={net:.1f} Mt  (limit=65.58)')

# ── 4. Why is sheep unprofitable? Profit components ─────────────────────────
print('\n=== 4. Sheep profit components (B AUD) AUSTRALIA 2022 vs 2023 ===')
for yr in [2022, 2023]:
    rev_f = os.path.join(BASE, f'out_{yr}', f'economics_ag_revenue_{yr}.csv')
    cost_f = os.path.join(BASE, f'out_{yr}', f'economics_ag_cost_{yr}.csv')
    for fname, label in [(rev_f, 'Revenue'), (cost_f, 'Cost')]:
        if not os.path.exists(fname): continue
        df = pd.read_csv(fname)
        aus = df.query("region == 'AUSTRALIA' and Water_supply == 'ALL' and `Land-use` == 'Sheep - modified land'")
        val_col = [c for c in df.columns if 'Value' in c or '$' in c][0]
        type_col = [c for c in df.columns if c == 'Type'][0] if 'Type' in df.columns else None
        total = aus[val_col].sum()
        print(f'  {yr} Sheep {label} total: {total/1e9:.2f} B')
        if type_col:
            by_type = aus.groupby(type_col)[val_col].sum().sort_values(ascending=False)
            for t, v in by_type.items():
                if abs(v) > 1e7:
                    print(f'    {t:<35}: {v/1e9:.2f} B')
    print()
