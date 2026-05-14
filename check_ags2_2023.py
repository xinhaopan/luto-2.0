"""Investigate AgS2 2023 ag2ag transition cost spike despite tree planting availability."""
import pandas as pd, os

S1 = r'F:/Users/s222552331/Work/LUTO2_XH/luto-2.0/output/20260513_Paper3_aquila/Run_1_SCN_AgS1/output/2026_05_13__19_24_46_RF5_2010-2050'
S2 = r'F:/Users/s222552331/Work/LUTO2_XH/luto-2.0/output/20260513_Paper3_aquila/Run_2_SCN_AgS2/output/2026_05_13__19_24_45_RF5_2010-2050'

# ── 1. Non-ag area year-by-year (AgS2 only) ──────────────────────────────────
print('=== 1. Non-ag area AUSTRALIA (Mha), AgS2 ===')
for yr in range(2015, 2030):
    f = os.path.join(S2, f'out_{yr}', f'area_non_agricultural_landuse_{yr}.csv')
    if not os.path.exists(f): continue
    da = pd.read_csv(f)
    aus = da.query("region == 'AUSTRALIA'")
    area_col = [c for c in da.columns if 'Area' in c][0]
    lu_col = [c for c in da.columns if 'Land' in c or 'land' in c][0]
    total = aus[area_col].sum()
    top = aus.nlargest(3, area_col)[[lu_col, area_col]]
    top_str = '  '.join(f'{r[lu_col][:20]}:{r[area_col]/1e6:.2f}Mha' for _, r in top.iterrows())
    print(f'  {yr}: total={total/1e6:.2f} Mha   top: {top_str}')

# ── 2. Ag2Ag transition cost by year: AgS1 vs AgS2 ───────────────────────────
print('\n=== 2. Ag2Ag transition cost (B AUD): AgS1 vs AgS2 ===')
print(f'{"Year":<6} | {"AgS1":>10} | {"AgS2":>10}')
for yr in range(2015, 2030):
    vals = []
    for base in [S1, S2]:
        f = os.path.join(base, f'out_{yr}', f'transition_cost_ag2ag_{yr}.csv')
        if not os.path.exists(f):
            vals.append(None); continue
        df = pd.read_csv(f)
        aus_all = df.query("region == 'AUSTRALIA' and `From-land-use` == 'ALL' and `To-land-use` == 'ALL' and Type == 'ALL'")
        vals.append(aus_all['Cost ($)'].sum() / 1e9)
    if any(v is not None for v in vals):
        v1 = f'{vals[0]:.2f}' if vals[0] is not None else 'N/A'
        v2 = f'{vals[1]:.2f}' if vals[1] is not None else 'N/A'
        mark = ' <---' if any(v is not None and v > 5 for v in vals) else ''
        print(f'{yr:<6} | {v1:>10} | {v2:>10}{mark}')

# ── 3. Top ag2ag transitions in 2023 (AgS2) ─────────────────────────────────
print('\n=== 3. AgS2 top ag2ag transitions in 2023 (Establishment cost) ===')
for yr in [2022, 2023, 2024]:
    f = os.path.join(S2, f'out_{yr}', f'transition_cost_ag2ag_{yr}.csv')
    if not os.path.exists(f): continue
    df = pd.read_csv(f)
    sub = df.query("region == 'AUSTRALIA' and `From-land-use` != 'ALL' and `To-land-use` != 'ALL' and Type == 'Establishment cost'")
    sub = sub.nlargest(8, 'Cost ($)')
    print(f'\n  {yr} top transitions:')
    print(sub[['From-land-use', 'To-land-use', 'Cost ($)']].to_string(index=False))

# ── 4. Non-ag transition cost year-by-year (AgS2) ───────────────────────────
print('\n=== 4. Non-ag (tree planting) transition cost (B AUD), AgS2 ===')
for yr in range(2015, 2030):
    f = os.path.join(S2, f'out_{yr}', f'transition_cost_non_ag_{yr}.csv')
    if not os.path.exists(f): continue
    df = pd.read_csv(f)
    aus = df.query("region == 'AUSTRALIA'") if 'region' in df.columns else df
    cost_col = [c for c in df.columns if 'Cost' in c or '$' in c][0]
    total = aus[cost_col].sum()
    if total > 1e8:
        print(f'  {yr}: {total/1e9:.2f} B')
    else:
        print(f'  {yr}: {total/1e9:.3f} B')

# ── 5. Area change 2022-2024 (AgS2): key agricultural LU ────────────────────
print('\n=== 5. Area (Mha) AUSTRALIA key ag LU, AgS2 ===')
for yr in [2020, 2021, 2022, 2023, 2024]:
    f = os.path.join(S2, f'out_{yr}', f'area_agricultural_landuse_{yr}.csv')
    if not os.path.exists(f): continue
    da = pd.read_csv(f)
    aus = da.query("region == 'AUSTRALIA' and Water_supply == 'ALL'")
    area_col = [c for c in da.columns if 'Area' in c][0]
    vals = []
    for lu in ['Beef - modified land', 'Sheep - modified land', 'Winter cereals',
               'Unallocated - modified land', 'Dairy - modified land']:
        row = aus[aus['Land-use'] == lu]
        v = row[area_col].sum() if len(row) else 0
        vals.append(f'{lu[:18]:18s}:{v/1e6:.2f}')
    print(f'  {yr}:  ' + '  '.join(vals))

# ── 6. GHG budget AgS2 ───────────────────────────────────────────────────────
print('\n=== 6. AgS2 GHG budget (Mt CO2e) ===')
for yr in [2020, 2021, 2022, 2023, 2024, 2025]:
    ag_f = os.path.join(S2, f'out_{yr}', f'GHG_emissions_separate_agricultural_landuse_{yr}.csv')
    am_f = os.path.join(S2, f'out_{yr}', f'GHG_emissions_separate_agricultural_management_{yr}.csv')
    na_f = os.path.join(S2, f'out_{yr}', f'GHG_emissions_separate_no_ag_reduction_{yr}.csv')
    tr_f = os.path.join(S2, f'out_{yr}', f'GHG_emissions_separate_transition_penalty_{yr}.csv')
    if not os.path.exists(ag_f): continue

    ag_df = pd.read_csv(ag_f)
    ag = ag_df.query("region == 'AUSTRALIA' and Source == 'ALL' and Water_supply == 'ALL' and `Land-use` == 'ALL'")['Value (t CO2e)'].sum()

    am_df = pd.read_csv(am_f)
    am = am_df.query("region == 'AUSTRALIA' and Water_supply == 'ALL' and `Land-use` == 'ALL'")['Value (t CO2e)'].sum()

    na = 0
    if os.path.exists(na_f):
        na_df = pd.read_csv(na_f)
        q = "region == 'AUSTRALIA'"
        if 'Source' in na_df.columns: q += " and Source == 'ALL'"
        if 'Water_supply' in na_df.columns: q += " and Water_supply == 'ALL'"
        if 'Land-use' in na_df.columns: q += " and `Land-use` == 'ALL'"
        na = na_df.query(q)['Value (t CO2e)'].sum()

    tr = 0
    if os.path.exists(tr_f):
        tr_df = pd.read_csv(tr_f)
        tr = tr_df.query("region == 'AUSTRALIA'")['Value (t CO2e)'].sum() if 'region' in tr_df.columns else tr_df['Value (t CO2e)'].sum()

    net = (ag + am + na + tr) / 1e6
    # Get GHG limit from the settings file
    limit_str = ''
    sf = os.path.join(S2, 'model_run_settings.txt')
    print(f'  {yr}: Ag={ag/1e6:.1f}  AgMgt={am/1e6:.1f}  NonAg={na/1e6:.1f}  Trans={tr/1e6:.2f}  Net={net:.1f} Mt')

# ── 7. Sheep profit/ha AgS1 vs AgS2, 2020-2025 ──────────────────────────────
print('\n=== 7. Sheep profit/ha AgS1 vs AgS2 ($/ha) ===')
for yr in [2020, 2021, 2022, 2023, 2024]:
    row = []
    for base, label in [(S1, 'S1'), (S2, 'S2')]:
        fp = os.path.join(base, f'out_{yr}', f'economics_ag_profit_{yr}.csv')
        fa = os.path.join(base, f'out_{yr}', f'area_agricultural_landuse_{yr}.csv')
        if not os.path.exists(fp): row.append('N/A'); continue
        de = pd.read_csv(fp); da = pd.read_csv(fa)
        val_col = [c for c in de.columns if 'Value' in c or '$' in c][0]
        area_col = [c for c in da.columns if 'Area' in c][0]
        ep = de.query("region == 'AUSTRALIA' and Water_supply == 'ALL' and `Land-use` == 'Sheep - modified land'")
        ea = da.query("region == 'AUSTRALIA' and Water_supply == 'ALL' and `Land-use` == 'Sheep - modified land'")
        profit = ep[val_col].sum() if len(ep) else 0
        area = ea[area_col].sum() if len(ea) else 1
        row.append(f'{profit/area:.0f}')
    print(f'  {yr}: AgS1={row[0]:>7} $/ha  AgS2={row[1]:>7} $/ha')

# ── 8. GHG limit from settings ───────────────────────────────────────────────
print('\n=== 8. AgS2 key settings ===')
sf = os.path.join(S2, 'model_run_settings.txt')
with open(sf) as fh:
    for line in fh:
        k = line.split(':')[0]
        if k in ('GHG_EMISSIONS_LIMITS', 'DEMAND_CONSTRAINT_TYPE', 'AG2050_SCENARIO',
                 'GHG_CONSTRAINT_TYPE', 'BIODIVERSITY_TARGET_GBF_2', 'GBF2_CONSTRAINT_TYPE'):
            print(' ', line.strip()[:120])
