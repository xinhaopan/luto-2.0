"""Check FLC multipliers for ALL land use types and profit for all LU in AgS3 2022."""
import pandas as pd, os, sys

sys.path.insert(0, r'F:/Users/s222552331/Work/LUTO2_XH/luto-2.0')
BASE_OLD = r'F:/Users/s222552331/Work/LUTO2_XH/luto-2.0/output/20260512_Paper3_aquila/Run_3_SCN_AgS3/output/2026_05_13__00_29_59_RF5_2010-2050'
BASE_NEW_ROOT = r'F:/Users/s222552331/Work/LUTO2_XH/luto-2.0/output/20260513_Paper3_test/Run_3_SCN_AgS3/output'
subdirs = [d for d in os.listdir(BASE_NEW_ROOT) if os.path.isdir(os.path.join(BASE_NEW_ROOT, d))]
BASE_NEW = os.path.join(BASE_NEW_ROOT, subdirs[0])

# ── 1. FLC multiplier for ALL land use columns in 2018-2026 ──────────────────
print('=== 1. FLC_multiplier_medium ALL columns (subset years) ===')
xl = pd.ExcelFile(r'F:/Users/s222552331/Work/LUTO2_XH/luto-2.0/input/FLC_cost_multipliers.xlsx')
flc = xl.parse('FLC_multiplier_medium')
yr_col = [c for c in flc.columns if 'year' in str(c).lower() or c == 'Year'][0]
sub = flc[(flc[yr_col] >= 2018) & (flc[yr_col] <= 2026)]
print(sub.to_string(index=False))

# ── 2. Profit/ha ALL land uses OLD vs NEW in 2022 ───────────────────────────
print('\n=== 2. All LU profit/ha 2022: OLD (FLC_medium) vs NEW (flat FLC) ===')
for yr in [2021, 2022, 2023]:
    print(f'\n  {yr}:')
    print(f'  {"Land-use":<40} {"OLD":>8} {"NEW":>8}')
    profits = {}
    for base, label in [(BASE_OLD, 'OLD'), (BASE_NEW, 'NEW')]:
        fp = os.path.join(base, f'out_{yr}', f'economics_ag_profit_{yr}.csv')
        fa = os.path.join(base, f'out_{yr}', f'area_agricultural_landuse_{yr}.csv')
        if not os.path.exists(fp): continue
        de = pd.read_csv(fp); da = pd.read_csv(fa)
        val_col = [c for c in de.columns if 'Value' in c or '$' in c][0]
        area_col = [c for c in da.columns if 'Area' in c][0]
        aus_p = de.query("region == 'AUSTRALIA' and Water_supply == 'ALL'")
        aus_a = da.query("region == 'AUSTRALIA' and Water_supply == 'ALL'")
        for lu in aus_p['Land-use'].unique():
            if lu == 'ALL': continue
            ep = aus_p[aus_p['Land-use'] == lu]
            ea = aus_a[aus_a['Land-use'] == lu]
            profit = ep[val_col].sum()
            area = ea[area_col].sum()
            if area > 1e5:
                profits.setdefault(lu, {})[label] = profit / area
    for lu, pdict in sorted(profits.items()):
        old = f'{pdict.get("OLD", 0):.0f}' if 'OLD' in pdict else 'N/A'
        new = f'{pdict.get("NEW", 0):.0f}' if 'NEW' in pdict else 'N/A'
        mark = ' ***' if abs(float(old) - float(new)) > 100 and old != 'N/A' and new != 'N/A' else ''
        print(f'  {lu:<40} {old:>8} {new:>8}{mark}')

# ── 3. Sheep yield/ha (kg/ha) OLD vs NEW ─────────────────────────────────────
print('\n=== 3. Sheep meat yield/ha (kg/ha) OLD vs NEW ===')
for yr in [2020, 2021, 2022, 2023]:
    row = []
    for base, label in [(BASE_OLD, 'OLD'), (BASE_NEW, 'NEW')]:
        fq = os.path.join(base, f'out_{yr}', f'quantity_comparison_{yr}.csv')
        fa = os.path.join(base, f'out_{yr}', f'area_agricultural_landuse_{yr}.csv')
        if not os.path.exists(fq): row.append(f'{label}=N/A'); continue
        dq = pd.read_csv(fq); da = pd.read_csv(fa)
        area_col = [c for c in da.columns if 'Area' in c][0]
        aus_a = da.query("region == 'AUSTRALIA' and Water_supply == 'ALL' and `Land-use` == 'Sheep - modified land'")
        sheep_area = aus_a[area_col].sum()
        r = dq[dq['Commodity'] == 'Sheep meat']
        if len(r) and sheep_area > 0:
            prod = r['Prod_targ_year (tonnes, KL)'].values[0]
            row.append(f'{label}:{prod/sheep_area*1e6:.1f}kg/ha({prod/1e3:.0f}kt/{sheep_area/1e6:.2f}Mha)')
        else:
            row.append(f'{label}=N/A')
    print(f'  {yr}: ' + '  '.join(row))

# ── 4. Area of ALL sheep types (modified + natural) ──────────────────────────
print('\n=== 4. Sheep area by type OLD vs NEW ===')
for yr in [2020, 2021, 2022, 2023]:
    print(f'  {yr}:')
    for base, label in [(BASE_OLD, 'OLD'), (BASE_NEW, 'NEW')]:
        fa = os.path.join(base, f'out_{yr}', f'area_agricultural_landuse_{yr}.csv')
        if not os.path.exists(fa): continue
        da = pd.read_csv(fa)
        area_col = [c for c in da.columns if 'Area' in c][0]
        aus = da.query("region == 'AUSTRALIA' and Water_supply == 'ALL'")
        sheep_rows = aus[aus['Land-use'].str.contains('Sheep', na=False)]
        parts = []
        for _, r in sheep_rows[sheep_rows['Land-use'] != 'ALL'].iterrows():
            if r[area_col] > 1e5:
                parts.append(f'{r["Land-use"]}:{r[area_col]/1e6:.2f}Mha')
        print(f'    {label}: ' + '  '.join(parts))
