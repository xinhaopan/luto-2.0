"""Check Area_cost multipliers and the actual FLC cost impact on sheep profit."""
import pandas as pd, os

# ── 1. Area_cost multiplier ───────────────────────────────────────────────────
print('=== 1. Area_cost.xlsx ===')
xl = pd.ExcelFile(r'F:/Users/s222552331/Work/LUTO2_XH/luto-2.0/input/Area_cost.xlsx')
print('Sheets:', xl.sheet_names)
for sht in xl.sheet_names[:3]:
    ac = xl.parse(sht)
    print(f'\nSheet: {sht}, columns: {ac.columns.tolist()[:8]}')
    yr_candidates = [c for c in ac.columns if 'year' in str(c).lower() or c == 'Year']
    if not yr_candidates:
        print(ac.head(5).to_string()); continue
    yr_col = yr_candidates[0]
    sub = ac[(ac[yr_col] >= 2019) & (ac[yr_col] <= 2026)]
    print(sub.to_string(index=False))

# ── 2. Compute implied FLC cost per ha for sheep: profit difference × area ────
print('\n=== 2. Implied FLC base cost for sheep ($/ha) ===')
# OLD profit - NEW profit = FLC_multiplier_diff × FLC_base_cost
# FLC_multiplier_diff = 1.367 - 1.0 = 0.367 in 2022
BASE_OLD = r'F:/Users/s222552331/Work/LUTO2_XH/luto-2.0/output/20260512_Paper3_aquila/Run_3_SCN_AgS3/output/2026_05_13__00_29_59_RF5_2010-2050'
BASE_NEW_ROOT = r'F:/Users/s222552331/Work/LUTO2_XH/luto-2.0/output/20260513_Paper3_test/Run_3_SCN_AgS3/output'
subdirs = [d for d in os.listdir(BASE_NEW_ROOT) if os.path.isdir(os.path.join(BASE_NEW_ROOT, d))]
BASE_NEW = os.path.join(BASE_NEW_ROOT, subdirs[0])

flc_mults = {2020: 1.302507, 2021: 1.335140, 2022: 1.367134, 2023: 1.398507}
for yr in [2020, 2021, 2022, 2023]:
    profits = {}
    for base, label in [(BASE_OLD, 'OLD'), (BASE_NEW, 'NEW')]:
        fp = os.path.join(base, f'out_{yr}', f'economics_ag_cost_{yr}.csv')
        fa = os.path.join(base, f'out_{yr}', f'area_agricultural_landuse_{yr}.csv')
        if not os.path.exists(fp): continue
        dc = pd.read_csv(fp); da = pd.read_csv(fa)
        val_col = [c for c in dc.columns if 'Value' in c or '$' in c][0]
        area_col = [c for c in da.columns if 'Area' in c][0]
        aus_c = dc.query("region == 'AUSTRALIA' and Water_supply == 'ALL' and `Land-use` == 'Sheep - modified land'")
        aus_a = da.query("region == 'AUSTRALIA' and Water_supply == 'ALL' and `Land-use` == 'Sheep - modified land'")
        cost = aus_c[val_col].sum() if len(aus_c) else 0
        area = aus_a[area_col].sum() if len(aus_a) else 1
        cost_type = aus_c.groupby('Type')[val_col].sum().sort_values(ascending=False) if 'Type' in dc.columns else None
        profits[label] = {'cost_per_ha': cost/area, 'area': area/1e6, 'types': cost_type}
    m = flc_mults.get(yr, 1.0)
    print(f'\n  {yr} (FLC_mult={m:.3f}):')
    for label, d in profits.items():
        print(f'    {label}: cost/ha = {d["cost_per_ha"]:.0f} $/ha  area = {d["area"]:.2f} Mha')
        if d['types'] is not None:
            for t, v in d['types'].items():
                if abs(v) > 1e7:
                    print(f'      {t:<35}: {v/1e9:.2f} B')

# ── 3. Sheep commodity price / revenue year-by-year ─────────────────────────
print('\n=== 3. Sheep revenue/ha OLD vs NEW ===')
for yr in [2020, 2021, 2022, 2023]:
    row = []
    for base, label in [(BASE_OLD, 'OLD'), (BASE_NEW, 'NEW')]:
        fp = os.path.join(base, f'out_{yr}', f'economics_ag_revenue_{yr}.csv')
        fa = os.path.join(base, f'out_{yr}', f'area_agricultural_landuse_{yr}.csv')
        if not os.path.exists(fp): continue
        dr = pd.read_csv(fp); da = pd.read_csv(fa)
        val_col = [c for c in dr.columns if 'Value' in c or '$' in c][0]
        area_col = [c for c in da.columns if 'Area' in c][0]
        aus_r = dr.query("region == 'AUSTRALIA' and Water_supply == 'ALL' and `Land-use` == 'Sheep - modified land'")
        aus_a = da.query("region == 'AUSTRALIA' and Water_supply == 'ALL' and `Land-use` == 'Sheep - modified land'")
        rev = aus_r[val_col].sum()
        area = aus_a[area_col].sum()
        row.append(f'{label}:{rev/area:.0f}$/ha')
    print(f'  {yr}: ' + '  '.join(row))
