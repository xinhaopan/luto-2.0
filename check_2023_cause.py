"""Find the root cause of the 2023 ag2ag spike common to all scenarios."""
import pandas as pd, numpy as np, os

# ── 1. Transition cost multiplier ─────────────────────────────────────────────
print('=== 1. Transitions_cost_multiplier by year ===')
xl = pd.ExcelFile('input/cost_multipliers.xlsx')
print('Sheets:', xl.sheet_names)
df = xl.parse('Transitions cost multiplier', index_col='Year')
print(df.loc[2018:2028])

# ── 2. Establishment cost and water license cost multipliers ──────────────────
print('\n=== 2. Other cost multipliers ===')
for sht in xl.sheet_names:
    try:
        d = xl.parse(sht, index_col='Year')
        sub = d.loc[2018:2028] if 2018 in d.index else d.head(10)
        print(f'\nSheet: {sht}')
        print(sub.to_string())
    except Exception as e:
        print(f'Sheet {sht}: {e}')

# ── 3. T_MAT (transition cost matrix) - per-ha base cost ─────────────────────
print('\n=== 3. Transition matrix (base cost $/ha) - checking structure ===')
import luto.settings as settings
tmat_files = [f for f in os.listdir('input') if 'tmatrix' in f.lower() or 'transition' in f.lower()]
print('Transition-related input files:', tmat_files)

# ── 4. Commodity prices year by year ─────────────────────────────────────────
print('\n=== 4. Sheep commodity prices 2018-2026 ===')
price_files = [f for f in os.listdir('input') if 'price' in f.lower() and f.endswith('.csv')]
print('Price CSV files:', price_files)
for pf in price_files[:3]:
    try:
        dp = pd.read_csv(os.path.join('input', pf))
        print(f'\n  {pf}: columns={dp.columns.tolist()[:6]}')
        yr_col = [c for c in dp.columns if 'year' in c.lower() or c == 'Year']
        if yr_col:
            sub = dp[(dp[yr_col[0]] >= 2020) & (dp[yr_col[0]] <= 2026)]
            sheep_cols = [c for c in dp.columns if 'sheep' in c.lower() or 'wool' in c.lower()]
            if sheep_cols:
                print(sub[yr_col + sheep_cols[:4]].to_string(index=False))
    except Exception as e:
        print(f'  Error: {e}')

# ── 5. Check ag_price_multipliers.xlsx for sheep ─────────────────────────────
print('\n=== 5. ag_price_multipliers.xlsx ===')
try:
    xlp = pd.ExcelFile('input/ag_price_multipliers.xlsx')
    print('Sheets:', xlp.sheet_names)
    for sht in xlp.sheet_names:
        d = xlp.parse(sht)
        yr_col = [c for c in d.columns if 'year' in str(c).lower() or c == 'Year']
        if not yr_col: print(f'  {sht}: no Year col, cols={d.columns.tolist()[:5]}'); continue
        sub = d[(d[yr_col[0]] >= 2020) & (d[yr_col[0]] <= 2026)]
        sheep_cols = [c for c in d.columns if 'sheep' in str(c).lower() or 'wool' in str(c).lower() or 'lexp' in str(c).lower()]
        if sheep_cols:
            print(f'\n  Sheet: {sht}, sheep cols: {sheep_cols}')
            print(sub[yr_col + sheep_cols[:6]].to_string(index=False))
        else:
            print(f'  Sheet: {sht}, no sheep cols, showing all:')
            print(sub.to_string(index=False))
except Exception as e:
    print(f'Error: {e}')

# ── 6. Water limits - year-specific targets ───────────────────────────────────
print('\n=== 6. Water-related data check ===')
water_files = [f for f in os.listdir('input') if 'water' in f.lower()]
print('Water input files:', water_files[:8])

# ── 7. Per-ha transition cost for cereals→sheep across years in AgS1 ─────────
print('\n=== 7. Per-ha transition cost for cereals→sheep (AgS1) ===')
BASE = r'F:/Users/s222552331/Work/LUTO2_XH/luto-2.0/output/20260513_Paper3_aquila/Run_1_SCN_AgS1/output/2026_05_13__19_24_46_RF5_2010-2050'
for yr in [2019, 2020, 2021, 2022, 2023, 2024, 2025]:
    ft = os.path.join(BASE, f'out_{yr}', f'transition_cost_ag2ag_{yr}.csv')
    fa = os.path.join(BASE, f'out_{yr}', f'area_agricultural_landuse_{yr}.csv')
    fp_prev = os.path.join(BASE, f'out_{yr-1}', f'area_agricultural_landuse_{yr-1}.csv')
    if not os.path.exists(ft) or not os.path.exists(fa) or not os.path.exists(fp_prev): continue
    dt = pd.read_csv(ft); da = pd.read_csv(fa); dp = pd.read_csv(fp_prev)
    area_col = [c for c in da.columns if 'Area' in c][0]
    # Cost of cereals→sheep transitions
    aus_t = dt[(dt['region']=='AUSTRALIA') & (dt['From-land-use']=='Winter cereals') &
               (dt['To-land-use']=='Sheep - modified land') & (dt['Type']=='ALL')]
    cost = aus_t['Cost ($)'].sum()
    # Area change: cereals area decrease = proxy for area transitioned (rough)
    aus_a = da[(da['region']=='AUSTRALIA') & (da['Water_supply']=='ALL') & (da['Land-use']=='Winter cereals')]
    aus_p = dp[(dp['region']=='AUSTRALIA') & (dp['Water_supply']=='ALL') & (dp['Land-use']=='Winter cereals')]
    area_now = aus_a[area_col].sum(); area_prev = aus_p[area_col].sum()
    delta = area_prev - area_now  # ha lost from cereals
    cost_per_ha = cost / delta if delta > 1e4 else 0
    print(f'  {yr}: cereals→sheep cost={cost/1e9:.3f}B  Δcereals={delta/1e6:.3f}Mha  cost/ha={cost_per_ha:.0f}$/ha')
