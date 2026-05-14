"""Compare AgS3: old run (with FLC medium peak) vs new flat FLC run to isolate FLC effect."""
import pandas as pd, os

OLD = r'F:/Users/s222552331/Work/LUTO2_XH/luto-2.0/output/20260512_Paper3_aquila/Run_3_SCN_AgS3/output/2026_05_13__00_29_59_RF5_2010-2050'
NEW_BASE = r'F:/Users/s222552331/Work/LUTO2_XH/luto-2.0/output/20260513_Paper3_test/Run_3_SCN_AgS3/output'
new_subdirs = [d for d in os.listdir(NEW_BASE) if os.path.isdir(os.path.join(NEW_BASE, d))]
NEW = os.path.join(NEW_BASE, new_subdirs[0])
print(f'New run: {new_subdirs[0]}')

# ── 1. FLC multiplier values for medium scenario (from input) ────────────────
print('\n=== 1. FLC_multiplier_medium values (sheep) from input ===')
import sys
sys.path.insert(0, r'F:/Users/s222552331/Work/LUTO2_XH/luto-2.0')
try:
    xl = pd.ExcelFile(r'F:/Users/s222552331/Work/LUTO2_XH/luto-2.0/input/FLC_cost_multipliers.xlsx')
    flc_med = xl.parse('FLC_multiplier_medium')
    yr_col = [c for c in flc_med.columns if 'year' in str(c).lower() or c == 'Year'][0]
    sub = flc_med[(flc_med[yr_col] >= 2018) & (flc_med[yr_col] <= 2027)]
    # Show sheep-related columns
    sheep_cols = [c for c in flc_med.columns if 'sheep' in str(c).lower() or 'Sheep' in str(c)]
    if sheep_cols:
        print(sub[[yr_col] + sheep_cols[:3]].to_string(index=False))
    else:
        print(sub.iloc[:, :5].to_string(index=False))
except Exception as e:
    print(f'  Error: {e}')

# ── 2. Compare transition costs year-by-year: OLD vs NEW for AgS3 ────────────
print('\n=== 2. Ag2Ag transition cost (B AUD): OLD (FLC medium) vs NEW (flat FLC) ===')
print(f'{"Year":<6} | {"OLD FLC_med":>12} | {"NEW flat":>10}')
for yr in range(2015, 2030):
    vals = []
    for base, label in [(OLD, 'old'), (NEW, 'new')]:
        f = os.path.join(base, f'out_{yr}', f'transition_cost_ag2ag_{yr}.csv')
        if not os.path.exists(f):
            vals.append(None)
            continue
        df = pd.read_csv(f)
        aus_all = df.query("region == 'AUSTRALIA' and `From-land-use` == 'ALL' and `To-land-use` == 'ALL' and Type == 'ALL'")
        vals.append(aus_all['Cost ($)'].sum() / 1e9)
    if any(v is not None for v in vals):
        v1 = f'{vals[0]:.2f}' if vals[0] is not None else 'N/A'
        v2 = f'{vals[1]:.2f}' if vals[1] is not None else 'N/A'
        mark = ' <---' if any(v is not None and v > 5 for v in vals) else ''
        print(f'{yr:<6} | {v1:>12} | {v2:>10}{mark}')

# ── 3. Profit/ha comparison for sheep 2020-2025: OLD vs NEW ─────────────────
print('\n=== 3. Sheep profit/ha ($/ha) OLD vs NEW, AUSTRALIA ===')
for yr in [2020, 2021, 2022, 2023, 2024]:
    row_data = []
    for base, label in [(OLD, 'old'), (NEW, 'new')]:
        fp = os.path.join(base, f'out_{yr}', f'economics_ag_profit_{yr}.csv')
        fa = os.path.join(base, f'out_{yr}', f'area_agricultural_landuse_{yr}.csv')
        if not os.path.exists(fp):
            row_data.append(None)
            continue
        de = pd.read_csv(fp)
        da = pd.read_csv(fa)
        val_col = [c for c in de.columns if 'Value' in c or '$' in c][0]
        area_col = [c for c in da.columns if 'Area' in c][0]
        ep = de.query("region == 'AUSTRALIA' and Water_supply == 'ALL' and `Land-use` == 'Sheep - modified land'")
        ea = da.query("region == 'AUSTRALIA' and Water_supply == 'ALL' and `Land-use` == 'Sheep - modified land'")
        profit = ep[val_col].sum() if len(ep) else 0
        area = ea[area_col].sum() if len(ea) else 1
        row_data.append(profit / area)
    v1 = f'{row_data[0]:.0f}' if row_data[0] is not None else 'N/A'
    v2 = f'{row_data[1]:.0f}' if row_data[1] is not None else 'N/A'
    print(f'  {yr}: OLD={v1:>6} $/ha   NEW={v2:>6} $/ha')

# ── 4. Sheep area change: OLD vs NEW ────────────────────────────────────────
print('\n=== 4. Sheep area (Mha) OLD vs NEW ===')
for yr in [2020, 2021, 2022, 2023, 2024]:
    vals = []
    for base in [OLD, NEW]:
        f = os.path.join(base, f'out_{yr}', f'area_agricultural_landuse_{yr}.csv')
        if not os.path.exists(f):
            vals.append(None)
            continue
        da = pd.read_csv(f)
        area_col = [c for c in da.columns if 'Area' in c][0]
        ea = da.query("region == 'AUSTRALIA' and Water_supply == 'ALL' and `Land-use` == 'Sheep - modified land'")
        vals.append(ea[area_col].sum() / 1e6)
    v1 = f'{vals[0]:.3f}' if vals[0] is not None else 'N/A'
    v2 = f'{vals[1]:.3f}' if vals[1] is not None else 'N/A'
    print(f'  {yr}: OLD={v1} Mha   NEW={v2} Mha')
