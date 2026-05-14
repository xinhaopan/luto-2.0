"""Investigate why 2023 has high ag2ag transition costs in the new smooth/flat FLC runs."""
import pandas as pd, os, sys

configs = [
    ('aquila_smooth', r'F:/Users/s222552331/Work/LUTO2_XH/luto-2.0/output/20260513_Paper3_aquila/Run_1_SCN_AgS1/output/2026_05_13__19_24_46_RF5_2010-2050'),
    ('test_flat',     r'F:/Users/s222552331/Work/LUTO2_XH/luto-2.0/output/20260513_Paper3_test/Run_1_SCN_AgS1/output/2026_05_13__22_14_05_RF5_2010-2050'),
]

print('=== Ag2Ag Transition Cost AUSTRALIA ALL->ALL (B AUD) ===')
print(f'{"Year":<6} | {"aquila_smooth":>13} | {"test_flat":>9}')
for yr in range(2010, 2051):
    vals = []
    for label, base in configs:
        f = os.path.join(base, f'out_{yr}', f'transition_cost_ag2ag_{yr}.csv')
        if not os.path.exists(f):
            vals.append(None)
            continue
        df = pd.read_csv(f)
        aus_all = df.query("region == 'AUSTRALIA' and `From-land-use` == 'ALL' and `To-land-use` == 'ALL'")
        total = aus_all['Cost ($)'].sum()
        vals.append(total / 1e9)
    if any(v is not None for v in vals):
        v1 = f'{vals[0]:.2f}' if vals[0] is not None else 'N/A'
        v2 = f'{vals[1]:.2f}' if vals[1] is not None else 'N/A'
        mark = ' <---' if any(v is not None and v > 10 for v in vals) else ''
        print(f'{yr:<6} | {v1:>13} | {v2:>9}{mark}')

# --- Detail the 2023 spike for test_flat ---
print('\n=== 2023 Top AUSTRALIA ag2ag transitions by cost (test_flat, Establishment) ===')
base_t = configs[1][1]
for yr in [2022, 2023, 2024]:
    f = os.path.join(base_t, f'out_{yr}', f'transition_cost_ag2ag_{yr}.csv')
    if not os.path.exists(f): continue
    df = pd.read_csv(f)
    sub = df.query("region == 'AUSTRALIA' and `From-land-use` != 'ALL' and `To-land-use` != 'ALL' and Type == 'Establishment cost'")
    sub = sub.nlargest(6, 'Cost ($)')
    print(f'\n  {yr} top transitions (Establishment):')
    print(sub[['From-land-use', 'To-land-use', 'Cost ($)']].to_string(index=False))

# --- Compare: which LU has biggest area change 2022→2023→2024 in test_flat ---
print('\n=== Area by LU AUSTRALIA (ALL water) in test_flat ===')
for yr in [2021, 2022, 2023, 2024, 2025]:
    f = os.path.join(base_t, f'out_{yr}', f'area_{yr}.csv')
    if not os.path.exists(f): continue
    da = pd.read_csv(f)
    aus = da.query("region == 'AUSTRALIA' and Water_supply == 'ALL'") if 'Water_supply' in da.columns else da.query("region == 'AUSTRALIA'")
    for lu in ['Beef - modified land', 'Sheep - modified land', 'Winter cereals', 'Unallocated - modified land']:
        row = aus[aus['Land-use'] == lu]
        v = row['Area (ha)'].sum() if 'Area (ha)' in row.columns else row.iloc[:, -1].sum()
        print(f'  {yr} {lu:35s}: {v/1e6:.2f} Mha')
    print()

# --- Check GHG constraint in test_flat ---
print('=== GHG constraint vs actual (test_flat) ===')
for yr in [2021, 2022, 2023, 2024, 2025]:
    f = os.path.join(base_t, f'out_{yr}', f'GHG_emissions_{yr}.csv')
    if not os.path.exists(f): continue
    df = pd.read_csv(f)
    limit = df.loc[df['Variable'] == 'GHG_EMISSIONS_LIMIT_TCO2e', 'Emissions (t CO2e)'].values
    actual = df.loc[df['Variable'] == 'GHG_EMISSIONS_TCO2e', 'Emissions (t CO2e)'].values
    if len(limit) and len(actual):
        slack = (actual[0] - limit[0]) / 1e6
        print(f'  {yr}: limit={limit[0]/1e6:.1f} Mt, actual={actual[0]/1e6:.1f} Mt, slack={slack:+.3f} Mt')

# --- Check FLC multiplier values for key years ---
print('\n=== FLC multiplier check in test_flat model_run_settings ===')
settings_f = os.path.join(base_t.replace('/output/2026_05_13__22_14_05_RF5_2010-2050', ''), 'output', '2026_05_13__22_14_05_RF5_2010-2050', 'model_run_settings.txt')
if os.path.exists(settings_f):
    with open(settings_f) as fh:
        for line in fh:
            if 'FLC' in line or 'TRANS_COST' in line or 'GHG_LIMIT' in line or 'DEMAND_CON' in line:
                print(' ', line.strip())
