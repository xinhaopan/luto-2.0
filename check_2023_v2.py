"""Find the root cause of 2023 ag2ag spike using correct file names."""
import pandas as pd, os

BASE = r'F:/Users/s222552331/Work/LUTO2_XH/luto-2.0/output/20260513_Paper3_test/Run_1_SCN_AgS1/output/2026_05_13__22_14_05_RF5_2010-2050'

# ── 1. Area by LU using correct file name ─────────────────────────────────────
print('=== 1. Ag LU Area AUSTRALIA ALL water (Mha) ===')
for yr in [2020, 2021, 2022, 2023, 2024]:
    f = os.path.join(BASE, f'out_{yr}', f'area_agricultural_landuse_{yr}.csv')
    if not os.path.exists(f): continue
    da = pd.read_csv(f)
    aus = da.query("region == 'AUSTRALIA' and Water_supply == 'ALL'")
    area_col = [c for c in da.columns if 'Area' in c][0]
    print(f'  {yr}:')
    for lu in ['Beef - modified land', 'Sheep - modified land', 'Winter cereals',
               'Unallocated - modified land', 'Dairy - modified land']:
        row = aus[aus['Land-use'] == lu]
        v = row[area_col].sum() if len(row) else 0
        print(f'    {lu:35s}: {v/1e6:.3f} Mha')

# ── 2. Quantity production vs demand ─────────────────────────────────────────
print('\n=== 2. Quantity comparison (demand met?) for sheep/beef 2020-2024 ===')
for yr in [2020, 2021, 2022, 2023, 2024]:
    f = os.path.join(BASE, f'out_{yr}', f'quantity_comparison_{yr}.csv')
    if not os.path.exists(f): continue
    dq = pd.read_csv(f)
    aus = dq.query("region == 'AUSTRALIA'") if 'region' in dq.columns else dq
    print(f'\n  {yr}:')
    print(aus.to_string(index=False))

# ── 3. Economics profit for key LU 2020-2024 ─────────────────────────────────
print('\n=== 3. Economics profit AUSTRALIA (B AUD/yr) ===')
for yr in [2020, 2021, 2022, 2023, 2024]:
    f = os.path.join(BASE, f'out_{yr}', f'economics_ag_profit_{yr}.csv')
    if not os.path.exists(f): continue
    de = pd.read_csv(f)
    aus = de.query("region == 'AUSTRALIA' and Water_supply == 'ALL'")
    val_col = [c for c in de.columns if 'Value' in c or '$' in c][0]
    print(f'  {yr}:')
    for lu in ['Beef - modified land', 'Sheep - modified land', 'Winter cereals',
               'Unallocated - modified land', 'Dairy - modified land']:
        row = aus[aus['Land-use'] == lu]
        v = row[val_col].sum() if len(row) else 0
        print(f'    {lu:35s}: {v/1e9:.2f} B')

# ── 4. Quantity production kt for sheep 2020-2025 ────────────────────────────
print('\n=== 4. Sheep/beef production (kt) 2020-2025 ===')
for yr in [2020, 2021, 2022, 2023, 2024]:
    f = os.path.join(BASE, f'out_{yr}', f'quantity_production_kt_separate_{yr}.csv')
    if not os.path.exists(f): continue
    dq = pd.read_csv(f)
    aus = dq.query("region == 'AUSTRALIA' and Water_supply == 'ALL'") if 'Water_supply' in dq.columns else dq
    val_col = [c for c in dq.columns if 'qty' in c.lower() or 'prod' in c.lower() or 'kt' in c.lower() or 'Value' in c][0]
    for lu in ['Beef - modified land', 'Sheep - modified land']:
        row = aus[aus['Land-use'] == lu] if 'Land-use' in aus.columns else pd.DataFrame()
        v = row[val_col].sum() if len(row) else 0
        print(f'  {yr} {lu:35s}: {v:.1f} kt')

# ── 5. Check switches-lumap 2022 vs 2023 (area that switched) ────────────────
print('\n=== 5. Area switching (Mha) from lumap switches 2022 vs 2023 ===')
for yr in [2020, 2021, 2022, 2023, 2024]:
    f = os.path.join(BASE, f'out_{yr}', f'switches-lumap_{yr}.csv')
    if not os.path.exists(f): continue
    ds = pd.read_csv(f)
    print(f'\n  {yr} switches (all columns): {ds.columns.tolist()}')
    if 'Area (ha)' in ds.columns or 'area' in ' '.join(ds.columns).lower():
        area_col = [c for c in ds.columns if 'area' in c.lower()][0]
        total = ds[area_col].sum()
        to_sheep = ds[ds.apply(lambda r: 'Sheep' in str(r.values), axis=1)][area_col].sum() if 'Sheep' in ds.to_string() else 0
        print(f'    Total area switching: {total/1e6:.2f} Mha')
        print(f'    Of which to sheep: {to_sheep/1e6:.2f} Mha')

# ── 6. Check crosstab 2022 vs 2023 ───────────────────────────────────────────
print('\n=== 6. Land use transitions matrix (cells): 2022 vs 2023 ===')
for yr in [2022, 2023]:
    f = os.path.join(BASE, f'out_{yr}', f'crosstab-lumap_{yr}.csv')
    if not os.path.exists(f): continue
    dc = pd.read_csv(f, index_col=0)
    print(f'\n  {yr} crosstab shape: {dc.shape}')
    # Top 5 transitions by cell count
    dc_flat = dc.stack().reset_index()
    dc_flat.columns = ['From', 'To', 'Cells']
    dc_flat = dc_flat.query("Cells > 0 and From != To").nlargest(8, 'Cells')
    print(dc_flat.to_string(index=False))
