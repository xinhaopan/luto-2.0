"""AgS1 transition cost breakdown by year and by transition type."""
import pandas as pd, os

BASE = r'F:/Users/s222552331/Work/LUTO2_XH/luto-2.0/output/20260513_Paper3_aquila/Run_1_SCN_AgS1/output/2026_05_13__19_24_46_RF5_2010-2050'

# ── 1. Year-by-year summary (ag2ag + ag2nonag + nonag2ag) ───────────────────
print('=== 1. AgS1 transition costs by year ===')
costs = {}
for yr in range(2011, 2051):
    out_dir = os.path.join(BASE, f'out_{yr}')
    if not os.path.isdir(out_dir): continue
    totals = {}
    for tag in ['ag2ag', 'ag2non_ag', 'non_ag2ag']:
        fname = f'transition_cost_{tag}_{yr}.csv'
        fp = os.path.join(out_dir, fname)
        if not os.path.exists(fp): continue
        d = pd.read_csv(fp)
        val_col = [c for c in d.columns if 'Value' in c or '$' in c][0]
        aus = d[d['region'] == 'AUSTRALIA']
        totals[tag] = {'val': aus[val_col].sum(), 'df': aus, 'val_col': val_col}
    total = sum(v['val'] for v in totals.values())
    costs[yr] = {'total': total, 'parts': totals}
    ag2ag = totals.get('ag2ag', {}).get('val', 0)
    ag2non = totals.get('ag2non_ag', {}).get('val', 0)
    non2ag = totals.get('non_ag2ag', {}).get('val', 0)
    print(f'  {yr}: ag2ag={ag2ag/1e9:.2f}B  ag2non={ag2non/1e9:.2f}B  non2ag={non2ag/1e9:.2f}B  total={total/1e9:.2f}B')

# ── 2. Spike year: drill into ag2ag transitions ──────────────────────────────
spike_yr = max(costs, key=lambda y: costs[y]['parts'].get('ag2ag', {}).get('val', 0))
print(f'\n=== 2. Spike year (ag2ag): {spike_yr} ===')
ag2ag_parts = costs[spike_yr]['parts']
for tag, info in ag2ag_parts.items():
    df = info['df']; val_col = info['val_col']
    print(f'\n  [{tag}] total={info["val"]/1e9:.2f}B  columns: {df.columns.tolist()}')
    # Try to get from/to breakdown
    from_cols = [c for c in df.columns if 'From' in c and 'land-use' in c.lower()]
    to_cols   = [c for c in df.columns if 'To' in c and 'land-use' in c.lower()]
    if from_cols and to_cols:
        grp = df.groupby(from_cols + to_cols)[val_col].sum().sort_values(ascending=False)
        for idx, val in grp.head(20).items():
            if abs(val) > 3e8:
                print(f'    {str(idx):<80}: {val/1e9:.2f} B')

# ── 3. Area changes for key land uses around spike year ──────────────────────
print(f'\n=== 3. Area changes (Mha) around {spike_yr} ===')
prev_areas = {}
for yr in range(max(2011, spike_yr-2), min(2023, spike_yr+3)):
    fa = os.path.join(BASE, f'out_{yr}', f'area_agricultural_landuse_{yr}.csv')
    if not os.path.exists(fa): continue
    da = pd.read_csv(fa)
    area_col = [c for c in da.columns if 'Area' in c][0]
    aus = da[(da['region']=='AUSTRALIA') & (da['Water_supply']=='ALL') & (da['Land-use']!='ALL')]
    cur = aus.groupby('Land-use')[area_col].sum()
    print(f'\n  {yr}:')
    for lu, area in cur.sort_values(ascending=False).items():
        if area > 5e6:
            delta = cur[lu] - prev_areas.get(lu, cur[lu])
            mark = f'  Δ{delta/1e6:+.2f}' if lu in prev_areas else ''
            print(f'    {lu:<40}: {area/1e6:.2f} Mha{mark}')
    prev_areas = cur.to_dict()

# ── 4. GHG limit and sheep demand for context ────────────────────────────────
print('\n=== 4. Model settings relevant to AgS1 ===')
sf = os.path.join(BASE, 'model_run_settings.txt')
if os.path.exists(sf):
    with open(sf) as f:
        for line in f:
            if any(k in line for k in ['GHG', 'DEMAND', 'SHEEP', 'BIODIVERSITY', 'SCENARIO']):
                print(' ', line.strip()[:120])
