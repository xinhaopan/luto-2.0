"""Dig into the ag2ag transition cost file structure to find the spike cause."""
import pandas as pd, os

BASE = r'F:/Users/s222552331/Work/LUTO2_XH/luto-2.0/output/20260513_Paper3_aquila/Run_1_SCN_AgS1/output/2026_05_13__19_24_46_RF5_2010-2050'

# ── 1. Peek at 2023 ag2ag file ───────────────────────────────────────────────
yr = 2023
fp = os.path.join(BASE, f'out_{yr}', f'transition_cost_ag2ag_{yr}.csv')
d = pd.read_csv(fp)
print('=== ag2ag 2023 columns:', d.columns.tolist())
print(f'Total rows: {len(d)}')
print('\nUnique From-land-use values (head 5):', d['From-land-use'].unique()[:8].tolist())
print('Unique To-land-use values (head 5):', d['To-land-use'].unique()[:8].tolist())
print('Unique Type values:', d['Type'].unique().tolist())

# ── 2. Check AUSTRALIA + filter out ALL rows ──────────────────────────────────
aus = d[d['region'] == 'AUSTRALIA']
print(f'\nAUSTRALIA rows: {len(aus)}')
print('Type distribution:')
print(aus.groupby('Type')['Cost ($)'].sum().sort_values(ascending=False).apply(lambda x: f'{x/1e9:.2f}B'))

# ── 3. Non-ALL rows only (no aggregates) ─────────────────────────────────────
non_all = aus[(aus['From-land-use'] != 'ALL') & (aus['To-land-use'] != 'ALL')]
print(f'\nNon-ALL rows: {len(non_all)}')
print('Type distribution (non-ALL):')
print(non_all.groupby('Type')['Cost ($)'].sum().sort_values(ascending=False).apply(lambda x: f'{x/1e9:.2f}B'))

# ── 4. Correct total: Type=='ALL' and From/To != 'ALL' ───────────────────────
# (Each transition pair summarised in one ALL row)
type_all = non_all[non_all['Type'] == 'ALL']
print(f'\nCorrect total (Type=ALL, no agg land-use): {type_all["Cost ($)"].sum()/1e9:.2f} B')
print('\nTop transitions (Type=ALL):')
grp = type_all.groupby(['From-land-use', 'To-land-use'])['Cost ($)'].sum().sort_values(ascending=False)
for idx, val in grp.head(20).items():
    if abs(val) > 3e8:
        print(f'  {idx[0]:<40} → {idx[1]:<40}: {val/1e9:.2f} B')

# ── 5. Year-by-year correct totals ───────────────────────────────────────────
print('\n=== Correct ag2ag by year ===')
for yr in range(2011, 2051):
    fp = os.path.join(BASE, f'out_{yr}', f'transition_cost_ag2ag_{yr}.csv')
    if not os.path.exists(fp): continue
    d = pd.read_csv(fp)
    aus = d[d['region'] == 'AUSTRALIA']
    non_all = aus[(aus['From-land-use'] != 'ALL') & (aus['To-land-use'] != 'ALL') & (aus['Type'] == 'ALL')]
    total = non_all['Cost ($)'].sum()
    print(f'  {yr}: {total/1e9:.2f} B')
