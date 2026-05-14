"""
Check base transition cost ($/ha) from transition_cost_ag2ag files,
and find what drives the 2022 spike.
"""
import pandas as pd, numpy as np, os

BASE = r"F:\Users\s222552331\Work\LUTO2_XH\luto-2.0\output\20260512_Paper3_test\Run_1_SCN_AgS1\output\2026_05_12__21_13_13_RF15_2010-2050"

# ── Structure of transition_cost_ag2ag ───────────────────────────────────────
f22 = os.path.join(BASE, "out_2022", "transition_cost_ag2ag_2022.csv")
df22 = pd.read_csv(f22)
print("Columns:", df22.columns.tolist())
print("Australia/ALL/ALL rows:")
aus = df22.query("region == 'AUSTRALIA' and `From-land-use` == 'ALL' and `To-land-use` == 'ALL'")
print(aus.to_string(index=False))

# ── Top AUSTRALIA from→to transitions by cost for 2022 ───────────────────────
print("\n=== 2022 Top AUSTRALIA To→ transitions by cost (Establishment cost) ===")
aus_detail = df22.query("region == 'AUSTRALIA' and `From-land-use` != 'ALL' and `To-land-use` != 'ALL' and Type == 'Establishment cost'")
top = aus_detail.nlargest(10, "Cost ($)")
print(top[["From-land-use", "To-land-use", "Cost ($)"]].to_string(index=False))

# ── Compare Beef→Sheep: transition_cost_ag2ag for 2019–2024 ──────────────────
print("\n=== Beef→Sheep transition cost in AUSTRALIA (Establishment) ===")
print("Year  | Total(B AUD) | note")
for yr in [2019, 2020, 2021, 2022, 2023, 2024, 2025]:
    f = os.path.join(BASE, f"out_{yr}", f"transition_cost_ag2ag_{yr}.csv")
    if not os.path.exists(f): continue
    df = pd.read_csv(f)
    row = df.query(
        "region == 'AUSTRALIA' and `From-land-use` == 'Beef - modified land'"
        " and `To-land-use` == 'Sheep - modified land' and Type == 'Establishment cost'"
    )
    total = row["Cost ($)"].sum() if len(row) else 0
    print(f"  {yr}: {total/1e9:.3f} B AUD  (n_rows={len(row)})")

# ── Cross-check: economics vs transition_cost (multiplier effect) ─────────────
print("\n=== Cross: economics vs raw transition cost for Beef→Sheep in 2022 ===")
# Raw
ft = os.path.join(BASE, "out_2022", "transition_cost_ag2ag_2022.csv")
dt = pd.read_csv(ft)
raw_beef_sheep = dt.query(
    "region == 'AUSTRALIA' and `From-land-use` == 'Beef - modified land'"
    " and `To-land-use` == 'Sheep - modified land' and Type == 'Establishment cost'"
)["Cost ($)"].sum()

# Economics (with multipliers)
fe = os.path.join(BASE, "out_2022", "economics_ag_transition_ag2ag_2022.csv")
de = pd.read_csv(fe)
val = [c for c in de.columns if "Value" in c][0]
econ_beef_sheep = de.query(
    "region == 'AUSTRALIA' and `To_Land-use` == 'Sheep - modified land'"
    " and Type == 'Establishment cost'"
)[val].sum()
print(f"  2022 raw transition (Beef→Sheep, AUS): {raw_beef_sheep/1e9:.3f} B AUD")
print(f"  2022 economics (→Sheep, AUS, est.cost): {econ_beef_sheep/1e9:.3f} B AUD")
if raw_beef_sheep > 0:
    print(f"  implied multiplier: {econ_beef_sheep/raw_beef_sheep:.3f}x")

# ── Biggest contributor to 2022 spike: look at ALL from→to transitions ─────────
print("\n=== 2022 vs 2021: AUSTRALIA 'Sheep - modified land' destination, top sources ===")
for yr in [2021, 2022]:
    f = os.path.join(BASE, f"out_{yr}", f"transition_cost_ag2ag_{yr}.csv")
    df = pd.read_csv(f)
    sheep_to = df.query(
        "region == 'AUSTRALIA' and `To-land-use` == 'Sheep - modified land'"
        " and `From-land-use` != 'ALL' and Type == 'Establishment cost'"
    ).sort_values("Cost ($)", ascending=False)
    print(f"\n  {yr} → Sheep (AUSTRALIA, top 5):")
    print(sheep_to.head(5)[["From-land-use", "Cost ($)"]].to_string(index=False))
    print(f"  TOTAL: {sheep_to['Cost ($)'].sum()/1e9:.3f} B AUD")
