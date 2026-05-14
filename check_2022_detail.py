"""
Deep-dive: what drives large transitions specifically in 2022 for AgS1.
Steps:
  A. FLC_multiplier_high full series (show year-over-year change)
  B. Area_cost high full series
  C. Profitability of Beef vs Sheep 2021/2022/2023
  D. GHG per hectare of Beef vs Sheep 2021/2022/2023
  E. Transition cost base (what's the transition matrix look like?)
  F. Check if 2022 is special in other input files
"""
import pandas as pd, numpy as np, os, sys

INPUT = r"F:\Users\s222552331\Work\LUTO2_XH\luto-2.0\input"
BASE  = r"F:\Users\s222552331\Work\LUTO2_XH\luto-2.0\output\20260512_Paper3_test\Run_1_SCN_AgS1\output\2026_05_12__21_13_13_RF15_2010-2050"

# ── A. FLC_multiplier_high: year-on-year change ───────────────────────────────
print("=== A. FLC_multiplier_high (Beef column, 2013-2030) ===")
xl = pd.ExcelFile(os.path.join(INPUT, "FLC_cost_multipliers.xlsx"))
flc = xl.parse("FLC_multiplier_high")
flc_beef = flc[["Year", "Beef"]].query("Year >= 2013 and Year <= 2030")
flc_beef["YoY_change"] = flc_beef["Beef"].pct_change().round(4)
print(flc_beef.to_string(index=False))

# ── B. Area_cost high: year-on-year change (first few cols) ──────────────────
print("\n=== B. Area_cost 'high' sheet (first 3 land-use columns, 2013-2030) ===")
xl2 = pd.ExcelFile(os.path.join(INPUT, "Area_cost.xlsx"))
ac = xl2.parse("high")
# Find year column
yr_col = ac.columns[0]
ac_sub = ac[(ac[yr_col] >= 2013) & (ac[yr_col] <= 2030)].copy()
# Show first 3 data cols
print(ac_sub[[yr_col] + list(ac_sub.columns[1:4])].to_string(index=False))

# ── C. Revenue comparison: Beef vs Sheep 2021/2022/2023 ──────────────────────
print("\n=== C. Revenue per land use (AUSTRALIA, ALL water) 2021/2022/2023 ===")
for yr in [2020, 2021, 2022, 2023, 2024]:
    f = os.path.join(BASE, f"out_{yr}", f"economics_ag_revenue_{yr}.csv")
    if not os.path.exists(f): continue
    df = pd.read_csv(f)
    val = [c for c in df.columns if "Value" in c][0]
    aus = df.query("region == 'AUSTRALIA' and Water_supply == 'ALL'") if "Water_supply" in df.columns else df
    beef = aus[aus["Land-use"] == "Beef - modified land"][val].sum() if "Land-use" in aus.columns else 0
    sheep = aus[aus["Land-use"] == "Sheep - modified land"][val].sum() if "Land-use" in aus.columns else 0
    print(f"  {yr}: Beef-mod={beef/1e9:.2f}B  Sheep-mod={sheep/1e9:.2f}B  ratio={sheep/max(beef,1):.2f}")

# ── D. Cost comparison: Beef vs Sheep ────────────────────────────────────────
print("\n=== D. Cost per land use (AUSTRALIA, ALL water) 2021/2022/2023 ===")
for yr in [2020, 2021, 2022, 2023, 2024]:
    f = os.path.join(BASE, f"out_{yr}", f"economics_ag_cost_{yr}.csv")
    if not os.path.exists(f): continue
    df = pd.read_csv(f)
    val = [c for c in df.columns if "Value" in c][0]
    aus = df.query("region == 'AUSTRALIA' and Water_supply == 'ALL'") if "Water_supply" in df.columns else df
    beef = aus[aus["Land-use"] == "Beef - modified land"][val].sum() if "Land-use" in aus.columns else 0
    sheep = aus[aus["Land-use"] == "Sheep - modified land"][val].sum() if "Land-use" in aus.columns else 0
    print(f"  {yr}: Beef-mod={beef/1e9:.2f}B  Sheep-mod={sheep/1e9:.2f}B")

# ── E. Transition cost matrix: Beef→Sheep and Sheep→Beef ─────────────────────
print("\n=== E. Transition cost: Beef-mod → Sheep-mod (per hectare, from raw input) ===")
sys.path.insert(0, r"F:\Users\s222552331\Work\LUTO2_XH\luto-2.0")
import luto.settings as settings
settings.AG2050_MODE = True
settings.AG2050_SCENARIO = "AgS1"
# Check if transition cost file exists
trans_files = [f for f in os.listdir(INPUT) if "transition" in f.lower() and "ag" in f.lower()]
print(f"  Transition cost input files: {trans_files[:5]}")

# ── F. Year-by-year: what % of cells switched Beef→Sheep ─────────────────────
print("\n=== F. Beef→Sheep switch count and area by year ===")
for yr in range(2015, 2030):
    f = os.path.join(BASE, f"out_{yr}", f"switches-lumap_{yr}.csv")
    if not os.path.exists(f): continue
    sw = pd.read_csv(f)
    b2s = sw[(sw["From-land-use"] == "Beef - modified land") & (sw["To-land-use"] == "Sheep - modified land")]
    if "Area (ha)" in sw.columns:
        area = b2s["Area (ha)"].sum()
    else:
        area = len(b2s)
    n = len(b2s)
    print(f"  {yr}: Beef→Sheep n={n:4d}  area={area/1e6:.2f} Mha")
