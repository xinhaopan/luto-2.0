"""
Find the root cause of the per-hectare transition cost spike in 2022.
Check: transition_cost_ag2ag CSV (base costs), FLC applied, and
       compare the actual $/ha for Beef→Sheep transitions 2020-2024.
"""
import pandas as pd, numpy as np, os, sys

BASE  = r"F:\Users\s222552331\Work\LUTO2_XH\luto-2.0\output\20260512_Paper3_test\Run_1_SCN_AgS1\output\2026_05_12__21_13_13_RF15_2010-2050"
INPUT = r"F:\Users\s222552331\Work\LUTO2_XH\luto-2.0\input"

# ── Check transition_cost_ag2ag (base costs, before multipliers) ──────────────
print("=== transition_cost_ag2ag_2022.csv columns ===")
f = os.path.join(BASE, "out_2022", "transition_cost_ag2ag_2022.csv")
df = pd.read_csv(f)
print("Columns:", df.columns.tolist())
print("Shape:", df.shape)
print(df.head(5).to_string())

# ── Per-hectare transition cost: Beef → Sheep, Beef → Beef (same) ─────────────
print("\n=== Per-hectare transition costs (AUSTRALIA, Dryland): Beef-mod → various ===")
for yr in [2019, 2020, 2021, 2022, 2023, 2024, 2025]:
    ft = os.path.join(BASE, f"out_{yr}", f"transition_cost_ag2ag_{yr}.csv")
    if not os.path.exists(ft):
        continue
    dt = pd.read_csv(ft)
    cols = dt.columns.tolist()
    print(f"  [{yr}] cols: {cols}")
    break  # Just check structure once

# ── Per-hectare cost from economics_ag_transition (includes multipliers) ───────
print("\n=== economics_ag_transition_ag2ag: Beef-mod → Sheep-mod $/ha by year ===")
print("    (using 'to' land use, Dryland, Type=Establishment, AUSTRALIA)")
for yr in [2019, 2020, 2021, 2022, 2023, 2024, 2025]:
    ft = os.path.join(BASE, f"out_{yr}", f"economics_ag_transition_ag2ag_{yr}.csv")
    fs = os.path.join(BASE, f"out_{yr}", f"switches-lumap_{yr}.csv")
    if not (os.path.exists(ft) and os.path.exists(fs)):
        continue
    dt = pd.read_csv(ft)
    sw = pd.read_csv(fs)
    val = [c for c in dt.columns if "Value" in c][0]

    # Sheep-mod transition cost total
    sheep_cost = dt.query("region == 'AUSTRALIA' and `To_Land-use` == 'Sheep - modified land' and Water_supply == 'Dryland' and Type == 'Establishment cost'")
    sheep_cost_total = sheep_cost[val].sum() if len(sheep_cost) else 0

    # Sheep area switched
    sw_cols = sw.columns.tolist()
    to_col = [c for c in sw_cols if "To" in c][0]
    from_col = [c for c in sw_cols if "From" in c][0]
    sheep_area = sw[sw[to_col] == "Sheep - modified land"]["Area (ha)"].sum() if "Area (ha)" in sw.columns else len(sw[sw[to_col] == "Sheep - modified land"])

    unit_cost = sheep_cost_total / max(sheep_area, 1)
    print(f"  {yr}: Sheep transition total={sheep_cost_total/1e9:.2f}B  sheep_area={sheep_area/1e6:.2f}Mha  unit={unit_cost:.0f} AUD/ha")

# ── Check FLC: how it is applied to transition costs ────────────────────────
print("\n=== FLC multiplier effect: how does it multiply transition costs? ===")
print("    Settings: AMORTISE_UPFRONT_COSTS =", end=" ")
sys.path.insert(0, r"F:\Users\s222552331\Work\LUTO2_XH\luto-2.0")
import luto.settings as settings
print(getattr(settings, 'AMORTISE_UPFRONT_COSTS', 'not found'))

# Check if there is FLC applied differently
flc = pd.ExcelFile(os.path.join(INPUT, "FLC_cost_multipliers.xlsx")).parse("FLC_multiplier_high")
print("\nFLC_high Beef values (2019-2025):")
print(flc[["Year","Beef"]].query("Year >= 2019 and Year <= 2025").to_string(index=False))

# ── Check cost_multipliers.xlsx ───────────────────────────────────────────────
print("\n=== cost_multipliers.xlsx sheets ===")
xl3 = pd.ExcelFile(os.path.join(INPUT, "cost_multipliers.xlsx"))
print("Sheets:", xl3.sheet_names)
# Check first sheet
df3 = xl3.parse(xl3.sheet_names[0])
yr_col = [c for c in df3.columns if "year" in c.lower() or c=="Year"][0] if any("year" in c.lower() or c=="Year" for c in df3.columns) else None
if yr_col:
    print("2019-2025:")
    print(df3[(df3[yr_col]>=2019)&(df3[yr_col]<=2025)].head(10).to_string(index=False))
else:
    print(df3.head(5).to_string())

# ── Check if transition BASE cost (before multipliers) is consistent ──────────
print("\n=== Base transition cost file in input/ ===")
for f in os.listdir(INPUT):
    if "transit" in f.lower():
        print(" ", f)
