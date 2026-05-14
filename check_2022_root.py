"""
Find the root cause of the RAW transition cost jumping 5x from 2021 to 2022.
The raw transition cost already shows 5.6→29.4 B before economics multipliers.
This must be due to: (a) more switches, (b) different unit costs in 2022,
                     or (c) a data/model artifact.

Approach:
  1. Check Establishment cost multiplier and Transitions cost multiplier in cost_multipliers.xlsx
  2. Compare total area switching TO Sheep: 2021 vs 2022 from lumap
  3. Check the net transition matrix (lumap 2021 vs 2022)
  4. Check if water license costs spike in 2022
"""
import pandas as pd, numpy as np, os, sys

BASE  = r"F:\Users\s222552331\Work\LUTO2_XH\luto-2.0\output\20260512_Paper3_test\Run_1_SCN_AgS1\output\2026_05_12__21_13_13_RF15_2010-2050"
INPUT = r"F:\Users\s222552331\Work\LUTO2_XH\luto-2.0\input"

# ── 1. Cost multipliers: Establishment + Transitions sheets ───────────────────
print("=== 1. cost_multipliers.xlsx - Establishment cost multiplier ===")
xl = pd.ExcelFile(os.path.join(INPUT, "cost_multipliers.xlsx"))
ec = xl.parse("Establishment cost multiplier")
yr_col = [c for c in ec.columns if "year" in c.lower() or c == "Year"][0] if any("year" in c.lower() or c=="Year" for c in ec.columns) else ec.columns[0]
print(f"  Columns (first 5): {ec.columns.tolist()[:5]}, shape={ec.shape}")
sub = ec[(ec[yr_col]>=2019)&(ec[yr_col]<=2025)]
print(sub.iloc[:,:5].to_string(index=False))

print("\n=== Transitions cost multiplier ===")
tc_mul = xl.parse("Transitions cost multiplier")
yr_col2 = [c for c in tc_mul.columns if "year" in c.lower() or c == "Year"][0] if any("year" in c.lower() or c=="Year" for c in tc_mul.columns) else tc_mul.columns[0]
print(f"  Columns (first 5): {tc_mul.columns.tolist()[:5]}, shape={tc_mul.shape}")
sub2 = tc_mul[(tc_mul[yr_col2]>=2019)&(tc_mul[yr_col2]<=2025)]
print(sub2.iloc[:,:5].to_string(index=False))

# ── 2. lumap area: how many ha are allocated to each LU in 2021 vs 2022 ────────
print("\n=== 2. Land-use area comparison: 2021 vs 2022 (AUSTRALIA, ALL water) ===")
for yr in [2020, 2021, 2022, 2023]:
    f = os.path.join(BASE, f"out_{yr}")
    area_f = os.path.join(f, f"area_{yr}.csv")
    if not os.path.exists(area_f):
        # find area file
        area_files = [x for x in os.listdir(f) if "area" in x.lower() and x.endswith(".csv")]
        area_f = os.path.join(f, area_files[0]) if area_files else None
    if area_f and os.path.exists(area_f):
        da = pd.read_csv(area_f)
        val = [c for c in da.columns if "area" in c.lower() or "ha" in c.lower() or "Value" in c][0]
        if "region" in da.columns and "Land-use" in da.columns:
            aus = da.query("region == 'AUSTRALIA'")
        else:
            aus = da
        if "Water_supply" in aus.columns:
            aus = aus.query("Water_supply == 'ALL'") if "Water_supply" in aus.columns else aus
        for lu in ["Beef - modified land", "Sheep - modified land", "Unallocated - modified land", "Winter cereals"]:
            row = aus[aus["Land-use"]==lu] if "Land-use" in aus.columns else pd.DataFrame()
            v = row[val].sum() if len(row) else 0
            print(f"  {yr} {lu:35s}: {v/1e6:.2f} Mha")
        print()

# ── 3. Transition cost: ALL types that drive the spike ────────────────────────
print("=== 3. RAW transition cost breakdown by TYPE (AUSTRALIA, ALL, 2021 vs 2022) ===")
for yr in [2021, 2022]:
    f = os.path.join(BASE, f"out_{yr}", f"transition_cost_ag2ag_{yr}.csv")
    df = pd.read_csv(f)
    aus_all = df.query("region == 'AUSTRALIA' and `From-land-use` == 'ALL' and `To-land-use` == 'ALL'")
    print(f"\n  {yr}: AUSTRALIA total by Type:")
    print(aus_all[["Type", "Cost ($)"]].to_string(index=False))

# ── 4. Detailed from→to matrix 2021 vs 2022 ────────────────────────────────
print("\n=== 4. ALL RAW Ag-to-Ag Transition Costs 2021 vs 2022 (AUSTRALIA) ===")
for yr in [2021, 2022]:
    f = os.path.join(BASE, f"out_{yr}", f"transition_cost_ag2ag_{yr}.csv")
    df = pd.read_csv(f)
    sub = df.query("region == 'AUSTRALIA' and `From-land-use` != 'ALL' and `To-land-use` != 'ALL' and Type == 'Establishment cost'")
    sub = sub.sort_values("Cost ($)", ascending=False).head(8)
    print(f"\n  {yr} top transitions (AUSTRALIA, Establishment):")
    print(sub[["From-land-use", "To-land-use", "Cost ($)"]].to_string(index=False))

# ── 5. FLC as applied in code ─────────────────────────────────────────────────
print("\n=== 5. How FLC multiplier is applied to transition costs in data.py ===")
sys.path.insert(0, r"F:\Users\s222552331\Work\LUTO2_XH\luto-2.0")
import luto.settings as settings
print(f"  AG2050_MODE={settings.AG2050_MODE}")
print(f"  AG2050_FLC_MAP={settings.AG2050_FLC_MAP}")
print(f"  AG2050_AC_MAP={settings.AG2050_AC_MAP}")
