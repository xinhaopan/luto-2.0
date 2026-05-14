"""
Investigate why ag-to-ag transition costs spike in 2022 for AgS1.
Checks: demand, FLC multipliers, area cost multipliers, productivity trends,
        switches-lumap, and per-year revenue/cost differences around 2022.
"""
import pandas as pd, numpy as np, os

INPUT = r"F:\Users\s222552331\Work\LUTO2_XH\luto-2.0\input"
BASE  = r"F:\Users\s222552331\Work\LUTO2_XH\luto-2.0\output\20260512_Paper3_test\Run_1_SCN_AgS1\output\2026_05_12__21_13_13_RF15_2010-2050"

# ── 1. Demand: check if target demand shifts sharply around 2022 ──────────────
print("=" * 60)
print("1. DEMAND TARGETS (All_LUTO_demand_scenarios_with_convergences.csv)")
demand = pd.read_csv(os.path.join(INPUT, "All_LUTO_demand_scenarios_with_convergences.csv"))
print("  Columns:", demand.columns.tolist()[:8])
# Filter to AgS1-relevant scenario
scens = demand.columns.tolist()
yr_col = [c for c in demand.columns if "year" in c.lower() or c == "Year"][0]
print(f"  Year col: {yr_col},  Shape: {demand.shape}")
# Show rows around 2020-2025
yr_slice = demand[(demand[yr_col] >= 2019) & (demand[yr_col] <= 2025)]
print(yr_slice.to_string(max_cols=8, max_rows=10))

# ── 2. FLC cost multipliers ───────────────────────────────────────────────────
print("\n" + "=" * 60)
print("2. FLC COST MULTIPLIERS (FLC_cost_multipliers.xlsx)")
xl = pd.ExcelFile(os.path.join(INPUT, "FLC_cost_multipliers.xlsx"))
print("  Sheets:", xl.sheet_names)

# ── 3. Area cost multipliers ──────────────────────────────────────────────────
print("\n" + "=" * 60)
print("3. AREA COST MULTIPLIERS (Area_cost.xlsx)")
xl2 = pd.ExcelFile(os.path.join(INPUT, "Area_cost.xlsx"))
print("  Sheets:", xl2.sheet_names)

# ── 4. Productivity trends (settings.py AG2050_PRODUCTIVITY_TREND) ───────────
print("\n" + "=" * 60)
print("4. PRODUCTIVITY TREND (from settings.py)")
import sys; sys.path.insert(0, r"F:\Users\s222552331\Work\LUTO2_XH\luto-2.0")
import luto.settings as settings
if hasattr(settings, 'AG2050_PRODUCTIVITY_TREND_DICT'):
    pt = settings.AG2050_PRODUCTIVITY_TREND_DICT
    print("  AG2050_PRODUCTIVITY_TREND_DICT keys:", list(pt.keys()))
    # Show AgS1 values
    if 'AgS1' in pt:
        df_pt = pd.DataFrame(list(pt['AgS1'].items()), columns=['year','mult'])
        print(df_pt[(df_pt['year'] >= 2019) & (df_pt['year'] <= 2025)].to_string(index=False))
elif hasattr(settings, 'AG2050_PRODUCTIVITY_TREND'):
    print("  AG2050_PRODUCTIVITY_TREND:", settings.AG2050_PRODUCTIVITY_TREND)
else:
    print("  No AG2050_PRODUCTIVITY_TREND found in settings")

# ── 5. Revenue changes around 2022 ───────────────────────────────────────────
print("\n" + "=" * 60)
print("5. Ag Revenue changes 2021 vs 2022 vs 2023 (AUSTRALIA ALL)")
for yr in [2020, 2021, 2022, 2023, 2024]:
    f = os.path.join(BASE, f"out_{yr}", f"economics_ag_revenue_{yr}.csv")
    if not os.path.exists(f):
        continue
    df = pd.read_csv(f)
    val = [c for c in df.columns if "Value" in c or "value" in c][0]
    row = df.query("region == 'AUSTRALIA' and `Land-use` == 'ALL'") if "Land-use" in df.columns else df
    if "Water_supply" in df.columns:
        row = row.query("Water_supply == 'ALL'") if "Water_supply" in row.columns else row
    total = row[val].sum() if len(row) else df[val].sum()
    print(f"  {yr}: {total/1e9:.2f} B AUD (revenue)")

# ── 6. Ag Cost changes around 2022 ───────────────────────────────────────────
print("\n" + "=" * 60)
print("6. Ag Cost changes 2021 vs 2022 vs 2023 (AUSTRALIA ALL)")
for yr in [2020, 2021, 2022, 2023, 2024]:
    f = os.path.join(BASE, f"out_{yr}", f"economics_ag_cost_{yr}.csv")
    if not os.path.exists(f):
        continue
    df = pd.read_csv(f)
    val = [c for c in df.columns if "Value" in c or "value" in c][0]
    row = df
    if "region" in df.columns and "Land-use" in df.columns and "Water_supply" in df.columns:
        row = df.query("region == 'AUSTRALIA' and `Land-use` == 'ALL' and Water_supply == 'ALL'")
    total = row[val].sum() if len(row) else df[val].sum()
    print(f"  {yr}: {total/1e9:.2f} B AUD (cost)")

# ── 7. What land uses are switching in 2022? ──────────────────────────────────
print("\n" + "=" * 60)
print("7. Land-use switches in 2021, 2022, 2023")
for yr in [2021, 2022, 2023]:
    f = os.path.join(BASE, f"out_{yr}", f"switches-lumap_{yr}.csv")
    if not os.path.exists(f):
        continue
    df = pd.read_csv(f)
    n = len(df)
    top_from = df["From-land-use"].value_counts().head(3).to_dict()
    top_to   = df["To-land-use"].value_counts().head(3).to_dict()
    print(f"  {yr}: {n} switches | top-from: {top_from} | top-to: {top_to}")
