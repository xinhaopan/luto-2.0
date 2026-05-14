"""
Check FLC and Area cost multipliers around 2022 for AgS1,
and verify per-switch transition cost spike.
"""
import pandas as pd, numpy as np, os

INPUT = r"F:\Users\s222552331\Work\LUTO2_XH\luto-2.0\input"
BASE  = r"F:\Users\s222552331\Work\LUTO2_XH\luto-2.0\output\20260512_Paper3_test\Run_1_SCN_AgS1\output\2026_05_12__21_13_13_RF15_2010-2050"

# ── Per-switch cost ───────────────────────────────────────────────────────────
print("=== Cost per switch (B AUD total / n_switches) ===")
for yr in range(2015, 2030):
    ft = os.path.join(BASE, f"out_{yr}", f"economics_ag_transition_ag2ag_{yr}.csv")
    fs = os.path.join(BASE, f"out_{yr}", f"switches-lumap_{yr}.csv")
    if not (os.path.exists(ft) and os.path.exists(fs)):
        continue
    dt = pd.read_csv(ft)
    val = [c for c in dt.columns if "Value" in c][0]
    row = dt.query("region == 'AUSTRALIA' and `To_Land-use` == 'ALL' and Water_supply == 'ALL' and Type == 'ALL'")
    total = row[val].values[0] if len(row) else 0
    n_sw = len(pd.read_csv(fs))
    print(f"  {yr}: {total/1e9:7.2f} B  n={n_sw:5d}  per_switch={total/n_sw/1e6:.2f} M AUD")

# ── FLC Multipliers around 2022 ───────────────────────────────────────────────
print("\n=== FLC Cost Multipliers (AgS1 sheet or default) ===")
xl = pd.ExcelFile(os.path.join(INPUT, "FLC_cost_multipliers.xlsx"))
print("  Sheets:", xl.sheet_names)
# AgS1 uses 'maintain_historical' or might map to a specific sheet
# Try each sheet, show years 2019-2025
for sheet in xl.sheet_names:
    if sheet == "ReadMe":
        continue
    df = xl.parse(sheet)
    print(f"\n  Sheet: {sheet}")
    print(f"    Cols: {df.columns.tolist()[:6]}, shape={df.shape}")
    yr_col = [c for c in df.columns if "year" in c.lower() or c == "Year"][0] if any("year" in c.lower() or c=="Year" for c in df.columns) else None
    if yr_col:
        sub = df[(df[yr_col] >= 2019) & (df[yr_col] <= 2025)]
        print(sub.to_string(max_cols=6, index=False))
    else:
        print("    (no year column, first 5 rows:)")
        print(df.head(5).to_string())

# ── Area Cost Multipliers ─────────────────────────────────────────────────────
print("\n=== Area Cost Multipliers (AgS1-relevant sheet) ===")
xl2 = pd.ExcelFile(os.path.join(INPUT, "Area_cost.xlsx"))
for sheet in xl2.sheet_names:
    if sheet == "Readme":
        continue
    df = xl2.parse(sheet)
    print(f"\n  Sheet: {sheet}")
    print(f"    Cols: {df.columns.tolist()[:6]}, shape={df.shape}")
    yr_col = [c for c in df.columns if "year" in c.lower() or c == "Year"][0] if any("year" in c.lower() or c=="Year" for c in df.columns) else None
    if yr_col:
        sub = df[(df[yr_col] >= 2019) & (df[yr_col] <= 2025)]
        print(sub.to_string(max_cols=6, index=False))
    break  # Just check the first data sheet for structure

# ── Check which FLC/Area sheet is loaded for AgS1 ─────────────────────────────
print("\n=== settings.py: AG2050 cost multiplier mapping ===")
import sys; sys.path.insert(0, r"F:\Users\s222552331\Work\LUTO2_XH\luto-2.0")
import luto.settings as s
for attr in dir(s):
    if "FLC" in attr or "AREA_COST" in attr or "AG2050" in attr:
        v = getattr(s, attr)
        if not callable(v):
            print(f"  {attr} = {v}")
