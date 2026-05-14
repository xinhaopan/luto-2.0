import pandas as pd, os

BASE = r"F:\Users\s222552331\Work\LUTO2_XH\luto-2.0\output\20260512_Paper3_test\Run_1_SCN_AgS1\output\2026_05_12__21_13_13_RF15_2010-2050"

print("=== AgS1: Ag-to-Ag Transition Cost by Year (from economics_ag_transition_ag2ag) ===")
rows = []
for yr in range(2011, 2051):
    f = os.path.join(BASE, f"out_{yr}", f"economics_ag_transition_ag2ag_{yr}.csv")
    if not os.path.exists(f):
        continue
    df = pd.read_csv(f)
    if "Value (AUD)" in df.columns:
        total = df[df["Land-use"] == "ALL"]["Value (AUD)"].sum() if "Land-use" in df.columns else df["Value (AUD)"].sum()
    else:
        total = df.select_dtypes("number").sum().sum()
    rows.append((yr, total))

for yr, v in rows:
    bar = "#" * int(abs(v) / max(abs(r[1]) for r in rows) * 40)
    print(f"  {yr}: {v/1e9:8.3f} B AUD  {bar}")

# Check 2022 detail
print("\n=== 2022 ag2ag transition detail ===")
f = os.path.join(BASE, "out_2022", "economics_ag_transition_ag2ag_2022.csv")
df = pd.read_csv(f)
print("Columns:", df.columns.tolist())
print(df.head(3))
print("\n Top from-land-use transitions in 2022:")
if "Land-use" in df.columns and "Value (AUD)" in df.columns:
    top = df[df["Land-use"] != "ALL"].nlargest(5, "Value (AUD)")
    print(top[["Land-use", "Value (AUD)"]].to_string())
