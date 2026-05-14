import pandas as pd, os

BASE = r"F:\Users\s222552331\Work\LUTO2_XH\luto-2.0\output\20260512_Paper3_test\Run_1_SCN_AgS1\output\2026_05_12__21_13_13_RF15_2010-2050"

print("=== AgS1: Ag-to-Ag Transition Cost by Year ===")
for yr in range(2011, 2051):
    f = os.path.join(BASE, f"out_{yr}", f"cost_transitions_{yr}.csv")
    if not os.path.exists(f):
        continue
    df = pd.read_csv(f)
    # Keep only ag-to-ag (exclude non-ag)
    if "Cost Type" in df.columns:
        ag2ag = df[df["Cost Type"].str.contains("Ag", na=False)]
    elif "Type" in df.columns:
        ag2ag = df
    else:
        ag2ag = df
    total = df["Value (AUD)"].sum() if "Value (AUD)" in df.columns else df.iloc[:, 1].sum()
    if abs(total) > 1e6:
        print(f"  {yr}: {total/1e9:.3f} B AUD")

# Also check what columns are there
f2022 = os.path.join(BASE, "out_2022", "cost_transitions_2022.csv")
if os.path.exists(f2022):
    df = pd.read_csv(f2022)
    print("\n=== 2022 cost_transitions columns ===")
    print(df.columns.tolist())
    print(df.head(5))
else:
    # Find transition-related files in 2022
    d2022 = os.path.join(BASE, "out_2022")
    files = os.listdir(d2022)
    trans_files = [f for f in files if "trans" in f.lower() or "switch" in f.lower()]
    print(f"Transition-related files in 2022: {trans_files}")
    # Check economics summary
    econ_files = [f for f in files if "econ" in f.lower()]
    print(f"Economics files: {econ_files}")
