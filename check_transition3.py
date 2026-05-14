import pandas as pd, os

BASE = r"F:\Users\s222552331\Work\LUTO2_XH\luto-2.0\output\20260512_Paper3_test\Run_1_SCN_AgS1\output\2026_05_12__21_13_13_RF15_2010-2050"

# Find the correct value column
f = os.path.join(BASE, "out_2022", "economics_ag_transition_ag2ag_2022.csv")
df = pd.read_csv(f)
val_col = [c for c in df.columns if "Value" in c or "value" in c][0]
print(f"Value column: {val_col}")

# National ALL/ALL/ALL row
aus_all = df.query("region == 'AUSTRALIA' and `To_Land-use` == 'ALL' and Water_supply == 'ALL' and Type == 'ALL'")
print(f"2022 AUSTRALIA ALL total: {aus_all[val_col].values[0]/1e9:.2f} B AUD")

# Annual transition totals (national ALL/ALL/ALL)
print("\n=== AgS1: Annual Ag-to-Ag Transition (AUSTRALIA, ALL) ===")
for yr in range(2011, 2051):
    f = os.path.join(BASE, f"out_{yr}", f"economics_ag_transition_ag2ag_{yr}.csv")
    if not os.path.exists(f):
        continue
    df = pd.read_csv(f)
    val_col = [c for c in df.columns if "Value" in c or "value" in c][0]
    row = df.query("region == 'AUSTRALIA' and `To_Land-use` == 'ALL' and Water_supply == 'ALL' and Type == 'ALL'")
    if len(row):
        v = row[val_col].values[0]
        print(f"  {yr}: {v/1e9:8.2f} B AUD")

# Check what happens in 2022 specifically - top transitions
print("\n=== 2022: Top transitions (AUSTRALIA) ===")
f = os.path.join(BASE, "out_2022", "economics_ag_transition_ag2ag_2022.csv")
df = pd.read_csv(f)
val_col = [c for c in df.columns if "Value" in c or "value" in c][0]
top = df.query("region == 'AUSTRALIA' and `To_Land-use` != 'ALL' and Water_supply != 'ALL' and Type == 'Establishment cost'") \
       .nlargest(10, val_col)[["To_Land-use", "Water_supply", val_col]]
print(top.to_string())

# Check switches in 2022 (land use change map)
f_sw = os.path.join(BASE, "out_2022", "switches-lumap_2022.csv")
if os.path.exists(f_sw):
    sw = pd.read_csv(f_sw)
    print(f"\n=== 2022 Land Use Switches ===")
    print("Columns:", sw.columns.tolist())
    print(f"Total switches: {len(sw)}")
