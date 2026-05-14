import json, re, os, pandas as pd

BASE = r"F:\Users\s222552331\Work\LUTO2_XH\luto-2.0\output\20260512_Paper3_test\Run_2_SCN_AgS2\output\2026_05_12__21_13_13_RF15_2010-2050"

# --- 1. Read GHG_overview_sum.js ---
f = os.path.join(BASE, "DATA_REPORT", "data", "GHG_overview_sum.js")
with open(f, "r", encoding="utf-8") as fh:
    content = fh.read()
json_str = re.sub(r'^window\["GHG_overview_sum"\]\s*=\s*', '', content.strip().rstrip(';'))
data = json.loads(json_str)

# Use AUSTRALIA level
region = "AUSTRALIA" if "AUSTRALIA" in data else list(data.keys())[0]
print(f"=== GHG Overview Sum - Region: {region} ===")
series_by_name = {}
for s in data[region]:
    name = s["name"]
    pts = s.get("data", [])
    series_by_name[name] = pts
    print(f"  Series: {name!r:40s}  ({len(pts)} points)  sample: {pts[:3]}")

# --- 2. Compute net emission from report series ---
print("\n=== Net Emission from report vs solver limit (Mt CO2e) ===")
print(f"  {'Year':4}  {'Net(report)':12}  {'Solver Limit':12}  {'Over?':8}")

# Sum all series to get net (exclude limit-related series)
years = list(range(2010, 2051))
net_vals = {}
excl_keys = [k for k in series_by_name if 'limit' in k.lower() or 'target' in k.lower()]
incl_keys = [k for k in series_by_name if k not in excl_keys]
print(f"\nSeries included in net: {incl_keys}")
print(f"Series excluded (limits): {excl_keys}\n")

# Each series has data as list of [year, value] or just values indexed by year
for k in incl_keys:
    pts = series_by_name[k]
    for i, pt in enumerate(pts):
        yr = years[i] if i < len(years) else None
        if yr:
            net_vals[yr] = net_vals.get(yr, 0) + (pt if isinstance(pt, (int, float)) else pt[1])

# Get solver limits from CSV
for yr in range(2010, 2051, 5):
    csv_f = os.path.join(BASE, f"out_{yr}", f"GHG_emissions_{yr}.csv")
    if os.path.exists(csv_f):
        df = pd.read_csv(csv_f)
        lim = df[df["Variable"]=="GHG_EMISSIONS_LIMIT_TCO2e"]["Emissions (t CO2e)"].values
        act_csv = df[df["Variable"]=="GHG_EMISSIONS_TCO2e"]["Emissions (t CO2e)"].values
        net_r = net_vals.get(yr, None)
        l = lim[0]/1e6 if len(lim) else None
        a_csv = act_csv[0]/1e6 if len(act_csv) else None
        print(f"  {yr}  net_report={net_r!r:12}  csv_act={a_csv:.2f}  limit={l:.2f}")
