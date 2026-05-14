import pandas as pd, os

BASE = r"F:\Users\s222552331\Work\LUTO2_XH\luto-2.0\output\20260414_Paper3_NCI"

for run, label in [("Run_1_SCN_AgS1", "AgS1"), ("Run_2_SCN_AgS2", "AgS2")]:
    out_root = os.path.join(BASE, run, "output")
    if not os.path.exists(out_root):
        print(f"{label}: no output dir"); continue
    subdirs = [d for d in os.listdir(out_root) if os.path.isdir(os.path.join(out_root, d))]
    if not subdirs:
        print(f"{label}: empty output dir"); continue
    outdir = os.path.join(out_root, subdirs[0])
    print(f"\n=== {label} ({subdirs[0]}) ===")
    print(f"  {'Year':4}  {'Limit (Mt)':12}  {'Actual (Mt)':12}  Status")
    over, found = 0, 0
    for yr in range(2010, 2051):
        f = os.path.join(outdir, f"out_{yr}", f"GHG_emissions_{yr}.csv")
        if not os.path.exists(f):
            continue
        found += 1
        df = pd.read_csv(f)
        lim = df[df["Variable"]=="GHG_EMISSIONS_LIMIT_TCO2e"]["Emissions (t CO2e)"].values
        act = df[df["Variable"]=="GHG_EMISSIONS_TCO2e"]["Emissions (t CO2e)"].values
        if len(lim) and len(act):
            l, a = lim[0]/1e6, act[0]/1e6
            if a > l + 0.01:
                over += 1
                print(f"  {yr}  {l:12.2f}  {a:12.2f}  *** OVER by {a-l:.2f} Mt")
            elif yr % 5 == 0:
                print(f"  {yr}  {l:12.2f}  {a:12.2f}  ok")
    print(f"  -> {found} years checked, {over} violations")
