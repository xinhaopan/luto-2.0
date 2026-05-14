import pandas as pd, os, sys

BASE = r"F:\Users\s222552331\Work\LUTO2_XH\luto-2.0\output\20260512_Paper3_test"
OUT1 = os.path.join(BASE, "Run_1_SCN_AgS1", "output", "2026_05_12__21_13_13_RF15_2010-2050")
OUT2 = os.path.join(BASE, "Run_2_SCN_AgS2", "output", "2026_05_12__21_13_13_RF15_2010-2050")

for label, outdir in [('AgS1 (maintain_historical)', OUT1), ('AgS2 (low)', OUT2)]:
    over_years = []
    found = 0
    for yr in range(2010, 2051):
        f = os.path.join(outdir, f"out_{yr}", f"GHG_emissions_{yr}.csv")
        if os.path.exists(f):
            found += 1
            df = pd.read_csv(f)
            lim = df[df['Variable'] == 'GHG_EMISSIONS_LIMIT_TCO2e']['Emissions (t CO2e)'].values
            act = df[df['Variable'] == 'GHG_EMISSIONS_TCO2e']['Emissions (t CO2e)'].values
            if len(lim) and len(act):
                l, a = lim[0] / 1e6, act[0] / 1e6
                if a > l + 0.01:
                    over_years.append((yr, l, a, a - l))
    print(f"\n=== {label} ({found} years found) ===")
    if over_years:
        print(f"  {len(over_years)} years OVER target:")
        for yr, l, a, diff in over_years:
            print(f"  {yr}: limit={l:.2f}  actual={a:.2f}  over_by={diff:.2f} Mt")
    else:
        print(f"  ALL years within GHG limit -- hard constraint satisfied")
