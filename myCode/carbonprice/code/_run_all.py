"""
Run all carbonprice plotting scripts for each discount rate directory.
Path substitution: /carbon_price/ → /carbon_price_{rate}/
"""
import os
import sys
import traceback

RATES = ["0.03", "0.05", "0.07"]

SCRIPTS = [
    "01_create_GBF2_targets_csv.py",
    "02_draw_original_data.py",
    "03_draw_processed_data.py",
    "04_draw_average_price.py",
    "05_draw_sol_average_price.py",
    "06_make_cost_map.py",
    "07_make_GHG_benefits_map.py",
    "08_make_BIO_benefits_map.py",
    "09_make_price_map.py",
    "10_make_agmgt_map.py",
    "11_make_non_ag_map.py",
    "12_biodiversity_contribution_curve.py",
    "13_Rename_figure.py",
]


def patch_source(source: str, rate: str) -> str:
    """Replace carbon_price directory name in path strings."""
    source = source.replace('/carbon_price/', f'/carbon_price_{rate}/')
    source = source.replace('"carbon_price"', f'"carbon_price_{rate}"')
    source = source.replace("'carbon_price'", f"'carbon_price_{rate}'")
    # handle end-of-string cases like .../carbon_price"
    source = source.replace('/carbon_price"', f'/carbon_price_{rate}"')
    source = source.replace("/carbon_price'", f"/carbon_price_{rate}'")
    return source


for rate in RATES:
    print(f"\n{'='*60}")
    print(f"  RATE = {rate}")
    print(f"{'='*60}")

    for script in SCRIPTS:
        print(f"\n--- {script} ---")
        if not os.path.exists(script):
            print(f"  SKIP: file not found")
            continue

        with open(script, 'r', encoding='utf-8') as f:
            source = f.read()

        source = patch_source(source, rate)

        ns = {'__name__': '__main__', '__file__': os.path.abspath(script)}
        try:
            exec(compile(source, script, 'exec'), ns)
            print(f"  OK")
        except Exception as e:
            print(f"  ERROR in {script}: {e}")
            traceback.print_exc()
            print("  Continuing to next script...")

print("\n\nAll done.")
