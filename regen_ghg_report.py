"""
Regenerate GHG_overview_sum.js for all aquila runs after fixing the GHG double-counting bug.
Only re-runs process_ghg_data (fast), not the full report regeneration.
"""
import os, sys, json
sys.path.insert(0, r"F:\Users\s222552331\Work\LUTO2_XH\luto-2.0")

AQUILA = r"F:\Users\s222552331\Work\LUTO2_XH\luto-2.0\output\20260512_Paper3_aquila"

runs = {
    "Run_1_SCN_AgS1": "AgS1",
    "Run_2_SCN_AgS2": "AgS2",
    "Run_3_SCN_AgS3": "AgS3",
    "Run_4_SCN_AgS4": "AgS4",
}

for run_folder, scn in runs.items():
    import luto.settings as settings
    settings.AG2050_MODE = True
    settings.AG2050_SCENARIO = scn

    run_base = os.path.join(AQUILA, run_folder, "output")
    output_dirs = [d for d in os.listdir(run_base) if os.path.isdir(os.path.join(run_base, d))]
    if not output_dirs:
        print(f"[{run_folder}] No output dir found, skipping.")
        continue
    raw_data_dir = os.path.join(run_base, output_dirs[0])
    SAVE_DIR = os.path.join(raw_data_dir, "DATA_REPORT", "data")

    from luto.tools.report.data_tools import get_all_files
    from luto.tools.report.create_report_data import process_ghg_data
    from luto.tools.report.data_tools.parameters import GROUP_LU

    years = sorted(settings.SIM_YEARS)
    files = get_all_files(raw_data_dir).reset_index(drop=True)
    files["Year"] = files["Year"].astype(int)
    files = files.query("Year.isin(@years)")
    lu_group_map = {lu: cat for cat, lus in GROUP_LU.items() for lu in lus}

    print(f"\n[{run_folder}] Regenerating GHG data → {SAVE_DIR}")
    result = process_ghg_data(files, SAVE_DIR, lu_group_map, years)
    print(f"  Done: {result}")

print("\nAll runs complete. GHG_overview_sum.js updated for all aquila scenarios.")
