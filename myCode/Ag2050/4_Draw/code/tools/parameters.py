# =============================================================================
# Parameters for Ag2050 Paper3 drawing scripts
# Scenarios: AgS1–AgS4 (4-panel 2×2 layout)
# =============================================================================

# ── Task root ────────────────────────────────────────────────────────────────
# Switch between NCI results and local test run as needed
TASK_ROOT = "20260311_Paper3_NCI"    # NCI run (zip archives, auto-read by data_helper)
# TASK_ROOT = "20260312_Paper3_test" # local test run (RF15, immediately available)

INPUT_DIR  = '../../../input'
# All outputs under ag2050/ alongside run results
AG2050_DIR  = f"../../../../output/{TASK_ROOT}/ag2050"
OUTPUT_DIR  = f"{AG2050_DIR}/figures"   # charts and assembled maps
TIFF_DIR    = f"{AG2050_DIR}/tiffs"     # extracted GeoTIFFs from zip
EXCEL_DIR   = f"{AG2050_DIR}/excel"     # exported long tables for figures

# ── Scenario definitions ─────────────────────────────────────────────────────
run_number_origin = [1, 2, 3, 4]
senerios_origin = ['AgS1', 'AgS2', 'AgS3', 'AgS4']

# Display labels for each scenario (used in figure titles / annotation)
SCENARIO_LABELS = {
    'Run_1_SCN_AgS1': 'AgS1\n(Ag AM only, no non-ag)',
    'Run_2_SCN_AgS2': 'AgS2\n(Ag AM + all non-ag)',
    'Run_3_SCN_AgS3': 'AgS3\n(Min AM, no non-ag)',
    'Run_4_SCN_AgS4': 'AgS4\n(Min AM, no non-ag v2)',
}

run_number = [num for num in run_number_origin]
senerios = [f"Run_{num}_SCN_{scen}" for num, scen in zip(run_number, senerios_origin)]
input_files = [scen for scen in senerios]

# ── Map subtasks: (output_stem, nc_stem, nc_sel, color_sheet) ────────────────
# output_stem → output PNG name prefix
# nc_stem     → xr_{nc_stem}_2050.nc inside out_2050/
# nc_sel      → dict passed to .sel() to pick the right layer
# color_sheet → sheet name in 'tools/land use colors.xlsx'
sub_tasks = [
    ('lumap_2050',  'map_lumap',  {'lm': 'ALL'},                          'ag_group_map'),  # Ag land use map    — layer=(lm,)
    ('ammap_2050',  'dvar_am',    {'am': 'ALL', 'lm': 'ALL', 'lu': 'ALL'}, 'am'),           # Ag management map  — layer=(am,lm,lu)
    ('non_ag_2050', 'dvar_non_ag',{'lu': 'ALL'},                           'non_ag'),        # Non-ag land use map — layer=(lu,)
]

# Generate task list for 01_Mapping.py: (scenario, output_stem, nc_stem, nc_sel, color_sheet)
tasks = [
    (main_task, sub_task[0], sub_task[1], sub_task[2], sub_task[3])
    for main_task in input_files
    for sub_task in sub_tasks
]

# ── Figure layout ─────────────────────────────────────────────────────────────
# 4 scenarios → 2 rows × 2 columns
n_rows = 2
n_cols = 2

# ── Chart style ───────────────────────────────────────────────────────────────
COLUMN_WIDTH  = 0.8    # bar width for stacked bar charts
X_OFFSET      = 0.5   # year-axis padding
font_size     = 25
axis_linewidth = 2
