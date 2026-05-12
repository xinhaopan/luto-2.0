"""
Task runner for AG2050 scenarios (AgS1 – AgS4).

Scenarios:
  AgS1 – Regional Ag capitals:   Non-ag OFF, full AG managements, maintain historical GHG & biodiversity
  AgS2 – Landscape stewardship:  Non-ag ON,  full AG managements, low GHG, restore 50% biodiversity
  AgS3 – Climate survival:       Non-ag OFF, limited AG managements (Eco Grazing/Savanna/HIR only), GHG off
  AgS4 – System decline:         Non-ag OFF, limited AG managements (Eco Grazing/Savanna/HIR only), GHG off

AG management availability per scenario:
  ─────────────────────────────────────────────────────────────────────
  Management                      AgS1  AgS2  AgS3  AgS4
  Asparagopsis taxiformis          ✓     ✓     ✗     ✗
  Precision Agriculture            ✓     ✓     ✗     ✗
  AgTech EI                        ✓     ✓     ✗     ✗
  Biochar                          ✓     ✓     ✗     ✗
  Ecological Grazing               ✓     ✓     ✓     ✓
  Savanna Burning                  ✓     ✓     ✓     ✓
  HIR - Beef                       ✓     ✓     ✓     ✓
  HIR - Sheep                      ✓     ✓     ✓     ✓
  ─────────────────────────────────────────────────────────────────────

Key differences from Paper2_tasks.py:
  • AG2050_MODE = True   activates all AG2050 overrides in data.py
  • AG2050_SCENARIO      drives PRODUCTIVITY_TREND, GHG_EMISSIONS_LIMITS and
                        BIODIVERSITY_TARGET_GBF_2 automatically via the mapping
                        tables in settings.py
  • NON_AG_LAND_USES and AG_MANAGEMENTS are set per scenario via conditional_rules
  • Demand is loaded from All_LUTO_demand_scenarios_with_convergences.csv
  • FLC multipliers from FLC_cost_multipliers.xlsx (scenario-specific sheet)
  • AC  multipliers from Area_cost.xlsx            (scenario-specific sheet)
  • Feedlot correction ratios applied per year for cost/revenue/water/GHG

Reference: myCode/tasks_run/Paper2_tasks.py
"""

import os
import pandas as pd
from tools.helpers import create_grid_search_template, create_task_runs


# ---------------------------------------------------------------------------
# Per-scenario NON_AG_LAND_USES configs
# ---------------------------------------------------------------------------
_non_ag_off = {
    'Environmental Plantings':      False,
    'Riparian Plantings':           False,
    'Sheep Agroforestry':           False,
    'Beef Agroforestry':            False,
    'Carbon Plantings (Block)':     False,
    'Sheep Carbon Plantings (Belt)': False,
    'Beef Carbon Plantings (Belt)': False,
    'BECCS':                        False,
    'Destocked - natural land':     False,
}

_non_ag_on = {
    'Environmental Plantings':      True,
    'Riparian Plantings':           True,
    'Sheep Agroforestry':           True,
    'Beef Agroforestry':            True,
    'Carbon Plantings (Block)':     True,
    'Sheep Carbon Plantings (Belt)': True,
    'Beef Carbon Plantings (Belt)': True,
    'BECCS':                        False,
    'Destocked - natural land':     True,
}

# ---------------------------------------------------------------------------
# Per-scenario AG_MANAGEMENTS configs
# ---------------------------------------------------------------------------
_ag_man_full = {                         # AgS1 & AgS2
    'Asparagopsis taxiformis': True,
    'Precision Agriculture':   True,
    'Ecological Grazing':      True,
    'Savanna Burning':         True,
    'AgTech EI':               True,
    'Biochar':                 True,
    'HIR - Beef':              True,
    'HIR - Sheep':             True,
    'Utility Solar PV':        False,
    'Onshore Wind':            False,
}

_ag_man_limited = {                      # AgS3 & AgS4
    'Asparagopsis taxiformis': False,
    'Precision Agriculture':   False,
    'Ecological Grazing':      True,
    'Savanna Burning':         True,
    'AgTech EI':               False,
    'Biochar':                 False,
    'HIR - Beef':              True,
    'HIR - Sheep':             True,
    'Utility Solar PV':        False,
    'Onshore Wind':            False,
}


grid_search = {
    'TASK_NAME': ['20260324_Paper3_aquila'],
    'KEEP_OUTPUTS': [True],
    'QUEUE': ['normalsr'],
    'NUMERIC_FOCUS': [3],
    # ---- Computational settings (not model parameters) ----------------------
    'MEM': ['68GB'],
    'NCPUS': ['17'],
    'WRITE_THREADS': ['2'],
    'TIME': ['48:00:00'],

    # ---- AG2050 scenario switch and selector ---------------------------------
    # Set AG2050_MODE=True to activate all AG2050 overrides.
    # AG2050_SCENARIO automatically drives PRODUCTIVITY_TREND, GHG_EMISSIONS_LIMITS,
    # and BIODIVERSITY_TARGET_GBF_2 via the mapping tables in settings.py.
    'AG2050_MODE': [True],
    'AG2050_SCENARIO': ['AgS1', 'AgS2', 'AgS3', 'AgS4'],

    # ---- Model settings -----------------------------------------------------
    'SOLVE_WEIGHT_ALPHA': [1],
    'SOLVE_WEIGHT_BETA': [0.9],
    'OBJECTIVE': ['maxprofit'],
    'WRITE_OUTPUT_GEOTIFFS': [True],
    'RESFACTOR': [5],
    'SIM_YEARS': [[i for i in range(2010, 2051, 1)]],

    # ---- GHG settings -------------------------------------------------------
    # GHG_EMISSIONS_LIMITS is auto-set by AG2050_SCENARIO; listed here for
    # documentation only – the per-scenario values are shown as comments.
    #   AgS1 → 'maintain_historical'  (keep ≤ 2010 level)
    #   AgS2 → 'low'                  (1.8 °C 67 % pathway)
    #   AgS3 → 'off'
    #   AgS4 → 'off'
    'GHG_CONSTRAINT_TYPE': ['hard'],
    'CARBON_PRICES_FIELD': ['CONSTANT'],
    'CARBON_EFFECTS_WINDOW': [60],
    'GHG_TARGETS_DICT': [{
        'off': None,
        'low': '1.8C (67%) excl. avoided emis SCOPE1',
        'medium': '1.5C (50%) excl. avoided emis SCOPE1',
        'high': '1.5C (67%) excl. avoided emis SCOPE1',
    }],

    # ---- Biodiversity settings ----------------------------------------------
    # BIODIVERSITY_TARGET_GBF_2 is auto-set by AG2050_SCENARIO; shown as comments:
    #   AgS1 → 'maintain_historical'  (floor = 2010 GBF-2 score)
    #   AgS2 → 'high'                 (restore 50 % in top-30 % priority areas)
    #   AgS3 → 'off'
    #   AgS4 → 'off'
    'GBF2_CONSTRAINT_TYPE': ['hard'],
    'GBF2_PRIORITY_DEGRADED_AREAS_PERCENTAGE_CUT': [40],
    'GBF2_TARGETS_DICT': [{
        'low':    {2030: 0,    2050: 0,    2100: 0},
        'medium': {2030: 0.30, 2050: 0.30, 2100: 0.30},
        'high':   {2030: 0.30, 2050: 0.50, 2100: 0.50},
    }],
    'BIO_QUALITY_LAYER': ['Suitability'],
    'BIODIVERSITY_TARGET_GBF_3': ['off'],
    'BIODIVERSITY_TARGET_GBF_4_SNES': ['off'],
    'BIODIVERSITY_TARGET_GBF_4_ECNES': ['off'],
    'BIODIVERSITY_TARGET_GBF_8': ['off'],

    # ---- Water settings -----------------------------------------------------
    'WATER_STRESS': [0.6],
    'WATER_LIMITS': ['on'],
    'WATER_CONSTRAINT_TYPE': ['hard'],
    'INCLUDE_WATER_LICENSE_COSTS': [1],
    'WATER_REGION_DEF': ['Drainage Division'],
    'WATER_CLIMATE_CHANGE_IMPACT': ['on'],

    # ---- Demand settings ----------------------------------------------------
    # Demand is read from All_LUTO_demand_scenarios_with_convergences.csv when
    # AG2050_MODE is True; APPLY_DEMAND_MULTIPLIERS is skipped automatically.
    'DYNAMIC_PRICE': [False],
    'DEMAND_CONSTRAINT_TYPE': ['soft'],

    # ---- Per-scenario land use & management options -------------------------
    # Two variants each; conditional_rules below select the correct pair.
    'REGIONAL_ADOPTION_CONSTRAINTS': ['off'],
    'NON_AG_LAND_USES': [_non_ag_off, _non_ag_on],
    'AG_MANAGEMENTS':   [_ag_man_full, _ag_man_limited],
}


# Conditional rules: for each scenario keep only its matching NON_AG / AG_MAN pair.
conditional_rules = [
    # AgS1 – non-ag OFF, full managements
    {'conditions':    {'AG2050_SCENARIO': ['AgS1']},
     'restrictions':  {'NON_AG_LAND_USES': [_non_ag_off], 'AG_MANAGEMENTS': [_ag_man_full]}},
    # AgS2 – non-ag ON, full managements
    {'conditions':    {'AG2050_SCENARIO': ['AgS2']},
     'restrictions':  {'NON_AG_LAND_USES': [_non_ag_on],  'AG_MANAGEMENTS': [_ag_man_full]}},
    # AgS3 – non-ag OFF, limited managements
    {'conditions':    {'AG2050_SCENARIO': ['AgS3']},
     'restrictions':  {'NON_AG_LAND_USES': [_non_ag_off], 'AG_MANAGEMENTS': [_ag_man_limited]}},
    # AgS4 – non-ag OFF, limited managements
    {'conditions':    {'AG2050_SCENARIO': ['AgS4']},
     'restrictions':  {'NON_AG_LAND_USES': [_non_ag_off], 'AG_MANAGEMENTS': [_ag_man_limited]}},
]

settings_name_dict = {
    'AG2050_SCENARIO': 'SCN',
}

task_root_dir = f"../../output/{grid_search['TASK_NAME'][0]}"

# ---- Uncomment to generate the template CSV (first run only) ----------------
grid_search_settings_df = create_grid_search_template(
    grid_search, settings_name_dict, conditional_rules=conditional_rules
)
print(grid_search_settings_df.columns)

# ---- Load existing template and launch tasks --------------------------------
# grid_search_settings_df = pd.read_csv(
#     os.path.join(task_root_dir, 'grid_search_template.csv'), index_col=0
# )
create_task_runs(
    task_root_dir,
    grid_search_settings_df,
    platform="aquila",
    n_workers=min(len(grid_search_settings_df.columns), 50),
    use_parallel=True,
)
