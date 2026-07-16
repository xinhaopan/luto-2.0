"""
Task runner for AG2050 scenarios (AgS1 – AgS4).

Scenarios:
  AgS1 – Regional Ag capitals:   Non-ag ON, full AG managements, maintain historical GHG and 2010 national biodiversity, High productivity
  AgS2 – Landscape stewardship:  Non-ag ON,  full AG managements, low GHG, restore 50% biodiversity, Very High productivity
  AgS3 – Climate survival:       Non-ag OFF, limited AG managements (Eco Grazing/Savanna/HIR only), GHG off
  AgS4 – System decline:         Non-ag OFF, limited AG managements (Eco Grazing/Savanna/HIR only), GHG off

  AgS1 keeps the national all-cell Suitability score at or above its 2010 level.
  AgS2 restores 50% of the top-20% biodiversity-priority areas. Both cap total
  non-ag land at <=15% per NRM region (REGIONAL_ADOPTION_CONSTRAINTS='NON_AG_CAP').
  Transitions use jinzhu's per-source delta-flow model (now the only mode).

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
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

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
    'TASK_NAME': ['20260718_Paper3_NCI'],
    'KEEP_OUTPUTS': [False],
    'QUEUE': ['normalsr'],
    # 'NUMERIC_FOCUS': [0],  # [merge] removed in jinzhu; solver NumericFocus no longer configurable via settings
    # ---- Computational settings (not model parameters) ----------------------
    # NCI RESFACTOR=3 production run: 200 GB / 50 CPUs / 48 h.
    #
    # Note on the CPUs: with Crossover on auto (-1), most of the wall time is single-threaded.
    # Gurobi's barrier parallelises (measured: 22 cores busy), but the crossover clean-up that
    # follows it is a sequential simplex -- AgS1/2019 at RESFACTOR=5 spent 143 minutes on ONE
    # core after the barrier had found the optimum in 8 minutes on 22. Extra cores speed up the
    # barrier and do nothing for the crossover, so 50 buys less than it looks like it should.
    'MEM': ['200GB'],
    'NCPUS': ['50'],
    # 'WRITE_THREADS': ['2'],  # [merge] removed in jinzhu; write threading is now internal (n_jobs auto)
    'TIME': ['720:00:00'],  # 30 days
    'SOLVE_TIME_LIMIT_SECONDS': [30 * 24 * 3600],  # 30 days per Gurobi attempt

    # ---- AG2050 scenario switch and selector ---------------------------------
    # Set AG2050_MODE=True to activate all AG2050 overrides.
    # AG2050_SCENARIO automatically drives PRODUCTIVITY_TREND, GHG_EMISSIONS_LIMITS,
    # and BIODIVERSITY_TARGET_GBF_2 via the mapping tables in settings.py.
    'AG2050_MODE': [True],
    'AG2050_SCENARIO': ['AgS1', 'AgS2', 'AgS3', 'AgS4'],

    # ---- Model settings -----------------------------------------------------
    # 'SOLVE_WEIGHT_ALPHA': [1],  # [merge] removed in jinzhu; objective now uses SOLVE_WEIGHT_BETA only
    'SOLVE_WEIGHT_BETA': [0.9],
    'OBJECTIVE': ['maxprofit'],
    # Upstream's retry params -- Crossover on auto (-1). None of the RESFACTOR=5 infeasibilities
    # were a solver-configuration problem: every one of them was a cell whose base state held more
    # land than the cell has, and the IIS for each is a handful of rows (one const_cell_usage plus
    # that cell's bal_a/bal_n), never a biodiversity, GHG, water or regional-adoption constraint.
    # `_project_base_into_cell` in solvers/input_data.py is what fixes them.
    #
    # Tolerance 1e-4, and it has to sit in that window from both sides. Above it, a loose tolerance
    # buys a sloppy solution -- 1e-2 permits a 1% constraint violation. Below it, the base dvars are
    # float32, so projecting agriculture onto `capacity - non_ag` leaves a rounding residual of ~2
    # float32 ULP (measured at RESFACTOR=5: max 2.4e-07, mean 4e-10), and FEASIBILITY_TOLERANCE has
    # to clear it: 1e-4 leaves 400x of margin, 1e-6 would leave 4x.
    'RETRY_PARAMS': [[(0, 2, -1, -1, -1), (0, 1, 0, -1, 0)]],
    'FEASIBILITY_TOLERANCE': [1e-4],
    'OPTIMALITY_TOLERANCE': [1e-4],
    'RESFACTOR': [3],
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
    # GBF2_TARGET is auto-set from AG2050_BIO_MAP[scenario] at runtime (data.py).
    # Override the map here so:
    #   AgS1 Regional Ag capitals  → 'maintain_historical' (national 2010 floor)
    #   AgS2 Landscape stewardship → 'high' (restore 50% by 2050)
    #   AgS3 / AgS4                → 'off'
    'AG2050_BIO_MAP': [{
        'AgS1': 'maintain_historical',
        'AgS2': 'high',
        'AgS3': 'off',
        'AgS4': 'off',
    }],
    'GBF2_CONSTRAINT_TYPE': ['hard'],
    # top-20% priority areas: CUT = normalised area % (0=none, 100=all) → 20 = top 20%
    'GBF2_PRIORITY_DEGRADED_AREAS_PERCENTAGE_CUT': [20],
    'GBF2_TARGETS_DICT': [{
        # The 2030 milestone is half of the corresponding 2050 target.
        'low':    {2030: 0.025, 2050: 0.05, 2100: 0.05},  # Unused in Paper3; AgS1 uses maintain_historical.
        'medium': {2030: 0.15,  2050: 0.30, 2100: 0.30},
        'high':   {2030: 0.25,  2050: 0.50, 2100: 0.50},  # AgS2 Landscape stewardship: restore 50% by 2050
    }],
    'BIO_QUALITY_LAYER': ['Suitability'],
    'GBF3_NVIS_TARGET': ['off'],
    'GBF4_TARGET_SNES': ['off'],
    'GBF4_TARGET_ECNES': ['off'],
    'GBF8_TARGET': ['off'],

    # ---- Water settings -----------------------------------------------------
    'WATER_STRESS': [0.6],
    'WATER_LIMITS': ['off'],
    'WATER_CONSTRAINT_TYPE': ['hard'],
    'INCLUDE_WATER_LICENSE_COSTS': [1],
    'WATER_REGION_DEF': ['Drainage Division'],
    'WATER_CLIMATE_CHANGE_IMPACT': ['on'],

    # ---- Demand settings ----------------------------------------------------
    # Demand is read from All_LUTO_demand_scenarios_with_convergences.csv when
    # AG2050_MODE is True; APPLY_DEMAND_MULTIPLIERS is skipped automatically.
    'DYNAMIC_PRICE': [False],
    'DEMAND_CONSTRAINT_TYPE': ['hard'],
    'DEMAND_BOUNDS': [{
        'sheep lexp': [1.0, 1.0],
        'sheep meat': [1.0, 1.0],
        'sheep wool': [0, 1.5],

        # all other commodities
        'apples': [1.0, 1.0],
        'beef lexp': [1.0, 1.0],
        'beef meat': [1.0, 1.0],
        'citrus': [1.0, 1.0],
        'cotton': [1.0, 1.0],
        'dairy': [1.0, 1.0],
        'grapes': [1.0, 1.0],
        'hay': [1.0, 1.0],
        'nuts': [1.0, 1.0],
        'other non-cereal crops': [1.0, 1.0],
        'pears': [1.0, 1.0],
        'plantation fruit': [1.0, 1.0],
        'rice': [1.0, 1.0],
        'stone fruit': [1.0, 1.0],
        'sugar': [1.0, 1.0],
        'summer cereals': [1.0, 1.0],
        'summer legumes': [1.0, 1.0],
        'summer oilseeds': [1.0, 1.0],
        'tropical stone fruit': [1.0, 1.0],
        'vegetables': [1.0, 1.0],
        'winter cereals': [1.0, 1.0],
        'winter legumes': [1.0, 1.0],
        'winter oilseeds': [1.0, 1.0],
    }],


    # ---- Regional adoption: 15% NRM non-ag cap ------------------------------
    # Total combined non-ag land <= 15% of each NRM region. Binds AgS1/AgS2
    # (non-ag ON); harmless for AgS3/AgS4 (non-ag OFF).
    'REGIONAL_ADOPTION_CONSTRAINTS': ['NON_AG_CAP'],
    'REGIONAL_ADOPTION_NON_AG_REGION': ['NRM'],
    'REGIONAL_ADOPTION_NON_AG_CAP': [15],

    # ---- Per-scenario land use & management options -------------------------
    # Two variants each; conditional_rules below select the correct pair.
    'NON_AG_LAND_USES': [_non_ag_off, _non_ag_on],
    'AG_MANAGEMENTS':   [_ag_man_full, _ag_man_limited],
}


# Conditional rules: for each scenario keep only its matching NON_AG / AG_MAN pair.
conditional_rules = [
    # AgS1 – non-ag ON, full managements
    {'conditions':    {'AG2050_SCENARIO': ['AgS1']},
     'restrictions':  {'NON_AG_LAND_USES': [_non_ag_on],  'AG_MANAGEMENTS': [_ag_man_full]}},
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

# Optional comma-separated filter for resubmitting only changed scenarios.
requested_scenarios = os.environ.get('AG2050_RUN_SCENARIOS', '').strip()
if requested_scenarios:
    selected = [item.strip() for item in requested_scenarios.split(',') if item.strip()]
    unknown = sorted(set(selected) - set(grid_search['AG2050_SCENARIO']))
    if unknown:
        raise ValueError(f'Unknown AG2050 scenario(s): {unknown}')
    grid_search['AG2050_SCENARIO'] = selected

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
if os.environ.get('AG2050_DRY_RUN') == '1':
    print('AG2050_DRY_RUN=1: generated task tables without submitting jobs.')
else:
    create_task_runs(
        task_root_dir,
        grid_search_settings_df,
        platform="NCI",
        n_workers=min(len(grid_search_settings_df.columns), 50),
        use_parallel=True,
    )
