import math

TASK_NAME = "20260416_paper4"

# ---------------------------------------------------------------------------
# Carbon price levels used in grid search (CARBON_PRICE_COSTANT values)
# ---------------------------------------------------------------------------
CARBON_PRICES = [0, 8.92, 17.85, 26.77, 35.69, 44.61, 53.54, 62.46,
                 75.84, 89.23, 133.84, 178.46, 233.07, 267.69, 312.3, 356.92]

# Biodiversity CUT levels
BIO_CUTS = [10, 20, 30, 40, 50]

# Total runs: BIO_off × 1 cut × 16 prices  +  BIO_high × 5 cuts × 16 prices = 96
N_JOBS = math.ceil(96 / 1)

# ---------------------------------------------------------------------------
# Scenario groups
# ---------------------------------------------------------------------------
# BIO off: one cut level (50), all carbon prices
bio_off_scenarios = [f'GBF2_off_CUT_50_CarbonPrice_{cp}' for cp in CARBON_PRICES]

# BIO high: five cut levels, all carbon prices
bio_high_scenarios = [
    f'GBF2_high_CUT_{cut}_CarbonPrice_{cp}'
    for cut in BIO_CUTS
    for cp in CARBON_PRICES
]

all_scenarios = bio_off_scenarios + bio_high_scenarios

# Reference scenario (BIO off, carbon price = 0)
REFERENCE_SCENARIO = 'GBF2_off_CUT_50_CarbonPrice_0'

# ---------------------------------------------------------------------------
# File categories
# ---------------------------------------------------------------------------
economic_files = [
    'xr_economics_ag_cost', 'xr_economics_am_cost', 'xr_economics_non_ag_cost',
    'xr_transition_cost_ag2ag', 'xr_transition_cost_ag2non_ag',
    'xr_transition_cost_ag2non_ag_amortised',
    'xr_economics_ag_revenue', 'xr_economics_am_revenue', 'xr_economics_non_ag_revenue',
]
carbon_files = [
    'xr_GHG_ag', 'xr_GHG_ag_management', 'xr_GHG_non_ag', 'xr_transition_GHG',
]
bio_files = [
    'xr_biodiversity_overall_priority_ag',
    'xr_biodiversity_overall_priority_ag_management',
    'xr_biodiversity_overall_priority_non_ag',
]

# ---------------------------------------------------------------------------
# Column mapping: NetCDF file key → display label
# ---------------------------------------------------------------------------
KEY_TO_COLUMN_MAP = {
    "xr_economics_ag_cost":                        "Ag cost",
    "xr_economics_am_cost":                        "AgMgt cost",
    "xr_economics_non_ag_cost":                    "Non-ag cost",
    "xr_transition_cost_ag2ag":                    "Transition(ag→ag) cost",
    "xr_transition_cost_ag2non_ag":                "Transition(ag→non-ag) cost",
    "xr_transition_cost_ag2non_ag_amortised":      "Transition(ag→non-ag) amortised cost",
    "xr_economics_ag_revenue":                     "Ag revenue",
    "xr_economics_am_revenue":                     "AgMgt revenue",
    "xr_economics_non_ag_revenue":                 "Non-ag revenue",
    # GHG
    "xr_GHG_ag":                                   "Ag GHG",
    "xr_GHG_ag_management":                        "AgMgt GHG",
    "xr_GHG_non_ag":                               "Non-ag GHG",
    "xr_transition_GHG":                           "Transition GHG",
    # Biodiversity
    "xr_biodiversity_overall_priority_ag":            "Ag biodiversity",
    "xr_biodiversity_overall_priority_ag_management": "AgMgt biodiversity",
    "xr_biodiversity_overall_priority_non_ag":        "Non-ag biodiversity",
}

# ---------------------------------------------------------------------------
# Cost / revenue land-use groups
# ---------------------------------------------------------------------------
COST_DICT = {
    'cost_am': [
        'Asparagopsis taxiformis',
        'Precision Agriculture',
        'Savanna Burning',
        'AgTech EI',
        'Biochar',
        'HIR - Beef',
        'HIR - Sheep',
    ],
    'cost_non-ag': [
        'Environmental Plantings',
        'Riparian Plantings',
        'Sheep Agroforestry',
        'Beef Agroforestry',
        'Carbon Plantings (Block)',
        'Sheep Carbon Plantings (Belt)',
        'Beef Carbon Plantings (Belt)',
        'Destocked - natural land',
    ],
}

REVENUE_DICT = {key.replace('cost', 'revenue'): value for key, value in COST_DICT.items()}

# ---------------------------------------------------------------------------
# Axis / title helpers
# ---------------------------------------------------------------------------
NAME_DICT = {
    "cp":   {"title": "Carbon Price",       "unit": "(AU$/tCO2e)"},
    "bp":   {"title": "Biodiversity Price", "unit": "(AU$/ha)"},
    "ghg":  {"title": "GHG Emissions",      "unit": "(Mt CO2e)"},
    "cost": {"title": "Cost",               "unit": "(MAU$)"},
}

START_YEAR = 2025
COLUMN_NAME = ["Ag", "AM", "Non-ag", "Transition(ag2ag)", "Transition(ag2non-ag)"]

# ---------------------------------------------------------------------------
# Title maps for plots
# ---------------------------------------------------------------------------
# Short label for each scenario group used in figure titles / legends
ORIGINAL_TITLE_MAP = {
    'GBF2_off_CUT_50_CarbonPrice_0':   'Reference',
    **{f'GBF2_off_CUT_50_CarbonPrice_{cp}': f'CP={cp}' for cp in CARBON_PRICES if cp != 0},
    **{f'GBF2_high_CUT_{cut}_CarbonPrice_0': f'NP{cut}%' for cut in BIO_CUTS},
}

# Full label map: every scenario
ALL_TITLE_MAP = {
    **{f'GBF2_off_CUT_50_CarbonPrice_{cp}':
           ('Reference' if cp == 0 else f'CP={cp}')
       for cp in CARBON_PRICES},
    **{f'GBF2_high_CUT_{cut}_CarbonPrice_{cp}':
           f'NP{cut}%, CP={cp}'
       for cut in BIO_CUTS for cp in CARBON_PRICES},
}

# Carbon-price axis label map (scenario name → carbon price string)
CP_TITLE_MAP = {s: str(cp) for s, cp in
                [(f'GBF2_off_CUT_50_CarbonPrice_{cp}', cp) for cp in CARBON_PRICES] +
                [(f'GBF2_high_CUT_{cut}_CarbonPrice_{cp}', cp)
                 for cut in BIO_CUTS for cp in CARBON_PRICES]}

# Bio-price axis label map (scenario name → cut label)
BP_TITLE_MAP = {f'GBF2_high_CUT_{cut}_CarbonPrice_{cp}': f'NP{cut}%'
                for cut in BIO_CUTS for cp in CARBON_PRICES}
