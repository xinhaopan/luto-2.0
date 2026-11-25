# Settings shared by every Paper 2 revision run (luto/settings.py names), in the grid format of our task scripts (each value a one-item list). Anything not listed stays at the code default.
# Code: /g/data/jk53/LUTO_XH/LUTO2_paper2_v1, branch paper2_v1, commit dcb03540 (your jinzhu branch b578e1d7 merged in).
# GHG_NET_ROW is ours: luto/settings.py lines 883-899, luto/solvers/row_builder.py lines 356-378.

COMMON = {
# ---------------------------------- Model settings ------------------------------
    'OBJECTIVE': ['maxprofit'],
    'RESFACTOR': [1],
    'SIM_YEARS': [list(range(2010, 2051))],
    'WRITE_OUTPUTS': [False],

    # ------------------------- Cell grouping (Jinzhu, 2026-11-18) -------------------------
    'CELL_GROUPING': [True],
    'BIO_ROW_GENERATION': [False],        # raises an error with cell grouping
    'ELASTIC_FAMILIES': [[]],             # raises an error with cell grouping

    # ----------------------------------- GHG settings --------------------------------
    'GHG_NET_ROW': [True],
    'GHG_NET_TARGETS_DICT': [{
        'low':  '1.8C (67%) excl. avoided emis SCOPE1',
        'high': '1.5C (67%) excl. avoided emis SCOPE1',
    }],
    'CARBON_PRICES_FIELD': ['CONSTANT'],
    'CARBON_EFFECTS_WINDOW': [60],

    # ----------------------------- Biodiversity settings -------------------------------
    'GBF2_TARGETS_DICT': [{
        'off': None,
        'low': {2030: 0, 2050: 0, 2100: 0},
        'medium': {2030: 0.30, 2050: 0.30, 2100: 0.30},
        'high': {2030: 0.30, 2050: 0.50, 2100: 0.50},
    }],
    'BIO_QUALITY_LAYER': ['Suitability'],
    'GBF3_NVIS_TARGET': ['off'],
    'GBF4_TARGET_SNES': ['off'],
    'GBF4_TARGET_ECNES': ['off'],
    'GBF8_TARGET': ['off'],

    # ----------------------------------- Water settings --------------------------------
    'WATER_STRESS': [0.6],
    'WATER_LIMITS': ['on'],
    'INCLUDE_WATER_LICENSE_COSTS': [1],
    'WATER_REGION_DEF': ['Drainage Division'],
    'WATER_CLIMATE_CHANGE_IMPACT': ['on'],
    # ----------------------------------- Demand settings --------------------------------
    'DYNAMIC_PRICE': [False],             # the current code defaults to True, the first submission used False
    #----------------------------------- other settings --------------------------------
    'REGIONAL_ADOPTION_CONSTRAINTS': ['off'],
    'NON_AG_LAND_USES': [{
        'Environmental Plantings': True,
        'Riparian Plantings': True,
        'Sheep Agroforestry': True,
        'Beef Agroforestry': True,
        'Carbon Plantings (Block)':  True,
        'Sheep Carbon Plantings (Belt)':  True,
        'Beef Carbon Plantings (Belt)':  True,
        'BECCS': False,
        'Destocked - natural land': True,
    }]

}
