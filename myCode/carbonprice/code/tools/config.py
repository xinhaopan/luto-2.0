N_JOBS = -1
COST_DICT = {
    'cost_am': [
        'Asparagopsis taxiformis',
        'Precision Agriculture',
        # 'Ecological Grazing',
        'Savanna Burning',
        'AgTech EI',
        'Biochar',
        'HIR - Beef',
        'HIR - Sheep'
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
        # 'BECCS'
    ]
}

KEY_TO_COLUMN_MAP = {
    "ag_profit": "Ag profit(M$)",
    "am_profit": "AM profit(M$)",
    "non-ag_profit": "Non-ag profit(M$)",
    "transition(ag2ag)_profit": "Transition(ag2ag) profit(M$)",
    "transition(ag2non-ag)_profit": "Transition(ag2non-ag) profit(M$)",
    "ghg_ag": "Ag GHG reductions (MtCO2e)",
    "ghg_am": "AM GHG reductions and removals (MtCO2e)",
    "ghg_non-ag": "Non-ag GHG removals (MtCO2e)",
    "ghg_tran": "Transition GHG reductions (MtCO2e)",
    "ghg": "GHG reductions and removals (MtCO2e)",
    "bio_ag": "Ag biodiversity restoration (Mha)",
    "bio_am": "AM biodiversity restoration (Mha)",
    "bio_non-ag": "Non-ag biodiversity restoration (Mha)",
    "bio": "Biodiversity restoration (Mha)",
}

# 创建 revenue_dict，主键从 cost 换成 revenue
REVENUE_DICT = {key.replace('cost', 'revenue'): value for key, value in COST_DICT.items()}


TASK_NAME = "20250823_Paper2_Results"

INPUT_FILES = [
    'Run_4_GHG_off_BIO_off',
    'Run_2_GHG_high_BIO_off',
    'Run_1_GHG_high_BIO_high',
    # 'Run_3_off_on',
    # 'Run_4_off_off',
]


NAME_DICT = {
    "cp": {"title": "Carbon Price", "unit": "(AU$/tCO2e)"},
    "bp": {"title": "Biodiversity Price", "unit": "(AU$/ha)"},
    "ghg": {"title": "GHG Emissions", "unit": "(Mt CO2e)"},
    "cost": {"title": "Cost", "unit": "(MAU$)"},
}

START_YEAR = 2025
COLUMN_NAME = ["Ag", "AM",  "Non-ag", "Transition(ag2ag)","Transition(ag2non-ag)"]
TASK_DIR = f'../../../output/{TASK_NAME}'


