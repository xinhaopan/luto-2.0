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
    "opportunity_cost": "AG opportunity cost(M$)",
    "am_net_cost": "AM net Cost(M$)",
    "non-ag_net_cost": "NON-AG net cost(M$)",
    "transition_cost": "Transition cost(M$)",
    "cost": "All cost(M$)",
    "ghg_ag": "AG carbon sequestration (MtCO2e)",
    "ghg_am": "AM carbon sequestration (MtCO2e)",
    "ghg_non-ag": "NON-AG carbon sequestration (MtCO2e)",
    "ghg_tran": "Transition carbon sequestration (MtCO2e)",
    "ghg": "Carbon sequestration (MtCO2e)",
    "bio_ag": "AG biodiversity restoration (Mha)",
    "bio_am": "AM biodiversity restoration (Mha)",
    "bio_non-ag": "NON-AG biodiversity restoration (Mha)",
    "bio": "Biodiversity restoration (Mha)",
}

# 创建 revenue_dict，主键从 cost 换成 revenue
REVENUE_DICT = {key.replace('cost', 'revenue'): value for key, value in COST_DICT.items()}


TASK_NAME = "20250618_Paper2_Results"

INPUT_FILES = [
    'Run_4_GHG_high_BIO_high',
    'Run_3_GHG_high_BIO_off',
    # 'Run_3_off_on',
    # 'Run_4_off_off',
]


NAME_DICT = {
    "cp": {"title": "Carbon Price", "unit": "(AU$/tCO2e)"},
    "bp": {"title": "Biodiversity Price", "unit": "(AU$/ha)"},
    "ghg": {"title": "GHG Emissions", "unit": "(Mt CO2e)"},
    "cost": {"title": "Cost", "unit": "(MAU$)"},
}

START_YEAR = 2010
COST_COLUMN = ["AG opportunity cost(M$)", "AM net cost(M$)",  "NON-AG net cost(M$)", "Transition cost(M$)"]
TASK_DIR = f'../../../output/{TASK_NAME}'
