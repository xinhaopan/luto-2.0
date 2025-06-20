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
    'cost_non_ag': [
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
    "opportunity_cost": "Opportunity Cost(M$)",
    "transition_ag2ag_cost": "Transition AG2AG Cost(M$)",
    "transition_ag2non-ag_cost": "Transition AG2Non-AG Cost(M$)",
    "am_net_cost": "AM Net Cost(M$)",
    "non_ag_net_cost": "Non-AG Net Cost(M$)",
    "cost": "All cost(M$)",
    "biodiversity_cost": "BIO cost(M$)",
    "ghg": "GHG Abatement(MtCOe2)",
    "bio": "BIO(Mha)",
    "carbon_price": "carbon price($/tCOe2)",
    "biodiversity_price": "biodiversity price($/ha)"
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
COST_COLUMN = ["Agricultural Cost(M$)", "Agricultural Cost(M$)",  "Non-agricultural Cost(M$)", "Transition cost(M$)"]
TASK_DIR = f'../../../output/{TASK_NAME}'
