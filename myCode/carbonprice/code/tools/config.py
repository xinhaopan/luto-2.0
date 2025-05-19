n_jobs = -1
cost_dict = {
    'cost_am': [
        'Asparagopsis taxiformis',
        'Precision Agriculture',
        # 'Ecological Grazing',
        'Savanna Burning',
        'AgTech EI',
        'Biochar',
    ],
    'cost_non_ag': [
        'Environmental Plantings',
        'Riparian Plantings',
        'Sheep Agroforestry',
        'Beef Agroforestry',
        'Carbon Plantings (Block)',
        'Sheep Carbon Plantings (Belt)',
        'Beef Carbon Plantings (Belt)',
        # 'BECCS'
    ]
}


# 创建 revenue_dict，主键从 cost 换成 revenue
revenue_dict = {key.replace('cost', 'revenue'): value for key, value in cost_dict.items()}

time = '20250510'
refactor = '5'
include_water_cost = '1'
suffix = include_water_cost + '_' + refactor

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

senerios = [
    'Run_1_on_on',
    'Run_2_on_off',
    # 'Run_3_off_on',
    # 'Run_4_off_off',
]
input_files = [
    f"{time}_{senerio}_{suffix}"
    for senerio in senerios
]

name_dict = {
    "cp": {"title": "Carbon Price", "unit": "(AU$/tCO2e)"},
    "bp": {"title": "Biodiversity Price", "unit": "(AU$/ha)"},
    "ghg": {"title": "GHG Emissions", "unit": "(Mt CO2e)"},
    "cost": {"title": "Cost", "unit": "(MAU$)"},
}

start_year = 2010
cost_columns = ["Opportunity Cost(M$)", "Transition AG2AG Cost(M$)", "Transition AG2Non-AG Cost(M$)",
                "AM Net Cost(M$)", "Non-AG Net Cost(M$)"]

