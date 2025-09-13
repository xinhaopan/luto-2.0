import math

N_JOBS = math.ceil(41 / 41)
TASK_NAME = "20250908_Paper2_Results_NCI"



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
    "xr_cost_ag": "Ag cost(M$)",
    "xr_cost_agricultural_management": "Agmgt cost(M$)",
    "xr_cost_non_ag": "Non-ag cost(M$)",
    "xr_cost_transition_ag2ag": "Transition(ag→ag) cost(M$)",
    "xr_transition_cost_ag2non_ag": "Transition(ag→non-ag) cost(M$)",
    "xr_transition_cost_ag2non_ag_amortised": "Transition(ag→non-ag) amortised cost(M$)",
    "xr_revenue_ag": "Ag revenue(M$)",
    "xr_revenue_agricultural_management": "Agmgt revenue(M$)",
    "xr_revenue_non_ag": "Non-ag revenue(M$)",

    # GHG Files
    "xr_GHG_ag": "Ag GHG reductions (MtCO2e)",
    "xr_GHG_ag_management": "Agmgt GHG reductions and removals (MtCO2e)",
    "xr_GHG_non_ag": "Non-ag GHG removals (MtCO2e)",
    "xr_transition_GHG": "Transition GHG reductions (MtCO2e)",

    # Biodiversity Files
    "xr_biodiversity_GBF2_priority_ag": "Ag biodiversity restoration (Mha)",
    "xr_biodiversity_GBF2_priority_ag_management": "AM biodiversity restoration (Mha)",
    "xr_biodiversity_GBF2_priority_non_ag": "Non-ag biodiversity restoration (Mha)",


    "ag_profit": "Ag profit(M$)",
    "am_profit": "AM profit(M$)",
    "non-ag_profit": "Non-ag profit(M$)",
    "transition(ag2ag)_profit": "Transition(ag→ag) profit(M$)",
    "transition(ag2non-ag)_profit": "Transition(ag2non→ag) profit(M$)",

}



# 创建 revenue_dict，主键从 cost 换成 revenue
REVENUE_DICT = {key.replace('cost', 'revenue'): value for key, value in COST_DICT.items()}


NAME_DICT = {
    "cp": {"title": "Carbon Price", "unit": "(AU$/tCO2e)"},
    "bp": {"title": "Biodiversity Price", "unit": "(AU$/ha)"},
    "ghg": {"title": "GHG Emissions", "unit": "(Mt CO2e)"},
    "cost": {"title": "Cost", "unit": "(MAU$)"},
}

START_YEAR = 2025
COLUMN_NAME = ["Ag", "AM",  "Non-ag", "Transition(ag2ag)","Transition(ag2non-ag)"]


