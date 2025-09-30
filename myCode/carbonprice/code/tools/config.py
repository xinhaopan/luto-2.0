import math

N_JOBS = math.ceil(41 / 1)
TASK_NAME = "20250926_Paper2_Results_HPC" # "20250922_Paper2_Results_NCI"  #

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
    "xr_cost_ag": "Ag cost",
    "xr_cost_agricultural_management": "AgMgt cost",
    "xr_cost_non_ag": "Non-ag cost",
    "xr_cost_transition_ag2ag": "Transition(ag→ag) cost",
    "xr_transition_cost_ag2non_ag": "Transition(ag→non-ag) cost",
    "xr_transition_cost_ag2non_ag_amortised": "Transition(ag→non-ag) amortised cost",
    "xr_revenue_ag": "Ag revenue",
    "xr_revenue_agricultural_management": "AgMgt revenue",
    "xr_revenue_non_ag": "Non-ag revenue",

    # GHG Files
    "xr_GHG_ag": "Ag GHG",
    "xr_GHG_ag_management": "Agmgt GHG",
    "xr_GHG_non_ag": "Non-ag GHG",
    "xr_transition_GHG": "Transition GHG",
    "xr_GHG_ag_diff": "Ag GHG",

    # Biodiversity Files
    "xr_biodiversity_GBF2_priority_ag": "Ag biodiversity",
    "xr_biodiversity_GBF2_priority_ag_management": "AgMgt biodiversity",
    "xr_biodiversity_GBF2_priority_non_ag": "Non-ag biodiversity",
    "xr_biodiversity_GBF2_priority_ag_diff": "Ag biodiversity",

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


economic_files = ['xr_cost_ag', 'xr_cost_agricultural_management', 'xr_cost_non_ag', 'xr_cost_transition_ag2ag','xr_transition_cost_ag2non_ag','xr_transition_cost_ag2non_ag_amortised','xr_revenue_ag', 'xr_revenue_agricultural_management', 'xr_revenue_non_ag']
carbon_files = ['xr_GHG_ag', 'xr_GHG_ag_management', 'xr_GHG_non_ag', 'xr_transition_GHG']
bio_files = ['xr_biodiversity_GBF2_priority_ag', 'xr_biodiversity_GBF2_priority_ag_management','xr_biodiversity_GBF2_priority_non_ag']

input_files_0 = ['Run_18_GHG_off_BIO_off_CUT_50']
input_files_1 = ['Run_12_GHG_low_BIO_off_CUT_50', 'Run_06_GHG_high_BIO_off_CUT_50']
input_files_2 = [
    'Run_11_GHG_low_BIO_high_CUT_10',  'Run_10_GHG_low_BIO_high_CUT_20',  'Run_09_GHG_low_BIO_high_CUT_30',  'Run_08_GHG_low_BIO_high_CUT_40',  'Run_07_GHG_low_BIO_high_CUT_50',
    'Run_05_GHG_high_BIO_high_CUT_10', 'Run_04_GHG_high_BIO_high_CUT_20', 'Run_03_GHG_high_BIO_high_CUT_30', 'Run_02_GHG_high_BIO_high_CUT_40', 'Run_01_GHG_high_BIO_high_CUT_50'
]
input_files = input_files_0 + input_files_1 + input_files_2

carbon_names = ['carbon_low', 'carbon_high']
carbon_bio_names = [
        'carbon_low_bio_10', 'carbon_low_bio_20', 'carbon_low_bio_30', 'carbon_low_bio_40', 'carbon_low_bio_50',
        'carbon_high_bio_10', 'carbon_high_bio_20', 'carbon_high_bio_30', 'carbon_high_bio_40', 'carbon_high_bio_50'
    ]
counter_carbon_bio_names = ['Counterfactual_carbon_low_bio_10', 'Counterfactual_carbon_low_bio_20', 'Counterfactual_carbon_low_bio_30', 'Counterfactual_carbon_low_bio_40', 'Counterfactual_carbon_low_bio_50',
        'Counterfactual_carbon_high_bio_10', 'Counterfactual_carbon_high_bio_20', 'Counterfactual_carbon_high_bio_30', 'Counterfactual_carbon_high_bio_40', 'Counterfactual_carbon_high_bio_50'
    ]

PRICE_TITLE_MAP = {'carbon_low': r'Reference→$\mathrm{GHG}_{\mathrm{low}}$',
                'carbon_high': r'Reference→$\mathrm{GHG}_{\mathrm{high}}$',

                'carbon_low_bio_10': r'$\mathrm{GHG}_{\mathrm{low}}$→$\mathrm{GHG}_{\mathrm{low}}$,$\mathrm{Bio}_{\mathrm{10}}$',
                'carbon_low_bio_20': r'$\mathrm{GHG}_{\mathrm{low}}$→$\mathrm{GHG}_{\mathrm{low}}$,$\mathrm{Bio}_{\mathrm{20}}$',
                'carbon_low_bio_30': r'$\mathrm{GHG}_{\mathrm{low}}$→$\mathrm{GHG}_{\mathrm{low}}$,$\mathrm{Bio}_{\mathrm{30}}$',
                'carbon_low_bio_40': r'$\mathrm{GHG}_{\mathrm{low}}$→$\mathrm{GHG}_{\mathrm{low}}$,$\mathrm{Bio}_{\mathrm{40}}$',
                'carbon_low_bio_50': r'$\mathrm{GHG}_{\mathrm{low}}$→$\mathrm{GHG}_{\mathrm{low}}$,$\mathrm{Bio}_{\mathrm{50}}$',
                'carbon_high_bio_10': r'$\mathrm{GHG}_{\mathrm{high}}$→$\mathrm{GHG}_{\mathrm{high}}$,$\mathrm{Bio}_{\mathrm{10}}$',
                'carbon_high_bio_20': r'$\mathrm{GHG}_{\mathrm{high}}$→$\mathrm{GHG}_{\mathrm{high}}$,$\mathrm{Bio}_{\mathrm{20}}$',
                'carbon_high_bio_30': r'$\mathrm{GHG}_{\mathrm{high}}$→$\mathrm{GHG}_{\mathrm{high}}$,$\mathrm{Bio}_{\mathrm{30}}$',
                'carbon_high_bio_40': r'$\mathrm{GHG}_{\mathrm{high}}$→$\mathrm{GHG}_{\mathrm{high}}$,$\mathrm{Bio}_{\mathrm{40}}$',
                'carbon_high_bio_50': r'$\mathrm{GHG}_{\mathrm{high}}$→$\mathrm{GHG}_{\mathrm{high}}$,$\mathrm{Bio}_{\mathrm{50}}$',

                'Counterfactual_carbon_low_bio_10': r'Reference→$\mathrm{GHG}_{\mathrm{low}}$,$\mathrm{Bio}_{\mathrm{10}}$',
                'Counterfactual_carbon_low_bio_20': r'Reference→$\mathrm{GHG}_{\mathrm{low}}$,$\mathrm{Bio}_{\mathrm{20}}$',
                'Counterfactual_carbon_low_bio_30': r'Reference→$\mathrm{GHG}_{\mathrm{low}}$,$\mathrm{Bio}_{\mathrm{30}}$',
                'Counterfactual_carbon_low_bio_40': r'Reference→$\mathrm{GHG}_{\mathrm{low}}$,$\mathrm{Bio}_{\mathrm{40}}$',
                'Counterfactual_carbon_low_bio_50': r'Reference→$\mathrm{GHG}_{\mathrm{low}}$,$\mathrm{Bio}_{\mathrm{50}}$',
                'Counterfactual_carbon_high_bio_10': r'Reference→$\mathrm{GHG}_{\mathrm{high}}$,$\mathrm{Bio}_{\mathrm{10}}$',
                'Counterfactual_carbon_high_bio_20': r'Reference→$\mathrm{GHG}_{\mathrm{high}}$,$\mathrm{Bio}_{\mathrm{20}}$',
                'Counterfactual_carbon_high_bio_30': r'Reference→$\mathrm{GHG}_{\mathrm{high}}$,$\mathrm{Bio}_{\mathrm{30}}$',
                'Counterfactual_carbon_high_bio_40': r'Reference→$\mathrm{GHG}_{\mathrm{high}}$,$\mathrm{Bio}_{\mathrm{40}}$',
                'Counterfactual_carbon_high_bio_50': r'Reference→$\mathrm{GHG}_{\mathrm{high}}$,$\mathrm{Bio}_{\mathrm{50}}$',
                }

ORIGINAL_TITLE_MAP = {'Run_18_GHG_off_BIO_off_CUT_50': 'Reference',
                'Run_12_GHG_low_BIO_off_CUT_50': r'$\mathrm{GHG}_{\mathrm{low}}$',
                'Run_06_GHG_high_BIO_off_CUT_50': r'$\mathrm{GHG}_{\mathrm{high}}$',

                'Run_11_GHG_low_BIO_high_CUT_10': r'$\mathrm{GHG}_{\mathrm{low}}$,$\mathrm{Bio}_{\mathrm{10}}$',
                'Run_10_GHG_low_BIO_high_CUT_20': r'$\mathrm{GHG}_{\mathrm{low}}$,$\mathrm{Bio}_{\mathrm{20}}$',
                'Run_09_GHG_low_BIO_high_CUT_30': r'$\mathrm{GHG}_{\mathrm{low}}$,$\mathrm{Bio}_{\mathrm{30}}$',
                'Run_08_GHG_low_BIO_high_CUT_40': r'$\mathrm{GHG}_{\mathrm{low}}$,$\mathrm{Bio}_{\mathrm{40}}$',
                'Run_07_GHG_low_BIO_high_CUT_50': r'$\mathrm{GHG}_{\mathrm{low}}$,$\mathrm{Bio}_{\mathrm{50}}$',

                'Run_05_GHG_high_BIO_high_CUT_10': r'$\mathrm{GHG}_{\mathrm{high}}$,$\mathrm{Bio}_{\mathrm{10}}$',
                'Run_04_GHG_high_BIO_high_CUT_20': r'$\mathrm{GHG}_{\mathrm{high}}$,$\mathrm{Bio}_{\mathrm{20}}$',
                'Run_03_GHG_high_BIO_high_CUT_30': r'$\mathrm{GHG}_{\mathrm{high}}$,$\mathrm{Bio}_{\mathrm{30}}$',
                'Run_02_GHG_high_BIO_high_CUT_40': r'$\mathrm{GHG}_{\mathrm{high}}$,$\mathrm{Bio}_{\mathrm{40}}$',
                'Run_01_GHG_high_BIO_high_CUT_50': r'$\mathrm{GHG}_{\mathrm{high}}$,$\mathrm{Bio}_{\mathrm{50}}$',
                }
