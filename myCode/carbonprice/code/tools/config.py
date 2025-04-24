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

time = '20250424'
suffix = '0_20'

senerios = [
    'Run_1_on_on',
    'Run_2_on_off',
    'Run_3_off_on',
    'Run_4_off_off',
]
input_files = [
    f"{time}_{senerio}_{suffix}"
    for senerio in senerios
]

