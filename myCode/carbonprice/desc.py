cost_dict = {
    'cost_am': [
        'Asparagopsis taxiformis',
        'Precision Agriculture',
        'Ecological Grazing',
        'Savanna Burning',
        'AgTech EI'
    ],
    'cost_non_ag': [
        'Environmental Plantings',
        'Riparian Plantings',
        'Sheep Agroforestry',
        'Beef Agroforestry',
        'Carbon Plantings (Block)',
        'Sheep Carbon Plantings (Belt)',
        'Beef Carbon Plantings (Belt)',
        'BECCS'
    ]
}

# 创建 revenue_dict，主键从 cost 换成 revenue
revenue_dict = {key.replace('cost', 'revenue'): value for key, value in cost_dict.items()}
