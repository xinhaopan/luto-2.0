# 主任务列表
main_tasks = [
    '0_BIO_0_GHG_1_5C_67',
    '1_BIO_0_GHG_1_5C_50',
    '2_BIO_0_GHG_1_8C_67',
    '3_BIO_0_3_GHG_1_5C_67',
    '4_BIO_0_3_GHG_1_5C_50',
    '5_BIO_0_3_GHG_1_8C_67',
    '6_BIO_0_5_GHG_1_5C_67',
    '7_BIO_0_5_GHG_1_5C_50',
    '8_BIO_0_5_GHG_1_8C_67'
]

# 子任务及对应颜色配置文件
sub_tasks = [
    ('lumap_2050', 'lumap_colors_grouped.csv'),
    ('ammap_2050', 'ammap_colors.csv'),
    ('non_ag_2050', 'non_ag_colors.csv')
]

# 生成任务列表
tasks = [(main_task, sub_task[0], sub_task[1]) for main_task in main_tasks for sub_task in sub_tasks]
