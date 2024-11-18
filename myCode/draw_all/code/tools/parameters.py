INPUT_DIR = '../../../input'
input_files = [
    '2_BIO_0_GHG_1_8C_67',
    '1_BIO_0_GHG_1_5C_50',
    '0_BIO_0_GHG_1_5C_67',
    '5_BIO_0_3_GHG_1_8C_67',
    '4_BIO_0_3_GHG_1_5C_50',
    '3_BIO_0_3_GHG_1_5C_67',
    '8_BIO_0_5_GHG_1_8C_67',
    '7_BIO_0_5_GHG_1_5C_50',
    '6_BIO_0_5_GHG_1_5C_67',
]
input_files = [f"{file}_2" for file in input_files]

# 子任务及对应颜色配置文件
sub_tasks = [
    ('lumap_2050', 'ag_group_map'),
    ('ammap_2050', 'am'),
    ('non_ag_2050', 'non_ag')
]

# 生成任务列表
tasks = [(main_task, sub_task[0], sub_task[1]) for main_task in input_files for sub_task in sub_tasks]