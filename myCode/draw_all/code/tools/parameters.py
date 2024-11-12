INPUT_DIR = '../../../input'
input_files = [
    '2_BIO_0_GHG_1_8C_67_1',
    '1_BIO_0_GHG_1_5C_50_1',
    '0_BIO_0_GHG_1_5C_67_1',
    '5_BIO_0_3_GHG_1_8C_67_1',
    '4_BIO_0_3_GHG_1_5C_50_1',
    '3_BIO_0_3_GHG_1_5C_67_1',
    '8_BIO_0_5_GHG_1_8C_67',
    '7_BIO_0_5_GHG_1_5C_50',
    '6_BIO_0_5_GHG_1_5C_67',
]

# 子任务及对应颜色配置文件
sub_tasks = [
    ('lumap_2050', 'ag_group_map'),
    ('ammap_2050', 'am'),
    ('non_ag_2050', 'non_ag')
]

# 生成任务列表
tasks = [(main_task, sub_task[0], sub_task[1]) for main_task in input_files for sub_task in sub_tasks]