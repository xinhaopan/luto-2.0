INPUT_DIR = '../../../input'

time = ''
middle = ''
suffix = 'test'
senerios = [
    '20241213_Run_7_GHG_1_8C_67_BIO_0',
    '20241213_Run_4_GHG_1_5C_50_BIO_0',
    '20241213_Run_1_GHG_1_5C_67_BIO_0',
    '20241213_Run_5_GHG_1_5C_50_BIO_3',
    '20241213_Run_8_GHG_1_8C_67_BIO_3',
    '20241213_Run_2_GHG_1_5C_67_BIO_3',
    '20241213_Run_6_GHG_1_5C_50_BIO_5',
    '20241213_Run_9_GHG_1_8C_67_BIO_5',
    '20241213_Run_3_GHG_1_5C_67_BIO_5',
]

input_files = [
    f"{time}_{middle}_{senerio}_{suffix}".strip('_').replace('__', '_')
    for senerio in senerios
]

# 子任务及对应颜色配置文件
sub_tasks = [
    ('lumap_2050', 'ag_group_map'),
    ('ammap_2050', 'am'),
    ('non_ag_2050', 'non_ag')
]

# 生成任务列表
tasks = [(main_task, sub_task[0], sub_task[1]) for main_task in input_files for sub_task in sub_tasks]