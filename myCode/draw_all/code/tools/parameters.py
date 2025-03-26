INPUT_DIR = '../../../input'

time = '20250324'
middle = ''
suffix = '5'
# senerios = [
#     'Run_16_GHG_1_8C_67_BIO_0',
#     'Run_13_GHG_1_5C_50_BIO_0',
#     'Run_10_GHG_1_5C_67_BIO_0',
#     'Run_17_GHG_1_8C_67_BIO_3',
#     'Run_14_GHG_1_5C_50_BIO_3',
#     'Run_11_GHG_1_5C_67_BIO_3',
#     'Run_18_GHG_1_8C_67_BIO_5',
#     'Run_15_GHG_1_5C_50_BIO_5',
#     'Run_12_GHG_1_5C_67_BIO_5',
# ]

run_number_origin = [7,4,1,8,5,2,9,6,3]
run_number = [num for num in run_number_origin]
senerios_origin = [
    'GHG_1_8C_67_BIO_0',
    'GHG_1_5C_50_BIO_0',
    'GHG_1_5C_67_BIO_0',
    'GHG_1_8C_67_BIO_3',
    'GHG_1_5C_50_BIO_3',
    'GHG_1_5C_67_BIO_3',
    'GHG_1_8C_67_BIO_5',
    'GHG_1_5C_50_BIO_5',
    'GHG_1_5C_67_BIO_5',
]
senerios = [f"Run_{num}_{senerio}" for num, senerio in zip(run_number, senerios_origin)]
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

COLUMN_WIDTH = 0.8
X_OFFSET = 2
font_size = 25
axis_linewidth = 2