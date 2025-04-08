import os
import re
import shutil
import pandas as pd
from tools.data_helper import *

# 指定目标目录路径
directory = '../../../output'  # 替换为你的目标目录路径

# 获取目录下的所有文件名
file_names = os.listdir(directory)

# 提取包含特定模式的文件名
pattern = re.compile(r'20250404_Run_\d+_GHG_1_8C_67_BIO_5')
filtered_file_names = [file for file in file_names if pattern.search(file)]

# 执行清理任务
for file in filtered_file_names:
    path = os.path.join(get_path(file))

    for item in os.listdir(path):
        item_path = os.path.join(path, item)

        # 跳过 DATA_REPORT 文件夹
        if item == 'DATA_REPORT':
            continue

        try:
            if os.path.isfile(item_path) or os.path.islink(item_path):
                os.remove(item_path)
            elif os.path.isdir(item_path):
                shutil.rmtree(item_path)
        except Exception as e:
            print(f"⚠️ 删除失败: {item_path}，原因：{e}")
