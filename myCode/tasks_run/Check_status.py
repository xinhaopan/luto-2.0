import os
import re
import pandas as pd
import argparse

def parse_filename(filename):
    # 匹配 GHG 是小数（支持带小数点）和 BIO 是整数
    match = re.search(r'Run_\d+_GHG_([\d.]+)_BIO_(\d+)', filename)
    if match:
        ghg_value = float(match.group(1))  # 例如 '0.2'
        bio_value = int(match.group(2))    # 例如 '30'
        return ghg_value, bio_value
    else:
        print(f"{filename} No match found.")
        return None, None

def get_first_subfolder(output_dir):
    # 获取 output/ 目录下的第一个子文件夹
    subfolders = [f for f in os.listdir(output_dir) if os.path.isdir(os.path.join(output_dir, f))]
    return subfolders[0] if subfolders else None

def check_status(output_dir, first_subfolder):
    # 检查第一个子文件夹中是否有 DATA_REPORT 文件夹
    if first_subfolder:
        data_report_path = os.path.join(output_dir, first_subfolder, 'DATA_REPORT')
        if os.path.exists(data_report_path) and os.path.isdir(data_report_path):
            return 'End'
    return 'Pending'

def get_endyear(output_dir, first_subfolder):
    # 从 begin_end_compare_* 文件夹中提取 Endyear
    if first_subfolder:
        subfolder_path = os.path.join(output_dir, first_subfolder)
        for folder in os.listdir(subfolder_path):
            if folder.startswith('begin_end_compare_'):
                parts = folder.split('_')
                if len(parts) >= 4:
                    return parts[-1]
    return None


# 获取指定目录下的 Run_*/output/ 目录
base_dir = '../../output/20250730_price_task'
df = pd.read_csv(os.path.join(base_dir, 'grid_search_template.csv'), index_col=0)
run_dirs = [os.path.join(base_dir, d) for d in df.columns if d.startswith('Run_')]
data = []

for run_dir in run_dirs:
    name = os.path.basename(run_dir)
    output_dir = os.path.join(run_dir, 'output')
    if not os.path.exists(output_dir):
        continue

    ghg, bio = parse_filename(name)
    if ghg is None or bio is None:
        continue

    first_subfolder = get_first_subfolder(output_dir)
    status = check_status(output_dir, first_subfolder)
    endyear = get_endyear(output_dir, first_subfolder) if status == 'End' else None

    data.append([name, ghg, bio, status, endyear])

# 创建表格并保存为 CSV
df = pd.DataFrame(data, columns=['name', 'GHG', 'BIO', 'Status', 'Endyear'])
df.to_csv(f'{base_dir}/output_table.csv', index=False)
print("表格已保存至 output_table.csv")