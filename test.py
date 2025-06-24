import os

# 指定你要遍历的顶级目录
root_dir = 'output'


# 步骤1：收集所有需要重命名的文件路径
rename_tasks = []
for dirpath, dirnames, filenames in os.walk(root_dir):
    for filename in filenames:
        full_path = os.path.join(dirpath, filename)
        if 'data_for_carbon_price' in full_path and 'non_ag' in full_path:
            new_path = full_path.replace('non_ag', 'non-ag')
            rename_tasks.append((full_path, new_path))

# 步骤2：统一重命名
for src, dst in rename_tasks:
    new_dir = os.path.dirname(dst)
    os.makedirs(new_dir, exist_ok=True)
    print(f'Renaming: {src} -> {dst}')
    if os.path.exists(dst):
        print(f'File already exists: {dst}, skipping.')
        continue
    try:
        os.rename(src, dst)
    except FileNotFoundError:
        print(f'File not found: {src}, skipping.')