import os
from joblib import Parallel, delayed
import dill
import gzip
import subprocess
import re
from collections import defaultdict


def dir_has_target_file(dir_path, target_filename):
    """
    判断该目录及其所有子目录下是否有目标文件
    """
    for root, dirs, files in os.walk(dir_path):
        if target_filename in files:
            return os.path.join(root, target_filename)
    return None


def find_data_with_solution_all_subdirs(task_root_dir, n_jobs=4):
    """
    递归查找task_root_dir下所有子目录中data_with_solution.gz文件。
    返回：(所有找到的完整路径列表，所有完全未找到该文件的一级子目录的名称列表)
    """
    files = [f for f in os.listdir(task_root_dir)
             if os.path.isdir(os.path.join(task_root_dir, f))]
    dirs = [os.path.join(task_root_dir, f) for f in files if os.path.isdir(os.path.join(task_root_dir, f))]

    results = Parallel(n_jobs=n_jobs)(
        delayed(dir_has_target_file)(d, 'data_with_solution.gz') for d in dirs
    )
    found_paths = [res for res in results if res]
    not_found_dirs = [files[i] for i, res in enumerate(results) if not res]

    return found_paths, not_found_dirs


# Main execution
task_root_dir = '../../output/Paper1_results_test'
found, not_found = find_data_with_solution_all_subdirs(task_root_dir, n_jobs=8)
print("有解:")
for p in found:
    print(p)
print("\n无解:")
for d in not_found:
    print(d)

# Identify directories with and without production.html
with_htmls = []
without_htmls = []
for file_path in found:
    folder_path = os.path.dirname(file_path)
    html_path = os.path.join(folder_path, 'DATA_REPORT', 'REPORT_HTML', 'pages', 'production.html')
    if os.path.exists(html_path):
        with_htmls.append(folder_path)
    else:
        without_htmls.append(folder_path)

# Group directories by run_path and process efficiently
run_path_to_targets = defaultdict(list)
for target_dir in without_htmls:
    norm_path = os.path.normpath(target_dir)
    parts = norm_path.split(os.sep)
    run_path = os.sep.join(parts[:5])
    run_path_to_targets[run_path].append(norm_path)

for run_path, targets in run_path_to_targets.items():
    # Modify settings.py once per run_path
    settings_path = os.path.join(run_path, 'luto', 'settings.py')
    with open(settings_path, 'r', encoding='utf-8') as f:
        content = f.read()
    new_content = re.sub(
        r'INPUT_DIR\s*=\s*(?:r)?[\'"].*?[\'"]',
        "INPUT_DIR = '../../../input'",
        content
    )
    with open(settings_path, 'w', encoding='utf-8') as f:
        f.write(new_content)

    # Write a reusable write_output.py script
    script_content = '''import sys
import gzip
import dill
from luto.tools.write import write_outputs
import luto.settings as settings

gz_path = sys.argv[1]
with gzip.open(gz_path, 'rb') as f:
    data = dill.load(f)
write_outputs(data)
'''
    script_path = os.path.join(run_path, 'write_output.py')
    with open(script_path, 'w', encoding='utf-8') as f:
        f.write(script_content)
    print("Start processing run_path:", run_path)
    # Execute for each target_dir under this run_path
    for target_dir in targets:
        parts = target_dir.split(os.sep)
        gz_path = os.path.join(os.sep.join(parts[5:]), 'data_with_solution.gz')
        with open(os.path.join(run_path, 'write_stdout.log'), 'a') as std_file, \
                open(os.path.join(run_path, 'write_stderr.log'), 'a') as err_file:
            subprocess.run(
                ['python', 'write_output.py', gz_path],
                cwd=run_path,
                stdout=std_file,
                stderr=err_file,
                check=True
            )