import os
from joblib import Parallel, delayed
import dill
import gzip
import shutil
import subprocess
import re
from collections import defaultdict
from tqdm import tqdm
import time

def tprint(*args, **kwargs):
    """打印带时间戳的内容，自动换行，支持所有print参数。"""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print(f"{timestamp}   ", *args, **kwargs)


def dir_has_target_file(dir_path, target_filename):
    """
    判断该目录及其所有子目录下是否有目标文件
    """
    for root, dirs, files in os.walk(dir_path):
        if target_filename in files:
            return os.path.join(root, target_filename)
    return None


def find_data_with_solution_all_subdirs(task_root_dir, n_jobs=3):
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

def process_target(run_path, script_path, gz_path):
    tprint(f"PID {os.getpid()} is working on {run_path}")
    update_luto_code(run_path)
    try:
        with open(os.path.join(run_path, 'write_stdout.log'), 'a') as std_file, \
             open(os.path.join(run_path, 'write_stderr.log'), 'a') as err_file:
            subprocess.run(
                ['python', script_path, gz_path],
                cwd=run_path,
                stdout=std_file,
                stderr=err_file,
                check=True
            )
        tprint(f"Success: PID {os.getpid()} {run_path}")
        return gz_path, "success"
    except Exception as e:
        tprint(f"Failed: PID {os.getpid()} {run_path}, Error: {e}")
        return gz_path, f"failed: {e}"

def update_luto_code(run_path):
    """
    根据 run_path 自动推断 src_dir，并将 src_dir 下除了 settings.py 以外的内容复制到 run_path/luto
    """
    # 找到 LUTO2 路径
    abs_run_path = os.path.abspath(run_path)
    luto2_dir = abs_run_path.split('output')[0].rstrip(os.sep)  # 取 output 之前的路径
    src_dir = os.path.join(luto2_dir, 'luto')
    dst_dir = os.path.join(run_path, 'luto')

    if not os.path.exists(src_dir):
        raise FileNotFoundError(f"源代码目录不存在: {src_dir}")
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)

    for item in os.listdir(src_dir):
        src_item = os.path.join(src_dir, item)
        dst_item = os.path.join(dst_dir, item)
        if item == 'settings.py':
            continue
        if os.path.isdir(src_item):
            # 如果目标目录存在，先删除
            if os.path.exists(dst_item):
                shutil.rmtree(dst_item)
            shutil.copytree(src_item, dst_item)
        else:
            shutil.copy2(src_item, dst_item)

def write_repeat(task_root_dir,n_jobs=9):
    found, not_found = find_data_with_solution_all_subdirs(task_root_dir, n_jobs)
    tprint("有解:")
    for p in found:
        tprint(p)
    tprint("\n无解:")
    for d in not_found:
        tprint(d)

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
            tprint("没有html 的目录:", folder_path)


    # Group directories by run_path and process efficiently
    run_path_to_targets = defaultdict(list)
    for target_dir in without_htmls:
        norm_path = os.path.normpath(target_dir)
        parts = norm_path.split(os.sep)
        run_path = os.sep.join(parts[:5])
        run_path_to_targets[run_path].append(norm_path)

    tprint("\nStart write output.......")
    all_jobs = []
    for run_path, targets in run_path_to_targets.items():
        # 修改 settings.py 和写 script
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

        script_name = 'write_output.py'
        script_path = os.path.join(run_path, script_name)
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
        with open(script_path, 'w', encoding='utf-8') as f:
            f.write(script_content)

        for target_dir in targets:
            parts = target_dir.split(os.sep)
            gz_path = os.path.join(os.sep.join(parts[5:]), 'data_with_solution.gz')
            all_jobs.append((run_path, script_name, gz_path))

    success_list, failed_list = [], []
    results = []

    with tqdm(total=len(all_jobs), desc="Processing all targets") as pbar:
        parallel = Parallel(n_jobs=n_jobs, batch_size=1)
        tasks = (delayed(process_target)(run_path, script_name, gz_path)
                 for run_path, script_name, gz_path in all_jobs)
        for result in parallel(tasks):
            results.append(result)
            run_path, gz_path, status = result
            if status == "success":
                success_list.append(run_path)
            else:
                failed_list.append((run_path, gz_path, status))
            pbar.set_postfix(success=len(success_list), failed=len(failed_list))
            pbar.update()

    tprint("\n--- Success ---")
    for path in success_list:
        tprint(path)

    tprint("\n--- Failed ---")
    for path, error in failed_list:
        tprint(path, error)

if __name__ == "__main__":
    # Main execution
    import time
    # time.sleep(60*60*5)
    task_root_dir = '../../../output/20250608_Paper1_results_windows_BIO3'
    write_repeat(task_root_dir)




