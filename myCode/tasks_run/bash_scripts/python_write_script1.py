import os
import sys
import shutil
import gzip
import dill
import re
from collections import defaultdict
import time

def tprint(*args, **kwargs):
    """打印带时间戳的内容，自动换行，支持所有print参数。"""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print(f"{timestamp}   ", *args, **kwargs)

def dir_has_target_file(dir_path, target_filename):
    """判断该目录及其所有子目录下是否有目标文件"""
    for root, dirs, files in os.walk(dir_path):
        if target_filename in files:
            return os.path.join(root, target_filename)
    return None

def find_data_with_solution_all_subdirs(task_root_dir):
    """递归查找task_root_dir下所有子目录中data_with_solution.gz文件。"""
    files = [f for f in os.listdir(task_root_dir)
             if os.path.isdir(os.path.join(task_root_dir, f))]
    dirs = [os.path.join(task_root_dir, f) for f in files if os.path.isdir(os.path.join(task_root_dir, f))]

    found_paths = []
    not_found_dirs = []
    for i, d in enumerate(dirs):
        res = dir_has_target_file(d, 'data_with_solution.gz')
        if res:
            found_paths.append(res)
        else:
            not_found_dirs.append(files[i])
    return found_paths, not_found_dirs

def update_luto_code(run_path):
    """同步luto代码，排除settings.py"""
    abs_run_path = os.path.abspath(run_path)
    luto2_dir = abs_run_path.split('output')[0].rstrip(os.sep)
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
            if os.path.exists(dst_item):
                shutil.rmtree(dst_item)
            shutil.copytree(src_item, dst_item)
        else:
            shutil.copy2(src_item, dst_item)

def do_write_output(target_dir, gz_path):
    tprint(f"处理: {target_dir}")
    try:
        sys.path.insert(0, target_dir)
        import luto.settings as settings
        sys.path.pop(0)
        from luto.tools.write import write_outputs
        with gzip.open(gz_path, 'rb') as f:
            data = dill.load(f)
        write_outputs(data)
        tprint(f"写出成功: {target_dir}")
        return "success"
    except Exception as e:
        tprint(f"写出失败: {target_dir}, 错误: {e}")
        return f"failed: {e}"

def write_repeat(task_root_dir):
    found, not_found = find_data_with_solution_all_subdirs(task_root_dir)
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

    run_path_to_targets = defaultdict(list)
    for target_dir in without_htmls:
        norm_path = os.path.normpath(target_dir)
        parts = norm_path.split(os.sep)
        # 按你自己的结构决定是5还是其它层
        run_path = os.sep.join(parts[:5])
        run_path_to_targets[run_path].append(norm_path)

    tprint("\nStart write output.......")
    for run_path, targets in run_path_to_targets.items():
        # 修正 settings.py
        settings_path = os.path.join(run_path, 'luto', 'settings.py')
        if os.path.exists(settings_path):
            with open(settings_path, 'r', encoding='utf-8') as f:
                content = f.read()
            new_content = re.sub(
                r'INPUT_DIR\s*=\s*(?:r)?[\'"].*?[\'"]',
                "INPUT_DIR = '../../../input'",
                content
            )
            with open(settings_path, 'w', encoding='utf-8') as f:
                f.write(new_content)

        # 同步 luto 代码
        update_luto_code(run_path)

        for target_dir in targets:
            gz_path = os.path.join(target_dir, 'data_with_solution.gz')
            do_write_output(target_dir, gz_path)

if __name__ == "__main__":
    # 用法示例：python python_writes_script.py /path/to/task_root
    if len(sys.argv) < 2:
        print("Usage: python python_writes_script.py <task_root_dir>")
        sys.exit(1)
    task_root_dir = sys.argv[1]
    write_repeat(task_root_dir)
