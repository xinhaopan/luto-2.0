import os
import subprocess
import sys
import shutil
import pandas as pd
from joblib import Parallel, delayed

def delete_path(path):
    """删除单个文件或文件夹"""
    try:
        if os.path.isfile(path) or os.path.islink(path):
            os.remove(path)
        elif os.path.isdir(path):
            shutil.rmtree(path)
        print(f"Deleted: {path}")
    except Exception as e:
        print(f"Error deleting {path}: {e}")

def delete_folder_multiprocessing(folder_path, num_processes=128):
    """使用多进程删除单个目录内容"""
    if not os.path.exists(folder_path):
        print(f"Path does not exist: {folder_path}")
        return

    # 列出所有文件和子目录
    entries = [os.path.join(folder_path, entry) for entry in os.listdir(folder_path)]

    # 使用多进程删除每个子文件或子目录
    with Pool(processes=num_processes) as pool:
        pool.map(delete_path, entries)

    # 删除空的根文件夹
    os.rmdir(folder_path)
    print(f"Folder '{folder_path}' has been completely deleted.")


def execute_script_in_directory(script_name, working_directory):
    """
    在指定的工作目录环境下运行 Python 脚本，并将输出存储到指定的文件中。

    参数:
        script_name (str): 要运行的脚本名称。
        working_directory (str): 指定的工作目录。
        output_directory (str): 输出日志存储目录。
    """
    print(f"{script_name}: running in {working_directory}")
    output_directory = os.path.join(working_directory, "output")
    # 确保输出目录存在
    os.makedirs(output_directory, exist_ok=True)

    # 输出日志文件路径
    stdout_path = os.path.join(output_directory, f"{script_name}_stdout.log")
    stderr_path = os.path.join(output_directory, f"{script_name}_stderr.log")

    try:
        with open(stdout_path, "w") as stdout_file, open(stderr_path, "w") as stderr_file:
            # 使用 subprocess 在指定目录下运行脚本，并将输出写入日志文件
            result = subprocess.run(
                [sys.executable, script_name],  # 使用当前 Python 解释器执行脚本
                cwd=working_directory,  # 指定运行时的工作目录
                stdout=stdout_file,  # 重定向标准输出
                stderr=stderr_file,  # 重定向错误输出
                text=True  # 输出解码为字符串
            )
        # 检查返回码，判断成功或失败
        if result.returncode == 0:
            print(f"{working_directory}: success")
        else:
            print(f"{working_directory}: failed with return code {result.returncode}")
    except Exception as e:
        print(f"Failed '{working_directory}': {e}")

def process_file_dir(file_dir, script_to_run):
    copy_folders(file_dir)
    """
    处理单个 file_dir，包括检查 PKL 文件和执行脚本
    :param file_dir: 要处理的文件夹路径
    :param script_to_run: 要执行的脚本路径
    """
    try:
        working_directory = os.path.join("output", file_dir)

        # 确保工作目录存在
        time_file_path = os.path.join(working_directory, 'output')
        if not os.path.exists(time_file_path):
            print(f"Time file path does not exist: {time_file_path}")
            return

        # 获取 time_file_dirs
        time_file_dirs = [name for name in os.listdir(time_file_path) if os.path.isdir(os.path.join(time_file_path, name))]

        for time_file_dir in time_file_dirs:
            if "2010-2050" in time_file_dir:
                pkl_path = os.path.join(time_file_path, time_file_dir, 'data_with_solution.pkl')

                # 如果 PKL 文件不存在，删除子目录
                if not os.path.exists(pkl_path):
                    print(f"PKL file does not exist at path: {pkl_path}")
                    # sdelete_folder_multiprocessing(os.path.join(time_file_path, time_file_dir))

        # 执行脚本
                execute_script_in_directory(script_to_run, working_directory)

    except Exception as e:
        print(f"Error processing file_dir '{file_dir}': {e}")

def copy_folder_custom(source, destination, ignore_dirs=None):
    ignore_dirs = set() if ignore_dirs is None else set(ignore_dirs)

    jobs = []
    os.makedirs(destination, exist_ok=True)
    for item in os.listdir(source):

        if item in ignore_dirs: continue

        s = os.path.join(source, item)
        d = os.path.join(destination, item)
        jobs += copy_folder_custom(s, d) if os.path.isdir(s) else [(s, d)]

    return jobs
def copy_folders(col,worker=3):
    SOURCE_DIR=os.getcwd()
    # Copy codes to the each custom run folder, excluding {EXCLUDE_DIRS} directories
    EXCLUDE_DIRS = ['input', 'output', '.git', '.vscode', '__pycache__', 'jinzhu_inspect_code', 'myCode','luto/settings.py']
    from_to_files = copy_folder_custom(SOURCE_DIR, f'{SOURCE_DIR}/output/{col}', EXCLUDE_DIRS)
    for s, d in from_to_files:
        if not os.path.exists(s):
            print(f"Source file not found: {s}")
        if not os.path.exists(os.path.dirname(d)):
            print(f"Destination directory does not exist: {os.path.dirname(d)}")

    Parallel(n_jobs=worker, backend="threading")(delayed(shutil.copy2)(s, d) for s, d in from_to_files)
    # Create an output folder for the task
    os.makedirs(f'{SOURCE_DIR}/output/{col}/output', exist_ok=True)

if __name__ == "__main__":
    csv_path = "myCode/tasks_run/Custom_runs/setting_template_windows_100125.csv"
    df = pd.read_csv(csv_path)
    file_dirs = df.columns[2:]

    results = Parallel(n_jobs=3)(
        delayed(process_file_dir)(file_dir, "1_write.py") for file_dir in file_dirs
    )


