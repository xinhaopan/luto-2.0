from math import e
import os
import re
import shutil
import random
import itertools
import multiprocessing
import pandas as pd
import subprocess
import datetime
import time
import numpy as np

from joblib import delayed, Parallel

from myCode.tasks_run.tools.parameters import EXCLUDE_DIRS, PARAMS_NUM_AS_STR, PARAMS_TO_EVAL, TASK_ROOT_DIR,OUTPUT_DIR
from myCode.tasks_run.tools import calculate_total_cost
from luto import settings


def print_with_time(message):
    """
    打印带有时间戳的消息。
    """
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")

def create_settings_template(to_path: str = TASK_ROOT_DIR):
    # Save the settings template to the root task folder
    None if os.path.exists(to_path) else os.makedirs(to_path)

    # # Write the requirements to the task folder
    # conda_pkgs, pip_pkgs = get_requirements()
    # with open(f'{to_path}/requirements_conda.txt', 'w') as conda_f, \
    #         open(f'{to_path}/requirements_pip.txt', 'w') as pip_f:
    #     conda_f.write(conda_pkgs)
    #     pip_f.write(pip_pkgs)

    if os.path.exists(f'{to_path}/settings_template.csv'):
        print('settings_template.csv already exists! Skip creating a new one!')
    else:
        # Get the settings from luto.settings
        with open('../../luto/settings.py', 'r') as file:
            lines = file.readlines()

            # Regex patterns that matches variable assignments from settings
            parameter_reg = re.compile(r"^(\s*[A-Z].*?)\s*=")
            settings_order = [match[1].strip() for line in lines if (match := parameter_reg.match(line))]

            # Reorder the settings dictionary to match the order in the settings.py file
            settings_dict = {i: getattr(settings, i) for i in dir(settings) if i.isupper()}
            settings_dict = {i: settings_dict[i] for i in settings_order if i in settings_dict}

            # Add the NODE parameters
            settings_dict['NODE'] = 'Please specify the node name'
            settings_dict['MEM'] = 'auto'
            settings_dict['CPU_PER_TASK'] = settings_dict['THREADS']
            settings_dict['TIME'] = 'auto'
            settings_dict['JOB_NAME'] = 'auto'

        # Create a template for cutom settings
        settings_df = pd.DataFrame({k: [v] for k, v in settings_dict.items()}).T.reset_index()
        settings_df.columns = ['Name', 'Default_run']
        settings_df = settings_df.applymap(str)
        settings_df.to_csv(f'{to_path}/settings_template.csv', index=False)

def create_default_settings(to_path: str = TASK_ROOT_DIR, name="default"):
    # Save the settings template to the root task folder
    None if os.path.exists(to_path) else os.makedirs(to_path)


    if os.path.exists(f'{to_path}/settings_template0.csv'):
        print('settings_template.csv already exists! Skip creating a new one!')
    else:
        # Get the settings from luto.settings
        with open('../../luto/settings.py', 'r') as file:
            lines = file.readlines()

            # Regex patterns that matches variable assignments from settings
            parameter_reg = re.compile(r"^(\s*[A-Z].*?)\s*=")
            settings_order = [match[1].strip() for line in lines if (match := parameter_reg.match(line))]

            # Reorder the settings dictionary to match the order in the settings.py file
            settings_dict = {i: getattr(settings, i) for i in dir(settings) if i.isupper()}
            settings_dict = {i: settings_dict[i] for i in settings_order if i in settings_dict}

            # Add the NODE parameters
            settings_dict['NODE'] = 'Please specify the node name'
            settings_dict['MEM'] = 'auto'
            settings_dict['CPU_PER_TASK'] = settings_dict['THREADS']
            settings_dict['TIME'] = 'auto'
            settings_dict['JOB_NAME'] = 'auto'

        # Create a template for cutom settings
        settings_df = pd.DataFrame({k: [v] for k, v in settings_dict.items()}).T.reset_index()
        settings_df.columns = ['Name', 'Default_run']
        settings_df = settings_df.applymap(str)
        settings_df.to_csv(f'{to_path}/{name}.csv', index=False)
        df = pd.read_csv(f'{to_path}/{name}.csv')
        df["default"] = df.iloc[:, 1]
        df.to_csv(f'{to_path}/{name}.csv', index=False)



def process_column(col, custom_settings, num_task, cwd):
    os.chdir(cwd)
    """并行处理单个列的任务"""
    # 获取当前任务的自定义设置
    custom_dict = update_settings(custom_settings[col].to_dict(), num_task, col)
    # 检查列名是否有效，并报告更改后的设置
    check_settings_name(custom_settings, col)
    # 为每个任务创建文件夹
    create_run_folders(col)
    # 将自定义设置写入任务文件夹
    write_custom_settings(f'{OUTPUT_DIR}/{col}', custom_dict)
    # 如果系统为 Linux，更新线程设置并提交任务
    update_thread_settings(f'{OUTPUT_DIR}/{col}', custom_dict)
    if os.name == 'nt':
        run_task_windows(cwd, col)  # 执行任务
    elif os.name == 'posix':
        run_task_linux(cwd, col, custom_dict)  # 执行任务


def create_task_runs(from_path: str = f'{TASK_ROOT_DIR}/settings_template.csv',use_multithreading=True,  num_workers: int = 4):
    """读取设置模板文件并并行运行任务"""
    # 读取自定义设置文件
    if os.name == 'posix':
        calculate_total_cost(pd.read_csv(from_path))
    custom_settings = pd.read_csv(from_path, index_col=0)
    custom_settings = custom_settings.dropna(how='all', axis=1)

    # 将列名转换为有效的 Python 变量名
    custom_settings.columns = [format_name(col) for col in custom_settings.columns]

    # 处理需要评估的参数
    custom_settings = custom_settings.replace({'TRUE': 'True', 'FALSE': 'False'})
    custom_settings.loc[PARAMS_TO_EVAL] = custom_settings.loc[PARAMS_TO_EVAL].map(eval)

    # 获取列名列表，并排除 'Default_run'
    custom_cols = [col for col in custom_settings.columns if col not in ['Default_run']]
    num_task = len(custom_cols)

    if not custom_cols:
        raise ValueError('No custom settings found in the settings_template.csv file!')

    cwd = os.getcwd()
    if use_multithreading:
        with multiprocessing.Pool(processes=num_workers) as pool:
            pool.starmap(process_column, [(col, custom_settings, num_task, cwd) for col in custom_cols])
    else:
        for col in custom_cols:
            process_column(col, custom_settings, num_task, cwd)


def run_task_windows(cwd, col):
    print_with_time(f"{col}: running task for column...")
    start_time = time.time()  # 记录任务开始时间
    log_file = f'output/{col}/output/error_log.txt'  # 定义日志文件路径

    python_path = r'F:\xinhao\miniforge\envs\luto\python.exe'
    try:
        # 运行子进程，捕获标准输出和标准错误
        result = subprocess.run(
            [python_path, '0_runs.py'],
            cwd=f'output/{col}',
            capture_output=True,  # 捕获输出
            text=True  # 将输出转换为文本
        )

        end_time = time.time()  # 记录任务结束时间
        elapsed_time = (end_time - start_time) / 3600  # 计算用时，单位：小时

        # 检查子进程的返回码和错误输出
        if result.returncode == 0 and not result.stderr:
            # 如果成功运行，记录成功信息
            with open(log_file, 'a') as f:
                f.write(f"Success running temp_runs.py for {col}:\n")
                f.write(f"stdout:\n{result.stdout}\n")
            print_with_time(f"{col}: successfully completed. Elapsed time: {elapsed_time:.2f} h")
        else:
            # 如果运行失败，记录错误信息
            with open(log_file, 'a') as f:
                f.write(f"Error running temp_runs.py for {col}:\n")
                f.write(f"stdout:\n{result.stdout}\n")
                f.write(f"stderr:\n{result.stderr}\n")
            print_with_time(f"{col}: error occurred. Elapsed time: {elapsed_time:.2f} h")

    except Exception as e:
        # 捕获 Python 异常，并将其写入日志文件
        with open(log_file, 'a') as f:
            f.write(f"Exception occurred while running temp_runs.py for {col}:\n")
            f.write(f"{str(e)}\n")
        print_with_time(f"{col}: exception occurred during execution, see {log_file} for details.")




# Grid search to set grid search parameters
def create_grid_search_template(num_runs: int = 10):
    # Gird parameters for {AG_MANAGEMENTS} and {AG_MANAGEMENTS_REVERSIBLE}
    grid_am = {k: [True, False] for k in settings.AG_MANAGEMENTS}

    # Grid parameters for {NON_AG_LAND_USES} and {NON_AG_LAND_USES_REVERSIBLE}
    grid_non_ag = {k: [True, False] for k in settings.NON_AG_LAND_USES}

    # Grid parameters for {MODE}
    grid_mode = ['timeseries', 'snapshot']

    # Create grid search parameter space
    custom_settings = pd.read_csv(f'{TASK_ROOT_DIR}/settings_template.csv')
    custom_settings = custom_settings[['Name', 'Default_run']]

    seen_am = set()
    seen_non_ag = set()
    random_choices = num_runs // len(grid_mode)

    for idx, (mode, _) in enumerate(itertools.product(grid_mode, range(random_choices))):

        select_am = {key: random.choice(value) for key, value in grid_am.items()}
        select_non_ag = {key: random.choice(value) for key, value in grid_non_ag.items()}

        if str(select_am) in seen_am and str(select_non_ag) in seen_non_ag:
            continue

        seen_am.add(str(select_am))
        seen_non_ag.add(str(select_non_ag))

        custom_settings[f'run_{idx:02}'] = custom_settings['Default_run']
        custom_settings.loc[(custom_settings['Name'] == 'MODE'), f'run_{idx:02}'] = mode
        custom_settings.loc[(custom_settings['Name'] == 'AG_MANAGEMENTS'), f'run_{idx:02}'] = str(select_am)
        custom_settings.loc[(custom_settings['Name'] == 'AG_MANAGEMENTS_REVERSIBLE'), f'run_{idx:02}'] = str(select_am)
        custom_settings.loc[(custom_settings['Name'] == 'NON_AG_LAND_USES'), f'run_{idx:02}'] = str(select_non_ag)
        custom_settings.loc[(custom_settings['Name'] == 'NON_AG_LAND_USES_REVERSIBLE'), f'run_{idx:02}'] = str(
            select_non_ag)

    custom_settings = custom_settings[['Name', 'Default_run'] + sorted(custom_settings.columns[2:])]
    custom_settings.to_csv(f'{TASK_ROOT_DIR}/settings_template.csv', index=False)


def is_float(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def format_name(name):
    return re.sub(r'\W+', '_', name.strip())


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


def get_requirements():
    with open('requirements.txt', 'r') as file:
        requirements = file.read().splitlines()

    split_idx = requirements.index('# Below can only be installed with pip')

    conda_pkgs = " ".join(requirements[:split_idx])
    pip_pkgs = " ".join(requirements[split_idx + 1:])

    return conda_pkgs, pip_pkgs


def write_custom_settings(task_dir: str, settings_dict: dict):
    # Write the custom settings to the settings.py of each task
    with open(f'{task_dir}/luto/settings.py', 'w') as file, \
            open(f'{task_dir}/luto/settings_bash.py', 'w') as bash_file:
        for k, v in settings_dict.items():
            k = k.replace(' ', '_').replace('(', '').replace(')', '')
            # List values need to be converted to bash arrays
            if isinstance(v, list):
                bash_file.write(f'{k}=({" ".join([str(elem) for elem in v])})\n')
                file.write(f'{k}={v}\n')
            elif k == 'SSP':
                v = str(v)
                file.write(f'{k}="{v}"\n')
                bash_file.write(f'{k}="{v}"\n')
            elif k == 'CARBON_PRICES_FIELD':
                # print(f'{k}="{v}"\n')
                v = str(v)
                file.write(f'{k}="{v}"\n')
                bash_file.write(f'{k}="{v}"\n')
            # Dict values need to be converted to bash variables
            elif isinstance(v, dict):
                file.write(f'{k}={v}\n')
                bash_file.write(f'# {k} is a dictionary, which is not natively supported in bash\n')
                for key, value in v.items():
                    key = str(key).replace(' ', '_').replace('(', '').replace(')', '')
                    bash_file.write(f'{k}_{key}={value}\n')
            # If the value is a number, write it as number
            elif str(v).isdigit() or is_float(v):
                file.write(f'{k}="{v}"\n') if k in PARAMS_NUM_AS_STR else file.write(f'{k}={v}\n')
                bash_file.write(f'{k}="{v}"\n') if k in PARAMS_NUM_AS_STR else bash_file.write(f'{k}={v}\n')
            # If the value is a string, write it as a string
            elif isinstance(v, str):
                file.write(f'{k}="{v}"\n')
                bash_file.write(f'{k}="{v}"\n')
            # Write the rest as strings
            else:
                file.write(f'{k}={v}\n')
                bash_file.write(f'{k}={v}\n')


def update_thread_settings(task_dir: str, settings_dict: dict):
    # Read existing settings
    settings_file_path = f'{task_dir}/luto/settings.py'
    bash_file_path = f'{task_dir}/luto/settings_bash.py'

    # Read current content of settings.py
    with open(settings_file_path, 'r') as file:
        content = file.readlines()

    # Update or append THREADS and WRITE_THREADS in settings.py
    cpu_per_task = settings_dict.get('CPU_PER_TASK', 50)  # Default to 30 if not specified
    found_threads = False
    found_write_threads = False

    with open(settings_file_path, 'w') as file:
        for line in content:
            if 'THREADS=' in line:
                file.write(f'THREADS={cpu_per_task}\n')
                found_threads = True
            elif 'WRITE_THREADS=' in line:
                file.write(f'WRITE_THREADS={cpu_per_task}\n')
                found_write_threads = True
            else:
                file.write(line)
        if not found_threads:
            file.write(f'THREADS={cpu_per_task}\n')
        if not found_write_threads:
            file.write(f'WRITE_THREADS={cpu_per_task}\n')

    # Repeat the process for settings_bash.py
    with open(bash_file_path, 'r') as bash_file:
        bash_content = bash_file.readlines()

    found_threads = False  # Reset the flag for the bash file
    found_write_threads = False  # Reset the flag for WRITE_THREADS

    with open(bash_file_path, 'w') as bash_file:
        for line in bash_content:
            if 'THREADS=' in line:
                bash_file.write(f'THREADS={cpu_per_task}\n')
                found_threads = True
            elif 'WRITE_THREADS=' in line:
                bash_file.write(f'WRITE_THREADS={cpu_per_task}\n')
                found_write_threads = True
            else:
                bash_file.write(line)
        if not found_threads:
            bash_file.write(f'THREADS={cpu_per_task}\n')
        if not found_write_threads:
            bash_file.write(f'WRITE_THREADS={cpu_per_task}\n')


def update_settings(settings_dict: dict, n_tasks: int, col: str):
    if settings_dict['NODE'] == 'Please specify the node name':
        if os.name == 'nt':
            print(f'{col} running on windows')
            # If the os is windows, do nothing
            # print('This will only create task folders, and NOT submit job to run!')
            pass
        elif os.name == 'posix':
            print(f'{col} running on linux')
            # If the os is linux, submit the job
            # raise ValueError('NODE must be specified!')

    # The input dir for each task will point to the absolute path of the input dir
    settings_dict['INPUT_DIR'] = '../../input'
    settings_dict['DATA_DIR'] = settings_dict['INPUT_DIR']
    settings_dict['WRITE_THREADS'] = 10  # 10 threads for writing is a safe number to avoid out-of-memory issues

    # Set the memory and time based on the resolution factor
    if int(settings_dict['RESFACTOR']) == 1:
        MEM = "250G"
        TIME = "30-0:00:00"
    elif int(settings_dict['RESFACTOR']) == 2:
        MEM = "150G"
        TIME = "10-0:00:00"
    elif int(settings_dict['RESFACTOR']) <= 5:
        MEM = "100G"
        TIME = "5-0:00:00"
    else:
        MEM = "80G"
        TIME = "2-0:00:00"

    # If the MEM and TIME are not set to auto, set them to the custom values
    MEM = settings_dict['MEM'] if settings_dict['MEM'] != 'auto' else MEM
    TIME = settings_dict['TIME'] if settings_dict['TIME'] != 'auto' else TIME
    JOB_NAME = settings_dict['JOB_NAME'] if settings_dict['JOB_NAME'] != 'auto' else col

    # Update the settings dictionary
    settings_dict['MEM'] = MEM
    settings_dict['TIME'] = TIME
    settings_dict['JOB_NAME'] = JOB_NAME

    return settings_dict


def check_settings_name(settings: dict, col: str):
    # If the column name is not in the settings, do nothing
    if 'Default_run' not in settings.columns:
        return

    # Report the changed settings
    changed_params = 0
    for idx, _ in settings.iterrows():
        if settings.loc[idx, col] != settings.loc[idx, 'Default_run']:
            changed_params = changed_params + 1
            # print(f'"{col}" has changed <{idx}>: "{settings.loc[idx, "Default_run"]}" ==> "{settings.loc[idx, col]}">')

    print(f'"{col}" has no changed parameters compared to "Default_run"') if changed_params == 0 else None


def create_run_folders(col):
    # Copy codes to the each custom run folder, excluding {EXCLUDE_DIRS} directories
    from_to_files = copy_folder_custom(os.getcwd(), f'{OUTPUT_DIR}/{col}', EXCLUDE_DIRS)
    worker = min(settings.WRITE_THREADS, len(from_to_files))
    for s, d in from_to_files:
        if not os.path.exists(s):
            print(f"Source file not found: {s}")
        if not os.path.exists(os.path.dirname(d)):
            print(f"Destination directory does not exist: {os.path.dirname(d)}")

    Parallel(n_jobs=worker, backend="threading")(delayed(shutil.copy2)(s, d) for s, d in from_to_files)
    # Create an output folder for the task
    os.makedirs(f'{OUTPUT_DIR}/{col}/output', exist_ok=True)


def convert_to_unix(file_path):
    with open(file_path, 'rb') as file:
        content = file.read()
    content = content.replace(b'\r\n', b'\n')
    with open(file_path, 'wb') as file:
        file.write(content)


def run_task_linux(cwd, col, config):
    """
    提交单个 PBS 作业，并在作业成功完成后执行同步操作。
    :param cwd: 当前工作目录
    :param col: 作业列名称，用于区分任务
    :param config: 单个作业的配置字典，包含 walltime, ncpus, mem, queue, job_name, script_content 等参数。
    """
    # 从配置字典中读取参数
    job_name = config.get("JOB_NAME", "default_job")
    walltime = config.get("TIME", "04:00:00")
    ncpus = config.get("CPU_PER_TASK", "50")
    mem = config.get("MEM", "150") + "GB"
    queue = config.get("queue", "normal")
    dir = f"{cwd}/output/{col}"
    script_content = config.get("script_content",
                                f"/g/data/jk53/LUTO_XH/apps/miniforge3/envs/luto/bin/python {dir}/0_runs_linux.py")

    # 动态生成 PBS 脚本内容
    pbs_script = f"""#!/bin/bash
    # 作业名称
    #PBS -N {job_name}
    # 分配资源：CPU核心数和内存
    #PBS -l ncpus={ncpus}
    #PBS -l mem={mem}
    # 最大运行时间
    #PBS -l walltime={walltime}
    # 合并标准输出和错误输出到同一文件
    #PBS -j oe
    #PBS -o {dir}/output
    # 提交到指定队列
    #PBS -q {queue}

    # 指定需要的存储路径（按实际配置调整）
    #PBS -l storage=gdata/jk53+scratch/jk53

    cd {dir}
    # 设置日志目录
    LOG_DIR={dir}/output
    mkdir -p $LOG_DIR

    # 输出作业开始时间到日志
    echo "Job started at $(date)" > $LOG_DIR/$PBS_JOBID.log

    # 执行脚本内容
    {script_content} >> $LOG_DIR/$PBS_JOBID.log 2>&1

    # 输出作业结束时间到日志
    echo "Job finished at $(date)" >> $LOG_DIR/$PBS_JOBID.log
    """

    # 写入 PBS 脚本到文件
    script_file = f"{dir}/output/{job_name}.pbs"
    with open(script_file, "w") as f:
        f.write(pbs_script)

    # 提交作业
    try:
        result = subprocess.run(["qsub", script_file], check=True, capture_output=True, text=True)
        job_id = result.stdout.strip()
        submission_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        print(f"{submission_time}: Job '{job_name}' submitted successfully! ID: {job_id}.")
        # 等待作业完成
        # log_file = f"{dir}/output/simulation_log.txt"
        # if wait_for_job_to_complete(job_id, log_file):
        #     # 作业完成后执行同步操作
        #     sync_files(cwd,col)
    except subprocess.CalledProcessError as e:
        print(f"Error submitting job '{job_name}':", e.stderr)


def wait_for_job_to_complete(job_id, log_file):
    """
    检查 PBS 作业的状态，直到作业完成，并检查作业是否成功完成。
    :param job_id: PBS 作业 ID
    :param log_file: 作业输出的日志文件路径
    :return: 如果作业成功返回 True，失败返回 False
    """
    while True:
        # 使用 qstat 查询作业状态
        result = subprocess.run(["qstat", job_id], check=False, capture_output=True, text=True)

        # 如果作业没有返回结果，说明作业已经完成
        if result.returncode != 0:
            print(f"Job {job_id} completed.")

            # 检查日志文件，确认作业是否成功完成
            with open(log_file, 'r') as log:
                log_content = log.read()
                if 'Total time for simulation' in log_content:
                    print(f"Job {job_id} completed successfully.")
                    return True  # 作业成功，返回 True
                else:
                    print(f"Job {job_id} failed or was incomplete, check the log file for more details.")
                    return False  # 作业失败，返回 False
        time.sleep(30)


def sync_files(cwd,col):
    # 获取日志文件路径
    source_file = f"{cwd}/output/{col}/"  # 示例路径
    log_file_path = f"{source_file}/output/simulation_log.txt"

    try:
        # 读取 simulation_log 文件中的开始时间
        with open(log_file_path, 'r') as log_file:
            lines = log_file.readlines()

        # 查找 "Simulation started" 的时间行
        start_time_line = next((line for line in lines if "Simulation started" in line), None)
        if not start_time_line:
            print("Could not find 'Simulation started' in the log file.")
            return

        # 提取时间戳并转换为 datetime 对象
        start_time_str = start_time_line.split("Simulation started at ")[1].strip()
        start_time = datetime.datetime.strptime(start_time_str, "%a %b %d %H:%M:%S %Y")

        # 记录同步开始时间
        sync_start_time = time.time()
        with open(log_file_path, 'a') as log_file:
            log_file.write(f"Sync started at {time.ctime(sync_start_time)}\n")

        dest_file = f"/cygdrive/n/LUF-Modelling/LUTO2_XH/LUTO2/output/{col}"  # Windows 上的路径
        rsync_log_file = f"{source_file}/output/rsync_log.txt"
        # 使用rsync同步文件夹
        rsync_command = [
            'rsync', '-avz',  # 使用rsync的archive模式，压缩，显示详细信息
            '-e', f"ssh -p 2222",  # 使用指定的SSH端口
            f'--log-file={rsync_log_file}',  # 指定日志文件路径
            source_file,  # 源目录
            f"xinhao@localhost:{dest_file}"  # 目标目录（Windows上的路径）
        ]

        subprocess.run(rsync_command, check=True)
        print(f"Files successfully synchronized from {source_file} to {dest_file}")

        # 记录同步结束时间
        sync_end_time = time.time()
        sync_duration = (sync_end_time - sync_start_time) / 3600

        # 记录同步结束时间和同步用时
        with open(log_file_path, 'a') as log_file:
            log_file.write(f"Sync finished at {time.ctime(sync_end_time)}\n")
            log_file.write(f"Sync duration: {sync_duration:.2f} h\n")

        # 计算总的用时并记录
        total_duration = (sync_end_time - start_time.timestamp()) / 3600
        with open(log_file_path, 'a') as log_file:
            log_file.write(f"Total time: {total_duration:.2f} h\n")

    except Exception as e:
        print(f"Error occurred during the synchronization process: {e}")

def recommend_resources(df):
    """
    Recommend suitable CPU based on MEM or suitable MEM based on CPU.

    Args:
        df (pd.DataFrame): Input DataFrame with rows containing CPU_PER_TASK and MEM.

    Returns:
        None: Prints the recommendations for each task.
    """
    # 如果 DataFrame 中存在 'Default_run' 列，则删除它
    df = df.drop(columns=['Default_run']) if 'Default_run' in df.columns else df
    df.columns = df.columns.astype(str)
    df = df.loc[:, ~df.columns.str.startswith('Unnamed')]

    cpu_row = df[df['Name'] == 'CPU_PER_TASK']
    mem_row = df[df['Name'] == 'MEM']

    cpu_values = cpu_row.iloc[0, 1:].astype(float)
    mem_values = mem_row.iloc[0, 1:].astype(float)

    recommended_cpus = np.ceil(mem_values / 4)
    recommended_mem = cpu_values * 4

    for task, cpu, mem, rec_cpu, rec_mem in zip(cpu_values.index, cpu_values, mem_values, recommended_cpus, recommended_mem):
        print(f"Task {task}:")
        print(f"  - Current CPU: {cpu}, Recommended MEM: {rec_mem} GB")
        print(f"  - Current MEM: {mem} GB, Recommended MEM for CPU {rec_cpu} GB")
        break

def check_null_values(df):
    """
    Check for null values in the DataFrame and print their locations.

    Args:
        df (pd.DataFrame): DataFrame to check for null values.

    Returns:
        None: Raises a ValueError if null values are found.
    """
    if df.isnull().values.any():
        null_positions = df[df.isnull().any(axis=1)]
        for index, row in null_positions.iterrows():
            null_columns = row[row.isnull()].index.tolist()
            print(f"行 {index} 的以下列存在空值：{null_columns}")
        raise ValueError("DataFrame 中存在空值，请处理后再继续执行！")


def generate_column_names(new_df, df_revise,suffix='', ghg_name_map=None, bio_name_map=None):
    """
    Generate new column names based on mappings and input data.

    Args:
        new_df (pd.DataFrame): DataFrame to update column names.
        df_revise (pd.DataFrame): DataFrame with revision information.
        ghg_name_map (dict): Mapping for GHG names.
        bio_name_map (dict): Mapping for BIO names.

    Returns:
        list: List of new column names.
    """
    if ghg_name_map is None:
        ghg_name_map = {
            "1.8C (67%) excl. avoided emis": "GHG_1_8C_67",
            "1.5C (50%) excl. avoided emis": "GHG_1_5C_50",
            "1.5C (67%) excl. avoided emis": "GHG_1_5C_67"
        }

    if bio_name_map is None:
        bio_name_map = {
            "{2010: 0, 2030: 0, 2050: 0, 2100: 0}": "BIO_0",
            "{2010: 0, 2030: 0.3, 2050: 0.3, 2100: 0.3}": "BIO_3",
            "{2010: 0, 2030: 0.3, 2050: 0.5, 2100: 0.5}": "BIO_5"
        }
    current_time = datetime.datetime.now().strftime("%Y%m%d")

    # 获取 GHG 和 BIO 对应的行值
    ghg_limits_field = new_df.iloc[new_df[new_df.iloc[:, 0] == "GHG_LIMITS_FIELD"].index[0]]
    biodiv_gbf_target_2_dict = new_df.iloc[new_df[new_df.iloc[:, 0] == "BIODIV_GBF_TARGET_2_DICT"].index[0]]

    # 检查 Name1 是否存在
    name_column = df_revise.columns[0]
    if "Name1" not in df_revise[name_column].values:
        print("警告：Name1 不存在于修订模板中，相关内容将被省略。")
        name1_row = None
    else:
        name1_row = df_revise[df_revise[name_column] == "Name1"].iloc[0]

    new_column_names = []
    for col in new_df.columns[2:]:  # 跳过前两列
        # 获取 GHG 和 BIO 的映射值
        ghg_value = ghg_name_map.get(ghg_limits_field[col], "Unknown_GHG")
        bio_value = bio_name_map.get(biodiv_gbf_target_2_dict[col], "Unknown_BIO")

        # 初始化列名部分
        col = str(col).split('.')[0]
        new_name = f"{current_time}_{col}"

        # 如果 Name1 存在并且有值，则添加到列名中
        if name1_row is not None:
            name1_value = name1_row.get(col, "")
            if not pd.isna(name1_value) and name1_value != "":
                new_name += f"_{name1_value}"
            else:
                print(f"警告：列 {col} 中 Name1 没有有效值，已跳过添加相关内容。")

        # 添加 GHG 和 BIO 的内容
        new_name += f"_{ghg_value}_{bio_value}" + suffix
        new_column_names.append(new_name)

    return new_column_names


def generate_csv(
    input_csv="Custom_runs/settings_template.csv",
    revise_excel="Custom_runs/Revise_settings_template.xlsx",
    output_csv="Custom_runs/setting_template_windows.csv",
    ghg_name_map=None,
    bio_name_map=None
):
    """
    Process input CSV and Revise Excel, generate new CSV file with updated settings.

    Args:
        input_csv (str): Path to the input CSV file. Default: "Custom_runs/settings_template.csv".
        revise_excel (str): Path to the revise Excel file. Default: "Custom_runs/Revise_settings_template.xlsx".
        output_csv (str): Path for the output CSV file. Must be specified.
        ghg_name_map (dict): Mapping for GHG names. Default: provided example map.
        bio_name_map (dict): Mapping for BIO names. Default: provided example map.

    Returns:
        None
    """

    df = pd.read_csv(input_csv)
    df_revise = pd.read_excel(revise_excel, sheet_name="using")
    df_revise.columns = df_revise.columns.astype(str)
    df_revise = df_revise.loc[:, ~df_revise.columns.str.startswith('Unnamed')]

    check_null_values(df_revise)

    new_df = pd.DataFrame(df.iloc[:, :2])
    new_df.columns = df.columns[:2]

    for col_name in df_revise.columns[1:]:
        new_df[col_name] = df["Default_run"]

    for idx, row in df_revise.iterrows():
        match_value = row[df_revise.columns[0]]
        matching_condition = new_df.iloc[:, 0] == match_value
        if matching_condition.any():
            for col_name in df_revise.columns[1:]:
                new_df.loc[matching_condition, col_name] = row[col_name]

    new_column_names = generate_column_names(new_df, df_revise,suffix, ghg_name_map, bio_name_map)
    new_df.columns = new_df.columns[:2].tolist() + new_column_names

    if os.path.exists(output_csv):
        print(f"文件 '{output_csv}' 已经存在，未覆盖保存。")
    else:
        new_df.to_csv(output_csv, index=False)
        print(f"新的DataFrame已生成，并保存为 '{output_csv}'")

    total_cost = calculate_total_cost(df_revise)
    print(f"Job Cost: {total_cost}k")
    recommend_resources(df_revise)


def create_grid_search_template(grid_dict, map_dict, output_file,suffix="", template_df_dir='Custom_runs') -> pd.DataFrame:
    create_settings_template('Custom_runs')
    template_df = pd.read_csv(os.path.join('Custom_runs','settings_template.csv'))

    # Collect new columns in a list
    template_grid_search = template_df.copy()

    # Check if all keys in grid_dict exist in the first column of the template DataFrame
    template_keys = template_df["Name"].unique()  # Assuming the first column is named "Name"
    all_keys = set(grid_dict.keys()).union(set(map_dict.keys()))
    missing_keys = [key for key in all_keys if key not in template_keys]
    if missing_keys:
        raise KeyError(f"The following keys are not found in the template: {missing_keys}")

    # Convert all values in the grid_dict to string representations
    grid_dict = {k: [str(v) for v in v] for k, v in grid_dict.items()}

    # Create a list of dictionaries with all possible permutations
    keys, values = zip(*grid_dict.items())
    permutations = [dict(zip(keys, v)) for v in itertools.product(*values)]

    permutations_df = pd.DataFrame(permutations)
    permutations_df.insert(0, 'run_idx', range(1, len(permutations_df) + 1))

    # Reporting the grid search template
    print(f'Grid search template has been created with {len(permutations_df)} permutations!')
    permutations_df.to_csv(output_file, index=False)

    # Loop through the permutations DataFrame and create new columns with updated settings
    for _, row in permutations_df.iterrows():
        # Copy the default settings
        new_column = template_df['Default_run'].copy()

        # Replace the settings using the key-value pairs in the permutation item
        for k, v in row.items():
            if k != 'run_idx':
                new_column.loc[template_df['Name'] == k] = v

        # Rename the column and add it to the DataFrame
        new_column.name = f'Run_{row["run_idx"]}'
        template_grid_search = pd.concat([template_grid_search, new_column.rename(f'Run_{row["run_idx"]}')], axis=1)

    ghg_row = template_grid_search.loc[template_grid_search['Name'] == 'GHG_LIMITS_FIELD']

    for col in template_grid_search.columns[1:]:  # 跳过 'Name' 列
        ghg_value = ghg_row[col].values[0]  # 获取当前列 GHG_LIMITS_FIELD 的值
        corresponding_value = dict(zip(map_dict['GHG_LIMITS_FIELD'], map_dict['CARBON_PRICES_FIELD'])).get(ghg_value)
        template_grid_search.loc[template_grid_search['Name'] == 'CARBON_PRICES_FIELD', col] = corresponding_value

    # Save the grid search template to the root task folder

    template_grid_search.columns = template_grid_search.columns[:2].tolist() + generate_column_names(template_grid_search, template_grid_search, suffix)
    template_grid_search.to_csv(output_file, index=False)
    total_cost = calculate_total_cost(template_grid_search)
    print(f"Job Cost: {total_cost}k")
    recommend_resources(template_grid_search)
    return template_grid_search
