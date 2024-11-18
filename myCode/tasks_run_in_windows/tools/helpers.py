from math import e
import os
import re
import shutil
import random
import itertools
import multiprocessing
import pandas as pd
import subprocess
from datetime import datetime

from joblib import delayed, Parallel

from myCode.tasks_run_in_windows.tools.parameters import EXCLUDE_DIRS, PARAMS_NUM_AS_STR, PARAMS_TO_EVAL, TASK_ROOT_DIR,OUTPUT_DIR
from luto import settings


def print_with_time(message):
    """
    打印带有时间戳的消息。
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
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
        settings_df.to_csv(f'{to_path}/settings_template0.csv', index=False)


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
    run_task(cwd, col)  # 执行任务


def create_task_runs(from_path: str = f'{TASK_ROOT_DIR}/settings_template.csv',use_multithreading=True,  num_workers: int = 4):
    """读取设置模板文件并并行运行任务"""
    # 读取自定义设置文件
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


def run_task(cwd, col):
    print_with_time(f"{col}: running task for column...")
    start_time = time.time()  # 记录任务开始时间
    log_file = f'output/{col}/output/error_log.txt'  # 定义日志文件路径

    # 根据操作系统选择 Python 解释器路径
    if os.name == 'nt':
        python_path = r'F:\xinhao\miniforge\envs\luto\python.exe'  # 修改为 Windows 上的 Python 路径
    elif os.name == 'posix':
        python_path = '/scratch/jk53/miniforge/envs/luto/bin/python'  # 修改为 Linux 上的 Python 3 路径
    try:
        # 运行子进程，捕获标准输出和标准错误
        result = subprocess.run(
            [python_path, 'temp_runs.py'],
            cwd=f'output/{col}',
            capture_output=True,  # 捕获输出
            text=True  # 将输出转换为文本
        )

        # 如果子进程返回码不为0，或者有异常输出到stderr，说明有错误发生
        if result.returncode != 0 or result.stderr:
            with open(log_file, 'a') as f:
                f.write(f"Error running temp_runs.py for {col}:\n")
                f.write(f"stdout:\n{result.stdout}\n")
                f.write(f"stderr:\n{result.stderr}\n")
            print_with_time(f"{col}: error occurred while running temp_runs.py, see {log_file} for details.")

    except Exception as e:
        # 捕获 Python 异常，并将其写入日志文件
        with open(log_file, 'a') as f:
            f.write(f"Exception occurred while running temp_runs.py for {col}:\n")
            f.write(f"{str(e)}\n")
        print_with_time(f"{col}: exception occurred during execution, see {log_file} for details.")

    end_time = time.time()  # 记录任务结束时间
    elapsed_time = (end_time - start_time) / 3600  # 计算用时，单位：小时
    print_with_time(f"{col}: completed. Elapsed time: {elapsed_time:.2f} h")


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
    cpu_per_task = settings_dict.get('CPU_PER_TASK', 30)  # Default to 30 if not specified
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
            # If the os is windows, do nothing
            # print('This will only create task folders, and NOT submit job to run!')
            pass
        elif os.name == 'posix':
            # If the os is linux, submit the job
            raise ValueError('NODE must be specified!')

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
            print(f'"{col}" has changed <{idx}>: "{settings.loc[idx, "Default_run"]}" ==> "{settings.loc[idx, col]}">')

    print(f'"{col}" has no changed parameters compared to "Default_run"') if changed_params == 0 else None


def create_run_folders(col):
    # Copy codes to the each custom run folder, excluding {EXCLUDE_DIRS} directories
    from_to_files = copy_folder_custom(os.getcwd(), f'{OUTPUT_DIR}/{col}', EXCLUDE_DIRS)
    worker = min(settings.WRITE_THREADS, len(from_to_files))
    Parallel(n_jobs=worker, backend="threading")(delayed(shutil.copy2)(s, d) for s, d in from_to_files)
    # Create an output folder for the task
    os.makedirs(f'{OUTPUT_DIR}/{col}/output', exist_ok=True)


def convert_to_unix(file_path):
    with open(file_path, 'rb') as file:
        content = file.read()
    content = content.replace(b'\r\n', b'\n')
    with open(file_path, 'wb') as file:
        file.write(content)





















