import os
import re
import shutil
import random
import itertools
import pandas as pd
import subprocess
import datetime
import time
import numpy as np
import ast
import sys

from joblib import delayed, Parallel

from myCode.tasks_run.tools.parameters import EXCLUDE_DIRS, SOURCE_DIR
from myCode.tasks_run.tools import calculate_total_cost
from luto import settings


def print_with_time(message):
    """
    打印带有时间戳的消息。
    """
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")


def create_settings_template(to_path: str = 'Custom_runs'):
    # Save the settings template to the root task folder
    None if os.path.exists(to_path) else os.makedirs(to_path)

    # Check if the settings_template.csv already exists
    if os.path.exists(f'{to_path}/settings_template.csv'):
        print('settings_template.csv already exists! Skip creating a new one!')
        return pd.read_csv(f'{to_path}/settings_template.csv')
    else:
        # Get the settings from luto.settings
        with open(f'{SOURCE_DIR}/luto/settings.py', 'r') as file, \
                open(f'{to_path}/non_str_val.txt', 'w') as non_str_val_file:

            lines = file.readlines()

            # Regex patterns that matches variable assignments from settings
            parameter_reg = re.compile(r"^(\s*[A-Z].*?)\s*=")
            settings_order = [match[1].strip() for line in lines if (match := parameter_reg.match(line))]

            # Reorder the settings dictionary to match the order in the settings.py file
            settings_dict = {i: getattr(settings, i) for i in dir(settings) if i.isupper()}
            settings_dict = {i: settings_dict[i] for i in settings_order if i in settings_dict}

            # Add parameters
            settings_dict['JOB_NAME'] = 'auto'
            settings_dict['MEM'] = 'auto'
            settings_dict['QUEUE'] = 'normal'
            settings_dict['WRITE_THREADS'] = 10  # 10 threads for writing is a safe number to avoid out-of-memory issues
            # settings_dict['NCPUS'] = min(settings_dict['THREADS'] // 4 * 4, 48)  # max 48 cores
            settings_dict['NCPUS'] = 'auto'
            settings_dict['TIME'] = '10:00:00'

            # Write the non-string values to a file
            for k, v in settings_dict.items():
                if not isinstance(v, str):
                    non_str_val_file.write(f'{k}\n')

        # Create a template for custom settings
        settings_df = pd.DataFrame({k: [v] for k, v in settings_dict.items()}).T.reset_index()
        settings_df.columns = ['Name', 'Default_run']
        settings_df = settings_df.map(str)

    return settings_df



def process_column(col, custom_settings, script_name, delay):
    time.sleep(delay * 60)  # 让每个任务启动前等待固定时间
    """并行处理单个列的任务"""
    with open('Custom_runs/non_str_val.txt', 'r') as file:
        eval_vars = file.read().splitlines()
    # Evaluate the non-string values to their original types
    custom_settings.loc[eval_vars, col] = custom_settings.loc[eval_vars, col].map(eval)
    # Update the settings dictionary
    custom_dict = update_settings(custom_settings[col].to_dict(), col)
    for key in eval_vars:
        custom_dict[key] = ast.literal_eval(custom_dict[key]) if isinstance(custom_dict[key], str) else custom_dict[key]

    # Submit the task
    create_run_folders(col)
    task_dir = f'{SOURCE_DIR}/output/{col}'
    write_custom_settings(task_dir, custom_dict)

    if os.name == 'nt':
        submit_task_windows(task_dir, col, script_name)  # 执行任务
    elif os.name == 'posix':
        submit_task_linux(task_dir, custom_dict, script_name)  # 执行任务


def create_task_runs(csv_path: str,use_multithreading=True,  num_workers: int = 3, script_name='0_runs', delay=0):
    """读取设置模板文件并并行运行任务"""
    # 读取自定义设置文件
    custom_settings = pd.read_csv(csv_path, index_col=0)
    if os.name == 'posix':
        calculate_total_cost(custom_settings)
    custom_settings = custom_settings.dropna(how='all', axis=1)

    # 处理需要评估的参数
    custom_settings.columns = [re.sub(r'\W+', '_', col.strip()) for col in custom_settings.columns]
    custom_settings = custom_settings.replace({'TRUE': 'True', 'FALSE': 'False'})

    # 获取列名列表，并排除 'Default_run'
    custom_cols = [col for col in custom_settings.columns if col not in ['Default_run']]
    # num_task = len(custom_cols)

    if not custom_cols:
        raise ValueError('No custom settings found in the settings_template.csv file!')

    if use_multithreading:
        Parallel(n_jobs=num_workers, batch_size=1)(
            delayed(process_column)(col, custom_settings, script_name, i * delay)
            for i, col in enumerate(custom_cols)
        )

    else:
        delay = 0
        for col in custom_cols:
            process_column(col, custom_settings, script_name, delay)


def submit_task_windows(task_dir, col,script_name):
    print_with_time(f"{task_dir}: running task for column...")
    start_time = time.time()  # 记录任务开始时间
    log_file = f'{task_dir}/output/{script_name}_error_log.txt'  # 定义日志文件路径

    python_path = sys.executable
    task_dir = os.path.abspath(task_dir)

    try:
        # 运行子进程，捕获标准输出和标准错误
        result = subprocess.run(
            [python_path, f'{task_dir}/{script_name}.py'],
            cwd=f'{task_dir}',
            capture_output=True,  # 捕获输出
            text=True,  # 将输出转换为文本
            encoding="utf-8"
        )

        end_time = time.time()  # 记录任务结束时间
        elapsed_time = (end_time - start_time) / 3600  # 计算用时，单位：小时

        # 检查子进程的返回码和错误输出
        if result.returncode == 0 and not result.stderr:
            # 如果成功运行，记录成功信息
            with open(log_file, 'a', encoding="utf-8") as f:
                f.write(f"Success running temp_runs.py for {col}:\n")
                f.write(f"stdout:\n{result.stdout}\n")
            print_with_time(f"{col} {script_name}: successfully completed. Elapsed time: {elapsed_time:.2f} h")
        else:
            # 如果运行失败，记录错误信息
            with open(log_file, 'a', encoding="utf-8") as f:
                f.write(f"Error running temp_runs.py for {col}:\n")
                f.write(f"stdout:\n{result.stdout}\n")
                f.write(f"stderr:\n{result.stderr}\n")
            print_with_time(f"{col}: error occurred. Elapsed time: {elapsed_time:.2f} h")

    except Exception as e:
        # 捕获 Python 异常，并将其写入日志文件
        with open(log_file, 'a', encoding="utf-8") as f:
            f.write(f"Exception occurred while running temp_runs.py for {col}:\n")
            f.write(f"{str(e)}\n")
        print_with_time(f"{col}: exception occurred during execution, see {log_file} for details.")


def submit_task_linux(task_dir, config,script_name='0_runs_linux'):
    """
    提交单个 PBS 作业，并在作业成功完成后执行同步操作。
    :param col: 作业列名称，用于区分任务
    :param config: 单个作业的配置字典，包含 walltime, ncpus, mem, queue, job_name, script_content 等参数。
    """
    # 从配置字典中读取参数
    job_name = config.get("JOB_NAME", "default_job")
    walltime = config.get("TIME", "05:00:00")
    ncpus = config.get("NCPUS", "10")
    mem = config.get("MEM", "40") + "GB"
    queue = config.get("QUEUE", "normal")
    script_content = config.get("script_content",
                                f"/home/582/xp7241/apps/miniforge3/envs/luto/bin/python {script_name}.py")

    # 动态生成 PBS 脚本内容
    pbs_script = f"""#!/bin/bash
    # 作业名称
    #PBS -N {job_name}
    # 分配资源：CPU核心数和内存
    #PBS -l ncpus={ncpus}
    #PBS -l mem={mem}
    #PBS -l jobfs=10GB
    # 最大运行时间
    #PBS -l walltime={walltime}
    # 在提交作业时的当前工作目录下运行
    #PBS -l wd
    # 合并标准输出和错误输出到同一文件
    #PBS -j oe
    #PBS -o output
    # 提交到指定队列
    #PBS -q {queue}

    # 指定需要的存储路径（按实际配置调整）
    #PBS -l storage=gdata/jk53+scratch/jk53

    # 设置日志目录
    LOG_DIR=output
    mkdir -p $LOG_DIR

    # 加载 Gurobi 许可证
    source ~/.bashrc
    export GRB_LICENSE_FILE=/g/data/jk53/config/gurobi_cfg.txt
    export GRB_TOKEN_SERVER=gurobi.licensing.its.deakin.edu.au
    export GRB_TOKEN_PORT=41954
    
    # 输出作业开始时间到日志
    echo "Job started at $(date)" >> $LOG_DIR/$PBS_JOBID.log
    echo "Current directory: $(pwd)" >> $LOG_DIR/$PBS_JOBID.log
    
    # 执行脚本内容
    {script_content} >> $LOG_DIR/$PBS_JOBID.log 2>&1
    
    # 输出作业结束时间到日志
    echo "Job finished at $(date)" >> $LOG_DIR/$PBS_JOBID.log
    """

    # 写入 PBS 脚本到文件
    script_file = f"{task_dir}/output/{job_name}.pbs"
    with open(script_file, "w") as f:
        f.write(pbs_script)

    # 提交作业
    try:
        result = subprocess.run(["qsub", f"output/{job_name}.pbs"], check=True, capture_output=True, text=True, cwd=task_dir)
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


def write_custom_settings(task_dir: str, settings_dict: dict):
    # Write the custom settings to the settings.py of each task
    with open(f'{task_dir}/luto/settings.py', 'w') as file, \
            open(f'{task_dir}/luto/settings_bash.py', 'w') as bash_file:
        for k, v in settings_dict.items():
            # List values need to be converted to bash arrays
            if isinstance(v, list):
                bash_file.write(f'{k}=({" ".join([str(elem) for elem in v])})\n')
                file.write(f'{k}={v}\n')
            # Dict values need to be converted to bash variables
            elif isinstance(v, dict):
                file.write(f'{k}={v}\n')
                bash_file.write(f'# {k} is a dictionary, which is not natively supported in bash\n')
                for key, value in v.items():
                    key = str(key).replace(' ', '_').replace('(', '').replace(')', '')
                    bash_file.write(f'{k}_{key}={value}\n')
                    # If the value is a string, write it as a string
            elif isinstance(v, str):
                file.write(f'{k}="{v}"\n')
                bash_file.write(f'{k}="{v}"\n')
            # Write the rest as it is
            else:
                file.write(f'{k}={v}\n')


def update_settings(settings_dict: dict, col: str):
    # The input dir for each task will point to the absolute path of the input dir
    settings_dict['INPUT_DIR'] = os.path.join(SOURCE_DIR, settings_dict['INPUT_DIR']).replace('\\', '/')
    settings_dict['DATA_DIR'] = settings_dict['INPUT_DIR']
    # if settings_dict['CARBON_PRICES_FIELD'] != 'c0':
    #     settings_dict['CARBON_PRICES_FIELD'] = settings_dict['GHG_LIMITS_FIELD'][:9].replace('(', '')
    settings_dict['NO_GO_VECTORS'] = {
        'Winter cereals': f'{os.path.abspath(settings_dict['INPUT_DIR'])}/no_go_areas/no_go_Winter_cereals.shp',
        'Environmental Plantings': f'{os.path.abspath(settings_dict['INPUT_DIR'])}/no_go_areas/no_go_Enviornmental_Plantings.shp'
    }
    if os.name == 'posix':
        # Set the memory and time based on the resolution factor
        # if int(settings_dict['RESFACTOR']) == 1:
        #     MEM = "250G"
        # elif int(settings_dict['RESFACTOR']) == 2:
        #     MEM = "150G"
        # elif int(settings_dict['RESFACTOR']) <= 5:
        #     MEM = "100G"
        # elif int(settings_dict['RESFACTOR']) <= 10:
        #     MEM = "80G"
        # else:
        #     MEM = "40G"

        # Update the settings dictionary
        settings_dict['JOB_NAME'] = settings_dict['JOB_NAME'] if settings_dict['JOB_NAME'] != 'auto' else col
        settings_dict['MEM'] = settings_dict['MEM'] if settings_dict['MEM'] != 'auto' else MEM
        settings_dict['NCPUS'] = settings_dict['NCPUS'] if settings_dict['NCPUS'] != 'auto' else NCPUS

        # Update the threads based on the number of cpus
        settings_dict['THREADS'] = settings_dict['NCPUS']
        settings_dict['WRITE_THREADS'] = settings_dict['NCPUS']

    return settings_dict


def create_run_folders(col):
    # Copy codes to the each custom run folder, excluding {EXCLUDE_DIRS} directories
    from_to_files = copy_folder_custom(SOURCE_DIR, f'{SOURCE_DIR}/output/{col}', EXCLUDE_DIRS)
    worker = min(settings.WRITE_THREADS, len(from_to_files))
    for s, d in from_to_files:
        if not os.path.exists(s):
            print(f"Source file not found: {s}")
        if not os.path.exists(os.path.dirname(d)):
            print(f"Destination directory does not exist: {os.path.dirname(d)}")

    Parallel(n_jobs=worker, backend="threading")(delayed(shutil.copy2)(s, d) for s, d in from_to_files)
    # Create an output folder for the task
    os.makedirs(f'{SOURCE_DIR}/output/{col}/output', exist_ok=True)


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

    cpu_row = df[df['Name'] == 'NCPUS']
    mem_row = df[df['Name'] == 'MEM']

    cpu_values = cpu_row.iloc[0, 1:].astype(float)
    mem_values = mem_row.iloc[0, 1:].astype(float)

    recommended_cpus = np.ceil(mem_values / 4)
    recommended_mem = cpu_values * 4

    for task, cpu, mem, rec_cpu, rec_mem in zip(cpu_values.index, cpu_values, mem_values, recommended_cpus, recommended_mem):
        print(f"Task {task}:")
        print(f"  - Current CPU: {cpu}, Recommended MEM: {rec_mem} GB")
        print(f"  - Current MEM: {mem} GB, Recommended CPU {rec_cpu}")
        break


def create_grid_search_template(grid_dict, output_file,suffixs=['GHG_LIMITS_FIELD', 'BIODIV_GBF_TARGET_2_DICT'], run_time=None) -> pd.DataFrame:
    # Collect new columns in a list
    template_df = create_settings_template('Custom_runs')
    template_grid_search = template_df.copy()

    # Convert all values in the grid_dict to string representations
    grid_dict = {k: [str(v) for v in v] for k, v in grid_dict.items()}

    # Create a list of dictionaries with all possible permutations
    keys, values = zip(*grid_dict.items())
    permutations = [dict(zip(keys, v)) for v in itertools.product(*values)]

    permutations_df = pd.DataFrame(permutations)
    permutations_df.insert(0, 'run_idx', range(1, len(permutations_df) + 1))

    # Reporting the grid search template
    print(f'Grid search template has been created with {len(permutations_df)} permutations!')
    # permutations_df.to_csv(f'{TASK_ROOT_DIR}/grid_search_parameters.csv', index=False)

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

    # Modify column names using the new function
    template_grid_search = modify_column_name(template_grid_search, suffixs, run_time)

    print(template_grid_search.columns)
    template_grid_search.to_csv(output_file, index=False)
    total_cost = calculate_total_cost(template_grid_search)
    print(f"Job Cost: {total_cost}k")
    recommend_resources(template_grid_search)
    return template_grid_search

def modify_column_name(template_grid_search, suffixs=['GHG_LIMITS_FIELD', 'BIODIV_GBF_TARGET_2_DICT'], run_time=None):
    """
    Modify column names in the template grid search DataFrame based on the provided suffixes.

    Args:
        template_grid_search (pd.DataFrame): DataFrame containing the template settings.
        suffixs (list): List of suffixes to append to column names.
        run_time (str): Custom timestamp to use in column names. If None, uses current time.

    Returns:
        pd.DataFrame: DataFrame with modified column names.
    """
    if run_time is None:
        run_time = datetime.datetime.now().strftime("%Y%m%d")

    # Default GHG and BIO mappings
    ghg_name_map = {
        "1.8C (67%) excl. avoided emis SCOPE1": "GHG_Low",
        "1.5C (50%) excl. avoided emis SCOPE1": "GHG_Moderate",
        "1.5C (67%) excl. avoided emis SCOPE1": "GHG_High"
    }
    bio_name_map = {
        "{2010: 0, 2030: 0, 2050: 0, 2100: 0}": "BIO_Low",
        "{2010: 0, 2030: 0.15, 2050: 0.15, 2100: 0.3}": "BIO_Moderate",
        "{2010: 0, 2030: 0.15, 2050: 0.3, 2100: 0.3}": "BIO_High"
    }

    # Initialize new column names list
    new_column_names = []

    for col in template_grid_search.columns[2:]:  # Skip the first two columns
        new_name = f"{run_time}_{col.split('.')[0]}"
        for suffix in suffixs:
            try:
                if suffix == 'GHG_LIMITS_FIELD' or suffix == 'BIODIV_GBF_TARGET_2_DICT':
                    if suffix == 'GHG_LIMITS_FIELD':
                        ghg_value = template_grid_search.loc[template_grid_search['Name'] == 'GHG_LIMITS_FIELD', col].values[0]
                        new_name += f"_{ghg_name_map.get(ghg_value, 'Unknown_GHG')}"
                    elif suffix == 'BIODIV_GBF_TARGET_2_DICT':
                        bio_value = template_grid_search.loc[template_grid_search['Name'] == 'BIODIV_GBF_TARGET_2_DICT', col].values[0]
                        new_name += f"_{bio_name_map.get(bio_value, 'Unknown_BIO')}"
                else:
                    # Use suffix as an index to get the corresponding value from template_grid_search
                    suffix_value = template_grid_search.loc[template_grid_search['Name'] == suffix, col].values[0]
                    new_name += f"_{suffix_value}"
            except Exception as e:
                print(f'Error processing suffix {suffix} for column {col}: {e}')
        new_column_names.append(new_name)

    # Update the column names of the DataFrame
    template_grid_search.columns = template_grid_search.columns[:2].tolist() + new_column_names

    return template_grid_search