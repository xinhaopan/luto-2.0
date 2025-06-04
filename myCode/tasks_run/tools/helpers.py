import os, re, json
import shutil, itertools, subprocess, zipfile
import pandas as pd
import numpy as np
import datetime

from tqdm.auto import tqdm
from typing import Literal
from joblib import delayed, Parallel

from myCode.tasks_run.tools.parameters import EXCLUDE_DIRS, SERVER_PARAMS
from myCode.tasks_run.tools import calculate_total_cost
from luto import settings


def print_with_time(message):
    """
    打印带有时间戳的消息。
    """
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")


def get_settings_df(task_root_dir:str) -> pd.DataFrame:
    # Save the settings template to the root task folder
    if not os.path.exists(task_root_dir):
        os.makedirs(task_root_dir, exist_ok=True)

    # Get the settings from luto.settings
    with open('../../luto/settings.py', 'r') as file, \
            open(f'{task_root_dir}/non_str_val.txt', 'w') as non_str_val_file:

        # Regex patterns that matches variable assignments from settings
        lines = file.readlines()
        parameter_reg = re.compile(r"^(\s*[A-Z].*?)\s*=")  # Keys are uppercase and start with a letter
        settings_keys = [match[1].strip() for line in lines if (match := parameter_reg.match(line))]

        # Reorder the settings dictionary to match the order in the settings.py file
        settings_dict = {i: getattr(settings, i) for i in dir(settings) if i.isupper()}
        settings_dict = {key: settings_dict[key] for key in settings_keys if key in settings_dict}

        # Write the non-string values to a file; this helps to evaluate the settings later
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
    custom_dict = update_settings(custom_settings[col].to_dict(),col)
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

def write_settings(task_dir:str, settings_dict:dict):
    with open(f'{task_dir}/luto/settings.py', 'w') as file:
        for k, v in settings_dict.items():
            if isinstance(v, str):
                file.write(f'{k}="{v}"\n')
            else:
                file.write(f'{k}={v}\n')

def write_terminal_vars(task_dir:str, col:str, settings_dict:dict):
    with open(f'{task_dir}/luto/settings_bash.py', 'w') as bash_file:
        for key, value in settings_dict.items():
            if key not in SERVER_PARAMS:
                continue
            if isinstance(value, str):
                bash_file.write(f'export {key}="{value}"\n')
            else:
                bash_file.write(f'export {key}={value}\n')


def submit_task(task_root_dir: str, col: str, mode: Literal['single', 'cluster'], max_concurrent_tasks):
    shutil.copyfile('bash_scripts/task_cmd.sh', f'{task_root_dir}/{col}/task_cmd.sh')
    shutil.copyfile('bash_scripts/python_script.py',
                    f'{task_root_dir}/{col}/python_script.py')

    # Wait until the number of running jobs is less than max_concurrent_tasks
    if os.name == 'posix':
        while True:
            try:
                running_jobs = int(
                    subprocess.run('qselect | wc -l', shell=True, capture_output=True, text=True).stdout.strip())
            except Exception as e:
                print(f"Error checking running jobs: {e}")
            if running_jobs < max_concurrent_tasks:
                break
            else:
                print(
                    f"Max concurrent tasks reached ({running_jobs}/{max_concurrent_tasks}), waiting to submit {col}...")
                import time;
                time.sleep(10)

    # Open log files for the task run
    with open(f'{task_root_dir}/{col}/run_std.log', 'w') as std_file, \
            open(f'{task_root_dir}/{col}/run_err.log', 'w') as err_file:
        if mode == 'single':
            subprocess.run(['python', 'python_script.py'], cwd=f'{task_root_dir}/{col}', stdout=std_file,
                           stderr=err_file, check=True)
        elif mode == 'cluster' and os.name == 'posix':
            subprocess.run(['bash', 'task_cmd.sh'], cwd=f'{task_root_dir}/{col}', stdout=std_file, stderr=err_file,
                           check=True)
        else:
            raise ValueError('Mode must be either "single" or "cluster"!')

def create_task_runs(
    task_root_dir:str,
    custom_settings:pd.DataFrame,
    mode:Literal['single','cluster']='single',
    n_workers:int=4,
    max_concurrent_tasks:int=300,
    use_parallel:bool=True
) -> None:
    if os.name == 'posix':
        calculate_total_cost(custom_settings)
    if mode not in ['single', 'cluster']:
        raise ValueError('Mode must be either "single" or "cluster"!')

    # Read the custom settings file
    custom_settings = custom_settings.dropna(how='all', axis=1)
    custom_settings = custom_settings.set_index('Name') if 'Name' in custom_settings.columns else custom_settings
    # Replace TRUE/FALSE (Excel) with True/False (Python)
    custom_settings = custom_settings.replace({'TRUE': 'True', 'FALSE': 'False'})
    # Check if there are any custom settings
    if custom_settings.columns.size == 0:
        raise ValueError('No custom settings found in the settings_template.csv file!')
    # Evaluate settings that are not originally strings
    with open(f'{task_root_dir}/non_str_val.txt', 'r') as file:
        eval_vars = file.read().splitlines()
        custom_settings.loc[eval_vars] = custom_settings.loc[eval_vars].map(str).map(eval)


    def task_wraper(col):
        settings_dict = custom_settings.loc[:, col].copy()
        create_run_folders(task_root_dir, col, n_workers)
        write_settings(f'{task_root_dir}/{col}', settings_dict)
        write_terminal_vars(f'{task_root_dir}/{col}', col, settings_dict)
        submit_task(task_root_dir, col, mode, max_concurrent_tasks)

    if use_parallel:
        tasks = [delayed(task_wraper)(col) for col in custom_settings.columns]
        for result in tqdm(Parallel(n_jobs=n_workers, return_as='generator')(tasks), total=len(tasks)):
            pass
    else:
        for col in tqdm(custom_settings.columns):
            task_wraper(col)





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
    os.makedirs(destination, exist_ok=True)
    jobs = []
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


def update_settings(settings_dict: pd.DataFrame,job_name) -> pd.DataFrame:
    '''
    Update the task run settings with parameters for the server, and change the data path to absolute path.
    E.g. job name, input directory, raw data directory, and threads.
    '''
    settings_dict['INPUT_DIR'] = os.path.abspath(f"../../{settings_dict['INPUT_DIR']}").replace('\\', '/')
    settings_dict['RAW_DATA'] = os.path.abspath(f"../../{settings_dict['RAW_DATA']}").replace('\\', '/')
    settings_dict['NO_GO_VECTORS'] = {
        'Winter cereals': f'{os.path.abspath(settings_dict['INPUT_DIR'])}/no_go_areas/no_go_Winter_cereals.shp',
        'Environmental Plantings': f'{os.path.abspath(settings_dict['INPUT_DIR'])}/no_go_areas/no_go_Enviornmental_Plantings.shp'
    }

    settings_dict['KEEP_OUTPUTS'] = eval(settings_dict['KEEP_OUTPUTS'])  # Convert string to boolean
    settings_dict['THREADS'] = settings_dict['NCPUS']
    settings_dict['JOB_NAME'] = job_name

    return settings_dict

def update_permutations(settings_df: pd.DataFrame) -> pd.DataFrame:
    # --------------------------------------- GHG_EMISSIONS_LIMITS process -------------------------------------------------
    ghg_map_name = {
        'off': None,
        'GHG_Low': 'low',
        'GHG_Medium': 'medium',
        'GHG_High': 'high'
    }
    settings_df['GHG_EMISSIONS_LIMITS'] = settings_df['GHG_NAME'].map(ghg_map_name)
    if settings_df['GHG_EMISSIONS_LIMITS'].isnull().any():
        bad_vals = settings_df.loc[settings_df['GHG_EMISSIONS_LIMITS'].isnull(), 'GHG_NAME'].unique()
        raise ValueError(
            f"Invalid value(s) for GHG_EMISSIONS_LIMITS: {bad_vals}. Must be 'off', 'GHG_Low', 'GHG_Medium', or 'GHG_High'.")

    # --------------------------------------- BIODIVERSTIY_TARGET_GBF_2 process -------------------------------------------------
    gbf2_map_name = {
        'off': 'off',
        'BIO_Low': 'low',
        'BIO_Medium': 'medium',
        'BIO_High': 'high'
    }
    settings_df['BIODIVERSTIY_TARGET_GBF_2'] = settings_df['GBF2_NAME'].map(gbf2_map_name)
    if settings_df['BIODIVERSTIY_TARGET_GBF_2'].isnull().any():
        bad_vals = settings_df.loc[settings_df['BIODIVERSTIY_TARGET_GBF_2'].isnull(), 'GBF2_NAME'].unique()
        raise ValueError(
            f"Invalid value(s) for BIODIVERSTIY_TARGET_GBF_2: {bad_vals}. Must be 'off', 'BIO_Low', 'BIO_Medium', or 'BIO_High'.")

    # remove the NAME columns
    settings_df = settings_df.loc[:, ~settings_df.columns.str.contains('NAME')]
    return settings_df


def create_run_folders(task_root_dir:str, col:str, n_workers:int):
    src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../'))
    dst_dir = f'{task_root_dir}/{col}'
    # Copy the files from the source to the destination
    from_to_files = copy_folder_custom(src_dir, dst_dir, EXCLUDE_DIRS)
    Parallel(n_jobs=n_workers)(delayed(shutil.copy2)(s, d) for s, d in from_to_files)
    # Create an output folder for the task
    os.makedirs(f'{dst_dir}/output', exist_ok=True)


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
    mem_values = mem_row.iloc[0, 1:].str.replace(r'([0-9]*\.?[0-9]+).*', r'\1', regex=True).astype(float).astype(float)

    recommended_cpus = np.ceil(mem_values / 4)
    recommended_mem = cpu_values * 4

    for task, cpu, mem, rec_cpu, rec_mem in zip(cpu_values.index, cpu_values, mem_values, recommended_cpus, recommended_mem):
        print(f"Task {task}:")
        print(f"  - Current CPU: {cpu}, Recommended MEM: {rec_mem} GB")
        print(f"  - Current MEM: {mem} GB, Recommended CPU {rec_cpu}")
        break



def create_grid_search_template(grid_dict, settings_name_dict=None, run_time=None) -> pd.DataFrame:
    task_root_dir = f'../../output/{grid_dict['TASK_NAME'][0]}'
    os.makedirs(os.path.dirname(task_root_dir), exist_ok=True)
    grid_search_param_df = get_settings_df(task_root_dir)

    # get_grid_search_param_df
    template_grid_search = grid_search_param_df.copy()
    # Create a list of dictionaries with all possible permutations
    grid_dict = {k: [str(i) for i in v] for k, v in grid_dict.items()}
    keys, values = zip(*grid_dict.items())
    permutations = [dict(zip(keys, v)) for v in itertools.product(*values)]

    # Save the grid search parameters to the root task folder

    permutations_df = pd.DataFrame(permutations)
    permutations_df.insert(0, 'run_idx', [i for i in range(1, len(permutations_df) + 1)])
    permutations_df.to_csv(f'{task_root_dir}/grid_search_parameters.csv', index=False)

    # Report the grid search parameters
    print(f'Grid search template has been created with {len(permutations_df)} permutations!')
    for k, v in grid_dict.items():
        if len(v) > 1:
            print(f'    {k:<30} : {len(v)} values')

    # get_grid_search_settings_df
    grid_search_param_df = permutations_df.copy()
    run_settings_dfs = []
    total = len(grid_search_param_df)
    for idx, (_, row) in enumerate(grid_search_param_df.iterrows()):
        settings_dict = template_grid_search.set_index('Name')['Default_run'].to_dict()
        settings_dict.update(row.to_dict())

        run_name = generate_run_name(row, idx, total, settings_name_dict, run_time=run_time)
        settings_dict = update_settings(settings_dict,run_name)
        run_settings_dfs.append(pd.Series(settings_dict, name=run_name))

    # grid_search_param_df = update_permutations(pd.DataFrame(run_settings_dfs)) # update the permutations
    template_grid_search = pd.concat(run_settings_dfs, axis=1).reset_index(names='Name')
    template_grid_search.index = template_grid_search['Name'].values

    print(template_grid_search.columns)
    template_grid_search.to_csv(f'{task_root_dir}/grid_search_template.csv', index=False)

    grid_search_param_df = grid_search_param_df.loc[:, grid_search_param_df.nunique() > 1]
    grid_search_param_df.to_csv(f'{task_root_dir}/grid_search_parameters_unique.csv', index=False)

    total_cost = calculate_total_cost(template_grid_search)
    print(f"Job Cost: {total_cost}k")
    recommend_resources(template_grid_search)
    return template_grid_search

def generate_run_name(row, idx, total, settings_name_dict, run_time=None):
    """
    生成如 20240521_Run_01_GHG_Low_BIO_Low 这样格式的run_id，编号宽度根据实验总数自动调整。
    """
    if run_time is None:
        run_time = datetime.datetime.now().strftime("%Y%m%d")
    width = len(str(total))
    run_num = str(idx + 1).zfill(width)
    run_name = f"{run_time}_Run_{run_num}"
    if settings_name_dict is not None:
        name = [
            f"_{settings_name_dict[k]}_{row[k]}"
            for k in settings_name_dict if k in row.index
        ]
        run_name = run_name + ''.join(name)
    return run_name

