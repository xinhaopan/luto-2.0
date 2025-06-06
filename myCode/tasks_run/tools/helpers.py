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


def submit_write_task(task_root_dir: str, col: str):
    # 1. 写入 write_output.py 脚本
    script_content = f'''import gzip
    import dill
    from luto.tools.write import write_outputs

    gz_path = r"{gz_path_escaped}"

    with gzip.open(gz_path, 'rb') as f:
        data = dill.load(f)

    write_outputs(data)
    '''
    script_path = os.path.join(f'{task_root_dir}/{col}/python_script.py', 'write_output.py')
    with open(script_path, 'w', encoding='utf-8') as f:
        f.write(script_content)
    subprocess.run(['python', 'python_script.py'], cwd=f'{task_root_dir}/{col}', stdout=std_file,
                   stderr=err_file, check=True)

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

def check_mode_system(mode):
    """
    检查运行模式与操作系统是否匹配，不匹配则报错退出
    """
    sys_type = os.name  # 'posix' (Linux/macOS), 'nt' (Windows)
    if mode == 'single' and sys_type != 'nt':
        raise RuntimeError("single just run on Windows, please switch to Windows or change mode to 'cluster'.")
    if mode == 'cluster' and sys_type != 'posix':
        raise RuntimeError("cluster just run on Linux, please switch to Linux or change mode to 'single'.")

def create_task_runs(
    task_root_dir:str,
    custom_settings:pd.DataFrame,
    mode:Literal['single','cluster']='single',
    n_workers:int=4,
    max_concurrent_tasks:int=300,
    use_parallel:bool=True
) -> None:
    check_mode_system(mode)
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


    use_parallel = False if os.name == 'posix' else use_parallel
    if use_parallel:
        tasks = [delayed(task_wraper)(col) for col in custom_settings.columns]
        for result in tqdm(Parallel(n_jobs=n_workers, return_as='generator')(tasks), total=len(tasks)):
            pass
    else:
        for col in tqdm(custom_settings.columns):
            task_wraper(col)






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



def create_grid_search_template(grid_dict, settings_name_dict=None, use_date=False) -> pd.DataFrame:
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

        run_name = generate_run_name(row, idx, total, settings_name_dict, use_date=use_date)
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

def generate_run_name(row, idx, total, settings_name_dict, use_date=True):
    """
    生成如 20240521_Run_01_GHG_Low_BIO_Low 这样格式的run_id，编号宽度根据实验总数自动调整。
    """
    parts = []
    if use_date:
        parts.append(datetime.datetime.now().strftime("%Y%m%d"))
    width = len(str(total))
    run_num = str(idx + 1).zfill(width)
    parts.append(f"Run_{run_num}")

    if settings_name_dict:
        for k in settings_name_dict:
            if k in row:
                parts.append(f"{settings_name_dict[k]}_{row[k]}")
    return "_".join(parts)

