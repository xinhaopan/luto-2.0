import os, re, json
import shutil, itertools, subprocess, zipfile
import pandas as pd
import numpy as np
import datetime
import time

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



def submit_task(task_root_dir: str, col: str, platform: Literal['Denethor','NCI','HPC'], max_concurrent_tasks,model_name):
    # 复制bash和python脚本到对应目录
    if model_name == 'Run':
        script_name = 'python_script.py'
    elif model_name == 'Write':
        script_name = 'python_write_script.py'
    elif model_name == 'Report':
        script_name = 'python_report_script.py'
    else:
        raise ValueError('model_name must be either "Run" or "Write"!')
    shutil.copyfile(f'bash_scripts/{script_name}', f'{task_root_dir}/{col}/{script_name}')
    if platform == 'NCI':
        shutil.copyfile('bash_scripts/task_cmd.sh', f'{task_root_dir}/{col}/task_cmd.sh')
    elif platform == 'HPC':
        shutil.copyfile('bash_scripts/task_cmd_HPC.sh', f'{task_root_dir}/{col}/task_cmd_HPC.sh')

    # 控制最大并发
    while True:
        if platform == 'NCI':
            cmd = "qselect | wc -l"
        elif platform == 'HPC':
            cmd = "squeue -u $USER | wc -l"
        else:
            cmd = None

        if cmd:
            try:
                running_jobs = int(subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout.strip())
            except Exception as e:
                print(f"Error checking running jobs: {e}")
                running_jobs = 0
        else:
            running_jobs = 0

        if running_jobs < max_concurrent_tasks:
            break
        else:
            print(f"Max concurrent tasks reached ({running_jobs}/{max_concurrent_tasks}), waiting to submit {col}...")
            time.sleep(10)

    # 输出
    print(f"[START] Submitting task for {col} in platform: {platform}")

    try:
        with open(f'{task_root_dir}/{col}/run_std.log', 'w') as std_file, \
             open(f'{task_root_dir}/{col}/run_err.log', 'w') as err_file:
            if platform == 'Denethor':
                result = subprocess.run(['python', script_name],
                                        cwd=f'{task_root_dir}/{col}',
                                        stdout=std_file, stderr=err_file)
            elif platform == 'NCI':
                result = subprocess.run(['bash', 'task_cmd.sh',script_name],
                                        cwd=f'{task_root_dir}/{col}',
                                        stdout=std_file, stderr=err_file)
            elif platform == 'HPC':
                result = subprocess.run(['bash', 'task_cmd_HPC.sh',script_name],
                                        cwd=f'{task_root_dir}/{col}',
                                        stdout=std_file, stderr=err_file)
            else:
                raise ValueError('platform must be either "Denethor", "NCI", or "HPC"!')

        if result.returncode == 0:
            print(f"[SUCCESS] Task for {col} finished successfully (submitted)!")
        else:
            print(f"[FAILED] Task for {col} failed with exit code {result.returncode}!")

    except Exception as e:
        print(f"[ERROR] Exception occurred for task {col}: {e}")


def submit_write_task(task_root_dir: str, col: str, platform: Literal['Denethor', 'NCI', 'HPC'], max_concurrent_tasks: int):
    # 1. 复制/准备 shell 脚本（如果需要）
    settings_path = f'{task_root_dir}/{col}/luto/settings_bash.py'
    if os.path.exists(settings_path):
        with open(settings_path, 'r', encoding='utf-8') as f:
            content = f.read()
        # 替换 MEM
        new_content = re.sub(
            r'export MEM="[^"]*"',
            'export MEM="100GB"',
            content
        )
        # 替换 TIME
        new_content = re.sub(
            r'export TIME="[^"]*"',
            'export TIME="1:00:00"',
            new_content
        )
        # 替换 NCPUS
        new_content = re.sub(
            r'export NCPUS="[^"]*"',
            'export NCPUS="25"',
            new_content
        )
        with open(settings_path, 'w', encoding='utf-8') as f:
            f.write(new_content)

    shutil.copyfile('bash_scripts/python_write_script.py', f'{task_root_dir}/{col}/python_write_script.py')
    shutil.copyfile('bash_scripts/task_cmd_HPC.sh', f'{task_root_dir}/{col}/task_cmd_HPC.sh')
    if platform == 'NCI':
        # 确保PBS脚本存在
        if not os.path.exists(f'{task_root_dir}/{col}/task_cmd.sh'):
            raise FileNotFoundError('PBS脚本 task_cmd.sh 未找到！')
    elif platform == 'HPC':
        # 确保Slurm脚本存在
        if not os.path.exists(f'{task_root_dir}/{col}/task_cmd_HPC.sh'):
            raise FileNotFoundError('Slurm脚本 task_cmd_HPC.sh 未找到！')

    # 2. 控制最大并发
    while True:
        if platform == 'NCI':
            cmd = "qselect | wc -l"
        elif platform == 'HPC':
            cmd = "squeue -u $USER | wc -l"
        else:
            cmd = None

        if cmd:
            try:
                running_jobs = int(subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout.strip())
            except Exception as e:
                print(f"Error checking running jobs: {e}")
                running_jobs = 0
        else:
            running_jobs = 0

        if running_jobs < max_concurrent_tasks:
            break
        else:
            print(f"Max concurrent tasks reached ({running_jobs}/{max_concurrent_tasks}), waiting to submit {col}...")
            time.sleep(10)

    # 3. 日志文件路径
    std_log = os.path.join(task_root_dir, col, 'write_std.log')
    err_log = os.path.join(task_root_dir, col, 'write_err.log')

    # 4. 提交/运行
    print(f"[START] Submitting write_output task for {col} in platform: {platform}")
    try:
        with open(std_log, 'w') as std_file, open(err_log, 'w') as err_file:
            if platform == 'Denethor':
                # 本地直接运行
                result = subprocess.run(['python', 'write_output.py'],
                                       cwd=f'{task_root_dir}/{col}',
                                       stdout=std_file, stderr=err_file)
            elif platform == 'NCI':
                # PBS
                result = subprocess.run(['bash', 'task_cmd.sh'],
                                       cwd=f'{task_root_dir}/{col}',
                                       stdout=std_file, stderr=err_file)
            elif platform == 'HPC':
                # Slurm
                result = subprocess.run(['bash', 'task_cmd_HPC.sh'],
                                       cwd=f'{task_root_dir}/{col}',
                                       stdout=std_file, stderr=err_file)
            else:
                raise ValueError('platform must be either "Denethor", "NCI" or "HPC"!')

        if result.returncode == 0:
            print(f"[SUCCESS] Write output task for {col} finished successfully (submitted)!")
        else:
            print(f"[FAILED] Write output task for {col} failed with exit code {result.returncode}!")

    except Exception as e:
        print(f"[ERROR] Exception occurred for write_output task {col}: {e}")


def check_platform_system(platform):
    """
    检查运行模式与操作系统是否匹配，不匹配则报错退出
    """
    sys_type = os.name  # 'posix' (Linux/macOS), 'nt' (Windows)
    if platform == 'Denethor' and sys_type != 'nt':
        raise RuntimeError("Denethor just run on Windows, please switch to Windows or change platform to 'NCI' or 'HPC'.")
    if platform == 'NCI' and sys_type != 'posix':
        raise RuntimeError("NCI just run on Linux, please switch to Linux or change platform to 'Denethor'.")
    if platform == 'HPC' and sys_type != 'posix':
        raise RuntimeError("HPC just run on Linux, please switch to Linux or change platform to 'Denethor'.")

def create_task_runs(
    task_root_dir:str,
    custom_settings:pd.DataFrame,
    platform:Literal['Denathor','NCI','HPC']='single',
    n_workers:int=4,
    max_concurrent_tasks:int=300,
    use_parallel:bool=True,
    model_name:Literal['Run','Write','Report']='Run'
) -> None:
    check_platform_system(platform)
    if platform == 'NCI':
        calculate_total_cost(custom_settings)
    if platform not in ['Denethor', 'NCI','HPC']:
        raise ValueError('Platform must be one of "Denethor", "NCI", or "HPC"!')

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
        submit_task(task_root_dir, col, platform, max_concurrent_tasks,model_name)


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
    Parallel(n_jobs=n_workers)(
        delayed(lambda s, d: (shutil.copy2(s, d), os.utime(d, (time.time(), time.time()))))(s, d)
        for s, d in from_to_files
    )
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


def create_conditional_combinations(grid_dict, conditional_rules=None):
    """
    根据条件规则创建网格搜索组合
    """
    if conditional_rules is None:
        conditional_rules = []

    # 转换为字符串格式
    grid_dict_str = {k: [str(i) for i in v] for k, v in grid_dict.items()}

    # 生成所有可能的组合
    keys, values = zip(*grid_dict_str.items())
    all_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

    # 应用条件规则过滤
    filtered_combinations = []

    for combo in all_combinations:
        # 检查是否需要应用任何条件规则
        rule_applied = False

        for rule in conditional_rules:
            if matches_conditions(combo, rule['conditions']):
                # 如果restrictions为空，表示排除该组合
                if not rule['restrictions']:
                    # 排除组合，不添加到结果中
                    rule_applied = True
                    break
                # 如果有restrictions，检查是否满足限制条件
                elif satisfies_restrictions(combo, rule['restrictions']):
                    # 满足限制条件，添加组合
                    filtered_combinations.append(combo)
                    rule_applied = True
                    break
                else:
                    # 不满足限制条件，排除组合
                    rule_applied = True
                    break

        # 如果没有规则匹配，保留原组合
        if not rule_applied:
            filtered_combinations.append(combo)

    return filtered_combinations


def matches_conditions(combo, conditions):
    """
    检查组合是否匹配所有条件
    """
    for param, values in conditions.items():
        if combo.get(param) not in [str(v) for v in values]:
            return False
    return True


def satisfies_restrictions(combo, restrictions):
    """
    检查组合是否满足所有限制条件
    """
    for param, allowed_values in restrictions.items():
        if combo.get(param) not in [str(v) for v in allowed_values]:
            return False
    return True


def create_grid_search_template(grid_dict, settings_name_dict=None, use_date=False,
                                conditional_rules=None) -> pd.DataFrame:
    """
    创建基于条件规则的网格搜索模板

    Parameters:
    - grid_dict: 参数字典
    - settings_name_dict: 设置名称映射字典
    - use_date: 是否使用日期
    - conditional_rules: 条件规则列表
    """
    task_root_dir = f'../../output/{grid_dict["TASK_NAME"][0]}'
    os.makedirs(os.path.dirname(task_root_dir), exist_ok=True)
    grid_search_param_df = get_settings_df(task_root_dir)

    # get_grid_search_param_df
    template_grid_search = grid_search_param_df.copy()

    # 使用条件规则创建组合
    permutations = create_conditional_combinations(grid_dict, conditional_rules)

    # 转换为DataFrame
    permutations_df = pd.DataFrame(permutations)
    permutations_df.insert(0, 'run_idx', [i for i in range(1, len(permutations_df) + 1)])

    # 保存参数到根任务文件夹
    permutations_df.to_csv(f'{task_root_dir}/grid_search_parameters.csv', index=False)

    # 报告网格搜索参数
    original_total = 1
    grid_dict_str = {k: [str(i) for i in v] for k, v in grid_dict.items()}
    for k, v in grid_dict_str.items():
        if len(v) > 1:
            original_total *= len(v)

    print(f'条件网格搜索模板已创建！')
    print(f'原始笛卡尔积组合数: {original_total}')
    print(f'优化后组合数: {len(permutations_df)}')

    if conditional_rules:
        print('应用的条件规则:')
        for i, rule in enumerate(conditional_rules, 1):
            conditions_str = ' & '.join([f"{k}={v}" for k, v in rule['conditions'].items()])
            restrictions_str = ' & '.join([f"{k}={v}" for k, v in rule['restrictions'].items()])
            print(f'  规则{i}: 当 {conditions_str} 时，限制 {restrictions_str}')

    print('参数详情:')
    for k, v in grid_dict_str.items():
        if len(v) > 1:
            print(f'    {k:<40} : {len(v)} values')

    # get_grid_search_settings_df - 保持原有逻辑
    grid_search_param_df = permutations_df.copy()
    run_settings_dfs = []
    total = len(grid_search_param_df)

    for idx, (_, row) in enumerate(grid_search_param_df.iterrows()):
        settings_dict = template_grid_search.set_index('Name')['Default_run'].to_dict()
        settings_dict.update(row.to_dict())

        run_name = generate_run_name(row, idx, total, settings_name_dict, use_date=use_date)
        settings_dict = update_settings(settings_dict, run_name)
        run_settings_dfs.append(pd.Series(settings_dict, name=run_name))

    template_grid_search = pd.concat(run_settings_dfs, axis=1).reset_index(names='Name')
    template_grid_search.index = template_grid_search['Name'].values

    print(f'生成的运行配置列数: {len(template_grid_search.columns) - 1}')  # -1 因为有Name列
    template_grid_search.to_csv(f'{task_root_dir}/grid_search_template.csv', index=False)

    # 保存只包含变化参数的版本
    grid_search_param_df = grid_search_param_df.loc[:, grid_search_param_df.nunique() > 1]
    grid_search_param_df.to_csv(f'{task_root_dir}/grid_search_parameters_unique.csv', index=False)

    # 计算成本和推荐资源
    total_cost = calculate_total_cost(template_grid_search)
    print(f"作业成本: {total_cost}k")
    recommend_resources(template_grid_search)

    return template_grid_search





