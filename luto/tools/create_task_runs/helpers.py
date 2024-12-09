import os
import re
import itertools
import shutil
import pandas as pd

from joblib import delayed, Parallel
from luto import settings

from luto.tools.create_task_runs.parameters import EXCLUDE_DIRS, TASK_ROOT_DIR


def create_settings_template(to_path:str=TASK_ROOT_DIR):

    # Save the settings template to the root task folder
    None if os.path.exists(to_path) else os.makedirs(to_path)
    
    # Check if the settings_template.csv already exists
    if os.path.exists(f'{to_path}/settings_template.csv'):
        print('settings_template.csv already exists! Skip creating a new one!')
    else:
        # Get the settings from luto.settings
        with open('luto/settings.py', 'r') as file, \
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
            settings_dict['WRITE_THREADS'] = 10 # 10 threads for writing is a safe number to avoid out-of-memory issues
            settings_dict['NCPUS'] = min(settings_dict['THREADS']//4*4, 48) # max 48 cores
            settings_dict['TIME'] = '10:00:00'

            # Write the non-string values to a file
            for k, v in settings_dict.items():
                if not isinstance(v, str):
                    non_str_val_file.write(f'{k}\n')

        # Create a template for custom settings
        settings_df = pd.DataFrame({k:[v] for k,v in settings_dict.items()}).T.reset_index()
        settings_df.columns = ['Name','Default_run']
        settings_df = settings_df.map(str)    
         
    return settings_df


def create_grid_search_template(template_df:pd.DataFrame, grid_dict: dict) -> pd.DataFrame:
    
    # Collect new columns in a list
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
    permutations_df.to_csv(f'{TASK_ROOT_DIR}/grid_search_parameters.csv', index=False)

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
        
    # Save the grid search template to the root task folder
    template_grid_search.to_csv(f'{TASK_ROOT_DIR}/grid_search_template.csv', index=False)

    return template_grid_search


def create_task_runs(custom_settings:pd.DataFrame):
     
    # Read the custom settings file
    custom_settings = custom_settings.dropna(how='all', axis=1)
    custom_settings = custom_settings.set_index('Name')
    
    # Replace special characters to underscore, making the column names valid python variables
    custom_settings.columns = [re.sub(r'\W+', '_', col.strip()) for col in custom_settings.columns]
    custom_settings = custom_settings.replace({'TRUE': 'True', 'FALSE': 'False'})
    custom_cols = [col for col in custom_settings.columns if col not in ['Default_run']]

    if not custom_cols:
        raise ValueError('No custom settings found in the settings_template.csv file!')

    cwd = os.getcwd()
    for col in custom_cols:
        # Evaluate the parameters that need to be evaluated
        with open(f'{TASK_ROOT_DIR}/non_str_val.txt', 'r') as file: eval_vars = file.read().splitlines()
        custom_settings.loc[eval_vars, col] = custom_settings.loc[eval_vars, col].map(eval)
        # Get the settings for the run
        custom_dict = update_settings(custom_settings[col].to_dict(), col)    
        # Create a folder for each run
        create_run_folders(col)    
        # Write the custom settings to the task folder
        write_custom_settings(f'{TASK_ROOT_DIR}/{col}', custom_dict)  
        # Submit the task if the os is linux
        submit_task(cwd, col)
        


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



def write_custom_settings(task_dir:str, settings_dict:dict):
    # Write the custom settings to the settings.py of each task
    with open(f'{task_dir}/luto/settings.py', 'w') as file, \
         open(f'{task_dir}/luto/settings_bash.py', 'w') as bash_file:
        
        for k, v in settings_dict.items():
            # List values need to be converted to bash arrays
            if isinstance(v, list):
                bash_file.write(f'{k}=({ " ".join([str(elem) for elem in v])})\n')
                file.write(f'{k}={v}\n')
            # Dict values need to be converted to bash variables
            elif isinstance(v, dict):
                file.write(f'{k}={v}\n')
                bash_file.write(f'# {k} is a dictionary, which is not natively supported in bash\n')
                for key, value in v.items():
                    key = str(key).replace(' ', '_').replace('(','').replace(')','')
                    bash_file.write(f'{k}_{key}={value}\n') 
            # If the value is a string, write it as a string
            elif isinstance(v, str):
                file.write(f'{k}="{v}"\n')
                bash_file.write(f'{k}="{v}"\n')
            # Write the rest as it is
            else:
                file.write(f'{k}={v}\n')


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
                bash_file.write(f'{k}={v}\n')
                


def update_settings(settings_dict:dict, col:str):

    # The input dir for each task will point to the absolute path of the input dir
    settings_dict['INPUT_DIR'] = os.path.abspath(settings_dict['INPUT_DIR']).replace('\\','/')
    settings_dict['DATA_DIR'] = settings_dict['INPUT_DIR']

    # Set the memory and time based on the resolution factor
    if int(settings_dict['RESFACTOR']) == 1:
        MEM = "250G"
    elif int(settings_dict['RESFACTOR']) == 2:
        MEM = "150G" 
    elif int(settings_dict['RESFACTOR']) <= 5:
        MEM = "100G"
    else:
        MEM = "80G"
        
   
    # Update the settings dictionary
    settings_dict['JOB_NAME'] = settings_dict['JOB_NAME'] if settings_dict['JOB_NAME'] != 'auto' else col
    settings_dict['MEM'] = settings_dict['MEM'] if settings_dict['MEM'] != 'auto' else MEM

    return settings_dict

    
    
    
    
def create_run_folders(col):
    # Copy codes to the each custom run folder, excluding {EXCLUDE_DIRS} directories
    from_to_files = copy_folder_custom(os.getcwd(), f'{TASK_ROOT_DIR}/{col}', EXCLUDE_DIRS)
    worker = min(settings.WRITE_THREADS, len(from_to_files))
    Parallel(n_jobs=worker)(delayed(shutil.copy2)(s, d) for s, d in from_to_files)
    # Create an output folder for the task
    os.makedirs(f'{TASK_ROOT_DIR}/{col}/output', exist_ok=True)



def submit_task(cwd:str, col:str):
    # Copy the slurm script to the task folder
    shutil.copyfile('luto/tools/create_task_runs/bash_scripts/task_cmd.sh', f'{TASK_ROOT_DIR}/{col}/task_cmd.sh')
    shutil.copyfile('luto/tools/create_task_runs/bash_scripts/python_script.py', f'{TASK_ROOT_DIR}/{col}/python_script.py')
    
    if os.name == 'nt':         
        # If the os is windows, do nothing
        print('This will only create task folders, and NOT submit job to run!')
    
    # Start the task if the os is linux
    if os.name == 'posix':
        os.chdir(f'{TASK_ROOT_DIR}/{col}')
        os.system('bash task_cmd.sh')
        os.chdir(cwd)    
    
