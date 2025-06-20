import os
import paramiko
import subprocess
from datetime import datetime
import sys
import functools
import traceback
from filelock import FileLock
from stat import S_ISDIR
import subprocess
import threading
import io
import logging

import sys
import functools
import traceback


# 日志配置
logging.basicConfig(
    filename='temp_0_log_file.txt',
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    encoding='utf-8'
)

# 读取上次运行时间，如果文件不存在或为空则返回默认时间
def read_last_sync_time(file_path):
    # 默认时间
    default_time = '2023-01-01 00:00:00'

    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            last_sync_time = f.read().strip()
            # 如果文件为空，则返回默认时间
            if not last_sync_time:
                return default_time
            return last_sync_time
    else:
        return default_time


# 更新时间戳文件为当前时间

def update_sync_time(file_path, current_time):
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(current_time)
    logging.info(f"[INFO] Sync time updated to: {current_time}")


def create_sync_list(hostname, port, username, private_key_path, remote_dir, sync_time_str, output_file):
    # 将字符串时间转换为 datetime 对象
    sync_time = datetime.strptime(sync_time_str, "%Y-%m-%d %H:%M:%S")

    # 创建 SSH 客户端
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())  # 自动接受 SSH 密钥

    # 使用私钥文件连接
    client.connect(hostname, port=port, username=username, key_filename=private_key_path)

    # 创建 SFTP 客户端
    sftp = client.open_sftp()

    # 存储需要同步的文件
    files_to_sync = []

    def recursive_list_files(directory):
        try:
            # 获取目录下的文件和子目录
            for entry in sftp.listdir_attr(directory):
                file_path = f"{directory}/{entry.filename}"
                # 判断是文件还是目录
                if S_ISDIR(entry.st_mode):
                    # 如果是目录，则递归遍历子目录
                    recursive_list_files(file_path)
                else:
                    # 如果是文件，则检查文件修改时间
                    file_mod_time = datetime.fromtimestamp(entry.st_mtime)  # 获取文件修改时间
                    if file_mod_time > sync_time:
                        relative_path = file_path[len(remote_dir) + 1:]  # 去掉remote_dir的路径部分
                        files_to_sync.append(relative_path)
        except Exception as e:
            logging.info(f"Error accessing {directory}: {e}")

    # 调用递归函数
    recursive_list_files(remote_dir)

    # 将文件列表写入到指定的文本文件中
    with open(output_file, 'w', encoding='utf-8') as f:
        for file in files_to_sync:
            f.write(file + '\n')

    # 关闭 SFTP 和 SSH 客户端
    sftp.close()
    client.close()

    # 输出文件路径
    logging.info(f"File list saved to {output_file}")

# 生成 WinSCP 脚本来同步文件

def generate_winscp_script(sync_file, script_path, local_dir, remote_dir, private_key_path):
    private_key_path = private_key_path + '.ppk' if not private_key_path.endswith('.ppk') else private_key_path
    with open(sync_file, 'r', encoding='utf-8') as f:
        files = f.readlines()

    # 创建 WinSCP 脚本
    with open(script_path, 'w', encoding='utf-8') as script_file:
        script_file.write(f"""# Open SCP connection
open scp://s222552331@hpclogin.deakingpuhpc.deakin.edu.au/ -privatekey="{private_key_path}"
# Enable batch mode
option batch on
# Disable confirmation prompts
option confirm off
""")

        # 添加同步文件的命令
        for file in files:
            file = file.strip()
            remote_path = f"{remote_dir}/{file}"
            local_path = f"{local_dir}\\{file}".replace("/","\\")
            script_file.write(f'get "{remote_path}" "{local_path}"\n')

        script_file.write("# Close connection\nexit\n")

    with open(script_path, 'r', encoding='utf-8') as file:
        content = file.read()

        # 替换所有的 \\ 为 \
    content = content.replace('\\', '\\')

    # 将修改后的内容写回文件
    with open(script_path, 'w', encoding='utf-8') as file:
        file.write(content)

    logging.info(f"WinSCP script saved to {script_path}")

def read_stream(stream, name):
    # 使用 io.TextIOWrapper 包装，指定 utf-8 编码读取字节流
    with io.TextIOWrapper(stream, encoding='utf-8', errors='replace') as text_stream:
        for line in text_stream:
            logging.info(f"[{name}] {line.strip()}")

# 调用 WinSCP 执行同步

def execute_winscp_script(winscp_path, script_path, log_file_path):
    logging.info("Starting synchronization...")

    process = subprocess.Popen(
        [winscp_path, '/script=' + script_path, '/log=' + log_file_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )

    # 启动异步线程分别读取 stdout 和 stderr，防止死锁
    stdout_thread = threading.Thread(target=read_stream, args=(process.stdout, 'STDOUT'))
    stderr_thread = threading.Thread(target=read_stream, args=(process.stderr, 'STDERR'))
    stdout_thread.start()
    stderr_thread.start()

    # 等待进程结束
    process.wait()

    # 等待两个线程都完成读取
    stdout_thread.join()
    stderr_thread.join()

    if process.returncode != 0:
        logging.info("Error: WinSCP execution failed.")
        return False
    else:
        logging.info("Synchronization completed successfully.")
        return True


def clean_large_txt_files(file_path, max_size_mb=10):
    # 将最大文件大小转换为字节
    max_size_bytes = max_size_mb * 1024 * 1024

    # 判断传入的是文件且是 .txt 文件
    if os.path.isfile(file_path) and file_path.endswith('.txt'):
        # 获取文件的大小
        file_size = os.path.getsize(file_path)

        # 如果文件大小大于指定的最大值，则删除文件
        if file_size > max_size_bytes:
            logging.info(f"Deleting {file_path} - Size: {file_size / (1024 * 1024):.2f} MB")
            os.remove(file_path)
            logging.info(f"Cleaned file: {file_path}")
        else:
            logging.info(f"File {file_path} is under the size limit. No cleaning required.")
    else:
        logging.info(f"Invalid file path or not a .txt file: {file_path}")

def check_and_create_lock():
    if os.path.exists(LOCK_FILE):
        logging.info(f"[INFO] Lock file exists at {LOCK_FILE}")
        return False
    with open(LOCK_FILE, 'w') as f:
        f.write(str(os.getpid()))  # 可选，写入当前进程ID
    logging.info(f"[INFO] Created lock file at {LOCK_FILE}")
    return True


def remove_lock():
    try:
        if os.path.exists(LOCK_FILE):
            os.remove(LOCK_FILE)
            logging.info(f"[INFO] Removed lock file at {LOCK_FILE}")
    except Exception as e:
        logging.info(f"[ERROR] Failed to remove lock file: {e}")

def create_folder(output_file, local_dir):
    with open(output_file, 'r', encoding='utf-8') as file:
        # 遍历每一行
        for line in file:
            # 去除行首和行尾的空白字符（包括换行符）
            file_path = line.strip()

            # 获取文件路径中的文件夹部分
            folder_path = os.path.dirname(file_path)
            folder_path = os.path.join(local_dir, folder_path)

            # 检查文件夹是否存在，如果不存在则创建
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)


def execute_sync_task():
    datetime_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    logging.info(f"{datetime_str}: Starting synchronization task...")
    if not check_and_create_lock():
        logging.info(f"{datetime_str}: Task is already running.")
        return  # 如果锁文件存在，任务终止

    try:
        logging.info(f"{datetime_str}: Starting task...")
        sync_time_file = 'temp_1_last_sync_time.txt' # 上次同步时间文件
        output_file = 'temp_2_sync_list.txt'  # 同步文件路径
        log_file = 'temp_3_sync_log.txt'  # 同步日志路径

        # 读取上次同步时间
        last_sync_time = read_last_sync_time(sync_time_file)

        # 配置参数
        hostname = 'hpclogin.deakingpuhpc.deakin.edu.au'   # 服务器地址
        port = 22  # SSH端口
        username = 's222552331'  # SSH用户名
        private_key_path = r"C:\Users\s222552331\.ssh\Denethor_HPC_key" # 私钥路径
        remote_dir = "/home/remote/s222552331/LUTO2_XH/LUTO2/output"   # 远程文件夹路径

        create_sync_list(hostname, port, username, private_key_path, remote_dir, last_sync_time, output_file)

        # 配置同步的本地路径和远程路径
        local_dir = r'N:\LUF-Modelling\LUTO2_XH\LUTO2\output'  # 本地目录
        winscp_path = r'C:\Program Files (x86)\WinSCP\WinSCP.com'  # WinSCP的路径
        script_path = 'temp_winscp_script.txt'  # WinSCP脚本文件路径
        current_dir = os.getcwd()
        script_path = os.path.join(current_dir, script_path)
        # 生成 WinSCP 脚本
        generate_winscp_script(output_file, script_path, local_dir, remote_dir, private_key_path)
        create_folder(output_file, local_dir)
        # 执行 WinSCP 脚本
        success = execute_winscp_script(winscp_path, script_path, log_file)
        if success:
            # 更新当前时间为最后同步时间
            update_sync_time(sync_time_file, datetime_str)
        clean_large_txt_files(log_file)
        clean_large_txt_files(output_file)
        clean_large_txt_files(sync_time_file)
        clean_large_txt_files('temp_0_log_file.txt')

        logging.info("")
    except Exception as e:
        logging.info(f"Task failed: {e}")
        logging.info("")
    finally:
        # 任务完成后删除锁文件
        remove_lock()
        logging.info("[INFO] Lock released.")


LOCK_FILE = os.path.join(os.path.dirname(__file__), "sync_task.lock")
execute_sync_task()