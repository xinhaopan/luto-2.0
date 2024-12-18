import os
import paramiko
import subprocess
from datetime import datetime
import sys
import functools
import traceback
from filelock import FileLock
from stat import S_ISDIR

import sys
import functools
import traceback

class LogToFile:
    def __init__(self, log_file, mode='w'):
        self.log_file = log_file
        self.mode = mode

    class StreamToLogger:
        def __init__(self, file, original_stream):  # 修正了这里，确保传入 original_stream
            self.file = file
            self.original_stream = original_stream  # 修正为 original_stream

        def write(self, message):
            self.file.write(message)  # 写入日志文件
            # 同时输出到控制台
            self.original_stream.write(message)

        def flush(self):
            self.file.flush()  # 刷新文件内容
            self.original_stream.flush()  # 刷新控制台内容

    def __call__(self, func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # 打开文件进行写入
            with open(self.log_file, self.mode) as file:
                original_stdout = sys.stdout
                original_stderr = sys.stderr
                try:
                    # 重定向 stdout 和 stderr 到文件和控制台
                    sys.stdout = self.StreamToLogger(file, original_stdout)
                    sys.stderr = self.StreamToLogger(file, original_stderr)
                    return func(*args, **kwargs)
                except Exception as e:
                    exc_info = traceback.format_exc()
                    sys.stderr.write(exc_info + '\n')
                    raise
                finally:
                    # 恢复标准输出和标准错误
                    sys.stdout = original_stdout
                    sys.stderr = original_stderr
        return wrapper



# 读取上次运行时间，如果文件不存在或为空则返回默认时间
@LogToFile('0_log_file.txt', mode='a')
def read_last_sync_time(file_path):
    # 默认时间
    default_time = '2023-01-01 00:00:00'

    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            last_sync_time = f.read().strip()
            # 如果文件为空，则返回默认时间
            if not last_sync_time:
                return default_time
            return last_sync_time
    else:
        return default_time


# 更新时间戳文件为当前时间
@LogToFile('0_log_file.txt', mode='a')
def update_sync_time(file_path):
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with open(file_path, 'w') as f:
        f.write(current_time)
    print(f"[INFO] Sync time updated to: {current_time}")

@LogToFile('0_log_file.txt', mode='a')
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
            print(f"Error accessing {directory}: {e}")

    # 调用递归函数
    recursive_list_files(remote_dir)

    # 将文件列表写入到指定的文本文件中
    with open(output_file, 'w') as f:
        for file in files_to_sync:
            f.write(file + '\n')

    # 关闭 SFTP 和 SSH 客户端
    sftp.close()
    client.close()

    # 输出文件路径
    print(f"File list saved to {output_file}")

# 生成 WinSCP 脚本来同步文件
@LogToFile('0_log_file.txt', mode='a')
def generate_winscp_script(sync_file, script_path, local_dir, remote_dir, private_key_path):
    with open(sync_file, 'r') as f:
        files = f.readlines()

    # 创建 WinSCP 脚本
    with open(script_path, 'w') as script_file:
        script_file.write(f"""# Open SCP connection
open scp://xp7241@gadi.nci.org.au/ -privatekey="{private_key_path}"
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

    with open(script_path, 'r') as file:
        content = file.read()

        # 替换所有的 \\ 为 \
    content = content.replace('\\', '\\')

    # 将修改后的内容写回文件
    with open(script_path, 'w') as file:
        file.write(content)

    print(f"WinSCP script saved to {script_path}")

# 调用 WinSCP 执行同步
@LogToFile('0_log_file.txt', mode='a')
def execute_winscp_script(winscp_path, script_path, log_file_path):
    print("Starting synchronization...")

    # 使用 Popen 来捕获 stdout 和 stderr
    process = subprocess.Popen([winscp_path, '/script=' + script_path, '/log=' + log_file_path],
                               stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE,
                               text=True,
                               encoding='utf-8')

    # 捕获标准输出和标准错误
    stdout, stderr = process.communicate()

    # 检查进程的返回码
    if process.returncode != 0:
        print(f"Error: {stderr}")
    else:
        print("Synchronization completed successfully.")
        print(f"Output: {stdout}")
@LogToFile('0_log_file.txt', mode='a')
def clean_large_txt_files(file_path, max_size_mb=10):
    # 将最大文件大小转换为字节
    max_size_bytes = max_size_mb * 1024 * 1024

    # 判断传入的是文件且是 .txt 文件
    if os.path.isfile(file_path) and file_path.endswith('.txt'):
        # 获取文件的大小
        file_size = os.path.getsize(file_path)

        # 如果文件大小大于指定的最大值，则删除文件
        if file_size > max_size_bytes:
            print(f"Deleting {file_path} - Size: {file_size / (1024 * 1024):.2f} MB")
            os.remove(file_path)
            print(f"Cleaned file: {file_path}")
        else:
            print(f"File {file_path} is under the size limit. No cleaning required.")
    else:
        print(f"Invalid file path or not a .txt file: {file_path}")
@LogToFile('0_log_file.txt', mode='a')
def check_and_create_lock():
    """
    检查锁文件是否存在，并创建锁文件。如果锁文件存在，返回 False，表示任务正在执行。
    """
    lock = FileLock(LOCK_FILE + ".lock")  # 使用 filelock 对锁文件进行锁定

    # 尝试获取锁，如果获取失败，则说明已有任务正在执行
    try:
        lock.acquire()  # 没有设置超时，任务执行完成后自动释放锁
        print("Lock acquired successfully.")
        return True
    except TimeoutError:
        print(f"Another task is already running.")
        return False  # 如果任务正在执行，返回 False
@LogToFile('0_log_file.txt', mode='a')
def remove_lock():
    """
    删除锁文件，任务完成后调用此方法
    """
    try:
        if os.path.exists(LOCK_FILE):
            os.remove(LOCK_FILE)
            print("Lock file removed.")
    except Exception as e:
        print(f"Error removing lock file: {str(e)}")

def create_folder(output_file, local_dir):
    with open(output_file, 'r') as file:
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

@LogToFile('0_log_file.txt', mode='a')
def execute_sync_task():
    datetime_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    if not check_and_create_lock():
        print(f"{datetime_str}: Task is already running.")
        return  # 如果锁文件存在，任务终止

    try:
        print(f"{datetime_str}: Starting task...")
        sync_time_file = '1_last_sync_time.txt' # 上次同步时间文件
        output_file = '2_sync_list.txt'  # 同步文件路径
        log_file = '3_sync_log.txt'  # 同步日志路径

        # 读取上次同步时间
        last_sync_time = read_last_sync_time(sync_time_file)
        # 更新当前时间为最后同步时间
        update_sync_time(sync_time_file)

        # 配置参数
        hostname = 'gadi.nci.org.au'  # 服务器地址
        port = 22  # SSH端口
        username = 'xp7241'  # SSH用户名
        private_key_path = r"C:\Users\s222552331\.ssh\id_rsa.ppk"  # 私钥路径
        remote_dir = "/g/data/jk53/LUTO_XH/LUTO2/output"   # 远程文件夹路径

        create_sync_list(hostname, port, username, private_key_path, remote_dir, last_sync_time, output_file)

        # 配置同步的本地路径和远程路径
        local_dir = r'N:\LUF-Modelling\LUTO2_XH\LUTO2\output'  # 本地目录
        winscp_path = r'C:\Program Files (x86)\WinSCP\WinSCP.com'  # WinSCP的路径
        script_path = 'winscp_script.txt'  # WinSCP脚本文件路径
        current_dir = os.getcwd()
        script_path = os.path.join(current_dir, script_path)
        # 生成 WinSCP 脚本
        generate_winscp_script(output_file, script_path, local_dir, remote_dir, private_key_path)
        create_folder(output_file, local_dir)
        # 执行 WinSCP 脚本
        execute_winscp_script(winscp_path, script_path, log_file)
        clean_large_txt_files(log_file)
        clean_large_txt_files(output_file)
        clean_large_txt_files(sync_time_file)
        clean_large_txt_files('0_log_file.txt')
        print("")
    except Exception as e:
        print(f"Task failed: {e}")
        print("")
    finally:
        # 任务完成后删除锁文件
        remove_lock()


LOCK_FILE = 'sync_task.lock'
execute_sync_task()