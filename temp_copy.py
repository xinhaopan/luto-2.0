import os
import shutil
import paramiko
from scp import SCPClient
import concurrent.futures
from tqdm import tqdm

def create_ssh_client(server, port, user, password):
    ssh = paramiko.SSHClient()
    ssh.load_system_host_keys()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(server, port, user, password)
    return ssh

def copy_file_from_remote(ssh, remote_file, local_file):
    scp = SCPClient(ssh.get_transport())
    scp.get(remote_file, local_file)
    scp.close()

def list_remote_files(ssh, remote_directory):
    stdin, stdout, stderr = ssh.exec_command(f'find {remote_directory} -type f')
    files = stdout.read().splitlines()
    return [file.decode('utf-8') for file in files]

def sanitize_path(path):
    invalid_chars = '<>:"/\\|?*\x08\x1b'
    for char in invalid_chars:
        path = path.replace(char, '')
    return path

def copy_files_concurrently(source_files, destination_files, max_threads=5):
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_threads) as executor:
        list(tqdm(executor.map(shutil.copy2, source_files, destination_files), total=len(source_files)))

def copy_files_from_remote_to_local(server, port, user, password, remote_directory, local_directory):
    ssh = create_ssh_client(server, port, user, password)
    remote_files = list_remote_files(ssh, remote_directory)
    local_files = []

    for remote_file in remote_files:
        relative_path = os.path.relpath(remote_file, remote_directory).replace('\\', '/')
        local_file = os.path.join(local_directory, sanitize_path(relative_path))
        local_files.append(local_file)

    for local_file in local_files:
        directory = os.path.dirname(local_file)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)

    for remote_file, local_file in zip(tqdm(remote_files, desc="Copying from remote"), local_files):
        copy_file_from_remote(ssh, remote_file, local_file)

    ssh.close()
    print("所有文件已从远程服务器复制到本地。")

def copy_files_within_local(source_root, target_root, max_threads=5):
    source_files = []
    destination_files = []

    for root, dirs, files in os.walk(source_root):
        # 跳过一级目录下的 CSV 文件
        if root == source_root:
            files = [f for f in files if not f.endswith('.csv')]

        for file in files:
            source_file = os.path.join(root, file)
            relative_path = os.path.relpath(source_file, source_root)
            destination_file = os.path.join(target_root, relative_path)
            source_files.append(source_file)
            destination_files.append(destination_file)

    for destination_file in destination_files:
        directory = os.path.dirname(destination_file)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)

    copy_files_concurrently(source_files, destination_files, max_threads)
    print("所有文件已在本地文件夹之间复制完成。")

if __name__ == "__main__":
    # 示例用法：
    # 从远程服务器复制文件到本地
    # copy_files_from_remote_to_local(
    #     server='10.136.177.2',
    #     port=22,
    #     user='s222552331',
    #     password='Whitesun13jin!!',
    #     remote_directory='/home/s222552331/LUTO2_XH/Custom_runs',
    #     local_directory='Custom_runs'
    # )

    # 在本地文件夹之间复制文件
    copy_files_within_local(
        source_root='../Custom_runs',
        target_root='output'
    )
