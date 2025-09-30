import time

import paramiko
import os
import pandas as pd
import posixpath
import stat
from tqdm import tqdm
from tools.ssh_config import ssh_config
import sys
from joblib import Parallel, delayed

def ensure_directory_exists(path):
    try:
        absolute_path = os.path.abspath(path)
        os.makedirs(absolute_path, exist_ok=True)
        return True
    except OSError as e:
        print(f"[错误] 创建目录 {path} 时发生错误: {e}")
        return False

def is_remote_dir(sftp, path):
    try:
        attr = sftp.stat(path)
        return stat.S_ISDIR(attr.st_mode)
    except Exception:
        return False

def download_directory(sftp, remote_path, local_path):
    if not ensure_directory_exists(local_path):
        return

    try:
        items = sftp.listdir(remote_path)
        with tqdm(total=len(items), desc=f"Downloading {remote_path}", unit="item") as progress_bar:
            for item in items:
                remote_item_path = posixpath.join(remote_path, item)
                local_item_path = os.path.join(local_path, item)

                if is_remote_dir(sftp, remote_item_path):
                    download_directory(sftp, remote_item_path, local_item_path)
                else:
                    sftp.get(remote_item_path, local_item_path)
                progress_bar.update(1)
    except Exception as e:
        print(f"[错误] 下载目录 {remote_path} 时发生错误: {e}")

def get_first_folder_and_download(sftp, remote_base_path, target_folder_name, local_download_path):
    try:
        items = sftp.listdir(remote_base_path)
        folders = [item for item in items if is_remote_dir(sftp, posixpath.join(remote_base_path, item))]
        if not folders:
            print(f"[警告] 远程路径 {remote_base_path} 下没有找到任何文件夹。")
            return

        first_folder = sorted(folders)[0]
        remote_target_folder = posixpath.join(remote_base_path, first_folder, target_folder_name)
        local_target_folder = os.path.join(local_download_path, first_folder, target_folder_name)

        if is_remote_dir(sftp, remote_target_folder):
            download_directory(sftp, remote_target_folder, local_target_folder)
        else:
            print(f"[错误] 目标路径 {remote_target_folder} 不是一个目录。")
    except Exception as e:
        print(f"[错误] 发生错误：{e}")

def download_one(platform, base_name, file_name, local_root):
    """
    一个并行任务：对单个 file_name 下的 output 目录做下载。
    """
    cfg = ssh_config(platform)
    linux_host     = cfg["linux_host"]
    linux_port     = cfg["linux_port"]
    linux_username = cfg["linux_username"]
    private_key    = cfg["private_key_path"]
    project_dir    = cfg["project_dir"]
    target_name    = cfg["target_file_name"]

    # 建立 SSH + SFTP
    pkey = paramiko.RSAKey.from_private_key_file(private_key)
    transport = paramiko.Transport((linux_host, linux_port))
    transport.connect(username=linux_username, pkey=pkey)
    sftp = paramiko.SFTPClient.from_transport(transport)

    try:
        # 本地输出目录
        local_dir = os.path.join(local_root, base_name, file_name, 'output')
        os.makedirs(local_dir, exist_ok=True)

        # 远程目录
        remote_base = f"{project_dir}/{base_name}/{file_name}/output"
        # 下载目标文件夹下第一个文件，或按你的业务逻辑筛选
        get_first_folder_and_download(
            sftp, remote_base, target_name, local_dir
        )
    finally:
        sftp.close()
        transport.close()

if __name__ == "__main__":
    # time.sleep(60*60*5) # 延时 5 小时，给你时间去提交作业
    platform = "HPC"
    base_names = ["20250926_Paper2_Results_HPC"]
    local_root = "../../output"

    for base_name in base_names:
        cfg = ssh_config(platform)
        project_dir = cfg["project_dir"]

        # 1) 下载并读取 CSV（只做一次）
        local_csv = os.path.join(local_root, base_name, "grid_search_template.csv")
        os.makedirs(os.path.dirname(local_csv), exist_ok=True)
        remote_csv = f"{project_dir}/{base_name}/grid_search_template.csv"
        print(f"[INFO] Downloading CSV: {remote_csv}")
        # 建立一次 SFTP 下载 CSV
        pkey = paramiko.RSAKey.from_private_key_file(cfg["private_key_path"])
        tr = paramiko.Transport((cfg["linux_host"], cfg["linux_port"]))
        tr.connect(username=cfg["linux_username"], pkey=pkey)
        s = paramiko.SFTPClient.from_transport(tr)
        s.get(remote_csv, local_csv)
        s.close(); tr.close()

        df = pd.read_csv(local_csv)
        file_names = [fn.replace('.', '.') for fn in df.columns[1:]]

        # 2) 并行下载各子目录
        Parallel(n_jobs=7)(
            delayed(download_one)(
                platform, base_name, fn, local_root
            )
            for fn in tqdm(file_names, desc="Overall")
        )


