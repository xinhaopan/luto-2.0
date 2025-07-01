import paramiko
import os
import pandas as pd
import posixpath
import stat
from tqdm import tqdm
from tools.ssh_config import ssh_config
import sys

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


# 主函数入口
if __name__ == "__main__":
    # ---------- 0. 导入配置 ----------
    cfg = ssh_config(platform="HPC")
    linux_host = cfg["linux_host"]
    linux_port = cfg["linux_port"]
    linux_username = cfg["linux_username"]
    private_key_path = cfg["private_key_path"]
    target_file_name = cfg["target_file_name"]  # e.g., DATA_REPORT
    project_dir = cfg["project_dir"]  # e.g., /home/remote/.../output

    # ---------- 1. 建立 SSH / SFTP 连接 ----------
    pkey = paramiko.RSAKey.from_private_key_file(private_key_path)
    transport = paramiko.Transport((linux_host, linux_port))
    transport.connect(username=linux_username, pkey=pkey)
    sftp = paramiko.SFTPClient.from_transport(transport)

    # ---------- 2. 确认项目目录 ----------
    base_names = ["20250629_Paper1_Results"]
    for base_name in base_names:
        print(f"[INFO] 处理任务: {base_name}")
        local_csv_path = f"../../output/{base_name}/grid_search_template.csv"

        # ---------- 3. 创建本地目录并下载 CSV ----------
        os.makedirs(os.path.dirname(local_csv_path), exist_ok=True)
        remote_csv_path = f"{project_dir}/{base_name}/grid_search_template.csv"
        print(f"[INFO] Downloading remote CSV: {remote_csv_path}")
        sftp.get(remote_csv_path, local_csv_path)

        # ---------- 4. 读取本地 CSV ----------
        df = pd.read_csv(local_csv_path)
        file_names = df.columns[1:]

        # 遍历文件名并显示进度条
        with tqdm(total=len(file_names), desc="Processing files", unit="file") as progress_bar:
            for file_name in file_names:
                file_name = file_name.replace('.', '_')
                remote_base_path = f"{project_dir}/{base_name}/{file_name}/output"
                local_download_path = os.path.join("../../output",base_name, file_name, 'output')
                os.makedirs(local_download_path, exist_ok=True)

                if ensure_directory_exists(local_download_path):
                    print(f"[成功] 目录可用: {local_download_path}")
                else:
                    print(f"[失败] 无法使用目录: {local_download_path}")

                print(f"[信息] 正在处理文件目录：{file_name}")

                # 调用函数下载文件
                get_first_folder_and_download(
                    sftp,
                    remote_base_path,
                    target_file_name,
                    local_download_path
                )

                progress_bar.update(1)  # 更新文件进度条

    # 关闭 SFTP 会话和 SSH 连接
    sftp.close()
    transport.close()


