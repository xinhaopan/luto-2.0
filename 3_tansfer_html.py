import paramiko
import os
import pandas as pd
import posixpath
import stat
from tqdm import tqdm  # 引入 tqdm 用于进度条显示

def ensure_directory_exists(path):
    """
    确保目标目录存在，如果不存在则创建。
    """
    try:
        absolute_path = os.path.abspath(path)
        os.makedirs(absolute_path, exist_ok=True)
    except OSError as e:
        print(f"[错误] 创建目录 {path} 时发生错误: {e}")
        return False
    return True

def is_remote_dir(sftp, path):
    """
    检查远程路径是否为目录
    """
    try:
        attr = sftp.stat(path)
        return stat.S_ISDIR(attr.st_mode)
    except Exception as e:
        return False

def download_directory(sftp, remote_path, local_path):
    """
    递归下载远程目录到本地，同时显示进度条
    """
    try:
        if not ensure_directory_exists(local_path):
            return

        items = sftp.listdir(remote_path)
        total_items = len(items)

        # 使用 tqdm 显示文件夹内容的进度条
        with tqdm(total=total_items, desc=f"Downloading {remote_path}", unit="item") as progress_bar:
            for item in items:
                remote_item_path = posixpath.join(remote_path, item)
                local_item_path = os.path.join(local_path, item)

                if is_remote_dir(sftp, remote_item_path):
                    download_directory(sftp, remote_item_path, local_item_path)
                else:
                    sftp.get(remote_item_path, local_item_path)
                progress_bar.update(1)  # 更新进度条

    except Exception as e:
        print(f"[错误] 下载目录 {remote_path} 时发生错误: {e}")

def get_first_folder_and_download(sftp, remote_base_path, target_folder_name, local_download_path):
    """
    从远程服务器的 output/ 目录获取第一个文件夹，并下载该文件夹中的目标目录到本地
    """
    try:
        items = sftp.listdir(remote_base_path)
        folders = [item for item in items if is_remote_dir(sftp, posixpath.join(remote_base_path, item))]

        if not folders:
            print(f"[警告] 远程路径 {remote_base_path} 下没有找到任何文件夹。")
            return

        first_folder = sorted(folders)[0]
        remote_target_folder = posixpath.join(remote_base_path, first_folder, target_folder_name)

        if not ensure_directory_exists(local_download_path):
            return

        local_target_folder = os.path.join(local_download_path, first_folder, target_folder_name)

        if is_remote_dir(sftp, remote_target_folder):
            download_directory(sftp, remote_target_folder, local_target_folder)
        else:
            print(f"[错误] 目标路径 {remote_target_folder} 不是一个目录。")

    except Exception as e:
        print(f"[错误] 发生错误：{e}")

# 示例用法
if __name__ == "__main__":
    try:
        # 读取 CSV 文件
        csv_path = "myCode/tasks_run/Custom_runs/setting_0409_test_BIO2.csv"
        df = pd.read_csv(csv_path)
        file_names = df.columns[2:]

        # 远程服务器配置
        linux_host = "gadi.nci.org.au"
        linux_port = 22
        linux_username = "xp7241"
        private_key_path = r"C:\Users\s222552331\.ssh\id_rsa"
        target_file_name = "DATA_REPORT"

        # 创建 SSH 连接和 SFTP 会话
        private_key = paramiko.RSAKey.from_private_key_file(private_key_path)
        transport = paramiko.Transport((linux_host, linux_port))
        transport.connect(username=linux_username, pkey=private_key)
        sftp = paramiko.SFTPClient.from_transport(transport)

        # 遍历文件名并显示进度条
        with tqdm(total=len(file_names), desc="Processing files", unit="file") as progress_bar:
            for file_name in file_names:
                file_name = file_name.replace('.', '_')
                remote_base_path = f"/g/data/jk53/LUTO_XH/LUTO2/output/{file_name}/output"
                local_download_path = os.path.join("output", f'{file_name}', 'output')
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

    except FileNotFoundError as e:
        print(f"[错误] CSV 文件未找到：{str(e)}")
    except Exception as e:
        print(f"[错误] 发生意外错误：{str(e)}")