import paramiko
import os
import sys
import posixpath  # 导入 posixpath 替代 os.path

def print_progress(transferred, total):
    """
    打印文件传输进度的函数。
    :param transferred: 到目前为止已传输的字节数。
    :param total: 文件总字节数。
    """
    progress_percentage = (transferred / total) * 100
    sys.stdout.write(f"\rTransferring... {progress_percentage:.2f}%")
    sys.stdout.flush()

def transfer_files(local_dir, remote_dir, hostname, port, username, password):
    try:
        # 创建SSH客户端实例
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(hostname, port, username, password)

        # 使用SSH客户端打开SFTP会话
        sftp = ssh.open_sftp()

        # 遍历本地目录及其子目录
        for dirpath, dirnames, filenames in os.walk(local_dir):
            # 计算远程目录路径
            remote_path = posixpath.join(remote_dir, posixpath.relpath(dirpath, local_dir))
            # 尝试创建远程目录
            try:
                sftp.chdir(remote_path)  # 测试是否存在
            except IOError:
                sftp.mkdir(remote_path)  # 创建远程目录
                sftp.chdir(remote_path)

            # 遍历文件进行上传
            for filename in filenames:
                local_file = os.path.join(dirpath, filename)
                remote_file = posixpath.join(remote_path, filename)
                try:
                    remote_file_stat = sftp.stat(remote_file)
                    local_file_stat = os.stat(local_file)
                    # 检查文件大小和修改日期
                    if local_file_stat.st_size != remote_file_stat.st_size or local_file_stat.st_mtime > remote_file_stat.st_mtime:
                        raise IOError # 强制更新
                except (IOError, FileNotFoundError):
                    print(f"\nTransferring {local_file} to {remote_file}")
                    sftp.put(local_file, remote_file, callback=print_progress)
                else:
                    print(f"Skipping {local_file}, no changes detected.")

        # 关闭SFTP会话和SSH连接
        sftp.close()
        ssh.close()
    except Exception as e:
        print(f"An error occurred: {e}")


# 示例用法（你可能需要根据实际情况调整路径或凭据）
local_dir = "../input"
remote_dir = '/home/s222552331/LUTO2_XH/LUTO2/input'
hostname = '52.255.34.63'
username = 's222552331'
password = 'Whitesun13jin!!'  # 更改密码为你的实际密码
transfer_files(local_dir, remote_dir, hostname, 8080, username, password)
