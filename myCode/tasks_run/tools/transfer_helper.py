import paramiko
import os
import posixpath
import stat
from tqdm import tqdm
from tools.ssh_config import ssh_config

def ensure_local_dir_exists(path):
    abs_path = os.path.abspath(path)
    os.makedirs(abs_path, exist_ok=True)

def is_remote_dir(sftp, path):
    try:
        attr = sftp.stat(path)
        return stat.S_ISDIR(attr.st_mode)
    except Exception:
        return False

def find_all_data_gz(sftp, remote_base_dir):
    """
    只查找 /.../output/{file_name}/Run_xxx/output/xxxx/data_with_solution.gz 路径
    """
    result = []
    try:
        # 1. 列出所有 Run_* 目录
        run_dirs = [
            d for d in sftp.listdir(remote_base_dir)
            if d.startswith("Run_") and is_remote_dir(sftp, posixpath.join(remote_base_dir, d))
        ]
        for run_dir in run_dirs:
            run_path = posixpath.join(remote_base_dir, run_dir)
            output_path = posixpath.join(run_path, "output")
            # 2. 检查 output 目录
            if is_remote_dir(sftp, output_path):
                # 3. 列出 output 下所有子目录
                for sub_dir in sftp.listdir(output_path):
                    sub_path = posixpath.join(output_path, sub_dir)
                    if is_remote_dir(sftp, sub_path):
                        gz_path = posixpath.join(sub_path, "data_with_solution.gz")
                        # 4. 检查文件是否存在
                        try:
                            sftp.stat(gz_path)
                            result.append(gz_path)
                        except FileNotFoundError:
                            continue
    except Exception as e:
        print(f"[错误] {remote_base_dir}出错: {e}")
    return result

def download_file(sftp, remote_path, local_path):
    ensure_local_dir_exists(os.path.dirname(local_path))
    sftp.get(remote_path, local_path)
    remote_size = sftp.stat(remote_path).st_size
    local_size = os.path.getsize(local_path)
    if remote_size != local_size:
        raise ValueError(f"文件大小不一致: 远程{remote_size}字节，本地{local_size}字节")

def download_all_data_gz(file_name, platform="NCI"):
    cfg = ssh_config(platform)
    linux_host = cfg["linux_host"]
    linux_port = cfg["linux_port"]
    linux_username = cfg["linux_username"]
    private_key_path = cfg["private_key_path"]
    project_dir = cfg["project_dir"]

    remote_base_dir = f"{project_dir}/{file_name}"
    local_base_dir = os.path.join(r"N:\LUF-Modelling\LUTO2_XH\LUTO2\output", file_name)

    # 建立连接
    private_key = paramiko.RSAKey.from_private_key_file(private_key_path)
    transport = paramiko.Transport((linux_host, linux_port))
    transport.connect(username=linux_username, pkey=private_key)
    sftp = paramiko.SFTPClient.from_transport(transport)

    try:
        print(f"正在递归查找所有 data_with_solution.gz 文件...")
        data_gz_paths = find_all_data_gz(sftp, remote_base_dir)
        if not data_gz_paths:
            print("未找到任何 data_with_solution.gz 文件。")
            exit(0)

        print(f"{file_name}共找到 {len(data_gz_paths)} 个 data_with_solution.gz 文件，开始下载...")
        for remote_path in tqdm(data_gz_paths, desc="Downloading files"):
            # 构造相对路径
            # remote_path: /g/data/jk53/LUTO_XH/LUTO2/output/20250608_Paper1_results_test_99/Run_1_GHG_low_BIO_low/output/xxx/data_with_solution.gz
            # 我们要保留 20250608_Paper1_results_test_99/Run_1_GHG_low_BIO_low/output/xxx/data_with_solution.gz
            rel_path = remote_path.split(f"/output/{file_name}/")[-1]
            rel_path = posixpath.join(file_name, rel_path)  # file_name/Run_xxx/output/...
            rel_path = rel_path.replace("/", os.sep)  # 转为本地路径分隔符
            local_path = os.path.join(r"N:\LUF-Modelling\LUTO2_XH\LUTO2\output", rel_path)
            download_file(sftp, remote_path, local_path)

            settings_path = os.path.dirname(os.path.dirname(os.path.dirname(rel_path)))
            local_settings_path = os.path.join(r"N:\LUF-Modelling\LUTO2_XH\LUTO2\output",settings_path, 'luto', 'settings.py')
            remote_settings_path = os.path.join(project_dir,settings_path, 'luto', 'settings.py').replace('\\', '/')
            download_file(sftp, remote_settings_path, local_settings_path)

        print("全部下载完成！")
    finally:
        sftp.close()
        transport.close()

if __name__ == "__main__":
    # 配置区
    file_name = "20250608_Paper1_results_test_99"  # ===> 你的输入


