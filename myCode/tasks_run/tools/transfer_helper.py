import paramiko
import os
import posixpath
import stat
import re
import subprocess
import sys
import time
import tarfile
from datetime import datetime
from tqdm import tqdm
from tools.ssh_config import ssh_config
from joblib import Parallel, delayed

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
    查找所有 Data_RES*.lz4 文件 (如 Data_RES5.lz4)
    路径格式: /.../output/{file_name}/Run_xxx/output/xxxx/Data_RES*.lz4
    
    RES数字可能不同（如 Data_RES3.lz4, Data_RES5.lz4等）
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
                        # 4. 查找 Data_RES*.lz4 文件（使用正则表达式匹配 RES 后跟数字）
                        try:
                            files = sftp.listdir(sub_path)
                            for file in files:
                                # 匹配 Data_RES\d+.lz4 格式
                                if re.match(r'Data_RES\d+\.lz4$', file):
                                    lz4_path = posixpath.join(sub_path, file)
                                    result.append(lz4_path)
                        except Exception:
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

def download_all_data_gz(file_name, platform="NCI", n_jobs=9):
    def download_worker(remote_path, file_name, project_dir, cfg):
        private_key = paramiko.RSAKey.from_private_key_file(cfg["private_key_path"])
        transport = paramiko.Transport((cfg["linux_host"], cfg["linux_port"]))
        transport.connect(username=cfg["linux_username"], pkey=private_key)
        sftp = paramiko.SFTPClient.from_transport(transport)
        try:
            rel_path = remote_path.split(f"/output/{file_name}/")[-1]
            rel_path = posixpath.join(file_name, rel_path)
            rel_path = rel_path.replace("/", os.sep)
            local_path = os.path.join(r"N:\LUF-Modelling\LUTO2_XH\LUTO2\output", rel_path)
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            download_file(sftp, remote_path, local_path)

            settings_path = os.path.dirname(os.path.dirname(os.path.dirname(rel_path)))
            local_settings_path = os.path.join(r"N:\LUF-Modelling\LUTO2_XH\LUTO2\output", settings_path, 'luto', 'settings.py')
            remote_settings_path = os.path.join(project_dir, settings_path, 'luto', 'settings.py').replace('\\', '/')
            os.makedirs(os.path.dirname(local_settings_path), exist_ok=True)
            download_file(sftp, remote_settings_path, local_settings_path)
        finally:
            sftp.close()
            transport.close()

    cfg = ssh_config(platform)
    project_dir = cfg["project_dir"]
    remote_base_dir = f"{project_dir}/{file_name}"

    # 先用主进程连接查找所有文件
    private_key = paramiko.RSAKey.from_private_key_file(cfg["private_key_path"])
    transport = paramiko.Transport((cfg["linux_host"], cfg["linux_port"]))
    transport.connect(username=cfg["linux_username"], pkey=private_key)
    sftp = paramiko.SFTPClient.from_transport(transport)

    try:
        print(f"正在递归查找所有 Data_RES*.lz4 文件...")
        data_gz_paths = find_all_data_gz(sftp, remote_base_dir)
        if not data_gz_paths:
            print("未找到任何 Data_RES*.lz4 文件。")
            return

        print(f"{file_name}共找到 {len(data_gz_paths)} 个 Data_RES*.lz4 文件，开始并行下载...")
        Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(download_worker)(remote_path, file_name, project_dir, cfg)
            for remote_path in tqdm(data_gz_paths, desc="Downloading files")
        )
        print("全部下载完成！")
    finally:
        sftp.close()
        transport.close()

if __name__ == "__main__":
    # 配置区
    file_name = "20250608_Paper1_results_test_99"  # ===> 你的输入


# ==============================================================================
# rsync + delete 传输函数 (支持TAR打包模式和rsync直传模式)
# ==============================================================================

_DEFAULT_RSYNC_OPTIONS = [
    "-av",           # archive mode, verbose（不加 -z：NetCDF已压缩，再压缩浪费CPU）
    "--progress",    # 占位，_run_rsync 会替换为 --info=progress2（整体进度）
    "--partial",     # 保留部分传输的文件（断点续传）
]


def _log(message, level="INFO", log_file="rsync_download.log"):
    """记录日志消息到控制台和文件"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_msg = f"[{timestamp}] [{level}] {message}"
    try:
        print(log_msg)
    except UnicodeEncodeError:
        print(log_msg.encode(sys.stdout.encoding or "ascii", errors="replace")
                     .decode(sys.stdout.encoding or "ascii"))
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(log_msg + "\n")


def _windows_path_to_wsl(windows_path):
    """将Windows路径转换为WSL路径"""
    if not windows_path or sys.platform != "win32":
        return windows_path
    windows_path = os.path.normpath(windows_path)
    if len(windows_path) > 1 and windows_path[1] == ':':
        drive = windows_path[0].lower()
        path_without_drive = windows_path[2:].replace('\\', '/')
        return f"/mnt/{drive}{path_without_drive}"
    return windows_path.replace('\\', '/')


def _ensure_wsl_key_permissions(windows_key_path, log_file="rsync_download.log"):
    """将SSH密钥复制到WSL原生文件系统并设置600权限（解决NTFS挂载777权限问题）。
    返回WSL中可用的密钥路径。
    """
    wsl_key = _windows_path_to_wsl(windows_key_path)
    key_name = os.path.basename(windows_key_path)
    wsl_safe_path = f"~/.ssh/luto_{key_name}"
    cmd = f"mkdir -p ~/.ssh && cp {wsl_key} {wsl_safe_path} && chmod 600 {wsl_safe_path} && echo OK"
    try:
        result = subprocess.run(["wsl", "bash", "-c", cmd],
                                capture_output=True, text=True, timeout=10)
        if result.returncode == 0 and "OK" in result.stdout:
            _log(f"SSH密钥已同步到WSL: {wsl_safe_path}（权限600）", "INFO", log_file)
            return wsl_safe_path
    except Exception as e:
        _log(f"同步SSH密钥到WSL失败: {e}，回退使用 {wsl_key}", "WARNING", log_file)
    return wsl_key


def _build_ssh_command(remote_host, remote_command, ssh_key_path=None, log_file="rsync_download.log"):
    """构建SSH命令（自动处理WSL和密钥路径）"""
    ssh_cmd = ["ssh"]
    use_wsl = False
    if sys.platform == "win32":
        try:
            subprocess.run(["ssh", "-V"], capture_output=True, check=True, timeout=5)
            _log("使用原生SSH命令", "INFO", log_file)
        except Exception:
            use_wsl = True
            _log("使用WSL执行SSH命令", "INFO", log_file)

    if ssh_key_path:
        if use_wsl:
            wsl_key_path = _windows_path_to_wsl(ssh_key_path)
            _log(f"SSH密钥路径(WSL): {wsl_key_path}", "INFO", log_file)
            ssh_cmd.extend(["-i", wsl_key_path])
        else:
            _log(f"SSH密钥路径: {ssh_key_path}", "INFO", log_file)
            ssh_cmd.extend(["-i", ssh_key_path])

    ssh_cmd.extend([remote_host, remote_command])
    if use_wsl:
        ssh_cmd = ["wsl"] + ssh_cmd
    return ssh_cmd, use_wsl


def check_remote_dir(remote_host, remote_dir, ssh_key_path=None, log_file="rsync_download.log"):
    """检查远程目录是否存在，返回 True/False"""
    _log("=" * 80, log_file=log_file)
    _log("检查远程目录...", log_file=log_file)
    _log(f"远程主机: {remote_host}", log_file=log_file)
    _log(f"远程目录: {remote_dir}", log_file=log_file)
    try:
        remote_command = f"[ -d '{remote_dir}' ] && du -sh '{remote_dir}' || echo 'DIR_NOT_FOUND'"
        ssh_cmd, _ = _build_ssh_command(remote_host, remote_command, ssh_key_path, log_file)
        _log(f"执行命令: {' '.join(ssh_cmd)}", log_file=log_file)
        result = subprocess.run(ssh_cmd, capture_output=True, text=True, timeout=60)
        if result.stderr:
            _log(f"SSH stderr: {result.stderr.strip()}", "WARNING", log_file)
        if "DIR_NOT_FOUND" in result.stdout:
            _log(f"错误: 远程目录不存在: {remote_dir}", "ERROR", log_file)
            return False
        output = result.stdout.strip()
        size_info = output.split()[0] if output.split() else "Unknown"
        _log(f"✓ 远程目录存在，大小: {size_info}", "SUCCESS", log_file)
        return True
    except subprocess.TimeoutExpired:
        _log("检查超时（60秒）", "ERROR", log_file)
        return False
    except Exception as e:
        _log(f"检查远程目录出错: {e}", "ERROR", log_file)
        return False


def _create_tar_on_remote(remote_host, remote_dir, use_compression=False, ssh_key_path=None, log_file="rsync_download.log"):
    """在远程服务器创建tar包，返回远程tar路径或None"""
    _log("=" * 80, log_file=log_file)
    _log("步骤1: 在远程服务器打包目录...", log_file=log_file)
    folder_name = os.path.basename(remote_dir.rstrip("/"))
    parent_dir = os.path.dirname(remote_dir.rstrip("/"))
    if use_compression:
        tar_filename = f"{folder_name}.tar.gz"
        tar_cmd = f"cd '{parent_dir}' && tar -czf '{tar_filename}' '{folder_name}'"
        _log("打包模式: 压缩 (gzip)", log_file=log_file)
    else:
        tar_filename = f"{folder_name}.tar"
        tar_cmd = f"cd '{parent_dir}' && tar -cf '{tar_filename}' '{folder_name}'"
        _log("打包模式: 仅打包不压缩（适合NetCDF）", log_file=log_file)
    try:
        ssh_cmd, _ = _build_ssh_command(remote_host, tar_cmd, ssh_key_path, log_file)
        _log(f"执行远程命令: {tar_cmd}", log_file=log_file)
        start_time = time.time()
        result = subprocess.run(ssh_cmd, capture_output=True, text=True)
        elapsed = time.time() - start_time
        if result.returncode == 0:
            _log(f"✓ 远程打包完成！耗时: {elapsed:.1f}s ({elapsed/60:.1f}min)", "SUCCESS", log_file)
            return f"{parent_dir}/{tar_filename}"
        else:
            _log(f"✗ 远程打包失败: {result.stderr}", "ERROR", log_file)
            return None
    except Exception as e:
        _log(f"✗ 远程打包过程出错: {e}", "ERROR", log_file)
        return None


def _download_tar_file(remote_host, remote_tar_path, local_dir, ssh_key_path=None, log_file="rsync_download.log"):
    """从远程下载tar文件到本地，返回本地路径或None"""
    _log("=" * 80, log_file=log_file)
    _log("步骤2: 开始下载tar文件到本地...", log_file=log_file)
    tar_filename = os.path.basename(remote_tar_path)
    local_tar_path = os.path.join(local_dir, tar_filename)
    os.makedirs(local_dir, exist_ok=True)
    try:
        use_wsl = False
        if sys.platform == "win32":
            try:
                subprocess.run(["rsync", "--version"], capture_output=True, check=True, timeout=5)
            except Exception:
                use_wsl = True
                _log("使用WSL运行rsync", log_file=log_file)
        cmd = ["rsync", "-av", "--progress", "--partial"]
        if ssh_key_path:
            key = _ensure_wsl_key_permissions(ssh_key_path, log_file) if use_wsl else ssh_key_path
            cmd.extend(["-e", f"ssh -i {key}"])
        cmd.append(f"{remote_host}:{remote_tar_path}")
        if use_wsl:
            wsl_local = _windows_path_to_wsl(local_tar_path)
            cmd.append(wsl_local)
            cmd = ["wsl"] + cmd
            _log(f"本地目标路径(WSL): {wsl_local}", log_file=log_file)
        else:
            cmd.append(local_tar_path)
        start_time = time.time()
        success = _run_rsync(cmd, log_file)
        elapsed = time.time() - start_time
        _log(f"下载耗时: {elapsed:.1f}s ({elapsed/60:.1f}min)", log_file=log_file)
        if success and os.path.exists(local_tar_path):
            size_gb = os.path.getsize(local_tar_path) / (1024**3)
            _log(f"✓ tar文件下载完成！大小: {size_gb:.2f} GB", "SUCCESS", log_file)
            return local_tar_path
        else:
            _log("✗ 下载失败", "ERROR", log_file)
            return None
    except Exception as e:
        _log(f"✗ 下载过程出错: {e}", "ERROR", log_file)
        return None


def _extract_tar_local(tar_path, extract_dir, log_file="rsync_download.log"):
    """在本地解压tar文件，返回True/False"""
    _log("=" * 80, log_file=log_file)
    _log(f"解压 {tar_path} -> {extract_dir}", log_file=log_file)
    try:
        os.makedirs(extract_dir, exist_ok=True)
        start_time = time.time()
        with tarfile.open(tar_path, 'r:*') as tar:
            tar.extractall(path=extract_dir)
        elapsed = time.time() - start_time
        _log(f"✓ 本地解压完成！耗时: {elapsed:.1f}s ({elapsed/60:.1f}min)", "SUCCESS", log_file)
        return True
    except Exception as e:
        _log(f"✗ 本地解压失败: {e}", "ERROR", log_file)
        return False


def _delete_local_tar(tar_path, log_file="rsync_download.log"):
    """删除本地tar文件"""
    try:
        if os.path.exists(tar_path):
            os.remove(tar_path)
            _log(f"✓ 本地tar文件已删除: {tar_path}", "SUCCESS", log_file)
            return True
        else:
            _log(f"本地tar文件不存在: {tar_path}", "WARNING", log_file)
            return False
    except Exception as e:
        _log(f"✗ 删除本地tar文件失败: {e}", "ERROR", log_file)
        return False


def _delete_remote_files(remote_host, remote_dir, remote_tar_path, ssh_key_path=None,
                          confirm_delete=True, log_file="rsync_download.log"):
    """删除远程tar文件和源目录，返回True/False"""
    _log("=" * 80, log_file=log_file)
    _log("删除远程文件...", log_file=log_file)
    if not confirm_delete:
        _log("跳过删除远程文件（confirm_delete=False）", log_file=log_file)
        return False
    try:
        delete_cmd = f"rm -f '{remote_tar_path}' && rm -rf '{remote_dir}'"
        ssh_cmd, _ = _build_ssh_command(remote_host, delete_cmd, ssh_key_path, log_file)
        _log(f"执行远程命令: {delete_cmd}", log_file=log_file)
        result = subprocess.run(ssh_cmd, capture_output=True, text=True)
        if result.returncode == 0:
            _log("✓ 远程tar文件已删除", "SUCCESS", log_file)
            _log("✓ 远程源目录已删除", "SUCCESS", log_file)
            return True
        else:
            _log(f"✗ 删除远程文件失败: {result.stderr}", "ERROR", log_file)
            return False
    except Exception as e:
        _log(f"✗ 删除远程文件过程出错: {e}", "ERROR", log_file)
        return False


def _build_rsync_command(remote_host, remote_dir, local_base_dir, ssh_key_path=None,
                          rsync_options=None, log_file="rsync_download.log"):
    """构建下载rsync命令"""
    if rsync_options is None:
        rsync_options = _DEFAULT_RSYNC_OPTIONS
    local_base_dir = os.path.abspath(local_base_dir)
    os.makedirs(local_base_dir, exist_ok=True)
    folder_name = os.path.basename(remote_dir.rstrip("/"))
    remote_source = f"{remote_host}:{remote_dir}/"
    local_target = os.path.join(local_base_dir, folder_name)
    use_wsl = False
    if sys.platform == "win32":
        try:
            subprocess.run(["rsync", "--version"], capture_output=True, check=True, timeout=5)
            _log("使用原生rsync", log_file=log_file)
        except Exception:
            use_wsl = True
            _log("使用WSL运行rsync", log_file=log_file)
            local_target = _windows_path_to_wsl(local_target)
            _log(f"本地目标路径(WSL): {local_target}", log_file=log_file)
    local_target += "/"
    cmd = ["rsync"] + rsync_options
    if ssh_key_path:
        key = _ensure_wsl_key_permissions(ssh_key_path, log_file) if use_wsl else ssh_key_path
        cmd.extend(["-e", f"ssh -i {key}"])
    cmd.extend([remote_source, local_target])
    if use_wsl:
        cmd = ["wsl"] + cmd
    return cmd


def _run_rsync(cmd, log_file="rsync_download.log"):
    """运行rsync命令，带tqdm整体进度条，返回True/False"""
    _log(f"执行rsync命令: {' '.join(cmd)}", log_file=log_file)
    _log("=" * 80, log_file=log_file)

    # 替换 --progress 为 --info=progress2（整体进度而非逐文件）
    cmd_run = [c for c in cmd if c != '--progress']
    try:
        rsync_pos = next(i for i, c in enumerate(cmd_run) if c == 'rsync')
    except StopIteration:
        rsync_pos = 0
    cmd_run = (cmd_run[:rsync_pos + 1]
               + ['--info=progress2']
               + cmd_run[rsync_pos + 1:])

    _log(f"实际执行命令: {' '.join(cmd_run)}", log_file=log_file)

    try:
        # stdout 不捕获（输出到终端显示进度）；stderr 单独捕获以记录错误
        result = subprocess.run(cmd_run, text=True, stderr=subprocess.PIPE)
        if result.stderr.strip():
            for line in result.stderr.strip().splitlines():
                _log(f"rsync stderr: {line}", "WARNING", log_file)
        if result.returncode == 0:
            _log("rsync传输完成！", "SUCCESS", log_file)
            return True
        else:
            _log(f"rsync传输失败，退出码: {result.returncode}", "ERROR", log_file)
            return False
    except FileNotFoundError:
        _log("错误: 未找到rsync命令，请确保已安装", "ERROR", log_file)
        return False
    except Exception as e:
        _log(f"rsync执行出错: {e}", "ERROR", log_file)
        return False


def transfer_with_tar(remote_host, remote_dir, local_base_dir, ssh_key_path=None,
                      tar_use_compression=False, confirm_delete=True, log_file="rsync_download.log"):
    """使用tar打包方法从远程下载并删除远程文件"""
    local_download_dir = os.path.abspath(local_base_dir)
    os.makedirs(local_download_dir, exist_ok=True)

    remote_tar_path = _create_tar_on_remote(
        remote_host, remote_dir, tar_use_compression, ssh_key_path, log_file)
    if not remote_tar_path:
        _log("任务终止：远程打包失败", "ERROR", log_file)
        return False

    local_tar_path = _download_tar_file(
        remote_host, remote_tar_path, local_download_dir, ssh_key_path, log_file)
    if not local_tar_path:
        _log("任务终止：下载失败，尝试清理远程tar...", "ERROR", log_file)
        _delete_remote_files(remote_host, remote_dir, remote_tar_path, ssh_key_path,
                             confirm_delete=True, log_file=log_file)
        return False

    _log("=" * 80, log_file=log_file)
    _log("步骤3: 删除远程文件（释放远程存储空间）...", log_file=log_file)
    if not _delete_remote_files(remote_host, remote_dir, remote_tar_path, ssh_key_path,
                                 confirm_delete, log_file):
        _log("警告：远程文件删除失败，继续本地解压", "WARNING", log_file)

    _log("=" * 80, log_file=log_file)
    _log("步骤4: 解压到本地...", log_file=log_file)
    if not _extract_tar_local(local_tar_path, local_download_dir, log_file):
        _log(f"任务终止：本地解压失败，tar文件保留在: {local_tar_path}", "ERROR", log_file)
        return False

    _log("=" * 80, log_file=log_file)
    _log("步骤5: 清理本地tar文件...", log_file=log_file)
    _delete_local_tar(local_tar_path, log_file)
    return True


def transfer_with_rsync(remote_host, remote_dir, local_base_dir, ssh_key_path=None,
                        rsync_options=None, confirm_delete=True, log_file="rsync_download.log"):
    """使用rsync直接下载并删除远程目录"""
    _log("步骤1: 使用rsync从远程下载...", log_file=log_file)
    rsync_cmd = _build_rsync_command(
        remote_host, remote_dir, local_base_dir, ssh_key_path, rsync_options, log_file)
    if not _run_rsync(rsync_cmd, log_file):
        _log("rsync下载失败", "ERROR", log_file)
        return False
    _log("✓ rsync下载完成", "SUCCESS", log_file)

    _log("步骤2: 删除远程目录...", log_file=log_file)
    if not confirm_delete:
        _log("跳过删除远程目录（confirm_delete=False）", log_file=log_file)
        return True
    try:
        delete_cmd = f"rm -rf '{remote_dir}'"
        ssh_cmd, _ = _build_ssh_command(remote_host, delete_cmd, ssh_key_path, log_file)
        result = subprocess.run(ssh_cmd, capture_output=True, text=True, timeout=300)
        if result.returncode == 0:
            _log(f"✓ 远程目录已删除: {remote_host}:{remote_dir}", "SUCCESS", log_file)
            return True
        else:
            _log(f"删除远程目录失败: {result.stderr}", "ERROR", log_file)
            return False
    except Exception as e:
        _log(f"删除远程目录时出错: {e}", "ERROR", log_file)
        return False


def rsync_and_delete(platform, task_name, local_base_dir="../../output",
                     transfer_mode="full",
                     use_tar_mode=False, tar_use_compression=False,
                     confirm_delete=True, debug_mode=False, test_rsync_command=False,
                     lz4_n_jobs=9,
                     log_file="rsync_download.log"):
    """从远程服务器下载任务文件夹到本地。

    Args:
        platform:            "NCI" 或 "HPC"
        task_name:           远程 output 目录下的文件夹名
        local_base_dir:      本地下载基础目录
        transfer_mode:       传输模式
                               "full"   — 完整下载并删除远程（rsync 或 TAR）
                               "report" — 只下载 DATA_REPORT 目录，不删除远程
                               "lz4"    — 只下载 Data_RES*.lz4 文件（并行 SFTP），不删除远程
        use_tar_mode:        full模式：True=先远程打包再传输，False=rsync直传
        tar_use_compression: full+TAR模式：是否gzip压缩（NetCDF推荐False）
        confirm_delete:      full模式：True=下载完后直接删除远程，False=不删除远程
        debug_mode:          full模式：仅测试SSH连接，不执行实际传输
        test_rsync_command:  debug_mode下额外打印rsync命令
        lz4_n_jobs:          lz4模式：并行下载线程数
        log_file:            日志文件路径
    """
    if transfer_mode == "lz4":
        _log("=" * 80, log_file=log_file)
        _log(f"只下载 lz4 文件  [平台: {platform}]  [任务: {task_name}]", log_file=log_file)
        _log("=" * 80, log_file=log_file)
        download_all_data_gz(task_name, platform=platform, n_jobs=lz4_n_jobs)
        return

    if transfer_mode == "report":
        download_report_only(task_name, platform=platform,
                             local_base_dir=local_base_dir, log_file=log_file)
        return

    if transfer_mode != "full":
        raise ValueError(f"未知的 transfer_mode: {transfer_mode!r}，请选择 'full' / 'report' / 'lz4'")

    # ---- full 模式 ----
    cfg = ssh_config(platform)
    remote_host = f"{cfg['linux_username']}@{cfg['linux_host']}"
    remote_dir = f"{cfg['project_dir']}/{task_name}"
    ssh_key_path = cfg['private_key_path']

    _log("=" * 80, log_file=log_file)
    _log(f"完整下载任务  [平台: {platform}]  [任务: {task_name}]", log_file=log_file)
    _log(f"传输方式: {'TAR打包' if use_tar_mode else 'rsync直传'}", log_file=log_file)
    _log("=" * 80, log_file=log_file)

    if debug_mode:
        _log("【调试模式】测试SSH连接...", "WARNING", log_file)
        if check_remote_dir(remote_host, remote_dir, ssh_key_path, log_file):
            _log("✓ SSH连接测试成功！", "SUCCESS", log_file)
        else:
            _log("✗ SSH连接测试失败", "ERROR", log_file)
        if test_rsync_command:
            cmd = _build_rsync_command(remote_host, remote_dir, local_base_dir,
                                       ssh_key_path, log_file=log_file)
            _log(f"构建的rsync命令: {' '.join(cmd)}", log_file=log_file)
        _log("【调试完成】设 debug_mode=False 后再运行完整任务", "INFO", log_file)
        return

    if not check_remote_dir(remote_host, remote_dir, ssh_key_path, log_file):
        _log("任务终止：远程目录检查失败", "ERROR", log_file)
        sys.exit(1)

    _log("=" * 80, log_file=log_file)
    _log("开始下载文件...", log_file=log_file)
    start_time = time.time()

    if use_tar_mode:
        success = transfer_with_tar(
            remote_host, remote_dir, local_base_dir, ssh_key_path,
            tar_use_compression, confirm_delete, log_file)
    else:
        success = transfer_with_rsync(
            remote_host, remote_dir, local_base_dir, ssh_key_path,
            confirm_delete=confirm_delete, log_file=log_file)

    elapsed = time.time() - start_time
    _log(f"总耗时: {elapsed:.1f}s ({elapsed/60:.1f}min)", log_file=log_file)

    if not success:
        _log("任务终止：文件下载失败", "ERROR", log_file)
        sys.exit(1)

    _log("=" * 80, log_file=log_file)
    if use_tar_mode:
        _log("✓✓✓ 任务完成：文件已下载并解压，远程文件和tar已删除，本地tar已清理", "SUCCESS", log_file)
    else:
        _log("✓✓✓ 任务完成", "SUCCESS", log_file)
    _log("=" * 80, log_file=log_file)


def download_report_only(task_name, platform="NCI", local_base_dir="../../output",
                         log_file="rsync_download.log"):
    """只下载 DATA_REPORT 目录（HTML报告及相关文件），不删除远程。

    使用 rsync include/exclude 规则，仅同步每个 Run 下 output/{timestamp}/DATA_REPORT/ 内容。

    Args:
        task_name:     远程 output 目录下的文件夹名（等同于 rsync_and_delete 的 task_name）
        platform:      "NCI" 或 "HPC"
        local_base_dir: 本地下载基础目录
        log_file:      日志文件路径
    """
    cfg = ssh_config(platform)
    remote_host = f"{cfg['linux_username']}@{cfg['linux_host']}"
    remote_dir  = f"{cfg['project_dir']}/{task_name}"
    ssh_key_path = cfg['private_key_path']

    _log("=" * 80, log_file=log_file)
    _log(f"只下载报告文件  [平台: {platform}]  [任务: {task_name}]", log_file=log_file)
    _log(f"远程目录: {remote_dir}", log_file=log_file)
    _log("=" * 80, log_file=log_file)

    if not check_remote_dir(remote_host, remote_dir, ssh_key_path, log_file):
        _log("任务终止：远程目录检查失败", "ERROR", log_file)
        return False

    # rsync 过滤规则：遍历所有目录，只保留 DATA_REPORT 下的内容
    rsync_options = [
        "-avz",
        "--progress",
        "--partial",
        "--timeout=300",
        "--include=*/",            # 允许进入所有子目录（遍历用）
        "--include=DATA_REPORT/**", # 保留 DATA_REPORT 及其全部内容
        "--exclude=*",             # 排除其他所有文件
        "--prune-empty-dirs",      # 不在本地创建空目录
    ]

    start_time = time.time()
    cmd = _build_rsync_command(remote_host, remote_dir, local_base_dir,
                               ssh_key_path, rsync_options, log_file)
    success = _run_rsync(cmd, log_file)
    elapsed = time.time() - start_time

    _log(f"耗时: {elapsed:.1f}s ({elapsed/60:.1f}min)", log_file=log_file)
    if success:
        _log("✓✓✓ 报告文件下载完成", "SUCCESS", log_file)
    else:
        _log("✗ 报告文件下载失败", "ERROR", log_file)
    return success
