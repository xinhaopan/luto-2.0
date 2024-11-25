import subprocess
import time
from datetime import datetime
import os
import socket

# 定义远程服务器信息和脚本路径
remote_host = "xp7241@gadi.nci.org.au"
remote_directory = "/g/data/jk53/LUTO_XH/LUTO2/myCode/tasks_run/rsync"
shell_script_path = "sync_files.sh"  # 远程服务器上的脚本名
log_file_path = "Scheduled_log.txt"

def log_message(message):
    """将消息记录到日志文件"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_file_path, "a") as log_file:
        log_file.write(f"{timestamp} - {message}\n")

def run_script():
    """通过 SSH 在远程服务器上运行指定的 shell 脚本"""
    try:
        log_message("Starting script execution on remote server.")
        ssh_command = (
            f"ssh {remote_host} 'cd {remote_directory} && bash {shell_script_path}'"
        )
        subprocess.Popen(ssh_command, shell=True)
        log_message("Script submitted successfully on remote server.")
    except Exception as e:
        log_message(f"Unexpected error during remote script submission: {e}")

def clear_logs():
    """每周六晚上 3 点清理日志文件"""
    try:
        with open(log_file_path, "w") as log_file:
            log_file.write("")  # 清空文件内容
        log_message("Logs cleared successfully.")
    except Exception as e:
        log_message(f"Error clearing logs: {e}")

def main():
    # 记录当前节点和进程信息
    hostname = socket.gethostname()
    pid = os.getpid()
    log_message(f"Starting script on node '{hostname}' with PID {pid}.")
    while True:
        now = datetime.now()
        # 每小时的第 3 分钟执行脚本
        if now.minute == 3:
            run_script()
            # 等待一分钟，避免重复触发
            time.sleep(60)
        # 每周六晚上 3 点清理日志
        if now.weekday() == 5 and now.hour == 3 and now.minute == 2:
            clear_logs()
            # 等待一分钟，避免重复触发
            time.sleep(60)
        # 每秒钟检查一次时间
        time.sleep(1)

if __name__ == "__main__":
    main()
