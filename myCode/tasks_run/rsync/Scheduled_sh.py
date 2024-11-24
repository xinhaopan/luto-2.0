import subprocess
import time
from datetime import datetime

# 定义要运行的 shell 脚本路径
shell_script_path = "sync_files.sh"

def run_script():
    """运行指定的 shell 脚本"""
    try:
        # 使用 subprocess 运行脚本
        subprocess.run(["bash", shell_script_path], check=True)
        print(f"{datetime.now()}: Script executed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"{datetime.now()}: Error while executing script: {e}")
    except Exception as e:
        print(f"{datetime.now()}: Unexpected error: {e}")

def main():
    while True:
        now = datetime.now()
        # 每小时的第2分钟执行脚本
        if now.minute == 2 and now.second == 0:
            run_script()
            # 等待一分钟，避免重复触发
            time.sleep(60)
        # 每秒钟检查一次时间
        time.sleep(1)

if __name__ == "__main__":
    main()
