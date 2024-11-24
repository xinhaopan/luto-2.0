import subprocess
import os
import time

# 指定目标目录和脚本路径
file_dirs = [
    # '8_BIO_0_3_GHG_1_8C_67_5',
    # '7_BIO_0_5_GHG_1_5C_50_6',
    # '7_BIO_0_3_GHG_1_5C_50_5',
    # '6_BIO_0_3_GHG_1_5C_67_6',
    # '6_BIO_0_3_GHG_1_5C_67_5',
    # '0_GHG_1_8C_67',
    '1_GHG_1_8C_67'
]

for file_dir in file_dirs:
    print(f"Running script for {file_dir}...")

    # 记录开始时间
    start_time = time.time()

    target_dir = f"../output/{file_dir}"
    script_name = "temp_write.py"

    # 确保输出目录存在
    output_dir = os.path.join(target_dir, "output")
    os.makedirs(output_dir, exist_ok=True)

    stdout_path = os.path.join(output_dir, "write_stdout_log.txt")
    stderr_path = os.path.join(output_dir, "write_stderr_log.txt")

    # 打开文件以实时写入
    with open(stdout_path, "w") as stdout_file, open(stderr_path, "w") as stderr_file:
        # 构造命令并指定工作目录
        command = ["python", script_name]
        process = subprocess.Popen(
            command,
            cwd=target_dir,
            stdout=stdout_file,  # 实时写入标准输出到文件
            stderr=stderr_file,  # 实时写入标准错误到文件
            text=True
        )
        # 等待子进程完成
        process.wait()

    # 记录结束时间
    end_time = time.time()
    elapsed_time = (end_time - start_time) / 60  # 用时（分钟）

    # 确认执行完成
    print(f"Elapsed time for {file_dir}: {elapsed_time:.2f} minutes")
