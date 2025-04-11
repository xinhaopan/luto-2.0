import os
import paramiko
import pandas as pd
from tqdm import tqdm  # 引入 tqdm 用于进度条显示

from tools.config import linux_host, linux_username, linux_port, private_key_path
from tools.data_helper import  ensure_directory_exists, get_first_folder_and_download

csv_name = "setting_0410_test_weight1.csv"
csv_path = "../../tasks_run/Custom_runs/"
csv_file = os.path.join(csv_path, csv_name)
df = pd.read_csv(csv_file)
file_names = df.columns[2:]

target_file_names = ["data_with_solution.gz","model_run_settings.txt"]

# 创建 SSH 连接和 SFTP 会话
private_key = paramiko.RSAKey.from_private_key_file(private_key_path)
transport = paramiko.Transport((linux_host, linux_port))
transport.connect(username=linux_username, pkey=private_key)
sftp = paramiko.SFTPClient.from_transport(transport)

try:
    # 遍历文件名并显示进度条
    with tqdm(total=len(file_names), desc="Processing files", unit="file") as progress_bar:
        for file_name in file_names:
            file_name = file_name.replace('.', '_')
            remote_base_path = f"/g/data/jk53/LUTO_XH/LUTO2/output/{file_name}/output"
            local_download_path = os.path.join("../../../output", f'{file_name}', 'output')
            # 调用函数下载文件
            get_first_folder_and_download(
                sftp,
                remote_base_path,
                target_file_names,
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