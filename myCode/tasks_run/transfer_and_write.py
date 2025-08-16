from tools.transfer_helper import download_all_data_gz
from tools.write_helper import write_repeat
import time
from joblib import Parallel, delayed


print("starting transfer and write process...")
# time.sleep(60*60*1.5)
print("开始下载数据...")
file_names = ["20250812_price_task"]

def process_file(file_name):
    try:
        print(f"\n{file_name}")
        # download_all_data_gz(file_name,'HPC')
        print(f"\n{file_name}: 数据下载完成，开始写入输出...")
        write_repeat(f"../../output/{file_name}", force=False,write_threads=3)
        print(f"\n{file_name}: 输出写入完成。")
    except Exception as e:
        print(f"{file_name}: 处理过程中发生错误: {e}")
        return file_name
    return file_name


for file_name in file_names:
    process_file(file_name)