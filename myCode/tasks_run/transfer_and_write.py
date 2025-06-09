from transfer_gz import download_all_data_gz
from writes_repeat import write_repeat
import time

print("starting transfer and write process...")
# time.sleep(60*60*7.1)  # 等待7小时，确保服务器上的数据已经更新

print("开始下载数据...")
file_name = "20250608_Paper1_results_test_BIO"
download_all_data_gz(file_name)
print("数据下载完成，开始写入输出...")
write_repeat(f"../../output/{file_name}")
print("输出写入完成。")