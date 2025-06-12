from transfer_gz import download_all_data_gz
from writes_repeat import write_repeat
import time
from joblib import Parallel, delayed

print("starting transfer and write process...")
# time.sleep(60*60*2)  # 等待7小时，确保服务器上的数据已经更新

print("开始下载数据...")
file_names = ["20250608_Paper1_results_windows_BIO3"]

def process_file(file_name):
    try:
        print(f"\n{file_name}")
        # download_all_data_gz(file_name)
        print(f"\n{file_name}: 数据下载完成，开始写入输出...")
        write_repeat(f"../../output/{file_name}")
        print(f"\n{file_name}: 输出写入完成。")
    except Exception as e:
        print(f"{file_name}: 处理过程中发生错误: {e}")
        return file_name
    return file_name

# results = Parallel(n_jobs=3)(
#     delayed(process_file)(file_name) for file_name in file_names
# )
# for name in results:
#     print(f"\n{name} 处理完毕。")
#
for file_name in file_names:
    process_file(file_name)