import os
import shutil

def delete_data_report_contents(root_dir):
    """
    递归查找root_dir下所有DATA_REPORT文件夹，并删除其内容（保留空文件夹本身）
    """
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for dirname in dirnames:
            if dirname == "DATA_REPORT":
                report_dir = os.path.join(dirpath, dirname)
                # 删除文件夹内所有内容
                for item in os.listdir(report_dir):
                    item_path = os.path.join(report_dir, item)
                    if os.path.isfile(item_path) or os.path.islink(item_path):
                        os.remove(item_path)
                    elif os.path.isdir(item_path):
                        shutil.rmtree(item_path)
                print(f"已清空：{report_dir}")

# 用法举例
root_dir = "output/20250919_Paper2_Results"
delete_data_report_contents(root_dir)