import os
import shutil

def move_and_rename_files(root_path):
    files_to_fix = [
        "data_Production_overview_demand_type.js",
        "data_Production_overview_Domestic.js",
        "data_Production_overview_Exports.js",
        "data_Production_overview_Feed.js",
        "data_Production_overview_Imports.js"
    ]
    # 遍历所有子文件夹
    for dirpath, dirnames, filenames in os.walk(root_path):
        # 只在文件夹下操作，不在 data 目录本身再移动
        if os.path.basename(dirpath) == "data":
            continue
        # 目标 data 子目录
        target_dir = os.path.join(dirpath, "data")
        os.makedirs(target_dir, exist_ok=True)
        for fname in files_to_fix:
            src_path = os.path.join(dirpath, fname)
            if os.path.exists(src_path):
                new_name = fname.replace("data_", "")
                dst_path = os.path.join(target_dir, new_name)
                shutil.move(src_path, dst_path)
                print(f"Moved {src_path} -> {dst_path}")
            else:
                # 可选：只提示未找到
                pass


# 用法：输入你的路径
if __name__ == "__main__":
    input_path = r'output\20250922_Paper2_Results_HPC_test'
    move_and_rename_files(input_path)