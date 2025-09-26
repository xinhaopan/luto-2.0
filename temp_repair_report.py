import os
import shutil

def move_and_rename_files(root_path):
    files_to_fix = [
        "data_Production_overview_demand_type.js",
        "data_Production_overview_Domestic.js",
        "data_Production_overview_Exports.js",
        "data_Production_overview_Feed.js",
        "data_Production_overview_Imports.js",
    ]

    # topdown=True 以便剪枝 dirnames，阻止进入任何 data 目录
    for dirpath, dirnames, filenames in os.walk(root_path, topdown=True):
        # 不进入名为 "data" 的目录
        dirnames[:] = [d for d in dirnames if d != "data"]

        # 仅当当前目录下存在需要移动的文件时再创建 data/
        present = [f for f in files_to_fix if os.path.exists(os.path.join(dirpath, f))]
        if not present:
            continue

        target_dir = os.path.join(dirpath, "data")
        os.makedirs(target_dir, exist_ok=True)

        for fname in present:
            src_path = os.path.join(dirpath, fname)
            new_name = fname.replace("data_", "")
            dst_path = os.path.join(target_dir, new_name)
            # 如果目标已存在，可按需选择覆盖/跳过，这里选择覆盖
            if os.path.exists(dst_path):
                os.remove(dst_path)
            shutil.move(src_path, dst_path)
            print(f"Moved {src_path} -> {dst_path}")

# 用法：输入你的路径
if __name__ == "__main__":
    input_path = r'output\20250922_Paper2_Results_NCI'
    move_and_rename_files(input_path)