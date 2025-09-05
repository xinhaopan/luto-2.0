import os
import sys


import os


def rename_xr_files(root_dir):
    """
    遍历目录及子目录，把所有以 'xr_xr_' 开头的文件改名为以 'xr_' 开头。
    例如: xr_xr_ABC.nc -> xr_ABC.nc
    """
    abs_root_dir = os.path.abspath(root_dir)
    print(f"--- 开始诊断 ---")
    print(f"工作目录: '{os.getcwd()}'")
    print(f"搜索根目录: '{abs_root_dir}'")

    if not os.path.isdir(abs_root_dir):
        print(f"[错误] 指定的目录 '{abs_root_dir}' 不存在。")
        return

    found_files_to_rename = False

    for dirpath, _, filenames in os.walk(abs_root_dir):
        for filename in filenames:
            if filename.startswith("xr_xr_"):
                found_files_to_rename = True
                old_path = os.path.join(dirpath, filename)
                new_filename = "xr_" + filename[len("xr_xr_"):]
                new_path = os.path.join(dirpath, new_filename)

                print(f"[发现] {old_path}")
                print(f"  -> 重命名为 {new_path}")

                try:
                    os.rename(old_path, new_path)
                    print("  -> [成功]")
                except OSError as e:
                    print(f"  -> [失败] {e}")

    print("\n--- 诊断结束 ---")
    if not found_files_to_rename:
        print("最终结论: 没有发现以 'xr_xr_' 开头的文件。")
    else:
        print("最终结论: 已完成重命名操作。")


if __name__ == "__main__":
    # 设置要搜索的根目录。'.' 代表当前目录。
    # 你也可以指定一个绝对路径来确保正确性，例如:
    # target_directory = 'D:/my_data'  # Windows 示例
    # target_directory = '/home/xinhaopan/data' # Linux/macOS 示例
    target_directory = '../../../output/20250831_Price_Task_NCI/carbon_price/0_base_data'

    rename_xr_files(target_directory)