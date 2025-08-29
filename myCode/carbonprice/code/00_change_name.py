import os
import sys


def rename_ghg_files(root_dir):
    """
    在指定目录下查找并重命名文件（带有详细的诊断输出）。
    查找模式: root_dir/.../out_年份/xr_GHG_ag.nc
    重命名为: root_dir/.../out_年份/xr_GHG_ag_年份.nc
    """
    # 【诊断1】打印脚本的绝对根搜索路径
    abs_root_dir = os.path.abspath(root_dir)
    print(f"--- 开始诊断 ---")
    print(f"脚本当前的工作目录是: '{os.getcwd()}'")
    print(f"将要搜索的绝对路径是: '{abs_root_dir}'")

    if not os.path.isdir(abs_root_dir):
        print(f"\n[错误] 指定的目录 '{abs_root_dir}' 不存在。请检查路径。")
        return

    print("-" * 20)

    found_folders_to_check = False
    found_files_to_rename = False

    # 遍历根目录下的所有文件和文件夹
    for dirpath, dirnames, filenames in os.walk(abs_root_dir):
        # 检查当前文件夹的名称是否以 'out_' 开头
        base_dirname = os.path.basename(dirpath)
        if base_dirname.startswith("out_"):
            found_folders_to_check = True
            print(f"\n[发现目录] 找到一个可能是目标的文件夹: '{dirpath}'")

            # 尝试从文件夹名称中提取年份
            parts = base_dirname.split('_')
            if len(parts) == 2 and parts[1].isdigit():
                year = parts[1]
                print(f"  -> 从中提取到年份: '{year}'")

                original_filename = "xr_GHG_ag.nc"
                original_filepath = os.path.join(dirpath, original_filename)

                # 【诊断2】检查原始文件是否存在
                if os.path.isfile(original_filepath):
                    found_files_to_rename = True
                    print(f"  -> [文件确认] 原始文件 '{original_filepath}' 确实存在。")

                    new_filename = f"xr_GHG_ag_{year}.nc"
                    new_filepath = os.path.join(dirpath, new_filename)

                    # 执行重命名操作
                    try:
                        os.rename(original_filepath, new_filepath)
                        print(f"  -> [操作成功] 已重命名为: '{new_filepath}'")
                    except OSError as e:
                        print(f"  -> [操作失败] 尝试重命名 '{original_filepath}' 时发生错误: {e}")
                        print(f"  -> [重要提示] 这很可能是权限问题！请检查你是否有权限修改这个文件。")
                else:
                    print(f"  -> [文件未找到] 在此目录中没有找到名为 '{original_filename}' 的文件。")
            else:
                print(f"  -> 目录名 '{base_dirname}' 不符合 'out_年份' 的格式。")

    print("\n--- 诊断结束 ---")
    if not found_folders_to_check:
        print("最终结论: 在指定的路径下，没有找到任何以 'out_' 开头的文件夹。")
    elif not found_files_to_rename:
        print("最终结论: 找到了 'out_' 文件夹，但在其中没有找到名为 'xr_GHG_ag.nc' 的文件进行重命名。")
    else:
        print("最终结论: 脚本已执行完毕。请检查上面的日志确认操作结果。")


if __name__ == "__main__":
    # 设置要搜索的根目录。'.' 代表当前目录。
    # 你也可以指定一个绝对路径来确保正确性，例如:
    # target_directory = 'D:/my_data'  # Windows 示例
    # target_directory = '/home/xinhaopan/data' # Linux/macOS 示例
    target_directory = '../../../output/20250829_Price_Task_RES13'

    rename_ghg_files(target_directory)