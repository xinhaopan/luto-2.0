import os
import sys

if sys.platform == 'win32' and sys.version_info >= (3, 8):
    try:
        # 这是我们正确的DLL文件夹路径
        dll_path = r"F:\xinhao\miniforge\envs\luto-minimal\Library\bin"

        # 发出直接命令，将此路径添加到Python的DLL搜索列表
        os.add_dll_directory(dll_path)

        # 打印成功信息，让我们知道它执行了
        print(f"✅ [成功] 已将高优先级DLL路径添加到解释器: {dll_path}")

    except (FileNotFoundError, AttributeError) as e:
        print(f"❌ [警告] 添加DLL路径失败: {e}。继续尝试，但可能会遇到问题。")

print("\n现在开始导入 rioxarray...")

try:
    from osgeo import gdal
    import rioxarray
    print("✅ [成功] rioxarray 导入成功！")

except ImportError as e:
    print(f"❌ [错误] 导入 rioxarray 失败: {e}。请检查是否已正确安装 rioxarray。")
    sys.exit(1)