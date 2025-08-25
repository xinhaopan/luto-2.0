import os
import sys


print("\n现在开始导入 rioxarray...")

try:
    # from osgeo import gdal
    import rioxarray
    print("✅ [成功] rioxarray 导入成功！")

except ImportError as e:
    print(f"❌ [错误] 导入 rioxarray 失败: {e}。请检查是否已正确安装 rioxarray。")
    sys.exit(1)