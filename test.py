import rioxarray as rxr
import os
import rasterio.errors
proj_lib_path = r"F:\xinhao\miniforge\envs\luto\Library\share\proj"
os.environ['PROJ_LIB'] = proj_lib_path
# --- 请确保这个路径是正确的 ---
file_path = os.path.join('input', "NLUM_2010-11_mask.tif")

print(f"开始测试文件: {file_path}")

import rioxarray as rxr
import os
import rasterio.errors

# --- 请确保这个路径是正确的 ---
file_path = os.path.join('input', "NLUM_2010-11_mask.tif")

print(f"开始测试文件: {file_path}")

# 检查文件是否存在
if not os.path.exists(file_path):
    print(f"--- 错误: 文件未找到! ---")
    print(f"请确认路径 '{file_path}' 是否正确。")
else:
    try:
        # 1. 使用 rioxarray 打开栅格文件
        # 这会返回一个 xarray.DataArray 对象，上面有 .rio 访问器
        print("步骤 1: 正在打开文件...")
        rxr_arr = rxr.open_rasterio(file_path)
        print(f"文件打开成功。")
        print(f"文件的原始坐标系 (CRS): {rxr_arr.rio.crs}")
        print(f"原始尺寸: {rxr_arr.shape}")

        # 2. 尝试重投影到 'EPSG:3857'
        print("\n步骤 2: 正在尝试重投影到 'EPSG:3857'...")
        reprojected_arr = rxr_arr.rio.reproject('EPSG:3857')

        # 3. 如果代码能执行到这里，说明重投影成功
        print("\n--- success ---")
        print("文件已成功重投影。")
        print(f"新的坐标系 (CRS): {reprojected_arr.rio.crs}")
        print(f"新的尺寸: {reprojected_arr.shape}")

    except rasterio.errors.CRSError as e:
        # 4. 如果捕获到 CRSError，说明是坐标系解析失败
        print("\n--- 失败! ---")
        print("重投影失败，因为无法解析文件中的坐标系(CRS)。")
        print("这证实了文件元数据中存储的WKT字符串是无效或不受支持的。")
        print("\n捕获到的原始错误信息:")
        print(e)

    except Exception as e:
        # 5. 捕获其他可能的意外错误
        print(f"\n--- 发生未知错误! ---")
        print(e)
