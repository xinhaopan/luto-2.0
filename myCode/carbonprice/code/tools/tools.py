from typing import Union
import glob
import shutil, tempfile, time, random
import xarray as xr
from filelock import FileLock, Timeout
import numpy as np
import rasterio
import os
import zipfile
from io import BytesIO
import dill
import gzip

import tools.config as config


def get_year(parent_dir):
    """
    遍历指定目录，找到所有以 "out_" 开头的文件夹，并提取年份。

    Args:
        parent_dir (str): 要搜索的父目录的路径。

    Returns:
        list[int]: 一个包含所有有效年份的整数列表，并按升序排序。
                   如果目录不存在或没有找到匹配的文件夹，则返回一个空列表。
    """
    # 检查父目录是否存在
    if not os.path.isdir(parent_dir):
        print(f"错误: 目录 '{parent_dir}' 不存在或不是一个有效的目录。")
        return []

    years_list = []
    # 遍历父目录下的所有项目（文件和文件夹）
    for item_name in os.listdir(parent_dir):
        # 构造完整的项目路径
        full_path = os.path.join(parent_dir, item_name)

        # 检查该项目是否是一个文件夹，并且其名称是否以 "out_" 开头
        if os.path.isdir(full_path) and item_name.startswith("out_"):
            # 提取 "out_" 后面的部分作为年份字符串
            # "out_" 长度为 4，所以我们从第5个字符开始切片
            year_str = item_name[4:]

            # 尝试将提取的字符串转换为整数，以确保它是一个有效的年份
            # 如果转换失败（例如，文件夹名为 "out_final"），则忽略它
            try:
                year_int = int(year_str)
                years_list.append(year_int)
            except ValueError:
                # 转换失败，说明 "out_" 后面的不是纯数字，静默忽略
                print(f"提示: 忽略无效格式的文件夹 '{item_name}'")
                continue

    # 返回排序后的年份列表，使其更加整洁
    return sorted(years_list)


def get_path(task_name, path_name):
    """
    获取指定路径下的输出子目录。
    - path_name: 任务名称或路径名称
    - 返回: 输出子目录的完整路径
    """
    output_path = f"../../../output/{task_name}/{path_name}/output"
    try:
        if os.path.exists(output_path):
            subdirectories = os.listdir(output_path)
            # numeric_starting_subdir = [s for s in subdirectories if s[0].isdigit()][0]
            numeric_starting_subdir = [s for s in subdirectories if "2010-2050" in s][0]
            subdirectory_path = os.path.join(output_path, numeric_starting_subdir)
            return subdirectory_path
        else:
            raise FileNotFoundError(f"The specified output path does not exist: {output_path}")
    except (IndexError, FileNotFoundError) as e:
        print(f"Error occurred while getting path for {path_name}: {e}")
        print(
            f"Current directory content for {output_path}: {os.listdir(output_path) if os.path.exists(output_path) else 'Directory not found'}")




def save2nc(
        in_xr,
        save_path: str,
        *,
        engine: str = "h5netcdf",
        allow_overwrite: bool = True,
        compress: bool = True,
        lock_timeout: int = 600,
        lock_path: str | None = None,
        compute_before_write: bool = True,
        max_retries: int = 5,
        retry_delay: float = 1.0
):
    # 目录
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

    # 统一为 DataArray
    if isinstance(in_xr, xr.Dataset):
        if len(in_xr.data_vars) != 1:
            raise ValueError(f"输入 Dataset 含 {len(in_xr.data_vars)} 个变量，需单变量。")
        var_name = next(iter(in_xr.data_vars))
        da = in_xr[var_name]
    elif isinstance(in_xr, xr.DataArray):
        da = in_xr
        var_name = da.name or "data"
    else:
        raise TypeError("in_xr 必须是 xarray.DataArray 或 单变量 xarray.Dataset")

    if da.name != var_name:
        da = da.rename(var_name)

    # 仅保留维度坐标，剔除非维度坐标（避免写出失败）
    coords_to_drop = set(da.coords) - set(da.dims)
    if coords_to_drop:
        da = da.drop_vars(coords_to_drop, errors="ignore")

    # 预先计算，避免写出时仍有 dask 依赖
    if compute_before_write:
        da = da.load()

    # 编码
    enc = {var_name: {"dtype": "float32"}}
    if compress:
        enc[var_name].update({"zlib": True, "complevel": 4})
    if hasattr(da.data, "chunks") and da.data.chunks is not None:
        enc[var_name]["chunksizes"] = da.chunks

    # ✅ 关键修改：锁文件管理
    lockfile = lock_path or (save_path + ".lock")
    save_dir = os.path.dirname(save_path) or "."

    # ✅ 修改1：创建FileLock对象但不立即使用
    file_lock = FileLock(lockfile, timeout=lock_timeout)

    try:
        # ✅ 修改2：使用acquire()和release()方法
        file_lock.acquire()
        try:
            # 二次检查
            if os.path.exists(save_path) and not allow_overwrite:
                raise FileExistsError(f"目标已存在且不允许覆盖：{save_path}")

            # 独立临时文件路径（先关闭句柄，避免 Windows 上无法 move）
            with tempfile.NamedTemporaryFile(dir=save_dir, suffix=".nc", delete=False) as tmp_file:
                temp_path = tmp_file.name

            try:
                # 写入临时文件（xarray 会在函数返回时关闭句柄）
                da.astype("float32").to_netcdf(path=temp_path, engine=engine, encoding=enc)

                # 原子移动（带重试，处理偶发 PermissionError/杀毒软件扫描占用等）
                for attempt in range(max_retries):
                    try:
                        shutil.move(temp_path, save_path)
                        break
                    except PermissionError:
                        if attempt < max_retries - 1:
                            sleep_time = retry_delay * (1.5 ** attempt) + random.uniform(0, 0.5)
                            print(f"⚠️ 移动时权限被拒绝，{sleep_time:.2f}s 后重试...")
                            time.sleep(sleep_time)
                        else:
                            raise
            except Exception:
                if os.path.exists(temp_path):
                    try:
                        os.remove(temp_path)
                    except Exception:
                        pass
                raise
        finally:
            # ✅ 修改3：确保释放锁
            try:
                file_lock.release()
            except Exception as e:
                print(f"警告：释放锁时出错: {e}")
    except Timeout:
        raise TimeoutError(f"获取写锁超时（{lock_timeout}s）：{lockfile}")
    finally:
        # ✅ 修改4：强制清理锁文件
        try:
            if os.path.exists(lockfile):
                os.remove(lockfile)
                print(f"已清理锁文件: {lockfile}")
        except Exception as e:
            print(f"警告：清理锁文件失败 {lockfile}: {e}")



def filter_all_from_dims(ds: Union[xr.Dataset, xr.DataArray]) -> Union[xr.Dataset, xr.DataArray]:
    """
    从 xarray 对象的所有维度中筛选并移除值为 'ALL' 的坐标。

    该函数会遍历对象的所有维度，检查其坐标是否包含字符串 'ALL'。
    如果包含，则只选择不等于 'ALL' 的部分。

    Args:
        ds (Union[xr.Dataset, xr.DataArray]): 需要进行筛选的 xarray 数据集或数据数组。

    Returns:
        Union[xr.Dataset, xr.DataArray]: 一个新的、经过筛选的 xarray 对象。
    """
    # 将输入对象作为筛选的起点
    filtered_ds = ds

    # 遍历所有维度名称
    for dim_name in ds.dims:
        # 安全检查：确保维度有关联的坐标，并且坐标是字符串类型
        if dim_name in ds.coords and ds[dim_name].dtype.kind in ['U', 'S', 'O']:
            # 检查坐标值中是否含有 'ALL'
            if np.isin(ds[dim_name].values, ['ALL']).any():
                # 使用 .sel() 和布尔索引来选择不等于 'ALL' 的部分
                filtered_ds = filtered_ds.sel({dim_name: filtered_ds[dim_name] != 'ALL'})

    return filtered_ds


def create_xarray(years, base_path, env_category, env_name, mask=None,
                  engine="h5netcdf",
                  cell_dim="cell", cell_chunk="auto",
                  year_chunk=1, parallel=False):
    """
    以 year 维度拼接多个年度 NetCDF，懒加载+分块，避免过多文件句柄。
    """
    file_paths = [
        os.path.join(base_path, str(env_category), str(y), f"xr_{env_name}_{y}.nc")
        for y in years
    ]
    missing = [p for p in file_paths if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(f"以下文件未找到:\n" + "\n".join(missing))

    # 从文件名提取实际年份，确保坐标与文件顺序一致
    valid_years = [int(os.path.basename(p).split("_")[-1].split(".")[0]) for p in file_paths]

    ds = xr.open_mfdataset(
        file_paths,
        engine=engine,
        combine="nested",  # 明确“按给定顺序拼接”
        concat_dim="year",  # 新增 year 维度
        parallel=parallel,  # 一般 False 更稳，避免句柄并发
        chunks={cell_dim: cell_chunk, "year": year_chunk}  # year=1，cell 分块
    ).assign_coords(year=valid_years)

    if mask is not None:
        ds = ds.where(mask, other=0)  # 使用掩码，非掩码区域设为 0

    return ds


# def nc_to_tif(data, da, tif_path: str, nodata_value: float = -9999.0):
#     """
#     将 xarray DataArray 转换为 GeoTIFF，正确处理 nodata 和掩膜
#     """
#     # 仅支持 1D 'cell'
#     if "cell" not in da.dims or len(da.dims) != 1:
#         raise ValueError(f"维度是 {da.dims}，只支持一维 'cell'。")
#
#     arr1d = da.values.astype(np.float32)
#
#     # 创建有效性掩膜（在转换前）
#     valid_mask_1d = np.isfinite(arr1d)
#
#     # 将无效值替换为 nodata
#     arr1d = np.where(valid_mask_1d, arr1d, nodata_value)
#
#     # 铺回 2D
#     full_res_raw = (arr1d.size == data.LUMAP_NO_RESFACTOR.size)
#     if full_res_raw:
#         geo_meta = data.GEO_META_FULLRES.copy()
#         arr_2d = np.full(data.NLUM_MASK.shape, nodata_value, dtype=np.float32)
#         valid_mask_2d = np.zeros(data.NLUM_MASK.shape, dtype=bool)
#         # 只在有效位置填充数据和掩膜
#         mask_indices = np.where(data.NLUM_MASK)
#         arr_2d[mask_indices] = arr1d
#         valid_mask_2d[mask_indices] = valid_mask_1d
#     else:
#         geo_meta = data.GEO_META.copy()
#         arr_2d = data.LUMAP_2D_RESFACTORED.copy().astype(np.float32)
#         valid_mask_2d = np.ones_like(arr_2d, dtype=bool)
#
#         # 标记掩膜区域为无效
#         mask_condition = (arr_2d == data.MASK_LU_CODE) | (arr_2d == data.NODATA)
#         valid_mask_2d[mask_condition] = False
#         arr_2d[mask_condition] = nodata_value
#
#         # 填充有效数据
#         data_condition = ~mask_condition
#         data_indices = np.where(data_condition)
#         arr_2d[data_indices] = arr1d
#         valid_mask_2d[data_indices] = valid_mask_1d
#
#     # 确保所有无效位置都设为 nodata
#     arr_2d = np.where(valid_mask_2d, arr_2d, nodata_value).astype(np.float32)
#
#     # 配置元数据
#     meta = geo_meta.copy()
#     meta.update(
#         count=1,
#         dtype="float32",
#         nodata=nodata_value,  # 关键：设置 nodata 值
#         compress="deflate",
#         predictor=3,
#         tiled=True,
#         blockxsize=256,
#         blockysize=256,
#     )
#
#     os.makedirs(os.path.dirname(tif_path), exist_ok=True)
#
#     with rasterio.open(tif_path, "w", **meta) as dst:
#         dst.write(arr_2d, 1)
#         # 写入内部掩膜：255=有效，0=无效
#         dst.write_mask((valid_mask_2d.astype(np.uint8) * 255))
#
#     print(f"✅ 已保存: {tif_path}")

def nc_to_tif(data, da, tif_path: str, nodata_value: float = -9999.0):
    """
    将 xarray DataArray 转换为 GeoTIFF，正确处理 nodata 和掩膜
    """
    # 仅支持 1D 'cell'
    if "cell" not in da.dims or len(da.dims) != 1:
        raise ValueError(f"维度是 {da.dims}，只支持一维 'cell'。")

    arr1d = da.values.astype(np.float32)

    # 创建有效性掩膜（在转换前）
    valid_mask_1d = np.isfinite(arr1d)

    # 将无效值替换为 nodata
    arr1d = np.where(valid_mask_1d, arr1d, nodata_value)

    # 铺回 2D
    full_res_raw = (arr1d.size == data.LUMAP_NO_RESFACTOR.size)
    if full_res_raw:
        geo_meta = data.GEO_META_FULLRES.copy()
        arr_2d = np.full(data.NLUM_MASK.shape, nodata_value, dtype=np.float32)
        valid_mask_2d = np.zeros(data.NLUM_MASK.shape, dtype=bool)
        # 只在有效位置填充数据和掩膜
        mask_indices = np.where(data.NLUM_MASK)
        arr_2d[mask_indices] = arr1d
        valid_mask_2d[mask_indices] = valid_mask_1d
    else:
        geo_meta = data.GEO_META.copy()
        arr_2d = data.LUMAP_2D_RESFACTORED.copy().astype(np.float32)
        valid_mask_2d = np.ones_like(arr_2d, dtype=bool)

        # 标记掩膜区域为无效
        mask_condition = (arr_2d == data.MASK_LU_CODE) | (arr_2d == data.NODATA)
        valid_mask_2d[mask_condition] = False
        arr_2d[mask_condition] = nodata_value

        # 填充有效数据
        data_condition = ~mask_condition
        data_indices = np.where(data_condition)
        arr_2d[data_indices] = arr1d
        valid_mask_2d[data_indices] = valid_mask_1d

    # 确保所有无效位置都设为 nodata
    arr_2d = np.where(valid_mask_2d, arr_2d, nodata_value).astype(np.float32)

    # 配置元数据
    meta = geo_meta.copy()
    meta.update(
        count=1,
        dtype="float32",
        nodata=nodata_value,
        compress="deflate",
        predictor=3,
        tiled=True,
        blockxsize=256,
        blockysize=256,
    )

    os.makedirs(os.path.dirname(tif_path), exist_ok=True)

    # 删除已存在的文件及其关联文件
    if os.path.exists(tif_path):
        try:
            # 删除主文件
            os.remove(tif_path)
            # 删除常见的关联文件
            for ext in ['.aux.xml', '.ovr', '.msk', '.tfw', '.prj']:
                aux_file = tif_path + ext
                if os.path.exists(aux_file):
                    os.remove(aux_file)
            # 删除可能的其他扩展名
            base = os.path.splitext(tif_path)[0]
            for ext in ['.aux.xml', '.ovr', '.msk', '.tfw', '.prj']:
                aux_file = base + ext
                if os.path.exists(aux_file):
                    os.remove(aux_file)
        except Exception as e:
            print(f"⚠️ 删除旧文件时出错: {e}")
            # 继续尝试写入

    with rasterio.open(tif_path, "w", **meta) as dst:
        dst.write(arr_2d, 1)
        # 写入内部掩膜：255=有效，0=无效
        dst.write_mask((valid_mask_2d.astype(np.uint8) * 255))

    print(f"✅ 已保存: {tif_path}")

def get_data_RES(output_path="output"):
    zip_path = os.path.join(output_path, 'Run_Archive.zip')
    if not os.path.isfile(zip_path):
        raise FileNotFoundError(f"未找到指定的 zip 文件: {zip_path}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        matches = [name for name in zf.namelist() if glob.fnmatch.fnmatch(os.path.basename(name), "Data_RES*.gz")]
        if not matches:
            raise FileNotFoundError("在 zip 文件中未找到匹配 'Data_RES*.gz' 的文件。")
        # 读取 zip 内部 gz 文件内容为 bytes
        gz_bytes = zf.read(matches[0])
        # 用 BytesIO 包装后用 gzip.open 读取内容
        with gzip.open(BytesIO(gz_bytes), 'rb') as f:
            data = dill.load(f)
        return data
