from typing import Union, Optional, Iterable
import glob
import shutil, tempfile, time, random
import xarray as xr
from filelock import FileLock, Timeout
import numpy as np
import rasterio
import os
import joblib
import zipfile
import pandas as pd
import gc
import sys
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
        retry_delay: float = 1.0,
        encode_multi_index: bool = True,
        layer_dim: str = "layer"
):
    import cf_xarray as cfxr
    import pandas as pd

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

    # Dask chunking
    chunks_size = None
    if hasattr(in_xr, 'sizes'):
        chunks_size = {
            dim: (size if dim == 'cell' else 1)
            for dim, size in in_xr.sizes.items()
        }
        in_xr = in_xr.chunk(chunks_size)

    # 转换为 Dataset
    if isinstance(in_xr, xr.DataArray):
        var_name = in_xr.name or "data"
        ds = in_xr.to_dataset(name=var_name)
    elif isinstance(in_xr, xr.Dataset):
        if len(in_xr.data_vars) != 1:
            raise ValueError(f"输入 Dataset 含 {len(in_xr.data_vars)} 个变量，需单变量。")
        var_name = next(iter(in_xr.data_vars))
        ds = in_xr
    else:
        raise TypeError("in_xr 必须是 xarray.DataArray 或 单变量 xarray.Dataset")

    # MultiIndex 编码
    if encode_multi_index and layer_dim in ds.dims:
        if layer_dim in ds.indexes and isinstance(ds.indexes[layer_dim], pd.MultiIndex):
            ds = cfxr.encode_multi_index_as_compress(ds, layer_dim)

    # 去除非维度坐标
    for data_var in ds.data_vars:
        coords_to_drop = set(ds[data_var].coords) - set(ds[data_var].dims)
        if coords_to_drop:
            ds = ds.drop_vars(coords_to_drop, errors="ignore")

    # 预计算
    if compute_before_write:
        ds = ds.load()

    # ===== 编码设置 =====
    enc = {}

    # 数据变量编码
    for var in ds.data_vars:
        enc[var] = {"dtype": "float32"}
        if compress:
            enc[var].update({"zlib": True, "complevel": 4})

        if chunks_size is not None:
            chunk_tuple = tuple(
                chunks_size.get(dim, ds.sizes[dim])
                for dim in ds[var].dims
            )
            enc[var]["chunksizes"] = chunk_tuple

    # ✅ 关键修复：字符串坐标编码
    for coord in ds.coords:
        coord_data = ds.coords[coord]
        # 检查是否是字符串类型
        if coord_data.dtype.kind in ('U', 'S', 'O'):
            # 计算最大字符串长度
            try:
                if coord_data.size > 0:
                    max_len = max(len(str(x)) for x in coord_data.values.flat)
                    # 设置足够长的字符串类型
                    enc[coord] = {'dtype': f'U{max_len}'}
            except Exception:
                # 如果计算失败，转换为 object 类型
                ds = ds.assign_coords({coord: coord_data.astype(object)})

    # 文件锁和保存
    lockfile = lock_path or (save_path + ".lock")
    save_dir = os.path.dirname(save_path) or "."
    file_lock = FileLock(lockfile, timeout=lock_timeout)

    try:
        file_lock.acquire()
        try:
            if os.path.exists(save_path) and not allow_overwrite:
                raise FileExistsError(f"目标已存在且不允许覆盖：{save_path}")

            with tempfile.NamedTemporaryFile(dir=save_dir, suffix=".nc", delete=False) as tmp_file:
                temp_path = tmp_file.name

            try:
                ds.to_netcdf(path=temp_path, engine=engine, encoding=enc)

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
            try:
                file_lock.release()
            except Exception as e:
                print(f"警告：释放锁时出错: {e}")
    except Timeout:
        raise TimeoutError(f"获取写锁超时（{lock_timeout}s）：{lockfile}")
    finally:
        try:
            if os.path.exists(lockfile):
                os.remove(lockfile)
        except Exception as e:
            print(f"警告：清理锁文件失败 {lockfile}: {e}")

def _ensure_layer_multiindex_strict(
    da: xr.DataArray,
    layer_dim: str = "layer",
    level_order: Optional[Iterable[str]] = None,
) -> xr.DataArray:
    """
    严格确保 da 的 layer 是 MultiIndex。
    规则：
    - 如果 layer 已经是 MultiIndex：直接返回
    - 否则要求存在 coords(dims=(layer,)) 的 level 变量（如 am/lm/lu/source...）
      用 set_index(layer=[...]) 构造 MultiIndex
    - 如果没有 level 变量：直接报错（因为无法还原）
    """
    if layer_dim not in da.dims:
        return da

    idx = da.indexes.get(layer_dim, None)
    if isinstance(idx, pd.MultiIndex):
        return da

    layer_level_vars = [
        c for c in da.coords
        if c != layer_dim and da.coords[c].dims == (layer_dim,)
    ]
    if not layer_level_vars:
        raise ValueError(
            f"'{layer_dim}' is not a MultiIndex and no layer-level coords exist to rebuild it. "
            f"Available coords with dims=('layer',): {layer_level_vars}"
        )

    if level_order is not None:
        level_order = list(level_order)
        layer_level_vars = (
            [c for c in level_order if c in layer_level_vars]
            + [c for c in layer_level_vars if c not in level_order]
        )

    return da.set_index({layer_dim: layer_level_vars})


def filter_all_from_dims(
    obj: Union[xr.Dataset, xr.DataArray],
    layer_dim: str = "layer",
    strict_layer_multiindex: bool = True,
    layer_level_order: Optional[Iterable[str]] = None,
) -> Union[xr.Dataset, xr.DataArray]:
    """
    同时过滤：
    1) 正常 dims 中坐标值为 'ALL' 的部分
    2) 如果存在 (cell, layer) 且 'ALL' 出现在 layer-level coords（dims=('layer',)），也过滤掉这些 layer

    不使用 try；如果 strict 且 layer 无法成为 MultiIndex，则直接报错。
    """

    def _filter_da(da: xr.DataArray) -> xr.DataArray:
        out = da
        filter_words = ['ALL', 'AUSTRALIA']

        # ---------- A) 过滤真正 dims 中的 ALL ----------
        for filter_word in filter_words:
            for dim in list(out.dims):
                if dim in out.coords and out[dim].dtype.kind in ("U", "S", "O"):
                    vals = out[dim].values
                    if np.isin(vals, [filter_word]).any():
                        out = out.sel({dim: out[dim] != filter_word})

            # ---------- B) 过滤 layer-level coords 中的 ALL ----------
            if layer_dim in out.dims:
                layer_level_vars = [
                    c for c in out.coords
                    if c != layer_dim and out.coords[c].dims == (layer_dim,)
                ]
                if layer_level_vars:
                    if strict_layer_multiindex:
                        out = _ensure_layer_multiindex_strict(
                            out, layer_dim=layer_dim, level_order=layer_level_order
                        )

                    idx = out.indexes.get(layer_dim, None)
                    if strict_layer_multiindex and not isinstance(idx, pd.MultiIndex):
                        raise ValueError(
                            f"Expected '{layer_dim}' to be MultiIndex after ensure, got {type(idx)}"
                        )

                    if isinstance(idx, pd.MultiIndex):
                        keep = np.ones(len(idx), dtype=bool)
                        for lvl in idx.names:
                            keep &= (idx.get_level_values(lvl) != filter_word)
                        out = out.isel({layer_dim: keep})

        return out

    if isinstance(obj, xr.Dataset):
        new_vars = {v: _filter_da(obj[v]) for v in obj.data_vars}
        return xr.Dataset(new_vars, attrs=obj.attrs)

    return _filter_da(obj)


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


def nc_to_tif(data, da, tif_path: str, nodata_value: float = -9999.0,
              max_retries: int = 5, retry_delay: float = 2.0):
    """
    将 xarray DataArray 转换为 GeoTIFF，正确处理 nodata 和掩膜

    Parameters
    ----------
    data : 数据对象（包含地理元数据）
    da : xr.DataArray
        输入数据（必须是一维，维度名为 'cell'）
    tif_path : str
        输出 .tif 路径
    nodata_value : float
        NoData 值
    max_retries : int
        删除旧文件的最大重试次数
    retry_delay : float
        重试延迟（秒）
    """
    # 仅支持 1D 'cell'
    if "cell" not in da.dims or len(da.dims) != 1:
        raise ValueError(f"维度是 {da.dims}，只支持一维 'cell'。")

    arr = da.values.astype(np.float32)

    # ========== 根据数据长度判断类型并转换为 2D ==========
    # 参考 arr_to_xr 的逻辑
    if arr.size == data.LUMASK.size:
        # ===== 情况1: 全分辨率原始数据 (LUMASK.size = 6956407) =====
        geo_meta = data.GEO_META_FULLRES.copy()
        arr_2d = np.full(data.NLUM_MASK.shape, nodata_value, dtype=np.float32)

        # 创建有效性掩膜（1D）
        valid_mask_1d = np.isfinite(arr)
        arr_safe = np.where(valid_mask_1d, arr, nodata_value)

        # 使用 np.place 填充数据到 2D
        np.place(arr_2d, data.NLUM_MASK, arr_safe)

        # 创建 2D 有效性掩膜
        valid_mask_2d = np.zeros(data.NLUM_MASK.shape, dtype=bool)
        np.place(valid_mask_2d, data.NLUM_MASK, valid_mask_1d)

    elif arr.size == data.LUMASK.sum():
        # ===== 情况2: 部分有效的全分辨率数据 (LUMASK.sum()) =====
        # 【关键修复】先创建一个与 LUMASK.size 相同长度的全 0 数组
        arr_fulllen = np.zeros(data.NCELLS, dtype=np.float32)
        valid_mask_fulllen = np.zeros(data.NCELLS, dtype=bool)

        # 创建有效性掩膜
        valid_mask_1d = np.isfinite(arr)

        # 【关键】将 arr 填充到 arr_fulllen 的有效位置
        # data.MASK 是一个布尔数组，长度为 data.NCELLS
        # arr 的长度应该等于 data.MASK.sum()
        arr_fulllen[data.MASK] = np.where(valid_mask_1d, arr, nodata_value)
        valid_mask_fulllen[data.MASK] = valid_mask_1d

        # 再映射到 2D
        geo_meta = data.GEO_META_FULLRES.copy()
        arr_2d = np.full(data.NLUM_MASK.shape, nodata_value, dtype=np.float32)
        valid_mask_2d = np.zeros(data.NLUM_MASK.shape, dtype=bool)

        np.place(arr_2d, data.NLUM_MASK, arr_fulllen)
        np.place(valid_mask_2d, data.NLUM_MASK, valid_mask_fulllen)

    else:
        # ===== 情况3: 重采样数据 (data.NCELLS) =====
        geo_meta = data.GEO_META.copy()
        arr_2d = data.LUMAP_2D_RESFACTORED.copy().astype(np.float32)

        # 创建有效性掩膜
        valid_mask_1d = np.isfinite(arr)
        arr_safe = np.where(valid_mask_1d, arr, nodata_value)

        # 使用与 arr_to_xr 相同的方式填充
        arr_2d[*data.COORD_ROW_COL_RESFACTORED] = arr_safe

        # 创建 2D 有效性掩膜
        valid_mask_2d = np.ones(arr_2d.shape, dtype=bool)

        # 先将数据位置的掩膜设置为对应的有效性
        valid_mask_2d[*data.COORD_ROW_COL_RESFACTORED] = valid_mask_1d

        # 标记掩膜区域和 NODATA 区域为无效
        mask_condition = (arr_2d == data.MASK_LU_CODE) | (arr_2d == data.NODATA)
        valid_mask_2d[mask_condition] = False
        arr_2d[mask_condition] = nodata_value

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

    # ===== 确保目录存在 =====
    os.makedirs(os.path.dirname(tif_path), exist_ok=True)

    # ===== 使用文件锁 =====
    lock_path = tif_path + ".lock"
    file_lock = FileLock(lock_path, timeout=300)

    try:
        with file_lock:
            # 删除已存在的文件及其关联文件（带重试）
            if os.path.exists(tif_path):
                for attempt in range(max_retries):
                    try:
                        # 尝试修改权限
                        try:
                            os.chmod(tif_path, 0o777)
                        except:
                            pass

                        # 删除主文件
                        os.remove(tif_path)

                        # 删除常见的关联文件
                        for ext in ['.aux.xml', '.ovr', '.msk', '.tfw', '.prj']:
                            aux_file = tif_path + ext
                            if os.path.exists(aux_file):
                                try:
                                    os.remove(aux_file)
                                except:
                                    pass

                        # 删除可能的其他扩展名
                        base = os.path.splitext(tif_path)[0]
                        for ext in ['.aux.xml', '.ovr', '.msk', '.tfw', '.prj']:
                            aux_file = base + ext
                            if os.path.exists(aux_file):
                                try:
                                    os.remove(aux_file)
                                except:
                                    pass

                        # 删除成功，跳出重试循环
                        break

                    except (PermissionError, OSError) as e:
                        if attempt < max_retries - 1:
                            wait_time = retry_delay * (1.5 ** attempt)
                            print(f"⚠️ 删除 {os.path.basename(tif_path)} 失败"
                                  f"（尝试 {attempt + 1}/{max_retries}），"
                                  f"{wait_time:.1f}s 后重试...")
                            time.sleep(wait_time)
                            gc.collect()  # 强制垃圾回收
                        else:
                            print(f"❌ 无法删除文件 {os.path.basename(tif_path)}: {e}")
                            print(f"   提示：请关闭占用该文件的程序（QGIS、ArcGIS等）")
                            raise

            # 写入文件（带重试）
            for attempt in range(max_retries):
                try:
                    with rasterio.open(tif_path, "w", **meta) as dst:
                        dst.write(arr_2d, 1)
                        # 写入内部掩膜：255=有效，0=无效
                        dst.write_mask((valid_mask_2d.astype(np.uint8) * 255))

                    # print(f"✅ 已保存: {os.path.basename(tif_path)}")
                    break

                except Exception as e:
                    if attempt < max_retries - 1:
                        wait_time = retry_delay * (1.5 ** attempt)
                        print(f"⚠️ 写入 {os.path.basename(tif_path)} 失败"
                              f"（尝试 {attempt + 1}/{max_retries}）: {e}")
                        time.sleep(wait_time)
                        gc.collect()
                    else:
                        print(f"❌ 写入失败 {os.path.basename(tif_path)}: {e}")
                        raise

    except Timeout:
        print(f"❌ 获取文件锁超时: {os.path.basename(tif_path)}")
        raise

    finally:
        # 清理锁文件
        try:
            if os.path.exists(lock_path):
                os.remove(lock_path)
        except:
            pass

def get_data_RES(task_name, path_name, use_zip=False):
    """
    获取指定路径下的输出子目录。
    - path_name: 任务名称或路径名称
    - 返回: 输出子目录的完整路径
    """
    LUTO2_ROOT = f"../../../output/{task_name}/{path_name}"
    sys.path.insert(0, LUTO2_ROOT)
    if use_zip:
        output_path = f"../../../output/{task_name}/{path_name}"
        zip_path = os.path.join(output_path, 'Run_Archive.zip')
        if not os.path.isfile(zip_path):
            raise FileNotFoundError(f"未找到指定的 zip 文件: {zip_path}")
        with zipfile.ZipFile(zip_path, "r") as zf:
            matches = [name for name in zf.namelist() if glob.fnmatch.fnmatch(os.path.basename(name), "Data_RES*.lz4")]
            if not matches:
                raise FileNotFoundError("在 zip 文件中未找到匹配 'Data_RES*.lz4' 的文件。")
            # 读取 zip 内部 gz 文件内容为 bytes
            with zf.open(matches[0], "r") as file_in_zip:
                # joblib.load 可以直接处理这个文件对象
                data = joblib.load(file_in_zip)
    else:
        output_path = get_path(task_name, path_name)
        pattern = os.path.join(output_path, "Data_RES*.lz4")
        pkl_path = glob.glob(pattern)[0]
        data = joblib.load(pkl_path)

    return data
