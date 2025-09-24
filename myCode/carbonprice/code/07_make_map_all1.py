from rasterio.warp import reproject, Resampling
import cartopy.crs as ccrs
from matplotlib.lines import Line2D
import matplotlib.patches as patches
import matplotlib.image as mpimg
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib as mpl
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import rasterio
import geopandas as gpd
from matplotlib.colors import LogNorm
import numpy as np

import tools.config as config


class HistEqNorm(mcolors.Normalize):
    """
    直方图均衡化的 Normalize：把原始数值按经验 CDF 映射到 [0,1]。
    可直接用于 imshow(..., norm=HistEqNorm(...)).
    """

    def __init__(self, bin_edges, cdf, vmin=None, vmax=None, clip=False):
        super().__init__(vmin=vmin, vmax=vmax, clip=clip)
        self._bin_edges = np.asarray(bin_edges, dtype=float)
        self._cdf = np.asarray(cdf, dtype=float)

    def __call__(self, value, clip=None):
        # 允许掩膜数组 & NaN
        val = np.ma.asarray(value)
        out = np.ma.empty(val.shape, dtype=float)

        # 非有限值保持掩膜/NaN
        mask = ~np.isfinite(val)
        valid = ~mask

        if np.any(valid):
            x = val[valid].astype(float)
            # 截断到 [vmin, vmax]
            x = np.clip(x, self.vmin, self.vmax)
            # 用 CDF 做单调映射（上沿边界对应的 CDF）
            out_valid = np.interp(x, self._bin_edges[1:], self._cdf, left=0.0, right=1.0)
            out[valid] = out_valid

        # 无效值设为掩膜
        out = np.ma.array(out, mask=mask)
        return out


def make_hist_eq_norm(data, mask=None, bins=256, clip_percent=None):
    """
    基于 data 的经验直方图，构造直方图均衡化的 Normalize。
    - data: 2D/ndarray (可为掩膜数组)
    - mask: 可选，True 表示有效像元（会与 data 的掩膜/NaN 共同作用）
    - bins: 直方图箱数（越大越精细）
    - clip_percent: (low, high) 用分位裁剪极端值（例如 (2,98)）
    返回：norm 对象，可直接用于 imshow(..., norm=norm)
    """
    arr = np.ma.asarray(data)

    # 组合有效掩膜：非 NaN/Inf，且可选 mask
    finite = np.isfinite(arr)
    if mask is not None:
        finite = finite & (mask.astype(bool))

    vals = np.asarray(arr[finite], dtype=float)

    if vals.size == 0:
        # 没有有效数据，退回线性 0..1
        return mcolors.Normalize(vmin=0.0, vmax=1.0)

    # 分位裁剪（鲁棒，推荐）
    if clip_percent is not None:
        lo, hi = clip_percent
        vmin, vmax = np.percentile(vals, [lo, hi])
        vals = vals[(vals >= vmin) & (vals <= vmax)]
    else:
        vmin, vmax = np.min(vals), np.max(vals)

    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        return mcolors.Normalize(vmin=vmin if np.isfinite(vmin) else 0.0,
                                 vmax=vmax if np.isfinite(vmax) else 1.0)

    # 定义直方图的固定边界（和 ArcGIS 的 equalize 类似，按值域分箱）
    bin_edges = np.linspace(vmin, vmax, bins + 1)
    hist, edges = np.histogram(vals, bins=bin_edges)

    # 经验 CDF（上沿边界对应的累计频率），映射到 0..1
    cdf = hist.cumsum().astype(float)
    if cdf[-1] == 0:
        # 所有频次为 0（极端情况），退回线性
        return mcolors.Normalize(vmin=vmin, vmax=vmax)
    cdf /= cdf[-1]

    # 构造 Normalize
    return HistEqNorm(bin_edges=edges, cdf=cdf, vmin=vmin, vmax=vmax)


def set_plot_style(font_size=12, font_family='Arial'):
    mpl.rcParams.update({
        'font.size': font_size, 'font.family': font_family,
        'axes.titlesize': font_size, 'axes.labelsize': font_size,
        'xtick.labelsize': font_size, 'ytick.labelsize': font_size,
        'legend.fontsize': font_size, 'figure.titlesize': font_size
    })


def nice_round(values):
    """对数组 values 里的数进行数量级四舍五入，
    - 1 位数：个位
    - 2/3 位数：十位
    - 4 位数：百位
    - 5 位数：千位
    - 6 位及以上：只保留前两位
    """
    rounded = []
    for v in values:
        if v <= 1 or np.isnan(v):
            rounded.append(v)
            continue

        magnitude = int(np.floor(np.log10(abs(v))))  # 数量级
        if magnitude == 0:
            # 个位数
            r = round(v)
        elif magnitude in (1, 2):
            # 两位数或三位数 -> 十位
            r = round(v, -1)
        elif magnitude == 3:
            # 四位数 -> 百位
            r = round(v, -2)
        elif magnitude == 4:
            # 五位数 -> 千位
            r = round(v, -3)
        else:
            # 六位及以上 -> 保留前两位
            digits_to_keep = magnitude - 1  # 保留前两位
            r = round(v, -digits_to_keep)
        rounded.append(r)
    return np.array(rounded)


class StretchHighLowNormalize(mcolors.Normalize):
    """高低值拉伸、中间压缩的归一化"""

    def __init__(self, vmin=None, vmax=None, stretch=5, clip=False):
        super().__init__(vmin, vmax, clip)
        self.stretch = stretch

    def __call__(self, value, clip=None):
        # 线性归一化到 0~1
        x = (value - self.vmin) / (self.vmax - self.vmin)
        # S型拉伸：高低值变化快，中间变化慢
        x_stretched = 0.5 * (np.tanh(self.stretch * (x - 0.5)) + 1)
        return x_stretched

    def inverse(self, u):
        # 反归一化映射
        eps = 1e-6
        u = np.clip(u, eps, 1 - eps)
        x = 0.5 * (np.arctanh(2 * u - 1) / self.stretch + 1)
        return x * (self.vmax - self.vmin) + self.vmin


def efficient_tif_plot(
        ax,
        tif_file,
        cmap='terrain',
        interpolation='nearest',
        title_name='',
        unit_name='',
        shp=None, line_color='black', line_width=1,
        legend_width="55%", legend_height="6%",
        legend_loc='lower left', legend_bbox_to_anchor=(0, 0, 1, 1),
        legend_borderpad=1, legend_nbins=5,
        char_ticks_length=3, char_ticks_pad=1,
        title_y=1, unit_labelpad=3,
        normalization='hist_eq', stretch_factor=5,
        decimal_places=None, clip_percent=None,
        custom_tick_values=False,
):
    # --- 读取为 MaskedArray，并按 nodata 掩膜 ---
    with rasterio.open(tif_file) as src:
        bounds = src.bounds
        raw_data = src.read(1)  # 先读取原始数据，不使用masked=True
        nodata = src.nodata
        raster_crs = src.crs
        extent = (bounds.left, bounds.right, bounds.bottom, bounds.top)


    # ✅ 关键修改1：手动创建更严格的掩膜
    mask = np.zeros(raw_data.shape, dtype=bool)

    # 掩膜 nodata 值
    if nodata is not None:
        mask |= (raw_data == nodata)

    # 掩膜 NaN 和无穷值
    mask |= ~np.isfinite(raw_data)

    # ✅ 关键修改2：掩膜极小值（可能是填充的0值）
    mask |= (raw_data <= 0)  # 根据您的数据特点，成本应该>0

    # 调试信息
    print(f"Raw data range: {raw_data.min()} to {raw_data.max()}")
    print(f"Raw data unique values (first 10): {np.unique(raw_data)[:10]}")
    print(f"Mask percentage: {mask.sum() / mask.size * 100:.1f}%")

    # 创建MaskedArray
    data = np.ma.masked_array(raw_data, mask=mask)

    # ✅ 关键修改2：检查数据有效性
    if np.ma.is_masked(data) and data.count() == 0:
        print("Warning: All data is masked/invalid!")
        return None, None

    # ✅ 关键修改3：确保只有有限的有效值参与计算
    valid_data = data.compressed()  # 获取未被掩膜的数据
    if len(valid_data) == 0:
        print("Warning: No valid data found!")
        return None, None

    print(f"Valid data range: {valid_data.min():.6f} to {valid_data.max():.6f}")
    print(f"Valid pixel count: {len(valid_data)} / {data.size}")

    # CRS
    data_crs = _cartopy_crs_from_raster_crs(raster_crs)

    # 矢量
    if shp is not None:
        gdf = gpd.read_file(shp) if isinstance(shp, str) else shp
        gdf = gdf.to_crs(raster_crs)
        gdf.plot(ax=ax, edgecolor=line_color, linewidth=line_width, facecolor='none')
        minx, miny, maxx, maxy = gdf.total_bounds
        pad_x = (maxx - minx) * 0.02 or 1e-4
        pad_y = (maxy - miny) * 0.02 or 1e-4
        ax.set_extent((minx - pad_x, maxx + pad_x, miny - pad_y, maxy + pad_y), crs=data_crs)

    ax.set_title(title_name, y=0.95)
    ax.set_axis_off()

    # --- 归一化：使用有效数据计算范围 ---
    if normalization == 'log':
        # log 需要正数：把 <=0 的值也掩掉
        data_safe = np.ma.masked_less_equal(data, 0)
        if data_safe.count() == 0:
            print("Warning: No positive values for log normalization!")
            return None, None
        vmin = float(np.ma.min(data_safe))
        vmax = float(np.ma.max(data_safe))
        norm = LogNorm(vmin=vmin, vmax=vmax)
        data_to_plot = data_safe
    elif normalization == 'linear':
        vmin = float(np.ma.min(data))
        vmax = float(np.ma.max(data))
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        data_to_plot = data
    elif normalization == 'stretch':
        vmin = float(np.ma.min(data))
        vmax = float(np.ma.max(data))
        norm = StretchHighLowNormalize(vmin=vmin, vmax=vmax, stretch=stretch_factor)
        data_to_plot = data
    else:  # 'hist_eq'
        valid_mask = ~np.ma.getmaskarray(data)
        norm = make_hist_eq_norm(data, mask=valid_mask, bins=256, clip_percent=clip_percent)
        data_to_plot = data

    # --- ✅ 关键修改4：正确设置colormap的透明处理 ---
    if isinstance(cmap, str):
        cmap_obj = mpl.colormaps.get_cmap(cmap).copy()
    else:
        cmap_obj = cmap.copy()

    # 设置无效值（掩膜/NaN）为完全透明
    cmap_obj.set_bad(color=(0, 0, 0, 0))  # RGBA: 完全透明

    # --- ✅ 关键修改5：强制设置透明并使用alpha通道 ---
    # 创建一个RGBA数组，直接控制透明度
    from matplotlib.colors import Normalize

    # 先将数据标准化
    if normalization == 'log':
        norm_for_rgba = LogNorm(vmin=vmin, vmax=vmax)
    elif normalization == 'linear':
        norm_for_rgba = Normalize(vmin=vmin, vmax=vmax)
    elif normalization == 'stretch':
        norm_for_rgba = norm
    else:  # hist_eq
        norm_for_rgba = norm

    # 将数据转换为0-1范围
    normalized_data = norm_for_rgba(data_to_plot.filled(np.nan))

    # 获取colormap
    if isinstance(cmap, str):
        cmap_func = mpl.colormaps.get_cmap(cmap)
    else:
        cmap_func = cmap

    # 转换为RGBA
    rgba_data = cmap_func(normalized_data)

    # 设置掩膜区域为完全透明
    mask_2d = np.ma.getmaskarray(data_to_plot)
    rgba_data[mask_2d] = [0, 0, 0, 0]  # 完全透明

    # 绘制RGBA数组
    im = ax.imshow(
        rgba_data,
        origin='upper',
        extent=extent,
        transform=data_crs,
        interpolation=interpolation
    )

    # --- 色条：沿用"均匀色带"的线性 mappable ---
    linear_sm = mpl.cm.ScalarMappable(norm=mcolors.Normalize(0, 1), cmap=cmap_obj)
    cax = inset_axes(
        ax, width=legend_width, height=legend_height, loc=legend_loc,
        borderpad=legend_borderpad, bbox_to_anchor=legend_bbox_to_anchor,
        bbox_transform=ax.transAxes,
    )
    cbar = mpl.pyplot.colorbar(
        linear_sm, cax=cax, orientation='horizontal', extend='both',
        extendfrac=0.1, extendrect=False
    )

    # 刻度
    if custom_tick_values is not False:
        tick_vals_nice = custom_tick_values
        ticks_01_nice = _forward_map_values(tick_vals_nice, norm, normalization)
    else:
        ticks_01 = np.linspace(0, 1, legend_nbins if legend_nbins >= 2 else 2)
        tick_vals = _inverse_map_values(ticks_01, norm, normalization)
        tick_vals_nice = nice_round(tick_vals)
        ticks_01_nice = _forward_map_values(tick_vals_nice, norm, normalization)

    tick_labels = _format_tick_labels(tick_vals_nice, decimal_places)
    cbar.set_ticks(ticks_01_nice)
    cbar.set_ticklabels(tick_labels)
    cbar.outline.set_visible(False)
    cbar.ax.xaxis.set_label_position('top')
    cbar.ax.tick_params(length=char_ticks_length, pad=char_ticks_pad)
    if unit_name:
        cbar.set_label(unit_name, labelpad=unit_labelpad, family='Arial')

    ax.set_title(title_name, y=title_y, fontfamily='Arial', weight='bold')
    return im, cbar

# def efficient_tif_plot(
#         ax,
#         tif_file,
#         cmap='terrain',
#         interpolation='nearest',
#         title_name='',
#         unit_name='',
#         shp=None, line_color='black', line_width=1,
#         legend_width="55%", legend_height="6%",
#         legend_loc='lower left', legend_bbox_to_anchor=(0, 0, 1, 1),
#         legend_borderpad=1, legend_nbins=5,
#         char_ticks_length=3, char_ticks_pad=1,
#         title_y=1, unit_labelpad=3,
#         normalization='hist_eq', stretch_factor=5,
#         decimal_places=None, clip_percent=None,
#         custom_tick_values=False,
# ):
#     # --- 读取为 MaskedArray，并按 nodata 掩膜 ---
#     with rasterio.open(tif_file) as src:
#         bounds = src.bounds
#         data = src.read(1, masked=True)      # MaskedArray（GDAL内部mask会带上）
#         nodata = src.nodata
#         if nodata is not None:
#             data = np.ma.masked_equal(data, nodata)  # 再确保 -9999 等被掩掉
#         raster_crs = src.crs
#         extent = (bounds.left, bounds.right, bounds.bottom, bounds.top)
#
#     # CRS
#     data_crs = _cartopy_crs_from_raster_crs(raster_crs)
#
#     # 矢量
#     if shp is not None:
#         gdf = gpd.read_file(shp) if isinstance(shp, str) else shp
#         gdf = gdf.to_crs(raster_crs)
#         gdf.plot(ax=ax, edgecolor=line_color, linewidth=line_width, facecolor='none')
#         minx, miny, maxx, maxy = gdf.total_bounds
#         pad_x = (maxx - minx) * 0.02 or 1e-4
#         pad_y = (maxy - miny) * 0.02 or 1e-4
#         ax.set_extent((minx - pad_x, maxx + pad_x, miny - pad_y, maxy + pad_y), crs=data_crs)
#
#     ax.set_title(title_name, y=0.95)
#     ax.set_axis_off()
#
#     # --- 归一化：始终使用 np.ma.xxx，保持 MaskedArray ---
#     if normalization == 'log':
#         # log 需要正数：把 <=0 的值也掩掉
#         data_safe = np.ma.masked_less_equal(data, 0)
#         vmin = float(np.ma.min(data_safe))
#         vmax = float(np.ma.max(data_safe))
#         norm = LogNorm(vmin=vmin, vmax=vmax)
#         data_to_plot = data_safe
#     elif normalization == 'linear':
#         vmin = float(np.ma.min(data))
#         vmax = float(np.ma.max(data))
#         norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
#         data_to_plot = data
#     elif normalization == 'stretch':
#         vmin = float(np.ma.min(data))
#         vmax = float(np.ma.max(data))
#         norm = StretchHighLowNormalize(vmin=vmin, vmax=vmax, stretch=stretch_factor)
#         data_to_plot = data
#     else:  # 'hist_eq'
#         valid_mask = ~np.ma.getmaskarray(data)
#         norm = make_hist_eq_norm(data, mask=valid_mask, bins=256, clip_percent=clip_percent)
#         data_to_plot = data
#
#     # --- 让掩膜透明 ---
#     cmap_obj = mpl.colormaps.get_cmap(cmap).copy()
#     cmap_obj.set_bad((0, 0, 0, 0))   # 掩膜/NaN 透明
#
#     # --- 绘制 ---
#     # data_to_plot = data_to_plot.filled(np.nan)
#     im = ax.imshow(
#         data_to_plot, origin='upper', extent=extent, transform=data_crs,
#         cmap=cmap_obj, norm=norm, interpolation=interpolation
#     )
#
#     # --- 色条：沿用“均匀色带”的线性 mappable（不影响透明） ---
#     linear_sm = mpl.cm.ScalarMappable(norm=mcolors.Normalize(0, 1), cmap=cmap_obj)
#     cax = inset_axes(
#         ax, width=legend_width, height=legend_height, loc=legend_loc,
#         borderpad=legend_borderpad, bbox_to_anchor=legend_bbox_to_anchor,
#         bbox_transform=ax.transAxes,
#     )
#     cbar = mpl.pyplot.colorbar(
#         linear_sm, cax=cax, orientation='horizontal', extend='both',
#         extendfrac=0.1, extendrect=False
#     )
#
#     # 刻度
#     if custom_tick_values is not False:
#         tick_vals_nice = custom_tick_values
#         ticks_01_nice = _forward_map_values(tick_vals_nice, norm, normalization)
#     else:
#         ticks_01 = np.linspace(0, 1, legend_nbins if legend_nbins >= 2 else 2)
#         tick_vals = _inverse_map_values(ticks_01, norm, normalization)
#         tick_vals_nice = nice_round(tick_vals)
#         ticks_01_nice = _forward_map_values(tick_vals_nice, norm, normalization)
#
#     tick_labels = _format_tick_labels(tick_vals_nice, decimal_places)
#     cbar.set_ticks(ticks_01_nice)
#     cbar.set_ticklabels(tick_labels)
#     cbar.outline.set_visible(False)
#     cbar.ax.xaxis.set_label_position('top')
#     cbar.ax.tick_params(length=char_ticks_length, pad=char_ticks_pad)
#     if unit_name:
#         cbar.set_label(unit_name, labelpad=unit_labelpad, family='Arial')
#
#     ax.set_title(title_name, y=title_y, fontfamily='Arial', weight='bold')
#     return im, cbar


def _forward_map_values(values, norm, normalization):
    """将数据值映射到0-1区间"""
    values = np.array(values, dtype=float)

    if normalization == 'log':
        vmin, vmax = norm.vmin, norm.vmax
        values = np.clip(values, vmin, vmax)
        return (np.log(values) - np.log(vmin)) / (np.log(vmax) - np.log(vmin))
    elif normalization == 'linear':
        return (values - norm.vmin) / (norm.vmax - norm.vmin)
    elif normalization == 'stretch':
        return norm(values)
    else:  # hist_eq
        if hasattr(norm, "_cdf") and hasattr(norm, "_bin_edges"):
            cdf_x = np.concatenate(([norm.vmin], norm._bin_edges[1:], [norm.vmax]))
            cdf_y = np.concatenate(([0.0], norm._cdf, [1.0]))
            return np.interp(values, cdf_x, cdf_y)
        else:
            return (values - norm.vmin) / (norm.vmax - norm.vmin)


def _inverse_map_values(u_values, norm, normalization):
    """将0-1区间的值反映射为数据值"""
    u_values = np.array(u_values, dtype=float)

    if normalization == 'log':
        vmin, vmax = norm.vmin, norm.vmax
        return np.exp(u_values * (np.log(vmax) - np.log(vmin)) + np.log(vmin))
    elif normalization == 'linear':
        return norm.vmin + u_values * (norm.vmax - norm.vmin)
    elif normalization == 'stretch':
        return norm.inverse(u_values)
    else:  # hist_eq
        if hasattr(norm, "_cdf") and hasattr(norm, "_bin_edges"):
            cdf_x = np.concatenate(([0.0], norm._cdf, [1.0]))
            cdf_y = np.concatenate(([norm.vmin], norm._bin_edges[1:], [norm.vmax]))
            return np.interp(u_values, cdf_x, cdf_y)
        else:
            return norm.vmin + u_values * (norm.vmax - norm.vmin)


def _format_tick_labels(values, decimal_places=None):
    labels = []
    for i, v in enumerate(values):
        if v < 1:
            if decimal_places is not None:
                # 自动判断格式
                if i == 0:
                    # 首尾用整数格式
                    labels.append(f"{v:,.0f}")
                else:
                    # 其他用整数格式
                    labels.append(f"{v:,.{decimal_places}f}")
        else:
            labels.append(f"{v:,.0f}")

    return labels


def _cartopy_crs_from_raster_crs(r_crs):
    """从 rasterio CRS 推断 cartopy CRS"""
    if r_crs is None:
        return ccrs.PlateCarree()
    try:
        epsg = r_crs.to_epsg()
        if epsg:
            return ccrs.epsg(epsg)
    except Exception:
        pass
    if getattr(r_crs, 'is_geographic', False):
        return ccrs.PlateCarree()
    return ccrs.PlateCarree()


def align_raster_to_reference(src_path, ref_path, dst_path,
                              resampling="nearest", dtype=None, nodata=None, compress="deflate"):
    """
    将 src_path 的栅格重投影/重采样对齐到 ref_path 的网格，保存到 dst_path。
    - 分辨率、宽高、transform、CRS 与参考一致
    - 适合掩膜层(0/1)：resampling='nearest'
    - 适合连续值：resampling='bilinear' 或 'cubic'
    """
    # 选择重采样算法
    resampling_map = {
        "nearest": Resampling.nearest,
        "bilinear": Resampling.bilinear,
        "cubic": Resampling.cubic
    }
    rs = resampling_map.get(resampling, Resampling.nearest)

    # 打开参考栅格
    with rasterio.open(ref_path) as ref:
        ref_profile = ref.profile.copy()
        dst_crs = ref.crs
        dst_transform = ref.transform
        dst_width = ref.width
        dst_height = ref.height
        # 参考 nodata
        ref_nodata = ref.nodata

    # 打开源栅格
    with rasterio.open(src_path) as src:
        src_data = src.read(1)  # 单波段情况
        src_crs = src.crs
        src_transform = src.transform
        src_nodata = src.nodata

    # 目标 dtype / nodata 处理
    if dtype is None:
        dtype = src_data.dtype
    if nodata is None:
        # 对掩膜层：缺省 nodata = 0（或你可改成 255 视你的数据而定）
        nodata = src_nodata if src_nodata is not None else 0

    # 准备输出数组
    dst_data = np.full((dst_height, dst_width), nodata, dtype=dtype)

    # 重投影+重采样到参考网格
    reproject(
        source=src_data,
        destination=dst_data,
        src_transform=src_transform,
        src_crs=src_crs,
        src_nodata=src_nodata,
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        dst_nodata=nodata,
        resampling=rs,
    )

    # 写出
    profile_out = ref_profile.copy()
    profile_out.update(dtype=dtype, nodata=nodata, compress=compress, count=1)

    with rasterio.open(dst_path, "w", **profile_out) as dst:
        dst.write(dst_data, 1)

    return dst_path


def _crs_for_cartopy(rio_crs):
    """尽量把 rasterio 的 CRS 转成 cartopy 的 CRS。"""
    if rio_crs is None:
        return ccrs.PlateCarree(), False
    try:
        epsg = rio_crs.to_epsg()
        if epsg:
            return ccrs.epsg(epsg), True
    except Exception:
        pass
    # 尝试用字符串（WKT/PROJ4）创建——部分 cartopy 版本不支持所有字符串
    try:
        return ccrs.CRS.from_user_input(rio_crs.to_string()), True
    except Exception:
        return ccrs.PlateCarree(), False


def add_binary_gray_layer(ax, tif_file, gray_hex="#808080", alpha=1, zorder=10, debug=False):
    """
    将 tif 中 ==1 的像元画成灰色(可调透明度)，其它(0或NoData)透明。
    """
    with rasterio.open(tif_file) as src:
        band = src.read(1, masked=True)  # MaskedArray（nodata 已遮蔽为 mask=True）
        bounds = src.bounds
        rio_crs = src.crs

    # 1) 只把"未被掩膜 & 等于 1"的像元标为 1
    mask01 = np.zeros(band.shape, dtype=np.uint8)
    valid = ~band.mask
    mask01[valid & (band == 1)] = 1

    if debug:
        unique, counts = np.unique(mask01, return_counts=True)
        print("mask01 counts:", dict(zip(unique.tolist(), counts.tolist())))

    # 若全是 0，自然什么都看不到
    if mask01.sum() == 0 and debug:
        print("No pixels equal to 1 after masking; nothing to draw from this layer.")

    # 2) 颜色映射：0 -> 透明，1 -> 灰色(带透明度)
    rgba = list(mcolors.to_rgba(gray_hex))
    rgba[3] = alpha
    cmap = mcolors.ListedColormap([(0, 0, 0, 0), tuple(rgba)])

    # 3) CRS / 范围
    data_crs, ok = _crs_for_cartopy(rio_crs)
    if not ok and debug:
        print("Warning: raster CRS not recognized by cartopy; falling back to PlateCarree().")

    extent = (bounds.left, bounds.right, bounds.bottom, bounds.top)

    # 4) 画
    im = ax.imshow(
        mask01,
        origin="upper",
        extent=extent,
        transform=data_crs,  # 让 cartopy 负责坐标变换
        cmap=cmap,
        interpolation="nearest",
        vmin=0, vmax=1,  # 固定 0/1 映射
        zorder=zorder
    )
    return im


def _get_overlay_ax(fig):
    # 复用已存在的覆盖轴，避免重复创建
    for a in fig.axes:
        if getattr(a, "_overlay_for_annotations", False):
            return a
    ax = fig.add_axes([0, 0, 1, 1], frameon=False, zorder=1000)
    ax._overlay_for_annotations = True
    ax.set_axis_off()
    ax.set_facecolor('none')  # 透明，关键！
    ax.patch.set_alpha(0.0)
    return ax


def add_annotation(fig, x, y, width=None, height=None, text="",
                   style="box", anchor="ll",  # 'll' 左下角；'center' 中心
                   facecolor='white', edgecolor=None,
                   linecolor='black', linewidth=1.0,
                   textcolor='black', fontfamily='Arial',
                   gap=0.005):  # 正方形/线与文字的间距
    overlay = _get_overlay_ax(fig)
    trans = overlay.transAxes

    if anchor == "center":
        if style == "box":
            x0, y0 = x - width / 2, y - height / 2
            overlay.add_patch(patches.Rectangle(
                (x0, y0), width, height,
                facecolor=facecolor,
                edgecolor=('none' if edgecolor is None else edgecolor),
                linewidth=linewidth, transform=trans, zorder=1001))
            # 文字放在右侧
            overlay.text(x + width / 2 + width / 2 + gap, y, text, ha='left', va='center',
                         color=textcolor, fontfamily=fontfamily,
                         transform=trans, zorder=1002)

        elif style == "line":
            overlay.add_line(Line2D([x - width / 2, x + width / 2], [y, y],
                                    color=linecolor, linewidth=linewidth,
                                    transform=trans, zorder=1001))
            overlay.text(x + width / 2 + gap, y, text, ha='left', va='center',
                         color=textcolor, fontfamily=fontfamily,
                         transform=trans, zorder=1002)

    else:  # 左下角锚点
        if style == "box":
            overlay.add_patch(patches.Rectangle(
                (x, y), width, height,
                facecolor=facecolor,
                edgecolor=('none' if edgecolor is None else edgecolor),
                linewidth=linewidth, transform=trans, zorder=1001))
            # 文字放在右侧
            overlay.text(x + width + gap, y + height / 2, text, ha='left', va='center',
                         color=textcolor, fontfamily=fontfamily,
                         transform=trans, zorder=1002)

        elif style == "line":
            overlay.add_line(Line2D([x, x + width], [y, y],
                                    color=linecolor, linewidth=linewidth,
                                    transform=trans, zorder=1001))
            overlay.text(x + width + gap, y, text, ha='left', va='center',
                         color=textcolor, fontfamily=fontfamily,
                         transform=trans, zorder=1002)


def add_scalebar(fig, ax, x, y, length_km=500,
                 fontfamily='Arial',
                 color='black', linewidth=2,
                 transform_type='figure'):
    transform = ax.transAxes if transform_type == 'axes' else fig.transFigure

    # 估算比例尺宽度
    try:
        extent = ax.get_extent()
        lon_range = extent[1] - extent[0]
        km_per_degree = 100
        scale_length_deg = length_km / km_per_degree
        scale_width = scale_length_deg / lon_range * 0.8
        if transform_type == 'figure':
            scale_width *= 0.48
    except Exception:
        scale_width = 0.1 if transform_type == 'axes' else 0.05

    # 横线
    fig.add_artist(Line2D([x, x + scale_width], [y, y],
                          color=color, linewidth=linewidth,
                          transform=transform, clip_on=False))

    # 竖线（下端与横线齐平，向上）
    tick_h = 0.005 if transform_type == 'figure' else 0.02
    for tx in (x, x + scale_width):
        fig.add_artist(Line2D([tx, tx], [y, y + tick_h],
                              color=color, linewidth=linewidth,
                              transform=transform, clip_on=False))

    # 标签
    fig.text(x + scale_width / 2, y + tick_h, f'{length_km} km',
             ha='center', va='bottom',
             fontfamily=fontfamily, color=color, transform=transform)


def add_north_arrow(fig, x, y, size=0.1, img_path='../Map/north_arrow.png', transform_type='figure'):
    """
    在图上添加自定义指北针图像

    Parameters:
    -----------
    fig : matplotlib.figure.Figure
    x, y : float           # 左下角位置 (0-1)
    size : float           # 指北针相对于画布的宽度 (0-1)
    img_path : str         # 图片路径
    transform_type : str   # 'figure' 或 'axes'
    """
    img = mpimg.imread(img_path)

    if transform_type == 'figure':
        ax_img = fig.add_axes([x, y, size, size * img.shape[0] / img.shape[1]])  # 保持宽高比
    else:
        # 如果你想在某个ax的坐标系统内放，可以用 ax.inset_axes
        raise NotImplementedError("当前示例只做 figure 坐标系")

    ax_img.imshow(img)
    ax_img.axis('off')  # 不显示坐标轴


# ====================== 示例使用代码 ======================
if __name__ == "__main__":
    base_dir = f"../../../output/{config.TASK_NAME}/carbon_price"
    legend_nbins = 3


    env_category = 'carbon_high'
    tif_file = f"xr_total_cost_{env_category}_2050.tif"
    arr_path = f"{base_dir}/4_tif"
    ref_tif = f"{arr_path}/{tif_file}"
    src_tif = f'{arr_path}/public_area.tif'
    aligned_tif = f'{arr_path}/public_area_aligned.tif'

    set_plot_style(font_size=15, font_family='Arial')
    # 掩膜层用最近邻
    align_raster_to_reference(src_tif, ref_tif, aligned_tif, resampling="nearest")
    cost_cmap = LinearSegmentedColormap.from_list("cost", ["#FFFEC2", "#FA4F00", "#A80000"])

    title = 'Total cost'
    unit = 'MAU$'
    # 画图（两者像元网格完全一致了）
    fig = plt.figure(figsize=(6, 6), dpi=300, constrained_layout=False)
    ax1 = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

    from matplotlib.colors import LinearSegmentedColormap
    cost_cmap = LinearSegmentedColormap.from_list("cost", ["#FFFEC2", "#FA4F00", "#A80000"])

    efficient_tif_plot(
        ax1, f"{arr_path}/{tif_file}", cmap=cost_cmap,
        shp="../Map/AUS_line1.shp", line_width=0.3,
        title_name=title, unit_name=unit, legend_bbox_to_anchor=(0.1, 0.10, 0.8, 0.9),
        legend_nbins=3, title_y=0.95, char_ticks_length=1,
        unit_labelpad=1, normalization='log', decimal_places=2
    )
    add_binary_gray_layer(ax1, aligned_tif, gray_hex="#808080", alpha=1, zorder=15)
    plt.savefig(f"../../../output/{config.TASK_NAME}/carbon_price/3_Paper_figure/map {title} {env_category}.png", dpi=300,
                pad_inches=0.1, transparent=True)

    env_names = ['cost_ag', 'cost_agricultural_management', 'cost_non_ag', 'cost_transition_ag2ag_diff',
                 'transition_cost_ag2non_ag_amortised_diff']
    title_names = ['Agriculture cost', 'Agricultural management cost', 'Non-agriculture cost', 'Transition(ag→ag) cost',
                   'Transition(ag→non-ag) cost']

    for env, title in zip(env_names, title_names):
        unit = 'MAU$'
        # 画图（两者像元网格完全一致了）
        fig = plt.figure(figsize=(6, 6), dpi=300, constrained_layout=False)
        ax1 = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
        tif_file = f"xr_{env}_{env_category}_2050.tif"

        ticks = False
        decimal_places = 3
        # if env == 'xr_cost_ag':
        #     decimal_places = 2
        # if env == 'xr_cost_agricultural_management':
        #     ticks = [0,0.01,50]
        #     decimal_places = 2
        # elif env == 'xr_cost_non_ag':
        #     ticks = [0,0.001,0.3]
        # elif 'ag2ag' in env:
        #     decimal_places = 2
        #     ticks = [0, 0.01, 70]


        efficient_tif_plot(
            ax1, f"{arr_path}/{tif_file}", cmap=cost_cmap,
            shp="../Map/AUS_line1.shp", line_width=0.3,
            title_name=title, unit_name=unit, legend_bbox_to_anchor=(0.1, 0.10, 0.8, 0.9),
            legend_nbins=3, title_y=0.95, char_ticks_length=1, unit_labelpad=1,normalization='log',
            decimal_places=decimal_places, custom_tick_values=ticks
        )
        add_binary_gray_layer(ax1, aligned_tif, gray_hex="#808080", alpha=1, zorder=15)
        plt.savefig(f"../../../output/{config.TASK_NAME}/carbon_price/3_Paper_figure/map {title} {env_category}.png", dpi=300,
                    pad_inches=0.1, transparent=True)

        plt.show()

    env_category = 'carbon_high_bio_50'
    title = 'Total cost'
    unit = 'MAU$'
    # 画图（两者像元网格完全一致了）
    fig = plt.figure(figsize=(6,6), dpi=300,constrained_layout=False)
    ax1 = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    tif_file = f"xr_total_cost_{env_category}_2050.tif"
    from matplotlib.colors import LinearSegmentedColormap
    cost_cmap = LinearSegmentedColormap.from_list("cost", ["#FFFEC2", "#FA4F00", "#A80000"])

    efficient_tif_plot(
            ax1, f"{arr_path}/{tif_file}", cmap=cost_cmap,
            shp="../Map/AUS_line1.shp", line_width=0.3,
            title_name=title, unit_name=unit,legend_bbox_to_anchor=(0.1, 0.10, 0.8, 0.9),
            legend_nbins=3, title_y=0.95,
            char_ticks_length=1, unit_labelpad=1,  decimal_places=2
        )
    add_binary_gray_layer(ax1, aligned_tif, gray_hex="#808080", alpha=1, zorder=15)
    plt.savefig(f"../../../output/{config.TASK_NAME}/carbon_price/3_Paper_figure/map {title} {env_category}.png", dpi=300, pad_inches=0.1,transparent=True)
    plt.show()

    env_names = [ 'cost_ag','cost_agricultural_management','cost_non_ag','cost_transition_ag2ag_diff','transition_cost_ag2non_ag_amortised_diff']
    title_names = ['Agriculture cost','Agricultural management cost','Non-agriculture cost','Transition(ag→ag) cost','Transition(ag→non-ag) cost']
    for env, title in zip(env_names, title_names):
        unit = 'MAU$'
        # 画图（两者像元网格完全一致了）
        fig = plt.figure(figsize=(6, 6), dpi=300, constrained_layout=False)
        ax1 = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
        tif_file = f"xr_{env}_{env_category}_2050.tif"

        decimal_places = 2
        ticks = False
        # if env == 'xr_cost_ag':
        #     ticks = [0,0.1,0.3,90]
        #     decimal_places =1
        # elif 'agricultural_management' in env:
        #     print('xr_cost_agricultural_management')
        #     ticks = [0, 0.1, 0.2, 90]
        #     decimal_places =1
        # elif 'ag2ag' in env:
        #     decimal_places = 2
        #     ticks = [0, 0.15,16,60]
        # elif 'ag2non_ag' in env:
        #     decimal_places = 2
        #     ticks = [0, 0.7,1,30]

        efficient_tif_plot(
            ax1, f"{arr_path}/{tif_file}", cmap=cost_cmap,
            shp="../Map/AUS_line1.shp", line_width=0.3, title_name=title, unit_name=unit,
            legend_bbox_to_anchor=(0.1, 0.10, 0.8, 0.9), legend_nbins=3,
            title_y=0.95, char_ticks_length=1, unit_labelpad=1,decimal_places=decimal_places, custom_tick_values=ticks
        )
        add_binary_gray_layer(ax1, aligned_tif, gray_hex="#808080", alpha=1, zorder=15)
        plt.savefig(f"../../../output/{config.TASK_NAME}/carbon_price/3_Paper_figure/map {title} {env_category}.png", dpi=300,
                    pad_inches=0.1, transparent=True)

        plt.show()
#---------------------------------------------------ENV------------------------------------------------------------------------------------------------
    env_category = 'carbon_high'
    title = 'GHG benefit'
    unit = r"tCO$_2$e"
    # 画图（两者像元网格完全一致了）
    fig = plt.figure(figsize=(6,6), dpi=300,constrained_layout=False)
    ax1 = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    tif_file = f"xr_total_{env_category}_2050.tif"

    efficient_tif_plot(
            ax1, f"{arr_path}/{tif_file}", cmap="summer_r",
            shp="../Map/AUS_line1.shp", line_width=0.3,
            title_name=title, unit_name=unit,
            legend_bbox_to_anchor=(0.1, 0.10, 0.8, 0.9), legend_nbins=3, title_y=0.95,
            char_ticks_length=1, unit_labelpad=1,  decimal_places=2
        )
    add_binary_gray_layer(ax1, aligned_tif, gray_hex="#808080", alpha=1, zorder=15)
    plt.savefig(f"../../../output/{config.TASK_NAME}/carbon_price/3_Paper_figure/map {title} {env_category}.png", dpi=300, pad_inches=0.1,transparent=True)
    plt.show()

    env_category = 'carbon_high_bio_50'
    title = 'Biodiversity benefit'
    unit = "ha"
    # 画图（两者像元网格完全一致了）
    fig = plt.figure(figsize=(6, 6), dpi=300, constrained_layout=False)
    ax1 = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    tif_file = f"xr_total_{env_category}_2050.tif"

    efficient_tif_plot(
        ax1, f"{arr_path}/{tif_file}", cmap="summer_r",
        shp="../Map/AUS_line1.shp", line_width=0.3,
        title_name=title, unit_name=unit,
        legend_bbox_to_anchor=(0.1, 0.10, 0.8, 0.9), legend_nbins=3, title_y=0.95,
        char_ticks_length=1, unit_labelpad=1, decimal_places=2
    )
    add_binary_gray_layer(ax1, aligned_tif, gray_hex="#808080", alpha=1, zorder=15)
    plt.savefig(f"../../../output/{config.TASK_NAME}/carbon_price/3_Paper_figure/map {title} {env_category}.png",
                dpi=300, pad_inches=0.1, transparent=True)
    plt.show()
# ---------------------------------------------------Price----------------------------

    env_category = 'carbon_high'
    price_cmap = LinearSegmentedColormap.from_list("cost", ["#00ffff","#ff00ff"])

    title = 'Carbon price'
    unit = r"AU\$ CO$_2$e$^{-1}$"
    # 画图（两者像元网格完全一致了）
    fig = plt.figure(figsize=(6,6), dpi=300,constrained_layout=False)
    ax1 = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    tif_file = f"xr_price_{env_category}_2050.tif"

    efficient_tif_plot(
            ax1, f"{arr_path}/{tif_file}", cmap=price_cmap,
            shp="../Map/AUS_line1.shp", line_width=0.3,
            title_name=title, unit_name=unit,
            legend_bbox_to_anchor=(0.1, 0.10, 0.8, 0.9), legend_nbins=3, title_y=0.95,
            char_ticks_length=1, unit_labelpad=1,  decimal_places=2
        )
    add_binary_gray_layer(ax1, aligned_tif, gray_hex="#808080", alpha=1, zorder=15)
    plt.savefig(f"../../../output/{config.TASK_NAME}/carbon_price/3_Paper_figure/map {title} {env_category}.png", dpi=300, pad_inches=0.1,transparent=True)
    plt.show()

    env_category = 'carbon_high_bio_50'
    title = 'Biodiversity price'
    unit = r"AU\$ ha$^{-1}$"
    # 画图（两者像元网格完全一致了）
    fig = plt.figure(figsize=(6, 6), dpi=300, constrained_layout=False)
    ax1 = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    tif_file = f"xr_price_{env_category}_2050.tif"

    efficient_tif_plot(
        ax1, f"{arr_path}/{tif_file}", cmap=price_cmap,
        shp="../Map/AUS_line1.shp", line_width=0.3,
        title_name=title, unit_name=unit,
        legend_bbox_to_anchor=(0.1, 0.10, 0.8, 0.9), legend_nbins=3, title_y=0.95,
        char_ticks_length=1, unit_labelpad=1, decimal_places=2
    )
    add_binary_gray_layer(ax1, aligned_tif, gray_hex="#808080", alpha=1, zorder=15)
    plt.savefig(f"../../../output/{config.TASK_NAME}/carbon_price/3_Paper_figure/map {title} {env_category}.png",
                dpi=300, pad_inches=0.1, transparent=True)
    plt.show()

