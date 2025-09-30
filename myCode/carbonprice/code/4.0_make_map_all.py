
import os
import matplotlib.pyplot as plt
from rasterio.warp import reproject, Resampling
from matplotlib.lines import Line2D
import matplotlib.patches as patches
import matplotlib.image as mpimg
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
import matplotlib as mpl
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import rasterio
import geopandas as gpd
from matplotlib.colors import LogNorm
import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cartopy.crs as ccrs
import os
import json
import pylustrator
from cmcrameri import cm
import cmocean

# import matplotlib
# matplotlib.use("QtAgg")
# pylustrator.start()

import tools.config as config
from tools.helper_plot import set_plot_style


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


class PercentClipNormalize(mcolors.Normalize):
    """
    百分位裁剪拉伸归一化 - 类似ArcGIS的Percent Clip
    裁剪掉指定百分比的极端值，然后线性拉伸到0-1
    """

    def __init__(self, data, clip_percent=(2, 98), vmin=None, vmax=None, clip=False):
        """
        Parameters:
        -----------
        data : array-like
            用于计算百分位数的数据
        clip_percent : tuple
            (low_percent, high_percent) 要裁剪的百分位数，例如(2, 98)表示裁剪掉2%和98%的极值
        """
        # 计算百分位数
        valid_data = np.ma.compressed(np.ma.asarray(data))
        if len(valid_data) == 0:
            computed_vmin, computed_vmax = 0.0, 1.0
        else:
            computed_vmin, computed_vmax = np.percentile(valid_data, clip_percent)

        # 使用提供的vmin/vmax或计算的值
        final_vmin = vmin if vmin is not None else computed_vmin
        final_vmax = vmax if vmax is not None else computed_vmax

        super().__init__(vmin=final_vmin, vmax=final_vmax, clip=clip)
        self.clip_percent = clip_percent

    def __call__(self, value, clip=None):
        # 标准线性归一化，但会自动裁剪到vmin-vmax范围
        if clip is None:
            clip = self.clip

        if clip:
            value = np.clip(value, self.vmin, self.vmax)

        # 线性拉伸到0-1
        return (value - self.vmin) / (self.vmax - self.vmin)


class SigmoidNormalize(mcolors.Normalize):
    """
    Sigmoid拉伸归一化 - 类似ArcGIS的Sigmoid拉伸
    使用sigmoid函数进行非线性拉伸，增强中等值的对比度
    """

    def __init__(self, vmin=None, vmax=None, gain=6, cutoff=0.5, clip=False):
        """
        Parameters:
        -----------
        gain : float
            增益参数，控制S曲线的陡峭程度，默认6（类似ArcGIS）
        cutoff : float
            截止点，sigmoid曲线的中心点，默认0.5
        """
        super().__init__(vmin=vmin, vmax=vmax, clip=clip)
        self.gain = gain
        self.cutoff = cutoff

    def __call__(self, value, clip=None):
        if clip is None:
            clip = self.clip

        if clip:
            value = np.clip(value, self.vmin, self.vmax)

        # 先线性归一化到0-1
        x = (value - self.vmin) / (self.vmax - self.vmin)

        # 应用sigmoid变换
        # 标准sigmoid公式: 1 / (1 + exp(-gain * (x - cutoff)))
        sigmoid_x = 1.0 / (1.0 + np.exp(-self.gain * (x - self.cutoff)))

        # 重新缩放到0-1范围（因为sigmoid不会精确到达0和1）
        sigmoid_min = 1.0 / (1.0 + np.exp(-self.gain * (0 - self.cutoff)))
        sigmoid_max = 1.0 / (1.0 + np.exp(-self.gain * (1 - self.cutoff)))

        return (sigmoid_x - sigmoid_min) / (sigmoid_max - sigmoid_min)

    def inverse(self, u):
        """Sigmoid的反变换"""
        # 重新缩放
        sigmoid_min = 1.0 / (1.0 + np.exp(-self.gain * (0 - self.cutoff)))
        sigmoid_max = 1.0 / (1.0 + np.exp(-self.gain * (1 - self.cutoff)))

        sigmoid_u = u * (sigmoid_max - sigmoid_min) + sigmoid_min

        # 反sigmoid变换
        eps = 1e-10  # 避免log(0)
        sigmoid_u = np.clip(sigmoid_u, eps, 1 - eps)
        x = self.cutoff - np.log((1.0 / sigmoid_u) - 1.0) / self.gain

        return x * (self.vmax - self.vmin) + self.vmin




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


def make_percent_clip_norm(data, clip_percent=(2, 98)):
    """
    创建百分位裁剪归一化对象
    """
    return PercentClipNormalize(data, clip_percent=clip_percent)


def make_sigmoid_norm(data, gain=6, cutoff=0.5):
    """
    创建Sigmoid拉伸归一化对象

    Parameters:
    -----------
    data : array-like
        数据数组，用于计算vmin和vmax
    gain : float
        增益参数，控制S曲线陡峭程度
    cutoff : float
        截止点，sigmoid中心位置
    """
    valid_data = np.ma.compressed(np.ma.asarray(data))
    if len(valid_data) == 0:
        vmin, vmax = 0.0, 1.0
    else:
        vmin, vmax = np.min(valid_data), np.max(valid_data)

    return SigmoidNormalize(vmin=vmin, vmax=vmax, gain=gain, cutoff=cutoff)


def efficient_tif_plot(
        ax,
        tif_file,
        cmap='terrain',
        interpolation='nearest',
        title_name='',
        unit_name='',
        shp=None, line_color='black', line_width=1.5,
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

    # 归一化处理 - 新增percent_clip和sigmoid方法
    if normalization == 'log':
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

    elif normalization == 'percent_clip':
        norm = make_percent_clip_norm(data, clip_percent=(0, 95))
        data_to_plot = data

    elif normalization == 'sigmoid':
        norm = make_sigmoid_norm(data, gain=8, cutoff=0.4)
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

    ax.set_title(title_name, y=title_y, fontfamily='Arial') # , weight='bold'
    return im, cbar

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
                if i == 0:
                    # 首尾用整数格式
                    labels.append(f"{v:,.0f}")
                else:
                    # 常规先格式化
                    label = f"{v:,.{decimal_places}f}"
                    # 如果结果是 0，就继续增加小数位，直到显示出非零或达到限制
                    extra = decimal_places
                    while float(label.replace(",", "")) == 0 and extra < 10:  # 给个上限避免无限循环
                        extra += 1
                        label = f"{v:,.{extra}f}"
                    labels.append(label)
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
                   fontsize=12,
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
                         color=textcolor, fontfamily=fontfamily,fontsize=fontsize,
                         transform=trans, zorder=1002)

        elif style == "line":
            overlay.add_line(Line2D([x - width / 2, x + width / 2], [y, y],
                                    color=linecolor, linewidth=linewidth,
                                    transform=trans, zorder=1001))
            overlay.text(x + width / 2 + gap, y, text, ha='left', va='center',
                         color=textcolor, fontfamily=fontfamily,fontsize=fontsize,
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
                         color=textcolor, fontfamily=fontfamily,fontsize=fontsize,
                         transform=trans, zorder=1002)

        elif style == "line":
            overlay.add_line(Line2D([x, x + width], [y, y],
                                    color=linecolor, linewidth=linewidth,
                                    transform=trans, zorder=1001))
            overlay.text(x + width + gap, y, text, ha='left', va='center',
                         color=textcolor, fontfamily=fontfamily,fontsize=fontsize,
                         transform=trans, zorder=1002)


def add_scalebar(fig, ax, x, y, length_km=500,
                 fontfamily='Arial',fontsize=12,
                 color='black',linewidth=2,
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
             ha='center', va='bottom',fontsize=fontsize,
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



def plot_tif_layer(
        tif_file: str,
        title: str,
        unit: str,
        cmap,
        outfile: str = None,
        ax=None,
        normalization=None,
        decimal_places=2,
        custom_tick_values=False,
        line_width=0.3,
        title_y=0.95,
):
    """通用绘制函数：既可以保存单图，也可以在指定ax上绘制"""

    if ax is None:
        # 原来的单图模式
        fig = plt.figure(figsize=(6, 6), dpi=300, constrained_layout=False)
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
        should_save = True
    else:
        # 子图模式，ax已经提供
        should_save = False

    # 绘制逻辑（不变）
    efficient_tif_plot(
        ax,
        os.path.join(arr_path, tif_file),
        cmap=cmap,
        shp="../Map/AUS_line1.shp",
        line_width=line_width,
        title_name=title,
        unit_name=unit,
        legend_bbox_to_anchor=legend_bbox,
        legend_nbins=legend_nbins,
        title_y=title_y,
        char_ticks_length=1,
        unit_labelpad=1,
        normalization='linear', # normalization,
        decimal_places=decimal_places,
        custom_tick_values=custom_tick_values,
    )
    add_binary_gray_layer(ax, aligned_tif, gray_hex="#808080", alpha=1, zorder=15)

    # 只有单图模式才保存和显示
    if should_save and outfile:
        plt.savefig(outfile, dpi=300, pad_inches=0.1, transparent=True)
        plt.show()
        plt.close(fig)


def safe_plot(*, tif_file, title, unit, cmap, outfile=None, ax=None, **kwargs):
    """既支持单图保存，也支持子图绘制"""
    if ax is None:
        print(f"[INFO] Plotting {tif_file} -> {outfile}")
        os.makedirs(os.path.dirname(outfile), exist_ok=True)
    else:
        print(f"[INFO] Plotting {tif_file} to subplot: {title}")

    full = os.path.join(arr_path, tif_file)
    if not os.path.exists(full):
        print(f"[SKIP] Not found: {full}")
        if ax is not None:
            # 在子图上显示错误信息
            ax.text(0.5, 0.5, f"File not found:\n{os.path.basename(tif_file)}",
                    ha='center', va='center', transform=ax.transAxes,
                    fontsize=8, color='red')
            ax.set_title(title, fontsize=10)
        return

    plot_tif_layer(
        tif_file=tif_file,
        title=title,
        unit=unit,
        cmap=cmap,
        outfile=outfile,
        ax=ax,
        **kwargs
    )


def plot_all_for_scenario(env: str, cfg: dict, year: int = 2050, combined_mode=False,
                          combined_fig=None, plot_idx_ref=None):
    """
    既支持原来的单个保存模式，也支持合并到一张大图模式

    Parameters:
    -----------
    combined_mode : bool
        是否为合并模式
    combined_fig : matplotlib.figure.Figure
        合并模式下的图形对象
    plot_idx_ref : list
        合并模式下的子图索引引用（用列表包装以便修改）
    """
    print(f"\n===== SCENARIO: {env} (year={year}) =====")

    if combined_mode:
        nrows, ncols = 6, 4

    # 1) Total cost
    tif = f"{env}/xr_total_cost_{env}_{year}.tif"
    gs = gridspec.GridSpec(6, 4, figure=combined_fig, hspace=-0.45, wspace=0.03,
                           left=0.03, right=0.99, top=0.99, bottom=0.03)
    if combined_mode:
        if plot_idx_ref[0] < nrows * ncols:
            # 计算行列索引
            row, col = divmod(plot_idx_ref[0], 4)

            # 使用 GridSpec 的切片来定位
            ax = combined_fig.add_subplot(gs[row, col], projection=ccrs.PlateCarree())

            safe_plot(
                tif_file=tif,
                title="Total cost",
                unit=r"MAU\$ yr$^{-1}$",
                cmap=cost_cmap,
                normalization="log",
                decimal_places=2,
                ax=ax
            )
            plot_idx_ref[0] += 1
    else:
        out = os.path.join(out_dir, env, f"map Total cost {env}.png")
        safe_plot(
            tif_file=tif,
            title="Total cost",
            unit=r"MAU\$ yr$^{-1}$",
            cmap=cost_cmap,
            normalization="log",
            decimal_places=2,
            outfile=out
        )

    # 2) Cost components
    for key, title in zip(env_keys, title_keys):
        tif = f"{env}/xr_{key}_{env}_{year}.tif"
        kwargs = dict(normalization="log") | layer_overrides.get(key, {})

        if combined_mode:
            if plot_idx_ref[0] < nrows * ncols:
                ax = combined_fig.add_subplot(6, 4, plot_idx_ref[0] + 1, projection=ccrs.PlateCarree())
                safe_plot(
                    tif_file=tif,
                    title=f"{title}",
                    unit=r"MAU\$ yr$^{-1}$",
                    cmap=cost_cmap,
                    ax=ax,
                    **kwargs
                )
                plot_idx_ref[0] += 1
        else:
            out = os.path.join(out_dir, env, f"map {title} {env}.png")
            safe_plot(
                tif_file=tif,
                title=title,
                unit=r"MAU\$ yr$^{-1}$",
                cmap=cost_cmap,
                outfile=out,
                **kwargs
            )

    # 3) Benefits
    if "ghg" in cfg.get("benefits", []):
        tif = f"{env}/xr_total_carbon_{env}_{year}.tif"
        if combined_mode:
            if plot_idx_ref[0] < nrows * ncols:
                ax = combined_fig.add_subplot(6, 4, plot_idx_ref[0] + 1, projection=ccrs.PlateCarree())
                safe_plot(
                    tif_file=tif,
                    title=f"Change in GHG benefit",
                    unit=r"tCO$_2$e yr$^{-1}$",
                    cmap=benefit_cmap,
                    normalization='percent_clip',
                    decimal_places=2,
                    ax=ax
                )
                plot_idx_ref[0] += 1
        else:
            out = os.path.join(out_dir, env, f"map GHG benefit {env}.png")
            safe_plot(
                tif_file=tif,
                title="Change in GHG benefit",
                unit=r"tCO$_2$e yr$^{-1}$",
                cmap=benefit_cmap,
                normalization='percent_clip',
                decimal_places=2,
                outfile=out
            )

    if "bio" in cfg.get("benefits", []):
        tif = f"{env}/xr_total_bio_{env}_{year}.tif"
        if combined_mode:
            if plot_idx_ref[0] < nrows * ncols:
                ax = combined_fig.add_subplot(6, 4, plot_idx_ref[0] + 1, projection=ccrs.PlateCarree())
                safe_plot(
                    tif_file=tif,
                    title=f"Change in biodiversity benefit",
                    unit=r"ha yr$^{-1}$",
                    cmap=benefit_cmap,
                    decimal_places=2,
                    ax=ax
                )
                plot_idx_ref[0] += 1
        else:
            out = os.path.join(out_dir, env, f"map Biodiversity benefit {env}.png")
            safe_plot(
                tif_file=tif,
                title="Change in biodiversity benefit",
                unit=r"ha yr$^{-1}$",
                cmap=benefit_cmap,
                decimal_places=2,
                outfile=out
            )

    # 4) Prices
    if "carbon" in cfg.get("prices", []):
        if 'Counterfactual_carbon_high_bio_50' in env:
            title_name = f"Carbon price for GHG and biodiversity"
        elif 'carbon_high_bio_50' in env:
            title_name = f"Carbon price for biodiversity"
        else:
            title_name = f"Carbon price"
        tif = f"{env}/xr_carbon_price_{env}_{year}.tif"
        if combined_mode:
            if plot_idx_ref[0] < nrows * ncols:
                ax = combined_fig.add_subplot(6, 4, plot_idx_ref[0] + 1, projection=ccrs.PlateCarree())
                safe_plot(
                    tif_file=tif,
                    title=title_name,
                    unit=r"AU\$ CO$_2$e$^{-1}$ yr$^{-1}$",
                    cmap=price_cmap,
                    decimal_places=2,
                    ax=ax
                )
                plot_idx_ref[0] += 1
        else:
            out = os.path.join(out_dir, env, f"map Carbon price {env}.png")
            safe_plot(
                tif_file=tif,
                title=title_name,
                unit=r"AU\$ CO$_2$e$^{-1}$ yr$^{-1}$",
                cmap=price_cmap,
                decimal_places=2,
                outfile=out
            )

    if "bio" in cfg.get("prices", []):
        tif = f"{env}/xr_bio_price_{env}_{year}.tif"
        if combined_mode:
            if plot_idx_ref[0] < nrows * ncols:
                ax = combined_fig.add_subplot(6, 4, plot_idx_ref[0] + 1, projection=ccrs.PlateCarree())
                safe_plot(
                    tif_file=tif,
                    title=f"Biodiversity cost",
                    unit=r"AU\$ ha$^{-1}$ yr$^{-1}$",
                    cmap=price_cmap,
                    decimal_places=2,
                    ax=ax
                )
                plot_idx_ref[0] += 1
        else:
            out = os.path.join(out_dir, env, f"map Biodiversity price {env}.png")
            safe_plot(
                tif_file=tif,
                title="Biodiversity price",
                unit=r"AU\$ ha$^{-1}$ yr$^{-1}$",
                cmap=price_cmap,
                decimal_places=2,
                outfile=out
            )
    if combined_mode:
        return combined_fig,ax


def plot_all_scenarios_combined(scenarios: dict, year: int = 2050, figsize=(20, 30), combined_mode=True):
    """创建合并的所有情景图表"""
    print(f"\n===== CREATING COMBINED PLOT FOR ALL SCENARIOS (year={year}) =====")

    fig = plt.figure(figsize=figsize)
    plot_idx_ref = [0]  # 用列表包装以便在函数间修改

    for env, cfg in scenarios.items():
        fig,ax = plot_all_for_scenario(env, cfg, year=year, combined_mode=combined_mode,
                              combined_fig=fig, plot_idx_ref=plot_idx_ref)
    return fig,ax



# ==== Paths & global params ====
base_dir = f"../../../output/{config.TASK_NAME}/carbon_price"
arr_path = f"{base_dir}/4_tif"
out_dir  = f"{base_dir}/5_map"
os.makedirs(out_dir, exist_ok=True)

legend_nbins = 3
legend_bbox = (0.1, 0.10, 0.8, 0.9)

# 统一样式
set_plot_style(font_size=15, font_family='Arial')

# 参照/掩膜对齐（仅一次）
ref_tif   = f"{arr_path}/carbon_high/xr_total_cost_carbon_high_2050.tif"   # 用 carbon_high 的 total_cost 作为参考
src_tif   = f"{arr_path}/public_area.tif"
aligned_tif = f"{arr_path}/public_area_aligned.tif"
align_raster_to_reference(src_tif, ref_tif, aligned_tif, resampling="nearest")

# 统一色带
cost_cmap  = LinearSegmentedColormap.from_list("cost",  ["#FFFEC2", "#FA4F00", "#A80000","#5c2324"])
# cost_cmap  = cmocean.cm.matter
benefit_cmap = LinearSegmentedColormap.from_list("benefit", ["#ffff80", "#38e009","#1a93ab","#0c1078"])
# benefit_cmap = cmocean.cm.haline_r
price_cmap = LinearSegmentedColormap.from_list("price", ["#00ffff", "#ff00ff"])
# price_cmap = cm.buda_r
# ==== 你的既有配置（保持不变）====
scenarios = {
    "carbon_high": {
        "benefits": ["ghg"],          # 只画GHG
        "prices":   ["carbon"],       # 只画碳价
    },
    "carbon_high_bio_50": {
        "benefits": ["bio"],          # 只画Bio
        "prices":   ["carbon"]           # 只画生物多样性价
    },
    "Counterfactual_carbon_high_bio_50": {
        "benefits": ["ghg"],          # 只画GHG（一般没有Bio）
        "prices":   ["carbon"],       # 只画碳价
    }
}

env_keys = [
    "cost_ag",
    "cost_agricultural_management",
    "cost_non_ag",
    "cost_transition_ag2ag_diff",
    "transition_cost_ag2non_ag_amortised_diff",
]
title_keys = [
    "Agriculture cost",
    "Agricultural management cost",
    "Non-agriculture cost",
    "Transition(ag→ag) cost",
    "Transition(ag→non-ag) cost",
]

# 可选覆盖：别忘了它（生效于成本组件）
layer_overrides = {
    # "cost_ag": dict(decimal_places=3),
    # "cost_agricultural_management": dict(custom_tick_values=[0, 0.01, 50], decimal_places=2),
    # ...
}


# # ==== 逐个情景执行：一个情景所有图画完，再画下一个 ====
# for env, cfg in scenarios.items():
#     plot_all_for_scenario(env, cfg, year=2050)
# 合并模式
fig,ax = plot_all_scenarios_combined(scenarios, year=2050, figsize=(20, 30))

font_size = ax.xaxis.get_label().get_size()
font_family = ax.xaxis.get_label().get_family()[0]

plt.rcParams['font.family'] = font_family
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = font_family   # 直立
plt.rcParams['mathtext.it'] = font_family   # 斜体
plt.rcParams['mathtext.bf'] = font_family   # 粗体
plt.rcParams['mathtext.sf'] = font_family   # 无衬线

add_north_arrow(fig, 0.20, 0.063,size=0.012)
add_scalebar(fig,ax, 0.23, 0.069, length_km=500, fontsize=font_size,fontfamily=font_family,linewidth=1)
add_annotation(fig, 0.285, 0.073, width=0.015, text="Australian state boundary",linewidth=1,
               style="line", linecolor="black",fontsize=font_size, fontfamily=font_family)
add_annotation(fig, 0.435, 0.07, width=0.0122, height=0.0072,linewidth=1, text="No data",
               style="box", facecolor="white", edgecolor="black",fontsize=font_size, fontfamily=font_family)
add_annotation(fig, 0.52, 0.07, width=0.0122, height=0.0072,linewidth=1, text="Public, indigenous, urban, and other intensive land uses",
               style="box", facecolor="#808080",edgecolor="#808080",fontsize=font_size, fontfamily=font_family)
fig.text(
    0.015, 0.76, r'Reference→$\mathrm{GHG}_{\mathrm{high}}$',
    rotation=90, va="center", ha="left", fontsize=font_size,fontfamily=font_family,
    rotation_mode="anchor"  # 以锚点为中心旋转，贴边更稳
)
fig.text(
    0.015, 0.47, r'$\mathrm{GHG}_{\mathrm{high}}$→$\mathrm{GHG}_{\mathrm{high}}$,$\mathrm{Bio}_{\mathrm{50}}$',
    rotation=90, va="center", ha="left", fontsize=font_size,fontfamily=font_family,
    rotation_mode="anchor"  # 以锚点为中心旋转，贴边更稳
)
fig.text(
    0.015, 0.2, r'Reference→$\mathrm{GHG}_{\mathrm{high}}$,$\mathrm{Bio}_{\mathrm{50}}$',
    rotation=90, va="center", ha="left", fontsize=font_size,fontfamily=font_family,
    rotation_mode="anchor"  # 以锚点为中心旋转，贴边更稳
)

# plt.subplots_adjust(bottom=0.05)
output_path = os.path.join(out_dir, f"06_all_maps_line")
# save_figure_properly(fig, output_path, facecolor='white')
fig.savefig(f"{output_path}.png", dpi=300)
plt.show()