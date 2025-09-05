from rasterio.warp import reproject, Resampling
import cartopy.crs as ccrs
from matplotlib.lines import Line2D
import matplotlib.patches as patches
import matplotlib.image as mpimg
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib as mpl
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import rasterio
import geopandas as gpd
from matplotlib.colors import LogNorm
from matplotlib.colors import LinearSegmentedColormap

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
            digits_to_keep = magnitude - 1   # 保留前两位
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
        x = 0.5 * (np.arctanh(2 * u - 1) / self.stretch + 1)
        return x * (self.vmax - self.vmin) + self.vmin



# 你可以单独放在一个文件里
class StretchHighLowNormalize(mpl.colors.Normalize):
    """高低值拉伸，中间压缩的归一化"""
    def __init__(self, vmin=None, vmax=None, stretch=5, clip=False):
        super().__init__(vmin, vmax, clip)
        self.stretch = stretch

    def __call__(self, value, clip=None):
        x = (value - self.vmin) / (self.vmax - self.vmin)
        x_stretched = 0.5 * (np.tanh(self.stretch * (x - 0.5)) + 1)
        return x_stretched

    def inverse(self, u):
        eps = 1e-6
        u = np.clip(u, eps, 1-eps)
        x = 0.5 * (np.arctanh(2 * u - 1) / self.stretch + 1)
        return x * (self.vmax - self.vmin) + self.vmin

import matplotlib.colors as mcolors
import numpy as np

class HistEqNorm(mcolors.Normalize):
    """直方图均衡归一化：自动根据数据分布拉伸聚集区"""
    def __init__(self, data, vmin=None, vmax=None, bins=256, clip=False):
        vmin = np.min(data) if vmin is None else vmin
        vmax = np.max(data) if vmax is None else vmax
        super().__init__(vmin, vmax, clip)
        # 只用非nan数据
        data_flat = np.ravel(data[~np.isnan(data)])
        hist, bin_edges = np.histogram(data_flat, bins=bins, range=(vmin, vmax), density=True)
        # 累积分布函数（CDF）
        cdf = np.cumsum(hist)
        cdf = cdf / cdf[-1]
        self._cdf = cdf
        self._bin_edges = bin_edges

    def __call__(self, value, clip=None):
        # 将 value 映射到色带 0~1
        value = np.clip(value, self.vmin, self.vmax)
        # 插值：原始值 -> 坐标 0~1
        return np.interp(value, self._bin_edges[1:], self._cdf)

    def inverse(self, u):
        # 反归一化：色带 0~1 -> 原始值
        # 为避免插值边界问题，clip到[0,1]
        u = np.clip(u, 0, 1)
        return np.interp(u, self._cdf, self._bin_edges[1:])


class InvHistEqNorm(mcolors.Normalize):
    """反直方图均衡：聚集区拉伸，稀疏区压缩"""
    def __init__(self, data, vmin=None, vmax=None, bins=256, clip=False):
        vmin = np.min(data) if vmin is None else vmin
        vmax = np.max(data) if vmax is None else vmax
        super().__init__(vmin, vmax, clip)
        # 只用非nan数据
        data_flat = np.ravel(data[~np.isnan(data)])
        hist, bin_edges = np.histogram(data_flat, bins=bins, range=(vmin, vmax), density=True)
        # 累积分布函数（CDF）
        cdf = np.cumsum(hist)
        cdf = cdf / cdf[-1]
        self._cdf = cdf
        self._bin_edges = bin_edges

    def __call__(self, value, clip=None):
        # 反直方图均衡：插值 CDF → value
        value = np.clip(value, self.vmin, self.vmax)
        # 普通归一化：np.interp(value, self._bin_edges[1:], self._cdf)
        # 反归一化：np.interp(value, self._cdf, self._bin_edges[1:])
        # 先用普通均衡，把色带密集区压缩
        # 反均衡：让色带在高密度区拉伸
        return np.interp(value, self._cdf, self._bin_edges[1:])

    def inverse(self, u):
        # 色带0-1 → CDF → value
        u = np.clip(u, 0, 1)
        # 普通归一化是 value→cdf，反均衡是cdf→value
        return np.interp(u, self._cdf, self._bin_edges[1:])


def efficient_tif_plot(
        ax,
        tif_file,
        cmap='terrain',
        interpolation='nearest',
        title_name='',
        unit_name='',
        shp=None,
        line_color='black',
        line_width=1,
        legend_width="55%", legend_height="6%",
        legend_loc='lower left',
        legend_bbox_to_anchor=(0, 0, 1, 1),
        legend_borderpad=1,
        legend_nbins=5,
        char_ticks_length=3,
        char_ticks_pad=1,
        title_y=1,
        unit_labelpad=3,
        stretch=5 # not used for LogNorm, but kept for compatibility
):
    # === 读取栅格 ===
    with rasterio.open(tif_file) as src:
        bounds = src.bounds
        data = src.read(1, masked=True)
        extent = (bounds.left, bounds.right, bounds.bottom, bounds.top)
        raster_crs = src.crs

    # CRS 转换
    def _cartopy_crs_from_raster_crs(raster_crs):
        import cartopy.crs as ccrs
        try:
            epsg_code = int(str(raster_crs).split("EPSG:")[-1])
            return ccrs.epsg(epsg_code)
        except Exception:
            return ccrs.PlateCarree()

    data_crs = _cartopy_crs_from_raster_crs(raster_crs)

    # 绘制矢量
    if shp is not None:
        gdf = gpd.read_file(shp) if isinstance(shp, str) else shp
        gdf = gdf.to_crs(raster_crs)
        gdf.plot(ax=ax, edgecolor=line_color, linewidth=line_width, facecolor='none')
        minx, miny, maxx, maxy = gdf.total_bounds
        pad_x = (maxx - minx) * 0.02 or 1e-4
        pad_y = (maxy - miny) * 0.02 or 1e-4
        west, east = minx - pad_x, maxx + pad_x
        south, north = miny - pad_y, maxy + pad_y
        ax.set_extent((west, east, south, north), crs=data_crs)

    # 设置标题
    ax.set_title(title_name, y=0.95)
    ax.set_axis_off()

    # ==== LogNorm归一化 ====
    # 数据不能有0/负值，否则会报错
    data_safe = np.clip(data, 1e-6, None)
    vmin = float(np.nanmin(data_safe))
    vmax = float(np.nanmax(data_safe))
    norm = LogNorm(vmin=vmin, vmax=vmax)

    # ==== 绘制栅格 ====
    im = ax.imshow(
        data_safe,
        origin='upper',
        extent=extent,
        transform=data_crs,
        cmap=cmap,
        norm=norm,
        interpolation=interpolation
    )

    # === 色带/色条 ===
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.get_cmap(cmap))
    sm.set_array([])

    # 用线性 Normalize 构造“只给图例用”的 mappable（均匀使用色带）
    linear_sm = mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(0, 1), cmap=cmap)

    cax = inset_axes(
        ax,
        width=legend_width,
        height=legend_height,
        loc=legend_loc,
        borderpad=legend_borderpad,
        bbox_to_anchor=legend_bbox_to_anchor,
        bbox_transform=ax.transAxes,
    )

    cbar = plt.colorbar(
        linear_sm, cax=cax, orientation='horizontal',
        extend='both', extendfrac=0.1, extendrect=False
    )

    # 2) 均匀放置若干 tick（色带均匀）
    n_ticks = legend_nbins
    eps = 1e-6
    ticks_01 = np.linspace(eps, 1-eps, n_ticks)
    cbar.set_ticks(ticks_01)

    # 3) 0..1 反映射为数据值，并设置刻度标签
    # LogNorm的逆归一化
    def lognorm_inverse(norm, u):
        # norm(v) = (log(v) - log(vmin)) / (log(vmax) - log(vmin))
        # norm.inverse(u) = exp(u * (log(vmax) - log(vmin)) + log(vmin))
        log_vmin = np.log(norm.vmin)
        log_vmax = np.log(norm.vmax)
        val = np.exp(u * (log_vmax - log_vmin) + log_vmin)
        return val

    tick_vals = lognorm_inverse(norm, ticks_01)
    cbar.set_ticklabels([f"{v:,.0f}" for v in tick_vals])

    # 4) 其他外观保持不变
    cbar.outline.set_visible(False)
    cbar.ax.xaxis.set_label_position('top')
    cbar.formatter = mticker.StrMethodFormatter('{x:,.0f}')
    cbar.ax.tick_params(length=char_ticks_length, pad=char_ticks_pad)

    # 5) legend_nbins 支持 2/3/自定义
    if legend_nbins == 2:
        ticks_01 = np.array([eps, 1-eps])
    elif legend_nbins == 3:
        ticks_01 = np.linspace(eps, 1-eps, 3)
    else:
        ticks_01 = np.linspace(eps, 1-eps, legend_nbins)

    tick_vals = lognorm_inverse(norm, ticks_01)
    print(tick_vals)
    tick_vals_nice = nice_round(tick_vals)

    tick_vals_nice = [
        f"{v:,.3f}" if i != 0 and i != len(tick_vals_nice) - 1 and v < 1 else f"{v:,.0f}"
        for i, v in enumerate(tick_vals_nice)
    ]

    def fwd_map(v_array, norm):
        log_vmin = np.log(norm.vmin)
        log_vmax = np.log(norm.vmax)
        v_array = np.array(v_array, dtype=float)  # 强制转为 float
        v_array = np.clip(v_array, norm.vmin, norm.vmax)
        u_array = (np.log(v_array) - log_vmin) / (log_vmax - log_vmin)
        return u_array

    ticks_01_nice = fwd_map(tick_vals_nice, norm)
    cbar.set_ticks(ticks_01_nice)
    cbar.set_ticklabels(tick_vals_nice)

    if unit_name:
        cbar.set_label(unit_name, labelpad=unit_labelpad, family='Arial')

    ax.set_title(title_name, y=title_y, fontfamily='Arial', weight='bold')
    return im, cbar

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


def add_binary_gray_layer(ax, tif_file, gray_hex="#808080", alpha=0.6, zorder=10, debug=False):
    """
    将 tif 中 ==1 的像元画成灰色(可调透明度)，其它(0或NoData)透明。
    """
    with rasterio.open(tif_file) as src:
        band = src.read(1, masked=True)  # MaskedArray（nodata 已遮蔽为 mask=True）
        bounds = src.bounds
        rio_crs = src.crs

    # 1) 只把“未被掩膜 & 等于 1”的像元标为 1
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
    ax.set_facecolor('none')          # 透明，关键！
    ax.patch.set_alpha(0.0)
    return ax

def add_annotation(fig, x, y, width=None, height=None, text="",
                   style="box", anchor="ll",   # 'll' 左下角；'center' 中心
                   facecolor='white', edgecolor=None,
                   linecolor='black', linewidth=1.0,
                   textcolor='black', fontfamily='Arial',
                   gap=0.005):  # 正方形/线与文字的间距
    overlay = _get_overlay_ax(fig)
    trans = overlay.transAxes

    if anchor == "center":
        if style == "box":
            x0, y0 = x - width/2, y - height/2
            overlay.add_patch(patches.Rectangle(
                (x0, y0), width, height,
                facecolor=facecolor,
                edgecolor=('none' if edgecolor is None else edgecolor),
                linewidth=linewidth, transform=trans, zorder=1001))
            # 文字放在右侧
            overlay.text(x + width/2 + width/2 + gap, y, text, ha='left', va='center',
                         color=textcolor,  fontfamily=fontfamily,
                         transform=trans, zorder=1002)

        elif style == "line":
            overlay.add_line(Line2D([x - width/2, x + width/2], [y, y],
                                    color=linecolor, linewidth=linewidth,
                                    transform=trans, zorder=1001))
            overlay.text(x + width/2 + gap, y, text, ha='left', va='center',
                         color=textcolor,  fontfamily=fontfamily,
                         transform=trans, zorder=1002)

    else:  # 左下角锚点
        if style == "box":
            overlay.add_patch(patches.Rectangle(
                (x, y), width, height,
                facecolor=facecolor,
                edgecolor=('none' if edgecolor is None else edgecolor),
                linewidth=linewidth, transform=trans, zorder=1001))
            # 文字放在右侧
            overlay.text(x + width + gap, y + height/2, text, ha='left', va='center',
                         color=textcolor,  fontfamily=fontfamily,
                         transform=trans, zorder=1002)

        elif style == "line":
            overlay.add_line(Line2D([x, x + width], [y, y],
                                    color=linecolor, linewidth=linewidth,
                                    transform=trans, zorder=1001))
            overlay.text(x + width + gap, y, text, ha='left', va='center',
                         color=textcolor,  fontfamily=fontfamily,
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
    fig.text(x + scale_width/2, y + tick_h, f'{length_km} km',
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


base_dir = f"../../../output/{config.TASK_NAME}/carbon_price/0_base_data"
legend_nbins=3

tif_file = "xr_total_cost_carbon_100_amortised_2050.tif"
env_category = 'carbon_100'
arr_path = f"{base_dir}/map_data"
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
fig = plt.figure(figsize=(6,6), dpi=300,constrained_layout=False)
ax1 = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

from matplotlib.colors import LinearSegmentedColormap
cost_cmap = LinearSegmentedColormap.from_list("cost", ["#FFFEC2", "#FA4F00", "#A80000"])
efficient_tif_plot(
        ax1, f"{arr_path}/{tif_file}", cmap=cost_cmap,
        shp="../Map/AUS_line1.shp", line_width=0.3,
        title_name=title, unit_name=unit,
        legend_bbox_to_anchor=(0.1, 0.10, 0.8, 0.9), legend_nbins=legend_nbins,title_y=0.95,char_ticks_length=1, unit_labelpad=1
    )
add_binary_gray_layer(ax1, aligned_tif, gray_hex="#808080", alpha=0.6, zorder=15)
plt.savefig(f"../../../output/{config.TASK_NAME}/carbon_price/3_Paper_figure/map {title}.png", dpi=300, pad_inches=0.1,transparent=True)

env_names = [ 'xr_cost_ag','xr_cost_agricultural_management','xr_cost_non_ag','xr_cost_transition_ag2ag_diff','xr_transition_cost_ag2non_ag_amortised_diff']
title_names = ['Agriculture cost','Agricultural management cost','Non-agriculture cost','Transition(ag→ag) cost','Transition(ag→non-ag) cost']

for env, title in zip(env_names, title_names):
    unit = 'MAU$'
    # 画图（两者像元网格完全一致了）
    fig = plt.figure(figsize=(6, 6), dpi=300, constrained_layout=False)
    ax1 = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    tif_file = f"{env}_{env_category}_2050.tif"

    efficient_tif_plot(
        ax1, f"{arr_path}/{tif_file}", cmap=cost_cmap,
        shp="../Map/AUS_line1.shp", line_width=0.3,
        title_name=title, unit_name=unit,
        legend_bbox_to_anchor=(0.1, 0.10, 0.8, 0.9), legend_nbins=legend_nbins, title_y=0.95, char_ticks_length=1, unit_labelpad=1
    )
    add_binary_gray_layer(ax1, aligned_tif, gray_hex="#808080", alpha=0.6, zorder=15)
    plt.savefig(f"../../../output/{config.TASK_NAME}/carbon_price/3_Paper_figure/map {title}.png", dpi=300,
                pad_inches=0.1, transparent=True)

    plt.show()

# env_category = 'carbon_100_bio_50'
# title = 'Total cost'
# unit = 'MAU$'
# # 画图（两者像元网格完全一致了）
# fig = plt.figure(figsize=(6,6), dpi=300,constrained_layout=False)
# ax1 = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
# tif_file = f"xr_total_cost_{env_category}_amortised_2050.tif"
# from matplotlib.colors import LinearSegmentedColormap
# cost_cmap = LinearSegmentedColormap.from_list("cost", ["#FFFEC2", "#FA4F00", "#A80000"])
# efficient_tif_plot(
#         ax1, f"{arr_path}/{tif_file}", cmap=cost_cmap,
#         shp="../Map/AUS_line1.shp", line_width=0.3,
#         title_name=title, unit_name=unit,
#         legend_bbox_to_anchor=(0.1, 0.10, 0.8, 0.9), legend_nbins=legend_nbins,title_y=0.95,char_ticks_length=1, unit_labelpad=1
#     )
# add_binary_gray_layer(ax1, aligned_tif, gray_hex="#808080", alpha=0.6, zorder=15)
# plt.savefig(f"../../../output/{config.TASK_NAME}/carbon_price/3_Paper_figure/map {title} {env_category}.png", dpi=300, pad_inches=0.1,transparent=True)
#
# env_names = [ 'xr_cost_ag','xr_cost_agricultural_management','xr_cost_non_ag','xr_cost_transition_ag2ag_diff','xr_transition_cost_ag2non_ag_amortised_diff']
# title_names = ['Agriculture cost','Agricultural management cost','Non-agriculture cost','Transition(ag→ag) cost','Transition(ag→non-ag) cost']
#
# for env, title in zip(env_names, title_names):
#     unit = 'MAU$'
#     # 画图（两者像元网格完全一致了）
#     fig = plt.figure(figsize=(6, 6), dpi=300, constrained_layout=False)
#     ax1 = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
#     tif_file = f"{env}_{env_category}_2050.tif"
#
#     efficient_tif_plot(
#         ax1, f"{arr_path}/{tif_file}", cmap=cost_cmap,
#         shp="../Map/AUS_line1.shp", line_width=0.3,
#         title_name=title, unit_name=unit,
#         legend_bbox_to_anchor=(0.1, 0.10, 0.8, 0.9), legend_nbins=legend_nbins, title_y=0.95, char_ticks_length=1, unit_labelpad=1
#     )
#     add_binary_gray_layer(ax1, aligned_tif, gray_hex="#808080", alpha=0.6, zorder=15)
#     plt.savefig(f"../../../output/{config.TASK_NAME}/carbon_price/3_Paper_figure/map {title} {env_category}.png", dpi=300,
#                 pad_inches=0.1, transparent=True)
#
#     plt.show()




