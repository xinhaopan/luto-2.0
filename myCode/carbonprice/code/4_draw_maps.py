import geopandas as gpd
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.ticker as mticker
import matplotlib.pyplot as plt
from rasterio.warp import reproject, Resampling
import numpy as np
import rasterio
import cartopy.crs as ccrs
import matplotlib.colors as mcolors
import matplotlib as mpl
import matplotlib.patches as patches
from matplotlib.lines import Line2D
from matplotlib_map_utils.core.scale_bar import scale_bar as mmu_scale_bar
import matplotlib.patches as patches

import tools.config as config

import numpy as np
import matplotlib.colors as mcolors

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


def efficient_tif_plot(
        ax,
        # ==== 栅格参数 ====
        tif_file,  # GeoTIFF 路径
        cmap='terrain',  # 栅格颜色映射
        interpolation='nearest',  # 栅格插值方式

        # ==== 色条参数 ====
        title_name='',  # 图标题
        unit_name='',  # 色条单位

        # ==== 矢量参数 ====
        shp=None,  # 矢量文件路径或 GeoDataFrame
        line_color='black',  # 矢量线颜色
        line_width=1,  # 矢量线宽

        # ==== 图例（色条）参数 ====
        legend_width="55%", legend_height="6%",
        legend_loc='lower left',
        legend_bbox_to_anchor=(0, 0, 1, 1),
        legend_borderpad=1,
        legend_nbins=5,
        char_ticks_length=3,  # 色条刻度标签长度
        char_ticks_pad=1,  # 色条刻度标签与色条的距离

        # ==== 标题位置 ====
        title_y=1,
        unit_labelpad=3

):
    # === 读取栅格 ===
    with rasterio.open(tif_file) as src:
        bounds = src.bounds
        data = src.read(1, masked=True)
        extent = (bounds.left, bounds.right, bounds.bottom, bounds.top)
        raster_crs = src.crs

    # 自动取 min/max 作为色带范围
    import numpy as np
    finite_vals = data.compressed() if np.ma.is_masked(data) else data[np.isfinite(data)]
    vmin, vmax = (np.min(finite_vals), np.max(finite_vals)) if finite_vals.size else (0, 1)
    # norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    # 使用对数归一化，避免 0 或负值导致的错误
    # from matplotlib.colors import LogNorm
    # vals = data.compressed() if np.ma.is_masked(data) else data[np.isfinite(data)]
    # pos = vals[vals > 0]
    # vmin = np.percentile(pos, 0) if pos.size else 1e-6
    # vmax = np.percentile(pos, 100) if pos.size else 1
    # norm = LogNorm(vmin=vmin, vmax=vmax)
    norm = make_hist_eq_norm(data, mask=None, bins=216, clip_percent=(0, 100))


    # 栅格 CRS
    data_crs = _cartopy_crs_from_raster_crs(raster_crs)

    # 绘制栅格
    im = ax.imshow(
        data,
        origin='upper',
        extent=extent,
        transform=data_crs,
        cmap=cmap,
        norm=norm,
        interpolation=interpolation
    )

    # 绘制矢量
    if shp is not None:
        gdf = gpd.read_file(shp) if isinstance(shp, str) else shp
        gdf = gdf.to_crs(raster_crs)  # 投影到栅格 CRS
        gdf.plot(ax=ax, edgecolor=line_color, linewidth=line_width, facecolor='none')
        minx, miny, maxx, maxy = gdf.total_bounds  # (lon_min, lat_min, lon_max, lat_max)

        # 加一点 padding，防止是线要素导致高度/宽度为 0
        pad_x = (maxx - minx) * 0.02 or 1e-4
        pad_y = (maxy - miny) * 0.02 or 1e-4
        west, east = minx - pad_x, maxx + pad_x
        south, north = miny - pad_y, maxy + pad_y

        # 关键：用 PlateCarree，因为边界是经纬度
        ax.set_extent((west, east, south, north), crs=data_crs)

    # 设置标题
    ax.set_title(title_name, y=0.95)
    ax.set_axis_off()

    # === 色条 ===
    sm = plt.cm.ScalarMappable(norm=norm, cmap=plt.get_cmap(cmap))
    sm.set_array([])

    # cax = inset_axes(
    #     ax,
    #     width=legend_width,
    #     height=legend_height,
    #     loc=legend_loc,
    #     borderpad=legend_borderpad,
    #     bbox_to_anchor=legend_bbox_to_anchor,
    #     bbox_transform=ax.transAxes,
    # )
    # cbar = plt.colorbar(
    #     sm, cax=cax, orientation='horizontal',
    #     extend='both', extendfrac=0.1, extendrect=False
    # )
    # cbar.outline.set_visible(False)
    # cbar.ax.xaxis.set_label_position('top')
    # cbar.formatter = mticker.StrMethodFormatter('{x:,.0f}')
    # cbar.ax.tick_params(length=char_ticks_length, pad=char_ticks_pad)

    # if legend_nbins == 3:
    #     v0, v1 = float(vmin), float(vmax)
    #     ticks = [v0, (v0 + v1) / 2.0, v1]
    #     cbar.set_ticks(ticks)
    # if legend_nbins == 2:
    #     v0, v1 = float(vmin), float(vmax)
    #     ticks = [v0, v1]
    #     cbar.set_ticks(ticks)
    # else:
    #     # 自动刻度
    #     cbar.locator = mticker.MaxNLocator(nbins=legend_nbins)
    #
    # cbar.update_ticks()
    # if unit_name:
    #     cbar.set_label(unit_name, labelpad=unit_labelpad, family='Arial')


    # 1) 用线性的 Normalize 构造一个“只给图例用”的 mappable（均匀使用色带）
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
    n_ticks = 6
    ticks_01 = np.linspace(0, 1, n_ticks)
    cbar.set_ticks(ticks_01)

    # 3) 把 0..1 的位置“反映射”为数据值，并设置刻度标签
    def inv_map(u):
        norm = sm.norm
        # 直方图均衡（自定义 HistEqNorm）
        if hasattr(norm, "_cdf") and hasattr(norm, "_bin_edges"):
            # 扩展两端，保证端点正确
            cdf_x = np.concatenate(([0.0], norm._cdf, [1.0]))
            cdf_y = np.concatenate(([norm.vmin], norm._bin_edges[1:], [norm.vmax]))
            return np.interp(u, cdf_x, cdf_y)

        # 对数
        if isinstance(norm, mpl.colors.LogNorm):
            vmin, vmax = norm.vmin, norm.vmax
            return vmin * (vmax / vmin) ** u

        # 对称对数（简化近似；需更精确可再细化）
        if isinstance(norm, mpl.colors.SymLogNorm):
            vmin, vmax, lt = norm.vmin, norm.vmax, norm.linthresh
            s = 2 * u - 1
            out = np.empty_like(u, dtype=float)
            lin = np.abs(s) <= (lt / max(abs(vmin), abs(vmax)))
            out[lin] = s[lin] * lt
            out[~lin] = np.sign(s[~lin]) * lt * np.exp(
                (np.abs(s[~lin]) - (lt / max(abs(vmin), abs(vmax)))) * 5
            )
            return out

        # 线性
        vmin, vmax = norm.vmin, norm.vmax
        return vmin + u * (vmax - vmin)

    tick_vals = inv_map(ticks_01)
    cbar.set_ticklabels([f"{v:,.0f}" for v in tick_vals])

    # 4) 其他外观保持不变
    cbar.outline.set_visible(False)
    cbar.ax.xaxis.set_label_position('top')
    cbar.formatter = mticker.StrMethodFormatter('{x:,.0f}')
    cbar.ax.tick_params(length=char_ticks_length, pad=char_ticks_pad)

    # 均匀色带的 ticks（0–1 之间）
    if legend_nbins == 2:
        ticks_01 = np.array([0.0, 1.0])  # 位置在线性 0..1
    elif legend_nbins == 3:
        ticks_01 = np.linspace(0, 1, 3)  # 0, 0.5, 1
    else:
        ticks_01 = np.linspace(0, 1, legend_nbins)

    cbar.set_ticks(ticks_01)
    tick_vals = inv_map(ticks_01)
    cbar.set_ticklabels([f"{v:,.0f}" for v in tick_vals])

    if unit_name:
        cbar.set_label(unit_name, labelpad=unit_labelpad, family='Arial')

    ax.set_title(title_name, y=title_y, fontfamily='Arial')
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
                   textcolor='black', fontsize=8, fontfamily='Arial',
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
                         color=textcolor, fontsize=fontsize, fontfamily=fontfamily,
                         transform=trans, zorder=1002)

        elif style == "line":
            overlay.add_line(Line2D([x - width/2, x + width/2], [y, y],
                                    color=linecolor, linewidth=linewidth,
                                    transform=trans, zorder=1001))
            overlay.text(x + width/2 + gap, y, text, ha='left', va='center',
                         color=textcolor, fontsize=fontsize, fontfamily=fontfamily,
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
                         color=textcolor, fontsize=fontsize, fontfamily=fontfamily,
                         transform=trans, zorder=1002)

        elif style == "line":
            overlay.add_line(Line2D([x, x + width], [y, y],
                                    color=linecolor, linewidth=linewidth,
                                    transform=trans, zorder=1001))
            overlay.text(x + width + gap, y, text, ha='left', va='center',
                         color=textcolor, fontsize=fontsize, fontfamily=fontfamily,
                         transform=trans, zorder=1002)


from matplotlib_map_utils.core.scale_bar import ScaleBar, scale_bar
import matplotlib.patches as patches
from matplotlib.lines import Line2D


from matplotlib.lines import Line2D

def add_scalebar(fig, ax, x, y, length_km=500,
                 fontsize=9, fontfamily='Arial',
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
             ha='center', va='bottom', fontsize=fontsize,
             fontfamily=fontfamily, color=color, transform=transform)

import matplotlib.image as mpimg
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


arr_path = f"{config.TASK_DIR}/carbon_price/map_data"
ref_tif = f'{arr_path}/ghg_2050.tif'
src_tif = f'{arr_path}/public_area.tif'
aligned_tif = f'{arr_path}/public_area_aligned_to_ghg_2050.tif'

set_plot_style(font_size=4, font_family='Arial')
# 掩膜层用最近邻
align_raster_to_reference(src_tif, ref_tif, aligned_tif, resampling="nearest")

# 画图（两者像元网格完全一致了）
fig, axes = plt.subplots(3,2,subplot_kw={'projection': ccrs.PlateCarree()}, figsize=(4,6), dpi=300,constrained_layout=False)
axes = axes.flatten()

from matplotlib.colors import LinearSegmentedColormap
cmap_cost = LinearSegmentedColormap.from_list("cost", ["#4575B5", "#FFFFBF", "#9C551F"])
cmap_price = LinearSegmentedColormap.from_list("cost", ["#006100", "#FFFF00", "#FF2200"])

fields = {
    0: ("carbon_cost_2050.tif", "GHG reductions and removals cost", " AU$", cmap_cost,4),
    1: ("bio_cost_2050.tif","Biodiversity restoration cost", "AU$", cmap_cost,4),
    2: ("ghg_2050.tif","GHG reductions and removals", r"tCO$_2$e", "summer_r",5),
    3: ("bio_2050.tif","Biodiversity restoration", "ha", "summer_r",5),
    4: ("carbon_price_2050.tif","Shadow carbon price", r"AU\$ CO$_2$e$^{-1}$", cmap_price,4),
    5: ("bio_price_2050.tif","Shadow biodiversity price", r"AU\$ ha$^{-1}$", cmap_price,5)
}

for i, (tif_file, title, unit, cmap,ticks) in fields.items():
    efficient_tif_plot(
        axes[i], f"{arr_path}/{tif_file}", cmap=cmap,
        shp="../Map/AUS_line1.shp", line_width=0.3,
        title_name=title, unit_name=unit,
        legend_bbox_to_anchor=(0.1, 0.10, 0.8, 0.9), legend_nbins=ticks,title_y=0.95,char_ticks_length=1, unit_labelpad=1
    )
    add_binary_gray_layer(axes[i], aligned_tif, gray_hex="#808080", alpha=0.6, zorder=15)


plt.subplots_adjust(left=0.02, right=0.98, top=0.99, bottom=0.01, wspace=0, hspace=-0.4)
# 保存图像
# plt.savefig(f"{config.TASK_DIR}/carbon_price/Paper_figure/map_test.png", dpi=300, bbox_inches=None)
# plt.show()
font_size = axes[0].xaxis.get_label().get_size()
font_family = axes[0].xaxis.get_label().get_family()


# 新的调用方式
add_north_arrow(fig, 0.13, 0.07,size=0.015)
add_scalebar(fig, axes[0], 0.16, 0.079, length_km=500, linewidth=0.5, fontsize=font_size,fontfamily=font_family,)

add_annotation(fig, 0.22, 0.083, width=0.015, text="Australian state boundary",linewidth=0.5,
               style="line", linecolor="black",fontsize=font_size, fontfamily=font_family)
add_annotation(fig, 0.41, 0.08, width=0.0122, height=0.0072,linewidth=0.5, text="Invalid value",
               style="box", facecolor="white", edgecolor="black",fontsize=font_size, fontfamily=font_family)
add_annotation(fig, 0.52, 0.08, width=0.0122, height=0.0072,linewidth=0.5, text="Public, indigenous, urban, and other intensive land uses",
               style="box", facecolor="#808080",edgecolor="#808080",fontsize=font_size, fontfamily=font_family)

plt.savefig(f"{config.TASK_DIR}/carbon_price/Paper_figure/04_map.png", dpi=300, bbox_inches=None)
plt.show()