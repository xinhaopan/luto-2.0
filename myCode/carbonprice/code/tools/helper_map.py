import os

# gdal_dll_path = r"F:\xinhao\miniforge\envs\xpluto\Library\bin"
# if os.path.exists(gdal_dll_path):
#     os.environ['PATH'] = gdal_dll_path + os.pathsep + os.environ['PATH']
#     os.add_dll_directory(gdal_dll_path)

from osgeo import gdal, osr
from rasterio.warp import reproject, Resampling
import rasterio

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.patches as patches
import matplotlib.image as mpimg
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
import matplotlib as mpl
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import geopandas as gpd
from matplotlib.colors import LogNorm
import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cartopy.crs as ccrs
import os
import math
from matplotlib.colors import Normalize, LogNorm
import json
import pylustrator
from cmcrameri import cm
import cmocean


# import matplotlib
# matplotlib.use("QtAgg")
# pylustrator.start()

import tools.config as config
from tools.helper_plot import set_plot_style



def efficient_tif_plot(
        ax,
        tif_file,
        cmap='terrain',
        interpolation='nearest',
        title_name='',
        unit_name='',
        shp=None, line_color='black', line_width=2,
        legend_width="55%", legend_height="6%",
        legend_loc='lower left', legend_bbox_to_anchor=(0, 0, 1, 1),
        legend_borderpad=1, legend_nbins=5,
        char_ticks_length=3, char_ticks_pad=1,
        title_y=1, unit_labelpad=5,
        decimal_places=None, clip_percent=None,
        custom_tick_values=False,
):

    # 读取栅格数据
    with rasterio.open(tif_file) as src:
        bounds = src.bounds
        data = src.read(1)
        nodata = src.nodata
        raster_crs = src.crs
        extent = (bounds.left, bounds.right, bounds.bottom, bounds.top)

    # 处理无效值
    if nodata is not None:
        data = np.where(data == nodata, np.nan, data)

    # 移除非正值（如果需要）
    # data = np.where(data <= 0, np.nan, data)

    # 检查数据有效性
    valid_mask = ~np.isnan(data)
    if not np.any(valid_mask):
        print("Warning: No valid data found!")
        return None, None

    # CRS
    data_crs = _cartopy_crs_from_raster_crs(raster_crs)

    # 矢量边界
    if shp is not None:
        gdf = gpd.read_file(shp) if isinstance(shp, str) else shp
        gdf = gdf.to_crs(raster_crs)
        gdf.plot(ax=ax, edgecolor=line_color, linewidth=line_width, facecolor='none')
        minx, miny, maxx, maxy = gdf.total_bounds
        pad_x = (maxx - minx) * 0.02 or 1e-4
        pad_y = (maxy - miny) * 0.02 or 1e-4
        ax.set_extent((minx - pad_x, maxx + pad_x, miny - pad_y, maxy + pad_y), crs=data_crs)

    ax.set_title(title_name, y=title_y, fontfamily='Arial')
    ax.set_axis_off()

    # 计算数据范围
    valid_data = data[valid_mask]
    vmin_real = float(np.nanmin(valid_data))
    vmax_real = float(np.nanmax(valid_data))

    # 非负数据从 0 起；否则用真实最小值
    vmin_data = 0.0 if vmin_real >= 0 else vmin_real

    if clip_percent is not None:
        # 合法性保护：限制在 0~100 范围
        percentiles = [min(max(float(p), 0.0), 100.0) for p in clip_percent]
        vmin_data, vmax_data = np.nanpercentile(valid_data, percentiles)
    else:
        vmin_data = vmin_real
        vmax_data = vmax_real

    # 使用 get_y_axis_ticks 获取优化的范围和刻度
    if custom_tick_values is not False:
        tick_vals = np.asarray(list(custom_tick_values), dtype=float)
        # 对于自定义刻度，使用数据范围
        vmin_plot = vmin_data
        vmax_plot = vmax_data
    else:
        nb = max(int(legend_nbins), 2)  # 至少 2 个刻度
        vmin_plot, vmax_plot, ticks_list = get_y_axis_ticks(vmin_data, vmax_data, desired_ticks=nb,strict_count=True)
        tick_vals = np.asarray(ticks_list, dtype=float)

    # 创建归一化对象 - 使用实际的绘图范围
    norm = Normalize(vmin=vmin_plot, vmax=vmax_plot, clip=True)
    cmap_obj = mpl.colormaps.get_cmap(cmap).copy()

    # 绘制栅格 - 使用实际的绘图范围
    im = ax.imshow(
        data,
        origin='upper',
        extent=extent,
        transform=data_crs,
        interpolation=interpolation,
        cmap=cmap_obj,
        vmin=vmin_plot,
        vmax=vmax_plot,
    )

    # 创建colorbar - 直接使用 im 而不是创建新的 ScalarMappable
    cax = inset_axes(
        ax, width=legend_width, height=legend_height, loc=legend_loc,
        borderpad=legend_borderpad, bbox_to_anchor=legend_bbox_to_anchor,
        bbox_transform=ax.transAxes,
    )

    cbar = plt.colorbar(
        im, cax=cax, orientation='horizontal',
        extend='both',
        extendfrac=0.1, extendrect=False
    )

    # 仅保留落在绘图范围内的刻度
    eps = 1e-12
    in_range = (tick_vals >= vmin_plot - eps) & (tick_vals <= vmax_plot + eps)
    tick_vals_filtered = tick_vals[in_range]

    # 直接设置刻度值 - 不需要映射到0-1
    cbar.set_ticks(tick_vals_filtered)
    cbar.set_ticklabels(_format_tick_labels(tick_vals_filtered, decimal_places))

    cbar.outline.set_visible(False)
    cbar.ax.xaxis.set_label_position('top')
    cbar.ax.tick_params(length=char_ticks_length, pad=char_ticks_pad)
    if unit_name:
        cbar.set_label(unit_name, labelpad=unit_labelpad, family='Arial')

    return im, cbar


def get_y_axis_ticks(min_value, max_value, desired_ticks=5, min_upper=None, strict_count=False):
    """
    生成Y轴刻度，根据数据范围智能调整刻度间隔和范围。
    参数 strict_count:
      - True: 美化刻度算法，但最后严格输出 desired_ticks 个刻度（在美化后的范围内均分）
      - False: 原有行为，刻度数量美化
    新增参数：
      - min_upper: 若提供，则保证返回的 max_v 和最上端刻度 >= min_upper（确保最大值只增不减）
    """
    # 0) 如需保持上界不回退，先将 max_value 至少抬到 min_upper
    if min_upper is not None:
        max_value = max(max_value, float(min_upper))

    # 1. 快速处理特殊情况
    if min_value > 0 and max_value > 0:
        min_value = 0
    elif min_value < 0 and max_value < 0:
        max_value = 0

    range_value = max_value - min_value
    if range_value <= 0:
        return (0, 1, [0, 0.5, 1])

    # 2. 计算“nice”间隔
    ideal_interval = range_value / (desired_ticks - 1)
    e = math.floor(math.log10(ideal_interval))
    base = 10 ** e
    normalized_interval = ideal_interval / base
    nice_intervals = [1, 2, 5, 10]
    interval = min(nice_intervals, key=lambda x: abs(x - normalized_interval)) * base

    # 3. 对齐边界
    min_tick = math.floor(min_value / interval) * interval
    max_tick = math.ceil(max_value / interval) * interval

    # 4. 生成刻度
    tick_count = int((max_tick - min_tick) / interval) + 1
    ticks = np.linspace(min_tick, max_tick, tick_count)

    # 压缩刻度数量（尽量接近 desired_ticks）
    if len(ticks) > desired_ticks + 1:
        scale = math.ceil((len(ticks) - 1) / (desired_ticks - 1))
        interval *= scale
        min_tick = math.floor(min_value / interval) * interval
        max_tick = math.ceil(max_value / interval) * interval
        tick_count = int((max_tick - min_tick) / interval) + 1
        ticks = np.linspace(min_tick, max_tick, tick_count)

    # 5. 插入 0
    if min_value < 0 < max_value and 0 not in ticks:
        zero_idx = np.searchsorted(ticks, 0)
        ticks = np.insert(ticks, zero_idx, 0)

    close_threshold = 0.3 * interval
    max_v = max_tick
    min_v = min_tick

    # 6. 末端微调
    if len(ticks) >= 2:
        # 顶端
        if (ticks[-1] != 0 and
            (max_value - ticks[-2]) < close_threshold and
            (ticks[-1] - max_value) > close_threshold):
            ticks = ticks[:-1]
            max_v = max_value + 0.1 * interval

        # 底端
        if (ticks[0] != 0 and
            (ticks[1] - min_value) < close_threshold and
            (min_value - ticks[0]) > close_threshold):
            ticks = ticks[1:]
            min_v = min_value - 0.1 * interval
        elif abs(min_value) < interval:
            min_v = math.floor(min_value)

    # 7. 0-100 特例
    if ((abs(ticks[0]) < 1e-10 and abs(ticks[-1] - 100) < 1e-10)
        or (min_tick == 0 and max_tick == 100)):
        ticks = np.array([0, 25, 50, 75, 100])

    # 8) 强制不回退：若提供了 min_upper，确保上界与最上刻度不低于它
    if min_upper is not None:
        target = float(min_upper)
        if max_v < target:
            max_v = target
        # 若最上刻度低于 target，把上端刻度抬到不低于 target 的下一“格”
        top_tick_needed = math.ceil(target / interval) * interval
        if ticks[-1] < top_tick_needed:
            tick_count = int((top_tick_needed - ticks[0]) / interval) + 1
            ticks = np.linspace(ticks[0], top_tick_needed, tick_count)

    # === 核心部分：严格保证刻度数量 ===
    if strict_count:
        nice_rounds = nice_round([min_v, max_v])
        min_v, max_v = nice_rounds[0], nice_rounds[1]
        ticks = np.linspace(min_v, max_v, desired_ticks)

    return (min_v, max_v, ticks.tolist())

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
                         color=textcolor, fontfamily=fontfamily, fontsize=fontsize,
                         transform=trans, zorder=1002)

        elif style == "line":
            overlay.add_line(Line2D([x - width / 2, x + width / 2], [y, y],
                                    color=linecolor, linewidth=linewidth,
                                    transform=trans, zorder=1001))
            overlay.text(x + width / 2 + gap, y, text, ha='left', va='center',
                         color=textcolor, fontfamily=fontfamily, fontsize=fontsize,
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
                         color=textcolor, fontfamily=fontfamily, fontsize=fontsize,
                         transform=trans, zorder=1002)

        elif style == "line":
            overlay.add_line(Line2D([x, x + width], [y, y],
                                    color=linecolor, linewidth=linewidth,
                                    transform=trans, zorder=1001))
            overlay.text(x + width + gap, y, text, ha='left', va='center',
                         color=textcolor, fontfamily=fontfamily, fontsize=fontsize,
                         transform=trans, zorder=1002)


def add_scalebar(fig, ax, x, y, length_km=500,
                 fontfamily='Arial', fontsize=12,
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

    artists = []

    # 横线
    artists.append(Line2D([x, x + scale_width], [y, y],
                          color=color, linewidth=linewidth,
                          transform=transform, clip_on=False))

    # 竖线（下端与横线齐平，向上）
    tick_h = 0.005 if transform_type == 'figure' else 0.02
    for tx in (x, x + scale_width):
        artists.append(Line2D([tx, tx], [y, y + tick_h],
                              color=color, linewidth=linewidth,
                              transform=transform, clip_on=False))

    # 标签
    text_artist = fig.text(x + scale_width / 2, y + tick_h, f'{length_km} km',
                           ha='center', va='bottom', fontsize=fontsize,
                           fontfamily=fontfamily, color=color, transform=transform)
    artists.append(text_artist)

    # 一起加到 fig
    for artist in artists:
        artist.set_zorder(150)
        fig.add_artist(artist)


def add_north_arrow(fig, x, y, size=0.1, img_path='../Map/north_arrow.png', transform_type='figure'):
    """
    在图上添加自定义指北针图像

    Parameters:
    -----------
    fig : matplotlib.figure.Figure
    x, y : float           # 左上角位置 (0-1)
    size : float           # 指北针相对于画布的宽度 (0-1)
    img_path : str         # 图片路径
    transform_type : str   # 'figure' 或 'axes'
    """
    img = mpimg.imread(img_path)
    w = size
    h = size * img.shape[0] / img.shape[1]  # 保持宽高比

    if transform_type == 'figure':
        # 输入的 x, y 为左下角
        ax_img = fig.add_axes([x, y, w, h],zorder=120)
    else:
        raise NotImplementedError("当前示例只做 figure 坐标系")

    ax_img.imshow(img)
    ax_img.axis('off')  # 不显示坐标轴


def plot_tif_layer(
        tif_path: str,
        title: str,
        unit: str,
        cmap,
        outfile: str = None,
        ax=None,
        decimal_places=2,
        custom_tick_values=False,
        line_width=1,
        title_y=0.9,
        unit_labelpad=5,
        char_ticks_length=1,
        legend_nbins=3,
        legend_bbox=(0.1, 0.10, 0.8, 0.9),
        clip_percent=None
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
        tif_path,
        cmap=cmap,
        shp="../Map/AUS_line1.shp",
        line_width=line_width,
        title_name=title,
        unit_name=unit,
        legend_bbox_to_anchor=legend_bbox,
        legend_nbins=legend_nbins,
        title_y=title_y,
        char_ticks_length=char_ticks_length,
        unit_labelpad=unit_labelpad,
        decimal_places=decimal_places,
        custom_tick_values=custom_tick_values,
        clip_percent=clip_percent
    )
    aligned_tif = f"../Map/public_area_aligned.tif"
    add_binary_gray_layer(ax, aligned_tif, gray_hex="#808080", alpha=1, zorder=15)

    # 只有单图模式才保存和显示
    if should_save and outfile:
        plt.savefig(outfile, dpi=300, pad_inches=0.1, transparent=True)
        plt.show()
        plt.close(fig)


def safe_plot(*, tif_path, title, unit, cmap, outfile=None, ax=None,**kwargs):
    """既支持单图保存，也支持子图绘制"""
    if ax is None:
        print(f"[INFO] Plotting {tif_path}")
        os.makedirs(os.path.dirname(outfile), exist_ok=True)
    else:
        print(f"[INFO] Plotting {tif_path} to subplot: {title}")

    if not os.path.exists(tif_path):
        print(f"[SKIP] Not found: {tif_path}")
        if ax is not None:
            # 在子图上显示错误信息
            ax.text(0.5, 0.5, f"File not found:\n{os.path.basename(tif_path)}",
                    ha='center', va='center', transform=ax.transAxes,
                    fontsize=8, color='red')
            ax.set_title(title, fontsize=10)
        return

    plot_tif_layer(
        tif_path=tif_path,
        title=title,
        unit=unit,
        cmap=cmap,
        outfile=outfile,
        ax=ax,
        **kwargs
    )
