import matplotlib
matplotlib.use('Agg')
from rasterio.warp import reproject, Resampling
from matplotlib.lines import Line2D
import matplotlib.patches as patches
import matplotlib.image as mpimg
from matplotlib.colors import LinearSegmentedColormap
import rasterio
import matplotlib.gridspec as gridspec
import cartopy.crs as ccrs
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import Normalize, LogNorm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import geopandas as gpd
import numpy as np
import math
from cmcrameri import cm
import cmocean

# import matplotlib
# matplotlib.use("QtAgg")
# pylustrator.start()

import tools.config as config
from tools.helper_plot import set_plot_style

# 若你已定义以下函数/类，可直接使用你自己的版本
# from your_module import make_hist_eq_norm, StretchHighLowNormalize
# 这里假设它们已在当前作用域中可用

def draw_hex_map_agg_linear(
    ax: plt.Axes | None,
    shp_path: str,
    which: str = "sum",                  # "sum" / "mean" 或具体列名（如 "value_sum"）
    admin_shp: str | None = None,        # 可选：行政边界（只画边线）
    # —— 标题/单位 & 图例参数（沿用你的接口）——
    title_name: str = '',
    unit_name: str = '',
    line_color: str = 'black',
    line_width: float = 1.5,
    legend_width: str = "55%",
    legend_height: str = "6%",
    legend_loc: str = 'lower left',
    legend_bbox_to_anchor=(0, 0, 1, 1),
    legend_borderpad: float = 1,
    legend_nbins: int = 4,
    char_ticks_length: float = 3,
    char_ticks_pad: float = 1,
    title_y: float = 1.5,
    unit_labelpad: float = 4,
    decimal_places: int | None = None,
    custom_tick_values=False,            # False 或传入刻度值（原值空间）
    # —— 绘图外观 ——
    cmap: str = "viridis",
    edgecolor: str = "white",
    linewidth: float = 0.6,
    # 线性拉伸参数
    vmax_percentile: float | None = None,
):
    """
    读取聚合 shp，按 which 选择列着色；使用默认线性拉伸（Normalize）。
    色条采用线性 mappable + 线性映射到 0..1 的刻度位置。
    需要你环境里已有：nice_round、_format_tick_labels（若无，可自行替换为简单格式化）。
    """
    gdf = gpd.read_file(shp_path)

    # —— 选择列：sum/mean 或具体列名（兼容 *_sum / *_mean）——
    which_l = which.lower()
    if which_l in ("sum", "mean"):
        def _match(col: str) -> bool:
            cl = col.lower()
            return cl == which_l or cl.endswith(f"_{which_l}")
        matches = [c for c in gdf.columns if _match(c)]
        exact = [c for c in matches if c.lower() == which_l]
        chosen = exact[0] if exact else (matches[0] if matches else None)
        if chosen is None:
            if which in gdf.columns:
                chosen = which
            else:
                raise ValueError(f"未找到列 '{which}'（或 *_{which}）。现有列：{list(gdf.columns)}")
    else:
        if which not in gdf.columns:
            raise ValueError(f"列 '{which}' 不存在。现有列：{list(gdf.columns)}")
        chosen = which

    gdf = gdf.dropna(subset=[chosen])
    if gdf.empty:
        raise ValueError(f"列 {chosen} 全为缺失，无法绘制。")

    # —— 线性归一化（支持百分位截顶）——
    vals = gdf[chosen].to_numpy()
    vmin_real = float(np.nanmin(vals))
    vmax_real = float(np.nanmax(vals))

    # 非负数据从 0 起；否则用真实最小值
    vmin_plot = 0.0 if vmin_real >= 0 else vmin_real

    if vmax_percentile is not None:
        # 合法性保护
        p = float(vmax_percentile)
        p = min(max(p, 0.0), 100.0)
        vmax_plot = float(np.nanpercentile(vals, p))
    else:
        vmax_plot = vmax_real

    # 夹紧：>vmax_plot 的值统一用最右端颜色，<vmin_plot 用最左端颜色
    norm = Normalize(vmin=vmin_plot, vmax=vmax_plot, clip=True)

    # 只创建一次 cmap，并可按需定制 under 颜色（clip=True 时基本用不到）
    cmap_obj = mpl.colormaps.get_cmap(cmap).copy()
    cmap_obj.set_under(cmap_obj(0.0))

    # —— 主图层 ——
    coll = gdf.plot(column=chosen, cmap=cmap_obj, norm=norm,
                    edgecolor=edgecolor, linewidth=linewidth, legend=False, ax=ax)
    # 行政边界（只画边线）
    main_gdf = gpd.read_file(shp_path)
    main_crs = main_gdf.crs
    data_crs = _cartopy_crs_from_raster_crs(main_crs)
    if admin_shp is not None:
        gdf_admin = gpd.read_file(admin_shp) if isinstance(admin_shp, str) else admin_shp
        gdf_admin = gdf_admin.to_crs(main_crs)
        gdf_admin.plot(ax=ax, edgecolor=line_color, linewidth=line_width, facecolor='none')
        minx, miny, maxx, maxy = gdf_admin.total_bounds
        pad_x = (maxx - minx) * 0.02 or 1e-4
        pad_y = (maxy - miny) * 0.02 or 1e-4
        ax.set_extent((minx - pad_x, maxx + pad_x, miny - pad_y, maxy + pad_y), crs=data_crs)

    ax.set_axis_off()
    if title_name:
        ax.set_title(title_name, y=title_y, fontfamily='Arial')

    # —— 色条：线性 mappable（0..1） + 线性映射刻度 ——
    linear_sm = mpl.cm.ScalarMappable(norm=mcolors.Normalize(0, 1), cmap=cmap_obj)  # ← 用 cmap_obj
    cax = inset_axes(
        ax, width=legend_width, height=legend_height, loc=legend_loc,
        borderpad=legend_borderpad, bbox_to_anchor=legend_bbox_to_anchor,
        bbox_transform=ax.transAxes,
    )
    cbar = plt.colorbar(
        linear_sm, cax=cax, orientation='horizontal',
        extend='both', extendfrac=0.1, extendrect=False
    )

    # span = max(norm.vmax - norm.vmin, 1e-12)
    #
    # if custom_tick_values is not False:
    #     # 用户指定数值 -> 按真实值映射位置
    #     vals_for_pos = np.asarray(list(custom_tick_values), dtype=float)
    #     ticks_01 = np.clip((vals_for_pos - norm.vmin) / span, 0, 1)
    #     labels_vals = vals_for_pos
    # else:
    #     # 自动：位置等距，确保中间=0.5
    #     N = max(int(legend_nbins), 2)
    #     ticks_01 = np.linspace(0, 1, N)  # ★ 等距位置
    #     labels_vals = norm.vmin + ticks_01 * span  # 对应的原值
    #
    # # 标签做“好看”四舍五入，但不改变位置
    # labels_vals_rounded = nice_round(labels_vals)
    # tick_labels = _format_tick_labels(labels_vals_rounded, decimal_places)
    #
    # cbar.set_ticks(ticks_01)
    # cbar.set_ticklabels(tick_labels)

    if custom_tick_values is not False:
        tick_vals = np.asarray(list(custom_tick_values), dtype=float)
    else:
        nb = max(int(legend_nbins), 2)  # 至少 2 个刻度
        v_minv, v_maxv, ticks_list = get_y_axis_ticks(vmin_plot, vmax_plot, desired_ticks=nb)
        tick_vals = np.asarray(ticks_list, dtype=float)

    # 仅保留落在 [vmin, vmax] 范围内的刻度，避免超界标签
    eps = 1e-12
    in_range = (tick_vals >= v_minv - eps) & (tick_vals <= v_maxv + eps)
    tick_vals = tick_vals[in_range]

    # 映射到 0..1（线性 mappable 的坐标）
    span = max(v_maxv - v_minv, 1e-12)
    ticks_01 = np.clip((tick_vals - norm.vmin) / span, 0.0, 1.0)

    # 应用到 colorbar
    cbar.set_ticks(ticks_01)
    cbar.set_ticklabels(_format_tick_labels(tick_vals, decimal_places))



    cbar.outline.set_visible(False)
    cbar.ax.xaxis.set_label_position('top')
    cbar.ax.tick_params(length=char_ticks_length, pad=char_ticks_pad)
    if unit_name:
        cbar.set_label(unit_name, labelpad=unit_labelpad, family='Arial')

    return coll, cbar


def get_y_axis_ticks(min_value, max_value, desired_ticks=5, min_upper=None):
    """
    生成Y轴刻度，根据数据范围智能调整刻度间隔和范围。
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

    return (min_v, max_v, ticks.tolist())

# def nice_round(values):
#     """对数组 values 里的数进行数量级四舍五入，
#     - 1 位数：个位
#     - 2/3 位数：十位
#     - 4 位数：百位
#     - 5 位数：千位
#     - 6 位及以上：只保留前两位
#     """
#     rounded = []
#     for v in values:
#         if v <= 1 or np.isnan(v):
#             rounded.append(v)
#             continue
#
#         magnitude = int(np.floor(np.log10(abs(v))))  # 数量级
#         if magnitude == 0:
#             # 个位数
#             r = round(v)
#         elif magnitude in (1, 2):
#             # 两位数或三位数 -> 十位
#             r = round(v, -1)
#         elif magnitude == 3:
#             # 四位数 -> 百位
#             r = round(v, -2)
#         elif magnitude == 4:
#             # 五位数 -> 千位
#             r = round(v, -3)
#         else:
#             # 六位及以上 -> 保留前两位
#             digits_to_keep = magnitude - 1  # 保留前两位
#             r = round(v, -digits_to_keep)
#         rounded.append(r)
#     return np.array(rounded)



def nice_round(values):
    """
    按整组数据范围选取 1–2–5×10^e 的“nice”间隔，
    再把各值四舍五入到该间隔的整数倍。
    - NaN 保留
    - |v| <= 1 保留原值（与旧实现一致）
    """
    arr = np.asarray(values, dtype=float)
    out = arr.copy()

    # 有效值（排除 NaN/Inf）
    finite_mask = np.isfinite(arr)
    if not np.any(finite_mask):
        return out  # 全是 NaN/Inf，原样返回

    vmin = np.nanmin(arr[finite_mask])
    vmax = np.nanmax(arr[finite_mask])
    rng  = vmax - vmin
    if not np.isfinite(rng) or rng <= 0:
        return out  # 无范围或范围为0，不做处理

    desired_ticks = 5  # 与刻度函数保持一致的默认“目标刻度数”
    ideal_interval = rng / (desired_ticks - 1)

    # 选取最接近理想间隔的“nice”间隔（1/2/5/10 × 10^e）
    e = math.floor(math.log10(ideal_interval))
    base = 10 ** e
    normalized = ideal_interval / base
    nice_choices = [1, 2, 5, 10]
    interval = min(nice_choices, key=lambda x: abs(x - normalized)) * base

    if not np.isfinite(interval) or interval == 0:
        return out

    # 对 |v|>1 的有效值进行就近取整到 interval 的整数倍；其余保持原值
    mask_to_round = finite_mask & (np.abs(arr) > 1)
    out[mask_to_round] = np.round(arr[mask_to_round] / interval) * interval
    return out


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


# def _format_tick_labels(values, decimal_places=None):
#     labels = []
#     for i, v in enumerate(values):
#         if v < 1:
#             if decimal_places is not None:
#                 # 自动判断格式
#                 if i == 0:
#                     # 首尾用整数格式
#                     labels.append(f"{v:,.0f}")
#                 else:
#                     # 其他用整数格式
#                     labels.append(f"{v:,.{decimal_places}f}")
#         else:
#             labels.append(f"{v:,.0f}")
#
#     return labels

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
        shp_file: str,
        title: str,
        unit: str,
        cmap,
        outfile: str = None,
        ax=None,
        decimal_places=2,
        custom_tick_values=False,
        line_width=1,
        title_y=0.95,
        which="sum",
        vmax_percentile=None,
        **kwargs
):
    """通用绘制函数：既可以保存单图，也可以在指定ax上绘制"""

    if ax is None:
        fig = plt.figure(figsize=(6, 6), dpi=300, constrained_layout=False)
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
        should_save = True
    else:
        should_save = False

    # 把所有参数集中到一个字典
    args_dict = dict(
        ax=ax,
        shp_path=os.path.join(arr_path, shp_file),
        which=which,
        cmap=cmap,
        admin_shp="../Map/AUS_line1.shp",
        line_width=line_width,
        title_name=title,
        unit_name=unit,
        legend_bbox_to_anchor=(0.1, 0.02, 0.8, 0.9),
        title_y=title_y,
        char_ticks_length=1,
        unit_labelpad=5,
        decimal_places=decimal_places,
        custom_tick_values=custom_tick_values,
        vmax_percentile=vmax_percentile
    )
    args_dict.update(kwargs)  # 合并其它额外参数

    # 一次性传递所有参数
    draw_hex_map_agg_linear(**args_dict)

    if should_save and outfile:
        plt.savefig(outfile, dpi=300, pad_inches=0.1, transparent=True)
        plt.show()
        plt.close(fig)


def safe_plot(*, tif_file, title, unit, cmap, outfile=None, ax=None, which='sum',vmax_percentile=None,**kwargs):
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
        shp_file=tif_file,
        title=title,
        unit=unit,
        cmap=cmap,
        outfile=outfile,
        ax=ax,
        which=which,
        vmax_percentile=vmax_percentile,
        **kwargs
    )


def plot_all_for_scenario(env: str,shp_name: str, cfg: dict, year: int = 2050, combined_mode=False,
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
    tif = f"{env}/{shp_name}/{shp_name}_total_cost_{env}_{year}.shp"
    kwargs = layer_overrides.get('total_cost', {})

    gs = gridspec.GridSpec(6, 4, figure=combined_fig, hspace=0, wspace=0.03,
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
                ax=ax,
                ** kwargs
            )
            plot_idx_ref[0] += 1
    else:
        out = os.path.join(out_dir, env, f"map Total cost {env}.png")
        safe_plot(
            tif_file=tif,
            title="Total cost",
            unit=r"MAU\$ yr$^{-1}$",
            cmap=cost_cmap,
            outfile=out,
            ** kwargs
        )

    # 2) Cost components
    for key, title in zip(env_keys, title_keys):
        kwargs = layer_overrides.get(key, {})
        tif = f"{env}/{shp_name}/{shp_name}_{key}_{env}_{year}.shp"

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
                ** kwargs
            )

    # 3) Benefits
    if "ghg" in cfg.get("benefits", []):
        kwargs = layer_overrides.get('total_carbon', {})
        tif = f"{env}/{shp_name}/{shp_name}_total_carbon_{env}_{year}.shp"
        if combined_mode:
            if plot_idx_ref[0] < nrows * ncols:
                ax = combined_fig.add_subplot(6, 4, plot_idx_ref[0] + 1, projection=ccrs.PlateCarree())
                safe_plot(
                    tif_file=tif,
                    title=f"Change in GHG benefit",
                    unit=r"MtCO$_2$e yr$^{-1}$",
                    cmap=benefit_cmap,
                    ax=ax,
                    ** kwargs
                )
                plot_idx_ref[0] += 1
        else:
            out = os.path.join(out_dir, env, f"map GHG benefit {env}.png")
            safe_plot(
                tif_file=tif,
                title="Change in GHG benefit",
                unit=r"MtCO$_2$e yr$^{-1}$",
                cmap=benefit_cmap,
                outfile=out,
                ** kwargs
            )

    if "bio" in cfg.get("benefits", []):
        kwargs = layer_overrides.get('total_bio', {})
        tif = f"{env}/{shp_name}/{shp_name}_total_bio_{env}_{year}.shp"
        if combined_mode:
            if plot_idx_ref[0] < nrows * ncols:
                ax = combined_fig.add_subplot(6, 4, plot_idx_ref[0] + 1, projection=ccrs.PlateCarree())
                safe_plot(
                    tif_file=tif,
                    title=f"Change in biodiversity benefit",
                    unit="Contribution-weighted\narea, ha yr$^{-1}$",
                    cmap=benefit_cmap,
                    ax=ax,
                    ** kwargs
                )
                plot_idx_ref[0] += 1
        else:
            out = os.path.join(out_dir, env, f"map Biodiversity benefit {env}.png")
            safe_plot(
                tif_file=tif,
                title="Change in biodiversity benefit",
                unit="Contribution-weighted\narea, ha yr$^{-1}$",
                cmap=benefit_cmap,
                outfile=out,
                ** kwargs
            )

    # 4) Prices
    if "carbon" in cfg.get("prices", []):
        kwargs = layer_overrides.get('carbon_price', {})
        title_name = layer_overrides.get(env, {}).get("title_name", "Carbon price")

        tif = f"{env}/{shp_name}/{shp_name}_carbon_price_{env}_{year}.shp"
        if combined_mode:
            if plot_idx_ref[0] < nrows * ncols:
                ax = combined_fig.add_subplot(6, 4, plot_idx_ref[0] + 1, projection=ccrs.PlateCarree())
                safe_plot(
                    tif_file=tif,
                    title=title_name,
                    unit=r"AU\$ CO$_2$e$^{-1}$ yr$^{-1}$",
                    cmap=price_cmap,
                    ax=ax,
                    which="mean",
                    vmax_percentile=90,
                    **kwargs

                )
                plot_idx_ref[0] += 1
        else:
            out = os.path.join(out_dir, env, f"map Carbon price {env}.png")
            safe_plot(
                tif_file=tif,
                title=title_name,
                unit=r"AU\$ CO$_2$e$^{-1}$ yr$^{-1}$",
                cmap=price_cmap,
                outfile=out,
                which="mean",
                vmax_percentile=90,
                ** kwargs
            )

    if "bio" in cfg.get("prices", []):
        kwargs = layer_overrides.get('bio_price', {})
        tif = f"{env}/{shp_name}/{shp_name}_bio_price_{env}_{year}.shp"
        if combined_mode:
            if plot_idx_ref[0] < nrows * ncols:
                ax = combined_fig.add_subplot(6, 4, plot_idx_ref[0] + 1, projection=ccrs.PlateCarree())
                safe_plot(
                    tif_file=tif,
                    title=f"Biodiversity cost",
                    unit=r"AU\$ ha$^{-1}$ yr$^{-1}$",
                    cmap=price_cmap,
                    ax=ax,
                    vmax_percentile=90,
                    ** kwargs
                )
                plot_idx_ref[0] += 1
        else:
            out = os.path.join(out_dir, env, f"map Biodiversity price {env}.png")
            safe_plot(
                tif_file=tif,
                title="Biodiversity price",
                unit=r"AU\$ ha$^{-1}$ yr$^{-1}$",
                cmap=price_cmap,
                outfile=out,
                vmax_percentile=90,
                ** kwargs
            )
    if combined_mode:
        return combined_fig,ax


def plot_all_scenarios_combined(scenarios: dict,shp_name:str, year: int = 2050, figsize=(20, 30), combined_mode=True):
    """创建合并的所有情景图表"""
    print(f"\n===== CREATING COMBINED PLOT FOR ALL SCENARIOS (year={year}) =====")

    fig = plt.figure(figsize=figsize)
    plot_idx_ref = [0]  # 用列表包装以便在函数间修改

    for env, cfg in scenarios.items():
        fig,ax = plot_all_for_scenario(env,shp_name, cfg, year=year, combined_mode=combined_mode,
                              combined_fig=fig, plot_idx_ref=plot_idx_ref)
    return fig,ax



# ==== Paths & global params ====
base_dir = f"../../../output/{config.TASK_NAME}/carbon_price"
arr_path = f"{base_dir}/4_tif"
out_dir  = f"{base_dir}/5_map"
os.makedirs(out_dir, exist_ok=True)



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
    'Counterfactual_carbon_high_bio_50': dict(title_name = "Shadow carbon price under\nnet-zero targets and nature-positive targets"),
    'carbon_high_bio_50': dict(title_name = "Shadow carbon price\nunder nature-positive targets"),
    'carbon_high': dict(title_name = "Shadow carbon price\nunder net-zero targets"),
    'total_bio':dict(legend_nbins=3,unit_labelpad=5),
    'carbon_price':dict(legend_nbins=3),
}


# ==== 逐个情景执行：一个情景所有图画完，再画下一个 ====
shp_name = 'H_5kkm2'
# for env, cfg in scenarios.items():
#     plot_all_for_scenario(env,shp_name, cfg, year=2050)
# 合并模式
fig,ax = plot_all_scenarios_combined(scenarios,shp_name, year=2050, figsize=(20, 30))

font_size = ax.xaxis.get_label().get_size()
font_family = ax.xaxis.get_label().get_family()[0]

plt.rcParams['font.family'] = font_family
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = font_family   # 直立
plt.rcParams['mathtext.it'] = font_family   # 斜体
plt.rcParams['mathtext.bf'] = font_family   # 粗体
plt.rcParams['mathtext.sf'] = font_family   # 无衬线

add_north_arrow(fig, 0.20, 0.013,size=0.012)
add_scalebar(fig,ax, 0.24, 0.019, length_km=500, fontsize=font_size,fontfamily=font_family,linewidth=1)
add_annotation(fig, 0.3, 0.023, width=0.015, text="Australian state boundary",linewidth=1,
               style="line", linecolor="black",fontsize=font_size, fontfamily=font_family)
add_annotation(fig, 0.46, 0.02, width=0.0122, height=0.0072,linewidth=1, text="No data",
               style="box", facecolor="white", edgecolor="black",fontsize=font_size, fontfamily=font_family)
add_annotation(fig, 0.54, 0.02, width=0.0122, height=0.0072,linewidth=1, text="Public, indigenous, urban, and other intensive land uses",
               style="box", facecolor="#808080",edgecolor="#808080",fontsize=font_size, fontfamily=font_family)
fig.text(
    0.015, 0.80, r'Reference→$\mathrm{GHG}_{\mathrm{high}}$',
    rotation=90, va="center", ha="left", fontsize=font_size,fontfamily=font_family,
    rotation_mode="anchor"  # 以锚点为中心旋转，贴边更稳
)
fig.text(
    0.015, 0.47, r'$\mathrm{GHG}_{\mathrm{high}}$→$\mathrm{GHG}_{\mathrm{high}}$,$\mathrm{Bio}_{\mathrm{50}}$',
    rotation=90, va="center", ha="left", fontsize=font_size,fontfamily=font_family,
    rotation_mode="anchor"  # 以锚点为中心旋转，贴边更稳
)
fig.text(
    0.015, 0.16, r'Reference→$\mathrm{GHG}_{\mathrm{high}}$,$\mathrm{Bio}_{\mathrm{50}}$',
    rotation=90, va="center", ha="left", fontsize=font_size,fontfamily=font_family,
    rotation_mode="anchor"  # 以锚点为中心旋转，贴边更稳
)

# plt.subplots_adjust(bottom=0.05)
output_path = os.path.join(out_dir, f"06_all_maps")
# save_figure_properly(fig, output_path, facecolor='white')
fig.savefig(f"{output_path}.png", dpi=300)
plt.show()