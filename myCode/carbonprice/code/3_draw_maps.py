import tools.config as config
import rasterio
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import matplotlib as mpl
import h3pandas
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.ticker as mticker

def set_plot_style(font_size=12, font_family='Arial'):
    """
    设置 matplotlib 的统一字体风格和大小（适用于 ax 和 fig.text）

    参数:
    - font_size: int，字体大小
    - font_family: str，字体名称，如 'Arial', 'DejaVu Sans', 'Times New Roman'
    """
    mpl.rcParams.update({
        'font.size': font_size,
        'font.family': font_family,
        'axes.titlesize': font_size,
        'axes.labelsize': font_size,
        'xtick.labelsize': font_size,
        'ytick.labelsize': font_size,
        'legend.fontsize': font_size,
        'figure.titlesize': font_size
    })

# ====== 使用方法 ======
set_plot_style(font_size=9, font_family='Arial')

def create_hex_shape(raster_file, operation="sum",use_min=True):
    # 1) 用 masked=True 读入波段，正确识别 NoData
    raster_path = f"{config.TASK_DIR}/carbon_price/data/{raster_file}_2050.tif"
    with rasterio.open(raster_path) as src:
        band       = src.read(1, masked=True)   # numpy.ma.MaskedArray
        arr        = band.data                  # 原始数据（含 fill）
        valid_mask = ~band.mask                 # True 表示有效像元
        transform  = src.transform
        crs        = src.crs

    # 2) 提取所有有效像元的中心点与数值
    rows, cols = np.where(valid_mask)
    xs, ys     = rasterio.transform.xy(transform, rows, cols, offset="center")
    values     = arr[rows, cols] / 1e6 if use_min else arr[rows, cols]  # 转换为百万单位

    df = pd.DataFrame({
        "value":    values,
        "geometry": [Point(x, y) for x, y in zip(xs, ys)]
    })
    gdf_pts = gpd.GeoDataFrame(df, crs=crs)
    gdf_pts_wgs = gdf_pts.to_crs("EPSG:4326")
    # 1) 用 geo_to_h3_aggregate 一步完成聚合并生成几何
    hex_gdf = gdf_pts_wgs.h3.geo_to_h3_aggregate(
        resolution=3,             # H3 分辨率，12,426.85 km²
        operation=dict(value=operation),  # 对 'value' 列做均值
        return_geometry=True      # 生成六边形 polygon
    )  # :contentReference[oaicite:0]{index=0}
    # 2) 把 H3 ID（索引）变成普通列，并重命名
    hex_gdf = hex_gdf.reset_index().rename(
        columns={f"h3_{6:02d}": "h3_id", "value": "val"}
    )
    return hex_gdf


def draw_hex_shape(ax, hex_gdf, title_name, unit_name, cmap='viridis'):
    # 预先准备好 ScalarMappable，共用同一套色带
    norm = mpl.colors.Normalize(vmin=hex_gdf.val.min(), vmax=hex_gdf.val.max())
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])

    # 1) 绘制六边形
    hex_gdf.plot(column='val', cmap=cmap,
             edgecolor='white', linewidth=0.3,
             legend=False, ax=ax)
    ax.set_title(title_name, y=0.95)  # 设置标题并调整位置
    ax.set_axis_off()

    # 2) 在当前 ax 里插入一个水平 colorbar
    cax = inset_axes(
        ax,
        width="55%",  # 色条长度为 ax 宽度
        height="8%",  # 色条高度为 ax 高度
        loc='lower center',  # 放在 ax 的正下方中间
        borderpad=1,
        bbox_to_anchor=(-0.12, 0.07, 1, 0.8),  # (left, bottom, width, height)
        # 都是 ax.transAxes 坐标系下的 0–1
        bbox_transform=ax.transAxes,
    )
    cbar = fig.colorbar(
        sm, cax=cax, orientation='horizontal',
        extend='both',  # 在色带两端加箭头
        extendfrac=0.1,  # 比默认更小，箭头更尖
        extendrect=False  # 保留三角箭头（而不是平头）
    )
    # 隐藏 colorbar 整体外框
    cbar.outline.set_visible(False)
    # 将刻度线移到上方
    # cbar.ax.xaxis.set_ticks_position('top')
    # 将标签也移到上方
    cbar.ax.xaxis.set_label_position('top')
    cbar.locator = mticker.MaxNLocator(nbins=5)
    cbar.update_ticks()
    # cbar.set_ticks([])     # 如果不想显示刻度
    cbar.set_label(unit_name)

    plt.subplots_adjust(wspace=-0.05, hspace=-0.05)  # 子图左右间距

def calculate_shadow_price(hex_gdf0, hex_gdf1):
    """
    计算影子价格（shadow price），即两个列的比值。

    参数:
    - hex_gdf: GeoDataFrame，包含需要计算的列
    - base_col: str，基准列名
    - target_col: str，目标列名

    返回:
    - GeoDataFrame，新增一列 'shadow_price'
    """
    num = hex_gdf0['val']
    den = hex_gdf1['val']
    ratio = num.div(den)  # 等同于 num/den

    # 2) 把 inf/-inf 替换成 NaN
    ratio_clean = ratio.replace([np.inf, -np.inf], np.nan)

    # （可选）3) 再把 NaN 填成 0 或者某个统计量
    ratio_filled = ratio_clean.fillna(0)

    # 4) 赋回你的 GeoDataFrame
    hex_gdf3 = hex_gdf0.copy()
    hex_gdf3['val'] = ratio_clean  # 或 ratio_filled
    return hex_gdf3

set_plot_style(font_size=9, font_family='Arial')
fig, axes = plt.subplots(3, 2, figsize=(8, 12))
axes = axes.flatten()

raster_files = ["carbon_cost", "bio_cost","ghg","bio"]
title_name = [
    "Carbon sequestration cost",
    "Biodiversity restoration cost",
    "Carbon sequestration",
    "Biodiversity restoration",
]
unit_name = ["Million AU$", "Million AU$", "MtCO2e", "Mha"]
camps = ["autumn_r","autumn_r","summer_r","summer_r"]

hex_gdfs = []
for ax, raster_file, title_name, unit_name,camp in zip(axes, raster_files, title_name, unit_name,camps):
    hex_gdf = create_hex_shape(raster_file)
    draw_hex_shape(ax, hex_gdf, title_name, unit_name, camp)
    hex_gdfs.append(hex_gdf)

hex_gdf_cp = calculate_shadow_price(hex_gdfs[0], hex_gdfs[2])
draw_hex_shape(axes[4], hex_gdf_cp,"Shadow carbon price","AU$ tCO2e-1", "winter_r")
hex_gdf_bp = calculate_shadow_price(hex_gdfs[1], hex_gdfs[3])
draw_hex_shape(axes[5], hex_gdf_bp,"Shadow biodiversity price","AU$ ha-1", "winter_r")
plt.savefig(f"{config.TASK_DIR}/carbon_price/Paper_figure/3_maps.png", dpi=300, bbox_inches='tight')
plt.show()