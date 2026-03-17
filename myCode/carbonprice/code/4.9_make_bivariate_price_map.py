"""
4.9_make_bivariate_price_map.py
双变量地图（Bivariate Choropleth）：
  - 横轴：biodiversity price 低/中/高
  - 纵轴：carbon price      低/中/高
  → 9种组合颜色，图例为 3×3 色块矩阵

NaN 处理（方案B）：
  NaN 不是"未涉及土地"，而是 cost/benefit < 1 被过滤掉的极低价格像元。
  → 将 NaN 视为价格极低，归入 Low 档（class 0）。
  → 但只有两个 TIF 都是 NaN 的像元（真正非农业土地）才不画。
"""

import matplotlib
matplotlib.use('Agg')
import os, sys
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import gridspec
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.patheffects as pe
import cartopy.crs as ccrs
import geopandas as gpd

sys.path.insert(0, os.path.dirname(__file__))
import tools.config as config
from tools.helper_plot import set_plot_style

# ── 路径 ────────────────────────────────────────────────────────────────────
base_dir = f"../../../output/{config.TASK_NAME}/carbon_price"
arr_path = f"{base_dir}/4_tif"
out_dir  = f"{base_dir}/3_Paper_figure"
shp_path = "../Map/AUS_line1.shp"
gray_tif = "../Map/public_area_aligned.tif"
os.makedirs(out_dir, exist_ok=True)

carbon_tif = f"{arr_path}/carbon_high_50/xr_carbon_sol_price_carbon_high_50_2050.tif"
bio_tif    = f"{arr_path}/carbon_high_bio_50/xr_bio_sol_price_carbon_high_bio_50_2050.tif"

# ── 3×3 双变量颜色矩阵 ───────────────────────────────────────────────────────
# 行 = carbon price 低→高，列 = bio price 低→高
# 使用经典 Pink-Blue Stevens 双变量配色
#                  Bio Low     Bio Med     Bio High
BIVAR_COLORS = [
    ["#e8e8e8",  "#ace4e4",  "#5ac8c8"],   # Carbon Low
    ["#dfb0d6",  "#a5add3",  "#5698b9"],   # Carbon Medium
    ["#be64ac",  "#8c62aa",  "#3b4994"],   # Carbon High
]

# 展平为 9 色列表（按 combo_id = (carbon_class-1)*3 + (bio_class-1)）
FLAT_COLORS = [BIVAR_COLORS[r][c] for r in range(3) for c in range(3)]
# combo_id:  0  1  2  3  4  5  6  7  8
#            CL-BL CL-BM CL-BH  CM-BL ...  CH-BH

# ── 工具函数 ─────────────────────────────────────────────────────────────────

def read_tif(path):
    with rasterio.open(path) as src:
        data   = src.read(1).astype(float)
        nodata = src.nodata
        bounds = src.bounds
        crs    = src.crs
        profile = src.profile.copy()
    if nodata is not None:
        data[data == nodata] = np.nan
    return data, bounds, crs, profile


def align_to_reference(src_path, ref_profile):
    """将 src 重采样对齐到 ref_profile 的网格，返回 ndarray。"""
    with rasterio.open(src_path) as src:
        src_data = src.read(1).astype(float)
        src_nodata = src.nodata
        if src_nodata is not None:
            src_data[src_data == src_nodata] = np.nan
        dst = np.full((ref_profile['height'], ref_profile['width']), np.nan)
        reproject(
            source=src_data,
            destination=dst,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=ref_profile['transform'],
            dst_crs=ref_profile['crs'],
            src_nodata=np.nan,
            dst_nodata=np.nan,
            resampling=Resampling.nearest,
        )
    return dst


def classify_tercile(data):
    """
    按33/67百分位分三档。
    断点只用有效值（非NaN）计算；NaN像元本身保留 NaN，
    由调用方决定如何处理（方案B：归入 Low）。
    """
    valid = data[~np.isnan(data)]
    p33, p67 = np.nanpercentile(valid, [33, 67])
    vmin = float(np.nanmin(valid))
    vmax = float(np.nanmax(valid))
    cls = np.full(data.shape, np.nan)
    cls[~np.isnan(data) & (data <= p33)] = 0   # Low  → 0
    cls[~np.isnan(data) & (data > p33) & (data <= p67)] = 1  # Med → 1
    cls[~np.isnan(data) & (data > p67)] = 2    # High → 2
    return cls, (vmin, p33, p67, vmax)


def fmt_val(v):
    if v >= 1e6:
        return f"{v/1e6:.1f}M"
    if v >= 1000:
        return f"{v/1000:.1f}k"
    if v >= 10:
        return f"{v:,.0f}"
    return f"{v:.1f}"


def get_data_crs(crs):
    try:
        epsg = crs.to_epsg()
        return ccrs.epsg(epsg) if epsg else ccrs.PlateCarree()
    except Exception:
        return ccrs.PlateCarree()


# ── 读取 & 对齐 ───────────────────────────────────────────────────────────────
carbon_data, carbon_bounds, carbon_crs, carbon_profile = read_tif(carbon_tif)
bio_data_raw, _, _, _ = read_tif(bio_tif)

# 将 bio 对齐到 carbon 网格（防止分辨率/范围差异）
bio_data = align_to_reference(bio_tif, carbon_profile)

# ── 分类 ──────────────────────────────────────────────────────────────────────
carbon_cls, carbon_breaks = classify_tercile(carbon_data)
bio_cls,    bio_breaks    = classify_tercile(bio_data)

c_vmin, c_p33, c_p67, c_vmax = carbon_breaks
b_vmin, b_p33, b_p67, b_vmax = bio_breaks

print(f"Carbon  vmin={c_vmin:.1f}  p33={c_p33:.1f}  p67={c_p67:.1f}  vmax={c_vmax:.1f}")
print(f"Bio     vmin={b_vmin:.1f}  p33={b_p33:.1f}  p67={b_p67:.1f}  vmax={b_vmax:.1f}")

# ── 合并为单张 combo 图（方案B：NaN 归入 Low = 0）────────────────────────────
# 只有两者都是 NaN（真正的非农业土地）才不画
both_nan = np.isnan(carbon_cls) & np.isnan(bio_cls)

# NaN → 0（Low），有效值保留
carbon_cls_filled = np.where(np.isnan(carbon_cls), 0, carbon_cls)
bio_cls_filled    = np.where(np.isnan(bio_cls),    0, bio_cls)

# 至少一个有效像元才画
any_valid = ~np.isnan(carbon_data) | ~np.isnan(bio_data)

combo = np.full(carbon_data.shape, np.nan)
combo[any_valid] = carbon_cls_filled[any_valid] * 3 + bio_cls_filled[any_valid]

# ── 绘图 ──────────────────────────────────────────────────────────────────────
set_plot_style(font_size=12, font_family='Arial')

fig = plt.figure(figsize=(11, 8))
ax_map = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)

# ── 地图 ──────────────────────────────────────────────────────────────────────
data_crs = get_data_crs(carbon_crs)
extent   = (carbon_bounds.left, carbon_bounds.right,
            carbon_bounds.bottom, carbon_bounds.top)

cmap9 = ListedColormap(FLAT_COLORS)
norm9 = BoundaryNorm(np.arange(-0.5, 9.5, 1), ncolors=9)

if os.path.exists(shp_path):
    gdf = gpd.read_file(shp_path).to_crs(carbon_crs)
    gdf.plot(ax=ax_map, edgecolor='black', linewidth=0.3,
             facecolor='none', zorder=5)
    minx, miny, maxx, maxy = gdf.total_bounds
    pad_x = (maxx - minx) * 0.02
    pad_y = (maxy - miny) * 0.02
    ax_map.set_extent(
        (minx - pad_x, maxx + pad_x, miny - pad_y, maxy + pad_y),
        crs=data_crs
    )

ax_map.imshow(combo, origin='upper', extent=extent, transform=data_crs,
              cmap=cmap9, norm=norm9, interpolation='nearest', zorder=2)

# 灰色公共土地
if os.path.exists(gray_tif):
    with rasterio.open(gray_tif) as gsrc:
        gmask   = gsrc.read(1, masked=True)
        gbounds = gsrc.bounds
    mask01 = np.zeros(gmask.shape, dtype=np.uint8)
    mask01[~gmask.mask & (gmask == 1)] = 1
    gray_cmap = ListedColormap([(0, 0, 0, 0), (0.5, 0.5, 0.5, 1.0)])
    gext = (gbounds.left, gbounds.right, gbounds.bottom, gbounds.top)
    ax_map.imshow(mask01, origin='upper', extent=gext, transform=data_crs,
                  cmap=gray_cmap, vmin=0, vmax=1,
                  interpolation='nearest', zorder=10)

ax_map.set_axis_off()

# ── 3×3 图例：嵌入红框位置（南澳大利亚湾下方空白区域）──────────────────────
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

ax_leg = inset_axes(
    ax_map,
    width="28%", height="28%",
    loc='lower left',
    bbox_to_anchor=(0.13, 0.03, 1, 1),
    bbox_transform=ax_map.transAxes,
    borderpad=0,
)
ax_leg.set_aspect('equal')
ax_leg.set_axis_off()
ax_leg.patch.set_alpha(0)               # 背景透明

CELL = 1.0
GAP  = 0.025                            # 色块间距极小
SIDE = 3 * CELL + 2 * GAP              # 色块矩阵总边长

# 坐标范围：左侧留行标题空间，上方留列标题空间，下方留灰色图例
ax_leg.set_xlim(-0.72, SIDE + 0.05)
ax_leg.set_ylim(-0.52, SIDE + 0.70)

# 9 个色块：碳价从上(Low)到下(High)，bio从左(Low)到右(High)
CLASS_LABELS = ["Low", "Med", "High"]
for r in range(3):
    for c in range(3):
        y_pos = (2 - r) * (CELL + GAP)   # r=0(Low)→顶, r=2(High)→底
        rect = mpatches.FancyBboxPatch(
            (c * (CELL + GAP), y_pos),
            CELL, CELL,
            boxstyle="square,pad=0",
            facecolor=BIVAR_COLORS[r][c],
            edgecolor='white', linewidth=1.0,
            transform=ax_leg.transData, zorder=3,
        )
        ax_leg.add_patch(rect)

# 列标签（Bio price，上方）：Low / Med / High
for c, lbl in enumerate(CLASS_LABELS):
    cx = c * (CELL + GAP) + CELL / 2
    ax_leg.text(cx, SIDE + 0.06, lbl,
                ha='center', va='bottom', fontsize=7.5,
                fontweight='bold', fontfamily='Arial',
                transform=ax_leg.transData)

# 行标签（Carbon price，左侧）：Low 在上，High 在下
for r, lbl in enumerate(CLASS_LABELS):
    cy = (2 - r) * (CELL + GAP) + CELL / 2
    ax_leg.text(-0.08, cy, lbl,
                ha='right', va='center', fontsize=7.5,
                fontweight='bold', fontfamily='Arial',
                transform=ax_leg.transData)

# 轴标题（无单位）
ax_leg.text(
    SIDE / 2, SIDE + 0.38,
    "Biodiversity price →",
    ha='center', va='bottom', fontsize=7.5, fontfamily='Arial',
    fontstyle='italic', transform=ax_leg.transData,
)
ax_leg.text(
    -0.60, SIDE / 2,
    "Carbon price →",
    ha='center', va='center', fontsize=7.5, fontfamily='Arial',
    fontstyle='italic', rotation=90, transform=ax_leg.transData,
)

# 灰色图例：放在色块矩阵正下方
ax_leg.add_patch(mpatches.FancyBboxPatch(
    (0, -0.44), CELL * 0.55, CELL * 0.22,
    boxstyle="square,pad=0",
    facecolor='#808080', edgecolor='none',
    transform=ax_leg.transData, zorder=3,
))
ax_leg.text(
    CELL * 0.55 + 0.06, -0.33,
    "Public, indigenous, urban,\nwater bodies, and other land",
    ha='left', va='center', fontsize=7, fontfamily='Arial',
    transform=ax_leg.transData,
)

ax_map.set_axis_off()   # 确保地图轴不显示任何legend残余

# ── 保存 ──────────────────────────────────────────────────────────────────────
out_path = os.path.join(out_dir, "09_Sol_price_bivariate_map.png")
fig.savefig(out_path, dpi=300, bbox_inches='tight')
print(f"Saved → {out_path}")
plt.show()
