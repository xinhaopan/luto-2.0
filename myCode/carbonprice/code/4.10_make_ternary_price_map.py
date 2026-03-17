"""
4.10_make_ternary_price_map.py
三变量地图（Ternary Color Map）：
  Vertex Top       → Shadow carbon price under Net Zero       (carbon_high_50)
  Vertex Bottom-L  → Shadow biodiversity price                (carbon_high_bio_50)
  Vertex Bottom-R  → Shadow carbon price under Both targets   (Counterfactual)

各变量独立按 0–95th percentile 归一化到 [0,1]
NaN 视为 0（极低价格，不贡献颜色），三者全 NaN 的像元不画（Alpha=0）

混色策略（lerp-from-white）：
  1. 计算三个归一化值的合力方向（加权平均顶点颜色）
  2. 按合力幅度（三者之和 / 95pct）从白色线性插值到方向颜色
  → 低活动区域 = 白色；高单变量区域 = 顶点颜色；混合高区域 = 混合色
"""

import matplotlib
matplotlib.use('Agg')
import os, sys
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import geopandas as gpd
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

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

tif_r = f"{arr_path}/carbon_high_50/xr_carbon_sol_price_carbon_high_50_2050.tif"
tif_g = f"{arr_path}/carbon_high_bio_50/xr_bio_sol_price_carbon_high_bio_50_2050.tif"
tif_b = f"{arr_path}/Counterfactual_carbon_high_bio_50/xr_carbon_sol_price_Counterfactual_carbon_high_bio_50_2050.tif"

# ── 顶点颜色 ──────────────────────────────────────────────────────────────────
# 三色在感知上均匀分布，三者平均 ≈ 中性灰
# Top (R):    Shadow carbon price – Net Zero   → coral-red
# Bot-L (G):  Shadow biodiversity price        → forest-green
# Bot-R (B):  Shadow carbon price – Both       → periwinkle-blue
COLOR_R = np.array([0.92, 0.18, 0.18])   # vivid red
COLOR_G = np.array([0.10, 0.78, 0.28])   # vivid green
COLOR_B = np.array([0.12, 0.32, 0.95])   # vivid blue

LABEL_R = "Shadow carbon price\nunder Net Zero"
LABEL_G = "Shadow\nbiodiversity price"
LABEL_B = "Shadow carbon price\nunder both targets"

# 低活动像元的底色：浅灰而非纯白，确保所有有效农业像元可见
BASE_COLOR = np.array([0.91, 0.91, 0.91])

# ── 工具函数 ─────────────────────────────────────────────────────────────────

def read_tif(path):
    with rasterio.open(path) as src:
        data    = src.read(1).astype(float)
        nodata  = src.nodata
        bounds  = src.bounds
        crs     = src.crs
        profile = src.profile.copy()
    if nodata is not None:
        data[data == nodata] = np.nan
    return data, bounds, crs, profile


def align_to_reference(src_path, ref_profile):
    with rasterio.open(src_path) as src:
        src_data   = src.read(1).astype(float)
        src_nodata = src.nodata
        if src_nodata is not None:
            src_data[src_data == src_nodata] = np.nan
        dst = np.full((ref_profile['height'], ref_profile['width']), np.nan)
        reproject(
            source=src_data, destination=dst,
            src_transform=src.transform, src_crs=src.crs,
            dst_transform=ref_profile['transform'], dst_crs=ref_profile['crs'],
            src_nodata=np.nan, dst_nodata=np.nan,
            resampling=Resampling.nearest,
        )
    return dst


def normalize_independent(data, lo_pct=15, hi_pct=85):
    """NaN → 0；有效值按 lo_pct–hi_pct% 线性拉伸到 [0,1]，提高色彩对比度。"""
    valid = data[~np.isnan(data)]
    if valid.size == 0:
        return np.zeros_like(data)
    vmin = np.nanpercentile(valid, lo_pct)
    vmax = np.nanpercentile(valid, hi_pct)
    if vmax == vmin:
        return np.zeros_like(data)
    out = np.clip(data, vmin, vmax)
    out = (out - vmin) / (vmax - vmin)
    out = np.nan_to_num(out, nan=0.0)
    return out


def get_data_crs(crs):
    try:
        epsg = crs.to_epsg()
        return ccrs.epsg(epsg) if epsg else ccrs.PlateCarree()
    except Exception:
        return ccrs.PlateCarree()


def build_rgba_map(norm_r, norm_g, norm_b, any_valid):
    """
    lerp-from-white 混色：
      direction = weighted average of vertex colors (proportional to norms)
      magnitude = clip(sum / 95th-pct-of-sum, 0, 1)
      color     = lerp(white, direction, magnitude)
    """
    total = norm_r + norm_g + norm_b
    valid_total = total[any_valid]
    total_95 = np.percentile(valid_total, 95) if valid_total.size > 0 else 1.0
    if total_95 == 0:
        total_95 = 1.0
    mag = np.clip(total / total_95, 0, 1)          # (H, W)

    eps = 1e-10
    w_r = norm_r / (total + eps)
    w_g = norm_g / (total + eps)
    w_b = norm_b / (total + eps)
    direction = np.clip(
        w_r[..., None] * COLOR_R +
        w_g[..., None] * COLOR_G +
        w_b[..., None] * COLOR_B, 0, 1
    )                                               # (H, W, 3)

    # lerp: BASE_COLOR at mag=0 → vertex-color-mix at mag=1
    # 使用浅灰底色确保低价格有效像元可见（不与透明背景混淆）
    color = (1 - mag[..., None]) * BASE_COLOR + mag[..., None] * direction
    color = np.clip(color, 0, 1)

    alpha = np.where(any_valid, 1.0, 0.0)
    return np.concatenate([color, alpha[..., None]], axis=-1).astype(np.float32)


def make_triangle_legend(N=350):
    """
    等边三角形图例：顶点 = 顶点颜色（满饱和），内部 = 加权平均。
    顶点: 上=COLOR_R, 左下=COLOR_G, 右下=COLOR_B
    """
    v_r = np.array([0,       N // 2])   # 上
    v_g = np.array([N - 1,   0     ])   # 左下
    v_b = np.array([N - 1,   N - 1 ])   # 右下

    rows, cols = np.mgrid[0:N, 0:N]
    T = np.array([[v_r[0]-v_b[0], v_g[0]-v_b[0]],
                  [v_r[1]-v_b[1], v_g[1]-v_b[1]]], dtype=float)
    inv_T = np.linalg.inv(T)
    pts  = np.stack([rows - v_b[0], cols - v_b[1]], axis=-1)
    lam  = np.einsum('ij,...j->...i', inv_T, pts)
    lam_r = lam[..., 0]
    lam_g = lam[..., 1]
    lam_b = 1.0 - lam_r - lam_g

    inside = (lam_r >= -1e-9) & (lam_g >= -1e-9) & (lam_b >= -1e-9)
    lam_r  = np.clip(lam_r, 0, 1)
    lam_g  = np.clip(lam_g, 0, 1)
    lam_b  = np.clip(lam_b, 0, 1)

    # 三角形内部颜色 = 加权平均顶点颜色（满饱和）
    rgb = np.clip(
        lam_r[..., None] * COLOR_R +
        lam_g[..., None] * COLOR_G +
        lam_b[..., None] * COLOR_B, 0, 1
    )
    rgba = np.zeros((N, N, 4), dtype=float)
    rgba[inside, :3] = rgb[inside]
    rgba[inside,  3] = 1.0
    return rgba, (v_r, v_g, v_b)


# ── 读取 & 对齐 ───────────────────────────────────────────────────────────────
data_r, bounds, crs, profile = read_tif(tif_r)
data_g = align_to_reference(tif_g, profile)
data_b = align_to_reference(tif_b, profile)

# ── 归一化（各自独立） ─────────────────────────────────────────────────────────
norm_r = normalize_independent(data_r)
norm_g = normalize_independent(data_g)
norm_b = normalize_independent(data_b)

print(f"R (Carbon NZ)   95pct = {np.nanpercentile(data_r[~np.isnan(data_r)], 95):.1f}")
print(f"G (Bio)         95pct = {np.nanpercentile(data_g[~np.isnan(data_g)], 95):.1f}")
print(f"B (Carbon Both) 95pct = {np.nanpercentile(data_b[~np.isnan(data_b)], 95):.1f}")

# ── 构建 RGBA 图像 ─────────────────────────────────────────────────────────────
any_valid = ~(np.isnan(data_r) & np.isnan(data_g) & np.isnan(data_b))
rgba_map  = build_rgba_map(norm_r, norm_g, norm_b, any_valid)

# ── 绘图 ──────────────────────────────────────────────────────────────────────
set_plot_style(font_size=11, font_family='Arial')

fig    = plt.figure(figsize=(11, 8), facecolor='none')
ax_map = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
ax_map.set_facecolor('none')
fig.patch.set_alpha(0)
fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)

data_crs = get_data_crs(crs)
extent   = (bounds.left, bounds.right, bounds.bottom, bounds.top)

# 矢量边界 & 地图范围
if os.path.exists(shp_path):
    gdf = gpd.read_file(shp_path).to_crs(crs)
    gdf.plot(ax=ax_map, edgecolor='black', linewidth=0.3,
             facecolor='none', zorder=5)
    minx, miny, maxx, maxy = gdf.total_bounds
    pad_x = (maxx - minx) * 0.02
    pad_y = (maxy - miny) * 0.02
    ax_map.set_extent(
        (minx - pad_x, maxx + pad_x, miny - pad_y, maxy + pad_y),
        crs=data_crs
    )

# 三变量混色栅格
ax_map.imshow(rgba_map, origin='upper', extent=extent,
              transform=data_crs, interpolation='nearest', zorder=2)

# 灰色公共土地
if os.path.exists(gray_tif):
    with rasterio.open(gray_tif) as gsrc:
        gmask   = gsrc.read(1, masked=True)
        gbounds = gsrc.bounds
    mask01 = np.zeros(gmask.shape, dtype=np.uint8)
    mask01[~gmask.mask & (gmask == 1)] = 1
    from matplotlib.colors import ListedColormap
    gray_cmap = ListedColormap([(0, 0, 0, 0), (0.5, 0.5, 0.5, 1.0)])
    gext = (gbounds.left, gbounds.right, gbounds.bottom, gbounds.top)
    ax_map.imshow(mask01, origin='upper', extent=gext, transform=data_crs,
                  cmap=gray_cmap, vmin=0, vmax=1,
                  interpolation='nearest', zorder=10)

ax_map.set_axis_off()

# ── 三角形图例 ────────────────────────────────────────────────────────────────
# 整体上移：bbox_to_anchor y 从 0.03 → 0.10
FONT_SIZE = 12   # 全图统一字号

# 图像 11×8 in；inset 宽高比设为 8/11 ≈ 0.727 使物理上接近正方形
# width=13%, height=18%  →  11*0.98*0.13 ≈ 1.401 in,  8*0.98*0.18 ≈ 1.411 in  (近似正方)
ax_leg = inset_axes(
    ax_map,
    width="13%", height="18%",
    loc='lower left',
    bbox_to_anchor=(0.31, 0.11, 1, 1),
    bbox_transform=ax_map.transAxes,
    borderpad=0,
)
ax_leg.set_facecolor('none')
ax_leg.patch.set_alpha(0)
ax_leg.set_axis_off()

SQRT3_2 = 3 ** 0.5 / 2   # ≈ 0.866，等边三角形高度（底边=1）

# xlim == ylim 的 span → 配合物理正方形 inset，triangle 自然等边
ax_leg.set_xlim(-0.40, 1.40)   # span = 1.80
ax_leg.set_ylim(-0.40, 1.40)   # span = 1.80  (等于 xlim span)

N_TRI = 400
tri_rgba, _ = make_triangle_legend(N=N_TRI)

# extent 高度 = SQRT3_2 → 物理正方坐标系中为正等边三角形
ax_leg.imshow(tri_rgba, origin='upper',
              extent=[0, 1, 0, SQRT3_2], aspect='auto',
              interpolation='bilinear', zorder=2)

# 三顶点标签
lkw = dict(fontsize=FONT_SIZE, fontfamily='Arial', fontweight='bold',
           color='black', transform=ax_leg.transData, clip_on=False)

ax_leg.text(0.50,   SQRT3_2 + 0.06, LABEL_R, ha='center', va='bottom', **lkw)
ax_leg.text(-0.03,  -0.03,          LABEL_G, ha='right',  va='top',    **lkw)
ax_leg.text( 1.03,  -0.03,          LABEL_B, ha='left',   va='top',    **lkw)

# ── 灰色图例：主地图 transAxes，三角形下方右侧、不重叠 ───────────────────────
ax_map.plot(0.15, 0.07, 's',
            markersize=8, color='#808080',
            transform=ax_map.transAxes, clip_on=False, zorder=15)
ax_map.text(0.17, 0.07,
            "Public, indigenous, urban, water bodies, and other land",
            ha='left', va='center',
            fontsize=FONT_SIZE, fontfamily='Arial', fontweight='bold',
            color='black', transform=ax_map.transAxes, clip_on=False)

# ── 保存 ──────────────────────────────────────────────────────────────────────
out_path = os.path.join(out_dir, "10_Sol_price_ternary_map.png")
fig.savefig(out_path, dpi=300, bbox_inches='tight', transparent=True)
print(f"Saved → {out_path}")
plt.show()
