"""
4.12_make_price_panel.py
上下两个 panel：
  (上) 双 y 轴 GAM 拟合线 + 95% CI，2025–2050
       数据来自 xr_carbon_price.nc / xr_bio_price.nc（与 3.0 相同）
  (下) 三变量 Ternary Color Map（与 4.10 相同）
上面 panel 比下面略小；无 (a)(b) 标签；字体 Arial，FONT_SIZE=12
"""

import matplotlib
matplotlib.use('Agg')
import os, sys
import numpy as np
import xarray as xr
import rasterio
from rasterio.warp import reproject, Resampling
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.lines as mlines
from matplotlib.patches import Patch
import cartopy.crs as ccrs
import geopandas as gpd
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.colors import ListedColormap

sys.path.insert(0, os.path.dirname(__file__))
import tools.config as config
from tools.helper_plot import set_plot_style, draw_fit_line_ax

# ─────────────────────────── 路径 ────────────────────────────────────────────
base_dir = f"../../../output/{config.TASK_NAME}/carbon_price"
data_dir = f"{base_dir}/1_draw_data"
arr_path = f"{base_dir}/4_tif"
out_dir  = f"{base_dir}/3_Paper_figure"
shp_path = "../Map/AUS_line1.shp"
gray_tif = "../Map/public_area_aligned.tif"
os.makedirs(out_dir, exist_ok=True)

tif_r = f"{arr_path}/carbon_high_50/xr_carbon_sol_price_carbon_high_50_2050.tif"
tif_g = f"{arr_path}/carbon_high_bio_50/xr_bio_sol_price_carbon_high_bio_50_2050.tif"
tif_b = f"{arr_path}/Counterfactual_carbon_high_bio_50/xr_carbon_sol_price_Counterfactual_carbon_high_bio_50_2050.tif"

# ─────────────────────────── 共享配置 ────────────────────────────────────────
FONT_SIZE  = 12
COLOR_R    = np.array([0.92, 0.18, 0.18])
COLOR_G    = np.array([0.10, 0.78, 0.28])
COLOR_B    = np.array([0.12, 0.32, 0.95])
BASE_COLOR = np.array([0.91, 0.91, 0.91])
SQRT3_2    = 3 ** 0.5 / 2

LABEL_R = "Shadow carbon price\nunder Net Zero"
LABEL_G = "Shadow\nbiodiversity price"
LABEL_B = "Shadow carbon price\nunder both targets"

# ─────────────────────────── 地图工具函数 ────────────────────────────────────
def read_tif(path):
    with rasterio.open(path) as src:
        data = src.read(1).astype(float)
        if src.nodata is not None:
            data[data == src.nodata] = np.nan
        return data, src.bounds, src.crs, src.profile.copy()

def align_to_reference(src_path, ref_profile):
    with rasterio.open(src_path) as src:
        d = src.read(1).astype(float)
        if src.nodata is not None:
            d[d == src.nodata] = np.nan
        dst = np.full((ref_profile['height'], ref_profile['width']), np.nan)
        reproject(source=d, destination=dst,
                  src_transform=src.transform, src_crs=src.crs,
                  dst_transform=ref_profile['transform'], dst_crs=ref_profile['crs'],
                  src_nodata=np.nan, dst_nodata=np.nan,
                  resampling=Resampling.nearest)
    return dst

def normalize_independent(data, lo=15, hi=85):
    valid = data[~np.isnan(data)]
    if valid.size == 0:
        return np.zeros_like(data)
    vmin, vmax = np.nanpercentile(valid, lo), np.nanpercentile(valid, hi)
    if vmax == vmin:
        return np.zeros_like(data)
    out = np.clip(data, vmin, vmax)
    return np.nan_to_num((out - vmin) / (vmax - vmin), nan=0.0)

def build_rgba_map(nr, ng, nb, any_valid):
    total = nr + ng + nb
    mag   = np.clip(total / (np.percentile(total[any_valid], 95) or 1), 0, 1)
    eps   = 1e-10
    w_r, w_g, w_b = nr/(total+eps), ng/(total+eps), nb/(total+eps)
    direction = np.clip(w_r[...,None]*COLOR_R + w_g[...,None]*COLOR_G
                        + w_b[...,None]*COLOR_B, 0, 1)
    color = (1-mag[...,None])*BASE_COLOR + mag[...,None]*direction
    alpha = np.where(any_valid, 1.0, 0.0)
    return np.concatenate([np.clip(color,0,1), alpha[...,None]], axis=-1).astype(np.float32)

def make_triangle_legend(N=400):
    v_r = np.array([0, N//2]); v_g = np.array([N-1,0]); v_b = np.array([N-1,N-1])
    rows, cols = np.mgrid[0:N, 0:N]
    T = np.array([[v_r[0]-v_b[0], v_g[0]-v_b[0]],[v_r[1]-v_b[1], v_g[1]-v_b[1]]], dtype=float)
    lam = np.einsum('ij,...j->...i', np.linalg.inv(T),
                    np.stack([rows-v_b[0], cols-v_b[1]], axis=-1))
    lr, lg = lam[...,0], lam[...,1]; lb = 1-lr-lg
    inside = (lr>=-1e-9)&(lg>=-1e-9)&(lb>=-1e-9)
    rgb = np.clip(np.clip(lr,0,1)[...,None]*COLOR_R
                 +np.clip(lg,0,1)[...,None]*COLOR_G
                 +np.clip(lb,0,1)[...,None]*COLOR_B, 0, 1)
    rgba = np.zeros((N,N,4)); rgba[inside,:3]=rgb[inside]; rgba[inside,3]=1
    return rgba

def get_data_crs(crs):
    try:
        epsg = crs.to_epsg()
        return ccrs.epsg(epsg) if epsg else ccrs.PlateCarree()
    except Exception:
        return ccrs.PlateCarree()

# ─────────────────────────── 准备地图数据 ────────────────────────────────────
data_r, bounds, crs, profile = read_tif(tif_r)
data_g = align_to_reference(tif_g, profile)
data_b = align_to_reference(tif_b, profile)
norm_r, norm_g, norm_b = (normalize_independent(d) for d in (data_r, data_g, data_b))
any_valid = ~(np.isnan(data_r) & np.isnan(data_g) & np.isnan(data_b))
rgba_map  = build_rgba_map(norm_r, norm_g, norm_b, any_valid)

# ─────────────────────────── 准备线图数据（同 3.0）──────────────────────────
carbon_da = xr.open_dataarray(os.path.join(data_dir, 'xr_carbon_price.nc'))
bio_da    = xr.open_dataarray(os.path.join(data_dir, 'xr_bio_price.nc'))
df_carbon = carbon_da.to_dataframe(name='data').reset_index()
df_bio    = bio_da.to_dataframe(name='data').reset_index()

years = list(range(2025, 2051))
df_c_nz   = (df_carbon[(df_carbon['scenario'] == 'carbon_high_50')                  & df_carbon['Year'].isin(years)]
             .set_index('Year')[['data']])
df_c_both = (df_carbon[(df_carbon['scenario'] == 'Counterfactual_carbon_high_bio_50') & df_carbon['Year'].isin(years)]
             .set_index('Year')[['data']])
df_b_bio  = (df_bio[(df_bio['scenario']   == 'carbon_high_bio_50')     & df_bio['Year'].isin(years)]
             .set_index('Year')[['data']])

# ─────────────────────────── 建立画布 ────────────────────────────────────────
set_plot_style(font_size=FONT_SIZE, font_family='Arial')

fig = plt.figure(figsize=(11, 15), facecolor='none')
fig.patch.set_alpha(0)
# GridSpec 宽度用原始值（下面地图用），上面折线图单独用 add_axes 指定更窄的位置
gs  = gridspec.GridSpec(2, 1, figure=fig,
                         left=0.10, right=0.92, top=0.88, bottom=0.03,
                         height_ratios=[0.60, 1.0],
                         hspace=0.02)

# ══════════════════════════ Panel (上)：GAM 折线图 ════════════════════════════
# 从 GridSpec 取纵向位置，但横向缩窄（不影响下面地图）
_top = gs[0].get_position(fig)
_pad = 0.11
ax_line  = fig.add_axes([_top.x0 + _pad, _top.y0, _top.width - 2*_pad, _top.height])
ax_line.set_facecolor('none')
ax_right = ax_line.twinx()
ax_right.set_facecolor('none')

# GAM 平滑线 + 95% CI
draw_fit_line_ax(ax_line,  df_c_nz,   color=tuple(COLOR_R), ci=95)
draw_fit_line_ax(ax_line,  df_c_both, color=tuple(COLOR_B), ci=95)
draw_fit_line_ax(ax_right, df_b_bio,  color=tuple(COLOR_G), ci=95)
ax_line.set_title('')  # 清除 draw_fit_line_ax 可能设置的标题

# 坐标轴标签
ax_line.set_xlabel('')
ax_line.set_ylabel(r'Shadow carbon price (AU\$ tCO$_2$e$^{-1}$ yr$^{-1}$)',
                   fontsize=FONT_SIZE, fontfamily='Arial', fontweight='normal')
ax_right.set_ylabel(r'Shadow biodiversity price (AU\$ contribution-weighted area ha$^{-1}$ yr$^{-1}$)',
                    fontsize=FONT_SIZE, fontfamily='Arial', fontweight='normal',
                    color='black', rotation=270, labelpad=18)

# 右轴样式（全黑，覆盖 draw_fit_line_ax 对 spines 的设置）
ax_right.tick_params(axis='y', labelsize=FONT_SIZE, colors='black')
ax_right.spines['right'].set_color('black')
ax_right.yaxis.label.set_color('black')
ax_right.set_yticks(np.arange(0, 1201, 300))
ax_right.set_ylim(0, 1200)

ax_line.tick_params(axis='both', labelsize=FONT_SIZE)
ax_line.set_yticks(np.arange(0, 201, 50))
ax_line.set_ylim(0, 200)
ax_line.set_xticks(np.arange(2025, 2051, 5))
ax_line.set_xlim(2024.5, 2050.5)
ax_line.grid(axis='y', linestyle='--', lw=0.5, alpha=0.4, color='gray')
ax_line.set_axisbelow(True)

# 确保所有刻度标签字体 Arial 黑色
for label in (ax_line.get_xticklabels() + ax_line.get_yticklabels()
              + ax_right.get_yticklabels()):
    label.set_fontfamily('Arial'); label.set_fontsize(FONT_SIZE)
    label.set_color('black')

# 确保轴标签字体 Arial 黑色
for ax_obj in [ax_line, ax_right]:
    ax_obj.yaxis.label.set_fontfamily('Arial')
    ax_obj.yaxis.label.set_fontsize(FONT_SIZE)
    ax_obj.yaxis.label.set_color('black')

# ── 图例：两列，第一列全为线，第二列全为 95% CI 色块 ──────────────────────────
# 用两个独立 legend 对象确保绝对的列对齐，不依赖 ncol 的填充顺序
h_nz   = mlines.Line2D([], [], color=tuple(COLOR_R), lw=2, marker='o', ms=4,
                        label='Shadow carbon price under Net Zero')
h_both = mlines.Line2D([], [], color=tuple(COLOR_B), lw=2, marker='o', ms=4,
                        label='Shadow carbon price under both targets')
h_bio  = mlines.Line2D([], [], color=tuple(COLOR_G), lw=2, marker='o', ms=4,
                        label='Shadow biodiversity price')
p_nz   = Patch(facecolor=(*tuple(COLOR_R), 0.40), edgecolor='none',
               label='95% Confidence Interval')
p_both = Patch(facecolor=(*tuple(COLOR_B), 0.40), edgecolor='none',
               label='95% Confidence Interval')
p_bio  = Patch(facecolor=(*tuple(COLOR_G), 0.40), edgecolor='none',
               label='95% Confidence Interval')

_lkw = dict(fontsize=FONT_SIZE, prop={'family': 'Arial', 'size': FONT_SIZE},
            frameon=False, handlelength=2.0, handleheight=0.8,
            borderpad=0, labelspacing=0.5)

# 第一列：三条线
leg1 = ax_line.legend(
    handles=[h_nz, h_both, h_bio],
    loc='upper left', bbox_to_anchor=(0.01, 0.55), **_lkw)
ax_line.add_artist(leg1)

# 第二列：三个 CI 色块（x 偏移量与第一列宽度对齐）
leg2 = ax_line.legend(
    handles=[p_nz, p_both, p_bio],
    loc='upper left', bbox_to_anchor=(0.55, 0.55), **_lkw)

for leg in [leg1, leg2]:
    for t in leg.get_texts():
        t.set_fontfamily('Arial'); t.set_fontsize(FONT_SIZE); t.set_color('black')

# ══════════════════════════ Panel (下)：Ternary 地图 ══════════════════════════
data_crs = get_data_crs(crs)
extent   = (bounds.left, bounds.right, bounds.bottom, bounds.top)

ax_map = fig.add_subplot(gs[1], projection=ccrs.PlateCarree())
ax_map.set_facecolor('none')

if os.path.exists(shp_path):
    gdf = gpd.read_file(shp_path).to_crs(crs)
    gdf.plot(ax=ax_map, edgecolor='black', lw=0.3, facecolor='none', zorder=5)
    minx, miny, maxx, maxy = gdf.total_bounds
    px, py = (maxx-minx)*0.02, (maxy-miny)*0.02
    ax_map.set_extent((minx-px, maxx+px, miny-py, maxy+py), crs=data_crs)

ax_map.imshow(rgba_map, origin='upper', extent=extent,
              transform=data_crs, interpolation='nearest', zorder=2)

if os.path.exists(gray_tif):
    with rasterio.open(gray_tif) as gsrc:
        gmask = gsrc.read(1, masked=True); gbounds = gsrc.bounds
    mask01 = np.zeros(gmask.shape, dtype=np.uint8)
    mask01[~gmask.mask & (gmask==1)] = 1
    gray_cmap = ListedColormap([(0,0,0,0),(0.5,0.5,0.5,1.0)])
    ax_map.imshow(mask01, origin='upper',
                  extent=(gbounds.left,gbounds.right,gbounds.bottom,gbounds.top),
                  transform=data_crs, cmap=gray_cmap, vmin=0, vmax=1,
                  interpolation='nearest', zorder=10)

ax_map.set_axis_off()

# 三角形图例 inset
ax_leg = inset_axes(ax_map, width="13%", height="18%", loc='lower left',
                    bbox_to_anchor=(0.31, 0.11, 1, 1),
                    bbox_transform=ax_map.transAxes, borderpad=0)
ax_leg.set_facecolor('none'); ax_leg.patch.set_alpha(0); ax_leg.set_axis_off()
ax_leg.set_xlim(-0.40, 1.40); ax_leg.set_ylim(-0.40, 1.40)

tri_rgba = make_triangle_legend()
ax_leg.imshow(tri_rgba, origin='upper', extent=[0,1,0,SQRT3_2],
              aspect='auto', interpolation='bilinear', zorder=2)

lkw = dict(fontsize=FONT_SIZE, fontfamily='Arial', fontweight='normal',
           color='black', transform=ax_leg.transData, clip_on=False)
ax_leg.text(0.50, SQRT3_2+0.06, LABEL_R, ha='center', va='bottom', **lkw)
ax_leg.text(-0.03, -0.03,       LABEL_G, ha='right',  va='top',    **lkw)
ax_leg.text( 1.03, -0.03,       LABEL_B, ha='left',   va='top',    **lkw)

# 灰色图例
ax_map.plot(0.15, 0.07, 's', markersize=8, color='#808080',
            transform=ax_map.transAxes, clip_on=False, zorder=15)
ax_map.text(0.17, 0.07, "Public, indigenous, urban, water bodies, and other land",
            ha='left', va='center',
            fontsize=FONT_SIZE, fontfamily='Arial', fontweight='normal',
            color='black', transform=ax_map.transAxes, clip_on=False)

# ─────────────────────────── 保存 ────────────────────────────────────────────
out_path = os.path.join(out_dir, "12_Sol_price_panel.png")
fig.savefig(out_path, dpi=300, bbox_inches='tight', transparent=True)
print(f"Saved → {out_path}")

out_svg = os.path.join(out_dir, "12_Sol_price_panel.svg")
fig.savefig(out_svg, format='svg', bbox_inches='tight', transparent=True)
print(f"Saved → {out_svg}")
plt.show()
