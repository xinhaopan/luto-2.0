import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib import gridspec
from matplotlib.colors import LinearSegmentedColormap, BoundaryNorm
import cartopy.crs as ccrs
import numpy as np
import rasterio
import os

import tools.config as config
from tools.helper_map import (safe_plot, add_scalebar, add_north_arrow, add_annotation,
                               align_raster_to_reference, get_y_axis_ticks)
from tools.helper_plot import set_plot_style


def plot_tif_grid(tif_title_list, figsize=(10, 11)):
    """
    画2x2网格图
    tif_title_list: [(tif, title, unit, clip_percent, cmap, vmin_override, vmax_override), ...]
    """
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=-0.2, wspace=0.02,
                           left=0.01, right=0.99, top=0.99, bottom=0.12)
    axes = []

    for idx, (tif, title, unit_name, clip_percent, cmap, vmin_ov, vmax_ov) in enumerate(tif_title_list):
        row = idx // 2
        col = idx % 2
        ax = fig.add_subplot(gs[row, col], projection=ccrs.PlateCarree())

        safe_plot(
            tif_path=tif,
            title=title,
            ax=ax,
            unit=unit_name,
            cmap=cmap,
            clip_percent=clip_percent,
            title_y=0.95,
            create_colorbar=False,
            vmin_override=vmin_ov,
            vmax_override=vmax_ov,
        )
        axes.append(ax)
    return fig, axes


# ==== Paths & global params ====
base_dir = f"../../../output/{config.TASK_NAME}/{config.CARBON_PRICE_DIR}"
arr_path = f"{base_dir}/4_tif"
out_dir = f"{base_dir}/3_Paper_figure"
os.makedirs(out_dir, exist_ok=True)

# 统一样式
set_plot_style(font_size=15, font_family='Arial')

# 色带：图1-3（碳价格，绿蓝色系），图4（生物多样性价格，粉紫色系）
price_cmap = LinearSegmentedColormap.from_list("price", ["#ffff80", "#38e009", "#1a93ab", "#0c1078"])
bio_cmap   = LinearSegmentedColormap.from_list("bio",   ["#FFF1F3", "#FF69B4", "#E91E63", "#880E4F"])

# ==== tif 文件路径 ====
tif_files_price = [
    f"{arr_path}/carbon_high_50/xr_carbon_sol_price_carbon_high_50_2050.tif",
    f"{arr_path}/carbon_high_bio_50/xr_carbon_sol_price_carbon_high_bio_50_2050.tif",
    f"{arr_path}/Counterfactual_carbon_high_bio_50/xr_carbon_sol_price_Counterfactual_carbon_high_bio_50_2050.tif",
]
tif_file_bio = f"{arr_path}/carbon_high_bio_50/xr_bio_sol_price_carbon_high_bio_50_2050.tif"

clip_pct = [0, 95]


# ==== 预扫描全局 vmax ====
def scan_vmax(tif_path, clip_percent):
    if not os.path.exists(tif_path):
        return None
    with rasterio.open(tif_path) as src:
        arr = src.read(1).astype(float)
        nd = src.nodata
    if nd is not None:
        arr = np.where(arr == nd, np.nan, arr)
    valid = arr[~np.isnan(arr)]
    return float(np.nanpercentile(valid, clip_percent[1])) if len(valid) > 0 else None


price_vmaxes = [v for v in [scan_vmax(t, clip_pct) for t in tif_files_price] if v is not None]
price_global_vmax = max(price_vmaxes) if price_vmaxes else 2400.0
bio_vmax = scan_vmax(tif_file_bio, clip_pct) or 6000.0

# 计算刻度（9 刻度 = 8 色段）
_, price_vmax_shared, price_ticks = get_y_axis_ticks(0, price_global_vmax, desired_ticks=9,  strict_count=True)
_, bio_vmax_shared,   bio_ticks   = get_y_axis_ticks(0, bio_vmax,          desired_ticks=6, strict_count=True)
price_vmin_shared = bio_vmin_shared = 0.0
price_ticks = [float(t) for t in price_ticks]
bio_ticks   = [float(t) for t in bio_ticks]

print(f"[Price colorbar] ticks={price_ticks}")
print(f"[Bio colorbar]   ticks={bio_ticks}")

# ==== tif_title_list（含 cmap 和 vmin/vmax）====
tif_title_list = [
    (tif_files_price[0], "Shadow carbon price\nunder Net Zero targets",
     r"AU\$ tCO$_2$e$^{-1}$ yr$^{-1}$", clip_pct, price_cmap, price_vmin_shared, price_vmax_shared),
    (tif_files_price[1], "Shadow carbon price\nunder Nature Positive",
     r"AU\$ tCO$_2$e$^{-1}$ yr$^{-1}$", clip_pct, price_cmap, price_vmin_shared, price_vmax_shared),
    (tif_files_price[2], "Shadow carbon price\nunder both targets",
     r"AU\$ tCO$_2$e$^{-1}$ yr$^{-1}$", clip_pct, price_cmap, price_vmin_shared, price_vmax_shared),
    (tif_file_bio, "Shadow biodiversity price",
     r"AU\$ ha$^{-1}$ yr$^{-1}$", clip_pct, bio_cmap, bio_vmin_shared, bio_vmax_shared),
]

# ==== 创建网格图 ====
fig, axes = plot_tif_grid(tif_title_list)

# 获取字体设置
font_size = axes[0].xaxis.get_label().get_size()
font_family = axes[0].xaxis.get_label().get_family()[0]
plt.rcParams['font.family'] = font_family
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = font_family
plt.rcParams['mathtext.it'] = font_family
plt.rcParams['mathtext.bf'] = font_family
plt.rcParams['mathtext.sf'] = font_family

# ==== 共享分段色标（图1-3，左列，第3张图下方）====
price_norm = BoundaryNorm(np.array(price_ticks), price_cmap.N)
price_sm = plt.cm.ScalarMappable(cmap=price_cmap, norm=price_norm)
price_sm.set_array([])

price_cbar_ax = fig.add_axes([0.058, 0.2, 0.284, 0.015])  # 左列居中，80% 宽度
price_cbar = fig.colorbar(price_sm, cax=price_cbar_ax, orientation='horizontal', extend='both')
price_cbar.set_ticks(price_ticks)
price_cbar.set_ticklabels([f"{int(v):,}" if v != 0 else "0" for v in price_ticks])
price_cbar.set_label(r"AU\$ tCO$_2$e$^{-1}$ yr$^{-1}$",
                     fontsize=font_size, fontfamily=font_family, labelpad=5)
price_cbar.ax.xaxis.set_label_position('top')
price_cbar.ax.tick_params(labelsize=font_size - 1, length=3, pad=2)
price_cbar.outline.set_visible(False)
plt.setp(price_cbar.ax.get_xticklabels(), rotation=45, ha='right')

# ==== 单独分段色标（图4，右列）====
bio_norm = BoundaryNorm(np.array(bio_ticks), bio_cmap.N)
bio_sm = plt.cm.ScalarMappable(cmap=bio_cmap, norm=bio_norm)
bio_sm.set_array([])

bio_cbar_ax = fig.add_axes([0.558, 0.2, 0.284, 0.015])   # 右列居中，80% 宽度
bio_cbar = fig.colorbar(bio_sm, cax=bio_cbar_ax, orientation='horizontal', extend='both')
bio_cbar.set_ticks(bio_ticks)
bio_cbar.set_ticklabels([f"{int(v):,}" if v != 0 else "0" for v in bio_ticks])
bio_cbar.set_label(r"AU\$ ha$^{-1}$ yr$^{-1}$",
                   fontsize=font_size, fontfamily=font_family, labelpad=5)
bio_cbar.ax.xaxis.set_label_position('top')
bio_cbar.ax.tick_params(labelsize=font_size - 1, length=3, pad=2)
bio_cbar.outline.set_visible(False)
plt.setp(bio_cbar.ax.get_xticklabels(), rotation=45, ha='right')

# ==== 添加图例注释 ====
add_north_arrow(fig, 0.21, 0.11, size=0.018)
add_scalebar(fig, axes[0], 0.26, 0.118, length_km=500, fontsize=font_size,
             fontfamily=font_family, linewidth=2)
add_annotation(fig, 0.34, 0.122, width=0.015, text="State/Territory boundaries",
               linewidth=2, style="line", linecolor="black",
               fontsize=font_size, fontfamily=font_family)
add_annotation(fig, 0.61, 0.116, width=0.011, height=0.011, linewidth=2,
               text="No data", style="box", facecolor="white", edgecolor="black",
               fontsize=font_size, fontfamily=font_family)
add_annotation(fig, 0.20, 0.087, width=0.012, height=0.011, linewidth=2,
               text="Public, indigenous, urban, water bodies, and other land",
               style="box", facecolor="#808080", edgecolor="#808080",
               fontsize=font_size, fontfamily=font_family)

# 保存图片
output_path = os.path.join(out_dir, f"06_Sol_price_maps_line")
fig.savefig(f"{output_path}.png", dpi=300)
plt.show()
