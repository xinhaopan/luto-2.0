from matplotlib import pyplot as plt
from matplotlib import gridspec
from matplotlib.colors import LinearSegmentedColormap
import cartopy.crs as ccrs
import os

import tools.config as config
from tools.helper_map import (safe_plot, add_scalebar, add_north_arrow, add_annotation, align_raster_to_reference)
from tools.helper_plot import set_plot_style
import cmocean
def plot_tif_grid(scenarios, tif_title_list, title_names):
    """
    画3x8网格图，每行一个场景，每列一个tif和title
    tif_title_list: [(tif1, title1, unit_name), ...]
    """
    nrows = len(tif_title_list)
    ncols = len(scenarios)
    figsize = (ncols * 5, nrows * 5)
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(nrows, ncols, figure=fig, hspace=-0.6, wspace=0.02,
                           left=0.03, right=0.99, top=0.99, bottom=0.03)
    axes = []

    for row, tif in enumerate(tif_title_list):
        for col,scenario in enumerate(scenarios):
            title_name = ''
            if row == 0:
                title_name = title_names[col]
            if tif == "Total":
                tif_path = f"../../../output/{config.TASK_NAME}/carbon_price/4_tif/{scenario}/xr_total_area_agricultural_management_{scenario}_2050.tif"
            else:
                tif_path = f"../../../output/{config.TASK_NAME}/carbon_price/4_tif/{scenario}/xr_area_agricultural_management_{scenario}_{tif}_2050.tif"
            ax = fig.add_subplot(gs[row, col], projection=ccrs.PlateCarree())
            safe_plot(
                tif_path=tif_path,
                title=title_name,
                ax=ax,
                unit='ha',
                cmap=area_cmap,
                title_y=0.95,
                force_one_start=True,
                custom_tick_values=[0, 0.5, 1],
            )
            axes.append(ax)
    return fig, axes

base_dir = f"../../../output/{config.TASK_NAME}/carbon_price"
arr_path = f"{base_dir}/4_tif"
out_dir = f"{base_dir}/5_map"
os.makedirs(out_dir, exist_ok=True)
# price_cmap = LinearSegmentedColormap.from_list("price", ["#00ffff", "#ff00ff"])
area_cmap = "YlGn"

legend_nbins = 3
# 统一样式
set_plot_style(font_size=15, font_family='Arial')

scenarios = ["Run_18_GHG_off_BIO_off_CUT_50", "Run_06_GHG_high_BIO_off_CUT_50", "Run_01_GHG_high_BIO_high_CUT_50"]
title_names = ['Reference', r'$\mathrm{GHG}_{\mathrm{high}}$',
               r'$\mathrm{GHG}_{\mathrm{high}}$,$\mathrm{Bio}_{\mathrm{50}}$']
tif_title_list = ["Biochar", "Asparagopsis taxiformis", "Savanna burning", "Precision agriculture", "AgTech EI", "HIR - Beef", "HIR - Sheep"]
row_labels = ["Biochar", "Methane reduction (livestock)", "Early dry-season savanna burning", "Agricultural technology (fertiliser)", "Agricultural technology (energy)", "Managed regeneration (beef)", "Managed regeneration (sheep)"]

fig, axes = plot_tif_grid(scenarios, tif_title_list, title_names)

# 获取字体设置
font_size = axes[0].xaxis.get_label().get_size()
font_family = axes[0].xaxis.get_label().get_family()[0]

nrows = len(tif_title_list)
ncols = len(scenarios)
# 添加行标签到每行第一列的左侧
for row in range(nrows):
    ax = axes[row * ncols]  # 每行第一列的子图索引：0, 3, 6, 9
    ax.text(
        -0.01, 0.5, row_labels[row],
        fontsize=font_size,
        fontfamily=font_family,
        rotation=90,
        va='center', ha='right',
        transform=ax.transAxes,
        clip_on=False
    )

# 设置字体
plt.rcParams['font.family'] = font_family
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = font_family
plt.rcParams['mathtext.it'] = font_family
plt.rcParams['mathtext.bf'] = font_family
plt.rcParams['mathtext.sf'] = font_family

# 添加图例元素
add_north_arrow(fig, 0.14, 0.096, size=0.012)
add_scalebar(fig, axes[0], 0.17, 0.104, length_km=500, fontsize=font_size,
             fontfamily=font_family, linewidth=2)
add_annotation(fig, 0.23, 0.106, width=0.015, text="State/Territory boundaries",
               linewidth=2, style="line", linecolor="black",
               fontsize=font_size, fontfamily=font_family)
add_annotation(fig, 0.42, 0.105, width=0.007, height=0.003, linewidth=2,
               text="No data", style="box", facecolor="white", edgecolor="black",
               fontsize=font_size, fontfamily=font_family)
add_annotation(fig, 0.49, 0.105, width=0.007, height=0.003, linewidth=2,
               text="Public, indigenous, urban, water bodies, and other land",
               style="box", facecolor="#808080", edgecolor="#808080",
               fontsize=font_size, fontfamily=font_family)

# 保存图片
output_path = os.path.join(out_dir, f"06_area_agmgt_maps_line")
fig.savefig(f"{output_path}.png", dpi=300)
plt.show()