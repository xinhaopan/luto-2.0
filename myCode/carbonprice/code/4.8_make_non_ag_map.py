from matplotlib import pyplot as plt
from matplotlib import gridspec
from matplotlib.colors import LinearSegmentedColormap
import cartopy.crs as ccrs
import os

import tools.config as config
from tools.helper_map import (check_and_create_missing_tifs, safe_plot, add_scalebar, add_north_arrow, add_annotation, align_raster_to_reference)
from tools.helper_plot import set_plot_style
import cmocean
import matplotlib.colors as mcolors


def plot_tif_grid(scenarios, tif_title_list, title_names):
    """
    画3x8网格图，每行一个场景，每列一个tif和title
    tif_title_list: [(tif1, title1, unit_name), ...]
    """
    nrows = len(tif_title_list)
    ncols = len(scenarios)
    figsize = (ncols * 5, nrows * 4.2)

    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(
        nrows, ncols, figure=fig, hspace=-0.2, wspace=0.02,
        left=0.05, right=0.99, top=1, bottom=0.04
    )
    axes = []

    # ---------- 1) 先收集所有 tif 路径 ----------
    tif_paths = []
    cell_jobs = []  # 保存每个子图需要的信息，后面画图用

    for row, tif in enumerate(tif_title_list):
        for col, scenario in enumerate(scenarios):
            title_name = title_names[col] if row == 0 else ""

            if tif == "Total":
                tif_path = (
                    f"../../../output/{config.TASK_NAME}/carbon_price/4_tif/{scenario}/"
                    f"xr_total_area_non_agricultural_landuse_{scenario}_2050.tif"
                )
                title_name = title_names[col]
            else:
                tif_path = (
                    f"../../../output/{config.TASK_NAME}/carbon_price/4_tif/{scenario}/"
                    f"xr_area_non_agricultural_landuse_{scenario}_{tif}_2050.tif"
                )

            tif_paths.append(tif_path)
            cell_jobs.append((row, col, tif_path, title_name))

    # ---------- 2) 关键：绘图前检查并创建缺失 tif ----------
    # 直接调用你写的函数即可
    existing, created = check_and_create_missing_tifs(tif_paths)

    # 可选：如果缺失但没有模板，就提前提示（避免后面每个子图都报 not found）
    if len(existing) == 0 and len(tif_paths) > 0:
        print("[ERROR] 所有 tif 都不存在，无法用模板创建缺失文件。")
        # 这里仍然可以继续画，safe_plot 会在子图上写 File not found
        # return fig, axes  # 你也可以选择直接返回

    # ---------- 3) 画图 ----------
    for row, col, tif_path, title_name in cell_jobs:
        ax = fig.add_subplot(gs[row, col], projection=ccrs.PlateCarree())
        safe_plot(
            tif_path=tif_path,
            title=title_name,
            ax=ax,
            unit='%',
            cmap=area_cmap,
            title_y=0.95,
            force_one_start=False,
            custom_tick_values=[0, 0.5, 1],
            create_colorbar=False
        )
        axes.append(ax)

    return fig, axes


# --- Configuration ---
base_dir = f"../../../output/{config.TASK_NAME}/carbon_price"
out_dir = f"{base_dir}/3_Paper_figure"
os.makedirs(out_dir, exist_ok=True)

# area_cmap = cmocean.cm.speed
orig_cmap = plt.get_cmap("YlGn")
new_colors = [mcolors.to_rgba("white")] + [orig_cmap(i/255) for i in range(256)]
area_cmap = mcolors.LinearSegmentedColormap.from_list("white_YlGn", new_colors)

set_plot_style(font_size=15, font_family='Arial')

# --- Data and Labels ---
scenarios = ["Run_21_GHG_off_BIO_off_CUT_50", "Run_06_GHG_high_BIO_off_CUT_50", "Run_01_GHG_high_BIO_high_CUT_50"]

# Column headers for the plot grid
title_names = ['Reference', r'$\mathrm{NZ}_{\mathrm{high}}$',
               r'$\mathrm{NZ}_{\mathrm{high}}$,$\mathrm{NP}_{\mathrm{50}}$']
tif_title_list = ["Destocked - natural land",  "Riparian plantings",
                    "Environmental plantings", "Sheep agroforestry", "Beef agroforestry"]
row_labels = ["Destocked (natural land)","Riparian buffer restoration\n(mixed species)",
                "Environmental plantings\n(mixed local native species)",
              "Agroforestry\n(mixed species + sheep)","Agroforestry\n(mixed species + beef)", ]

# --- Plotting ---
fig, axes = plot_tif_grid(scenarios, tif_title_list, title_names)

# Get font settings from a plotted axis to ensure consistency
font_size = axes[0].xaxis.get_label().get_size()
font_family = axes[0].xaxis.get_label().get_family()[0]

# Add row labels to the left of the first column
nrows = len(tif_title_list)
ncols = len(scenarios)
for row in range(nrows):
    ax = axes[row * ncols]  # Get the first axis of each row
    ax.text(
        -0.05, 0.5, row_labels[row],  # Position the text to the left of the axis
        fontsize=font_size,
        fontfamily=font_family,
        rotation=90,
        va='center', ha='right',
        multialignment='center',
        transform=ax.transAxes
    )

# Configure font settings for mathematical text in plots
plt.rcParams['font.family'] = font_family
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = font_family
plt.rcParams['mathtext.it'] = font_family
plt.rcParams['mathtext.bf'] = font_family
plt.rcParams['mathtext.sf'] = font_family

im = axes[0].images[0]
cax = fig.add_axes([0.2, 0.04, 0.6, 0.015])  # [left, bottom, width, height]，可调整
cbar = fig.colorbar(im, cax=cax, orientation='horizontal', extend='both')
cbar.ax.xaxis.set_label_position('top')
cbar.set_ticks([0, 0.25, 0.5, 0.75, 1])
cbar.set_ticklabels(['0', '0.25', '0.50', '0.75', '1'])
cbar.ax.tick_params(labelsize=font_size)
cbar.set_label('Proportion', fontsize=font_size, fontfamily=font_family)

# --- Add Legend and Annotations ---
# Coordinates are adjusted for the new tall figure layout
# 添加图例元素
add_north_arrow(fig, 0.14, 0.001, size=0.012)
add_scalebar(fig, axes[0], 0.17, 0.009, length_km=500, fontsize=font_size,
             fontfamily=font_family, linewidth=2)
add_annotation(fig, 0.23, 0.012, width=0.015, text="State/Territory boundaries",
               linewidth=2, style="line", linecolor="black",
               fontsize=font_size, fontfamily=font_family)
add_annotation(fig, 0.42, 0.01, width=0.009, height=0.0045, linewidth=2,
               text="No data", style="box", facecolor="white", edgecolor="black",
               fontsize=font_size, fontfamily=font_family)
add_annotation(fig, 0.49, 0.01, width=0.009, height=0.0045, linewidth=2,
               text="Public, indigenous, urban, water bodies, and other land",
               style="box", facecolor="#808080", edgecolor="#808080",
               fontsize=font_size, fontfamily=font_family)

# --- Save and Show Plot ---
output_path = os.path.join(out_dir, "06_area_non_ag_maps_line")
fig.savefig(f"{output_path}.png", dpi=300, bbox_inches='tight')
plt.show()