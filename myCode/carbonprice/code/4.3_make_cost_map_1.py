import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib import gridspec
from matplotlib.colors import LinearSegmentedColormap, BoundaryNorm
import cartopy.crs as ccrs
import numpy as np
import os

import tools.config as config
from tools.helper_map import (safe_plot, add_scalebar, add_north_arrow, add_annotation,
                               align_raster_to_reference, get_y_axis_ticks)
from tools.helper_plot import set_plot_style

def plot_cost_grid(scenarios: dict, year: int = 2050, figsize=None, nrows=4, ncols=3,
                   vmin_shared=None, vmax_shared=None):
    """
    创建成本网格图：每行一种成本类型，每列一个场景

    Parameters:
    -----------
    scenarios : dict
        场景字典，键为场景名称
    year : int
        年份
    figsize : tuple
        图像尺寸
    nrows : int
        行数（成本类型数量）
    ncols : int
        列数（场景数量）
    vmin_shared : float
        所有子图共享的色标最小值
    vmax_shared : float
        所有子图共享的色标最大值
    """
    print(f"\n===== CREATING COST GRID FOR ALL SCENARIOS (year={year}) =====")

    # 自动计算图像尺寸（底部多留 1 英寸用于共享色标）
    if figsize is None:
        figsize = (ncols * 5, nrows * 4.2 + 1.0)

    fig = plt.figure(figsize=figsize)

    # 创建网格规范（bottom 留出空间给底部共享色标和注释）
    gs = gridspec.GridSpec(nrows, ncols, figure=fig, hspace=-0.3, wspace=0.015,
                           left=0.03, right=0.99, top=0.99, bottom=0.10)

    axes_list = []
    scenario_names = list(scenarios.keys())

    # 按行循环：每行一种成本类型
    for row, (cost_key, cost_title) in enumerate(zip(env_keys, row_labels)):
        print(f"\n--- Row {row}: {cost_title} ---")

        # 按列循环：每列一个场景
        for col, env in enumerate(scenario_names):
            print(f"  Plotting {env} - {cost_title}")

            # 创建子图
            ax = fig.add_subplot(gs[row, col], projection=ccrs.PlateCarree())

            # 构建tif文件路径
            tif = f"{arr_path}/{env}/xr_{cost_key}_ha_{env}_{year}.tif"

            # 获取覆盖参数
            kwargs = layer_overrides.get(cost_key, {})

            # 绘图（禁用单图色标，使用共享 vmin/vmax）
            safe_plot(
                tif_path=tif,
                title='',
                unit=r"AU\$ ha$^{-1}$ yr$^{-1}$",
                cmap=cost_cmap,
                ax=ax,
                force_one_start=True,
                create_colorbar=False,
                vmin_override=vmin_shared,
                vmax_override=vmax_shared,
                **kwargs
            )

            axes_list.append(ax)

    return fig, axes_list


# ==== Paths & global params ====
base_dir = f"../../../output/{config.TASK_NAME}/carbon_price"
arr_path = f"{base_dir}/4_tif"
out_dir = f"{base_dir}/3_Paper_figure"
os.makedirs(out_dir, exist_ok=True)

legend_nbins = 3
legend_bbox = (0.1, 0.10, 0.8, 0.9)

# 统一样式
set_plot_style(font_size=15, font_family='Arial')

# 参照/掩膜对齐（仅一次）
ref_tif = f"{arr_path}/carbon_high_50/xr_total_cost_ha_carbon_high_50_2050.tif"
src_tif = f"../Map/public_area.tif"
aligned_tif = f"../Map/public_area_aligned.tif"
align_raster_to_reference(src_tif, ref_tif, aligned_tif, resampling="nearest")

# 统一色带
cost_cmap = LinearSegmentedColormap.from_list("cost", ["#FFFEC2", "#FA4F00", "#A80000", "#5c2324"])

# ==== 场景配置 ====
scenarios = {
    "carbon_high_50": {},
    "carbon_high_bio_50": {},
    "Counterfactual_carbon_high_bio_50": {}
}

# 成本组件的键（按行顺序）
env_keys = [
    "total_sol_cost",
    # "cost_agricultural_management",
    "cost_non_ag",
    "transition_cost_ag2non_ag_amortised_diff",
]

# 行标签（左侧）
row_labels = [
    'Total solution cost',
    # 'Agricultural management cost',
    'Non-agriculture cost',
    'Transition(ag→non-ag) cost',
]

# 列标题（顶部）
column_titles = [
    r'Reference→$\mathrm{NZ}_{\mathrm{high}}$',
    r'$\mathrm{NZ}_{\mathrm{high}}$→$\mathrm{NZ}_{\mathrm{high}}$,$\mathrm{NP}_{\mathrm{50}}$',
    r'Reference→$\mathrm{NZ}_{\mathrm{high}}$,$\mathrm{NP}_{\mathrm{50}}$'
]

# 可选覆盖
layer_overrides = {
    'total_cost': {"clip_percent": [0, 100]},
    # 'cost_agricultural_management': {"clip_percent": [1, 99]},
    'cost_non_ag': {"clip_percent": [0, 100]},
    'transition_cost_ag2non_ag_amortised_diff': {"clip_percent": [1, 100]},
}

# ==== 全局 vmax（固定值）====
year_plot = 2050
global_vmax = 1500.0

# 计算共享刻度：以 150 为步长
seg_step = 150
vmin_shared = 0.0
n_segs = int(np.ceil(global_vmax / seg_step))
vmax_shared = float(n_segs * seg_step)
ticks_shared = [float(i * seg_step) for i in range(n_segs + 1)]

print(f"[Shared colorbar] vmin={vmin_shared}, vmax={vmax_shared}, ticks={ticks_shared}")

# ==== 创建网格图 ====
nrows = len(env_keys)
ncols = len(scenarios)
fig, axes = plot_cost_grid(scenarios, year=year_plot, nrows=nrows, ncols=ncols,
                           vmin_shared=vmin_shared, vmax_shared=vmax_shared)

# 获取字体设置
font_size = axes[0].xaxis.get_label().get_size()
font_family = axes[0].xaxis.get_label().get_family()[0]

# 添加列标题到第一行的每个子图上方
for col in range(ncols):
    ax = axes[col]
    ax.set_title(column_titles[col], fontsize=font_size, fontfamily=font_family, pad=5)

# 添加行标签到每行第一列的左侧
for row in range(nrows):
    ax = axes[row * ncols]
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

# ==== 底部共享分段色标 ====
bounds_cb = np.array(ticks_shared)
norm_cb = BoundaryNorm(bounds_cb, cost_cmap.N)
sm = plt.cm.ScalarMappable(cmap=cost_cmap, norm=norm_cb)
sm.set_array([])

# 色标轴：紧贴地图底部（地图 bottom=0.10，色标顶端对齐）
cbar_ax = fig.add_axes([0.25, 0.11, 0.5, 0.02])
cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal', extend='both')
cbar.set_ticks(ticks_shared)
cbar.set_ticklabels([f"{int(v):,}" if v != 0 else "0" for v in ticks_shared])
cbar.set_label(r"AU\$ ha$^{-1}$ yr$^{-1}$", fontsize=font_size, fontfamily=font_family, labelpad=5)
cbar.ax.xaxis.set_label_position('top')
cbar.ax.xaxis.set_ticks_position('bottom')
cbar.ax.tick_params(labelsize=font_size - 1, length=3, pad=2)
cbar.outline.set_visible(False)

# 添加图例元素（注释行，位于色标下方）
add_north_arrow(fig, 0.15, 0.052, size=0.012)
add_scalebar(fig, axes[0], 0.19, 0.058, length_km=500, fontsize=font_size,
             fontfamily=font_family, linewidth=1.5)
add_annotation(fig, 0.25, 0.062, width=0.015, text="State/Territory boundaries",
               linewidth=1.5, style="line", linecolor="black",
               fontsize=font_size, fontfamily=font_family)
add_annotation(fig, 0.44, 0.059, width=0.01, height=0.0075, linewidth=1.5,
               text="No data", style="box", facecolor="white", edgecolor="black",
               fontsize=font_size, fontfamily=font_family)
add_annotation(fig, 0.52, 0.059, width=0.01, height=0.0075, linewidth=1.5,
               text="Public, indigenous, urban, water bodies, and other land",
               style="box", facecolor="#808080", edgecolor="#808080",
               fontsize=font_size, fontfamily=font_family)

# 保存图片
output_path = os.path.join(out_dir, f"06_cost_maps_line_clip")
fig.savefig(f"{output_path}.png", dpi=300)
plt.show()
