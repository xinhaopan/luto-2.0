from matplotlib import pyplot as plt
from matplotlib import gridspec
from matplotlib.colors import LinearSegmentedColormap
import cartopy.crs as ccrs
import os

import tools.config as config
from tools.helper_map import (safe_plot, add_scalebar, add_north_arrow, add_annotation, align_raster_to_reference)
from tools.helper_plot import set_plot_style

def plot_cost_grid(scenarios: dict, year: int = 2050, figsize=None, nrows=4, ncols=3):
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
    """
    print(f"\n===== CREATING COST GRID FOR ALL SCENARIOS (year={year}) =====")

    # 自动计算图像尺寸
    if figsize is None:
        figsize = (ncols * 5, nrows * 5)

    fig = plt.figure(figsize=figsize)

    # 创建网格规范
    gs = gridspec.GridSpec(nrows, ncols, figure=fig, hspace=-0.45, wspace=0.03,
                           left=0.03, right=0.99, top=0.99, bottom=0.03)

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
            tif = f"{arr_path}/{env}/xr_{cost_key}_{env}_{year}.tif"

            # 获取覆盖参数
            kwargs = layer_overrides.get(cost_key, {})

            # 绘图
            safe_plot(
                tif_path=tif,
                title='',  # 标题统一在外部添加
                unit=r"%",


                cmap=benefit_cmap,
                ax=ax,
                **kwargs
            )

            axes_list.append(ax)

    return fig, axes_list


# ==== Paths & global params ====
base_dir = f"../../../output/{config.TASK_NAME}/carbon_price"
arr_path = f"{base_dir}/4_tif"
out_dir = f"{base_dir}/5_map"
os.makedirs(out_dir, exist_ok=True)

legend_nbins = 3
legend_bbox = (0.1, 0.10, 0.8, 0.9)

# 统一样式
set_plot_style(font_size=15, font_family='Arial')

# 参照/掩膜对齐（仅一次）
ref_tif = f"{arr_path}/carbon_high/xr_total_cost_carbon_high_2050.tif"
src_tif = f"../Map/public_area.tif"
aligned_tif = f"../Map/public_area_aligned.tif"
align_raster_to_reference(src_tif, ref_tif, aligned_tif, resampling="nearest")

# 统一色带
benefit_cmap = LinearSegmentedColormap.from_list("benefit", ["#ffff80", "#38e009","#1a93ab","#0c1078"])

# ==== 场景配置 ====
scenarios = {
    # "carbon_high": {},
    "carbon_high_bio_50": {},
    "Counterfactual_carbon_high_bio_50": {}
}

# 成本组件的键（按行顺序）
env_keys = [
    "total_bio",
    "biodiversity_GBF2_priority_ag_management",
    "biodiversity_GBF2_priority_non_ag",
]

# 行标签（左侧）
row_labels = [
    'Total',
    'Agricultural management',
    'Non-agriculture',
]

# 列标题（顶部）
column_titles = [
    # r'Reference→$\mathrm{GHG}_{\mathrm{high}}$',
    r'$\mathrm{GHG}_{\mathrm{high}}$→$\mathrm{GHG}_{\mathrm{high}}$,$\mathrm{Bio}_{\mathrm{50}}$',
    r'Reference→$\mathrm{GHG}_{\mathrm{high}}$,$\mathrm{Bio}_{\mathrm{50}}$'
]

# 可选覆盖
layer_overrides = {
    # 'total_carbon': {"clip_percent": [0, 99]},
    # 'GHG_ag_management': {"clip_percent": [0, 99]},
    # 'GHG_non_ag': {"clip_percent": [0, 99]},
    # 'transition_cost_ag2non_ag_amortised_diff': {"clip_percent": [0, 95]},
}

# ==== 创建网格图 ====
nrows = len(env_keys)  # 4种成本类型
ncols = len(scenarios)  # 3个场景
fig, axes = plot_cost_grid(scenarios, year=2050, nrows=nrows, ncols=ncols)

# 获取字体设置
font_size = axes[0].xaxis.get_label().get_size()
font_family = axes[0].xaxis.get_label().get_family()[0]

# 添加列标题到第一行的每个子图上方
for col in range(ncols):
    ax = axes[col]  # 第一行的子图索引：0, 1, 2
    ax.set_title(column_titles[col], fontsize=font_size, fontfamily=font_family, pad=5)

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
add_north_arrow(fig, 0.03, 0.08, size=0.02)
add_scalebar(fig, axes[0], 0.08, 0.089, length_km=500, fontsize=font_size,
             fontfamily=font_family, linewidth=2)
add_annotation(fig, 0.15, 0.093, width=0.015, text="Australian state boundary",
               linewidth=2, style="line", linecolor="black",
               fontsize=font_size, fontfamily=font_family)
add_annotation(fig, 0.35, 0.090, width=0.008, height=0.0072, linewidth=2,
               text="No data", style="box", facecolor="white", edgecolor="black",
               fontsize=font_size, fontfamily=font_family)
add_annotation(fig, 0.44, 0.090, width=0.008, height=0.0072, linewidth=2,
               text="Public, indigenous, urban, and other intensive land uses",
               style="box", facecolor="#808080", edgecolor="#808080",
               fontsize=font_size, fontfamily=font_family)

# 保存图片
output_path = os.path.join(out_dir, f"06_BIO_maps_line")
fig.savefig(f"{output_path}.png", dpi=300)
plt.show()