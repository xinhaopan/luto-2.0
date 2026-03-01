import matplotlib
matplotlib.use('Agg')
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
        figsize = (ncols * 5, nrows * 4.5)

    fig = plt.figure(figsize=figsize)

    # 创建网格规范
    gs = gridspec.GridSpec(nrows, ncols, figure=fig, hspace=-0.2, wspace=0.03,
                           left=0.03, right=0.99, top=0.99, bottom=0.01)

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

            # 绘图
            safe_plot(
                tif_path=tif,
                title='',  # 标题统一在外部添加
                unit=r"tCO$_2$e ha$^{-1}$ yr$^{-1}$",


                cmap=benefit_cmap,
                ax=ax,

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
# benefit_cmap = LinearSegmentedColormap.from_list("benefit", ["#ffff80", "#38e009","#1a93ab","#0c1078"])
benefit_cmap = "BrBG"  # 预设色带

# ==== 场景配置 ====
scenarios = {
    "carbon_high_50": {},
    "carbon_high_bio_50": {},
    "Counterfactual_carbon_high_bio_50": {}
}

# 成本组件的键（按行顺序）
env_keys = [
    # "total_sol_ghg_benefit",
    # "GHG_ag_management",
    "GHG_non_ag",
]

# 行标签（左侧）
row_labels = [
    # 'Total solution',
    # 'Agricultural management',
    'Non-agriculture',
]

# 列标题（顶部）
column_titles = [
    r'Reference→$\mathrm{NZ}_{\mathrm{high}}$',
    r'$\mathrm{NZ}_{\mathrm{high}}$→$\mathrm{NZ}_{\mathrm{high}}$,$\mathrm{NP}_{\mathrm{50}}$',
    r'Reference→$\mathrm{NZ}_{\mathrm{high}}$,$\mathrm{NP}_{\mathrm{50}}$'
]

# 可选覆盖
layer_overrides = {
    # 'total_sol_ghg_benefit': {"clip_percent": [1,99],"force_zero_center": True,},
    # 'GHG_ag_management': {"clip_percent": [1,99],"force_zero_center": True,},
    'GHG_non_ag': {"clip_percent": [0,100],"force_zero_center": True,},
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
# for row in range(nrows):
#     ax = axes[row * ncols]  # 每行第一列的子图索引：0, 3, 6, 9
#     ax.text(
#         -0.01, 0.5, row_labels[row],
#         fontsize=font_size,
#         fontfamily=font_family,
#         rotation=90,
#         va='center', ha='right',
#         transform=ax.transAxes,
#         clip_on=False
#     )

# 设置字体
plt.rcParams['font.family'] = font_family
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = font_family
plt.rcParams['mathtext.it'] = font_family
plt.rcParams['mathtext.bf'] = font_family
plt.rcParams['mathtext.sf'] = font_family

# 添加图例元素
add_north_arrow(fig, 0.19, 0.008, size=0.03)
add_scalebar(fig, axes[0], 0.22, 0.014, length_km=500, fontsize=font_size,
             fontfamily=font_family, linewidth=2)
add_annotation(fig, 0.28, 0.028, width=0.015, text="State/Territory boundaries",
               linewidth=2, style="line", linecolor="black",
               fontsize=font_size, fontfamily=font_family)
add_annotation(fig, 0.470, 0.014, width=0.008, height=0.008, linewidth=2,
               text="No data", style="box", facecolor="white", edgecolor="black",
               fontsize=font_size, fontfamily=font_family)
add_annotation(fig, 0.54, 0.0145, width=0.008, height=0.008, linewidth=2,
               text="Public, indigenous, urban, water bodies, and other land",
               style="box", facecolor="#808080", edgecolor="#808080",
               fontsize=font_size, fontfamily=font_family)

# 保存图片
output_path = os.path.join(out_dir, f"06_GHG_maps_line")
fig.savefig(f"{output_path}.png", dpi=300)
plt.show()