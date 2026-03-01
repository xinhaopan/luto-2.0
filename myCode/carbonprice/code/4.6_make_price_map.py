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
def plot_tif_grid(tif_title_list, figsize=(10, 10)):
    """
    画2x2网格图，每个子图显示一个tif和title
    tif_title_list: [(tif1, title1), (tif2, title2), (tif3, title3), (tif4, title4)]
    """
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=-0.2, wspace=0.02,
                           left=0.01, right=0.99, top=0.99, bottom=0.01)
    axes = []

    for idx, (tif, title, unit_name, clip_percent) in enumerate(tif_title_list):
        row = idx // 2
        col = idx % 2
        ax = fig.add_subplot(gs[row, col], projection=ccrs.PlateCarree())
        # 获取覆盖参数
        key = os.path.splitext(os.path.basename(tif))[0]  # 得到 'xr_ghg_sol_price_carbon_high_2050'
        main_key = key.replace('xr_', '').replace('_2050', '')  # 得到 'ghg_sol_price_carbon_high'
        kwargs = layer_overrides.get(main_key, {})

        safe_plot(
            tif_path=tif,
            title=title,
            ax=ax,
            unit=unit_name,
            cmap=price_cmap,
            clip_percent=clip_percent,
            title_y=0.95,
            **kwargs
            # 其它参数按需传
        )
        axes.append(ax)
    return fig, axes

base_dir = f"../../../output/{config.TASK_NAME}/carbon_price"
arr_path = f"{base_dir}/4_tif"
out_dir = f"{base_dir}/3_Paper_figure"
os.makedirs(out_dir, exist_ok=True)
# price_cmap = LinearSegmentedColormap.from_list("price", ["#00ffff", "#ff00ff"])
price_cmap = LinearSegmentedColormap.from_list("price", ["#ffff80", "#38e009","#1a93ab","#0c1078"])


layer_overrides = {
    # 'carbon_sol_price_carbon_high': {"custom_tick_values": [0,50,100]},
    'carbon_sol_price_carbon_high_bio_50': {"custom_tick_values": [0,1200,2400]},
    'carbon_sol_price_Counterfactual_carbon_high_bio_50': {"custom_tick_values": [0,1200,2400]},
}

legend_nbins = 3
# 统一样式
set_plot_style(font_size=15, font_family='Arial')

tif_title_list = [
    (f"{arr_path}/carbon_high_50/xr_carbon_sol_price_carbon_high_50_2050.tif", "Shadow carbon price\nunder Net Zero targets",'AU\$ tCO$_2$e$^{-1}$ yr$^{-1}$',[0,95]),
    (f"{arr_path}/carbon_high_bio_50/xr_carbon_sol_price_carbon_high_bio_50_2050.tif", "Shadow carbon price\nunder Nature Positive",'AU\$ tCO$_2$e$^{-1}$ yr$^{-1}$',[0,95]),
    (f"{arr_path}/Counterfactual_carbon_high_bio_50/xr_carbon_sol_price_Counterfactual_carbon_high_bio_50_2050.tif", "Shadow carbon price\nunder both targets",'AU\$ tCO$_2$e$^{-1}$ yr$^{-1}$',[0,95]),
    (f"{arr_path}/carbon_high_bio_50/xr_bio_sol_price_carbon_high_bio_50_2050.tif", "Shadow biodiversity price",'AU\$ ha$^{-1}$ yr$^{-1}$',[0,95]),
]

fig, axes = plot_tif_grid(tif_title_list)

# 获取字体设置
font_size = axes[0].xaxis.get_label().get_size()
font_family = axes[0].xaxis.get_label().get_family()[0]
# 设置字体
plt.rcParams['font.family'] = font_family
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = font_family
plt.rcParams['mathtext.it'] = font_family
plt.rcParams['mathtext.bf'] = font_family
plt.rcParams['mathtext.sf'] = font_family

# 添加图例元素
add_north_arrow(fig, 0.21, 0.038, size=0.018)
add_scalebar(fig, axes[0], 0.26, 0.046, length_km=500, fontsize=font_size,
             fontfamily=font_family, linewidth=2)
add_annotation(fig, 0.34, 0.050, width=0.015, text="State/Territory boundaries",
               linewidth=2, style="line", linecolor="black",
               fontsize=font_size, fontfamily=font_family)
add_annotation(fig, 0.61, 0.044, width=0.011, height=0.011, linewidth=2,
               text="No data", style="box", facecolor="white", edgecolor="black",
               fontsize=font_size, fontfamily=font_family)
add_annotation(fig, 0.20, 0.015, width=0.012, height=0.011, linewidth=2,
               text="Public, indigenous, urban, water bodies, and other land",
               style="box", facecolor="#808080", edgecolor="#808080",
               fontsize=font_size, fontfamily=font_family)

# 保存图片
output_path = os.path.join(out_dir, f"06_Sol_price_maps_line")
fig.savefig(f"{output_path}.png", dpi=300)
plt.show()