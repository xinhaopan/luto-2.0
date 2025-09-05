import xarray as xr
from matplotlib.ticker import MaxNLocator
from matplotlib.ticker import FuncFormatter
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt

import tools.config as config


import seaborn as sns
sns.set_theme(style="darkgrid",font="Arial", font_scale=2)
plt.rcParams.update({
    "xtick.bottom": True, "ytick.left": True,   # 打开刻度
    "xtick.top": False,  "ytick.right": False,  # 需要的话也可开
    "xtick.direction": "out", "ytick.direction": "out",
    "xtick.major.size": 4, "ytick.major.size": 4,
    "xtick.major.width": 1.2, "ytick.major.width": 1.2,
})
plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 20


def plot_lines_with_buffer(
        ax,
        da,  # DataArray, dims: (env_category, year, cell)
        q_low=0.1, q_high=0.9,  # buffer 的分位数范围，比如 10%~90%
        colors=None,  # 主线颜色列表
        buffer_colors=None,  # buffer 填充颜色列表
        name_map=None,  # { "carbon_100": "100%", ... }
        xlabel="Year",
        ylabel="Value",
        title="Lines with buffer",
        alpha=1,  # buffer 透明度
        linestyle="-",  # 中位数线型
        marker="o",  # 中位数点型
        reverse_env_order=True  # 顺序是否反转
):

    for spine in ax.spines.values():
        spine.set_visible(True)  # 显示所有边框
        spine.set_linewidth(1.2)  # 边框粗细
        spine.set_edgecolor("black")  # 边框颜色
    if isinstance(da, xr.Dataset):
        da = da["data"]

    years = da.coords["year"].values
    envs = list(da.coords["env_category"].values)

    if colors is None:
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    if buffer_colors is None:
        buffer_colors = colors
    if name_map is None:
        name_map = {}
    if reverse_env_order:
        envs = envs[::-1]
        colors = colors[::-1]
        buffer_colors = buffer_colors[::-1]

    # 存储信息供图例使用
    ax._buffer_legend_info = {
        'envs': envs,
        'colors': colors,
        'buffer_colors': buffer_colors,
        'name_map': name_map,
        'alpha': alpha,
        'linestyle': linestyle,
        'marker': marker
    }

    for i, env in enumerate(envs):
        da_env = da.sel(env_category=env)  # (year, cell)
        low = da_env.quantile(q_low, dim="cell")  # (year,)
        high = da_env.quantile(q_high, dim="cell")  # (year,)

        # 获得每个 cell 是否在 buffer 区间
        buffer_mask = ((da_env >= low) & (da_env <= high))  # (year, cell)

        # 把在buffer区间的cell值提取出来，对这些cell再求中位数
        buffered = da_env.where(buffer_mask, drop=True)  # (year, cell_in_buffer)
        buffer_median = buffered.median(dim="cell")  # (year,)

        # 提取 numpy 数组
        line = buffer_median.values
        low = low.values
        high = high.values

        env_label = name_map.get(str(env), str(env))

        # 绘制 buffer（无标签）
        ax.fill_between(
            years, low, high,
            color=buffer_colors[i % len(buffer_colors)],
            alpha=alpha,
            edgecolor='none'
        )

        # 绘制中位数线（带标签）
        ax.plot(
            years, line,
            color=colors[i % len(colors)],
            label=env_label,
            linestyle=linestyle,
            marker=marker,
            markersize=4,
        )
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xticks(years[::5])
    ax.set_title(title)

    return ax


class CombinedLegendHandler:
    """自定义图例处理器，显示buffer+线条的组合"""

    def __init__(self, buffer_color, line_color, alpha, linestyle, marker):
        self.buffer_color = buffer_color
        self.line_color = line_color
        self.alpha = alpha
        self.linestyle = linestyle
        self.marker = marker

    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        x0, y0 = handlebox.xdescent, handlebox.ydescent
        width, height = handlebox.width, handlebox.height

        # 绘制buffer背景
        buffer_patch = plt.Rectangle((x0, y0), width, height,
                                     facecolor=self.buffer_color,
                                     alpha=self.alpha,
                                     edgecolor='none',
                                     transform=handlebox.get_transform())
        handlebox.add_artist(buffer_patch)

        # 绘制中心线条
        line = plt.Line2D([x0, x0 + width], [y0 + height / 2, y0 + height / 2],
                          color=self.line_color,
                          linestyle=self.linestyle,
                          linewidth=1.5,
                          transform=handlebox.get_transform())
        handlebox.add_artist(line)

        # 添加标记点
        if self.marker and self.marker != 'None':
            marker_patch = plt.Line2D([x0 + width / 2], [y0 + height / 2],
                                      color=self.line_color,
                                      marker=self.marker,
                                      markersize=3,
                                      linestyle='',
                                      transform=handlebox.get_transform())
            handlebox.add_artist(marker_patch)

        return buffer_patch


def draw_legend(ax, bbox_to_anchor=(0.98, 0.69), ncol=6, legend_title="GHG targets percentage (%)"):
    """绘制包含buffer+线条的组合图例"""

    # 获取存储的信息
    if not hasattr(ax, '_buffer_legend_info'):
        # 如果没有信息，使用标准图例
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            fig = ax.get_figure()
            leg = ax.get_legend()
            if leg is not None:
                leg.remove()
            fig.legend(handles=handles, labels=labels,
                       loc='upper right', bbox_to_anchor=bbox_to_anchor,
                       ncol=ncol, frameon=False)
        return

    info = ax._buffer_legend_info
    fig = ax.get_figure()

    # 移除现有图例
    leg = ax.get_legend()
    if leg is not None:
        leg.remove()

    # 创建组合图例
    legend_elements = []

    for i, env in enumerate(info['envs']):
        env_label = info['name_map'].get(str(env), str(env))

        # 创建自定义图例句柄
        handler = CombinedLegendHandler(
            buffer_color=info['buffer_colors'][i % len(info['buffer_colors'])],
            line_color=info['colors'][i % len(info['colors'])],
            alpha=info['alpha'],
            linestyle=info['linestyle'],
            marker=info['marker']
        )

        # 创建一个虚拟的Line2D对象作为key
        proxy = Line2D([0], [0], color=info['colors'][i % len(info['colors'])],
                       label=env_label)
        legend_elements.append(proxy)

    if legend_elements:
        # 创建图例，使用自定义处理器
        legend_handlers = {}
        for element in legend_elements:
            i = legend_elements.index(element)
            legend_handlers[element] = CombinedLegendHandler(
                buffer_color=info['buffer_colors'][i % len(info['buffer_colors'])],
                line_color=info['colors'][i % len(info['colors'])],
                alpha=info['alpha'],
                linestyle=info['linestyle'],
                marker=info['marker']
            )

        fig.legend(handles=legend_elements,
                   loc='upper right', bbox_to_anchor=bbox_to_anchor,
                   title=legend_title,
                   ncol=ncol, frameon=False,
                   handler_map=legend_handlers,
                   handlelength=2.0, handleheight=1.0,
                   handletextpad=0.4, labelspacing=0.3)

def plot_xarray_lines(
    ax,
    da,                      # xarray.Dataset 或 DataArray, dims=(env_category, year)
    colors=None,             # 颜色列表
    name_map=None,           # dict: { "carbon_100": "100%", ... }
    xlabel="Year",
    ylabel="Value",
    title="",
    markersize=4,
    linewidth=2,
):
    for spine in ax.spines.values():
        spine.set_visible(True)  # 显示所有边框
        spine.set_linewidth(1.2)  # 边框粗细
        spine.set_edgecolor("black")  # 边框颜色
    # 如果传入的是 Dataset，先取出 data 变量
    if isinstance(da, xr.Dataset):
        da = da["data"]

    years = da.coords["year"].values
    envs = da.coords["env_category"].values

    if colors is None:
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    if name_map is None:
        name_map = {}

    for i, env in enumerate(envs):
        y = da.sel(env_category=env).values
        ax.plot(
            years, y,
            marker="o", linestyle="-",
            markersize=markersize, linewidth=linewidth,
            color=colors[i % len(colors)],
            label=name_map.get(str(env), str(env))
        )

    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.set_xlabel(xlabel)
    ax.set_xticks(years[::5])
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    # ax.legend(title="Scenario", bbox_to_anchor=(1.05, 1), loc="upper left", frameon=False)

    return ax

input_dir = f'../../../output/{config.TASK_NAME}/carbon_price/0_base_data/Results'
output_dir = f"../../../output/{config.TASK_NAME}/carbon_price/3_Paper_figure"

xr_carbon = xr.open_dataset(f'{input_dir}/xr_carbon_scenario_sum.nc') / 1e6
xr_carbon_cost = xr.open_dataset(f'{input_dir}/xr_carbon_cost_scenario_sum.nc') / 1e6
xr_carbon_price_avg = xr.open_dataset(f'{input_dir}/xr_carbon_price_scenario_avg.nc')
xr_carbon_price_cell = xr.open_dataset(f'{input_dir}/xr_carbon_price_scenario_cell.nc')
years = xr_carbon.coords["year"].values
mask = years >= 2025

name_map={
        "carbon_20": "20",
        "carbon_40": "40",
        "carbon_60": "60",
        "carbon_80": "80",
        "carbon_100": "100",
    }
colors = ["#88B89C", "#53A8BE","#73689C", "#F0912C", "#EB3236"]
buffer_colors = ["#D5E8DB", "#C0D8E2", "#CBC5DA", "#F0C8B5", "#F2AA96"]
fig = plt.figure(figsize=(13, 13))
ax1 = fig.add_subplot(221)
plot_xarray_lines(
    ax1,
    xr_carbon["data"].sel(year=mask),
    colors=colors,
    name_map=name_map,
    xlabel="",
    ylabel=r"MtCO$_2$e yr$^{-1}$",
    title="GHG reductions and removals",
)

ax2 = fig.add_subplot(222)
plot_xarray_lines(
    ax2,
    xr_carbon_cost["data"].sel(year=mask),
    colors=colors,
    name_map=name_map,
    xlabel="",
    ylabel=r"MAU\$ yr$^{-1}$",
    title="GHG reductions and removals cost",
)

ax3 = fig.add_subplot(223)
plot_xarray_lines(
    ax3,
    xr_carbon_price_avg["data"].sel(year=mask),
    colors=colors,
    name_map=name_map,
    xlabel="",
    ylabel=r"AU\$ CO$_2$e$^{-1}$",
    title="Carbon average price",
)

ax4 = fig.add_subplot(224)
plot_lines_with_buffer(
    ax4,
    xr_carbon_price_cell["data"].sel(year=mask),                 # 你的 DataArray (env_category, year, cell)
    q_low=0.95, q_high=0.99,
    colors=colors,
    buffer_colors=colors,
    name_map=name_map,
    xlabel="",
    ylabel=r"AU\$ CO$_2$e$^{-1}$",
    title="Uniform carbon price",
    alpha=0.3,
    reverse_env_order=True
)
draw_legend(ax4, bbox_to_anchor=(0.9, 0.15),legend_title="GHG targets percentage (%)")

fig.subplots_adjust(left=0.10, right=0.95, top=0.92, bottom=0.2, wspace=0.4, hspace=0.4)
plt.savefig(f"{output_dir}/3_carbon&cost.png", dpi=300)
plt.show()
del fig, ax1, ax2, ax3, ax4
# ------------------------------------------------  ------------------------            ----------------

xr_bio = xr.open_dataset(f'{input_dir}/xr_bio_scenario_sum.nc') / 1e6
xr_bio_cost = xr.open_dataset(f'{input_dir}/xr_bio_cost_scenario_sum.nc') / 1e6
xr_bio_price_avg = xr.open_dataset(f'{input_dir}/xr_bio_price_scenario_avg.nc')
xr_bio_price_cell = xr.open_dataset(f'{input_dir}/xr_bio_price_scenario_cell.nc')
years = xr_bio.coords["year"].values
mask = years >= 2025

name_map={
        "carbon_100_bio_50": "50",
        "carbon_100_bio_40": "40",
        "carbon_100_bio_30": "30",
        "carbon_100_bio_20": "20",
        "carbon_100_bio_10": "10",
    }
colors = ["#88B89C", "#53A8BE","#73689C", "#F0912C", "#EB3236"]
buffer_colors = ["#D5E8DB", "#C0D8E2", "#CBC5DA", "#F0C8B5", "#F2AA96"]
fig = plt.figure(figsize=(13, 13))
ax1 = fig.add_subplot(221)
plot_xarray_lines(
    ax1,
    xr_bio["data"].sel(year=mask),
    colors=colors,
    name_map=name_map,
    xlabel="",
    ylabel=r"Mha yr$^{-1}$",
    title="Biodiversity restoration",
)

ax2 = fig.add_subplot(222)
plot_xarray_lines(
    ax2,
    xr_bio_cost["data"].sel(year=mask),
    colors=colors,
    name_map=name_map,
    xlabel="",
    ylabel=r"MAU\$ yr$^{-1}$",
    title="Biodiversity restoration cost",
)

ax3 = fig.add_subplot(223)
plot_xarray_lines(
    ax3,
    xr_bio_price_avg["data"].sel(year=mask),
    colors=colors,
    name_map=name_map,
    xlabel="",
    ylabel=r"AU\$ ha$^{-1}$",
    title="Biodiversity average price",
)

ax4 = fig.add_subplot(224)
plot_lines_with_buffer(
    ax4,
    xr_bio_price_cell["data"].sel(year=mask),                 # 你的 DataArray (env_category, year, cell)
    q_low=0.90, q_high=0.94,     # buffer 范围 10%~90%
    colors=colors,
    buffer_colors=colors,
    name_map=name_map,
    xlabel="",
    ylabel=r"AU\$ ha$^{-1}$",
    title="Uniform biodiversity price",
    alpha=0.3,
    reverse_env_order=True
)

draw_legend(ax4, bbox_to_anchor=(0.9, 0.15),legend_title="Degraded areas percentage (%)")

fig.subplots_adjust(left=0.10, right=0.95, top=0.92, bottom=0.2, wspace=0.4, hspace=0.4)
plt.savefig(f"{output_dir}/3_biodiversity&cost.png", dpi=300)
plt.show()

