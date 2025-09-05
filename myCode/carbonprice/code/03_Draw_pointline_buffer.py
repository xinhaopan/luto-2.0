import xarray as xr
import tools.config as config
from matplotlib.ticker import MaxNLocator
from matplotlib.legend_handler import HandlerBase
from matplotlib.ticker import FuncFormatter

import numpy as np
import matplotlib.pyplot as plt


import seaborn as sns
sns.set_theme(style="darkgrid",font="Arial", font_scale=2.2)
plt.rcParams.update({
    "xtick.bottom": True, "ytick.left": True,   # 打开刻度
    "xtick.top": False,  "ytick.right": False,  # 需要的话也可开
    "xtick.direction": "out", "ytick.direction": "out",
    "xtick.major.size": 4, "ytick.major.size": 4,
    "xtick.major.width": 1.2, "ytick.major.width": 1.2,
})
plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 12



class CombinedLegendHandler(HandlerBase):
    """自定义图例处理器，显示 buffer+线条+marker 的组合"""

    def __init__(self, buffer_color, line_color, alpha, linestyle, marker):
        HandlerBase.__init__(self)
        self.buffer_color = buffer_color
        self.line_color = line_color
        self.alpha = alpha
        self.linestyle = linestyle
        self.marker = marker

    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        x0, y0 = handlebox.xdescent, handlebox.ydescent
        width, height = handlebox.width, handlebox.height

        # buffer 背景
        buffer_patch = plt.Rectangle(
            (x0, y0), width, height,
            facecolor=self.buffer_color,
            alpha=self.alpha,
            edgecolor="none",
            transform=handlebox.get_transform()
        )
        handlebox.add_artist(buffer_patch)

        # 中间线条
        line = plt.Line2D(
            [x0, x0+width], [y0+height/2, y0+height/2],
            color=self.line_color,
            linestyle=self.linestyle,
            linewidth=1.5,
            transform=handlebox.get_transform()
        )
        handlebox.add_artist(line)

        # 添加 marker
        if self.marker and self.marker != "None":
            marker_patch = plt.Line2D(
                [x0+width/2], [y0+height/2],
                color=self.line_color,
                marker=self.marker,
                markersize=4,
                linestyle="",
                transform=handlebox.get_transform()
            )
            handlebox.add_artist(marker_patch)

        return buffer_patch


def plot_lines_with_buffer_split(
    da,                         # DataArray, dims: (env_category, year, cell)
    q_low=0.1, q_high=0.9,      # buffer 的分位数范围
    colors=None, buffer_colors=None,
    name_map=None,
    xlabel="Year", ylabel="Value",
    alpha=0.3,
    linestyle="-", marker="o",
    figsize=(15, 10),
    legend_title="GHG targets percentage (%)"
):
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

    n_envs = len(envs)
    ncols = 2
    nrows = (n_envs + 1) // ncols

    fig, axes = plt.subplots(
        nrows=nrows, ncols=ncols, figsize=figsize, sharex=True, sharey=True
    )
    axes = np.atleast_1d(axes).ravel()

    # 保存 legend handler 信息
    legend_entries = {}
    for i, env in enumerate(envs):
        ax = axes[i]
        da_env = da.sel(env_category=env)

        low = da_env.quantile(q_low, dim="cell", skipna=True)
        high = da_env.quantile(q_high, dim="cell", skipna=True)
        line = da_env.where((da_env >= low) & (da_env <= high), drop=True).median(dim="cell")

        low_v, high_v, line_v = low.values, high.values, line.values

        # buffer
        ax.fill_between(years, low_v, high_v,
                        color=buffer_colors[i % len(buffer_colors)],
                        alpha=alpha, edgecolor="none")
        # 中位数线
        ax.plot(
            years, line_v,
            color=colors[i % len(colors)],
            linestyle=linestyle, marker=marker, markersize=4, linewidth=2,
        )
        for spine in ax.spines.values():
            spine.set_visible(True)  # 显示所有边框
            spine.set_linewidth(1.2)  # 边框粗细
            spine.set_edgecolor("black")  # 边框颜色

        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{int(x):,}"))
        ax.set_xlabel(xlabel)

        # 保存 legend 信息
        label = name_map.get(str(env), str(env))
        handler = CombinedLegendHandler(
            buffer_color=buffer_colors[i % len(buffer_colors)],
            line_color=colors[i % len(colors)],
            alpha=alpha,
            linestyle=linestyle,
            marker=marker
        )
        legend_entries[label] = handler

    # 删除多余子图
    total_slots = nrows * ncols
    if total_slots > n_envs:
        for j in range(n_envs, total_slots):
            fig.delaxes(axes[j])

    # 只保留左列中间的 y 标签
    mid_left_idx = (nrows // 2) * ncols
    for idx, ax in enumerate(axes[:n_envs]):
        if idx == mid_left_idx:
            ax.set_ylabel(ylabel)
        else:
            ax.set_ylabel("")

    # 图例放 figure 右下角（或自定义）
    fig.legend(
        legend_entries.keys(),
        legend_entries.keys(),
        handler_map=legend_entries,
        title=legend_title,
        loc="lower right", bbox_to_anchor=(0.95, 0.1),
        ncol=2, frameon=False
    )

    plt.tight_layout()
    return fig, axes

input_dir = f'../../../output/{config.TASK_NAME}/carbon_price/0_base_data/Results'
output_dir = f"../../../output/{config.TASK_NAME}/carbon_price/3_Paper_figure"


xr_carbon_price_cell = xr.open_dataset(f'{input_dir}/xr_carbon_price_scenario_cell.nc')
years = xr_carbon_price_cell.coords["year"].values
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

fig, axes = plot_lines_with_buffer_split(
    xr_carbon_price_cell["data"].sel(year=mask),
    q_low=0.95, q_high=0.99,
    colors=colors, buffer_colors=colors,
    name_map=name_map,
    xlabel="",
    ylabel=r"AU\$ CO$_2$e$^{-1}$",
    alpha=0.3,
    linestyle="-",
    marker="o",
    figsize=(15, 10)
)
plt.savefig(f"{output_dir}/4_carbon_price_scenarios_split.png", dpi=300, bbox_inches='tight')
plt.show()

xr_bio_price_cell = xr.open_dataset(f'{input_dir}/xr_bio_price_scenario_cell.nc')
years = xr_bio_price_cell.coords["year"].values
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

fig, axes = plot_lines_with_buffer_split(
    xr_bio_price_cell["data"].sel(year=mask),
    q_low=0.90, q_high=0.94,
    colors=colors, buffer_colors=colors,
    name_map=name_map,
    xlabel="",
    ylabel=r"AU\$ ha$^{-1}$",
    alpha=0.3,
    linestyle="-",
    marker="o",
    legend_title="Degraded areas percentage (%)",
    figsize=(15, 10)
)
plt.savefig(f"{output_dir}/4_bio_price_scenarios_split.png", dpi=300, bbox_inches='tight')
plt.show()