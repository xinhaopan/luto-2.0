import xarray as xr
import tools.config as config

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.ticker import FuncFormatter

import math
# %matplotlib notebook
import matplotlib as mpl
import matplotlib
matplotlib.use("QtAgg")
mpl.rcParams["figure.facecolor"] = "white"
mpl.rcParams["axes.facecolor"] = "white"

plt.rcParams.update({
    "xtick.bottom": True, "ytick.left": True,  # 打开刻度
    "xtick.top": False, "ytick.right": False,  # 需要的话也可开
    "xtick.direction": "out", "ytick.direction": "out",
    "xtick.major.size": 4, "ytick.major.size": 4,
    "xtick.major.width": 1.2, "ytick.major.width": 1.2,
})
# 设置全局字体为Arial
plt.rcParams['font.family'] = 'Arial'

import seaborn as sns
sns.set_theme(
    style="white",
    palette=None,
    rc={
        'font.family': 'Arial',
        'font.size': 12,            # 全局默认字体大小
        'axes.titlesize': 12,       # 子图标题
        'axes.labelsize': 12,       # x/y 轴标签
        'xtick.labelsize': 12,      # 刻度标签
        'ytick.labelsize': 12,      # 刻度标签
        'legend.fontsize': 12,      # 图例文字
        'figure.titlesize': 12,     # suptitle（如果你有的话）

        "mathtext.fontset": "custom",
        "mathtext.rm":      "Arial",
        "mathtext.it":      "Arial:italic",
        "mathtext.bf":      "Arial:italic",

        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 5,   # 刻度长度可选
        "ytick.major.size": 5,
    }
)


def plot_3d_env(
        ax,
        x_years,  # 1D: year 数组（已经筛好，比如 >=2025）
        y_da,  # DataArray: dims (env_category, year)
        z_da,  # DataArray: dims (env_category, year)
        envs,  # 1D: env_category 列表/数组
        name_map=None,
        colors=None,
        xlabel="",
        ylabel="Cost (mAUD$)",
        zlabel="GHG reductions and removals (MtCO2e)",
        n_yticks=4,
        n_zticks=4,
        y_tick=None,
        legend_title="Scenario",
        legend_cols=1
):
    # 默认颜色和映射
    if colors is None:
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    if name_map is None:
        name_map = {}

    # 逐条曲线绘制（确保 y/z 都是一维，与 x_years 等长）
    for i, env in enumerate(envs):
        y = y_da.sel(env_category=env).values  # -> shape (len(x_years),)
        z = z_da.sel(env_category=env).values
        ax.plot(
            x_years, y, z,
            marker="o", linestyle="-", markersize=4, linewidth=2,
            color=colors[i % len(colors)],
            label=name_map.get(str(env), str(env))
        )

    # 轴&刻度
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel, labelpad=15)
    if n_yticks:
        ax.yaxis.set_major_locator(MaxNLocator(nbins=n_yticks))
    if n_zticks:
        ax.zaxis.set_major_locator(MaxNLocator(nbins=n_zticks))
    if y_tick:
        ax.set_yticks(y_tick)

    ax.tick_params(axis="x", pad=2)  # x 轴刻度值往外移
    ax.tick_params(axis="y", pad=2)  # y 轴也可以调整
    ax.tick_params(axis="z", pad=8)  # z 轴刻度值往外移

    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.zaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{int(x):,}"))

    ax.legend(
        title=legend_title,
        loc="lower center",  # 底部居中
        bbox_to_anchor=(0.5, -0.18),  # (x, y)，x=0.5 表示居中，y 设为负值把图例往下移
        ncol=math.ceil(len(envs) / legend_cols),  # 图例分几列，这里等于 envs 数量 → 全部在一行
        frameon=False  # 去掉边框，可选
    )

    return ax


input_dir = f'../../../output/{config.TASK_NAME}/carbon_price/0_base_data/Results'

fig = plt.figure(figsize=(14, 8), facecolor="white")

ax1 = fig.add_subplot(121, projection="3d")
ax2 = fig.add_subplot(122, projection="3d")

# 自定义图例名称 & 颜色
name_map = {
    "carbon_100": "100",
    "carbon_80":  "80",
    "carbon_60":  "60",
    "carbon_40":  "40",
    "carbon_20":  "20",
}
colors = ["#8CB266", "#F2A930", "#7575EB", "#D89CCD", "#9D1D1D"]

xr_carbon_cost = xr.open_dataset(f'{input_dir}/xr_all_carbon_cost_sum.nc') / 1e6
xr_carbon = xr.open_dataset(f'{input_dir}/xr_all_carbon_sum.nc') / 1e6
years = xr_carbon.coords["year"].values
# 只选 2025 年及之后
mask = years >= 2025
x_data = years[mask]
y_data = xr_carbon["data"].sel(year=mask)
z_data = xr_carbon_cost["data"].sel(year=mask)
envs = xr_carbon.coords["env_category"].values

plot_3d_env(ax1,
    x_data, y_data, z_data, envs,
    name_map=name_map, colors=colors,
    xlabel="", ylabel=r"GHG reductions and removals (MtCO$_2$e yr$^{-1}$)", zlabel=r"Cost (MAU\$ yr$^{-1}$)",
    n_yticks=4, n_zticks=4,legend_title="GHG target percentage (%)"
)

xr_bio_cost = xr.open_dataset(f'{input_dir}/xr_all_bio_cost_sum.nc') / 1e6
xr_bio = xr.open_dataset(f'{input_dir}/xr_all_bio_sum.nc') / 1e6
mask = years >= 2025
x_data = years[mask]
y_data = xr_bio["data"].sel(year=mask)
z_data = xr_bio_cost["data"].sel(year=mask)
envs = xr_bio.coords["env_category"].values

name_map={
        "carbon_100_bio_50": "50",
        "carbon_100_bio_40": "40",
        "carbon_100_bio_30": "30",
        "carbon_100_bio_20": "20",
        "carbon_100_bio_10": "10",
    }
plot_3d_env(ax2,
    x_data, y_data, z_data, envs,
    name_map=name_map, colors=colors,
    xlabel="", ylabel="Biodiversity restoration (Mha yr$^{-1}$)", zlabel=r"Cost (MAU\$ yr$^{-1}$)",
    n_yticks=4, n_zticks=4,legend_title="Degraded areas percentage (%)"
)
plt.subplots_adjust(left=0, right=0.85, wspace=0.1, top=0.9, bottom=0.2)

output_dir = f"../../../output/{config.TASK_NAME}/carbon_price/3_Paper_figure"
plt.savefig(f"{output_dir}/03_3D_year_env_cost.png", dpi=300)
plt.show()