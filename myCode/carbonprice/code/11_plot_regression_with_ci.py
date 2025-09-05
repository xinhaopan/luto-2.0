import xarray as xr
import tools.config as config

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.ticker import FuncFormatter

import math
import matplotlib as mpl
# %matplotlib notebook
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

def plot_regression_with_ci(ax, x, y):
    # 去掉 NaN/inf
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    n = x.size
    if n < 3:
        raise ValueError("样本点少于 3 个，无法做线性回归和置信区间。")

    # 2) 线性回归（最小二乘）
    # y = a*x + b
    a, b = np.polyfit(x, y, 1)
    y_hat = a * x + b

    # R^2
    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

    # 3) 计算 95% 置信区间（针对回归均值的置信带）
    x_line = np.linspace(x.min(), x.max(), 200)
    y_line = a * x_line + b

    # 残差标准差
    s2 = ss_res / (n - 2)
    s = np.sqrt(s2)

    x_mean = np.mean(x)
    Sxx = np.sum((x - x_mean) ** 2)

    # t 分位数（95% 置信水平）
    try:
        from scipy import stats
        tcrit = stats.t.ppf(0.975, df=n - 2)
    except Exception:
        tcrit = 1.96  # 大样本近似

    # 回归均值的标准误
    se_mean = s * np.sqrt(1 / n + (x_line - x_mean) ** 2 / Sxx)
    ci_lo = y_line - tcrit * se_mean
    ci_hi = y_line + tcrit * se_mean

    ax.scatter(x, y, s=40, alpha=0.85, color='black')
    ax.plot(x_line, y_line, linewidth=2, label=f"y = {a:.3g}·x + {b:.3g} \n$R^2$ = {r2:.3f}", color='black')
    ax.fill_between(x_line, ci_lo, ci_hi, alpha=0.2, label="95% CI", color='gray')

    ax.set_xlabel("GHG reductions and removals (MtCO2e)")
    ax.set_ylabel("Cost (mAUD$)")
    ax.set_title("")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.legend(loc="best", title="2050")
    ax.grid(True, alpha=0.3)

    plt.show()


input_dir = f'../../../output/{config.TASK_NAME}/carbon_price/0_base_data/Results'
output_dir = f"../../../output/{config.TASK_NAME}/carbon_price/3_Paper_figure"

xr_carbon = xr.open_dataset(f'{input_dir}/xr_carbon_scenario_sum.nc') / 1e6
xr_carbon_cost = xr.open_dataset(f'{input_dir}/xr_carbon_cost_scenario_sum.nc') / 1e6


years = xr_carbon.coords["year"].values
mask = years >= 2025


x_data = years[mask]


import numpy as np
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(13, 6))
ax1 = fig.add_subplot(121)
x = np.asarray(xr_carbon["data"].sel(year=2050).values).ravel()       # GHG (MtCO2e)
y = np.asarray(xr_carbon_cost["data"].sel(year=2050).values).ravel()  # Cost (mAUD$)
plot_regression_with_ci(ax1, x, y)

# ax2 = fig.add_subplot(122)
# x = np.asarray(xr_bio["data"].sel(year=2050).values).ravel()       # GHG (MtCO2e)
# y = np.asarray(xr_bio_cost["data"].sel(year=2050).values).ravel()  # Cost (mAUD$)
# plot_regression_with_ci(ax2, x, y)
# plt.savefig(f"{output_dir}/03_cost_regression.png", dpi=300, bbox_inches='tight')
#
# xr_bio["data"].sel(year=2050), xr_bio_cost["data"].sel(year=2050)