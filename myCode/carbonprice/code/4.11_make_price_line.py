"""
4.11_make_price_line.py
双 y 轴折线图：2025–2050 shadow price 时间序列
  左 y 轴（红 + 蓝）：Shadow carbon price under Net Zero / Both targets
  右 y 轴（绿）     ：Shadow biodiversity price
颜色、字体与 4.10_make_ternary_price_map.py 保持一致，图像尺寸相同（11×8 in）
"""

import matplotlib
matplotlib.use('Agg')
import os, sys
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

sys.path.insert(0, os.path.dirname(__file__))
import tools.config as config
from tools.helper_plot import set_plot_style

# ── 路径 ────────────────────────────────────────────────────────────────────
base_dir  = f"../../../output/{config.TASK_NAME}/carbon_price"
data_dir  = f"{base_dir}/1_draw_data"
out_dir   = f"{base_dir}/3_Paper_figure"
os.makedirs(out_dir, exist_ok=True)

# ── 颜色（与 4.10 完全一致）──────────────────────────────────────────────────
COLOR_R = (0.92, 0.18, 0.18)   # Carbon NZ   → red
COLOR_G = (0.10, 0.78, 0.28)   # Bio price   → green
COLOR_B = (0.12, 0.32, 0.95)   # Carbon Both → blue

FONT_SIZE = 12   # 与 4.10 统一

# ── 载入数据 ──────────────────────────────────────────────────────────────────
ds_c = xr.open_dataset(os.path.join(data_dir, 'xr_carbon_sol_price.nc'))
ds_b = xr.open_dataset(os.path.join(data_dir, 'xr_bio_sol_price.nc'))

# 选取 2025–2050
years = np.arange(2025, 2051)
c_nz   = ds_c['data'].sel(scenario='carbon_high_50',
                           Year=years).values
c_both = ds_c['data'].sel(scenario='Counterfactual_carbon_high_bio_50',
                           Year=years).values
b_bio  = ds_b['data'].sel(scenario='carbon_high_bio_50',
                           Year=years).values

# ── 绘图 ──────────────────────────────────────────────────────────────────────
set_plot_style(font_size=FONT_SIZE, font_family='Arial')

fig, ax_left = plt.subplots(figsize=(11, 8))
fig.patch.set_alpha(0)
ax_left.set_facecolor('none')
fig.subplots_adjust(left=0.10, right=0.88, top=0.95, bottom=0.10)

ax_right = ax_left.twinx()
ax_right.set_facecolor('none')

# ── 三条线 ────────────────────────────────────────────────────────────────────
lw = 2.0
mk = 'o'
ms = 5

line_nz,   = ax_left.plot(years, c_nz,   color=COLOR_R, lw=lw,
                           marker=mk, ms=ms, label='Shadow carbon price under Net Zero')
line_both, = ax_left.plot(years, c_both, color=COLOR_B, lw=lw,
                           marker=mk, ms=ms, label='Shadow carbon price under Both targets')
line_bio,  = ax_right.plot(years, b_bio, color=COLOR_G, lw=lw,
                            marker=mk, ms=ms, label='Shadow biodiversity price')

# ── 坐标轴样式 ────────────────────────────────────────────────────────────────
ax_left.set_xlabel('Year', fontsize=FONT_SIZE, fontfamily='Arial', fontweight='bold')
ax_left.set_ylabel('Shadow carbon price (AUD/tCO2e)',
                   fontsize=FONT_SIZE, fontfamily='Arial', fontweight='bold', color='black',
                   labelpad=8)
ax_right.set_ylabel('Shadow biodiversity price (AUD/ha)',
                    fontsize=FONT_SIZE, fontfamily='Arial', fontweight='bold', color=COLOR_G)

ax_left.tick_params(axis='both', labelsize=FONT_SIZE)
ax_right.tick_params(axis='y',   labelsize=FONT_SIZE, colors=COLOR_G)
ax_right.spines['right'].set_color(COLOR_G)
ax_right.yaxis.label.set_color(COLOR_G)

# x 轴刻度：每 5 年一个
ax_left.set_xticks(np.arange(2025, 2051, 5))
ax_left.set_xlim(2024.5, 2050.5)

# 网格
ax_left.grid(axis='y', linestyle='--', linewidth=0.5, alpha=0.4, color='gray')
ax_left.set_axisbelow(True)

# ── 统一图例（合并左右轴的线）────────────────────────────────────────────────
lines  = [line_nz, line_both, line_bio]
labels = [l.get_label() for l in lines]
legend = ax_left.legend(
    lines, labels,
    loc='upper left',
    fontsize=FONT_SIZE,
    prop={'family': 'Arial', 'size': FONT_SIZE},
    framealpha=0.8,
    edgecolor='none',
)

# 设置字体
for t in legend.get_texts():
    t.set_fontfamily('Arial')
    t.set_fontsize(FONT_SIZE)

for label in (ax_left.get_xticklabels() + ax_left.get_yticklabels()
              + ax_right.get_yticklabels()):
    label.set_fontfamily('Arial')
    label.set_fontsize(FONT_SIZE)

# ── 保存 ──────────────────────────────────────────────────────────────────────
out_path = os.path.join(out_dir, "11_Sol_price_line.png")
fig.savefig(out_path, dpi=300, bbox_inches='tight', transparent=True)
print(f"Saved → {out_path}")
plt.show()
