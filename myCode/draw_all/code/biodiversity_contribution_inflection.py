import os
import pandas as pd
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import luto.settings as settings

INPUT_DIR = '../../../input'

# === 1. 加载数据 ===
bio_values = xr.open_dataarray(os.path.join(INPUT_DIR, 'GBF2_conserve_priority.nc')).sel(ssp=f'ssp{settings.SSP}').compute().values
area_values = pd.read_hdf(os.path.join(INPUT_DIR, "real_area.h5")).to_numpy()

# === 2. 构建 DataFrame ===
df = pd.DataFrame({
    'biodiversity': bio_values.flatten(),
    'area': area_values.flatten(),
})
df['bio_contribution'] = df['biodiversity'] * df['area']

# === 3. 按像元从高到低排序，不聚合 ===
df_sorted = df.sort_values(by='biodiversity', ascending=False).reset_index(drop=True)

# === 4. 计算累计面积和生物多样性贡献 ===
total_area = df_sorted['area'].sum()
total_contribution = df_sorted['bio_contribution'].sum()
df_sorted['cumulative_area'] = df_sorted['area'].cumsum()
df_sorted['cumulative_area_percent'] = df_sorted['cumulative_area'] / total_area * 100
df_sorted['cumulative_biodiv_contribution'] = df_sorted['bio_contribution'].cumsum() / total_contribution * 100


# === 额外标注面积为 50% 的点 ===
x = df_sorted['cumulative_area_percent'].values
y = df_sorted['cumulative_biodiv_contribution'].values

x_label = 50
# 插值计算对应 y 值（生物多样性贡献）
y_label = np.interp(x_label, x, y)

plt.scatter(x_label, y_label, color='red', s=50, zorder=5)
plt.text(x_label + 2, y_label-1, f'Area: {x_label}%, biodiversity: {y_label:.1f}%', fontsize=10, fontname='Arial', color='black')
plt.plot(x, y, color='black', linewidth=1.5, label='Cumulative Biodiversity Contribution')
# 图形美化
plt.xlabel('Cumulative Area', fontname='Arial', fontsize=10)
plt.ylabel('Cumulative Biodiversity Contribution', fontname='Arial', fontsize=10)
plt.xlim(0, 100)
plt.ylim(0, 100)
plt.xticks([0, 25, 50, 75, 100], ['0%', '25%', '50%', '75%', '100%'], fontname='Arial')
plt.yticks([0, 25, 50, 75, 100], ['0%', '25%', '50%', '75%', '100%'], fontname='Arial')
plt.tick_params(axis='both', direction='out', length=5, width=1)
for spine in plt.gca().spines.values():
    spine.set_linewidth(1)
    spine.set_color('black')
plt.grid(False)
plt.tight_layout()
plt.legend(frameon=False, fontsize=10, prop={'family': 'Arial'})
plt.savefig("../output/biodiversity_curve.png", dpi=300)
plt.show()
