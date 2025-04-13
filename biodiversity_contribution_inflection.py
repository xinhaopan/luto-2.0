import os
import pandas as pd
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import luto.settings as settings

INPUT_DIR = 'input'

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



# # === 5A. 使用 Kneedle 算法找拐点 ===
# x_k = df_sorted['cumulative_area_percent'].values
# y_k = df_sorted['cumulative_biodiv_contribution'].values
#
# mask_k = x_k > 0
# x_k = x_k[mask_k]
# y_k = y_k[mask_k]
#
# knee_locator = KneeLocator(x_k, y_k, curve='concave', direction='increasing')
# kneedle_x = knee_locator.knee
# kneedle_y = knee_locator.knee_y
#
# # === 5B. 使用二阶导数方法找拐点 ===
# x_d = df_sorted['cumulative_area_percent'].values
# y_d = df_sorted['cumulative_biodiv_contribution'].values
#
# dy_dx = np.gradient(y_d, x_d)
# d2y_dx2 = np.gradient(dy_dx, x_d)
# inflection_index = np.argmax(np.abs(d2y_dx2))
# deriv_x = x_d[inflection_index]
# deriv_y = y_d[inflection_index]
#
# # === 5C. 使用最大梯度法找拐点 ===
# max_grad_index = np.argmax(dy_dx)
# grad_x = x_d[max_grad_index]
# grad_y = y_d[max_grad_index]
#
# # === 6. 绘图（包含三种方法的拐点）===
# plt.figure(figsize=(6, 5))
# plt.plot(x_d, y_d, color='black', linewidth=1.5, label='Biodiversity Contribution')
#
# # 拐点1：Kneedle
# if kneedle_x is not None:
#     plt.scatter(kneedle_x, kneedle_y, color='red', s=60, zorder=5, label=f'Kneedle ({kneedle_x:.1f}%, {kneedle_y:.1f}%)')
#
# # 拐点2：二阶导数
# plt.scatter(deriv_x, deriv_y, color='blue', s=60, zorder=5, label=f'Deriv ({deriv_x:.1f}%, {deriv_y:.1f}%)')
#
# # 拐点3：最大梯度
# plt.scatter(grad_x, grad_y, color='green', s=60, zorder=5, label=f'Max Gradient ({grad_x:.1f}%, {grad_y:.1f}%)')

# === 额外标注面积为 30% 的点 ===
x = df_sorted['cumulative_area_percent'].values
y = df_sorted['cumulative_biodiv_contribution'].values

x_20 = 50
# 插值计算对应 y 值（生物多样性贡献）
y_20 = np.interp(x_20, x, y)

plt.scatter(x_20, y_20, color='red', s=50, zorder=5)
plt.text(x_20 + 2, y_20-1, f'Area: {x_20}%, biodiversity: {y_20:.1f}%', fontsize=10, fontname='Arial', color='black')
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
plt.savefig("biodiversity_curve_three_methods.png", dpi=300)
plt.show()

# # === 7. 拐点输出 ===
# if kneedle_x is not None:
#     print(f"Kneedle 拐点位置：面积累计约 {kneedle_x:.2f}%，贡献累计约 {kneedle_y:.2f}%")
# print(f"二阶导数法 拐点位置：面积累计约 {deriv_x:.2f}%，贡献累计约 {deriv_y:.2f}%")
# print(f"最大梯度法 拐点位置：面积累计约 {grad_x:.2f}%，贡献累计约 {grad_y:.2f}%")