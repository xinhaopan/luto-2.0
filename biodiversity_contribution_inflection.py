import os
import pandas as pd
import numpy as np
import xarray as xr
import luto.settings as settings

INPUT_DIR='input'
bio_values  = xr.open_dataarray(os.path.join(INPUT_DIR, 'GBF2_conserve_priority.nc')).sel(ssp=f'ssp{settings.SSP}').compute().values
area_values = pd.read_hdf(os.path.join(INPUT_DIR, "real_area.h5")).to_numpy()

# === 1. 计算每个像元的生物多样性贡献值 ===
bio_contribution = bio_values * area_values

# === 2. 构建 DataFrame 方便聚合 ===
df = pd.DataFrame({
    'biodiversity': bio_values,
    'area': area_values,
    'bio_contribution': bio_contribution
})

# === 3. 按 biodiversity 分组聚合：面积和生物多样性贡献 ===
area_by_biodiv = (
    df.groupby('biodiversity', as_index=False)
      .agg({'area': 'sum', 'bio_contribution': 'sum'})
      .sort_values(by='biodiversity', ascending=False)
      .reset_index(drop=True)
)

# === 4. 计算面积占比和累计面积 ===
total_area = area_by_biodiv['area'].sum()
area_by_biodiv['area_percent'] = area_by_biodiv['area'] / total_area * 100
area_by_biodiv['cumulative_area'] = area_by_biodiv['area'].cumsum()
area_by_biodiv['cumulative_area_percent'] = area_by_biodiv['cumulative_area'] / total_area * 100

# === 5. 计算生物多样性贡献累计百分比 ===
total_biodiv = area_by_biodiv['bio_contribution'].sum()
area_by_biodiv['cumulative_biodiv_contribution'] = area_by_biodiv['bio_contribution'].cumsum() / total_biodiv

# === 6. 查看结果 ===
print(area_by_biodiv.head())
# area_by_biodiv.to_csv("area_by_biodiv_with_contribution.csv", index=False)


import numpy as np
import matplotlib.pyplot as plt

# === 1. 提取数据 ===
x = area_by_biodiv['cumulative_area_percent'].values
y = area_by_biodiv['cumulative_biodiv_contribution'].values

# === 2. 计算导数找拐点 ===
dy_dx = np.gradient(y, x)
d2y_dx2 = np.gradient(dy_dx, x)
inflection_index = np.argmax(np.abs(d2y_dx2))
inflection_x = x[inflection_index]
inflection_y = y[inflection_index]

# === 3. 美化图形（符合期刊图风格）===
plt.figure(figsize=(6, 5))  # 可根据论文图大小调整

# 画主线
plt.plot(x, y * 100, color='black', linewidth=1.5, label='Biodiversity Contribution')  # y转为百分比

# 拐点标记
plt.scatter(inflection_x, inflection_y * 100, color='red', s=40, zorder=5)
# plt.text(inflection_x + 1, inflection_y * 100,
#          f'Inflection\n({inflection_x:.1f}%, {inflection_y*100:.1f}%)',
#          fontsize=10, fontname='Arial', color='black')

# 设置坐标轴样式
plt.xlim(0, 100)
plt.ylim(0, 100)
plt.xticks([0, 25, 50, 75, 100], labels=['0', '25', '50', '75', '100'], fontname='Arial')
plt.yticks([0, 25, 50, 75, 100], labels=['0', '25', '50', '75', '100'], fontname='Arial')

# 坐标轴标题
plt.xlabel('Cumulative Area (%)', fontname='Arial', fontsize=12)
plt.ylabel('Cumulative Biodiversity Contribution (%)', fontname='Arial', fontsize=12)

# 去除网格
plt.grid(False)

# 刻度朝外
plt.tick_params(axis='both', direction='out', length=5, width=1)

# 边框美化
for spine in plt.gca().spines.values():
    spine.set_linewidth(1)
    spine.set_color('black')

plt.tight_layout()
plt.savefig("biodiversity_curve_inflection.png", dpi=300)  # 可选保存为论文用图
plt.show()

# 打印拐点位置
print(f"拐点位置：面积累计约 {inflection_x:.2f}%，贡献累计约 {inflection_y:.4f}")