import os
import re
import numpy as np
import pandas as pd
import rasterio
from joblib import Parallel, delayed
from pyproj import CRS, Transformer
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from tools.data_helper import *
from tools.plot_helper import *
from tools.parameters import *


# ----------------------- 函数定义 -----------------------
def get_lon_lat(tif_path):
    """计算面积加权经纬度。"""
    with rasterio.open(tif_path) as dataset:
        data = dataset.read(1)
        transform = dataset.transform

    valid_mask = data > 0
    values = data[valid_mask]
    rows, cols = np.where(valid_mask)

    lon, lat = rasterio.transform.xy(transform, rows, cols)
    lon = np.array(lon)
    lat = np.array(lat)

    weighted_lon = np.sum(lon * values) / np.sum(values)
    weighted_lat = np.sum(lat * values) / np.sum(values)
    return weighted_lon, weighted_lat


path = get_path(input_files[4])
pattern = re.compile(r'(?<!Non-)Ag_LU.*\.tif{1,2}$', re.IGNORECASE)
# ----------------------- 提取土地利用类型与文件 -----------------------
folder_path_2050 = os.path.join(path, "out_2050", "lucc_separate")
files = [f for f in os.listdir(folder_path_2050) if pattern.match(f)]
names = [f.split("_")[3] for f in files]

# 创建多级列索引
tuples = [(name, coord) for name in names for coord in ['Lon', 'Lat']]
columns = pd.MultiIndex.from_tuples(tuples, names=["Land Use", "Coordinate"])
df = pd.DataFrame(index=range(2010, 2051), columns=columns)

# ----------------------- 并行计算经纬度 -----------------------
for year in range(2010, 2051):
    folder_path = os.path.join(path, f"out_{year}", "lucc_separate")
    files = [f for f in os.listdir(folder_path) if pattern.match(f)]
    results = Parallel(n_jobs=-1)(delayed(get_lon_lat)(os.path.join(folder_path, f)) for f in files)

    for name, (weighted_lon, weighted_lat) in zip(names, results):
        df.loc[year, (name, 'Lon')] = weighted_lon
        df.loc[year, (name, 'Lat')] = weighted_lat

# ----------------------- 坐标转换与偏移计算 -----------------------
with rasterio.open(os.path.join(path, "out_2050", "lucc_separate", files[0])) as dataset:
    crs_from = dataset.crs

crs_to = CRS.from_epsg(3577)  # 澳大利亚等距投影
transformer = Transformer.from_crs(crs_from, crs_to, always_xy=True)

# 结果 DataFrame 初始化
years = range(2010, 2051, 5)
tuples = [(land_use, metric) for land_use in names for metric in ['Distance (km)', 'Angle (degrees)']]
result_df = pd.DataFrame(index=years, columns=pd.MultiIndex.from_tuples(tuples, names=["Land Use", "Metric"]))

for land_use in names:
    origin_lon, origin_lat = df.loc[2010, (land_use, 'Lon')], df.loc[2010, (land_use, 'Lat')]
    origin_x, origin_y = transformer.transform(origin_lon, origin_lat)

    for year in years:
        curr_lon, curr_lat = df.loc[year, (land_use, 'Lon')], df.loc[year, (land_use, 'Lat')]
        curr_x, curr_y = transformer.transform(curr_lon, curr_lat)

        distance = np.sqrt((curr_x - origin_x) ** 2 + (curr_y - origin_y) ** 2) / 1000.0
        angle_deg = (np.degrees(np.arctan2(curr_x - origin_x, curr_y - origin_y)) + 360) % 360

        result_df.loc[year, (land_use, 'Distance (km)')] = round(distance, 3)
        result_df.loc[year, (land_use, 'Angle (degrees)')] = round(angle_deg, 2)

# ----------------------- 保存结果 -----------------------
output_path = "../output/10_land_use_movement_multilevel.xlsx"
result_df.to_excel(output_path)

# ----------------------- 绘制极坐标图 -----------------------
color_df = pd.read_excel('tools/land use colors.xlsx', sheet_name='ag')
color_mapping = dict(zip(color_df['desc'], color_df['color']))

plt.rcParams.update({'font.size': 10})
fig, ax = plt.subplots(figsize=(8, 6), subplot_kw={'projection': 'polar'})

for land_use in names:
    distances = result_df[(land_use, 'Distance (km)')].values.astype(float)
    angles_rad = np.radians(result_df[(land_use, 'Angle (degrees)')].values.astype(float))
    color = color_mapping.get(land_use, '#C8C8C8')
    ax.plot(angles_rad, distances, marker='o', label=land_use, color=color)

ax.set_theta_zero_location('N')
ax.set_theta_direction(-1)
ax.set_xticks(np.radians(np.arange(0, 360, 45)))
ax.set_xticklabels(['North', 'Northeast', 'East', 'Southeast', 'South', 'Southwest', 'West', 'Northwest'])
ax.set_rlabel_position(0)
ax.tick_params(axis='x', labelsize=10)
ax.tick_params(axis='y', labelsize=10)
ax.grid(True, linestyle='--', alpha=0.7)

# 保存图像
output_file = "../output/10_land_use_movement_polar"
fig.savefig(f"{output_file}.png", dpi=300, bbox_inches='tight')

# ----------------------- 保存图例 -----------------------
handles = [
    Line2D([0], [0], color=color_mapping.get(land_use, '#C8C8C8'), marker='o', linestyle='-', linewidth=1, markersize=3)
    for land_use in names]
labels = names

fig_legend, ax_legend = plt.subplots(figsize=(6, 3))
ax_legend.axis('off')
ax_legend.legend(handles, labels, loc='center', fontsize=10, ncol=2, frameon=False)
fig_legend.savefig(f"{output_file}_legend.svg", dpi=300, bbox_inches='tight', transparent=True)

# 提取 2050 年的 Distance (km) 数据
distance_2050 = df.xs('Distance (km)', axis=1, level='Metric').loc[2050]

# 找到 2050 年偏移量最大的三个土地利用类型
top_3_land_uses_2050 = distance_2050.nlargest(3)

# 转换为 DataFrame
top_3_2050_df = pd.DataFrame({
    'Land Use': top_3_land_uses_2050.index,
    'Distance (km)': top_3_land_uses_2050.values
})

# 输出结果
print(top_3_2050_df)
