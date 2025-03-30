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

from openpyxl import load_workbook

def save_result_to_excel(result_df, excel_path, sheet_name):
    if os.path.exists(excel_path):
        with pd.ExcelWriter(excel_path, mode='a', engine='openpyxl', if_sheet_exists='replace') as writer:
            result_df.to_excel(writer, sheet_name=sheet_name)
    else:
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            result_df.to_excel(writer, sheet_name=sheet_name)



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
    # print(tif_path, weighted_lon, weighted_lat)
    return weighted_lon, weighted_lat

def plot_land_use_shift(input_file, cal_excel=True):
    path = get_path(input_file)
    pattern = re.compile(r'(?<!Non-)Ag_LU.*\.tif{1,2}$', re.IGNORECASE)
    color_df = pd.read_excel('tools/land use colors.xlsx', sheet_name='ag')
    color_mapping = dict(zip(color_df['desc'], color_df['color']))

    # 初始化
    names, result_df, tuples, years = None, None, None, range(2010, 2051, 5)

    if cal_excel:
        use_parallel = True
        folder_path_2050 = os.path.join(path, "out_2050", "lucc_separate")
        files = [f for f in os.listdir(folder_path_2050) if pattern.match(f)]
        names = [f.split("_")[3] for f in files]

        # 经纬度数据框
        tuples = [(name, coord) for name in names for coord in ['Lon', 'Lat']]
        columns = pd.MultiIndex.from_tuples(tuples, names=["Land Use", "Coordinate"])
        df = pd.DataFrame(index=range(2010, 2051), columns=columns)

        for year in range(2010, 2051):
            folder_path = os.path.join(path, f"out_{year}", "lucc_separate")
            files = [f for f in os.listdir(folder_path) if pattern.match(f) and "mercator" not in f]
            if use_parallel:
                results = Parallel(n_jobs=-1)(delayed(get_lon_lat)(os.path.join(folder_path, f)) for f in files)
            else:
                results = [get_lon_lat(os.path.join(folder_path, f)) for f in files]
            for name, (weighted_lon, weighted_lat) in zip(names, results):
                df.loc[year, (name, 'Lon')] = weighted_lon
                df.loc[year, (name, 'Lat')] = weighted_lat

        # 坐标转换
        with rasterio.open(os.path.join(folder_path_2050, files[0])) as dataset:
            crs_from = dataset.crs
        crs_to = CRS.from_epsg(3577)
        transformer = Transformer.from_crs(crs_from, crs_to, always_xy=True)

        # 计算距离和角度
        tuples = [(land_use, metric) for land_use in names for metric in ['Distance (km)', 'Angle (degrees)']]
        result_df = pd.DataFrame(index=years, columns=pd.MultiIndex.from_tuples(tuples, names=["Land Use", "Metric"]))

        for land_use in names:
            origin_lon, origin_lat = df.loc[2010, (land_use, 'Lon')], df.loc[2010, (land_use, 'Lat')]
            origin_x, origin_y = transformer.transform(origin_lon, origin_lat)
            for year in years:
                curr_lon, curr_lat = df.loc[year, (land_use, 'Lon')], df.loc[year, (land_use, 'Lat')]
                curr_x, curr_y = transformer.transform(curr_lon, curr_lat)
                distance = np.sqrt((curr_x - origin_x)**2 + (curr_y - origin_y)**2) / 1000.0
                angle_deg = (np.degrees(np.arctan2(curr_x - origin_x, curr_y - origin_y)) + 360) % 360
                result_df.loc[year, (land_use, 'Distance (km)')] = round(distance, 3)
                result_df.loc[year, (land_use, 'Angle (degrees)')] = round(angle_deg, 2)

        excel_path = "../output/12_land_use_movement_all.xlsx"
        save_result_to_excel(result_df, excel_path, input_file)

    else:
        result_df = pd.read_excel("../output/12_land_use_movement_all.xlsx", sheet_name=input_file, header=[0, 1], index_col=0)
        names = result_df.columns.levels[0]

    # --- 绘图 ---
    plt.rcParams.update({'font.size': 10, 'font.family': 'Arial'})
    fig, ax = plt.subplots(figsize=(8, 6), subplot_kw={'projection': 'polar'})

    for land_use in names:
        distances = result_df[(land_use, 'Distance (km)')].astype(float).values
        angles_rad = np.radians(result_df[(land_use, 'Angle (degrees)')].astype(float).values)
        color = color_mapping.get(land_use, '#C8C8C8')
        ax.plot(angles_rad, distances, marker='o', label=land_use, color=color, markersize=2)

    # 极坐标设置
    angles_deg = np.arange(0, 360, 45)
    labels = ['North (km)', 'Northeast', 'East', 'Southeast',
              'South', 'Southwest', 'West', 'Northwest']
    offset_angles = [45, 135, 225, 315]
    masked_labels = [label if angle not in offset_angles else "" for angle, label in zip(angles_deg, labels)]

    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.set_xticks(np.radians(angles_deg))
    ax.set_xticklabels(masked_labels, fontname='Arial', fontsize=10)
    ax.set_yticks([0, 200, 400, 600, 800, 1000, 1200, 1400])

    for angle_deg in offset_angles:
        angle_rad = np.radians(angle_deg)
        label = labels[angles_deg.tolist().index(angle_deg)]
        ax.text(angle_rad, ax.get_rmax() * 1.12, label, ha='center', va='center', fontsize=10, fontname='Arial')

    ax.set_rlabel_position(180) # y 轴的标签（即径向标签）移动
    ax.grid(True, linestyle='--', alpha=0.7)

    # 保存图像
    output_file = f"../output/12_{input_file}_move"
    fig.savefig(f'{output_file}.pdf', dpi=300, bbox_inches='tight')
    save_figure(fig, output_file)
    plt.show()


Parallel(n_jobs=-1)(
    delayed(plot_land_use_shift)(input_file, True) for input_file in input_files
)