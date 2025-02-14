# 基础绘图和数据处理库
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyArrow, Circle
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from PIL import Image
from filelock import FileLock
from collections import defaultdict

# 空间数据处理库
import rasterio
from rasterio.plot import show
from rasterio.warp import reproject, Resampling
from rasterio.transform import from_origin
from rasterio.coords import BoundingBox
from rasterio.features import geometry_mask
from rasterio.windows import Window

# 地理空间库
import geopandas as gpd
from shapely.geometry import box

# 投影和坐标转换库
import pyproj
from pyproj import CRS

from tools.parameters import *


def show_image(image_path):
    """
    打开并显示指定路径的图片。

    Parameters:
    - image_path: str, 图片文件路径
    """
    # 打开图片
    img = Image.open(image_path)

    # 显示图片
    img.show()



def convert_tif_to_png(tif_path,output_png=None):
    """
    将带有渲染信息的 GeoTIFF 文件转换为 PNG 格式并保存。
    """
    # 打开 TIFF 文件
    with rasterio.open(tif_path) as src:
        # 假设 TIFF 是 RGB 三通道的
        r = src.read(1)  # 读取红色通道
        g = src.read(2)  # 读取绿色通道
        b = src.read(3)  # 读取蓝色通道

        # 将三个通道堆叠为 RGB 图像
        rgb_image = np.stack([r, g, b], axis=-1)

        # 将 numpy 数组转换为 PIL 图像
        pil_image = Image.fromarray(rgb_image)

        # 保存为 PNG 文件
        if output_png is None:
            output_png = tif_path.replace('.tif', '.svg')
        pil_image.save(output_png)

    # 显示图像
    # plt.imshow(rgb_image)
    plt.axis('off')  # 关闭坐标轴
    plt.show()
    print(f"PNG image saved to {output_png}")


def generate_basemap(basemap_geo_tif, basemap_rgb_tif,write_png=False):
    """
    生成带有坐标的底图，并将其保存为 RGB 格式的 GeoTIFF 文件。

    Parameters:
    - basemap_geo_tif: str, 输入的底图 GeoTIFF 文件路径，具有地理坐标信息。
    - basemap_rgb_tif: str, 输出的 RGB GeoTIFF 文件路径。
    """
    # 创建自定义颜色映射（BFE9FF 到 FFFFFF，使用 CIE LAB 色彩空间）
    colors = ['#BFE9FF', '#FFFFFF']
    cmap = LinearSegmentedColormap.from_list('custom_cmap', colors, N=256)
    lock_path = basemap_rgb_tif + ".lock"  # 创建一个与文件同名的锁文件
    with FileLock(lock_path):
        # 读取地理坐标信息的底图
        with rasterio.open(basemap_geo_tif) as src:
            basemap_data = src.read(1)  # 假设底图是单波段的
            basemap_transform = src.transform  # 获取地理变换信息
            basemap_crs = src.crs  # 获取坐标参考系统 (CRS)

            # 对数据进行渲染
            norm_data = (basemap_data - basemap_data.min()) / (basemap_data.max() - basemap_data.min())  # 归一化
            rgba_image = cmap(norm_data)  # 将数据映射为 RGBA 格式
            rgba_image = (rgba_image[:, :, :3] * 255).astype(np.uint8)  # 转换为 8 位 RGB

            # 获取栅格尺寸
            height, width = basemap_data.shape

        # 使用 `rasterio` 保存带有颜色信息的 RGB GeoTIFF
        with rasterio.open(
                basemap_rgb_tif,
                'w',
                driver='GTiff',
                height=height,
                width=width,
                count=3,  # RGB 三通道
                dtype=rgba_image.dtype,
                crs=basemap_crs,
                transform=basemap_transform
        ) as dst:
            # 写入 RGB 通道数据
            for i in range(3):
                dst.write(rgba_image[:, :, i], i + 1)
        if write_png:
            convert_tif_to_png(basemap_rgb_tif)

    # print(f"Basemap with coordinates saved to {basemap_rgb_tif}")


def reproject_overlay(overlay_tif, basemap_tif, output_tif, background_value=-9999,write_png=False):
    """
    将 overlay_tif 叠加图重投影到与 basemap_tif 相同的 CRS，并保存为带有背景值的 GeoTIFF 文件。

    参数:
    - overlay_tif: 叠加图的文件路径
    - basemap_tif: 底图的文件路径
    - output_tif: 输出文件的路径
    - background_value: 设置的背景值，默认为 -9999
    """
    # 1. 读取叠加图信息
    with rasterio.open(overlay_tif) as overlay_src:
        overlay_image = overlay_src.read(1)  # 读取单波段
        overlay_transform = overlay_src.transform  # 获取叠加图的地理变换信息
        overlay_crs = overlay_src.crs  # 获取叠加图的 CRS

    # 2. 读取底图信息
    with rasterio.open(basemap_tif) as basemap_src:
        basemap_transform = basemap_src.transform  # 获取底图的地理变换信息
        basemap_crs = basemap_src.crs  # 获取底图的 CRS
        basemap_shape = basemap_src.shape  # 获取底图的形状（高度、宽度）

    # 3. 获取源文件的 nodata 值，并确保有合适的背景值
    with rasterio.open(overlay_tif) as overlay_src:
        background_value_overlay = overlay_src.nodata  # 获取源图像的背景值
        if background_value_overlay is None:
            background_value_overlay = background_value  # 如果未定义背景值，使用目标背景值作为默认

    # 4. 创建填充了目标背景值的数组
    reprojected_image = np.full(basemap_shape, background_value, dtype=overlay_image.dtype)

    # 5. 执行重投影，确保传递源和目标的 nodata 值
    reproject(
        source=overlay_image,
        destination=reprojected_image,
        src_transform=overlay_transform,
        src_crs=overlay_crs,
        dst_transform=basemap_transform,
        dst_crs=basemap_crs,
        resampling=Resampling.nearest,
        src_nodata=background_value_overlay,  # 源图像的背景值
        dst_nodata=background_value  # 目标图像的背景值
    )

    # 6. 保存结果为 GeoTIFF 文件
    with rasterio.open(
            output_tif,
            'w',
            driver='GTiff',
            height=reprojected_image.shape[0],
            width=reprojected_image.shape[1],
            count=1,  # 单波段
            dtype=reprojected_image.dtype,
            crs=basemap_crs,
            transform=basemap_transform,
    ) as dst:
        dst.write(reprojected_image, 1)  # 写入单波段图像

    # 7. 显示重投影后的叠加图
    if write_png:
        plt.imshow(reprojected_image, cmap='jet')
        plt.savefig(output_tif.replace('.tif', '.svg'), dpi=300)
        plt.show()
        print(f"Reprojected overlay saved to {output_tif}")


def assign_colors_to_overlay(overlay_reproject_geo_tif, colors_sheet, overlay_reproject_rgb_tif, background_value=-9999,write_png=False):
    """
    根据唯一值设置颜色，并将叠加图保存为带颜色和坐标的 GeoTIFF 文件。

    参数:
    - overlay_reproject_geo_tif: str, 重投影后叠加图的文件路径（带有坐标的 GeoTIFF 文件）。
    - colors_csv: str, 颜色映射 CSV 文件路径。
    - overlay_reproject_rgb_tif: str, 输出的带颜色的 GeoTIFF 文件路径。
    - background_value: int or float, 表示背景的值，将忽略该值的渲染。默认值为 -9999。
    """
    # 读取叠加的栅格数据
    with rasterio.open(overlay_reproject_geo_tif) as overlay_src:
        overlay_data = overlay_src.read(1)
        overlay_transform = overlay_src.transform
        overlay_crs = overlay_src.crs

    # 读取颜色映射
    color_mapping = pd.read_excel('tools/land use colors.xlsx',sheet_name=colors_sheet)

    # 获取 lu_code 和 lu_color_HEX
    lu_codes = color_mapping['code'].values
    lu_colors = color_mapping['color'].values

    # 创建颜色映射字典，确保每个 lu_code 对应一个颜色
    color_dict = {code: color for code, color in zip(lu_codes, lu_colors)}

    # 检查 overlay_data 中的值是否都能映射到 color_dict，并忽略背景值
    unique_values = set(np.unique(overlay_data))
    unmapped_values = unique_values - set(color_dict.keys()) - {background_value}
    for value in unmapped_values:
        overlay_data[overlay_data == value] = -2
        # raise ValueError(f"Some lu_codes in the overlay_data do not have a corresponding color: {unmapped_values}")

    # 创建用于绘图的 RGB 图像
    rgb_image = np.zeros((overlay_data.shape[0], overlay_data.shape[1], 3), dtype=np.uint8)

    # 遍历颜色字典，将 overlay_data 的值与颜色进行映射
    for lu_code, color_hex in color_dict.items():
        # 跳过背景值
        if lu_code == background_value:
            continue

        # 将十六进制颜色转换为 RGB
        color_rgb = [int(color_hex[i:i + 2], 16) for i in (1, 3, 5)]

        # 将 overlay_data 中对应 lu_code 的像素设置为 RGB 颜色
        rgb_image[overlay_data == lu_code] = color_rgb

    # 保存为带颜色和坐标的 GeoTIFF 文件
    with rasterio.open(
            overlay_reproject_rgb_tif, 'w',
            driver='GTiff',
            height=rgb_image.shape[0],
            width=rgb_image.shape[1],
            count=3,
            dtype='uint8',
            crs=overlay_crs,
            transform=overlay_transform
    ) as dst:
        dst.write(rgb_image[:, :, 0], 1)  # 写入 R 通道
        dst.write(rgb_image[:, :, 1], 2)  # 写入 G 通道
        dst.write(rgb_image[:, :, 2], 3)  # 写入 B 通道

    # 调用一个外部函数将 tif 转换为 png（假设该函数已经定义）
    if write_png:
        convert_tif_to_png(overlay_reproject_rgb_tif)

def overlay_map(basemap_rgb_tif, overlay_reproject_rgb_tif, result_tif,write_png=False):
    """
    合并叠加图和底图，保留 overlay_reproject_rgb_tif 非背景部分，使用 basemap_rgb_tif 作为背景，并保存为带有坐标信息的 GeoTIFF 文件。

    Parameters:
    - basemap_rgb_tif: str, 底图文件路径 (具有 RGB 波段)
    - overlay_reproject_rgb_tif: str, 重投影后的叠加图文件路径 (具有 RGB 波段)
    - result_tif: str, 输出的 GeoTIFF 文件路径
    """

    # 读取底图（RGB 三波段）
    with rasterio.open(basemap_rgb_tif) as basemap_src:
        basemap_image = np.stack([basemap_src.read(i) for i in range(1, 4)], axis=-1)  # 读取 RGB 波段
        basemap_transform = basemap_src.transform  # 获取地理变换信息
        basemap_crs = basemap_src.crs  # 获取坐标参考系统 (CRS)

    # 读取重投影后的叠加图（RGB 三波段）
    with rasterio.open(overlay_reproject_rgb_tif) as overlay_src:
        overlay_image = np.stack([overlay_src.read(i) for i in range(1, 4)], axis=-1)  # 读取 RGB 波段

    # 创建合并后的 RGBA 图像（4个通道：RGB + Alpha）
    combined_image = np.zeros((basemap_image.shape[0], basemap_image.shape[1], 4), dtype=np.uint8)

    # 创建透明度掩膜：overlay_reproject_rgb_tif 中非透明部分保留，背景部分使用底图
    mask = np.all(overlay_image == 0, axis=-1)  # 假设 RGB 值全为 0 表示背景

    # 对于非背景部分，使用叠加图的 RGB
    combined_image[~mask, :3] = overlay_image[~mask]
    combined_image[~mask, 3] = 255  # 非背景部分设置为不透明

    # 对于背景部分，使用底图的 RGB
    combined_image[mask, :3] = basemap_image[mask]
    combined_image[mask, 3] = 255  # 设置背景部分为不透明（如果需要透明，可以设置为 0）

    # 保存结果为带有坐标信息的 GeoTIFF 文件
    with rasterio.open(
            result_tif,
            'w',
            driver='GTiff',
            height=combined_image.shape[0],
            width=combined_image.shape[1],
            count=4,  # 4 个波段：R, G, B, A
            dtype=combined_image.dtype,
            crs=basemap_crs,
            transform=basemap_transform
    ) as dst:
        # 分别写入 RGB 和 Alpha 通道
        for i in range(4):
            dst.write(combined_image[:, :, i], i + 1)

    # 可选：转换为 PNG 文件（需要确保 `convert_tif_to_png` 已定义）
    if write_png:
        convert_tif_to_png(result_tif)

def calculate_extended_bounds(bounds, buffer_size):
    """
    在原始边界的四周留下一些空白，返回扩展后的边界。

    Parameters:
    - bounds: BoundingBox, 原始的边界框 (min_x, min_y, max_x, max_y)
    - buffer_size: float, 扩展的空白大小（以单位坐标为准，如米或度）

    Returns:
    - extended_bounds: BoundingBox, 扩展后的边界框
    """
    return BoundingBox(
        bounds.left - buffer_size,
        bounds.bottom - buffer_size,
        bounds.right + buffer_size,
        bounds.top + buffer_size
    )


def crop_map(input_tif, reference_tif, output_tif, buffer_size=1, write_png=False):
    """
    使用 `reference_tif` 的坐标范围裁剪 `input_tif`，并在四周留一些空白。

    Parameters:
    - input_tif: str, 要裁剪的输入 TIF 文件路径
    - reference_tif: str, 参考范围的 TIF 文件路径，用来确定裁剪范围
    - output_tif: str, 裁剪后的输出 TIF 文件路径
    - buffer_size: float, 扩展的空白大小

    Returns:
    - None
    """

    # 读取 reference_tif，获取坐标范围
    with rasterio.open(reference_tif) as reference_src:
        reference_bounds = reference_src.bounds  # 获取参考范围的坐标

    # 在四周留一些空白
    extended_bounds = calculate_extended_bounds(reference_bounds, buffer_size)

    # 读取 input_tif
    with rasterio.open(input_tif) as input_src:
        # 获取 input_tif 的地理变换
        input_transform = input_src.transform

        # 获取 input_tif 中对应的窗口，用于裁剪
        window = input_src.window(*extended_bounds)

        # 读取裁剪后的数据
        cropped_image = input_src.read(window=window)

        # 获取裁剪后的 transform 和分辨率
        cropped_transform = input_src.window_transform(window)
        dst_resolution = input_src.res  # 获取输入图像的分辨率

        # 将裁剪后的图像写入输出文件
        with rasterio.open(
                output_tif,
                'w',
                driver='GTiff',
                height=cropped_image.shape[1],
                width=cropped_image.shape[2],
                count=input_src.count,  # 保持波段数量不变
                dtype=cropped_image.dtype,
                crs=input_src.crs,
                transform=cropped_transform,
                res=dst_resolution  # 保持输出图像的分辨率
        ) as dst:
            dst.write(cropped_image)

    if write_png:
        convert_tif_to_png(output_tif)
    # print(f"Cropped image saved to {output_tif}")


def add_shp(tif_path, shp_path, output_tif, shapefile_color='#8F8F8F', linewidth=1, write_png=False):
    """
    将 Shapefile 叠加到 TIFF 上，并将结果保存为带有坐标信息的 GeoTIFF 文件。

    参数:
    - tif_path: str, 输入 TIFF 文件路径
    - shp_path: str, 输入 Shapefile 文件路径
    - output_tif: str, 输出 GeoTIFF 文件路径
    - shapefile_color: str, Shapefile 边界的颜色
    - linewidth: int, Shapefile 边界线的宽度
    """

    # 1. 读取 TIFF 文件
    with rasterio.open(tif_path) as src:
        # 获取 TIFF 的元数据，用于保存结果
        profile = src.profile
        profile.update(count=4)  # 添加第四个波段用于透明度 (alpha)

        # 读取底图的 RGB 数据
        basemap_image = np.stack([src.read(i) for i in range(1, 3 + 1)], axis=-1)
        basemap_extent = src.bounds
        transform = src.transform
        crs = src.crs

    # 2. 读取 Shapefile 并重投影为与 TIFF 相同的 CRS
    gdf = gpd.read_file(shp_path)
    gdf_reprojected = gdf.to_crs(crs)

    # 3. 将 Shapefile 转换为掩膜并应用到 TIFF 图像上
    out_image = np.zeros((basemap_image.shape[0], basemap_image.shape[1], 4), dtype=np.uint8)

    # 复制 RGB 数据到输出图像
    out_image[:, :, :3] = basemap_image

    # 创建掩膜，设置 Shapefile 区域为指定颜色，并设置透明度
    shapes = ((geom, 1) for geom in gdf_reprojected.geometry)
    mask = rasterio.features.geometry_mask(shapes, out_shape=basemap_image.shape[:2], transform=transform, invert=True)

    # 设置掩膜区域的颜色
    color_rgb = [int(shapefile_color[i:i + 2], 16) for i in (1, 3, 5)]
    out_image[mask, :3] = color_rgb  # RGB 颜色
    out_image[mask, 3] = 255  # Alpha 通道，完全不透明

    # 4. 将结果保存为带有坐标信息的 GeoTIFF 文件
    with rasterio.open(
            output_tif,
            'w',
            driver='GTiff',
            height=out_image.shape[0],
            width=out_image.shape[1],
            count=4,  # 4 个波段：R, G, B, A
            dtype=out_image.dtype,
            crs=crs,
            transform=transform
    ) as dst:
        # 写入 RGB 和 Alpha 通道
        for i in range(4):
            dst.write(out_image[:, :, i], i + 1)
    if write_png:
        convert_tif_to_png(output_tif)



def add_scalebar_to_ax(ax, pixel_size, length, unit='km', location=(0.1, 0.95), color="black", lw=1, fontsize=4,
                       fontname='Arial'):
    """
    手动在图上添加比例尺。

    Parameters:
    - ax: matplotlib axes
    - pixel_size: 每个像素对应的实际距离（米或公里）
    - length: 比例尺的实际长度（米或公里，取决于单位）
    - unit: 比例尺的单位，'km' 或 'm'，默认是 'km'
    - location: 比例尺的位置，按图像宽度和高度的相对位置来设置
    - color: 比例尺的颜色
    - lw: 比例尺线的宽度
    - fontsize: 比例尺文字的字体大小
    - fontname: 字体名称，默认是 'Arial'
    """
    # 计算比例尺长度对应的像素数
    length_in_pixels = length / pixel_size

    # 设置比例尺的起点和终点坐标
    x_start = location[0] * ax.get_xlim()[1]
    y_start = location[1] * ax.get_ylim()[0]
    x_end = x_start + length_in_pixels

    # 绘制比例尺
    ax.add_line(Line2D([x_start, x_end], [y_start, y_start], color=color, lw=lw))

    # 添加文本标注比例尺长度
    ax.text(x_end + 10, y_start, f"{length} {unit}", verticalalignment='center', fontsize=fontsize, color=color,
            fontname=fontname)

def add_scalebar(tif_path, output_tif, dpi=300, unit='km', length=500, location=(0.1, 0.95), lw=1, fontsize=4,
                 fontname='Arial', write_png=False):
    """
    给带有坐标信息的 TIFF 文件添加比例尺，并保存为带有比例尺的 TIFF 和 PNG 文件。
    """
    # 读取 TIFF 文件并获取信息
    with rasterio.open(tif_path) as src:
        tiff_width = src.width
        tiff_height = src.height
        figsize = (tiff_width / dpi, tiff_height / dpi)  # 根据 TIFF 大小自动计算 figsize
        # 获取 CRS 和像素大小
        crs = src.crs
        transform_matrix = src.transform
        pixel_size_x = transform_matrix[0]

        # 如果是地理坐标系，计算每度的距离（米）
        if crs.is_geographic:
            center_lat = (src.bounds.top + src.bounds.bottom) / 2
            geod = pyproj.Geod(ellps='WGS84')
            _, _, distance_m_per_degree = geod.inv(src.bounds.left, center_lat, src.bounds.right, center_lat)
            pixel_size_m = distance_m_per_degree / src.width
        else:
            # 如果 CRS 是投影坐标系，直接使用 pixel_size_x
            pixel_size_m = pixel_size_x

        # 将像素大小转换为用户选择的单位（公里或米）
        if unit == 'km':
            pixel_size = pixel_size_m / 1000
        elif unit == 'm':
            pixel_size = pixel_size_m

        # 创建图像并保留原始的 RGB 颜色
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        rgb_image = np.dstack([src.read(i) for i in range(1, 4)])  # 读取 RGB 波段
        ax.imshow(rgb_image, interpolation='none')

    # 添加比例尺
    add_scalebar_to_ax(ax, pixel_size, length, unit=unit, location=location, lw=lw, fontsize=fontsize,
                       fontname=fontname)

    # 隐藏坐标轴
    ax.set_axis_off()

    # 强制刷新绘图，确保所有元素都显示
    fig.canvas.draw()

    # 保存为带有坐标信息的 TIFF 文件
    tiff_output = output_tif
    plt.savefig(tiff_output, dpi=dpi, bbox_inches='tight', pad_inches=0, transparent=True)

    # 保存为 PNG 文件（带比例尺）如果 write_png 为 True
    png_output = output_tif.replace('.tif', '.svg')
    plt.savefig(png_output, dpi=dpi, bbox_inches='tight', pad_inches=0, transparent=True)
    # print(f"PNG image saved to {png_output}")
    # 关闭图像
    # plt.show()
    plt.close(fig)


def create_legend(colors_sheet, legend_png_name = "legend_png.svg", legend_title="Legend", legend_fontsize=8,
                  legend_title_fontsize=10, legend_ncol=1, legend_figsize=(4, 4)):
    """
    创建图例并单独保存为 PNG 文件，背景为透明。

    Parameters:
    - colors_csv: str, 颜色映射 CSV 文件路径。
    - output_legend_png: str, 输出的图例 PNG 文件路径。
    - title: str, 图例标题（默认 "Legend"）。
    - legend_fontsize: int, 图例字体大小（默认 8）。
    - legend_title_fontsize: int, 图例标题字体大小（默认 10）。
    - legend_ncol: int, 图例的列数（默认 1）。
    - legend_figsize: tuple, 图例的大小（默认 (4, 4)）。
    """

    # 读取 CSV 文件
    data = pd.read_excel('tools/land use colors.xlsx',sheet_name=colors_sheet)
    data['original_order'] = range(len(data))  # 为每行添加一个原始顺序索引

    # 根据 'lu_desc' 进行合并，并保留原始顺序
    df_merged = data.groupby('desc', as_index=False).first()

    # 按原始顺序排序
    df_merged = df_merged.sort_values('original_order')

    # 创建 patches 列表，用于显示图例
    patches = [
        mpatches.Patch(color=row['color'], label=row['desc'])
        for _, row in df_merged.iterrows()
    ]

    # 创建图例
    fig, ax = plt.subplots(figsize=legend_figsize)
    ax.set_axis_off()  # 隐藏坐标轴
    # 确保背景颜色透明
    fig.patch.set_alpha(0)  # 确保整个图的背景透明
    ax.set_facecolor('none')  # 确保坐标轴的背景透明

    # 添加图例
    plt.legend(handles=patches, title=legend_title, loc='center', ncol=legend_ncol, fontsize=legend_fontsize,
               title_fontsize=legend_title_fontsize, frameon=False)

    # 保存图例为 PNG 文件，背景透明
    plt.savefig(legend_png_name, dpi=300, bbox_inches='tight', pad_inches=0, transparent=True)
    # plt.show()
    plt.close(fig)
    # print(f"Legend saved to {output_legend_png}")


def add_legend_to_image(image_path, legend_png, output_image_path, legend_location=(0.05, 0.05), legend_scale=0.5):
    """
    在图片上叠加图例并保存结果，背景为透明。

    Parameters:
    - image_path: str, 图片文件路径。
    - legend_png: str, 图例 PNG 文件路径。
    - output_image_path: str, 输出的带有图例的图片文件路径。
    - legend_location: tuple, 图例在图片上的位置（以图片宽度和高度的比例表示），默认 (0.05, 0.05)。
    - legend_scale: float, 图例缩放比例，默认 1.0（即保持原尺寸）。
    """
    # 打开原图和图例
    image = Image.open(image_path).convert("RGBA")  # 确保图像为 RGBA 模式
    legend = Image.open(legend_png).convert("RGBA")  # 确保图例为 RGBA 模式

    # 获取图片和图例的尺寸
    image_width, image_height = image.size
    legend_width, legend_height = legend.size

    # 缩放图例
    new_legend_width = int(legend_width * legend_scale)
    new_legend_height = int(legend_height * legend_scale)
    legend = legend.resize((new_legend_width, new_legend_height), Image.Resampling.LANCZOS)

    # 计算图例的位置
    x_pos = int(legend_location[0] * image_width)
    y_pos = int(legend_location[1] * image_height)

    # 扩展图片的尺寸以适应图例的超出部分
    new_image_width = max(image_width, x_pos + new_legend_width)
    new_image_height = max(image_height, y_pos + new_legend_height)

    # 创建一个新的大图片，背景为透明（"RGBA" 模式，(255, 255, 255, 0) 代表透明）
    new_image = Image.new("RGBA", (new_image_width, new_image_height), (255, 255, 255, 0))
    new_image.paste(image, (0, 0), image)  # 将原图粘贴到新图上

    # 在新图片上叠加图例
    new_image.paste(legend, (x_pos, y_pos), legend)

    # 保存新的图片，背景保持透明
    new_image.save(output_image_path, format="PNG")
    new_image.show()

    # print(f"Image with legend saved to {output_image_path}")


def add_north_arrow_to_png(png_path, output_png, arrow_image_path, location=(0.9, 0.9), zoom=0.1):
    """
    给 PNG 图片添加指北针。

    Parameters:
    - png_path: str, 原始 PNG 图片路径
    - output_png: str, 输出 PNG 文件路径
    - arrow_image_path: str, 指北针图标的路径
    - location: tuple, 指北针的位置，按图像宽度和高度的相对位置来设置
    - zoom: float, 指北针图标的缩放大小，默认 0.1
    """
    # 读取 PNG 图像
    img = Image.open(png_path)

    # 创建图像的 Matplotlib 画布
    fig, ax = plt.subplots()
    ax.imshow(img)
    ax.set_axis_off()

    # 加载指北针图标
    arrow_img = plt.imread(arrow_image_path)

    # 创建 OffsetImage 对象，并设置缩放比例
    imagebox = OffsetImage(arrow_img, zoom=zoom)

    # 设置指北针位置
    ab = AnnotationBbox(imagebox, location, xycoords='axes fraction', frameon=False)

    # 将指北针添加到图像上
    ax.add_artist(ab)

    # 保存结果为 PNG 文件
    plt.savefig(output_png, bbox_inches='tight', pad_inches=0, transparent=True,dpi=300)
    # plt.show()
    plt.close(fig)

    # print(f"Image with north arrow saved to {output_png}")


def concatenate_images(image_files, output_image, rows=3, cols=3):
    """
    将多张图片拼接成一个网格，保持原始大小，背景透明。

    Parameters:
    - image_files: List[str], 要拼接的图片文件路径列表。
    - output_image: str, 输出的拼接图片文件路径。
    - rows: int, 网格的行数，默认 3。
    - cols: int, 网格的列数，默认 3。
    """
    # 打开所有图片，并确保每张图片使用 RGBA 模式（支持透明背景）
    images = [Image.open(image_file).convert("RGBA") for image_file in image_files]

    # 获取每张图片的宽度和高度
    widths = [img.width for img in images]
    heights = [img.height for img in images]

    # 计算网格的最大宽度和总高度
    max_width_per_col = [max(widths[i::cols]) for i in range(cols)]
    total_width = sum(max_width_per_col)
    total_height = sum(max(heights[row * cols:(row + 1) * cols]) for row in range(rows))

    # 创建一个空白的透明画布，用来放置拼接后的图片
    new_img = Image.new('RGBA', (total_width, total_height), (255, 255, 255, 0))

    # 逐行拼接图片
    y_offset = 0
    for row in range(rows):
        x_offset = 0
        row_height = max(heights[row * cols:(row + 1) * cols])
        for col in range(cols):
            img_index = row * cols + col
            if img_index >= len(images):
                break
            img = images[img_index]

            # 粘贴图片到新画布
            new_img.paste(img, (x_offset, y_offset), img)
            x_offset += max_width_per_col[col]
        y_offset += row_height

    # 保存拼接后的图片
    new_img.save(output_image)




