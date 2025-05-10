import numpy as np
import os
import rasterio
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from sklearn.preprocessing import MinMaxScaler
from PIL import Image
import pandas as pd

from .tools import get_path, npy_to_map


def clip_outliers(arr, quantile=0.005):
    """
    对数组进行双侧百分位裁剪，去除极端值（默认去除 0.5% 和 99.5%）。

    参数:
        arr (ndarray): 输入数组
        quantile (float): 裁剪比例（默认 0.005 表示 0.5%）

    返回:
        ndarray: 裁剪后的数组
    """
    low_val, high_val = np.nanquantile(arr, [0, 1 - quantile])
    return np.clip(arr, low_val, high_val)


import os
import numpy as np
import pandas as pd
import rasterio
from rasterio.plot import show
import geopandas as gpd
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib.colors import to_rgb
from .tools import npy_to_map, get_path  # 还原 tif 的工具

def plot_bivariate_rgb_map(
    input_file,
    col_path='cp_2050_bins.npy',
    row_path='bp_2050_bins.npy',
    output_png='bivariate_map_5x5.png',
    proj_file='ammap_2050.tiff',
    show=True,
    dpi=300,
    n_bins=5,
):
    # 构造路径
    path_name = get_path(input_file)
    col_full_path = os.path.join(path_name, 'data_for_carbon_price', col_path)
    row_full_path = os.path.join(path_name, 'data_for_carbon_price', row_path)
    proj_file = os.path.join(path_name, 'out_2050', proj_file)
    # 还原为 tif
    col_tif = col_full_path.replace('.npy', '_restored.tif')
    row_tif = row_full_path.replace('.npy', '_restored.tif')

    npy_to_map(col_full_path, col_tif, proj_file)
    npy_to_map(row_full_path, row_tif, proj_file)

    # 读取 .tif 数据
    with rasterio.open(col_tif) as src1:
        col_mn_bin = src1.read(1)
        mask_row = src1.read_masks(1) > 0
        meta = src1.profile
    with rasterio.open(row_tif) as src2:
        row_mn_bin = src2.read(1)
        mask_col = src2.read_masks(1) > 0
    mask = mask_row & mask_col
    # 去除无效区域
    col_mn_bin[~mask] = np.nan
    row_mn_bin[~mask] = np.nan

    # 统计分布（行=bio，列=carbon）
    joint_counts = np.zeros((n_bins, n_bins), dtype=int)
    for i in range(n_bins):
        for j in range(n_bins):
            joint_counts[i, j] = np.sum((row_mn_bin == n_bins-1-i) & (col_mn_bin == j))
    df = pd.DataFrame(joint_counts,
                      index=[f'{row_path.split("_")[0]} {i}' for i in reversed(range(n_bins))],
                      columns=[f'{col_path.split("_")[0]} {j}' for j in range(n_bins)]
                      )

    print("📊 每个 bin 组合的像元数量 (bio row × carbon col):")
    print(df)

    # 颜色矩阵：行=bio（0-4），列=carbon（0-4），左下最浅，右上最深
    color_matrix_hex = [
        ['#0000ff', '#4000bf', '#800080', '#bf0040', '#ff0000'],
        ['#0020bf', '#40288f', '#803060', '#bf3830', '#ff4000'],
        ['#004080', '#405060', '#806040', '#bf7020', '#ff8000'],
        ['#006040', '#407830', '#809020', '#bfa710', '#ffbf00'],
        ['#008000', '#40a000', '#80c000', '#bfdf00', '#ffff00'],
    ]

    color_matrix = np.array([[to_rgb(c) for c in row] for row in color_matrix_hex])
    # 图例
    output_legend = output_png.replace('.png', '_legend.png')
    plot_bivariate_legend(color_matrix, save_path=output_legend)


    # 构建 RGB 图像
    rgb = np.zeros((row_mn_bin.shape[0], row_mn_bin.shape[1], 3))

    for i in range(n_bins):
        for j in range(n_bins):
            mask_mn = (row_mn_bin == i) & (col_mn_bin == j)
            rgb[mask_mn] = color_matrix[n_bins - 1 - i, j]
    rgb[~mask] = np.nan

    # 显示图像（不翻转）
    if show:
        plt.figure(figsize=(10, 8))
        plt.imshow(rgb)
        plt.axis("off")
        plt.title("5×5 Bivariate Price Map (Carbon price × Biodiversity price)")
        plt.show()

    # 保存 PNG
    rgb_clean = np.nan_to_num(rgb, nan=1.0)  # 白色背景
    img = Image.fromarray((rgb_clean * 255).astype(np.uint8))
    img.save(output_png, dpi=(dpi, dpi))
    img.show()
    print(f"✅ 图像已保存：{os.path.abspath(output_png)}")

    output_geotiff = output_png.replace('.png', '.tif')
    # 更新 profile：3 波段，uint8
    meta.update({
        'driver': 'GTiff',
        'count': 3,
        'dtype': 'uint8',
        'nodata': 0,
    })
    # 写出 3 波段
    with rasterio.open(output_geotiff, 'w', **meta) as dst:
        for b in range(3):
            band = (rgb_clean[:, :, b] * 255).astype(np.uint8)
            dst.write(band, b + 1)


def plot_bivariate_legend(color_matrix, labels=('Low', 'High'), figsize=(3, 3), save_path=None):
    n = color_matrix.shape[0]
    fig, ax = plt.subplots(figsize=figsize)

    for i in range(n):
        for j in range(n):
            ax.add_patch(plt.Rectangle((j, n - 1 - i), 1, 1, color=color_matrix[i, j]))
            # 将 (i=0, j=0) 画在左下：行号 i 要从上往下翻

    ax.set_xticks([0, n - 1])
    ax.set_yticks([0, n - 1])
    ax.set_xticklabels([labels[0] + '\nCarbon price', labels[1]])
    ax.set_yticklabels([labels[0] + '\nBiodiversity price', labels[1]])

    ax.set_xlim(0, n)
    ax.set_ylim(0, n)
    ax.tick_params(left=False, bottom=False, labeltop=False, labelright=False)
    ax.set_aspect('equal')

    for spine in ax.spines.values():
        spine.set_visible(False)

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ 图例已保存：{save_path}")
    plt.show()


