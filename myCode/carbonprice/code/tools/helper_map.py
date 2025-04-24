import numpy as np
import os
import rasterio
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from sklearn.preprocessing import MinMaxScaler
from PIL import Image
import pandas as pd

from .tools import get_path



def restore_npy_to_map(arr1_path, proj_file, output_tif, fill_value=np.nan, shift=0, dtype=rasterio.float32):
    with rasterio.open(proj_file) as src:
        mask2D = src.read(1) > 0
        transform = src.transform
        crs = src.crs
        profile = src.profile.copy()
        shape = src.shape

    nonzeroes = np.where(mask2D)
    lumap = np.load(arr1_path)

    if lumap.ndim != 1:
        raise ValueError(f"{arr1_path} 中的数组不是一维的")
    if len(lumap) != len(nonzeroes[0]):
        raise ValueError("lumap 的长度与 proj_file 中的有效像元数量不一致")

    themap = np.full(shape, fill_value=fill_value, dtype=float)
    themap[nonzeroes] = lumap + shift

    profile.update({
        'dtype': dtype,
        'count': 1,
        'compress': 'lzw',
        'nodata': fill_value
    })
    with rasterio.open(output_tif, 'w', **profile) as dst:
        dst.write(themap.astype(dtype), 1)

    print(f"✅ 已保存为 GeoTIFF: {os.path.abspath(output_tif)}")
    return output_tif  # 返回输出路径

def clip_outliers(arr, quantile=0.005):
    """
    对数组进行双侧百分位裁剪，去除极端值（默认去除 0.5% 和 99.5%）。

    参数:
        arr (ndarray): 输入数组
        quantile (float): 裁剪比例（默认 0.005 表示 0.5%）

    返回:
        ndarray: 裁剪后的数组
    """
    low_val, high_val = np.nanquantile(arr, [quantile, 1 - quantile])
    return np.clip(arr, low_val, high_val)


def plot_bivariate_rgb_map(
    input_file,
    arr_path1,
    arr_path2,
    output_png='bivariate_map_5x5.png',
    proj_file='ammap_2050.tiff',
    show=True,
    dpi=300,
    n_bins=5,
    quantile=0.005
):
    # 构造路径
    path_name = get_path(input_file)
    arr1_path = os.path.join(path_name, 'data_for_carbon_price', arr_path1)
    arr2_path = os.path.join(path_name, 'data_for_carbon_price', arr_path2)
    proj_path = os.path.join(path_name, 'out_2050', proj_file)

    # 还原为 tif
    carbon_tif = arr1_path.replace('.npy', '_restored.tif')
    biodiv_tif = arr2_path.replace('.npy', '_restored.tif')
    restore_npy_to_map(arr1_path, proj_path, carbon_tif)
    restore_npy_to_map(arr2_path, proj_path, biodiv_tif)

    # 读取 .tif 数据
    with rasterio.open(carbon_tif) as src1:
        carbon = src1.read(1)
        mask = src1.read_masks(1) > 0
    with rasterio.open(biodiv_tif) as src2:
        biodiv = src2.read(1)

    # 去除无效区域
    carbon[~mask] = np.nan
    biodiv[~mask] = np.nan

    # 剔除极端值
    carbon = clip_outliers(carbon, quantile)
    biodiv = clip_outliers(biodiv, quantile)

    # 标准化
    carbon_scaled = np.full(carbon.shape, np.nan)
    biodiv_scaled = np.full(biodiv.shape, np.nan)
    valid_carbon = ~np.isnan(carbon)
    valid_biodiv = ~np.isnan(biodiv)
    carbon_scaled[valid_carbon] = MinMaxScaler().fit_transform(carbon[valid_carbon].reshape(-1, 1)).flatten()
    biodiv_scaled[valid_biodiv] = MinMaxScaler().fit_transform(biodiv[valid_biodiv].reshape(-1, 1)).flatten()

    # 分箱
    bins = np.linspace(0, 1, n_bins + 1)
    carbon_bin = np.clip(np.digitize(carbon_scaled, bins) - 1, 0, n_bins - 1)
    biodiv_bin = np.clip(np.digitize(biodiv_scaled, bins) - 1, 0, n_bins - 1)

    # 统计分布（行=bio，列=carbon）
    joint_counts = np.zeros((n_bins, n_bins), dtype=int)
    for i in range(n_bins):
        for j in range(n_bins):
            joint_counts[i, j] = np.sum((biodiv_bin == i) & (carbon_bin == j))
    df = pd.DataFrame(joint_counts,
                      index=[f'{arr_path2.split("_")[0]} {i}' for i in reversed(range(n_bins))],
                      columns=[f'{arr_path1.split("_")[0]} {j}' for j in range(n_bins)]
                      )

    print("📊 每个 bin 组合的像元数量 (bio row × carbon col):")
    print(df)

    # 颜色矩阵：行=bio（0-4），列=carbon（0-4），左下最浅，右上最深
    color_matrix_hex = [
        ['#b0b0b0', '#a89880', '#a08060', '#986840', '#900820'],
        ['#c0c0c0', '#b8a890', '#b09070', '#a87850', '#a06030'],
        ['#d0d0d0', '#c8bfb0', '#c0a090', '#b88870', '#b07050'],
        ['#e0e0e0', '#d8c8c0', '#d0b0a0', '#c89880', '#c08060'],
        ['#f0f0f0', '#e8d7d0', '#e0bfb0', '#d8a790', '#d08f70'],
    ]
    color_matrix = np.array([[to_rgb(c) for c in row] for row in color_matrix_hex])
    # 图例
    output_legend = output_png.replace('.png', '_legend.png')
    plot_bivariate_legend(color_matrix, save_path=output_legend)


    # 构建 RGB 图像
    rgb = np.zeros((carbon.shape[0], carbon.shape[1], 3))
    # for i in range(n_bins):
    #     for j in range(n_bins):
    #         mask_ij = (biodiv_bin == i) & (carbon_bin == j)
    #         rgb[mask_ij] = color_matrix[i, j]
    #
    # for i in range(n_bins):
    #     for j in range(n_bins):
    #         mask_ij = (biodiv_bin == (n_bins - 1 - i)) & (carbon_bin == j)
    #         rgb[mask_ij] = color_matrix[i, j]

    for i in range(n_bins):
        for j in range(n_bins):
            mask_ij = (biodiv_bin == i) & (carbon_bin == j)
            rgb[mask_ij] = color_matrix[n_bins - 1 - i, j]
    rgb[~mask] = np.nan

    # 显示图像（不翻转）
    if show:
        plt.figure(figsize=(10, 8))
        plt.imshow(rgb)
        plt.axis("off")
        plt.title("5×5 Bivariate Price Map (Carbon × Biodiversity)")
        plt.show()

    # 保存 PNG
    rgb_clean = np.nan_to_num(rgb, nan=1.0)  # 白色背景
    img = Image.fromarray((rgb_clean * 255).astype(np.uint8))
    img.save(output_png, dpi=(dpi, dpi))
    print(f"✅ 图像已保存：{os.path.abspath(output_png)}")


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
