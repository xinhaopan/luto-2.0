import tools.config as config
from tools.tools import get_path

import os
import rasterio
import numpy as np
from rasterio import features
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import geopandas as gpd
import pandas as pd


def reclassify_and_calculate_proportion(input_tif, shp_path):
    """
    重分类 TIF 并按 shp 分区计算值为 1 的像元比例（包含 nodata）。

    输入:
    - input_tif: str，TIF 文件路径
    - shp_path: str，Shapefile 文件路径（自动识别 ID 字段）

    输出:
    - study_arr: numpy.ndarray，重分类后的数组
    - pd.DataFrame，列为 ['id', 'proportion_of_1']
    """
    # 读取并重分类 tif：-1 → 0，其余为 1
    with rasterio.open(input_tif) as src:
        arr = src.read(1)
        nodata = src.nodata
        transform = src.transform

        # 先记录 nodata mask
        mask = (arr == nodata)
        # 处理非nodata区
        arr2 = np.where(arr == -1, 0, 1)
        # 恢复nodata
        arr2[mask] = nodata

    # 读取 shapefile
    gdf = gpd.read_file(shp_path)

    results = []
    for i, row in gdf.iterrows():
        geom = row.geometry
        mask = features.geometry_mask([geom], transform=transform, invert=True, out_shape=arr2.shape)

        zone_values = arr2[mask]
        total = zone_values.size
        count_1 = np.count_nonzero(zone_values == 1)
        proportion = count_1 / total if total > 0 else np.nan

        results.append({'id': gdf.index[i], 'proportion_of_1': proportion})

    df_result = pd.DataFrame(results).set_index('id')
    return arr2, df_result


from joblib import Parallel, delayed
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import geometry_mask


def zonal_stats_masked(input_tif, study_arr, shp, stats='mean',
                                column_name=None, n_jobs=-1):
    """
    并行版：对 shp 分区，统计 input_tif 在 study_arr==1 部分的像元，
    返回 DataFrame，index 与 shp 一致。

    参数:
    - input_tif: str，tif 文件路径
    - study_arr: numpy.ndarray，与 tif 同尺寸，值为 1 的区域参与统计
    - shp: str 或 GeoDataFrame
    - stats: 'mean' 或 'sum'
    - column_name: 输出列名
    - n_jobs: 并行进程数，-1 表示用所有核

    返回:
    - pd.DataFrame，行索引同 shp，单列统计结果
    """
    # 载入矢量
    if isinstance(shp, str):
        gdf = gpd.read_file(shp)
    else:
        gdf = shp.copy()
    # 载入栅格
    with rasterio.open(input_tif) as src:
        img = src.read(1)
        transform = src.transform

    assert study_arr.shape == img.shape, "study_arr 尺寸必须和 tif 一致"

    def _compute(idx, geom):
        # 对单个几何做掩膜 & 统计
        mask = geometry_mask([geom], img.shape, transform, invert=True)
        zone = mask & (study_arr == 1)
        vals = img[zone]
        if vals.size == 0:
            return idx, np.nan
        if stats == 'mean':
            return idx, float(np.nanmean(vals))
        else:  # 'sum'
            return idx, float(np.nansum(vals))

    # 并行执行
    results = Parallel(n_jobs=n_jobs)(
        delayed(_compute)(i, row.geometry)
        for i, row in gdf.iterrows()
    )

    # 收集结果
    idxs, vals = zip(*results)
    col = column_name or f"{stats}_val"
    df = pd.DataFrame({col: vals}, index=idxs)
    return df


def set_plot_style(font_size=12, font_family='Arial'):
    mpl.rcParams.update({
        'font.size': font_size,
        'font.family': font_family,
        'axes.titlesize': font_size,
        'axes.labelsize': font_size,
        'xtick.labelsize': font_size,
        'ytick.labelsize': font_size,
        'legend.fontsize': font_size,
        'figure.titlesize': font_size
    })


def draw_hex_shape(
        ax,
        hex_gdf,
        value_col,
        title_name,
        unit_name,
        cmap='viridis',
        threshold=None,
        aus_border_gdf=None,  # 新增参数
        H_linewidth=1
):
    """
    只显示 'proportion_of_1' > threshold 的六边形，color 由 value_col 指定。
    可叠加澳大利亚国界线。
    如果没有负数，colorbar最左边刻度是0；否则保持原始最小值。
    """
    plot_gdf = hex_gdf.copy()
    if threshold is not None:
        plot_gdf = plot_gdf[plot_gdf['proportion_of_1'] > threshold]

    if plot_gdf.empty:
        ax.set_title(f"{title_name}\n(No hex where proportion_of_1 > {threshold})")
        ax.set_axis_off()
        return

    vmin = plot_gdf[value_col].min()
    vmax = plot_gdf[value_col].max()
    if vmin < 0:
        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    else:
        norm = mpl.colors.Normalize(vmin=0, vmax=vmax)

    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])

    # 画hex
    plot_gdf.plot(
        column=value_col,
        cmap=cmap,
        edgecolor='white',
        linewidth=H_linewidth,
        legend=False,
        ax=ax
    )

    # 画国界线
    if aus_border_gdf is not None:
        aus_border_gdf.plot(ax=ax, color='grey', linewidth=1.0, zorder=10)

    ax.set_title(title_name, y=0.95)
    ax.set_axis_off()

    cax = inset_axes(
        ax,
        width="55%",
        height="8%",
        loc='lower center',
        borderpad=1,
        bbox_to_anchor=(-0.12, 0.07, 1, 0.8),
        bbox_transform=ax.transAxes,
    )
    cbar = plt.colorbar(
        sm, cax=cax, orientation='horizontal',
        extend='both', extendfrac=0.1, extendrect=False
    )
    cbar.outline.set_visible(False)
    cbar.ax.xaxis.set_label_position('top')
    cbar.locator = mticker.MaxNLocator(nbins=5)
    cbar.update_ticks()
    cbar.set_label(unit_name)

    # 保证最左侧tick是0（无负数时）
    if vmin >= 0:
        ticks = cbar.get_ticks()
        # 如果已有0就不用加，如果最左不是0则手动加0
        if ticks[0] > 0:
            ticks = [0] + list(ticks)
            cbar.set_ticks(ticks)
        elif ticks[0] < 0:
            # 理论上不会进入这里，但以防某些float误差
            ticks = [0] + [t for t in ticks if t > 0]
            cbar.set_ticks(ticks)

    plt.subplots_adjust(wspace=-0.05, hspace=-0.05)


def set_extreme_quantile_nan(df, cols, quantile=0.05, masks=None):
    """
    - masks: dict, 只要提供了，就直接作用于 DataFrame，不管key是否在cols里；
    - cols: 只对这些列做quantile裁剪；
    """
    if isinstance(cols, str):
        cols = [cols]
    if masks is None:
        masks = {}

    # 1️⃣ 应用所有 mask，不管是否在 cols 里
    for mask_col, mask in masks.items():
        if mask_col in df.columns:
            df.loc[mask, mask_col] = np.nan
        else:
            print(f"⚠️ Warning: mask for '{mask_col}' provided but this column is not in DataFrame. Skipped.")

    # 2️⃣ 只对 cols 做 quantile 裁剪
    for col in cols:
        if col not in df.columns:
            print(f"⚠️ Warning: '{col}' not found in DataFrame columns. Skipped quantile.")
            continue

        valid = df[col].notna()
        if valid.sum() == 0:
            continue

        q_low = df.loc[valid, col].quantile(quantile)
        q_high = df.loc[valid, col].quantile(1 - quantile)

        mask_extreme = (df[col] <= q_low) | (df[col] >= q_high)
        df.loc[mask_extreme & valid, col] = np.nan

    return df

import numpy as np
import geopandas as gpd
import pandas as pd
import rasterio
from rasterio import features

def reclassify_and_calculate_proportion_fast(input_tif, shp_path):
    """
    向量化版：
    1) 重分类（-1→0，其它→1，保留 nodata）
    2) 一次性 rasterize 把每个面烧成一个整数 ID 图层
    3) 用 bincount 统计每个 ID 区域的总像元数和值==1 的像元数
    4) 计算比例并返回 DataFrame

    返回：
    - study_arr: np.ndarray，重分类后的数组（含 nodata）
    - df: pd.DataFrame，index 对应原 shp 的索引，列 ['proportion_of_1']
    """
    # 1) 读入栅格并重分类
    with rasterio.open(input_tif) as src:
        arr = src.read(1)
        nodata = src.nodata
        transform = src.transform
        shape = src.shape

    # 重分类：-1→0，其它有效值都设为1，nodata 保持原 nodata
    arr2 = np.where(arr == -1, 0, 1)
    if nodata is not None:
        arr2[arr == nodata] = nodata

    # 2) 读入矢量，并为每个面生成唯一 ID（1,2,3…）
    gdf = gpd.read_file(shp_path)
    shapes = ((geom, int(idx)+1) for idx, geom in enumerate(gdf.geometry))

    # 一次性 rasterize
    id_raster = features.rasterize(
        shapes,
        out_shape=shape,
        transform=transform,
        fill=0,
        dtype='int32'
    )
    # 0 表示不属于任何面；1..N 对应 gdf 的 0..N-1

    # 3) 掩膜：只保留 study_arr == 1 区域（含 nodata 仍算 total）
    # “包含 nodata” 这里 total = 区域所有像元数，无论值是什么
    mask_zone = id_raster > 0
    ids_all = id_raster[mask_zone]       # 区域内的面 ID 列表
    total_count = np.bincount(ids_all)   # length = max_id+1

    # 再统计值==1 的像元
    mask_one = (arr2 == 1) & (id_raster > 0)
    ids_one = id_raster[mask_one]
    one_count = np.bincount(ids_one, minlength=total_count.size)

    # 4) 计算比例
    # 注意 skip ID=0
    # 为避免除 0，先填 1，然后再把真正 total=0 的设置为 nan
    total = total_count.astype(float)
    total[total == 0] = np.nan
    proportion = one_count / total

    # 5) 构建结果 DataFrame
    # proportion[1] 对应 gdf.index[0]，依此类推
    df = pd.DataFrame({
        'proportion_of_1': proportion[1:]  # 丢掉背景 ID=0
    }, index=gdf.index)

    return arr2, df

import numpy as np
import geopandas as gpd
import pandas as pd
import rasterio
from rasterio.features import rasterize

def zonal_stats_vectorized(input_tif, study_arr, shp,
                            stats='mean', column_name=None):
    """
    向量化版 Zonal Stats，输出行数始终和 shp 面数量一致。

    参数
    ----
    input_tif   : str
                  栅格路径
    study_arr   : np.ndarray，和栅格同尺寸的掩膜数组（==1 的像元参与统计）
    shp         : str 或 GeoDataFrame
    stats       : 'sum' 或 'mean'
    column_name : 输出列名，默认为 '{stats}_val'

    返回
    ----
    pd.DataFrame，index 与 shp.index 对齐，列为统计结果
    """
    # 1) 读取面和栅格
    if isinstance(shp, str):
        gdf = gpd.read_file(shp)
    else:
        gdf = shp.copy()
    n_shapes = len(gdf)

    with rasterio.open(input_tif) as src:
        img = src.read(1)
        transform = src.transform
        shape = src.shape

    assert study_arr.shape == img.shape, "study_arr 和 input_tif 必须同尺寸"

    # 2) 一次性 rasterize：给每个面编号 1..n_shapes
    shapes = ((geom, i+1) for i, geom in enumerate(gdf.geometry))
    id_raster = rasterize(
        shapes,
        out_shape=shape,
        transform=transform,
        fill=0,           # 0 表示背景／不属于任何面
        dtype='int32'
    )

    # 3) 掩膜出要统计的像元：study_arr==1 且 属于某个面
    mask = (study_arr == 1) & (id_raster > 0)
    vals = img[mask]
    ids  = id_raster[mask]

    # 4) 用 bincount，长度设为 n_shapes+1，确保从 1..n_shapes 全都有值
    sum_per_id   = np.bincount(ids, weights=vals, minlength=n_shapes+1)
    count_per_id = np.bincount(ids, minlength=n_shapes+1)

    # 5) 取出 1..n_shapes 部分并计算
    if stats == 'sum':
        stat = sum_per_id[1:]       # 丢掉背景 ID=0
    elif stats == 'mean':
        with np.errstate(divide='ignore', invalid='ignore'):
            stat = sum_per_id[1:] / count_per_id[1:]
    else:
        raise ValueError("stats 参数只能是 'sum' 或 'mean'")

    # 6) 构造 DataFrame
    col = column_name or f"{stats}_val"
    df = pd.DataFrame({col: stat}, index=gdf.index)

    return df

# ==== 主体代码 ====
set_plot_style(font_size=10, font_family='Arial')

# ------- 字段信息结构化 -------
fields = {
    'carbon_cost': ("GHG reductions and removals cost(Million AU$)", "Million AU$", "cool"),
    'bio_cost':    ("Biodiversity restoration cost(Million AU$)", "Million AU$", "autumn_r"),
    'ghg':         ("GHG reductions and removals(MtCO2e)", "MtCO2e", "summer_r"),
    'bio':         ("Biodiversity restoration(Mha)", "Mha", "summer_r"),
}
# 自动生成相关列表
rename_map = {k: v[0] for k, v in fields.items()}
value_col_list = [v[0] for v in fields.values()] + [
    "Shadow carbon price(AU$ tCO2e-1)",
    "Shadow biodiversity price(AU$ ha-1)"
]
title_list = [
    'GHG reductions and removals cost',
    'Biodiversity restoration cost',
    'GHG reductions and removals',
    'Biodiversity restoration',
    'Shadow carbon price',
    'Shadow biodiversity price'
]
unit_list = [v[1] for v in fields.values()] + ["AU$ tCO2e-1", "AU$ ha-1"]
cmap_list = [v[2] for v in fields.values()] + ["winter_r", "winter_r"]


# 字段结构化部分...

set_plot_style(font_size=9, font_family='Arial')

# 路径准备
input_tif = f"{get_path(config.INPUT_FILES[0])}/out_2010/lmmap_2010.tiff"
# shp_path = "../Map/H_1wkm2.shp"
shp_name = 'H_1wkm2'
shp_path = f"../Map/{shp_name}.shp"
arr_path = f"{config.TASK_DIR}/carbon_price/data"
out_gpkg = "../Map/hex_gdf_output.gpkg"

# ------------------------------------------------ 计算部分 ------------------------------------------------
# 步骤1: 获得比例与分区统计
study_arr, df_proportion = reclassify_and_calculate_proportion_fast(input_tif, shp_path)
raster_files = list(fields.keys())

# 依次统计，自动确保 index 类型一致
for raster_file in raster_files:
    print(f"Processing {raster_file}...")
    raster_path = os.path.join(arr_path, f"{raster_file}_2050.tif")
    df_stat = zonal_stats_vectorized(raster_path, study_arr, shp_path, stats='sum', column_name=raster_file)
    df_stat.index = df_stat.index.astype(df_proportion.index.dtype)
    df_proportion = df_proportion.join(df_stat)
df_proportion_1 = df_proportion.copy()

mask_cc = df_proportion_1["carbon_cost"].between(-1e10, 1, inclusive="both")
mask_bc = df_proportion_1["bio_cost"].between(-1e10, 1, inclusive="both")
mask_c = df_proportion_1["ghg"].between(-1e10, 1, inclusive="both")
mask_b = df_proportion_1["bio"].between(-1e10, 1, inclusive="both")

# 步骤3: Shadow Price计算
df_proportion["Shadow carbon price(AU$ tCO2e-1)"] = (
    df_proportion["carbon_cost"]  / df_proportion["ghg"]
)
df_proportion["Shadow biodiversity price(AU$ ha-1)"] = (
    df_proportion["bio_cost"] / df_proportion["bio"]
)

mask_cp = df_proportion["Shadow carbon price(AU$ tCO2e-1)"].between(-1e10, 1, inclusive="both")
mask_bp = df_proportion["Shadow biodiversity price(AU$ ha-1)"].between(-1e10, 1, inclusive="both")


# 步骤3: 单位换算与重命名
unit_div = 1e6
df_proportion[raster_files] /= unit_div
df_proportion.rename(columns=rename_map, inplace=True)

# 前四个移除(-inf,1)，后两个移除后四个的(-inf,1)后又移除了前后5%
cols = list(df_proportion.columns[1:])
# print(cols)
masks = {
    "GHG reductions and removals cost(Million AU$)": mask_cc,
    "Biodiversity restoration cost(Million AU$)": mask_bc,
    "GHG reductions and removals(MtCO2e)": mask_c,
    "Biodiversity restoration(Mha)": mask_b,
    "Shadow carbon price(AU$ tCO2e-1)": mask_cp | mask_c,
    "Shadow biodiversity price(AU$ ha-1)": mask_bp | mask_b
}
df_proportion = set_extreme_quantile_nan(df_proportion, cols, quantile=0.05, masks=masks)

# 步骤4: 合并空间属性
gdf_hex = gpd.read_file(shp_path)
gdf_hex.index = gdf_hex.index.astype(df_proportion.index.dtype)
hex_gdf = gdf_hex.join(df_proportion, how="left")

# 步骤5: 导出
hex_gdf.to_file(out_gpkg, driver="GPKG")

# ------------------------------------------------------------------------------------------------
hex_gdf = gpd.read_file(out_gpkg)

threshold = 0.2
n_plot = len(value_col_list)
nrow, ncol = 3, 2
fig, axes = plt.subplots(nrow, ncol, figsize=(8, 12))
axes = axes.flatten()
aus_border_gdf = gpd.read_file("../Map/AU_boundary_line_main.shp")
for i in range(n_plot):
    draw_hex_shape(
        axes[i], hex_gdf, value_col_list[i],
        title_list[i], unit_list[i], cmap_list[i], threshold=threshold,aus_border_gdf=aus_border_gdf,H_linewidth=0.5
    )

for j in range(n_plot, len(axes)):
    axes[j].axis('off')

plt.savefig(f"{config.TASK_DIR}/carbon_price/Paper_figure/03_{shp_name}_maps.png", dpi=300, bbox_inches='tight')
plt.show()