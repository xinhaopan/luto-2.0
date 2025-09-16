import matplotlib.pyplot as plt
import tools.config as config
import xarray as xr
import seaborn as sns
import os
import time
# 引入 joblib
from joblib import Parallel, delayed

sns.set_theme(style="whitegrid")

def create_mask(years, base_path, env_category, env_name, chunks="auto"):
    masks = []
    for y in years:
        data_tmpl = f"{base_path}/{env_category}/{y}/xr_total_{env_category}_{y}.nc"
        data_xr = xr.open_dataarray(data_tmpl.format(year=y), chunks=chunks)
        cost_tml = f"{base_path}/{env_category}/{y}/xr_{env_name}_{y}.nc"
        cost_xr = xr.open_dataarray(cost_tml.format(year=y), chunks=chunks)
        m = (abs(data_xr) >= 1)  # & (abs(cost_xr >= 1))
        masks.append(m.expand_dims(year=[y]))
    mask = xr.concat(masks, dim="year")
    mask.name = "mask"
    mask.attrs["description"] = "True if both env and cost >= 1, else False"
    return mask


def create_xarray(years, base_path, env_category, env_name, mask=None,
                  engine="h5netcdf",
                  cell_dim="cell", cell_chunk="auto",
                  year_chunk=1, parallel=False):
    """
    以 year 维度拼接多个年度 NetCDF，懒加载+分块，避免过多文件句柄。
    """
    file_paths = [
        os.path.join(base_path, str(env_category), str(y), f"xr_{env_name}_{y}.nc")
        for y in years
    ]
    missing = [p for p in file_paths if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(f"以下文件未找到:\n" + "\n".join(missing))

    # 从文件名提取实际年份，确保坐标与文件顺序一致
    valid_years = [int(os.path.basename(p).split("_")[-1].split(".")[0]) for p in file_paths]

    ds = xr.open_mfdataset(
        file_paths,
        engine=engine,
        combine="nested",  # 明确“按给定顺序拼接”
        concat_dim="year",  # 新增 year 维度
        parallel=parallel,  # 一般 False 更稳，避免句柄并发
        chunks={cell_dim: cell_chunk, "year": year_chunk}  # year=1，cell 分块
    ).assign_coords(year=valid_years)

    if mask is not None:
        ds = ds.where(mask, other=0)  # 使用掩码，非掩码区域设为 0

    return ds

# prepare_data_for_plot 和 draw_prepared_data_on_ax 函数与之前完全相同
def prepare_data_for_plot(env_category, env_category_name):
    """
    重量级函数：读取、聚合(求和)和加载数据。将被并行执行。
    """
    print(f"开始准备数据和求和: {env_category}")
    # 1. 懒加载方式打开数据
    xr_bio = create_xarray(years, base_path, env_category, env_category_name)
    da = xr_bio["data"]

    # 2. 对 cell 维度进行求和。这步操作是懒加载的，速度极快。
    #    'cell' 是 create_xarray 函数中的默认 cell_dim
    summed_da = da.sum(dim='cell')

    # 3. 获取年份坐标
    year_values = summed_da["year"].values

    # 4. 触发计算(包括求和)并将结果加载到内存
    #    现在加载的是一个 (40,) 的小数组，而不是 (40, 168778) 的大数组
    data_values = summed_da.load().values

    print(f"数据准备完成: {env_category}, 最终数据形状: {data_values.shape}")
    return year_values, data_values


def draw_prepared_data_on_ax(ax, year_vals, data_vals, title_name, unit, label):
    """轻量级函数，只负责绘图。"""
    ax.plot(year_vals, data_vals, marker="o", linestyle="-", label=label)
    ax.set_title(title_name)
    ax.set_ylabel(unit)


# --- 脚本主干 ---
# (路径和列表定义与之前相同)
task_name = config.TASK_NAME
base_path = f"../../../output/{task_name}/carbon_price/0_base_data"
# ... 其他路径和列表定义 ...
env_category_lists = [
    ["carbon_low_bio_10", "carbon_low_bio_20", "carbon_low_bio_30", "carbon_low_bio_40", "carbon_low_bio_50"],
    ["carbon_high_bio_10", "carbon_high_bio_20", "carbon_high_bio_30", "carbon_high_bio_40", "carbon_high_bio_50"]
]

title_name_lists = [
    [
        '$\mathrm{GHG}_{\mathrm{low}}$,$\mathrm{Bio}_{\mathrm{10}}$',
        '$\mathrm{GHG}_{\mathrm{low}}$,$\mathrm{Bio}_{\mathrm{20}}$',
        '$\mathrm{GHG}_{\mathrm{low}}$,$\mathrm{Bio}_{\mathrm{30}}$',
        '$\mathrm{GHG}_{\mathrm{low}}$,$\mathrm{Bio}_{\mathrm{40}}$',
        '$\mathrm{GHG}_{\mathrm{low}}$,$\mathrm{Bio}_{\mathrm{50}}$'
    ],
    [
        '$\mathrm{GHG}_{\mathrm{high}}$,$\mathrm{Bio}_{\mathrm{10}}$',
        '$\mathrm{GHG}_{\mathrm{high}}$,$\mathrm{Bio}_{\mathrm{20}}$',
        '$\mathrm{GHG}_{\mathrm{high}}$,$\mathrm{Bio}_{\mathrm{30}}$',
        '$\mathrm{GHG}_{\mathrm{high}}$,$\mathrm{Bio}_{\mathrm{40}}$',
        '$\mathrm{GHG}_{\mathrm{high}}$,$\mathrm{Bio}_{\mathrm{50}}$'
    ]
]

years = list(range(2011, 2051))

fig2, axes2 = plt.subplots(nrows=2, ncols=5, figsize=(25, 10), sharey=True)
fig2.suptitle('Biodiversity amortised cost', fontsize=18)

# --- 使用 joblib 并行准备数据 ---
start_time = time.time()

# 1. 准备要执行的任务列表
tasks = []
for row in range(1):
    for col in range(1):
        env_cat = env_category_lists[row][col]
        task = delayed(prepare_data_for_plot)(
            env_cat,
            f'total_cost_{env_cat}_amortised'
        )
        tasks.append(task)

# 2. 并行执行
# n_jobs=-1 表示使用所有可用的CPU核心作为线程数
# backend='threading' 是关键！这告诉 joblib 使用线程而不是进程。
print("--- 开始使用 joblib (threading backend) 并行准备数据 ---")
results = Parallel(n_jobs=-1, backend='threading')(tasks)

data_prep_time = time.time()
print(f"\n--- 数据并行准备耗时: {data_prep_time - start_time:.2f} 秒 ---\n")

# --- 串行快速绘图 ---
# joblib 返回一个列表，我们需要根据顺序把它放回正确的位置
task_index = 0
for row in range(1):
    for col in range(1):
        print(f"绘制图表: 行 {row+1}, 列 {col+1}")
        ax = axes2[row, col]
        title = title_name_lists[row][col]

        year_vals, data_vals = results[task_index]

        draw_prepared_data_on_ax(
            ax=ax,
            year_vals=year_vals,
            data_vals=data_vals,
            title_name=title,
            unit="AU$",
            label=env_category_lists[row][col]
        )

        if col != 0:
            ax.tick_params(axis='y', labelleft=False)

        task_index += 1

# (设置Y轴标签和布局的代码与之前相同)
axes2[0, 0].set_ylabel('Cost (AU$)', fontsize=12)
axes2[1, 0].set_ylabel('Cost (AU$)', fontsize=12)
fig2.tight_layout(rect=[0, 0.03, 1, 0.95])

plot_time = time.time()
print(f"--- 绘图及布局耗时: {plot_time - data_prep_time:.2f} 秒 ---")
print(f"--- 总耗时: {plot_time - start_time:.2f} 秒 ---")
plt.savefig(f'biodiversity_amortised_cost.png', dpi=300)
plt.show()