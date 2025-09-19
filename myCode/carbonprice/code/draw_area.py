import tools.config as config
from tools.tools import get_path, filter_all_from_dims
import pickle

import os
from pathlib import Path
import pandas as pd
import numpy as np
import xarray as xr
from joblib import Parallel, delayed
from typing import Dict, Iterable, List, Optional

import matplotlib.pyplot as plt


### MODIFICATION ###
# Updated the function to handle shared y-axes and global limits.
def get_global_ylim(dict_df: Dict[str, pd.DataFrame]) -> tuple:
    """
    Calculate the global y-axis limits for a dictionary of DataFrames,
    considering separate stacking for positive and negative values.
    """
    global_max = 0
    global_min = 0

    for df in dict_df.values():
        # Calculate the max height of the positive stack for each row
        positive_sums = df[df > 0].sum(axis=1)
        if not positive_sums.empty:
            current_max = positive_sums.max()
            if current_max > global_max:
                global_max = current_max

        # Calculate the min depth of the negative stack for each row
        negative_sums = df[df < 0].sum(axis=1)
        if not negative_sums.empty:
            current_min = negative_sums.min()
            if current_min < global_min:
                global_min = current_min

    # Return a tuple (min, max) with a small padding at the top and bottom
    return (global_min * 1.05, global_max * 1.05)


### MODIFICATION ###
# Re-implemented to manually stack positive and negative values.
def plot_dict_stacked(dict_df, title_names, figsize=(20, 12), nrows=2, ncols=5, n_col=5, y_label="", y_lim=None,xtick_step=10):
    """
    Plots a dictionary of DataFrames as stacked bar charts, with separate
    stacking for positive and negative values.

    Parameters
    ----
    dict_df : dict[str, pd.DataFrame]
        key -> DataFrame (index=year, columns=categories)
    title_names : list
        List of titles for each subplot.
    figsize : tuple
        Figure size.
    nrows, ncols : int
        Grid dimensions for subplots.
    n_col : int
        Number of columns for the legend.
    y_label : str
        The y-axis label for the entire figure.
    y_lim : tuple, optional
        The y-axis range (min, max).
    """
    # Set sharey=True to link all y-axes.
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, sharex=False, sharey=True)
    axes = axes.flatten()

    # Get a list of colors to use for all plots
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    # Loop to plot each subplot
    for i, (key, df) in enumerate(dict_df.items()):
        ax = axes[i]

        if 2010 not in df.index:
            df = df.reindex(df.index.union([2010])).sort_index().fillna(0)

        # 初始化两个栈的“底座”
        y_offset_pos = pd.Series(0.0, index=df.index, dtype=float)
        y_offset_neg = pd.Series(0.0, index=df.index, dtype=float)

        for j, col_name in enumerate(df.columns):
            color = colors[j % len(colors)]

            # 正数：从 y_offset_pos 往上堆
            pos_data = df[col_name].clip(lower=0)
            ax.bar(df.index, pos_data, bottom=y_offset_pos, color=color,
                   label=col_name, width=0.8)
            y_offset_pos = y_offset_pos + pos_data  # 累积

            # 负数：从 y_offset_neg（<=0）继续向下堆
            neg_data = df[col_name].clip(upper=0)
            ax.bar(df.index, neg_data, bottom=y_offset_neg, color=color,
                   width=0.8, label=None)  # 不重复添加图例
            y_offset_neg = y_offset_neg + neg_data  # 累积

        ax.set_title(title_names[i])
        ax.set_xlabel("")
        ax.set_ylabel("")  # Individual labels are off

        years_sorted = sorted(int(y) for y in df.index)
        tick_years = [y for y in years_sorted if (y - years_sorted[0]) % xtick_step == 0]

        # 关键：刻度位置用“年份”，不是位置索引
        ax.set_xticks(tick_years)
        ax.set_xticklabels([str(y) for y in tick_years])

        # --- X-axis tick formatting ---
        #
        # if years:
        #     # For bar charts, ticks are at integer positions 0, 1, 2...
        #     # We map these positions back to the year labels.
        #     tick_years = [2010,2020,2030,2040,2050]
        #     ax.set_xticks([years_sorted.index(y) for y in tick_years])  # 位置
        #     ax.set_xticklabels(tick_years)

            # Apply the global y-limit to the first axis; it will propagate to all.
    if y_lim:
        axes[0].set_ylim(y_lim)

    fig.tight_layout(rect=[0.05, 0.04, 1, 0.98])

    # --- Legend Handling ---
    # Get handles and labels, then remove duplicates
    handles, labels = axes[0].get_legend_handles_labels()
    unique_labels = {}
    for handle, label in zip(handles, labels):
        if label not in unique_labels:
            unique_labels[label] = handle

    fig.legend(
        unique_labels.values(), unique_labels.keys(),
        loc="lower center",
        ncol=n_col,
        frameon=False
    )

    # Add a single, centered y-label for the entire figure.
    if y_label:
        fig.supylabel(y_label)

    fig.suptitle('', fontsize=16)

    plt.show()
    return fig


# ---------- 并行配置 ----------

# ========= 1) 汇总成三列（Ag / Mgt / non-ag），按年一行 =========
def _row_totals_for_year(origin_path: str, year: int, strict: bool = True) -> Optional[dict]:
    """返回单年 {'year', 'Ag','Mgt','non-ag'} 的字典；文件缺失时按 strict 处理。"""
    base = Path(origin_path) / f"out_{year}"

    p_land = base / f"xr_area_agricultural_landuse_{year}.nc"
    p_mgt = base / f"xr_area_agricultural_management_{year}.nc"
    p_non = base / f"xr_area_non_agricultural_landuse_{year}.nc"

    if strict and (not p_land.exists() or not p_mgt.exists() or not p_non.exists()):
        missing = [str(p) for p in [p_land, p_mgt, p_non] if not p.exists()]
        raise FileNotFoundError(f"[{year}] 缺少文件：{missing}")

    print(f"Processing year {year}...")

    def _safe_sum(p: Path) -> float:
        if not p.exists():
            return np.nan
        with xr.open_dataarray(p) as da:
            da = filter_all_from_dims(da)
            return (da.sum(dim=list(da.dims)).item() / 1e6)

    return {
        "year": year,
        "Ag": _safe_sum(p_land),
        "Mgt": _safe_sum(p_mgt),
        "non-ag": _safe_sum(p_non),
    }


def build_summary_tables(
        task_name: str,
        input_files: Iterable[str],
        years: Iterable[int],
        n_jobs: int = 41,
        strict: bool = False,
) -> Dict[str, pd.DataFrame]:
    """
    对每个 input_file 生成一个 DataFrame（index=year, columns=['Ag','Mgt','non-ag']）。
    """
    results: Dict[str, pd.DataFrame] = {}
    years = list(years)

    for input_file in input_files:
        origin_path = get_path(task_name, input_file)

        rows = Parallel(n_jobs=n_jobs)(
            delayed(_row_totals_for_year)(origin_path, year, strict)
            for year in years
        )
        df = pd.DataFrame([r for r in rows if r is not None]).set_index("year").sort_index()
        results[input_file] = df

    return results


# ========= 2) 管理维度按 am 聚合（只剩 am），列为年份 =========
def _series_for_year(origin_path: str, year: int,xr_name:str,keep_dim='am', strict: bool = True) -> Optional[pd.Series]:
    base = Path(origin_path) / f"out_{year}"
    p_mgt = base / f"{xr_name}_{year}.nc"

    if strict and not p_mgt.exists():
        raise FileNotFoundError(f"[{year}] 缺少文件：{p_mgt}")

    if not p_mgt.exists():
        return None

    print(f"Processing {p_mgt}...")
    with xr.open_dataarray(p_mgt) as da:
        da = filter_all_from_dims(da)
        # 对除 'am' 外的所有维求和，保留 am
        da_am = da.sum(dim=[d for d in da.dims if d != keep_dim]) / 1e6
        s = da_am.to_pandas()  # index=am, values=面积
        s.name = year
        return s


def build_tables(
        task_name: str,
        input_files: Iterable[str],
        years: Iterable[int],
        xr_name: str = "xr_GHG_ag_management",
        keep_dim: str = 'am',
        *,
        n_jobs: int = 41,
        strict: bool = False,
) -> Dict[str, pd.DataFrame]:
    """
    对每个 input_file 生成一个 DataFrame（index=am，columns=years）。
    """
    results_mgt: Dict[str, pd.DataFrame] = {}
    years = list(years)

    for input_file in input_files:
        origin_path = get_path(task_name, input_file)
        series_list = Parallel(n_jobs=n_jobs)(
            delayed(_series_for_year)(origin_path, year,xr_name,keep_dim, strict) for year in years
        )
        series_list = [s for s in series_list if s is not None]

        # 对齐所有 am 索引后按列拼接
        if series_list:
            df_all = pd.concat(series_list, axis=1).sort_index(axis=1)
        else:
            df_all = pd.DataFrame()
        results_mgt[input_file] = df_all.T

    return results_mgt

def build_tables_from_process(
        origin_path: str,
        input_files: Iterable[str],
        years: Iterable[int],
        xr_name: str = "xr_GHG_ag_management",
        keep_dim: str = 'am',
        *,
        n_jobs: int = 41,
        strict: bool = False,
) -> Dict[str, pd.DataFrame]:
    """
    对每个 input_file 生成一个 DataFrame（index=am，columns=years）。
    """
    results_mgt: Dict[str, pd.DataFrame] = {}
    years = list(years)

    for input_file in input_files:
        input_origin_path = f'{origin_path}/{input_file}'
        file_path = f"{xr_name}_{input_file}"
        series_list = Parallel(n_jobs=n_jobs)(
            delayed(_series_process_for_year)(input_origin_path, year,file_path,keep_dim, strict) for year in years
        )
        series_list = [s for s in series_list if s is not None]

        # 对齐所有 am 索引后按列拼接
        if series_list:
            df_all = pd.concat(series_list, axis=1).sort_index(axis=1)
        else:
            df_all = pd.DataFrame()
        results_mgt[input_file] = df_all.T

    return results_mgt

def _series_process_for_year(origin_path: str, year: int,xr_name:str,keep_dim='am', strict: bool = True) -> Optional[pd.Series]:
    base = Path(origin_path) / f"{year}"
    p_mgt = base / f"{xr_name}_{year}.nc"

    if strict and not p_mgt.exists():
        raise FileNotFoundError(f"[{year}] 缺少文件：{p_mgt}")

    if not p_mgt.exists():
        return None

    print(f"Processing {p_mgt}...")
    with xr.open_dataarray(p_mgt) as da:
        da = filter_all_from_dims(da)
        # 对除 'am' 外的所有维求和，保留 am
        da_am = da.sum(dim=[d for d in da.dims if d != keep_dim]) / 1e6
        s = da_am.to_pandas()  # index=am, values=面积
        s.name = year
        return s

def load_dict_pickle(save_path):
    with open(save_path, "rb") as f:
        return pickle.load(f)


def save_dict_pickle(dict_df, save_path):
    with open(save_path, "wb") as f:
        pickle.dump(dict_df, f)
    print(f"✅ 已保存到 {save_path}")


### MODIFICATION ###
def get_global_ylim(dict_df: Dict[str, pd.DataFrame]) -> tuple:
    """Calculate global y-limits (supporting positive and negative stacks)."""
    global_max = float("-inf")
    global_min = float("inf")

    for df in dict_df.values():
        arr = np.nan_to_num(df.to_numpy(dtype=float), nan=0.0)

        # 正向累积最大值
        pos_cumsum = np.where(arr > 0, arr, 0).cumsum(axis=1)
        pos_max = np.max(pos_cumsum) if pos_cumsum.size else 0.0

        # 负向累积最小值
        neg_cumsum = np.where(arr < 0, arr, 0).cumsum(axis=1)
        neg_min = np.min(neg_cumsum) if neg_cumsum.size else 0.0

        # 单个值极值
        data_max = np.max(arr) if arr.size else 0.0
        data_min = np.min(arr) if arr.size else 0.0

        row_max = max(pos_max, data_max)
        row_min = min(neg_min, data_min)

        if row_max > global_max:
            global_max = row_max
        if row_min < global_min:
            global_min = row_min

    # 保证 0 在线内
    if global_min > 0:
        global_min = 0
    if global_max < 0:
        global_max = 0

    # 上下各加 5% padding
    span = global_max - global_min
    if span == 0:
        span = 1.0
    return (global_min - 0.05 * span, global_max + 0.05 * span)



task_name = config.TASK_NAME
task_dir = f'../../../output/{task_name}'
years = [y for y in range(2011, 2051)]
n_jobs = 22
output_dir = f'../../../output/{task_name}/carbon_price/5_draw_area'
os.makedirs(output_dir, exist_ok=True)

origin_path = f"../../../output/{task_name}/carbon_price/0_base_data"

# input_files_1 = config.carbon_names
# results_t = build_tables_from_process(origin_path, input_files_1, years,xr_name='xr_transition_cost_ag2non_ag_amortised_diff',keep_dim='To land-use', strict=True, n_jobs=n_jobs)
# save_dict_pickle(results_t, os.path.join(output_dir, 'cost_GHG_tn.pkl'))
#
# results_mgt = build_tables_from_process(origin_path, input_files_1, years,xr_name='xr_cost_agricultural_management',keep_dim='am',  strict=True, n_jobs=n_jobs)
# save_dict_pickle(results_mgt, os.path.join(output_dir, 'cost_GHG_mgt.pkl'))
#
# results_nonag = build_tables_from_process(origin_path, input_files_1, years,xr_name='xr_cost_non_ag',keep_dim='lu', strict=True, n_jobs=n_jobs)
# save_dict_pickle(results_nonag, os.path.join(output_dir, 'cost_GHG_nonag_lu.pkl'))
#
# input_files_2 = config.carbon_bio_names
# results_t = build_tables_from_process(origin_path, input_files_2, years,xr_name='xr_transition_cost_ag2non_ag_amortised_diff',keep_dim='To land-use', strict=True, n_jobs=n_jobs)
# save_dict_pickle(results_t, os.path.join(output_dir, 'cost_bio_tn.pkl'))
#
# results_mgt = build_tables_from_process(origin_path, input_files_2, years,xr_name='xr_cost_agricultural_management',keep_dim='am',  strict=True, n_jobs=n_jobs)
# save_dict_pickle(results_mgt, os.path.join(output_dir, 'cost_bio_mgt.pkl'))
#
# results_nonag = build_tables_from_process(origin_path, input_files_2, years,xr_name='xr_cost_non_ag',keep_dim='lu', strict=True, n_jobs=n_jobs)
# save_dict_pickle(results_nonag, os.path.join(output_dir, 'cost_bio_nonag_lu.pkl'))


title_ghg_names = [r'$\mathrm{GHG}_{\mathrm{low}}$', r'$\mathrm{GHG}_{\mathrm{high}}$']


results_ghg_t = load_dict_pickle(os.path.join(output_dir, 'cost_GHG_tn.pkl'))
# results_ghg_mgt = {key: -df for key, df in results_ghg_mgt.items()}
summary_ylim = get_global_ylim(results_ghg_t)
fig_summary = plot_dict_stacked(results_ghg_t, title_ghg_names, figsize=(20, 12),nrows=1,ncols=2, y_label='Transition(ag2non-ag) GHG cost (m AU$)', y_lim=summary_ylim)
fig_summary.savefig(os.path.join(output_dir, 'cost_GHG_tn_stacked.png'), dpi=300)

results_ghg_mgt = load_dict_pickle(os.path.join(output_dir, 'cost_GHG_mgt.pkl'))
summary_ylim = get_global_ylim(results_ghg_mgt)
fig_summary = plot_dict_stacked(results_ghg_mgt, title_ghg_names, figsize=(20, 12),nrows=1,ncols=2, y_label='Mgt GHG cost (m AU$)', y_lim=summary_ylim)
fig_summary.savefig(os.path.join(output_dir, 'cost_GHG_mgt_stacked.png'), dpi=300)

results_ghg_nonag = load_dict_pickle(os.path.join(output_dir, 'cost_GHG_nonag_lu.pkl'))
summary_ylim = get_global_ylim(results_ghg_nonag)
fig_summary = plot_dict_stacked(results_ghg_nonag, title_ghg_names, figsize=(20, 12),nrows=1,ncols=2, y_label='Non-ag GHG cost (m AU$)', y_lim=summary_ylim)
fig_summary.savefig(os.path.join(output_dir, 'cost_GHG_nonag_stacked.png'), dpi=300)


title_bio_names = [r'$\mathrm{GHG}_{\mathrm{low}}$,$\mathrm{Bio}_{\mathrm{10}}$',
               r'$\mathrm{GHG}_{\mathrm{low}}$,$\mathrm{Bio}_{\mathrm{20}}$',
               r'$\mathrm{GHG}_{\mathrm{low}}$,$\mathrm{Bio}_{\mathrm{30}}$',
               r'$\mathrm{GHG}_{\mathrm{low}}$,$\mathrm{Bio}_{\mathrm{40}}$',
               r'$\mathrm{GHG}_{\mathrm{low}}$,$\mathrm{Bio}_{\mathrm{50}}$',

               r'$\mathrm{GHG}_{\mathrm{high}}$,$\mathrm{Bio}_{\mathrm{10}}$',
               r'$\mathrm{GHG}_{\mathrm{high}}$,$\mathrm{Bio}_{\mathrm{20}}$',
               r'$\mathrm{GHG}_{\mathrm{high}}$,$\mathrm{Bio}_{\mathrm{30}}$',
               r'$\mathrm{GHG}_{\mathrm{high}}$,$\mathrm{Bio}_{\mathrm{40}}$',
               r'$\mathrm{GHG}_{\mathrm{high}}$,$\mathrm{Bio}_{\mathrm{50}}$',
               ]

results_bio_t = load_dict_pickle(os.path.join(output_dir, 'cost_bio_tn.pkl'))
# results_bio_mgt = {key: -df for key, df in results_bio_mgt.items()}
summary_ylim = get_global_ylim(results_bio_t)
fig_summary = plot_dict_stacked(results_bio_t, title_bio_names, figsize=(20, 12),nrows=2,ncols=5, y_label='Transition(ag2non-ag) Biodiversity cost (m AU$)', y_lim=summary_ylim)
fig_summary.savefig(os.path.join(output_dir, 'cost_bio_tn_stacked.png'), dpi=300)

results_bio_mgt = load_dict_pickle(os.path.join(output_dir, 'cost_bio_mgt.pkl'))
summary_ylim = get_global_ylim(results_bio_mgt)
fig_summary = plot_dict_stacked(results_bio_mgt, title_bio_names, figsize=(20, 12),nrows=2,ncols=5, y_label='Mgt Biodiversity cost (m AU$)', y_lim=summary_ylim)
fig_summary.savefig(os.path.join(output_dir, 'cost_bio_mgt_stacked.png'), dpi=300)

results_bio_nonag = load_dict_pickle(os.path.join(output_dir, 'cost_bio_nonag_lu.pkl'))
summary_ylim = get_global_ylim(results_bio_nonag)
fig_summary = plot_dict_stacked(results_bio_nonag, title_bio_names, figsize=(20, 12),nrows=2,ncols=5, y_label='Non-ag Biodiversity cost (m AU$)', y_lim=summary_ylim)
fig_summary.savefig(os.path.join(output_dir, 'cost_bio_nonag_stacked.png'), dpi=300)
