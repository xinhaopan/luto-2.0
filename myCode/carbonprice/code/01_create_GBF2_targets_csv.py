import os
import glob
import zipfile
import io
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from collections import namedtuple
import tools.config as config
from tools.helper_plot import set_plot_style, xarray_to_dict


_CSV_PREFIX = "biodiversity_GBF2_priority_scores_"


def _extract_target_from_df(df: pd.DataFrame) -> tuple[int, float]:
    """
    从 DataFrame 中提取 (year, target_ha)。
    筛选条件：region=AUSTRALIA, Landuse=ALL, Water_supply=ALL,
              Type=Agricultural land-use, Agricultural Management=NaN
    公式：target_ha = Area Weighted Score (ha) / Contribution Relative to Pre-1750 Level (%) * Priority Target (%)
    """
    mask = (
        (df["region"].astype(str).str.strip() == "AUSTRALIA")
        & (df["Landuse"].astype(str).str.strip() == "ALL")
        & (df["Water_supply"].astype(str).str.strip() == "ALL")
        & (df["Type"].astype(str).str.strip() == "Agricultural land-use")
        & df["Agricultural Management"].isna()
    )
    rows = df[mask]
    if rows.empty:
        raise ValueError("未找到 AUSTRALIA/ALL/ALL/Agricultural land-use 行")
    row = rows.iloc[0]
    year = int(row["Year"])
    area = float(row["Area Weighted Score (ha)"])
    contrib = float(row["Contribution Relative to Pre-1750 Level (%)"])
    priority = float(row["Priority Target (%)"])
    return year, area / contrib * priority


def get_GBF2_target_list_from_csv(
    task_name: str,
    filename: str,
    start_year: int,
    end_year: int = 2050,
) -> list[float]:
    """
    从每年的 biodiversity_GBF2_priority_scores_{year}.csv 读取 GBF2 目标面积，
    避免加载 lz4 内存密集型数据。

    对每个可用年份计算：
        target_ha = Area Weighted Score (ha) / Contribution Relative to Pre-1750 Level (%) * Priority Target (%)

    然后对 start_year → end_year 做线性插值。

    文件路径：{run_dir}/output/{timestamp}/out_{year}/biodiversity_GBF2_priority_scores_{year}.csv
    zip 路径：{run_dir}/Run_Archive.zip 内 output/*/out_*/biodiversity_GBF2_priority_scores_*.csv

    Returns
    -------
    list[float]
        长度 = end_year - start_year + 1，对应每年 target（单位 ha）
    """
    run_dir = os.path.abspath(os.path.join("../../../output", task_name, filename))
    zip_path = os.path.join(run_dir, "Run_Archive.zip")

    year_target_map: dict[int, float] = {}

    # 尝试直接从文件系统读取
    direct_pattern = os.path.join(run_dir, "output", "*", "out_*", f"{_CSV_PREFIX}*.csv")
    direct_files = glob.glob(direct_pattern)

    if direct_files:
        for csv_path in direct_files:
            df = pd.read_csv(csv_path)
            yr, tgt = _extract_target_from_df(df)
            year_target_map[yr] = tgt
    elif os.path.isfile(zip_path):
        # 从 Run_Archive.zip 中读取
        with zipfile.ZipFile(zip_path, "r") as zf:
            matched = [
                n for n in zf.namelist()
                if _CSV_PREFIX in os.path.basename(n) and n.endswith(".csv")
            ]
            for name in matched:
                with zf.open(name) as f:
                    df = pd.read_csv(io.BytesIO(f.read()))
                yr, tgt = _extract_target_from_df(df)
                year_target_map[yr] = tgt
    else:
        raise FileNotFoundError(
            f"未找到 {_CSV_PREFIX}*.csv 文件，也无 Run_Archive.zip: {run_dir}"
        )

    if not year_target_map:
        raise FileNotFoundError(
            f"未找到任何 {_CSV_PREFIX}*.csv 文件: {run_dir}"
        )

    sorted_years = sorted(year_target_map)
    f_interp = interp1d(
        sorted_years,
        [year_target_map[y] for y in sorted_years],
        kind="linear",
        fill_value="extrapolate",
    )

    years = np.arange(int(start_year), int(end_year) + 1, dtype=int)
    return f_interp(years).astype(float).tolist()


def precompute_gbf2_targets_csv(
    *,
    task_name: str,
    filenames: list[str],
    start_year: int,
    end_year: int = 2050,
    out_dir: str | None = None,
    out_csv_name: str | None = None,
    scale: float = 1e6,          # ✅ 除以 1e6
    use_parallel: bool = True,
    n_jobs: int = 4,
    backend: str = "loky",
    overwrite: bool = True,
):
    """
    预计算 GBF2 targets，并把所有 filename 合并存为一个 CSV：
      - 行：Year
      - 列：每个 filename
      - 值：targets / scale（默认除以 1e6）

    输出路径默认：
      ../../../output/{task_name}/{config.CARBON_PRICE_DIR}/1_draw_data/gbf2_targets_{start_year}_{end_year}_scaled1e6.csv

    Returns
    -------
    str
        输出 CSV 的路径
    """
    if out_dir is None:
        out_dir = os.path.abspath(os.path.join("../../../output", task_name, config.CARBON_PRICE_DIR, "1_draw_data"))
    os.makedirs(out_dir, exist_ok=True)

    filenames = list(dict.fromkeys(filenames))  # 去重且保持原顺序

    if out_csv_name is None:
        out_csv_name = f"gbf2_targets.csv"
    out_csv_path = os.path.join(out_dir, out_csv_name)

    if (not overwrite) and os.path.exists(out_csv_path):
        print(f"✅ 已存在，跳过写入: {out_csv_path}")
        return out_csv_path

    years = list(range(int(start_year), int(end_year) + 1))

    def compute_one(fn: str):
        targets = get_GBF2_target_list_from_csv(
            task_name=task_name,
            filename=fn,
            start_year=int(start_year),
            end_year=int(end_year),
        )
        # ✅ 缩放：除以 1e6
        targets = [t / scale for t in targets]
        return fn, targets

    # 并行或串行计算
    if use_parallel:
        from joblib import Parallel, delayed
        results = Parallel(n_jobs=n_jobs, backend=backend, verbose=10)(
            delayed(compute_one)(fn) for fn in filenames
        )
    else:
        results = [compute_one(fn) for fn in filenames]

    # 拼成一个大表：Year + 每个 filename 一列
    df_out = pd.DataFrame({"Year": years})
    for fn, targets in results:
        if len(targets) != len(years):
            raise ValueError(f"{fn} targets 长度 {len(targets)} != years 长度 {len(years)}")
        df_out[fn] = targets

    df_out.to_csv(out_csv_path, index=False)
    print(f"✅ Saved one CSV: {out_csv_path}")
    print(f"   shape={df_out.shape} (rows={len(years)}, cols={1 + len(results)})")

    return out_csv_path


precompute_gbf2_targets_csv(
    task_name=config.TASK_NAME,
    filenames=sorted(set(config.input_files), reverse=True),   # 每个 filename 将成为一列
    start_year=config.START_YEAR,
    end_year=2050,
    use_parallel=True,
    n_jobs=1,                      # 注意：共享文件系统下不一定越大越快
    backend="loky",
    scale=1e6,                      # ✅ 除以 1e6
)
