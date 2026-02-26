import os
import pandas as pd
from collections import namedtuple
import tools.config as config
from tools.helper_plot import set_plot_style,xarray_to_dict

def get_GBF2_target_list_start_to_2050(
    data,
    task_name: str,
    filename: str,
    start_year: int,
    end_year: int = 2050,
) -> list[float]:
    """
    返回从 start_year 到 2050（含）的 GBF2 target 列表。

    Returns
    -------
    list[float]
        长度 = end_year - start_year + 1，对应每一年的 target（绝对值，不是百分比）
    """
    import os
    import importlib.util
    import numpy as np
    from scipy.interpolate import interp1d

    def _import_settings_from_output(task_name: str, filename: str):
        """
        从输出目录按路径导入 settings.py
        期望路径: ../../../output/{task_name}/{filename}/luto/settings.py
        """
        settings_path = os.path.abspath(
            os.path.join("../../../output", task_name, filename, "luto", "settings.py")
        )
        if not os.path.isfile(settings_path):
            raise FileNotFoundError(f"settings.py 不存在: {settings_path}")

        spec = importlib.util.spec_from_file_location("custom_luto_settings", settings_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"无法为 settings.py 创建 spec: {settings_path}")

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    settings = _import_settings_from_output(task_name, filename)

    target_config = settings.GBF2_TARGETS_DICT.get(settings.BIODIVERSITY_TARGET_GBF_2)
    if target_config is None:
        target_config = {2030: 0, 2050: 0, 2100: 0}

    bio_habitat_score_baseline_sum = (data.BIO_GBF2_MASK * data.REAL_AREA).sum()
    bio_habitat_score_base_yr_sum = data.BIO_GBF2_BASE_YR.sum()
    bio_habitat_score_base_yr_proportion = bio_habitat_score_base_yr_sum / bio_habitat_score_baseline_sum

    bio_habitat_target_proportion = [
        bio_habitat_score_base_yr_proportion + ((1 - bio_habitat_score_base_yr_proportion) * i)
        for i in target_config.values()
    ]

    targets_key_years = {
        data.YR_CAL_BASE: bio_habitat_score_base_yr_sum,
        **dict(zip(
            target_config.keys(),
            bio_habitat_score_baseline_sum * np.array(bio_habitat_target_proportion)
        ))
    }

    f = interp1d(
        list(targets_key_years.keys()),
        list(targets_key_years.values()),
        kind="linear",
        fill_value="extrapolate"
    )

    years = np.arange(int(start_year), int(end_year) + 1, dtype=int)
    targets = f(years).astype(float).tolist()
    return targets

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
      ../../../output/{task_name}/carbon_price/1_draw_data/gbf2_targets_{start_year}_{end_year}_scaled1e6.csv

    Returns
    -------
    str
        输出 CSV 的路径
    """
    from tools.tools import get_data_RES

    if out_dir is None:
        out_dir = os.path.abspath(os.path.join("../../../output", task_name, "carbon_price", "1_draw_data"))
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
        data = get_data_RES(task_name, fn)
        targets = get_GBF2_target_list_start_to_2050(
            data=data,
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
    n_jobs=25,                      # 注意：共享文件系统下不一定越大越快
    backend="loky",
    scale=1e6,                      # ✅ 除以 1e6
)