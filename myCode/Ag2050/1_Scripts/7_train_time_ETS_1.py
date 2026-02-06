from joblib import Parallel, delayed
from statsmodels.tsa.exponential_smoothing.ets import ETSModel
from scipy.stats import norm
import warnings

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pickle
from pathlib import Path
from matplotlib.gridspec import GridSpec

warnings.filterwarnings("ignore")


import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import least_squares
from scipy.stats import spearmanr
from statsmodels.tsa.exponential_smoothing.ets import ETSModel


# =========================
# 1) ETS 训练（支持 use_log）
# =========================
def train_single_pair_ets(df_pair: pd.DataFrame,
                          group_name: str,
                          element_name: str,
                          align_to_history: bool = True,
                          use_log: bool = False) -> dict:
    """
    ETS (阻尼趋势) 模型
    - use_log=False: 直接在原空间拟合
    - use_log=True : 传入的数据应已是 log1p(trade)，函数内部会在返回前 expm1 还原
                    （为避免对齐单位混乱，use_log=True 时建议 align_to_history=False）
    """

    # 1. 训练数据 (1990-2014)
    train_df = df_pair[(df_pair['year'] >= 1990) & (df_pair['year'] <= 2014)].copy()
    if len(train_df) < 5:
        print(f"[SKIP] {group_name} - {element_name}: 数据不足")
        return None

    train_df = train_df.sort_values('year').reset_index(drop=True)

    # 2. 时间序列
    y_ts = pd.Series(
        train_df['trade'].values.astype(float),
        index=pd.to_datetime(train_df['year'].astype(str) + '-01-01'),
        name='trade'
    )

    print(f"[TRAIN] {group_name} - {element_name} | ETS{'(log1p)' if use_log else ''}")

    # 3. 拟合 ETS
    try:
        mod = ETSModel(
            y_ts,
            error='add',
            trend='add',
            damped_trend=True
        )
        res = mod.fit(disp=False)
    except Exception as e:
        print(f"[ERROR] {group_name} - {element_name}: ETS 拟合失败 - {e}")
        return None

    # 4. 历史末年真实值（用于对齐；log 模式下默认不对齐）
    last_hist_year = int(train_df['year'].max())
    y_last = float(train_df.loc[train_df['year'] == last_hist_year, 'trade'].values[0])

    # 5. 预测到 2050
    start_dt = y_ts.index.min()
    end_dt = pd.to_datetime('2050-01-01')

    pred_80 = res.get_prediction(start=start_dt, end=end_dt)
    pred_95 = res.get_prediction(start=start_dt, end=end_dt)

    # 6. 提取区间
    def _to_pi_df(pred, level):
        alpha = 1 - level
        sf = pred.summary_frame(alpha=alpha)
        mean = sf['mean'].to_numpy()

        if {'pi_lower', 'pi_upper'}.issubset(sf.columns):
            lower = sf['pi_lower'].to_numpy()
            upper = sf['pi_upper'].to_numpy()
        elif {'obs_ci_lower', 'obs_ci_upper'}.issubset(sf.columns):
            lower = sf['obs_ci_lower'].to_numpy()
            upper = sf['obs_ci_upper'].to_numpy()
        else:
            z = norm.ppf(1 - alpha / 2.0)
            if 'mean_se' in sf.columns:
                mean_se = sf['mean_se'].to_numpy()
            elif hasattr(pred, 'var_pred_mean'):
                mean_se = np.sqrt(np.asarray(pred.var_pred_mean))
            else:
                mean_se = np.zeros_like(mean)

            sigma2 = float(res.params.get('sigma2', res.scale)) if hasattr(res, 'params') else float(res.scale)
            half_w = z * np.sqrt(mean_se ** 2 + sigma2)
            lower = mean - half_w
            upper = mean + half_w

        out = pd.DataFrame({
            'Year': sf.index.year,
            'Mean': mean,
            'Lower': lower,
            'Upper': upper
        }).drop_duplicates(subset=['Year']).reset_index(drop=True)
        return out

    df80 = _to_pi_df(pred_80, 0.80)
    df95 = _to_pi_df(pred_95, 0.95)

    # 7) 如果 use_log=True：先还原到原空间（Mean/Lower/Upper 都要还原）
    if use_log:
        for dfp in (df80, df95):
            for col in ['Mean', 'Lower', 'Upper']:
                dfp[col] = np.expm1(dfp[col])
                dfp[col] = np.maximum(dfp[col], 0.0)  # 防负
        # log 模式下 y_last 是 log 值，不适合直接用 offset 对齐
        align_to_history = False

    # 8) 对齐（原空间）
    if align_to_history:
        pred_mean_last = float(df80.loc[df80['Year'] == last_hist_year, 'Mean'].values[0])
        offset_mean = (y_last - pred_mean_last)
        print(f"    → 对齐偏移量: {offset_mean:.2f}")
    else:
        offset_mean = 0.0

    for dfp in (df80, df95):
        for col in ['Mean', 'Lower', 'Upper']:
            dfp[col] = dfp[col] + offset_mean

    # 9) Very High
    df80['Very_High'] = df95['Upper']

    return {
        'group': group_name,
        'element': element_name,
        'df_80': df80,
        'df_95': df95,
        'last_hist_year': last_hist_year,
        'model_type': 'ETS_LOG' if use_log else 'ETS',
        'aligned': align_to_history,
        'offset': float(offset_mean)
    }


# =======================================
# 2) 指数趋势（稳健）用于单调平滑外推
# =======================================
def train_single_pair_exp(df_pair: pd.DataFrame,
                          group_name: str,
                          element_name: str,
                          align_to_history: bool = False,
                          direction: str = "auto",
                          clip_q=(0.02, 0.98),
                          eps: float = 1e-6) -> dict:
    """
    指数趋势：y = exp(c0 + b*(year - year0)) - eps
    在 log 空间稳健拟合（soft_l1），自动选择递增/递减（direction="auto"）
    区间：log 空间残差 std 构造 80/95%（不会像 GAM 那样爆炸）
    """

    train_df = df_pair[(df_pair["year"] >= 1990) & (df_pair["year"] <= 2014)].copy()
    if len(train_df) < 8:
        print(f"[SKIP] {group_name} - {element_name}: 数据不足")
        return None

    train_df = train_df.sort_values("year").reset_index(drop=True)
    years = train_df["year"].astype(int).to_numpy()
    y_raw = train_df["trade"].astype(float).to_numpy()

    lo, hi = np.quantile(y_raw, clip_q)
    y = np.clip(y_raw, lo, hi)
    y_pos = np.maximum(y, 0.0) + eps

    year0 = float(years.min())
    x = years.astype(float) - year0
    z = np.log(y_pos)

    # 初值
    A = np.vstack([np.ones_like(x), x]).T
    c0_init, b_init = np.linalg.lstsq(A, z, rcond=None)[0]

    def resid(p):
        c0, b = p
        return (c0 + b * x) - z

    def fit_with_bounds(lb, ub, b0):
        r = least_squares(
            resid,
            x0=np.array([c0_init, b0], float),
            bounds=(np.array([-np.inf, lb]), np.array([np.inf, ub])),
            loss="soft_l1",
            f_scale=1.0
        )
        sse = float(np.sum(resid(r.x) ** 2))
        return r.x, sse

    if direction == "auto":
        x_inc, sse_inc = fit_with_bounds(0.0, np.inf, max(b_init, 0.0))
        x_dec, sse_dec = fit_with_bounds(-np.inf, 0.0, min(b_init, 0.0))
        if sse_inc <= sse_dec:
            c0_hat, b_hat = x_inc
            chosen = "inc"
        else:
            c0_hat, b_hat = x_dec
            chosen = "dec"
    elif direction == "inc":
        (c0_hat, b_hat), _ = fit_with_bounds(0.0, np.inf, max(b_init, 0.0))
        chosen = "inc"
    else:
        (c0_hat, b_hat), _ = fit_with_bounds(-np.inf, 0.0, min(b_init, 0.0))
        chosen = "dec"

    print(f"[TRAIN] {group_name} - {element_name} | EXP({chosen})")

    # 预测 1990-2050
    start_year = int(years.min())
    full_years = np.arange(start_year, 2050 + 1, dtype=int)
    x_full = full_years.astype(float) - year0

    z_hat_full = c0_hat + b_hat * x_full
    mean = np.exp(z_hat_full) - eps
    mean = np.maximum(mean, 0.0)

    # 区间（log残差）
    z_hat_train = c0_hat + b_hat * x
    resid_z = z - z_hat_train
    sigma = float(np.nanstd(resid_z, ddof=2)) if len(resid_z) > 2 else float(np.nanstd(resid_z))

    z80 = norm.ppf(0.90)
    z95 = norm.ppf(0.975)

    lower80 = np.exp(z_hat_full - z80 * sigma) - eps
    upper80 = np.exp(z_hat_full + z80 * sigma) - eps
    lower95 = np.exp(z_hat_full - z95 * sigma) - eps
    upper95 = np.exp(z_hat_full + z95 * sigma) - eps

    lower80 = np.maximum(lower80, 0.0)
    lower95 = np.maximum(lower95, 0.0)

    df80 = pd.DataFrame({"Year": full_years, "Mean": mean, "Lower": lower80, "Upper": upper80})
    df95 = pd.DataFrame({"Year": full_years, "Mean": mean, "Lower": lower95, "Upper": upper95})

    last_hist_year = int(years.max())
    y_last_true = float(y_raw[years == last_hist_year][0])

    if align_to_history:
        pred_last = float(df80.loc[df80["Year"] == last_hist_year, "Mean"].values[0])
        offset = y_last_true - pred_last
    else:
        offset = 0.0

    for dfp in (df80, df95):
        for col in ("Mean", "Lower", "Upper"):
            dfp[col] = dfp[col] + offset

    df80["Very_High"] = df95["Upper"]

    return {
        "group": group_name,
        "element": element_name,
        "df_80": df80,
        "df_95": df95,
        "last_hist_year": last_hist_year,
        "model_type": "EXP_TREND",
        "aligned": align_to_history,
        "offset": float(offset),
        "best_params": {"direction": chosen, "b": float(b_hat), "sigma_log": float(sigma)}
    }


# =========================
# 3) 诊断 + 选模型
# =========================
def diagnose_series(years, y):
    y = np.asarray(y, float)
    years = np.asarray(years, int)

    mask = np.isfinite(y)
    y = y[mask]
    years = years[mask]
    if len(y) < 8:
        return {
            "level_shift": False,
            "near_zero": False,
            "heterosk": False,
            "roughly_monotone": False,
            "tail_dominated": False
        }

    # level shift（前后均值）
    mid = len(y) // 2
    mean_ratio = (np.mean(y[mid:]) + 1e-6) / (np.mean(y[:mid]) + 1e-6)
    has_level_shift = (mean_ratio > 1.5) or (mean_ratio < 0.67)

    # near zero
    near_zero_frac = np.mean(y < 0.05 * np.max(y))
    has_near_zero = near_zero_frac > 0.15

    # heterosk（相对波动）
    rel_var = np.std(y) / (np.mean(y) + 1e-6)
    has_heterosk = rel_var > 0.6

    # monotone（Spearman）
    rho, _ = spearmanr(years, y)
    roughly_monotone = abs(rho) > 0.6

    # tail dominated（末段跳变）
    tail = y[-5:]
    tail_jump = np.max(np.abs(np.diff(tail))) / (np.mean(y) + 1e-6)
    tail_dominated = tail_jump > 0.8

    return {
        "level_shift": has_level_shift,
        "near_zero": has_near_zero,
        "heterosk": has_heterosk,
        "roughly_monotone": roughly_monotone,
        "tail_dominated": tail_dominated
    }


def choose_model(diag):
    if diag["level_shift"] or diag["near_zero"]:
        return "EXP_OR_MONOTONE"
    if diag["roughly_monotone"] and not diag["tail_dominated"]:
        return "EXP"
    if diag["heterosk"]:
        return "ETS_LOG"
    return "ETS_SIMPLE"


# =========================
# 4) Wrapper：避免未来泄漏 + 正确处理 log
# =========================
def train_one_pair_wrapper(group_name, element_name, df_pair, align_to_history):
    try:
        # 只用训练期做诊断（避免未来泄漏）
        df_train = df_pair[(df_pair["year"] >= 1990) & (df_pair["year"] <= 2014)].copy()
        if len(df_train) < 5:
            return None

        years = df_train["year"].values
        y = df_train["trade"].values

        diag = diagnose_series(years, y)
        choice = choose_model(diag)

        if choice in ("EXP", "EXP_OR_MONOTONE"):
            return train_single_pair_exp(df_pair, group_name, element_name, align_to_history=False)

        if choice == "ETS_LOG":
            df_log = df_pair.copy()
            df_log["trade"] = np.log1p(df_log["trade"].clip(lower=0))
            # log 模式下默认不对齐（单位问题），返回前会还原
            return train_single_pair_ets(df_log, group_name, element_name,
                                         align_to_history=False, use_log=True)

        # ETS_SIMPLE
        return train_single_pair_ets(df_pair, group_name, element_name, align_to_history=align_to_history, use_log=False)

    except Exception as e:
        print(f"[ERROR] {group_name} - {element_name}: {e}")
        return None


def plot_all_exports_ets(df_hist: pd.DataFrame, trained_models: list, save_path: str):
    """
    使用 ETS 训练结果画图，所有Export商品在一张大图上
    y轴最小值设为0
    """

    # 只选Export Quantity
    element_name = 'Export Quantity'

    # 筛选 Export 商品
    export_models = [m for m in trained_models if m['element'] == element_name]
    export_models = sorted(export_models, key=lambda x: x['group'])

    n_groups = len(export_models)
    print(f"总共 {n_groups} 个Export商品")

    # 计算子图布局
    n_cols = 6
    n_rows = int(np.ceil(n_groups / n_cols))

    print(f"布局:   {n_rows} 行 × {n_cols} 列")

    # 创建大图
    fig = plt.figure(figsize=(24, 3.5 * n_rows))
    gs = GridSpec(n_rows, n_cols, figure=fig, hspace=0.35, wspace=0.25,
                  left=0.04, right=0.98, top=0.98, bottom=0.05)

    # 为每个商品画子图
    for idx, model_result in enumerate(export_models):
        row = idx // n_cols
        col = idx % n_cols
        ax = fig.add_subplot(gs[row, col])

        group_name = model_result['group']
        df80 = model_result['df_80']
        df95 = model_result['df_95']
        last_hist_year = model_result['last_hist_year']

        # 筛选历史数据
        hist_data = df_hist[
            (df_hist['group'] == group_name) &
            (df_hist['Element'] == element_name) &
            (df_hist['year'] <= 2014)
            ].copy()

        if hist_data.empty and df80.empty:
            ax.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(group_name, fontsize=9, fontweight='bold')
            continue

        # 排序
        if not hist_data.empty:
            hist_data = hist_data.sort_values('year')

        # 1. 画预测区间（只画 2015 之后的）
        # 95% 预测区间（浅灰色，底层）
        env95 = df95[df95['Year'] > last_hist_year].copy()
        if not env95.empty:
            ax.fill_between(
                env95['Year'].to_numpy(),
                env95['Lower'].to_numpy(),
                env95['Upper'].to_numpy(),
                color='#082b6a', alpha=0.12, zorder=1
            )

        # 80% 预测区间（灰色，上层）
        env80 = df80[df80['Year'] > last_hist_year].copy()
        if not env80.empty:
            ax.fill_between(
                env80['Year'].to_numpy(),
                env80['Lower'].to_numpy(),
                env80['Upper'].to_numpy(),
                color='#0b3d91', alpha=0.18, zorder=2
            )

        # 2. 画预测均值线（深蓝色，从历史起始到 2050）
        if not df80.empty:
            ax.plot(df80['Year'], df80['Mean'],
                    color='darkblue', linewidth=2, zorder=3)

        # 3. 画历史数据（黑色点，在最上层）
        if not hist_data.empty:
            ax.scatter(hist_data['year'], hist_data['trade'],
                       color='black', s=30, zorder=5, alpha=0.8)

        # 4. 只保留标题（商品名称）
        ax.set_title(group_name, fontsize=9, fontweight='bold', pad=6)

        # 5. 网格
        ax.grid(True, alpha=0.25, linestyle='-', linewidth=0.5)

        # 6. 调整刻度字体大小
        ax.tick_params(labelsize=7)

        # 7. 设置x轴范围
        if not hist_data.empty:
            x_min = hist_data['year'].min() - 2
        else:
            x_min = 1988
        ax.set_xlim(x_min, 2052)

        # 8. 设置y轴最小值为0
        ax.set_ylim(bottom=0)

    # 隐藏多余的空白子图，并在最后一行放图例
    legend_placed = False
    for idx in range(n_groups, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        ax = fig.add_subplot(gs[row, col])

        # 在最后一行中间放图例
        if not legend_placed and row == n_rows - 1:
            ax.axis('off')
            # 创建图例元素
            from matplotlib.patches import Patch
            from matplotlib.lines import Line2D

            legend_elements = [
                Line2D([0], [0], marker='o', color='w', markerfacecolor='black',
                       markersize=8, label='Historical Data (≤2014)'),
                Line2D([0], [0], color='darkblue', linewidth=2.5, label='ETS Prediction'),
                Patch(facecolor='#0b3d91', alpha=0.18, label='80% Prediction Interval'),
                Patch(facecolor='#082b6a', alpha=0.12, label='95% Prediction Interval'),
            ]

            ax.legend(handles=legend_elements, loc='lower right', fontsize=11,
                      frameon=True, framealpha=0.95, ncol=2, bbox_to_anchor=(1.6, 0.02))
            legend_placed = True
        else:
            ax.axis('off')

    # 保存
    plt.savefig(save_path, dpi=150, bbox_inches='tight', pad_inches=0.1)
    print(f"\n✓ 保存到:   {save_path}")

    plt.show()
    plt.close(fig)

    return fig


def main():
    # ========== 配置 ==========
    DATA_PATH = "../2_processed_data/trade_model_data_all.csv"
    OUTPUT_DIR = "../2_processed_data/trained_models_ets"
    N_JOBS = 50  # 并行任务数

    # ========== 新增：对齐开关 ==========
    ALIGN_TO_HISTORY = False  # True:  对齐到历史末年真实值; False: 使用原始预测
    # ==================================

    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    # ========== 加载数据 ==========
    print("加载数据...")
    df = pd.read_csv(DATA_PATH)
    df = df[df["Report ISO"].str.upper() == "AUS"].copy()
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df["trade"] = pd.to_numeric(df["trade"], errors="coerce")
    df = df.dropna(subset=["year", "trade"])

    # 聚合数据
    df_agg = df.groupby(["group", "Element", "year"], as_index=False)["trade"].sum()
    print(f"聚合后数据: {len(df_agg)} 行")

    # 获取所有 (group, element) 组合
    pairs = df_agg[["group", "Element"]].drop_duplicates()
    print(f"总共 {len(pairs)} 个商品需要训练")
    print(f"对��模式: {'启用' if ALIGN_TO_HISTORY else '禁用'}\n")

    # ========== 准备任务 ==========
    tasks = []
    for _, row in pairs.iterrows():
        group_name = row["group"]
        element_name = row["Element"]
        df_pair = df_agg[
            (df_agg["group"] == group_name) &
            (df_agg["Element"] == element_name)
            ].copy()
        tasks.append((group_name, element_name, df_pair))

    # ========== 并行训练 ==========
    print(f"开始并行训练 (n_jobs={N_JOBS}).. .\n")
    results = Parallel(n_jobs=N_JOBS, verbose=10)(
        delayed(train_one_pair_wrapper)(g, e, df, ALIGN_TO_HISTORY) for g, e, df in tasks
    )

    # ========== 保存结果 ==========
    print("\n保存结果...")
    valid_results = [r for r in results if r is not None]
    print(f"成功训练:   {len(valid_results)}/{len(tasks)}")

    # 保存为单个文件
    align_suffix = "_aligned" if ALIGN_TO_HISTORY else "_raw"
    output_file = Path(OUTPUT_DIR) / f"all_trained_models_ets{align_suffix}.pkl"
    with open(output_file, 'wb') as f:
        pickle.dump(valid_results, f)

    print(f"\n✓ 所有结���已保存到:   {output_file}")

    # 显示样例
    print("\n样例结果:")
    for i, result in enumerate(valid_results[:3]):
        df80 = result['df_80']
        print(f"\n[{i + 1}] {result['group']} - {result['element']}")
        print(f"    模型类型: {result['model_type']}")
        print(f"    历史末年:   {result['last_hist_year']}")
        print(f"    是否对齐: {result['aligned']}")
        print(f"    偏移量:  {result['offset']:.2f}")
        print(f"    预测年份范围: {df80['Year'].min()} - {df80['Year'].max()}")

        # 显示 2050 年的预测
        pred_2050 = df80[df80['Year'] == 2050].iloc[0]
        print(f"    2050年预测: {pred_2050['Mean']:.2f}")
        print(f"    80% CI: [{pred_2050['Lower']:.2f}, {pred_2050['Upper']:.2f}]")

    # ========== 画图 ==========
    HIST_DATA_PATH = "../2_processed_data/trade_model_data_all.csv"
    OUTPUT_FILE = f"./export_predictions_ets_all{align_suffix}.png"

    print("\n加载历史数据用于画图...")

    # 历史数据
    df_hist = pd.read_csv(HIST_DATA_PATH)
    df_hist = df_hist[df_hist["Report ISO"].str.upper() == "AUS"].copy()
    df_hist["year"] = pd.to_numeric(df_hist["year"], errors="coerce")
    df_hist["trade"] = pd.to_numeric(df_hist["trade"], errors="coerce")
    df_hist = df_hist.dropna(subset=["year", "trade"])

    # 聚合历史数据
    df_hist_agg = df_hist.groupby(["group", "Element", "year"], as_index=False)["trade"].sum()
    print(f"历史数据: {len(df_hist_agg)} 行")

    print(f"训练模型:   {len(valid_results)} 个\n")

    # ========== 画图 ==========
    plot_all_exports_ets(df_hist_agg, valid_results, OUTPUT_FILE)

    print("\n完成！")


if __name__ == "__main__":
    main()