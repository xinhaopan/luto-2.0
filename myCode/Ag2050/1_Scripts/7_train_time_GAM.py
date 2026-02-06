from joblib import Parallel, delayed
from statsmodels.tsa.exponential_smoothing.ets import ETSModel
from scipy.stats import norm
import warnings
import statsmodels.api as sm
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from pygam import LinearGAM, s
from scipy.optimize import least_squares

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pickle
from pathlib import Path
from matplotlib.gridspec import GridSpec

warnings.filterwarnings("ignore")


import numpy as np
import pandas as pd
from statsmodels.tsa.exponential_smoothing.ets import ETSModel

# ======================
# 1) 小工具：sMAPE、clip、外推平滑惩罚
# ======================

def train_single_pair_exp(df_pair: pd.DataFrame,
                          group_name: str,
                          element_name: str,
                          align_to_history: bool = False,
                          direction: str = "auto",
                          clip_q=(0.02, 0.98),
                          eps: float = 1e-6) -> dict:
    """
    指数曲线：y = a * exp(b * (t - t0))   (a>0)
    在 log 空间：log(y+eps) = log(a) + b*(t-t0)

    - direction: "auto" / "inc" / "dec"
    - clip_q: 轻量去尖峰分位数
    - 区间：在 log 空间用残差 std 构造 80%/95% PI，再 exp 回原空间（稳定、可控）
    """

    # 1) 训练期：1990-2014
    train_df = df_pair[(df_pair["year"] >= 1990) & (df_pair["year"] <= 2014)].copy()
    if len(train_df) < 8:
        print(f"[SKIP] {group_name} - {element_name}: 数据不足")
        return None

    train_df = train_df.sort_values("year").reset_index(drop=True)
    years = train_df["year"].astype(int).to_numpy()
    y_raw = train_df["trade"].astype(float).to_numpy()

    # 2) 轻量去尖峰（减少极端年份把指数斜率拉爆）
    lo, hi = np.quantile(y_raw, clip_q)
    y = np.clip(y_raw, lo, hi)

    # 3) 指数拟合要求 y+eps > 0（用 eps 处理 0）
    y_pos = np.maximum(y, 0.0) + eps

    # 4) 准备自变量（中心化，数值更稳）
    t0 = float(years.min())
    x = years.astype(float) - t0

    # 5) 在 log 空间做稳健线性拟合：log(y) = c0 + b*x
    z = np.log(y_pos)

    # 初值：普通最小二乘
    A = np.vstack([np.ones_like(x), x]).T
    c0_init, b_init = np.linalg.lstsq(A, z, rcond=None)[0]

    # 方向约束（可选）
    if direction == "inc":
        b_init = max(b_init, 0.0)
        b_bounds = (0.0, np.inf)
    elif direction == "dec":
        b_init = min(b_init, 0.0)
        b_bounds = (-np.inf, 0.0)
    else:
        b_bounds = (-np.inf, np.inf)

    # 稳健拟合（soft_l1 对异常点不敏感）
    def resid(p):
        c0, b = p
        return (c0 + b * x) - z

    res = least_squares(
        resid,
        x0=np.array([c0_init, b_init], dtype=float),
        bounds=(np.array([-np.inf, b_bounds[0]]), np.array([np.inf, b_bounds[1]])),
        loss="soft_l1",
        f_scale=1.0
    )

    c0_hat, b_hat = res.x

    # 如果 direction=auto，就同时拟合 inc/dec，选残差更小的方向（更稳）
    if direction == "auto":
        # 再拟合一次 inc 和 dec，选 SSE 更小的
        def fit_with_bound(lb, ub, b0):
            r = least_squares(
                resid,
                x0=np.array([c0_init, b0], dtype=float),
                bounds=(np.array([-np.inf, lb]), np.array([np.inf, ub])),
                loss="soft_l1",
                f_scale=1.0
            )
            sse = float(np.sum(resid(r.x)**2))
            return r.x, sse

        x_inc, sse_inc = fit_with_bound(0.0, np.inf, max(b_init, 0.0))
        x_dec, sse_dec = fit_with_bound(-np.inf, 0.0, min(b_init, 0.0))

        if sse_inc <= sse_dec:
            c0_hat, b_hat = x_inc
            chosen_dir = "inc"
        else:
            c0_hat, b_hat = x_dec
            chosen_dir = "dec"
    else:
        chosen_dir = direction

    # 6) 生成 1990-2050 预测（均值）
    start_year = int(years.min())
    end_year = 2050
    full_years = np.arange(start_year, end_year + 1, dtype=int)
    x_full = full_years.astype(float) - t0

    z_hat_full = c0_hat + b_hat * x_full
    mean_full = np.exp(z_hat_full) - eps
    mean_full = np.maximum(mean_full, 0.0)

    # 7) 区间：用 log 空间残差 std（可控，不会爆炸到几十倍）
    z_hat_train = c0_hat + b_hat * x
    resid_z = z - z_hat_train
    sigma = float(np.nanstd(resid_z, ddof=2)) if len(resid_z) > 2 else float(np.nanstd(resid_z))

    # 你可以加一个上限，避免某些序列 sigma 特别大导致区间太宽
    # sigma = min(sigma, 1.0)  # 例如限制 log-std 最大为 1（约等于 *e 的倍数）

    z80 = norm.ppf(0.90)
    z95 = norm.ppf(0.975)

    lower80 = np.exp(z_hat_full - z80 * sigma) - eps
    upper80 = np.exp(z_hat_full + z80 * sigma) - eps
    lower95 = np.exp(z_hat_full - z95 * sigma) - eps
    upper95 = np.exp(z_hat_full + z95 * sigma) - eps

    lower80 = np.maximum(lower80, 0.0)
    lower95 = np.maximum(lower95, 0.0)

    df80 = pd.DataFrame({"Year": full_years, "Mean": mean_full, "Lower": lower80, "Upper": upper80})
    df95 = pd.DataFrame({"Year": full_years, "Mean": mean_full, "Lower": lower95, "Upper": upper95})

    # 8) 可选：对齐到历史末年（建议只用于展示）
    last_hist_year = int(years.max())
    y_last_true = float(y_raw[years == last_hist_year][0])

    if align_to_history:
        pred_last = float(df80.loc[df80["Year"] == last_hist_year, "Mean"].values[0])
        offset = y_last_true - pred_last
        print(f"[TRAIN] {group_name} - {element_name} EXP({chosen_dir})  offset={offset:.2f}")
    else:
        offset = 0.0
        print(f"[TRAIN] {group_name} - {element_name} EXP({chosen_dir})")

    for dfp in (df80, df95):
        for col in ("Mean", "Lower", "Upper"):
            dfp[col] = dfp[col] + offset

    df80["Very_High"] = df95["Upper"]

    return {
        "group": group_name,
        "element": element_name,
        "df_80": df80.reset_index(drop=True),
        "df_95": df95.reset_index(drop=True),
        "last_hist_year": last_hist_year,
        "model_type": "EXP_TREND",
        "aligned": align_to_history,
        "offset": float(offset),
        "best_params": {
            "direction": chosen_dir,
            "c0": float(c0_hat),
            "b": float(b_hat),
            "sigma_log": float(sigma),
            "clip_q": clip_q,
            "eps": eps
        }
    }


def train_one_pair_wrapper(group_name, element_name, df_pair, align_to_history):
    try:
        return train_single_pair_exp(df_pair, group_name, element_name, align_to_history,
                                     direction="auto", clip_q=(0.02, 0.98))
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