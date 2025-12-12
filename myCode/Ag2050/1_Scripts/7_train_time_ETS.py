
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


def train_single_pair_ets(df_pair: pd.DataFrame,
                          group_name: str,
                          element_name: str) -> dict:
    """
    ETS (阻尼趋势) 时间序列模型

    返回预测结果（含预测区间），用于后续画图
    """

    # 1. 准备训练数据 (1990-2014)
    train_df = df_pair[(df_pair['year'] >= 1990) & (df_pair['year'] <= 2014)].copy()

    if len(train_df) < 5:
        print(f"[SKIP] {group_name} - {element_name}:  数据不足")
        return None

    train_df = train_df.sort_values('year').reset_index(drop=True)

    # 2. 构建时间序列（带日期索引）
    y_ts = pd.Series(
        train_df['trade'].values.astype(float),
        index=pd.to_datetime(train_df['year'].astype(str) + '-01-01'),
        name='trade'
    )

    print(f"[TRAIN] {group_name} - {element_name}")

    # 3. 拟合 ETS 模型（阻尼趋势）
    try:
        mod = ETSModel(
            y_ts,
            error='add',  # 加法误差
            trend='add',  # 加法趋势
            damped_trend=True  # 阻尼趋势（防止过度外推）
        )
        res = mod.fit(disp=False)
    except Exception as e:
        print(f"[ERROR] {group_name} - {element_name}: ETS 拟合失败 - {e}")
        return None

    # 4. 获取历史末年的真实值（用于对齐）
    last_hist_year = int(train_df['year'].max())
    y_last = float(train_df.loc[train_df['year'] == last_hist_year, 'trade'].values[0])

    # 5. 预测到 2050 年（含预测区间）
    start_dt = y_ts.index.min()
    end_dt = pd.to_datetime('2050-01-01')

    pred_80 = res.get_prediction(start=start_dt, end=end_dt)
    pred_95 = res.get_prediction(start=start_dt, end=end_dt)

    # 6. 提取预测区间
    def _to_pi_df(pred, level):
        """将 prediction -> DataFrame(Year, Mean, Lower, Upper)"""
        alpha = 1 - level
        sf = pred.summary_frame(alpha=alpha)
        mean = sf['mean'].to_numpy()

        # 尝试获取预测区间
        if {'pi_lower', 'pi_upper'}.issubset(sf.columns):
            lower = sf['pi_lower'].to_numpy()
            upper = sf['pi_upper'].to_numpy()
        elif {'obs_ci_lower', 'obs_ci_upper'}.issubset(sf.columns):
            lower = sf['obs_ci_lower'].to_numpy()
            upper = sf['obs_ci_upper'].to_numpy()
        else:
            # 手动计算预测区间
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

    df80 = _to_pi_df(pred_80, 0.80)  # 80% 预测区间
    df95 = _to_pi_df(pred_95, 0.95)  # 95% 预测区间

    # 7. 对齐历史数据（可选，避免预测与历史跳跃）
    pred_mean_last = float(df80.loc[df80['Year'] == last_hist_year, 'Mean'].values[0])
    offset_mean = y_last - pred_mean_last
    # offset_mean = 0  # 如需关闭对齐，取消注释此行

    for dfp in (df80, df95):
        for col in ['Mean', 'Lower', 'Upper']:
            dfp[col] = dfp[col] + offset_mean

    # 8. Very High = 95% 上界
    df80['Very_High'] = df95['Upper']

    return {
        'group': group_name,
        'element': element_name,
        'df_80': df80,  # 包含 Year, Mean, Lower, Upper
        'df_95': df95,  # 包含 Year, Mean, Lower, Upper
        'last_hist_year': last_hist_year,
        'model_type': 'ETS'
    }


def train_one_pair_wrapper(group_name, element_name, df_pair):
    """
    包装函数，用于并行处理
    """
    try:
        result = train_single_pair_ets(df_pair, group_name, element_name)
        return result
    except Exception as e:
        print(f"[ERROR] {group_name} - {element_name}: {e}")
        return None


def plot_all_exports_ets(df_hist: pd.DataFrame, trained_models: list, save_path: str):
    """
    使用 ETS 训练结果画图，所有Export商品在一张大图上
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

    print(f"布局:  {n_rows} 行 × {n_cols} 列")

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
    print(f"\n✓ 保存到:  {save_path}")

    plt.show()
    plt.close(fig)

    return fig

def main():
    # ========== 配置 ==========
    DATA_PATH = "../2_processed_data/trade_model_data_all.csv"
    OUTPUT_DIR = "../2_processed_data/trained_models_ets"
    N_JOBS = 50  # 并行任务数

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
    print(f"总共 {len(pairs)} 个商品需要训练\n")

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
    print(f"开始并行训练 (n_jobs={N_JOBS})...\n")
    results = Parallel(n_jobs=N_JOBS, verbose=10)(
        delayed(train_one_pair_wrapper)(g, e, df) for g, e, df in tasks
    )

    # ========== 保存结果 ==========
    print("\n保存结果...")
    valid_results = [r for r in results if r is not None]
    print(f"成功训练:  {len(valid_results)}/{len(tasks)}")

    # 保存为单个文件
    output_file = Path(OUTPUT_DIR) / "all_trained_models_ets.pkl"
    with open(output_file, 'wb') as f:
        pickle.dump(valid_results, f)

    print(f"\n✓ 所有结果已保存到:  {output_file}")

    # 显示样例
    print("\n样例结果:")
    for i, result in enumerate(valid_results[:3]):
        df80 = result['df_80']
        print(f"\n[{i + 1}] {result['group']} - {result['element']}")
        print(f"    模型类型: {result['model_type']}")
        print(f"    历史末年:  {result['last_hist_year']}")
        print(f"    预测年份范围: {df80['Year'].min()} - {df80['Year'].max()}")

        # 显示 2050 年的预测
        pred_2050 = df80[df80['Year'] == 2050].iloc[0]
        print(f"    2050年预测: {pred_2050['Mean']:.2f}")
        print(f"    80% CI: [{pred_2050['Lower']:.2f}, {pred_2050['Upper']:.2f}]")
    HIST_DATA_PATH = "../2_processed_data/trade_model_data_all.csv"
    MODEL_FILE = "../2_processed_data/trained_models_ets/all_trained_models_ets.pkl"
    OUTPUT_FILE = "./export_predictions_ets_all.png"

    # ========== 加载数据 ==========
    print("加载数据...")

    # 历史数据
    df_hist = pd.read_csv(HIST_DATA_PATH)
    df_hist = df_hist[df_hist["Report ISO"].str.upper() == "AUS"].copy()
    df_hist["year"] = pd.to_numeric(df_hist["year"], errors="coerce")
    df_hist["trade"] = pd.to_numeric(df_hist["trade"], errors="coerce")
    df_hist = df_hist.dropna(subset=["year", "trade"])

    # 聚合历史数据
    df_hist_agg = df_hist.groupby(["group", "Element", "year"], as_index=False)["trade"].sum()
    print(f"历史数据: {len(df_hist_agg)} 行")

    # ETS 训练结果
    if not Path(MODEL_FILE).exists():
        print(f"错误: 找不到模型文件 {MODEL_FILE}")
        print("请先运行 ETS 训练代码!")
        return

    with open(MODEL_FILE, 'rb') as f:
        trained_models = pickle.load(f)

    print(f"训练模型:  {len(trained_models)} 个\n")

    # ========== 画图 ==========
    plot_all_exports_ets(df_hist_agg, trained_models, OUTPUT_FILE)

    print("\n完成！")

if __name__ == "__main__":
    main()