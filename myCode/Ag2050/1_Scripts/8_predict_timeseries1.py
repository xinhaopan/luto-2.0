import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import warnings

warnings.filterwarnings("ignore")


def classify_trend_pattern(y_data):
    """
    自动分类商品的趋势模式
    """
    if len(y_data) < 10:
        return 'insufficient'

    # 计算整体趋势
    overall_trend = np.polyfit(np.arange(len(y_data)), y_data, 1)[0]

    # 计算前半部分和后半部分的趋势
    mid = len(y_data) // 2
    first_half_trend = np.polyfit(np.arange(mid), y_data[:mid], 1)[0]
    second_half_trend = np.polyfit(np.arange(mid), y_data[mid:], 1)[0]

    # 计算波动率
    volatility = np.std(np.diff(y_data)) / (np.mean(y_data) + 1)

    # 分类逻辑
    if volatility > 0.3:
        return 'highly_volatile'
    elif abs(overall_trend) < 0.01 * np.mean(y_data):
        return 'stable'
    elif first_half_trend * second_half_trend < 0:
        return 'trend_reversal'
    elif abs(second_half_trend) > 2 * abs(first_half_trend):
        return 'accelerating'
    elif overall_trend > 0:
        return 'growing'
    else:
        return 'declining'


def smart_forecast(df_pair, group_name, element_name):
    """
    根据数据特征自动选择最合适的预测方法
    """
    train_df = df_pair[(df_pair['year'] >= 1990) & (df_pair['year'] <= 2014)].copy()

    if len(train_df) < 5:
        return None

    train_df = train_df.sort_values('year').reset_index(drop=True)
    years = train_df['year'].values
    y_data = train_df['trade'].values.astype(float)

    # 分类趋势模式
    pattern = classify_trend_pattern(y_data)

    print(f"[{group_name: 30s}] 模式:  {pattern}")

    last_year = int(years[-1])
    last_value = float(y_data[-1])

    # 根据模式选择策略
    if pattern == 'highly_volatile':
        # 高波动：使用中位数 + 最近趋势
        lookback = 5
        recent_trend = np.median(np.diff(y_data[-lookback:]))

        future_years = np.arange(2015, 2051)
        predictions = last_value + recent_trend * (future_years - last_year)
        predictions = np.maximum(predictions, 0)

        std = np.std(y_data) * 1.5  # 更大的不确定性

    elif pattern == 'stable':
        # 稳定：保持当前水平 + 小幅波动
        future_years = np.arange(2015, 2051)
        predictions = np.full_like(future_years, last_value, dtype=float)

        std = np.std(y_data[-10:])

    elif pattern == 'trend_reversal':
        # 趋势反转：只用最近10年
        lookback = min(10, len(y_data))
        recent_years = years[-lookback:]
        recent_values = y_data[-lookback:]

        slope, intercept = np.polyfit(recent_years, recent_values, 1)

        future_years = np.arange(2015, 2051)
        predictions = slope * future_years + intercept
        predictions = np.maximum(predictions, 0)

        std = np.std(recent_values - (slope * recent_years + intercept))

    elif pattern == 'accelerating':
        # 加速增长：二次多项式（但限制外推）
        coeffs = np.polyfit(years[-10:], y_data[-10:], 2)

        # 短期用二次，长期用线性
        short_term_years = np.arange(2015, 2031)
        short_term_pred = np.polyval(coeffs, short_term_years)

        # 长期：用2030年的值 + 线性外推
        long_term_years = np.arange(2031, 2051)
        long_term_slope = coeffs[1] + 2 * coeffs[0] * 2030  # 导数
        long_term_pred = short_term_pred[-1] + long_term_slope * (long_term_years - 2030)

        future_years = np.arange(2015, 2051)
        predictions = np.concatenate([short_term_pred, long_term_pred])
        predictions = np.maximum(predictions, 0)

        std = np.std(y_data[-10:]) * 1.2

    else:  # 'growing' or 'declining'
        # 正常增长/下降：使用最近10年线性趋势
        lookback = min(10, len(y_data))
        recent_years = years[-lookback:]
        recent_values = y_data[-lookback:]

        slope, intercept = np.polyfit(recent_years, recent_values, 1)

        future_years = np.arange(2015, 2051)
        predictions = slope * future_years + intercept
        predictions = np.maximum(predictions, 0)

        std = np.std(recent_values - (slope * recent_years + intercept))

    # 构建完整时间序列（包括历史）
    all_years = np.arange(1990, 2051)
    all_predictions = np.concatenate([y_data, predictions])

    # 计算预测区间
    n_hist = len(y_data)
    lower_80 = np.concatenate(
        [y_data - 0.5 * std, predictions - 1.28 * std * np.sqrt(1 + np.arange(len(predictions)) / 10)])
    upper_80 = np.concatenate(
        [y_data + 0.5 * std, predictions + 1.28 * std * np.sqrt(1 + np.arange(len(predictions)) / 10)])

    lower_95 = np.concatenate(
        [y_data - 1.0 * std, predictions - 1.96 * std * np.sqrt(1 + np.arange(len(predictions)) / 10)])
    upper_95 = np.concatenate(
        [y_data + 1.0 * std, predictions + 1.96 * std * np.sqrt(1 + np.arange(len(predictions)) / 10)])

    lower_80 = np.maximum(lower_80, 0)
    lower_95 = np.maximum(lower_95, 0)

    df80 = pd.DataFrame({
        'Year': all_years,
        'Mean': all_predictions,
        'Lower': lower_80,
        'Upper': upper_80
    })

    df95 = pd.DataFrame({
        'Year': all_years,
        'Mean': all_predictions,
        'Lower': lower_95,
        'Upper': upper_95
    })

    return {
        'group': group_name,
        'element': element_name,
        'df_80': df80,
        'df_95': df95,
        'last_hist_year': last_year,
        'model_type': f'Smart_{pattern}',
        'pattern': pattern
    }


def train_all_smart(df_agg):
    """��练所有出口商品"""
    df_export = df_agg[df_agg['Element'] == 'Export Quantity'].copy()
    pairs = df_export[['group']].drop_duplicates()

    print(f"总共 {len(pairs)} 个出口商品\n")

    results = []
    pattern_counts = {}

    for _, row in pairs.iterrows():
        group_name = row['group']
        element_name = 'Export Quantity'
        df_pair = df_export[df_export['group'] == group_name].copy()

        try:
            result = smart_forecast(df_pair, group_name, element_name)
            if result:
                results.append(result)
                pattern = result['pattern']
                pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
        except Exception as e:
            print(f"  ✗ 失败: {e}")

    print(f"\n成功:  {len(results)}/{len(pairs)}")
    print("\n模式分布:")
    for pattern, count in sorted(pattern_counts.items()):
        print(f"  {pattern: 20s}: {count}")

    return results


def plot_all_exports(df_hist, trained_models, save_path):
    """画图"""
    element_name = 'Export Quantity'
    export_models = sorted(trained_models, key=lambda x: x['group'])

    n_groups = len(export_models)
    n_cols = 6
    n_rows = int(np.ceil(n_groups / n_cols))

    fig = plt.figure(figsize=(24, 3.5 * n_rows))
    gs = GridSpec(n_rows, n_cols, figure=fig, hspace=0.35, wspace=0.25,
                  left=0.04, right=0.98, top=0.98, bottom=0.05)

    for idx, model_result in enumerate(export_models):
        row = idx // n_cols
        col = idx % n_cols
        ax = fig.add_subplot(gs[row, col])

        group_name = model_result['group']
        df80 = model_result['df_80']
        df95 = model_result['df_95']
        last_hist_year = model_result['last_hist_year']
        pattern = model_result.get('pattern', '')

        hist_data = df_hist[
            (df_hist['group'] == group_name) &
            (df_hist['Element'] == element_name) &
            (df_hist['year'] <= 2014)
            ].copy()

        if not hist_data.empty:
            hist_data = hist_data.sort_values('year')

        # 画预测区间
        env95 = df95[df95['Year'] > last_hist_year].copy()
        if not env95.empty:
            ax.fill_between(env95['Year'].to_numpy(), env95['Lower'].to_numpy(),
                            env95['Upper'].to_numpy(), color='#082b6a', alpha=0.12, zorder=1)

        env80 = df80[df80['Year'] > last_hist_year].copy()
        if not env80.empty:
            ax.fill_between(env80['Year'].to_numpy(), env80['Lower'].to_numpy(),
                            env80['Upper'].to_numpy(), color='#0b3d91', alpha=0.18, zorder=2)

        # 画预测线
        if not df80.empty:
            ax.plot(df80['Year'], df80['Mean'], color='darkblue', linewidth=2, zorder=3)

        # 画历史数据
        if not hist_data.empty:
            ax.scatter(hist_data['year'], hist_data['trade'], color='black', s=30, zorder=5, alpha=0.8)

        # 标题包含模式
        title = f"{group_name}\n({pattern})" if pattern else group_name
        ax.set_title(title, fontsize=8, fontweight='bold', pad=6)
        ax.grid(True, alpha=0.25, linestyle='-', linewidth=0.5)
        ax.tick_params(labelsize=7)

        x_min = hist_data['year'].min() - 2 if not hist_data.empty else 1988
        ax.set_xlim(x_min, 2052)
        ax.set_ylim(bottom=0)

    # 图例
    legend_placed = False
    for idx in range(n_groups, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        ax = fig.add_subplot(gs[row, col])

        if not legend_placed and row == n_rows - 1:
            ax.axis('off')
            from matplotlib.patches import Patch
            from matplotlib.lines import Line2D

            legend_elements = [
                Line2D([0], [0], marker='o', color='w', markerfacecolor='black',
                       markersize=8, label='Historical Data (≤2014)'),
                Line2D([0], [0], color='darkblue', linewidth=2.5, label='Smart Forecast'),
                Patch(facecolor='#0b3d91', alpha=0.18, label='80% Prediction Interval'),
                Patch(facecolor='#082b6a', alpha=0.12, label='95% Prediction Interval'),
            ]

            ax.legend(handles=legend_elements, loc='lower right', fontsize=11,
                      frameon=True, framealpha=0.95, ncol=2, bbox_to_anchor=(1.6, 0.02))
            legend_placed = True
        else:
            ax.axis('off')

    plt.savefig(save_path, dpi=150, bbox_inches='tight', pad_inches=0.1)
    print(f"\n✓ 保存:  {save_path}")
    plt.close(fig)


def main():
    DATA_PATH = "../2_processed_data/trade_model_data_all.csv"
    OUTPUT_DIR = "./smart_forecast"

    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    # 加载数据
    print("加载数据...")
    df = pd.read_csv(DATA_PATH)
    df = df[df["Report ISO"].str.upper() == "AUS"].copy()
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df["trade"] = pd.to_numeric(df["trade"], errors="coerce")
    df = df.dropna(subset=["year", "trade"])

    df_agg = df.groupby(["group", "Element", "year"], as_index=False)["trade"].sum()

    # 训练
    results = train_all_smart(df_agg)

    # 保存
    import pickle
    with open(Path(OUTPUT_DIR) / "smart_models.pkl", 'wb') as f:
        pickle.dump(results, f)

    # 画图
    plot_all_exports(df_agg, results, Path(OUTPUT_DIR) / "smart_forecast.png")

    print("\n完成!")


if __name__ == "__main__":
    main()