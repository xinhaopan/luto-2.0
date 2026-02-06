import numpy as np
import pandas as pd
import pickle
from pathlib import Path
import warnings
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from statsmodels.tsa.exponential_smoothing.ets import ETSModel
from statsmodels.tsa.arima.model import ARIMA
from scipy.stats import norm
from joblib import Parallel, delayed

warnings.filterwarnings("ignore")


# ==================== 模型 1:  ETS ====================
def train_ets(df_pair, group_name, element_name):
    """ETS (阻尼趋势)"""
    train_df = df_pair[(df_pair['year'] >= 1990) & (df_pair['year'] <= 2014)].copy()

    if len(train_df) < 5:
        return None

    train_df = train_df.sort_values('year').reset_index(drop=True)

    y_ts = pd.Series(
        train_df['trade'].values.astype(float),
        index=pd.to_datetime(train_df['year'].astype(str) + '-01-01'),
        name='trade'
    )

    try:
        mod = ETSModel(y_ts, error='add', trend='add', damped_trend=True)
        res = mod.fit(disp=False)
    except Exception as e:
        return None

    last_hist_year = int(train_df['year'].max())

    # 预测到2050
    start_dt = y_ts.index.min()
    end_dt = pd.to_datetime('2050-01-01')
    pred_80 = res.get_prediction(start=start_dt, end=end_dt)
    pred_95 = res.get_prediction(start=start_dt, end=end_dt)

    def _to_df(pred, level):
        alpha = 1 - level
        sf = pred.summary_frame(alpha=alpha)
        mean = sf['mean'].to_numpy()

        if {'pi_lower', 'pi_upper'}.issubset(sf.columns):
            lower, upper = sf['pi_lower'].to_numpy(), sf['pi_upper'].to_numpy()
        else:
            z = norm.ppf(1 - alpha / 2.0)
            sigma2 = float(res.params.get('sigma2', res.scale)) if hasattr(res, 'params') else float(res.scale)
            half_w = z * np.sqrt(sigma2)
            lower, upper = mean - half_w, mean + half_w

        return pd.DataFrame({
            'Year': sf.index.year,
            'Mean': mean,
            'Lower': lower,
            'Upper': upper
        }).drop_duplicates(subset=['Year']).reset_index(drop=True)

    df80 = _to_df(pred_80, 0.80)
    df95 = _to_df(pred_95, 0.95)

    return {
        'group': group_name,
        'element': element_name,
        'df_80': df80,
        'df_95': df95,
        'last_hist_year': last_hist_year,
        'model_type': 'ETS'
    }


# ==================== 模型 2: ARIMA ====================
def train_arima(df_pair, group_name, element_name):
    """ARIMA 自动选择参数"""
    train_df = df_pair[(df_pair['year'] >= 1990) & (df_pair['year'] <= 2014)].copy()

    if len(train_df) < 5:
        return None

    train_df = train_df.sort_values('year').reset_index(drop=True)
    y_data = train_df['trade'].values.astype(float)

    try:
        # 自动选择 ARIMA(p,d,q)，d=1 表示一阶差分
        mod = ARIMA(y_data, order=(1, 1, 1))
        res = mod.fit()
    except Exception as e:
        return None

    last_hist_year = int(train_df['year'].max())

    # 预测到2050
    n_forecast = 2050 - 1990 + 1
    forecast = res.get_forecast(steps=n_forecast - len(y_data))
    pred_df = forecast.summary_frame(alpha=0.20)  # 80% CI
    pred_df_95 = forecast.summary_frame(alpha=0.05)  # 95% CI

    # 拼接历史拟合+预测
    fitted = res.fittedvalues
    years_hist = np.arange(1990, last_hist_year + 1)
    years_pred = np.arange(last_hist_year + 1, 2051)

    df80 = pd.DataFrame({
        'Year': list(years_hist) + list(years_pred),
        'Mean': np.concatenate([fitted, pred_df['mean'].values]),
        'Lower': np.concatenate([fitted - 1.28 * np.std(y_data - fitted), pred_df['mean_ci_lower'].values]),
        'Upper': np.concatenate([fitted + 1.28 * np.std(y_data - fitted), pred_df['mean_ci_upper'].values])
    })

    df95 = pd.DataFrame({
        'Year': list(years_hist) + list(years_pred),
        'Mean': np.concatenate([fitted, pred_df_95['mean'].values]),
        'Lower': np.concatenate([fitted - 1.96 * np.std(y_data - fitted), pred_df_95['mean_ci_lower'].values]),
        'Upper': np.concatenate([fitted + 1.96 * np.std(y_data - fitted), pred_df_95['mean_ci_upper'].values])
    })

    return {
        'group': group_name,
        'element': element_name,
        'df_80': df80,
        'df_95': df95,
        'last_hist_year': last_hist_year,
        'model_type': 'ARIMA'
    }


# ==================== 模型 3: Prophet ====================
def train_prophet(df_pair, group_name, element_name):
    """Facebook Prophet"""
    try:
        from prophet import Prophet
    except ImportError:
        return None

    train_df = df_pair[(df_pair['year'] >= 1990) & (df_pair['year'] <= 2014)].copy()

    if len(train_df) < 5:
        return None

    train_df = train_df.sort_values('year').reset_index(drop=True)

    # Prophet 需要 'ds' (日期) 和 'y' (值) 列
    prophet_df = pd.DataFrame({
        'ds': pd.to_datetime(train_df['year'].astype(str) + '-01-01'),
        'y': train_df['trade'].values.astype(float)
    })

    try:
        # 配置:  允许变化点检测
        model = Prophet(
            changepoint_prior_scale=0.05,
            interval_width=0.80,
            yearly_seasonality=False,
            weekly_seasonality=False,
            daily_seasonality=False
        )
        model.fit(prophet_df)
    except Exception as e:
        return None

    last_hist_year = int(train_df['year'].max())

    # 预测到2050
    future = model.make_future_dataframe(periods=2050 - last_hist_year, freq='Y')
    forecast = model.predict(future)

    df80 = pd.DataFrame({
        'Year': forecast['ds'].dt.year.values,
        'Mean': forecast['yhat'].values,
        'Lower': forecast['yhat_lower'].values,
        'Upper': forecast['yhat_upper'].values
    })

    # Prophet 默认80%区间，手动计算95%
    df95 = df80.copy()
    df95['Lower'] = forecast['yhat'].values - 1.96 * (forecast['yhat_upper'].values - forecast['yhat'].values) / 1.28
    df95['Upper'] = forecast['yhat'].values + 1.96 * (forecast['yhat_upper'].values - forecast['yhat'].values) / 1.28

    return {
        'group': group_name,
        'element': element_name,
        'df_80': df80,
        'df_95': df95,
        'last_hist_year': last_hist_year,
        'model_type': 'Prophet'
    }


# ==================== 模型 4: 简单移动平均 + 趋势外推 ====================
def train_simple_trend(df_pair, group_name, element_name):
    """
    简单但稳健的方法：
    1.计算最近5年的平均增长率
    2.线性外推
    3.预测区间基于历史波动
    """
    train_df = df_pair[(df_pair['year'] >= 1990) & (df_pair['year'] <= 2014)].copy()

    if len(train_df) < 5:
        return None

    train_df = train_df.sort_values('year').reset_index(drop=True)
    years = train_df['year'].values
    y_data = train_df['trade'].values.astype(float)

    # 使用最近5年计算趋势
    recent_years = years[-5:]
    recent_values = y_data[-5:]

    # 线性回归
    slope = np.polyfit(recent_years, recent_values, 1)[0]
    last_value = y_data[-1]
    last_year = years[-1]

    # 预测
    all_years = np.arange(1990, 2051)
    predictions = last_value + slope * (all_years - last_year)
    predictions = np.maximum(predictions, 0)  # 避免负值

    # 预测区间：基于历史标准差
    std = np.std(y_data)

    df80 = pd.DataFrame({
        'Year': all_years,
        'Mean': predictions,
        'Lower': np.maximum(predictions - 1.28 * std, 0),
        'Upper': predictions + 1.28 * std
    })

    df95 = pd.DataFrame({
        'Year': all_years,
        'Mean': predictions,
        'Lower': np.maximum(predictions - 1.96 * std, 0),
        'Upper': predictions + 1.96 * std
    })

    return {
        'group': group_name,
        'element': element_name,
        'df_80': df80,
        'df_95': df95,
        'last_hist_year': int(last_year),
        'model_type': 'SimpleTrend'
    }


# ==================== 包装函数（用于并行）====================
def train_single_model_wrapper(df_pair, group_name, element_name, model_name, train_func):
    """
    包装函数，用于并行训练单个商品的单个模型
    """
    try:
        result = train_func(df_pair, group_name, element_name)
        if result:
            return (model_name, result)
        else:
            return (model_name, None)
    except Exception as e:
        print(f"[{model_name}] {group_name} ✗ 失败: {e}")
        return (model_name, None)


# ==================== 并行训练所有模型 ====================
def train_all_models(df_agg, n_jobs=-1):
    """
    对每个出口商品，并行训练4个模型

    参数: 
        n_jobs: 并行任务数，-1表示使用所有CPU核心
    """
    df_export = df_agg[df_agg['Element'] == 'Export Quantity'].copy()
    pairs = df_export[['group']].drop_duplicates()

    n_groups = len(pairs)
    print(f"总共 {n_groups} 个出口商品")
    print(f"将训练 {n_groups * 4} 个模型（每个商品4个模型）\n")

    # 准备所有任务（商品 × 模型）
    tasks = []
    model_configs = [
        ('ETS', train_ets),
        ('ARIMA', train_arima),
        ('Prophet', train_prophet),
        ('SimpleTrend', train_simple_trend)
    ]

    for _, row in pairs.iterrows():
        group_name = row['group']
        element_name = 'Export Quantity'
        df_pair = df_export[df_export['group'] == group_name].copy()

        for model_name, train_func in model_configs:
            tasks.append((df_pair, group_name, element_name, model_name, train_func))

    print(f"开始并行训练 (n_jobs={n_jobs})...\n")

    # 并行执行所有任务
    results = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(train_single_model_wrapper)(df, g, e, m, f)
        for df, g, e, m, f in tasks
    )

    # 整理结果
    all_results = {
        'ETS': [],
        'ARIMA': [],
        'Prophet': [],
        'SimpleTrend': []
    }

    for model_name, result in results:
        if result is not None:
            all_results[model_name].append(result)

    # 打印统计
    print("\n" + "=" * 60)
    print("训练完成统计:")
    print("=" * 60)
    for model_name in ['ETS', 'ARIMA', 'Prophet', 'SimpleTrend']:
        success_count = len(all_results[model_name])
        print(f"{model_name:12s}: {success_count}/{n_groups} 成功")
    print("=" * 60 + "\n")

    return all_results


# ==================== 画图函数 ====================
def plot_all_exports(df_hist, trained_models, save_path, model_name):
    """画图（复用你的函数）"""
    element_name = 'Export Quantity'
    export_models = [m for m in trained_models if m['element'] == element_name]
    export_models = sorted(export_models, key=lambda x: x['group'])

    n_groups = len(export_models)
    n_cols = 6
    n_rows = int(np.ceil(n_groups / n_cols))

    print(f"[{model_name}] 画图:  {n_groups} 个商品, {n_rows} 行 × {n_cols} 列")

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

        hist_data = df_hist[
            (df_hist['group'] == group_name) &
            (df_hist['Element'] == element_name) &
            (df_hist['year'] <= 2014)
            ].copy()

        if hist_data.empty and df80.empty:
            ax.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(group_name, fontsize=9, fontweight='bold')
            continue

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

        ax.set_title(group_name, fontsize=9, fontweight='bold', pad=6)
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
                Line2D([0], [0], color='darkblue', linewidth=2.5, label=f'{model_name} Prediction'),
                Patch(facecolor='#0b3d91', alpha=0.18, label='80% Prediction Interval'),
                Patch(facecolor='#082b6a', alpha=0.12, label='95% Prediction Interval'),
            ]

            ax.legend(handles=legend_elements, loc='lower right', fontsize=11,
                      frameon=True, framealpha=0.95, ncol=2, bbox_to_anchor=(1.6, 0.02))
            legend_placed = True
        else:
            ax.axis('off')

    plt.savefig(save_path, dpi=150, bbox_inches='tight', pad_inches=0.1)
    print(f"  ✓ 保存:  {save_path}\n")
    plt.close(fig)


# ==================== 主程序 ====================
def main():
    DATA_PATH = "../2_processed_data/trade_model_data_all.csv"
    OUTPUT_DIR = "./model_comparison"
    N_JOBS = 50  # -1 表示使用所有CPU核心，也可以设置具体数字如 4, 8

    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    # 加载数据
    print("加载数据...")
    df = pd.read_csv(DATA_PATH)
    df = df[df["Report ISO"].str.upper() == "AUS"].copy()
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df["trade"] = pd.to_numeric(df["trade"], errors="coerce")
    df = df.dropna(subset=["year", "trade"])

    df_agg = df.groupby(["group", "Element", "year"], as_index=False)["trade"].sum()
    print(f"数据:  {len(df_agg)} 行\n")

    # 并行训练所有模型
    all_results = train_all_models(df_agg, n_jobs=N_JOBS)

    # 保存训练结果
    model_file = Path(OUTPUT_DIR) / "all_trained_models.pkl"
    with open(model_file, 'wb') as f:
        pickle.dump(all_results, f)
    print(f"✓ 模型已保存到:  {model_file}\n")

    # 为每个模型生成图
    print("开始画图...\n")
    for model_name, results in all_results.items():
        if results:
            save_path = Path(OUTPUT_DIR) / f"export_predictions_{model_name}.png"
            plot_all_exports(df_agg, results, save_path, model_name)

    print("=" * 60)
    print("完成！请查看 ./model_comparison/ 文件夹")
    print("=" * 60)
    print("包含4张图：")
    print("  1.export_predictions_ETS.png")
    print("  2.export_predictions_ARIMA.png")
    print("  3.export_predictions_Prophet.png")
    print("  4.export_predictions_SimpleTrend.png")
    print("=" * 60)


if __name__ == "__main__":
    main()