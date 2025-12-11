#!/usr/bin/env python3
"""
Predict Australia's trade (all commodities, all countries) using posterior sampling.

Input: Future data CSV (2010-2050) with columns:
  - group, Report ISO, Partner ISO, country, Element, year
  - gdp_pc, distance, Population. WB, Urban. population. pct. WB

Output: Predictions with uncertainty scenarios (pessimistic/median/optimistic)

Method: Posterior sampling (draws N samples from trained posterior,
        predicts with each, then computes quantiles)

Usage:
    python predict_australia_trade_posterior_sampling.py future_data_2010_2050.csv
"""
import sys
import os
import pickle
import numpy as np
import pandas as pd
import arviz as az
from tqdm import tqdm
import matplotlib.pyplot as plt

# ======================== CONFIG ========================
FUTURE_DATA_PATH = sys.argv[1] if len(sys.argv) > 1 else "../2_processed_data/future_data_2010_2050.csv"
TRACE_PATH = "../2_processed_data/trade_model_trace.nc"
SCALER_PATH = "../2_processed_data/trade_scaler.pkl"
OUT_CSV = "../2_processed_data/australia_trade_predictions_2010_2050.csv"

N_POSTERIOR_SAMPLES = 500  # 推荐 200-1000，根据计算能力调整


def main():
    print("=" * 70)
    print(" Australia Trade Prediction - Posterior Sampling Method")
    print("=" * 70)

    # ======================== 1.  LOAD TRAINED MODEL ========================
    print("\n[Step 1/6] Loading trained model and scaler...")

    if not os.path.exists(TRACE_PATH):
        print(f"ERROR: Trace file not found: {TRACE_PATH}")
        print("Please run the training script first!")
        sys.exit(1)

    if not os.path.exists(SCALER_PATH):
        print(f"ERROR: Scaler file not found: {SCALER_PATH}")
        sys.exit(1)

    # Load scaler
    with open(SCALER_PATH, "rb") as f:
        scaler_data = pickle.load(f)
    scaler = scaler_data['scaler']
    preds_raw = scaler_data['preds_raw']
    preds_scaled = scaler_data['preds_scaled']
    train_years = scaler_data['train_years']

    print(f"  Scaler trained on: {train_years[0]}-{train_years[1]}")
    print(f"  Scaler means: {scaler.mean_}")
    print(f"  Scaler stds: {scaler.scale_}")

    # Load trace
    idata = az.from_netcdf(TRACE_PATH)
    groups = list(idata.posterior.coords['commodity'].values)
    gmap = {g: i for i, g in enumerate(groups)}

    n_chains = idata.posterior.dims['chain']
    n_draws = idata.posterior.dims['draw']
    n_total_samples = n_chains * n_draws

    print(f"  Model trained on {len(groups)} commodity groups")
    print(f"  Posterior samples: {n_chains} chains × {n_draws} draws = {n_total_samples} total")
    print(f"  Commodity groups: {groups}")

    # ======================== 2. LOAD FUTURE DATA ========================
    print(f"\n[Step 2/6] Loading future data from {FUTURE_DATA_PATH}...")

    if not os.path.exists(FUTURE_DATA_PATH):
        print(f"ERROR: Future data file not found: {FUTURE_DATA_PATH}")
        sys.exit(1)

    df = pd.read_csv(FUTURE_DATA_PATH, low_memory=False)
    print(f"  Loaded {len(df)} rows")
    print(f"  Columns: {df.columns.tolist()}")
    print(f"  Year range: {df['year'].min()}-{df['year'].max()}")
    print(f"  Unique groups: {df['group'].nunique()}")
    print(f"  Unique countries: {df['Partner ISO'].nunique() if 'Partner ISO' in df.columns else 'N/A'}")

    # Rename columns if needed
    if 'Report ISO' in df.columns:
        df = df.rename(columns={'Report ISO': 'Report_ISO'})
    if 'Partner ISO' in df.columns:
        df = df.rename(columns={'Partner ISO': 'Partner_ISO'})

    # ======================== 3. PREPROCESS FUTURE DATA ========================
    print("\n[Step 3/6] Preprocessing future data...")

    # Check required columns
    required_cols = ['group', 'Element', 'year', 'gdp_pc', 'Population.WB',
                     'Urban.population.pct. WB', 'distance']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"ERROR: Missing required columns: {missing_cols}")
        sys.exit(1)

    # Construct GDP PPP
    df['gdp_ppp'] = df['gdp_pc'] * df['Population.WB']

    # Log transforms
    df['gdp_ppp_lt'] = np.log1p(df['gdp_ppp'].astype(float))
    df['pop_lt'] = np.log1p(df['Population.WB'].astype(float))
    df['dist_lt'] = np.log1p(df['distance'].astype(float))

    # Check for NaN in raw predictors
    nan_counts = df[preds_raw].isna().sum()
    if nan_counts.sum() > 0:
        print(f"  WARNING: Found NaN values in predictors:")
        print(nan_counts[nan_counts > 0])
        print(f"  Dropping {df[preds_raw].isna().any(axis=1).sum()} rows with NaN...")
        df = df.dropna(subset=preds_raw)

    # Apply scaler (using training scaler - critical!)
    X_raw = df[preds_raw].values
    X_z = scaler.transform(X_raw)

    df['gdp_ppp_lt_z'] = X_z[:, 0]
    df['pop_lt_z'] = X_z[:, 1]
    df['urban_z'] = X_z[:, 2]
    df['dist_lt_z'] = X_z[:, 3]
    df['gdp_z_sq'] = df['gdp_ppp_lt_z'] ** 2

    # Export/import indicator
    df['is_export'] = (df['Element'] == "Export Quantity").astype(int)

    # Filter to groups present in training model
    df_before = len(df)
    df = df[df['group'].isin(groups)].copy()
    df_after = len(df)
    if df_after < df_before:
        print(f"  Filtered out {df_before - df_after} rows with unknown commodity groups")

    if len(df) == 0:
        print("ERROR: No matching commodity groups between training and future data!")
        print(f"  Training groups: {groups}")
        print(f"  Future groups: {sorted(pd.read_csv(FUTURE_DATA_PATH)['group'].unique())}")
        sys.exit(1)

    # Map group to index
    df['gidx'] = df['group'].map(gmap).astype(int)

    print(f"  Preprocessed {len(df)} rows for prediction")
    print(f"  Predictors (standardized): {preds_scaled}")

    # Prepare arrays for prediction
    X_pred = np.column_stack([df[p].values for p in preds_scaled])
    gidx = df['gidx'].values
    is_export = df['is_export'].values

    # ======================== 4. SAMPLE FROM POSTERIOR ========================
    print(f"\n[Step 4/6] Drawing {N_POSTERIOR_SAMPLES} samples from posterior distribution...")

    post = idata.posterior

    # Stack chain and draw dimensions into single sample dimension
    alpha_all = post["alpha"].stack(sample=("chain", "draw")).values.T  # (n_total, n_groups)
    beta_all = post["beta"].stack(sample=("chain", "draw")).values.T  # (n_total, n_preds)
    beta_export_all = post["beta_export"].stack(sample=("chain", "draw")).values  # (n_total,)

    print(f"  Posterior shapes: alpha {alpha_all.shape}, beta {beta_all.shape}, beta_export {beta_export_all.shape}")

    # Randomly select N samples
    rng = np.random.default_rng(42)
    n_samples = min(N_POSTERIOR_SAMPLES, n_total_samples)
    selected_idx = rng.choice(n_total_samples, size=n_samples, replace=False)

    alpha_samples = alpha_all[selected_idx, :]
    beta_samples = beta_all[selected_idx, :]
    beta_export_samples = beta_export_all[selected_idx]

    print(f"  Selected {n_samples} posterior samples (from {n_total_samples} available)")

    # ======================== 5. PREDICT WITH EACH SAMPLE ========================
    print(f"\n[Step 5/6] Computing predictions for each posterior sample...")
    print(f"  This will generate {n_samples} × {len(df)} = {n_samples * len(df):,} predictions")

    all_predictions = np.zeros((n_samples, len(df)))

    for i in tqdm(range(n_samples), desc="  Sampling progress"):
        # Linear predictor with i-th parameter set
        mu = (alpha_samples[i, gidx] +
              X_pred @ beta_samples[i, :] +
              beta_export_samples[i] * is_export)

        # Transform back to original scale (log(trade) -> trade)
        trade = np.expm1(mu).clip(min=0.0)

        all_predictions[i, :] = trade

    print(f"  Completed {n_samples} predictions")

    # ======================== 6. AGGREGATE & COMPUTE SCENARIOS ========================
    print("\n[Step 6/6] Aggregating predictions and computing uncertainty scenarios...")

    # Create expanded dataframe with all samples
    df_list = []
    for i in range(n_samples):
        tmp = df[['group', 'Partner_ISO', 'country', 'Element', 'year']].copy()
        tmp['sample_id'] = i
        tmp['trade_sample'] = all_predictions[i, :]
        df_list.append(tmp)

    df_expanded = pd.concat(df_list, ignore_index=True)
    print(f"  Created expanded dataset: {len(df_expanded):,} rows ({n_samples} samples × {len(df)} rows)")

    # Aggregate by (group, Element, year) for each sample
    print("  Aggregating across countries for each sample...")
    agg_samples = (df_expanded
                   .groupby(['group', 'Element', 'year', 'sample_id'], as_index=False)['trade_sample']
                   .sum())

    # Pivot to get (group, Element, year) × samples matrix
    print("  Computing quantiles from prediction distribution...")
    pivot = (agg_samples
             .pivot_table(index=['group', 'Element', 'year'],
                          columns='sample_id',
                          values='trade_sample')
             .values)

    keys = agg_samples.groupby(['group', 'Element', 'year'], as_index=False).first()[['group', 'Element', 'year']]

    print(f"  Pivot shape: {pivot.shape} (combinations × samples)")

    # Compute uncertainty scenarios from prediction distribution
    scenarios = {
        'pessimistic_p05': np.percentile(pivot, 5, axis=1),
        'lower_p25': np.percentile(pivot, 25, axis=1),
        'median_p50': np.percentile(pivot, 50, axis=1),
        'upper_p75': np.percentile(pivot, 75, axis=1),
        'optimistic_p95': np.percentile(pivot, 95, axis=1),
        'mean': pivot.mean(axis=1),
        'std': pivot.std(axis=1)
    }

    # Create output dataframe
    results = []
    for scenario_name, values in scenarios.items():
        tmp = keys.copy()
        tmp['scenario'] = scenario_name
        tmp['trade_volume'] = values
        results.append(tmp)

    final = pd.concat(results, ignore_index=True)
    final = final.sort_values(['scenario', 'group', 'Element', 'year'])

    # ======================== SAVE RESULTS ========================
    print("\n" + "=" * 70)
    print(" SAVING RESULTS")
    print("=" * 70)

    final.to_csv(OUT_CSV, index=False)
    print(f"\nSaved predictions to: {OUT_CSV}")
    print(f"  Total rows: {len(final):,}")
    print(f"  Scenarios: {final['scenario'].unique().tolist()}")
    print(f"  Commodity groups: {final['group'].nunique()}")
    print(f"  Years: {final['year'].min()}-{final['year'].max()}")

    # ======================== SUMMARY STATISTICS ========================
    print("\n" + "=" * 70)
    print(" PREDICTION SUMMARY")
    print("=" * 70)

    # Sample: Show one commodity's predictions
    sample_group = final['group'].iloc[0]
    sample_year = 2030
    sample_element = 'Export Quantity'

    print(f"\nExample: {sample_group} {sample_element} in {sample_year}")
    sample_data = final[(final['group'] == sample_group) &
                        (final['year'] == sample_year) &
                        (final['Element'] == sample_element)]

    if not sample_data.empty:
        print("\nUncertainty scenarios:")
        for _, row in sample_data.iterrows():
            print(f"  {row['scenario']:20s}: {row['trade_volume']:>15,. 0f}")

    # Overall statistics
    print("\n" + "-" * 70)
    print("Overall prediction statistics (median scenario, year 2030):")
    stats_2030 = final[(final['scenario'] == 'median_p50') & (final['year'] == 2030)]
    if not stats_2030.empty:
        export_2030 = stats_2030[stats_2030['Element'] == 'Export Quantity']['trade_volume'].sum()
        import_2030 = stats_2030[stats_2030['Element'] == 'Import Quantity']['trade_volume'].sum()
        print(f"  Total Export (all commodities): {export_2030:,. 0f}")
        print(f"  Total Import (all commodities): {import_2030:,. 0f}")

    # Top commodities
    print("\n" + "-" * 70)
    print("Top 10 export commodities (median scenario, year 2030):")
    top_exports = (stats_2030[stats_2030['Element'] == 'Export Quantity']
                   .sort_values('trade_volume', ascending=False)
                   .head(10))
    print(top_exports[['group', 'trade_volume']].to_string(index=False))

    print("\n" + "=" * 70)
    print(" COMPLETE!")
    print("=" * 70)
    print(f"\nMethod: Posterior sampling with {n_samples} samples")
    print(f"Uncertainty properly propagated through nonlinear transformations")
    print(f"\nNext steps:")
    print(f"  1. Check {OUT_CSV} for results")
    print(f"  2.  Visualize predictions by commodity/scenario")
    print(f"  3. Compare scenarios for risk assessment")

    # 读取预测结果
    df = pd.read_csv("../2_processed_data/australia_trade_predictions_2010_2050.csv")

    # 选择要画的商品
    commodities = ['Wheat', 'Beef', 'Wine']  # 改成你实际的商品名

    fig, axes = plt.subplots(len(commodities), 1, figsize=(12, 4 * len(commodities)))

    for i, commodity in enumerate(commodities):
        ax = axes[i] if len(commodities) > 1 else axes

        # 筛选该商品的出口数据
        data = df[(df['group'] == commodity) & (df['Element'] == 'Export Quantity')]

        # 提取不同情景
        years = sorted(data['year'].unique())
        median = data[data['scenario'] == 'median_p50'].set_index('year')['trade_volume']
        p05 = data[data['scenario'] == 'pessimistic_p05'].set_index('year')['trade_volume']
        p95 = data[data['scenario'] == 'optimistic_p95'].set_index('year')['trade_volume']

        # 画图
        ax.plot(years, median, linewidth=2, label='Median', color='blue')
        ax.fill_between(years, p05, p95, alpha=0.3, label='5th-95th percentile', color='blue')
        ax.axvline(2014, color='red', linestyle='--', alpha=0.5, label='Training data ends')

        ax.set_title(f'{commodity} Export Projections', fontsize=14, fontweight='bold')
        ax.set_xlabel('Year')
        ax.set_ylabel('Trade Volume')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('../2_processed_data/trade_predictions_plot.png', dpi=300, bbox_inches='tight')
    print("Saved plot to trade_predictions_plot.png")
    plt.show()


if __name__ == '__main__':
    main()