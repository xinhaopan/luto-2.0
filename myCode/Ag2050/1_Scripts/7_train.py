#!/usr/bin/env python3
"""
Fixed for Windows multiprocessing
"""
import sys
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import pymc as pm
import arviz as az

# ======================== CONFIG ========================
INPATH = "../2_processed_data/trade_model_data_all.csv"
TRACE_PATH = "../2_processed_data/trade_model_trace.nc"
SCALER_PATH = "../2_processed_data/trade_scaler.pkl"
DIAGNOSTICS_PATH = "../2_processed_data/model_diagnostics.txt"

TRAIN_Y0, TRAIN_Y1 = 1990, 2014
N_DRAWS = 1000
N_TUNE = 1000
N_CHAINS = 4


# ======================== MAIN FUNCTION ========================
def main():
    # ======================== 1.LOAD & FILTER DATA ========================
    print("=" * 60)
    print("STEP 1: Loading data (1990-2014 only)")
    print("=" * 60)
    df = pd.read_csv(INPATH, low_memory=False)
    print(f"Loaded {len(df)} rows from {INPATH}")

    # Filter to 1990-2014
    train = df[(df['year'] >= TRAIN_Y0) & (df['year'] <= TRAIN_Y1)].copy()
    print(f"Filtered to {TRAIN_Y0}-{TRAIN_Y1}: {len(train)} rows")

    if len(train) == 0:
        print("ERROR: No data in 1990-2014 range!  Check your 'year' column.")
        sys.exit(1)

    # ======================== 2.CONSTRUCT & TRANSFORM PREDICTORS ========================
    print("\n" + "=" * 60)
    print("STEP 2: Constructing and transforming predictors")
    print("=" * 60)

    train['gdp_ppp'] = train['gdp_pc'] * train['Population.WB']
    train['gdp_ppp_lt'] = np.log1p(train['gdp_ppp'].astype(float))
    train['pop_lt'] = np.log1p(train['Population.WB'].astype(float))
    train['dist_lt'] = np.log1p(train['distance'].astype(float))

    preds_raw = ['gdp_ppp_lt', 'pop_lt', 'Urban.population.pct.WB', 'dist_lt']

    X_train_raw = train[preds_raw].values
    scaler = StandardScaler().fit(X_train_raw)
    print("\nScaler fitted on 1990-2014 data:")
    print(f"  Means: {scaler.mean_}")
    print(f"  Stds: {scaler.scale_}")

    X_train_z = scaler.transform(X_train_raw)
    train['gdp_ppp_lt_z'] = X_train_z[:, 0]
    train['pop_lt_z'] = X_train_z[:, 1]
    train['urban_z'] = X_train_z[:, 2]
    train['dist_lt_z'] = X_train_z[:, 3]
    train['gdp_z_sq'] = train['gdp_ppp_lt_z'] ** 2
    train['y'] = np.log1p(train['trade'].astype(float))
    train['is_export'] = (train['Element'] == "Export Quantity").astype(int)

    preds = ['gdp_ppp_lt_z', 'pop_lt_z', 'urban_z', 'dist_lt_z', 'gdp_z_sq']

    with open(SCALER_PATH, "wb") as f:
        pickle.dump({
            'scaler': scaler,
            'preds_raw': preds_raw,
            'preds_scaled': preds,
            'train_years': [TRAIN_Y0, TRAIN_Y1]
        }, f)
    print(f"\nSaved scaler to {SCALER_PATH}")

    # ======================== 3.PREPARE TRAINING DATA ========================
    print("\n" + "=" * 60)
    print("STEP 3: Preparing training dataset")
    print("=" * 60)

    groups = sorted(train['group'].astype(str).unique())
    gmap = {g: i for i, g in enumerate(groups)}
    train['gidx'] = train['group'].astype(str).map(gmap).astype(int)

    print(f"Number of commodity groups: {len(groups)}")
    print(f"Groups: {groups}")

    y_train = train['y'].values
    is_export_train = train['is_export'].values
    X_train_z = np.column_stack([train[p].values for p in preds])
    gidx_train = train['gidx'].values

    print(f"\nTraining data shape:")
    print(f"  y: {y_train.shape}")
    print(f"  X: {X_train_z.shape}")

    # ======================== 4. BUILD & SAMPLE BAYESIAN MODEL ========================
    print("\n" + "=" * 60)
    print("STEP 4: Building and sampling Bayesian hierarchical model")
    print("=" * 60)

    coords = {"commodity": groups, "predictor": preds}

    with pm.Model(coords=coords) as model:
        mu_alpha = pm.Normal("mu_alpha", mu=0.0, sigma = 1.0)
        sigma_alpha = pm.HalfNormal("sigma_alpha", sigma=1.0)
        alpha_offset = pm.Normal("alpha_offset", mu=0.0, sigma=1.0, dims="commodity")
        alpha = pm.Deterministic("alpha", mu_alpha + alpha_offset * sigma_alpha, dims="commodity")

        beta = pm.Normal("beta", mu=0.0, sigma=1.0, dims="predictor")
        beta_export = pm.Normal("beta_export", mu=0.0, sigma=1.0)

        sigma = pm.HalfNormal("sigma", sigma=1.0)
        nu = pm.Exponential("nu", lam=1 / 10)

        mu_train = alpha[gidx_train] + pm.math.dot(X_train_z, beta) + beta_export * is_export_train
        y_obs = pm.StudentT("y_obs", nu=nu, mu=mu_train, sigma=sigma, observed=y_train)

        print(f"\nSampling with {N_CHAINS} chains, {N_DRAWS} draws, {N_TUNE} tuning steps...")
        print("This may take 10-30 minutes...\n")

        idata = pm.sample(
            draws=N_DRAWS,
            tune=N_TUNE,
            chains=N_CHAINS,
            target_accept=0.9,
            cores=N_CHAINS,
            return_inferencedata=True,
            random_seed=42
        )

    # ======================== 5. SAVE & DIAGNOSE ========================
    print("\n" + "=" * 60)
    print("STEP 5: Saving trace and diagnostics")
    print("=" * 60)

    az.to_netcdf(idata, TRACE_PATH)
    print(f"Saved trace to {TRACE_PATH}")

    summary = az.summary(idata, var_names=["mu_alpha", "sigma_alpha", "beta", "beta_export", "sigma", "nu"])
    print("\nPosterior summary:")
    print(summary)

    max_rhat = summary['r_hat'].max()
    min_ess_bulk = summary['ess_bulk'].min()
    min_ess_tail = summary['ess_tail'].min()
    divergences = idata.sample_stats.diverging.sum().values

    print("\nConvergence diagnostics:")
    print(f"  Max Rhat: {max_rhat:.4f} (should be < 1.01)")
    print(f"  Min ESS bulk: {min_ess_bulk:.0f} (should be > 400)")
    print(f"  Min ESS tail: {min_ess_tail:.0f} (should be > 400)")
    print(f"  Divergences: {divergences} (should be 0)")

    with open(DIAGNOSTICS_PATH, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("MODEL DIAGNOSTICS\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Training period: {TRAIN_Y0}-{TRAIN_Y1}\n")
        f.write(f"Training rows: {len(train)}\n")
        f.write(f"Commodities: {len(groups)}\n\n")
        f.write("Posterior summary:\n")
        f.write(summary.to_string())
        f.write(f"\n\nMax Rhat: {max_rhat:.4f}\n")
        f.write(f"Min ESS bulk: {min_ess_bulk:.0f}\n")
        f.write(f"Min ESS tail: {min_ess_tail:.0f}\n")
        f.write(f"Divergences: {divergences}\n")

    print(f"\nSaved diagnostics to {DIAGNOSTICS_PATH}")

    # ======================== 6.POSTERIOR ANALYSIS ========================
    print("\n" + "=" * 60)
    print("STEP 6: Posterior coefficient interpretation")
    print("=" * 60)

    post = idata.posterior
    alpha_mean = post["alpha"].mean(dim=("chain", "draw")).values
    beta_mean = post["beta"].mean(dim=("chain", "draw")).values
    beta_export_mean = float(post["beta_export"].mean(dim=("chain", "draw")).values)

    print("\nPosterior means:")
    print(f"  Global intercept: {float(post['mu_alpha'].mean()):.3f}")
    print(f"  Export coefficient: {beta_export_mean:.3f}")
    print("\nPredictor coefficients:")
    for i, pred in enumerate(preds):
        print(f"  {pred:20s}: {beta_mean[i]:7.3f}")

    print("\nCommodity intercepts (top 10):")
    alpha_df = pd.DataFrame({'group': groups, 'alpha': alpha_mean}).sort_values('alpha', ascending=False)
    print(alpha_df.head(10).to_string(index=False))

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)
    print(f"\nSaved files:")
    print(f"  1.{TRACE_PATH}")
    print(f"  2.{SCALER_PATH}")
    print(f"  3.{DIAGNOSTICS_PATH}")


# ======================== WINDOWS MULTIPROCESSING FIX ========================
if __name__ == '__main__':
    main()