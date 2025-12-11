#!/usr/bin/env python3
"""
FULL RUN VERSION with scenarios (low / med / high).

改进：
  - 保存每个 pair 的 StandardScaler（用于 year 标准化）
  - 保存后验 trace（idata）
  - 保存一个索引 JSON，记录每个 pair 对应的 scaler/trace 文件
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import pymc as pm
import arviz as az
import warnings
import re
import pickle
import json
from typing import Optional, Tuple, List

warnings.filterwarnings("ignore")

# ========== CONFIG ==========
INPATH = "../2_processed_data/trade_model_data_AUS_all.csv"
OUT_PRED_CSV = "../2_processed_data/predictions_AUS_scenarios_full.csv"
TRACE_DIR = "../2_processed_data/group_element_traces_AUS_scenarios_full"
SCALER_DIR = "../2_processed_data/group_element_scalers_AUS"
PARAMS_INDEX = "../2_processed_data/model_params_index.json"

TRAIN_Y0, TRAIN_Y1 = 1990, 2014
FUTURE_Y0, FUTURE_Y1 = 2015, 2050

MIN_OBS_PER_PAIR = 6
N_DRAWS = 500
N_TUNE = 500
N_CHAINS = 4

USE_JOBLIB = True
N_JOBS = 45

Path(TRACE_DIR).mkdir(parents=True, exist_ok=True)
Path(SCALER_DIR).mkdir(parents=True, exist_ok=True)


def _safe_fname(s: str) -> str:
    return re.sub(r"[^0-9A-Za-z._-]+", "_", str(s))[:200]


def fit_and_predict_pair(df_pair: pd.DataFrame,
                         group_name: str,
                         element_name: str) -> Optional[pd.DataFrame]:
    """
    对一个 (group, Element) pair 拟合 + 预测（low/med/high 三情景）。
    同时保存 scaler 和 trace。
    """
    # 1) 训练数据：1990–2014
    train = df_pair[(df_pair["year"] >= TRAIN_Y0) & (df_pair["year"] <= TRAIN_Y1)].copy()
    n_train = train["year"].nunique()
    if n_train < MIN_OBS_PER_PAIR:
        print(f"[INFO] Skip ({group_name}, {element_name}): only {n_train} training years")
        return None

    train = train.dropna(subset=["trade", "year"]).sort_values("year")
    y_train = np.log1p(train["trade"]. astype(float). values)
    years_train = train["year"].astype(int).values. reshape(-1, 1)

    # 2) 标准化 year
    scaler = StandardScaler(). fit(years_train)
    year_z_train = scaler.transform(years_train). ravel()

    future_years = np.arange(FUTURE_Y0, FUTURE_Y1 + 1)
    future_years_z = scaler.transform(future_years. reshape(-1, 1)).ravel()

    try:
        with pm.Model() as model:
            alpha = pm.Normal("alpha", mu=0.0, sigma=5.0)
            beta = pm.Normal("beta", mu=0.0, sigma=2.0)
            sigma = pm.HalfNormal("sigma", sigma=2.0)
            nu = pm.Exponential("nu", lam=1 / 10)

            mu_train = alpha + beta * year_z_train
            like = pm.logp(pm.StudentT. dist(nu=nu, mu=mu_train, sigma=sigma), y_train)
            pm. Potential("likelihood", like. sum())

            print(f"[RUN] Sampling ({group_name}, {element_name}), n_train_years={n_train} ...")
            idata = pm.sample(
                draws=N_DRAWS,
                tune=N_TUNE,
                chains=N_CHAINS,
                cores=1,
                target_accept=0.9,
                return_inferencedata=True,
                progressbar=False,
                random_seed=42,
            )

            # 保存 trace
            fname = f"trace_{_safe_fname(group_name)}__{_safe_fname(element_name)}.nc"
            trace_path = Path(TRACE_DIR) / fname
            az.to_netcdf(idata, trace_path)

            # 保存 scaler
            scaler_fname = f"scaler_{_safe_fname(group_name)}__{_safe_fname(element_name)}.pkl"
            scaler_path = Path(SCALER_DIR) / scaler_fname
            with open(scaler_path, "wb") as f:
                pickle.dump(scaler, f)

    except Exception as e:
        print(f"[WARN] Error fitting ({group_name}, {element_name}): {e}")
        return None

    # 4) 从后验中取 alpha, beta 样本，构造低/中/高三种情景
    post = idata. posterior
    alpha_samples = post["alpha"]. values. reshape(-1)
    beta_samples = post["beta"].values.reshape(-1)

    alpha_mean = float(alpha_samples.mean())
    beta_mean = float(beta_samples.mean())
    beta_low = float(np.quantile(beta_samples, 0.10))
    beta_high = float(np.quantile(beta_samples, 0.90))

    scenarios = {
        "low": beta_low,
        "med": beta_mean,
        "high": beta_high,
    }

    rows = []
    for scen_name, beta_scen in scenarios.items():
        mu_future = alpha_mean + beta_scen * future_years_z

        trade_pred = np.expm1(mu_future)
        trade_pred = np.maximum(trade_pred, 0.0)

        for year, pred in zip(future_years, trade_pred):
            rows.append({
                "group": group_name,
                "element": element_name,
                "year": int(year),
                "scenario": scen_name,
                "pred_mean": float(pred),
                "n_train_years": int(n_train),
            })

    df_out = pd.DataFrame(rows)
    return df_out


def run_one_pair(args: Tuple[str, str, pd.DataFrame]) -> Optional[pd. DataFrame]:
    group_name, element_name, df_pair = args
    try:
        return fit_and_predict_pair(df_pair, group_name, element_name)
    except Exception as e:
        print(f"[WARN] Exception in worker for ({group_name}, {element_name}): {e}")
        return None


def main():
    print("[RUN] Loading data:", INPATH)
    df = pd.read_csv(INPATH, low_memory=False)
    print("[RUN] Total rows loaded:", len(df))

    df = df[df["Report ISO"].astype(str).str.upper() == "AUS"].copy()
    print("[RUN] Rows after Report ISO == 'AUS':", len(df))
    if df.empty:
        print("[RUN] No rows for AUS, exit.")
        sys. exit(1)

    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df["trade"] = pd.to_numeric(df["trade"], errors="coerce")
    df = df.dropna(subset=["year", "trade"])
    df["year"] = df["year"].astype(int)

    agg_cols = ["group", "Element", "year"]
    df_agg = df.groupby(agg_cols, as_index=False)["trade"].sum()
    print("[RUN] After aggregation:", len(df_agg), "rows")

    pairs = df_agg[["group", "Element"]].drop_duplicates().values.tolist()
    print(f"[RUN] Total (group, element) pairs: {len(pairs)}")
    print(f"[RUN] USE_JOBLIB = {USE_JOBLIB}, N_JOBS = {N_JOBS if USE_JOBLIB else 1}")

    tasks: List[Tuple[str, str, pd.DataFrame]] = []
    for group_name, element_name in pairs:
        df_pair = df_agg[
            (df_agg["group"].astype(str) == str(group_name))
            & (df_agg["Element"] == element_name)
        ]
        if df_pair.empty:
            continue
        tasks.append((str(group_name), str(element_name), df_pair))

    results: List[pd.DataFrame] = []

    if USE_JOBLIB:
        from joblib import Parallel, delayed

        print(f"[RUN] Submitting {len(tasks)} tasks to joblib.Parallel ...")
        outs = Parallel(n_jobs=N_JOBS, verbose=10)(
            delayed(run_one_pair)(task) for task in tasks
        )
        for res in outs:
            if res is not None:
                results.append(res)
    else:
        for idx, (g, e, df_pair) in enumerate(tasks, start=1):
            print(f"\n[RUN] ===== Pair {idx}/{len(tasks)}: ({g}, {e}) =====")
            res = run_one_pair((g, e, df_pair))
            if res is not None:
                results.append(res)

    if not results:
        print("[RUN] No predictions produced in full run.")
        return

    df_preds = pd. concat(results, ignore_index=True)
    df_preds.to_csv(OUT_PRED_CSV, index=False)
    print("\n[RUN] Saved FULL predictions with scenarios to:", OUT_PRED_CSV)
    print("[RUN] Columns:", df_preds. columns.tolist())
    print("[RUN] Sample:")
    print(df_preds.head(30).to_string(index=False))

    # 生成索引 JSON（方便查找每个 pair 对应的文件）
    index_dict = {
        "train_y0": TRAIN_Y0,
        "train_y1": TRAIN_Y1,
        "future_y0": FUTURE_Y0,
        "future_y1": FUTURE_Y1,
        "trace_dir": str(TRACE_DIR),
        "scaler_dir": str(SCALER_DIR),
        "pairs": [
            {
                "group": g,
                "element": e,
                "trace_file": f"trace_{_safe_fname(g)}__{_safe_fname(e)}.nc",
                "scaler_file": f"scaler_{_safe_fname(g)}__{_safe_fname(e)}.pkl",
            }
            for g, e, _ in tasks
        ]
    }

    with open(PARAMS_INDEX, "w") as f:
        json.dump(index_dict, f, indent=2)
    print(f"\n[RUN] Saved model index to: {PARAMS_INDEX}")


if __name__ == "__main__":
    main()