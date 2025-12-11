#!/usr/bin/env python3
"""
重新生成预测 + 画图：读取已保存的 trace + scaler，用自定义分位数生成新预测，然后直接画图。

功能：
  1. 从保存的 model_params_index.json 读取 trace/scaler 信息；
  2. 用自定义的 LOW_QUANTILE/HIGH_QUANTILE 重新生成预测；
  3.  同时读取历史数据，画出历史 + 三情景预测图。

两个模式：
  - USE_SINGLE_GROUP_PLOTS = False：大图（多子图，出口和进口各一张）
  - USE_SINGLE_GROUP_PLOTS = True：小图（每个商品单独一张）
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import arviz as az
import pickle
import json
import warnings
import re
import os

warnings.filterwarnings("ignore")

# ========== CONFIG ==========
MODEL_INDEX_JSON = "../2_processed_data/model_params_index.json"
HIST_CSV = "../2_processed_data/trade_model_data_AUS_all.csv"

OUT_PRED_CSV = "../2_processed_data/predictions_AUS_scenarios_custom.csv"
OUT_EXPORT_PNG = "./export_quantity_scenarios_custom.png"
OUT_IMPORT_PNG = "./import_quantity_scenarios_custom.png"
OUT_GROUP_DIR = "./group_plots_custom"

FUTURE_Y0, FUTURE_Y1 = 2015, 2050

# ========== 调整这两个参数来改变 low/high 情景 ==========
LOW_QUANTILE = 0.40  # 改成 0.05, 0.10, 0.20 等
HIGH_QUANTILE = 0.60  # 改成 0.95, 0.90, 0.80 等

# ========== 选择画图模式 ==========
USE_SINGLE_GROUP_PLOTS = False  # True -> 每个商品单独一张图；False -> 大图（多子图）


# ========== 第一部分：重新生成预测 ==========

def regenerate_predictions() -> pd.DataFrame:
    """
    读取保存的 trace 和 scaler，用新的分位数重新生成预测。
    返回预测 DataFrame。
    """
    print(f"[REGEN] Loading model index from: {MODEL_INDEX_JSON}")
    with open(MODEL_INDEX_JSON, "r") as f:
        index_dict = json.load(f)

    trace_dir = Path(index_dict["trace_dir"])
    scaler_dir = Path(index_dict["scaler_dir"])

    print(f"[REGEN] Quantiles: low={LOW_QUANTILE}, med=0.50, high={HIGH_QUANTILE}")
    print(f"[REGEN] Total pairs to process: {len(index_dict['pairs'])}")

    results = []

    for idx, pair_info in enumerate(index_dict["pairs"], start=1):
        group_name = pair_info["group"]
        element_name = pair_info["element"]
        trace_file = pair_info["trace_file"]
        scaler_file = pair_info["scaler_file"]

        trace_path = trace_dir / trace_file
        scaler_path = scaler_dir / scaler_file

        # 检查文件是否存在
        if not trace_path.exists() or not scaler_path.exists():
            print(f"[REGEN] WARNING: Missing files for ({group_name}, {element_name}), skip")
            continue

        try:
            # 加载 trace
            idata = az.from_netcdf(str(trace_path))
            post = idata.posterior

            # 加载 scaler
            with open(scaler_path, "rb") as f:
                scaler = pickle.load(f)

            # 提取后验样本
            alpha_samples = post["alpha"].values.reshape(-1)
            beta_samples = post["beta"].values.reshape(-1)

            alpha_mean = float(alpha_samples.mean())
            beta_mean = float(beta_samples.mean())
            beta_low = float(np.quantile(beta_samples, LOW_QUANTILE))
            beta_high = float(np.quantile(beta_samples, HIGH_QUANTILE))

            # 未来年份
            future_years = np.arange(FUTURE_Y0, FUTURE_Y1 + 1)
            future_years_z = scaler.transform(future_years.reshape(-1, 1)).ravel()

            # 三个情景
            scenarios = {
                "low": beta_low,
                "med": beta_mean,
                "high": beta_high,
            }

            for scen_name, beta_scen in scenarios.items():
                mu_future = alpha_mean + beta_scen * future_years_z
                trade_pred = np.expm1(mu_future)
                trade_pred = np.maximum(trade_pred, 0.0)

                for year, pred in zip(future_years, trade_pred):
                    results.append({
                        "group": group_name,
                        "element": element_name,
                        "year": int(year),
                        "scenario": scen_name,
                        "pred_mean": float(pred),
                    })

            if (idx % 10) == 0:
                print(f"[REGEN] Processed {idx}/{len(index_dict['pairs'])} pairs")

        except Exception as e:
            print(f"[REGEN] ERROR processing ({group_name}, {element_name}): {e}")
            continue

    if not results:
        print("[REGEN] No predictions generated!")
        return pd.DataFrame()

    df_preds = pd.DataFrame(results)
    df_preds.to_csv(OUT_PRED_CSV, index=False)
    print(f"\n[REGEN] Saved {len(df_preds)} prediction rows to: {OUT_PRED_CSV}")
    print(f"[REGEN] Sample:")
    print(df_preds.head(20).to_string(index=False))

    return df_preds


# ========== 第二部分：加载历史数据 ==========

def load_history_data() -> pd.DataFrame:
    """
    加载和聚合历史数据。
    """
    print(f"\n[PLOT] Loading history from: {HIST_CSV}")
    hist_df = pd.read_csv(HIST_CSV)
    print(f"[PLOT] History shape: {hist_df.shape}")

    # 筛选 AUS
    hist_df = hist_df[hist_df["Report ISO"].astype(str).str.upper() == "AUS"].copy()
    hist_df["year"] = pd.to_numeric(hist_df["year"], errors="coerce")
    hist_df["trade"] = pd.to_numeric(hist_df["trade"], errors="coerce")
    hist_df = hist_df.dropna(subset=["year", "trade"])
    hist_df["year"] = hist_df["year"].astype(int)

    # 聚合历史数据
    hist_agg = hist_df.groupby(["group", "Element", "year"], as_index=False)["trade"].sum()
    print(f"[PLOT] History aggregated shape: {hist_agg.shape}")

    return hist_agg


# ========== 第三部分：画图（大图版） ==========

def plot_element_large(element_type: str, pred_df: pd.DataFrame, hist_agg: pd.DataFrame):
    """
    绘制某个 Element（"Export Quantity" 或 "Import Quantity"）的所有商品预测。
    所有商品放在一张大图里，分成多个子图。
    """
    print(f"\n[PLOT] ===== Plotting {element_type} (LARGE FIGURE) =====")

    # 筛选该 Element 的预测
    pred_element = pred_df[pred_df["element"] == element_type].copy()
    print(f"[PLOT] Predictions for '{element_type}': {len(pred_element)} rows")

    if pred_element.empty:
        print(f"[PLOT] No predictions for {element_type}, skip.")
        return None

    # 筛选该 Element 的历史数据
    hist_element = hist_agg[hist_agg["Element"] == element_type].copy()
    print(f"[PLOT] History for '{element_type}': {len(hist_element)} rows")

    # 获取所有商品
    groups = sorted(pred_element["group"].unique())
    print(f"[PLOT] Element '{element_type}' has {len(groups)} groups")

    # 根据商品数决定子图布局
    n_groups = len(groups)
    n_cols = 5
    n_rows = (n_groups + n_cols - 1) // n_cols

    # 图的高度要足够大
    print(f"[PLOT] Layout: {n_rows} rows × {n_cols} cols")

    fig = plt.figure(figsize=(5 * n_cols, 5 * n_rows + 2))
    fig.suptitle(
        f"Trade Predictions: {element_type} (Low/Med/High Scenarios)\n"
        f"Quantiles: low={LOW_QUANTILE}, high={HIGH_QUANTILE}",
        fontsize=18, fontweight="bold", y=0.995
    )
    gs = GridSpec(n_rows, n_cols, figure=fig, hspace=0.35, wspace=0.3)

    for idx, group in enumerate(groups):
        row = idx // n_cols
        col = idx % n_cols
        ax = fig.add_subplot(gs[row, col])

        # 历史数据
        hist_group = hist_element[hist_element["group"] == group].copy()
        hist_group = hist_group.sort_values("year")

        if not hist_group.empty:
            ax.plot(hist_group["year"], hist_group["trade"],
                    color="black", linewidth=2.5, label = "History (1990–2014)",
            marker = "o", markersize = 4, zorder = 3)

            # 预测数据（三个情景）
            pred_group = pred_element[pred_element["group"] == group].copy()

            # 低增长情景
            pred_low = pred_group[pred_group["scenario"] == "low"].sort_values("year")
            if not pred_low.empty:
                ax.plot(pred_low["year"], pred_low["pred_mean"],
                        color="blue", linestyle="--", linewidth=1.8, label=f"Low ({LOW_QUANTILE})",
                        marker="x", markersize=5, zorder=2)

            # 中等情景
            pred_med = pred_group[pred_group["scenario"] == "med"].sort_values("year")
            if not pred_med.empty:
                ax.plot(pred_med["year"], pred_med["pred_mean"],
                        color="green", linestyle="--", linewidth=1.8, label="Medium (0.50)",
                        marker="^", markersize=5, zorder=2)

            # 高增长情景
            pred_high = pred_group[pred_group["scenario"] == "high"].sort_values("year")
            if not pred_high.empty:
                ax.plot(pred_high["year"], pred_high["pred_mean"],
                        color="red", linestyle="--", linewidth=1.8, label=f"High ({HIGH_QUANTILE})",
                        marker="s", markersize=5, zorder=2)

            # 格式化
            ax.set_title(group, fontsize=10, fontweight="bold")
            ax.set_xlabel("Year", fontsize=9)
            ax.set_ylabel("Trade Volume", fontsize=9)
            ax.tick_params(labelsize=8)
            ax.grid(True, alpha=0.25, linestyle="-", linewidth=0.5)
            ax.legend(fontsize=8, loc="best", framealpha=0.9)

            # x 轴范围覆盖历史和预测
            all_years = []
            if not hist_group.empty:
                all_years.extend(hist_group["year"].tolist())
            if not pred_group.empty:
                all_years.extend(pred_group["year"].tolist())
            if all_years:
                ax.set_xlim(min(all_years) - 1, max(all_years) + 1)

            # 在历史和预测之间用竖线标记分界
            ax.axvline(x=2014.5, color = "gray", linestyle = ":", linewidth = 1.2, alpha = 0.6, zorder = 1)
            ax.text(2014.5, ax.get_ylim()[1] * 0.98, "↓ Forecast ↓",
                    fontsize=7, ha="center", color="gray", fontweight="bold")

    print(f"[PLOT] Figure created successfully for {element_type}")
    return fig

# ========== 第四部分：画图（小图版） ==========

def plot_single_group(group_name: str, element_type: str,
                      pred_df: pd.DataFrame, hist_agg: pd.DataFrame,
                      out_dir: str = OUT_GROUP_DIR):
    """
    为单个商品 + Element 组合画一张独立的图。
    """
    Path(out_dir).mkdir(exist_ok=True, parents=True)

    fig, ax = plt.subplots(figsize=(12, 6))

    # 历史数据
    hist_group = hist_agg[(hist_agg["group"] == group_name) &
                          (hist_agg["Element"] == element_type)].copy()
    hist_group = hist_group.sort_values("year")

    if not hist_group.empty:
        ax.plot(hist_group["year"], hist_group["trade"],
                color="black", linewidth=3, label="History (1990–2014)",
                marker="o", markersize=6, zorder=3)

    # 预测
    pred_group = pred_df[(pred_df["group"] == group_name) &
                         (pred_df["element"] == element_type)].copy()

    # 三个情景
    scenario_info = [
        ("low", "blue", "x", f"Low ({LOW_QUANTILE})"),
        ("med", "green", "^", "Medium (0.50)"),
        ("high", "red", "s", f"High ({HIGH_QUANTILE})"),
    ]

    for scen, color, marker, label in scenario_info:
        pred_scen = pred_group[pred_group["scenario"] == scen].sort_values("year")
        if not pred_scen.empty:
            ax.plot(pred_scen["year"], pred_scen["pred_mean"],
                    color=color, linestyle="--", linewidth=2.5,
            label = label, marker = marker, markersize = 7, zorder = 2)

            # 格式化
            ax.set_title(f"{group_name} - {element_type}",
                         fontsize=14, fontweight="bold", pad=20)
            ax.set_xlabel("Year", fontsize=12, fontweight="bold")
            ax.set_ylabel("Trade Volume", fontsize=12, fontweight="bold")
            ax.tick_params(labelsize=10)
            ax.grid(True, alpha=0.3, linestyle="-", linewidth=0.8)
            ax.legend(fontsize=11, loc="best", framealpha=0.95, edgecolor="black")

            # 在历史和预测之间用竖线标记分界
            ax.axvline(x=2014.5, color="gray", linestyle=":", linewidth=1.5, alpha=0.6, zorder=1)
            ax.text(2014.5, ax.get_ylim()[1] * 0.98, "↓ Forecast ↓",
                    fontsize=10, ha="center", color="gray", fontweight="bold")

            # 保存
            safe_group = re.sub(r"[^0-9A-Za-z._-]+", "_", group_name)[:50]
            safe_element = re.sub(r"[^0-9A-Za-z._-]+", "_", element_type)[:30]
            out_path = Path(out_dir) / f"{safe_group}_{safe_element}.png"

            fig.savefig(out_path, dpi=150, bbox_inches="tight", pad_inches=0.15)
            plt.show()
            plt.close(fig)

            return out_path

def plot_all_single_groups(pred_df: pd.DataFrame, hist_agg: pd.DataFrame):
    """
为所有商品 × Element 组合生成单独的图。
"""
    print(f"\n[PLOT] ===== Plotting all groups (INDIVIDUAL FIGURES) =====")

    pairs = pred_df[["group", "element"]].drop_duplicates().values.tolist()
    print(f"[PLOT] Total (group, element) pairs: {len(pairs)}")

    success_count = 0
    for idx, (group_name, element_type) in enumerate(pairs, start=1):
        try:
            out_path = plot_single_group(group_name, element_type, pred_df, hist_agg)
            if (idx % 10) == 0:
                print(f"[PLOT] Completed {idx}/{len(pairs)} figures")
            success_count += 1
        except Exception as e:
            print(f"[PLOT] WARNING: Failed to plot ({group_name}, {element_type}): {e}")

    print(
        f"[PLOT] Successfully created {success_count}/{len(pairs)} individual figures in {OUT_GROUP_DIR}/")

# ========== MAIN ==========

def main():
    print("=" * 80)
    print("REGENERATE PREDICTIONS + PLOT")
    print("=" * 80)

    # 第 1 步：重新生成预测
    pred_df = regenerate_predictions()
    if pred_df.empty:
        print("[ERROR] Failed to generate predictions, exit.")
        sys.exit(1)

    # 第 2 步：加载历史数据
    hist_agg = load_history_data()

    # 第 3 步：画图
    if USE_SINGLE_GROUP_PLOTS:
        # 小图模式
        plot_all_single_groups(pred_df, hist_agg)
    else:
        # 大图模式

        # 绘制出口
        print("\n[PLOT] ===== EXPORT QUANTITY =====")
        fig_export = plot_element_large("Export Quantity", pred_df, hist_agg)
        if fig_export is not None:
            try:
                fig_export.savefig(OUT_EXPORT_PNG, dpi=100, bbox_inches="tight", pad_inches=0.2)
                print(f"[PLOT] ✓ Saved export plot to: {OUT_EXPORT_PNG}")

                file_size_mb = os.path.getsize(OUT_EXPORT_PNG) / (1024 * 1024)
                print(f"[PLOT] File size: {file_size_mb:.2f} MB")
                plt.show()
                plt.close(fig_export)
            except Exception as e:
                print(f"[PLOT] ✗ Error saving export plot: {e}")
        else:
            print(f"[PLOT] ✗ Failed to create export figure")

        # 绘制进口
        print("\n[PLOT] ===== IMPORT QUANTITY =====")
        fig_import = plot_element_large("Import Quantity", pred_df, hist_agg)
        if fig_import is not None:
            try:
                fig_import.savefig(OUT_IMPORT_PNG, dpi=100, bbox_inches="tight", pad_inches=0.2)
                print(f"[PLOT] ✓ Saved import plot to: {OUT_IMPORT_PNG}")

                file_size_mb = os.path.getsize(OUT_IMPORT_PNG) / (1024 * 1024)
                print(f"[PLOT] File size: {file_size_mb:.2f} MB")

                plt.close(fig_import)
            except Exception as e:
                print(f"[PLOT] ✗ Error saving import plot: {e}")
        else:
            print(f"[PLOT] ✗ Failed to create import figure")

    print("\n" + "=" * 80)
    print("DONE!")
    print("=" * 80)

if __name__ == "__main__":
    main()