import matplotlib
matplotlib.use("Agg")

import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import t

from tools import predict_growth_index


SCRIPT_DIR = Path(__file__).resolve().parent
AG2050_DIR = SCRIPT_DIR.parent
ORIGINAL_DIR = AG2050_DIR / "0_original_data"
PROCESSED_DIR = AG2050_DIR / "2_processed_data"
RESULTS_DIR = AG2050_DIR / "3_Results"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# The production script used 75%. Here we compare wider intervals up to 95%.
# Avoid >95% because predict_growth_index defines Very_High as the 95% upper bound.
PI_LEVELS = [0.75, 0.85, 0.90, 0.95]
SCENARIO_COLS = ["Low", "Medium", "High", "Very_High"]
SLOPE_QUANTILES = {
    # Mirrors the earlier scenario logic: Low/High are the central 75% slope
    # interval, while Very High uses the upper edge of the 95% interval.
    "Low": 0.125,
    "Medium": 0.5,
    "High": 0.875,
    "Very_High": 0.975,
}
SCENARIO_LABELS = {
    "Low": "Low",
    "Medium": "Medium",
    "High": "High",
    "Very_High": "Very High",
}
COLORS = {
    "Historical": "black",
    "Fitted": "#0b3d91",
    "Low": "#f28e2b",
    "Medium": "#59a14f",
    "High": "#e15759",
    "Very_High": "#9467bd",
}


def load_labour_cost():
    """Read and harmonise the historical labour-cost series used by the test."""
    df = pd.read_excel(ORIGINAL_DIR / "labour_cost.xlsx", usecols="A,G")
    col0, col1 = df.columns[0], df.columns[1]

    # Parse values like "Aug-14" to 2014.
    df["Year"] = pd.to_datetime(df[col0], format="%b-%y", errors="coerce").dt.year
    df.rename(columns={col1: "Cost"}, inplace=True)
    df["Cost"] = pd.to_numeric(df["Cost"], errors="coerce")
    df = (
        df[["Year", "Cost"]]
        .dropna(subset=["Year"])
        .sort_values("Year")
        .reset_index(drop=True)
    )

    # Fill 2010-2013 with the first available 2014 cost, matching the original logic.
    cost_2014 = df.loc[df["Year"] == 2014, "Cost"].values[0]
    pre_years = pd.DataFrame({"Year": [2010, 2011, 2012, 2013], "Cost": cost_2014})
    df = (
        pd.concat([pre_years, df], ignore_index=True)
        .drop_duplicates("Year")
        .sort_values("Year")
        .reset_index(drop=True)
    )
    return df


def run_forecast(df, pi_level):
    """Run the same ETS forecast with a selected prediction interval width."""
    ax, df_result = predict_growth_index(
        df,
        var_name="Labour Cost",
        pi_level=pi_level,
        base_year=2010,
        draw_base_year=2010,
        model="ETS",
        align_scenarios_to_last_actual=False,
        use_fitted_for_history=True,
    )
    plt.close(ax.get_figure())
    return df_result


def collect_y_values(forecast_results):
    vals = []
    for df_result in forecast_results.values():
        for col in ["Historical", "Fitted"]:
            if col in df_result.columns:
                vals.extend(df_result[col].dropna().to_numpy(dtype=float))
        for col in SCENARIO_COLS:
            plot_col = f"Plot_{col}"
            if plot_col in df_result.columns:
                vals.extend(df_result[plot_col].dropna().to_numpy(dtype=float))
    return np.asarray(vals, dtype=float)


def plot_interval_comparison(forecast_results, out_path):
    y_values = collect_y_values(forecast_results)
    y_values = y_values[np.isfinite(y_values)]
    y_min = min(0.0, float(y_values.min()) * 1.05)
    y_max = float(y_values.max()) * 1.08

    fig, axes = plt.subplots(2, 2, figsize=(16, 10), sharex=True, sharey=True)
    axes = axes.ravel()

    for ax, (pi_level, df_result) in zip(axes, forecast_results.items()):
        hist = df_result["Historical"].dropna()
        fitted = df_result["Fitted"].dropna()

        ax.scatter(
            hist.index,
            hist.values,
            s=34,
            color=COLORS["Historical"],
            label="Historical",
            zorder=5,
        )
        ax.plot(
            fitted.index,
            fitted.values,
            color=COLORS["Fitted"],
            linewidth=2.2,
            label="ETS fitted",
            zorder=4,
        )

        for col in SCENARIO_COLS:
            plot_col = f"Plot_{col}"
            series = df_result[plot_col].dropna()
            ax.plot(
                series.index,
                series.values,
                color=COLORS[col],
                linewidth=2.0 if col == "Medium" else 1.8,
                linestyle="-" if col == "Medium" else "--",
                label=SCENARIO_LABELS[col],
                zorder=6,
            )

        ax.axhline(1.0, linestyle=":", color="0.55", linewidth=1.0, zorder=1)
        ax.set_title(f"{int(pi_level * 100)}% prediction interval", fontsize=15, weight="bold")
        ax.set_xlim(2010, 2050)
        ax.set_ylim(y_min, y_max)
        ax.set_xticks(np.arange(2010, 2051, 5))
        ax.grid(True, alpha=0.28)
        ax.tick_params(labelsize=11)

    for ax in axes[::2]:
        ax.set_ylabel("Labour cost multiplier", fontsize=13)
    for ax in axes[-2:]:
        ax.set_xlabel("Year", fontsize=13)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=6,
        frameon=False,
        fontsize=12,
    )
    fig.suptitle(
        "Labour cost forecast under alternative prediction interval widths",
        fontsize=18,
        weight="bold",
        y=0.985,
    )
    fig.tight_layout(rect=[0.02, 0.08, 1, 0.95])
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def write_forecast_workbook(forecast_results, out_path):
    summary_rows = []
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        for pi_level, df_result in forecast_results.items():
            sheet_name = f"PI_{int(pi_level * 100)}"
            df_result.to_excel(writer, sheet_name=sheet_name, index=True)

            summary = df_result.loc[2050, SCENARIO_COLS].copy()
            row = {"prediction_interval": f"{int(pi_level * 100)}%"}
            row.update(summary.to_dict())
            summary_rows.append(row)

        pd.DataFrame(summary_rows).to_excel(writer, sheet_name="2050_summary", index=False)

    return pd.DataFrame(summary_rows)


def build_statistical_slope_scenarios(df):
    """Fit log-linear slope uncertainty and convert slope quantiles to scenarios."""
    base_value = float(df.loc[df["Year"] == 2010, "Cost"].iloc[0])
    hist = df.copy()
    hist["Index"] = hist["Cost"] / base_value

    x = (hist["Year"] - hist["Year"].min()).to_numpy(dtype=float)
    y = np.log(hist["Index"].to_numpy(dtype=float))
    model = sm.OLS(y, sm.add_constant(x)).fit()

    slope = float(model.params[1])
    slope_se = float(model.bse[1])
    df_resid = float(model.df_resid)

    start_year = int(hist["Year"].max())
    end_year = 2050
    start_value = float(hist.loc[hist["Year"] == start_year, "Index"].iloc[0])
    years = np.arange(start_year, end_year + 1, dtype=int)
    elapsed = years - start_year

    scenarios = pd.DataFrame(index=years)
    summary_rows = []
    for scenario, quantile in SLOPE_QUANTILES.items():
        if quantile == 0.5:
            scenario_slope = slope
        else:
            scenario_slope = slope + t.ppf(quantile, df_resid) * slope_se

        scenarios[scenario] = start_value * np.exp(scenario_slope * elapsed)
        summary_rows.append(
            {
                "scenario": SCENARIO_LABELS[scenario],
                "slope_quantile": quantile,
                "annual_log_slope": scenario_slope,
                "annualised_growth_rate": np.exp(scenario_slope) - 1,
                "start_year": start_year,
                "start_multiplier": start_value,
                "multiplier_2050": float(scenarios.loc[end_year, scenario]),
            }
        )

    scenarios.index.name = "Year"
    hist_series = hist.set_index("Year")["Index"]
    fitted_series = pd.Series(
        np.exp(model.fittedvalues),
        index=hist["Year"].astype(int),
        name="Fitted",
    )
    summary = pd.DataFrame(summary_rows)
    return scenarios, hist_series, fitted_series, summary, model, start_year, end_year


def plot_statistical_slope_scenarios(df, out_path):
    scenarios, hist, fitted, summary, model, start_year, end_year = (
        build_statistical_slope_scenarios(df)
    )

    y_values = np.concatenate(
        [
            hist.to_numpy(dtype=float),
            fitted.to_numpy(dtype=float),
            scenarios[SCENARIO_COLS].to_numpy(dtype=float).ravel(),
        ]
    )
    y_min = min(0.0, float(np.nanmin(y_values)) * 1.05)
    y_max = float(np.nanmax(y_values)) * 1.08

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.scatter(
        hist.index,
        hist.values,
        s=42,
        color=COLORS["Historical"],
        label="Historical",
        zorder=5,
    )
    ax.plot(
        fitted.index,
        fitted.values,
        color=COLORS["Fitted"],
        linewidth=2.4,
        label="ETS fitted",
        zorder=4,
    )

    for col in SCENARIO_COLS:
        ax.plot(
            scenarios.index,
            scenarios[col],
            color=COLORS[col],
            linewidth=2.5 if col == "Medium" else 2.0,
            linestyle="-" if col == "Medium" else "--",
            label=f"{SCENARIO_LABELS[col]} (slope q={SLOPE_QUANTILES[col]:.3g})",
            zorder=6,
        )

    ax.axhline(1.0, linestyle=":", color="0.55", linewidth=1.0, zorder=1)
    ax.axvline(start_year, linestyle=":", color="0.6", linewidth=1.0, zorder=1)
    ax.set_title(
        "Labour cost scenarios from fitted trend-slope uncertainty",
        fontsize=17,
        weight="bold",
    )
    ax.set_xlabel("Year", fontsize=13)
    ax.set_ylabel("Labour cost multiplier", fontsize=13)
    ax.set_xlim(2010, end_year)
    ax.set_ylim(y_min, y_max)
    ax.set_xticks(np.arange(2010, end_year + 1, 5))
    ax.grid(True, alpha=0.28)
    ax.tick_params(labelsize=11)
    ax.legend(loc="upper left", frameon=True, fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)

    model_stats = pd.DataFrame(
        [
            {
                "r_squared": model.rsquared,
                "slope": model.params[1],
                "slope_standard_error": model.bse[1],
                "slope_p_value": model.pvalues[1],
                "n_observations": int(model.nobs),
                "df_resid": model.df_resid,
            }
        ]
    )
    return scenarios, hist, fitted, summary, model_stats


def main():
    df = load_labour_cost()
    forecast_results = {pi: run_forecast(df, pi) for pi in PI_LEVELS}

    plot_path = SCRIPT_DIR / "labour_cost_prediction_interval_comparison.png"
    workbook_path = PROCESSED_DIR / "labour_cost_prediction_interval_comparison.xlsx"
    slope_plot_path = SCRIPT_DIR / "labour_cost_statistical_slope_scenarios.png"
    slope_workbook_path = PROCESSED_DIR / "labour_cost_statistical_slope_scenarios.xlsx"

    plot_interval_comparison(forecast_results, plot_path)
    summary_2050 = write_forecast_workbook(forecast_results, workbook_path)
    slope_scenarios, hist, fitted, slope_summary, model_stats = plot_statistical_slope_scenarios(
        df,
        slope_plot_path,
    )

    with pd.ExcelWriter(slope_workbook_path, engine="openpyxl") as writer:
        slope_scenarios.to_excel(writer, sheet_name="slope_scenarios", index=True)
        hist.to_excel(writer, sheet_name="historical", index=True)
        fitted.to_excel(writer, sheet_name="log_linear_fitted", index=True)
        slope_summary.to_excel(writer, sheet_name="summary", index=False)
        model_stats.to_excel(writer, sheet_name="model_stats", index=False)

    for result_file in [plot_path, workbook_path, slope_plot_path, slope_workbook_path]:
        dst = RESULTS_DIR / result_file.name
        shutil.copy2(result_file, dst)
        print(f"Copied {result_file} to {dst}")

    print("\n2050 labour cost multipliers by prediction interval:")
    print(summary_2050.to_string(index=False))
    print("\n2050 labour cost multipliers by fitted slope scenario:")
    print(slope_summary.to_string(index=False))


if __name__ == "__main__":
    main()
