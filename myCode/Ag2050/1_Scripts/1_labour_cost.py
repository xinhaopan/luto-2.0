import matplotlib
matplotlib.use("Agg")

import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tools import predict_growth_index


SCRIPT_DIR = Path(__file__).resolve().parent
AG2050_DIR = SCRIPT_DIR.parent
ORIGINAL_DIR = AG2050_DIR / "0_original_data"
PROCESSED_DIR = AG2050_DIR / "2_processed_data"
RESULTS_DIR = AG2050_DIR / "3_Results"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SCENARIO_COLS = ["Low", "Medium", "High", "Very_High"]
GROWTH_RATE_FACTORS = {
    "Low": 0.5,
    "Medium": 1.0,
    "High": 1.5,
    "Very_High": 2.0,
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
    """Read and harmonise the historical labour-cost series."""
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

    # Fill 2010-2013 with the first available 2014 cost, matching previous logic.
    cost_2014 = df.loc[df["Year"] == 2014, "Cost"].values[0]
    pre_years = pd.DataFrame({"Year": [2010, 2011, 2012, 2013], "Cost": cost_2014})
    df = (
        pd.concat([pre_years, df], ignore_index=True)
        .drop_duplicates("Year")
        .sort_values("Year")
        .reset_index(drop=True)
    )
    return df


def run_ets_baseline(df):
    """Use ETS to define the Medium baseline path before scenario scaling."""
    ax, df_result = predict_growth_index(
        df,
        var_name="Labour Cost",
        pi_level=0.75,
        base_year=2010,
        draw_base_year=2010,
        model="ETS",
        align_scenarios_to_last_actual=True,
        use_fitted_for_history=True,
    )
    plt.close(ax.get_figure())
    df_result = df_result.copy()
    df_result.index = df_result.index.astype(int)
    return df_result


def build_growth_rate_scenarios(df_result):
    """Scale the ETS Medium cumulative growth path by scenario growth factors."""
    start_year = int(df_result["Historical"].dropna().index.max())
    end_year = int(df_result.index.max())
    start_value = float(df_result.loc[start_year, "Historical"])
    future_years = np.arange(start_year, end_year + 1, dtype=int)

    medium_future = (
        df_result["Plot_Medium"]
        .reindex(future_years)
        .interpolate()
        .ffill()
        .bfill()
    )
    medium_future.loc[start_year] = start_value
    growth_ratio = medium_future / start_value

    out = df_result.copy()

    # Keep the fitted historical path for the non-actual scenario sheets. The
    # *_actual sheets below replace these historical values with observations.
    fitted_history = out["Fitted"].combine_first(out["Medium"])
    hist_years = out.index[out.index < start_year]
    for scenario in SCENARIO_COLS:
        out.loc[hist_years, scenario] = fitted_history.loc[hist_years]
        out[f"Plot_{scenario}"] = np.nan
        out.loc[future_years, scenario] = (
            start_value * np.power(growth_ratio, GROWTH_RATE_FACTORS[scenario])
        )
        out.loc[future_years, f"Plot_{scenario}"] = out.loc[future_years, scenario]

    summary_rows = []
    n_years = end_year - start_year
    for scenario in SCENARIO_COLS:
        end_value = float(out.loc[end_year, scenario])
        cagr = (end_value / start_value) ** (1 / n_years) - 1
        summary_rows.append(
            {
                "scenario": SCENARIO_LABELS[scenario],
                "growth_rate_factor": GROWTH_RATE_FACTORS[scenario],
                "start_year": start_year,
                "start_multiplier": start_value,
                "multiplier_2050": end_value,
                "annualised_growth_rate": cagr,
            }
        )

    return out, pd.DataFrame(summary_rows)


def plot_growth_rate_scenarios(df_result, out_path):
    fig, ax = plt.subplots(figsize=(12, 6.5))

    hist = df_result["Historical"].dropna()
    fitted = df_result["Fitted"].dropna()
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

    for scenario in SCENARIO_COLS:
        series = df_result[f"Plot_{scenario}"].dropna()
        ax.plot(
            series.index,
            series.values,
            color=COLORS[scenario],
            linewidth=2.5 if scenario == "Medium" else 2.0,
            linestyle="-" if scenario == "Medium" else "--",
            label=f"{SCENARIO_LABELS[scenario]} ({GROWTH_RATE_FACTORS[scenario]:g}x growth rate)",
            zorder=6,
        )

    ax.axhline(1.0, linestyle=":", color="0.55", linewidth=1.0, zorder=1)
    ax.set_title("Labour Cost Growth Index", fontsize=16, weight="bold")
    ax.set_xlabel("Year", fontsize=12)
    ax.set_ylabel("Labour cost multiplier", fontsize=12)
    ax.set_xlim(int(df_result.index.min()), int(df_result.index.max()))
    ax.set_ylim(bottom=0)
    ax.set_xticks(np.arange(int(df_result.index.min()), int(df_result.index.max()) + 1, 5))
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", frameon=True, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def find_col(df_in, targets):
    """Return the first case-insensitive matching column, or None."""
    cols_lower = {c.lower(): c for c in df_in.columns}
    for target in targets:
        key = target.lower()
        if key in cols_lower:
            return cols_lower[key]

    norm_map = {c.replace("_", "").replace(" ", "").lower(): c for c in df_in.columns}
    for target in targets:
        key = target.replace("_", "").replace(" ", "").lower()
        if key in norm_map:
            return norm_map[key]
    return None


def main():
    df = load_labour_cost()
    df_base = run_ets_baseline(df)
    df_result, summary_2050 = build_growth_rate_scenarios(df_base)

    plot_path = SCRIPT_DIR / "labour_cost_growth_index.png"
    forecast_path = PROCESSED_DIR / "labour_cost_forecast.xlsx"
    excel_path = ORIGINAL_DIR / "FLC_cost_multipliers.xlsx"
    template_sheet = "FLC_multiplier"

    plot_growth_rate_scenarios(df_result, plot_path)

    if not excel_path.exists():
        raise FileNotFoundError(f"File does not exist: {excel_path}")

    df_template = pd.read_excel(excel_path, sheet_name=template_sheet)
    if "Year" not in df_template.columns:
        raise KeyError("Cannot find the 'Year' column in the template sheet.")

    years = df_template["Year"].astype(int).tolist()
    crop_cols = [c for c in df_template.columns if c != "Year"]

    print("Read years:", years)
    print("Read crop columns:", crop_cols)

    df_result_actual_history = df_result.copy()
    if "Historical" in df_result_actual_history.columns:
        hist_mask = df_result_actual_history["Historical"].notna()
        for scenario_col in SCENARIO_COLS:
            df_result_actual_history.loc[hist_mask, scenario_col] = (
                df_result_actual_history.loc[hist_mask, "Historical"]
            )

    with pd.ExcelWriter(forecast_path, engine="openpyxl") as writer:
        df_result.to_excel(writer, sheet_name="fitted_scenarios", index=True)
        df_result_actual_history.to_excel(writer, sheet_name="actual_history", index=True)
        summary_2050.to_excel(writer, sheet_name="growth_rate_method", index=False)

    scenario_map = {
        "FLC_multiplier_low": ["Low", "low"],
        "FLC_multiplier_medium": ["Medium", "medium"],
        "FLC_multiplier_high": ["High", "high"],
        "FLC_multiplier_very_high": [
            "Very_High",
            "Very High",
            "VeryHigh",
            "very_high",
            "very high",
            "veryhigh",
        ],
    }

    found_cols = {}
    for sheet, candidates in scenario_map.items():
        col = find_col(df_result, candidates)
        if col is None:
            print(
                f"Warning: no scenario column found for {sheet} in df_result "
                f"(tried {candidates}). This sheet will be NaN."
            )
        else:
            print(f"Scenario '{sheet}' maps to df_result column: {col}")
        found_cols[sheet] = col

    def build_multiplier_sheet(col_name, use_actual_history=False):
        df_sheet = pd.DataFrame(index=years, columns=crop_cols, dtype=float)
        df_sheet.index.name = "Year"

        for year in years:
            if (
                use_actual_history
                and "Historical" in df_result.columns
                and year in df_result.index
                and not pd.isna(df_result.at[year, "Historical"])
            ):
                value = df_result.at[year, "Historical"]
            elif col_name is not None and year in df_result.index:
                value = df_result.at[year, col_name]
            else:
                value = np.nan

            df_sheet.loc[year, :] = value

        return df_sheet.reset_index()

    sheets_to_write = {}
    for sheet_name, col_name in found_cols.items():
        sheets_to_write[sheet_name] = build_multiplier_sheet(col_name, use_actual_history=False)
        sheets_to_write[f"{sheet_name}_actual"] = build_multiplier_sheet(
            col_name,
            use_actual_history=True,
        )

    with pd.ExcelWriter(
        excel_path,
        engine="openpyxl",
        mode="a",
        if_sheet_exists="replace",
    ) as writer:
        for sheet_name, df_sheet in sheets_to_write.items():
            df_sheet.to_excel(writer, sheet_name=sheet_name, index=False)

    print(f"Wrote {len(sheets_to_write)} sheets to {excel_path} (same-name sheets replaced).")

    for result_file in [excel_path, forecast_path, plot_path]:
        dst = RESULTS_DIR / result_file.name
        shutil.copy2(result_file, dst)
        print(f"Copied {result_file} to {dst}")

    print("\n2050 labour cost multipliers by growth-rate scenario:")
    print(summary_2050.to_string(index=False))


if __name__ == "__main__":
    main()
