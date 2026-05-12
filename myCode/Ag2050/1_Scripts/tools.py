import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from scipy.stats import norm
from statsmodels.tsa.exponential_smoothing.ets import ETSModel
from statsmodels.tsa.statespace.structural import UnobservedComponents

warnings.filterwarnings("ignore")


def predict_growth_index(
    df,
    var_name="labour cost",
    pi_level=0.75,
    base_year=2010,
    model="UnobservedComponents",
    use_index=True,
    draw_base_year=2010,
    align_scenarios_to_last_actual=True,
):
    """
    Workflow:
    1. Fit on the raw scale.
    2. Shift intercepts so the fitted line and all scenarios pass through
       the last historical value.
    3. Scale everything by the adjusted fitted value at base_year.

    Returns:
        ax, df_return(index=Year, cols=['Low', 'Medium', 'High', 'Very_High'])
    """
    end_year = 2050

    df = df[["Year", "Cost"]].dropna().sort_values("Year").reset_index(drop=True)
    df["Year"] = df["Year"].astype(int)

    if base_year not in df["Year"].values:
        raise ValueError(f"base_year={base_year} is not present in the input data.")

    y_ts = pd.Series(
        df["Cost"].values.astype(float),
        index=pd.to_datetime(df["Year"].astype(str) + "-01-01"),
        name="Cost",
    )

    if model == "ETS":
        mod = ETSModel(y_ts, error="add", trend="add", damped_trend=True)
        res = mod.fit(disp=False)
    elif model == "OLS":
        x_hist = sm.add_constant(df["Year"].values.astype(float))
        res = sm.OLS(df["Cost"].values.astype(float), x_hist).fit()
    elif model == "UnobservedComponents":
        mod = UnobservedComponents(
            y_ts,
            trend=True,
            stochastic_level=True,
            stochastic_trend=True,
            irregular=True,
        )
        res = mod.fit(disp=False, maxiter=2000)
    else:
        raise ValueError(f"Unsupported model: {model}")

    last_hist_year = int(df["Year"].max())
    y_last = float(df.loc[df["Year"] == last_hist_year, "Cost"].iloc[0])

    if model == "OLS":
        pred_years = np.arange(int(df["Year"].min()), end_year + 1, dtype=int)
        x_pred = sm.add_constant(pred_years.astype(float))
        pred_80 = res.get_prediction(x_pred)
        pred_95 = res.get_prediction(x_pred)
    else:
        pred_years = None
        start_dt = y_ts.index.min()
        end_dt = pd.to_datetime(f"{end_year}-01-01")
        pred_80 = res.get_prediction(start=start_dt, end=end_dt)
        pred_95 = res.get_prediction(start=start_dt, end=end_dt)

    def _to_interval_df(pred, level):
        alpha = 1 - level
        sf = pred.summary_frame(alpha=alpha)
        mean = sf["mean"].to_numpy(dtype=float)

        if model == "OLS":
            lower = sf["mean_ci_lower"].to_numpy(dtype=float)
            upper = sf["mean_ci_upper"].to_numpy(dtype=float)
            years = pred_years.astype(int)
        else:
            if {"pi_lower", "pi_upper"}.issubset(sf.columns):
                lower = sf["pi_lower"].to_numpy(dtype=float)
                upper = sf["pi_upper"].to_numpy(dtype=float)
            elif {"obs_ci_lower", "obs_ci_upper"}.issubset(sf.columns):
                lower = sf["obs_ci_lower"].to_numpy(dtype=float)
                upper = sf["obs_ci_upper"].to_numpy(dtype=float)
            else:
                z = norm.ppf(1 - alpha / 2.0)
                if {"mean_ci_lower", "mean_ci_upper"}.issubset(sf.columns):
                    mean_se = (
                        sf["mean_ci_upper"].to_numpy(dtype=float)
                        - sf["mean_ci_lower"].to_numpy(dtype=float)
                    ) / (2.0 * z)
                elif "mean_se" in sf.columns:
                    mean_se = sf["mean_se"].to_numpy(dtype=float)
                elif hasattr(pred, "var_pred_mean"):
                    mean_se = np.sqrt(np.asarray(pred.var_pred_mean, dtype=float))
                else:
                    mean_se = np.zeros_like(mean)
                sigma2 = float(res.sigma2) if hasattr(res, "sigma2") else float(res.scale)
                half_w = z * np.sqrt(mean_se**2 + sigma2)
                lower = mean - half_w
                upper = mean + half_w
            years = sf.index.year.astype(int)

        return pd.DataFrame(
            {
                "Year": years,
                "Mean": mean,
                "Lower": lower,
                "Upper": upper,
            }
        ).drop_duplicates(subset=["Year"]).reset_index(drop=True)

    df80 = _to_interval_df(pred_80, pi_level)
    df95 = _to_interval_df(pred_95, 0.95)

    if model == "OLS":
        df80["Very_High"] = df95["Upper"] + (df95["Upper"] - df95["Mean"])
    else:
        df80["Very_High"] = df95["Upper"]

    df80["Raw_SES_Mean"] = df80["Mean"].copy()

    def _align_column_to_last_actual(df_in, col_name):
        last_pred = float(df_in.loc[df_in["Year"] == last_hist_year, col_name].iloc[0])
        df_in[col_name] = df_in[col_name] + (y_last - last_pred)

    if model == "OLS":
        if align_scenarios_to_last_actual:
            for col_name in ["Mean", "Lower", "Upper", "Very_High", "Raw_SES_Mean"]:
                _align_column_to_last_actual(df80, col_name)
            for col_name in ["Mean", "Lower", "Upper"]:
                _align_column_to_last_actual(df95, col_name)

        if use_index:
            base_val = float(df80.loc[df80["Year"] == base_year, "Raw_SES_Mean"].iloc[0])
        else:
            base_val = 1.0
    else:
        if align_scenarios_to_last_actual:
            pred_mean_last = float(df80.loc[df80["Year"] == last_hist_year, "Mean"].iloc[0])
            offset_mean = y_last - pred_mean_last
            for dfp in (df80, df95):
                for col_name in ["Mean", "Lower", "Upper"]:
                    dfp[col_name] = dfp[col_name] + offset_mean
            df80["Very_High"] = df80["Very_High"] + offset_mean
            df80["Raw_SES_Mean"] = df80["Raw_SES_Mean"] + offset_mean

        if use_index:
            base_val = float(df.loc[df["Year"] == base_year, "Cost"].iloc[0])
        else:
            base_val = 1.0

    for col_name in ["Mean", "Lower", "Upper", "Very_High", "Raw_SES_Mean"]:
        df80[f"{col_name}_Index"] = df80[col_name] / base_val
    for col_name in ["Mean", "Lower", "Upper"]:
        df95[f"{col_name}_Index"] = df95[col_name] / base_val

    df80 = df80[df80["Year"] >= draw_base_year].copy()
    df95 = df95[df95["Year"] >= draw_base_year].copy()

    df_hist = df[df["Year"] >= draw_base_year].copy()
    df_hist["Growth_Index"] = df_hist["Cost"] / base_val
    df_hist["Scenario"] = "Historical"

    def make_scenario_df(df_in, col_name, scenario, start_year):
        sub = df_in[df_in["Year"] >= start_year][["Year", col_name]].copy()
        sub.columns = ["Year", "Growth_Index"]
        sub["Scenario"] = scenario
        return sub

    df_medium = make_scenario_df(df80, "Mean_Index", "Medium", last_hist_year)
    df_low = make_scenario_df(df80, "Lower_Index", "Low", last_hist_year)
    df_high = make_scenario_df(df80, "Upper_Index", "High", last_hist_year)
    df_vhigh = make_scenario_df(df80, "Very_High_Index", "Very High", last_hist_year)
    fit_label = "Raw_OLS_Fit" if model == "OLS" else "Raw_SES_Fit"
    df_raw_ses = make_scenario_df(df80, "Raw_SES_Mean_Index", fit_label, draw_base_year)
    df_raw_ses = df_raw_ses[df_raw_ses["Year"] <= last_hist_year].copy()

    sns.set(style="whitegrid")
    fig, ax = plt.subplots(figsize=(12, 6))

    deep_blue = "#0b3d91"
    deeper_blue = "#082b6a"
    color_dict = {
        "Historical": "black",
        "Raw_OLS_Fit": deep_blue,
        "Raw_SES_Fit": deep_blue,
        "Medium": "#2ca02c",
        "Low": "#ff7f0e",
        "High": "#d62728",
        "Very High": "#9467bd",
    }

    min_hist_year = int(df_hist["Year"].min())
    if min_hist_year > draw_base_year:
        missing_years = list(range(draw_base_year, min_hist_year))
        first_row = df_hist.iloc[0]
        df_fill = pd.DataFrame(
            {
                "Year": missing_years,
                "Cost": float(first_row["Cost"]),
                "Growth_Index": float(first_row["Growth_Index"]),
                "Scenario": "Historical",
            }
        )
        df_hist = pd.concat([df_fill, df_hist], ignore_index=True)
        df_hist = df_hist.sort_values("Year").reset_index(drop=True)

    ax.scatter(
        df_hist["Year"],
        df_hist["Growth_Index"],
        s=40,
        c=color_dict["Historical"],
        label="Historical (actual data)",
        zorder=6,
    )

    ax.plot(
        df_raw_ses["Year"],
        df_raw_ses["Growth_Index"],
        color=color_dict[fit_label],
        linewidth=2.8,
        label=f"{model} mean",
        zorder=5,
    )

    env95 = df95[df95["Year"] >= last_hist_year].copy()
    if not env95.empty:
        ax.fill_between(
            env95["Year"].to_numpy(),
            env95["Lower_Index"].to_numpy(),
            env95["Upper_Index"].to_numpy(),
            color=deeper_blue,
            alpha=0.12,
            label="Prediction interval (95%)",
            zorder=2,
        )

    env80 = df80[df80["Year"] >= last_hist_year].copy()
    if not env80.empty:
        ax.fill_between(
            env80["Year"].to_numpy(),
            env80["Lower_Index"].to_numpy(),
            env80["Upper_Index"].to_numpy(),
            color=deep_blue,
            alpha=0.18,
            label=f"Prediction interval ({int(pi_level * 100)}%)",
            zorder=3,
        )
        ax.plot(
            env80["Year"],
            env80["Lower_Index"],
            color=deep_blue,
            linewidth=1.0,
            alpha=0.6,
            linestyle="--",
            zorder=4,
        )
        ax.plot(
            env80["Year"],
            env80["Upper_Index"],
            color=deep_blue,
            linewidth=1.0,
            alpha=0.6,
            linestyle="--",
            zorder=4,
        )

    for dfi, label, color, width in [
        (
            df_low,
            "Low = mean 95% CI lower" if model == "OLS" else f"Low = {int(pi_level * 100)}% PI lower",
            color_dict["Low"],
            1.6,
        ),
        (
            df_medium,
            "Medium = mean (OLS prediction)" if model == "OLS" else "Medium",
            color_dict["Raw_OLS_Fit"] if model == "OLS" else color_dict["Medium"],
            2.8,
        ),
        (
            df_high,
            "High = mean 95% CI upper" if model == "OLS" else f"High = {int(pi_level * 100)}% PI upper",
            color_dict["High"],
            1.6,
        ),
        (
            df_vhigh,
            "Very High = 95% CI upper + (95% CI upper - mean)" if model == "OLS" else "Very High = 95% PI upper",
            color_dict["Very High"],
            1.6,
        ),
    ]:
        if not dfi.empty:
            ax.plot(
                dfi["Year"],
                dfi["Growth_Index"],
                color=color,
                linewidth=width,
                linestyle="--",
                label=label,
                zorder=7,
            )

    ax.axhline(1.0, linestyle="dotted", color="gray", linewidth=1, zorder=1)
    ax.set_title(f"{var_name} Growth Index", fontsize=14, weight="bold")
    ax.set_xlabel("Year", fontsize=12)
    ax.set_ylabel("Growth Index", fontsize=12)
    ax.set_xlim(draw_base_year, end_year)
    ax.set_ylim(bottom=0)
    ax.set_xticks(range(draw_base_year, end_year + 1, 5))
    ax.legend(loc="upper left", frameon=True, fontsize=9, ncol=1)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    df_return = df80[["Year", "Lower_Index", "Mean_Index", "Upper_Index", "Very_High_Index"]].copy()
    df_return.columns = ["Year", "Low", "Medium", "High", "Very_High"]
    df_return = df_return.set_index("Year").reindex(np.arange(draw_base_year, end_year + 1))

    hist_map = (
        df[df["Year"] >= draw_base_year]
        .set_index("Year")["Cost"]
        .astype(float) / base_val
    )
    common_years = hist_map.index.intersection(df_return.index)
    if len(common_years) > 0:
        vals = hist_map.loc[common_years].values.reshape(-1, 1)
        df_return.loc[common_years, ["Low", "Medium", "High", "Very_High"]] = vals

    # Save the exact data used in plotting as additional columns.
    df_return["Historical"] = np.nan
    hist_idx = df_hist.set_index("Year")["Growth_Index"]
    common_hist_years = hist_idx.index.intersection(df_return.index)
    if len(common_hist_years) > 0:
        df_return.loc[common_hist_years, "Historical"] = hist_idx.loc[common_hist_years]

    df_return["Fitted"] = np.nan
    fit_idx = df_raw_ses.set_index("Year")["Growth_Index"]
    common_fit_years = fit_idx.index.intersection(df_return.index)
    if len(common_fit_years) > 0:
        df_return.loc[common_fit_years, "Fitted"] = fit_idx.loc[common_fit_years]

    plot_series_map = {
        "Plot_Low": df_low,
        "Plot_Medium": df_medium,
        "Plot_High": df_high,
        "Plot_Very_High": df_vhigh,
    }
    for col_name, df_plot in plot_series_map.items():
        df_return[col_name] = np.nan
        plot_idx = df_plot.set_index("Year")["Growth_Index"]
        common_plot_years = plot_idx.index.intersection(df_return.index)
        if len(common_plot_years) > 0:
            df_return.loc[common_plot_years, col_name] = plot_idx.loc[common_plot_years]

    df_return.fillna(1.0, inplace=True)

    return ax, df_return
