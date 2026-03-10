import pandas as pd

# ============================================================
# GHG conversion ratios (kg CO2e / kg liveweight, cradle-to-gate, excl. LUC)
# Source: Wiedemann et al. (2017) Animal Production Science 57:1149-1162
# System boundary: breeding + backgrounding + feedlot finishing
# ============================================================

# Grass-fed baseline (kg CO2e / kg LW)
# Back-calculated from Wiedemann et al. (2017) Discussion:
#   grain-finished was 15%, 23%, 13% LOWER than grass-fed (Wiedemann et al. 2016a)
#   short: 9.9 / 0.85 ≈ 11.65
#   mid:   9.4 / 0.77 ≈ 12.21
#   long: 10.6 / 0.87 ≈ 12.18  →  average ≈ 12.0
land_cr  = 12.0   # kg CO2e / kg LW  (grass-fed, same boundary as feedlot values)

# Feedlot systems (kg CO2e / kg LW finished, cradle-to-gate, excl. LUC)
short_cr = 9.9    # short-fed domestic  (55–80  days on feed, ~468 kg LW)
mid_cr   = 9.4    # mid-fed export      (108–164 days on feed, ~652 kg LW)
long_cr  = 10.6   # long-fed export     (>300    days on feed, ~784 kg LW)

cr_map = {
    "land":  land_cr,
    "short": short_cr,
    "mid":   mid_cr,
    "long":  long_cr,
}

# ============================================================
# Load cattle production by stage
# ============================================================
prod_df = pd.read_csv('../2_processed_data/cattle_production_by_stage.csv')
df = prod_df.copy()

# make sure types are OK
df["Year"]       = df["Year"].astype(int)
df["production"] = pd.to_numeric(df["production"], errors="coerce")  # tonnes LW

# map GHG intensity by stage (kg CO2e / kg LW)
df["cr_kgco2e_per_kg"] = df["stage"].map(cr_map)

# optional: check unmapped stages
unmapped = df.loc[df["cr_kgco2e_per_kg"].isna(), "stage"].unique()
if len(unmapped) > 0:
    raise ValueError(f"Unmapped stage values: {unmapped}")

# row-level emissions: production (t) × 1000 → kg LW, × CR (kg CO2e/kg LW) → kg CO2e
df["ghg_kgco2e"] = df["production"] * 1000 * df["cr_kgco2e_per_kg"]

# ---- aggregate: total emissions per Year & Scenario ----
ghg_year_scen = (
    df.groupby(["Scenario_ag", "Year"], as_index=False)["ghg_kgco2e"]
      .sum()
      .rename(columns={"ghg_kgco2e": "ghg_total_kgco2e"})
)

group_cols = ['Scenario_ag', 'Year']
total_production = (
    prod_df.groupby(group_cols, as_index=False)
           .agg(total_production_tonnes=('production', 'sum'))
)
merged_ghg = ghg_year_scen.merge(total_production, on=group_cols, how='left')

# Weighted average GHG intensity (kg CO2e / kg LW)
merged_ghg['FCR_scaled_merged'] = (
    merged_ghg["ghg_total_kgco2e"] / (merged_ghg['total_production_tonnes'] * 1000)
)

# Ratio vs. grass-fed baseline: >1 means higher GHG than pure grass-fed
merged_ghg['ratio'] = merged_ghg['FCR_scaled_merged'] / land_cr

# Pivot: Year as index, Scenario_ag as columns, ratio as values
merged_ghg = merged_ghg.pivot(index='Year', columns='Scenario_ag', values='ratio')
merged_ghg.to_csv('../3_Results/Feedlots_ghg_ratio_from_ag2050.csv')
