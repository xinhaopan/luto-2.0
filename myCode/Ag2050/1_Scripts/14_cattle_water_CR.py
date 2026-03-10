import pandas as pd

# ============================================================
# Water use conversion ratios (ML / tonne liveweight, cradle-to-gate)
# Source: Wiedemann et al. (2017) Animal Production Science 57:1149-1162
# System boundary: breeding + backgrounding + feedlot finishing
# Water scope: drinking water + feed irrigation + supply losses (full LCA)
# ============================================================

# Grass-fed baseline (L/kg LW → ML/tonne LW)
# Back-calculated from Wiedemann et al. (2017) Discussion:
#   grain-finished was 49–51% HIGHER than grass-fed (Wiedemann et al. 2016a)
#   Using short/mid average feedlot values:
#     short: 296 / 1.50 ≈ 197 L/kg LW
#     mid:   308 / 1.50 ≈ 205 L/kg LW
#   → average ≈ 200 L/kg LW
land_cr  = 200 / 1000   # ML / tonne LW  (grass-fed baseline)

# Feedlot systems — cradle-to-gate fresh water consumption (L/kg LW → ML/tonne LW)
# Table 9 + Results section: total fresh water including irrigation, supply losses
short_cr = 296 / 1000   # ML / tonne LW  short-fed domestic  (55–80  days on feed)
mid_cr   = 308 / 1000   # ML / tonne LW  mid-fed export      (108–164 days on feed)
long_cr  = 206 / 1000   # ML / tonne LW  long-fed export     (>300    days on feed)

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

# map water intensity by stage (ML / tonne LW)
df["cr_ML_per_tonne"] = df["stage"].map(cr_map)

# optional: check unmapped stages
unmapped = df.loc[df["cr_ML_per_tonne"].isna(), "stage"].unique()
if len(unmapped) > 0:
    raise ValueError(f"Unmapped stage values: {unmapped}")

# row-level water use: production (tonne LW) × CR (ML/tonne LW) → ML
df["water_ML"] = df["production"] * df["cr_ML_per_tonne"]

# ---- aggregate: total water use per Year & Scenario ----
water_year_scen = (
    df.groupby(["Scenario_ag", "Year"], as_index=False)["water_ML"]
      .sum()
      .rename(columns={"water_ML": "water_total_ML"})
)

group_cols = ['Scenario_ag', 'Year']
total_production = (
    prod_df.groupby(group_cols, as_index=False)
           .agg(total_production_tonnes=('production', 'sum'))
)
merged_water = water_year_scen.merge(total_production, on=group_cols, how='left')

# Weighted average water intensity (ML / tonne LW)
merged_water['FCR_scaled_merged'] = (
    merged_water["water_total_ML"] / merged_water['total_production_tonnes']
)

# Ratio vs. grass-fed baseline: >1 means higher water use than pure grass-fed
merged_water['ratio'] = merged_water['FCR_scaled_merged'] / land_cr

# Pivot: Year as index, Scenario_ag as columns, ratio as values
merged_water = merged_water.pivot(index='Year', columns='Scenario_ag', values='ratio')
merged_water.to_csv('../3_Results/Feedlots_water_ratio_from_ag2050.csv')
