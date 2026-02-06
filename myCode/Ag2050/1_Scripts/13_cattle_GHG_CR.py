import pandas as pd
prod_df = pd.read_csv('../2_processed_data/cattle_production_by_stage.csv')
land_cr = 5.626215057451551
short_cr = 9.9 * 468 / 468 * 0.6817 * (1-0.008)
mid_cr = 9.4 * 652 / 652 * 0.6817 * (1-0.007)
long_cr = 10.6 * 784 / 784 * 0.6817 * (1-0.021)
cr_map = {
    "land": land_cr,
    "short": short_cr,
    "mid": mid_cr,
    "long": long_cr,
}

df = prod_df.copy()

# make sure types are OK
df["Year"] = df["Year"].astype(int)
df["production"] = pd.to_numeric(df["production"], errors="coerce")

# map factor by stage
df["cr_kgco2e_per_kg"] = df["stage"].map(cr_map)

# optional: check unmapped stages
unmapped = df.loc[df["cr_kgco2e_per_kg"].isna(), "stage"].unique()
if len(unmapped) > 0:
    raise ValueError(f"Unmapped stage values: {unmapped}")

# row-level emissions
df["ghg_kgco2e"] = df["production"] * df["cr_kgco2e_per_kg"]

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
merged_ghg['FCR_scaled_merged'] = (
    merged_ghg["ghg_total_kgco2e"] / merged_ghg['total_production_tonnes']
)
merged_ghg['ratio'] = merged_ghg['FCR_scaled_merged'] / land_cr