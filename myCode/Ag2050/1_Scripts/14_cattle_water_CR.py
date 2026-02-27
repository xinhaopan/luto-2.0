import pandas as pd
import numpy as np
import os

input_dir = '../../../input'
AGEC_LVSTK = pd.read_hdf(os.path.join(input_dir, "agec_lvstk.h5"))
prod_ave = AGEC_LVSTK[('Q1', 'BEEF')].replace(0, np.nan).mean()

AGGHG_LVSTK = pd.read_hdf(os.path.join(input_dir, "agGHG_lvstk.h5"))
LVSTK_GHG_SCOPE_1 = ['CO2E_KG_HEAD_DUNG_URINE', 'CO2E_KG_HEAD_ENTERIC', 'CO2E_KG_HEAD_IND_LEACH_RUNOFF', 'CO2E_KG_HEAD_MANURE_MGT']
ghg_raw = AGGHG_LVSTK.loc[:, (AGGHG_LVSTK.columns.get_level_values(0) == 'BEEF') &
                                          (AGGHG_LVSTK.columns.get_level_values(1).isin(LVSTK_GHG_SCOPE_1))]
ghg_ave = ghg_raw.replace(0, np.nan).mean().sum()
ghg_per_kg = ghg_ave/prod_ave

prod_df = pd.read_csv('../2_processed_data/cattle_production_by_stage.csv')
land_cr = ghg_per_kg
short_cr = 27.25 * 672.6 / (468 * 0.6817 * (1-0.008)) / 1e3
mid_cr = 27.25 * 554.6 / (652 * 0.6817 * (1-0.007)) / 1e3
long_cr = 27.25 * 784 / (784 * 0.6817 * (1-0.021)) / 1e3
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

# Pivot the table to have Scenario_ag as columns, Year as index, and ratio as values
merged_ghg = merged_ghg.pivot(index='Year', columns='Scenario_ag', values='ratio')
merged_ghg.to_csv('../3_Results/Feedlots_ghg_ratio_from_ag2050.csv')
