import pandas as pd
import numpy as np
import os

# ============================================================
# Revenue conversion ratios (AUD / kg MEAT, = LUTO's natural production unit)
#
# land_cr : derived from LUTO input data
#           = (F1×Q1×P1 + F3×Q3×P3) / (F1×Q1×1000)  AUD/kg MEAT
#           Includes both meat revenue and live export revenue (same as revenue.py)
#
# feedlot_cr : absolute grain-fed beef prices (AUD/kg MEAT)
#   Source: Beef Central market reports + MLA boneless meat yield (74% of cwt)
#     short-fed (~90 days,  domestic/SE Asia): 13.2–13.3 AUD/kg MEAT  (use 13.2)
#     mid-fed   (~150 days, Japan/Korea):      ~16.1 AUD/kg MEAT
#     long-fed  (~300+ days, Wagyu-spec):      ~23.4 AUD/kg MEAT
#   Feedlot cattle sold as boxed beef (F3≈0, no live export component).
# ============================================================

input_dir = '../../../input'

# ---- Load LUTO input data ----
AGEC_LVSTK = pd.read_hdf(os.path.join(input_dir, 'agec_lvstk.h5'))

F1 = AGEC_LVSTK[('F1', 'BEEF')].to_numpy()   # fraction of herd producing meat
Q1 = AGEC_LVSTK[('Q1', 'BEEF')].to_numpy()   # carcass weight per head (t MEAT/head)
P1 = AGEC_LVSTK[('P1', 'BEEF')].to_numpy()   # meat price (AUD/tonne MEAT)

F3 = AGEC_LVSTK[('F3', 'BEEF')].to_numpy()   # fraction of herd going to live export
Q3 = AGEC_LVSTK[('Q3', 'BEEF')].to_numpy()   # live weight per head (t LW/head)
P3 = AGEC_LVSTK[('P3', 'BEEF')].to_numpy()   # live export price (AUD/tonne LW)

# ---- land_cr: LUTO grass-fed revenue per kg MEAT ----
# Mirrors revenue.py: rev_per_ha = yield_pot × (F1×Q1×P1 + F3×Q3×P3)
# yield_pot cancels → independent of stocking density
rev_per_head   = F1 * Q1 * P1 + F3 * Q3 * P3   # AUD/head
prod_meat_per_head = F1 * Q1 * 1000              # kg MEAT/head

with np.errstate(divide='ignore', invalid='ignore'):
    cr_cell = np.where(prod_meat_per_head > 0,
                       rev_per_head / prod_meat_per_head,
                       np.nan)

land_cr = np.nanmean(cr_cell)   # AUD/kg MEAT
P1_mean  = np.nanmean(P1)       # mean grass-fed meat price (AUD/tonne MEAT)

print(f"LUTO grass-fed revenue: {land_cr:.4f} AUD/kg MEAT")
print(f"  Mean P1 (meat price): {P1_mean:.2f} AUD/tonne MEAT  (= {P1_mean/1000:.4f} AUD/kg MEAT)")

# ============================================================
# Feedlot revenue per kg MEAT (absolute grain-fed prices)
# Source: Beef Central market reports + MLA boneless meat yield (68.4% of carcase weight)
#   MLA boneless meat yield source:
#     https://www.mla.com.au/contentassets/92b275844c0340a48f98646e2b9b8e6d/b.cch.2072_beef_final_report.pdf
##  land: 10.3171 AUD/kg MEAT (LUTO grass-fed revenue
#   short-fed (~90 days,  domestic/SE Asia):
#     100-day HGP-free export-weight: 900–910 c/kg cwt (Beef Central, 10 Feb 2022)
#     https://www.beefcentral.com/news/supermarket-grainfed-contract-cattle-break-through-magic-1000c-kg-milestone/
#     9.00 / 0.684 = 13.16, 9.10 / 0.684 = 13.30 → range: 13.2–13.3 AUD/kg MEAT
#
#   mid-fed   (~150 days, Japan/Korea):
#     150-day Angus grainfed program: 1100 c/kg cwt (Beef Central, 16 Nov 2022)
#     https://www.beefcentral.com/news/killara-feedlot-manager-delivers-sobering-assessment-about-feeder-prices-heading-into-2023/
#     11.00 / 0.684 = 16.08 → range: ~16.1 AUD/kg MEAT
#
#   long-fed  (~300+ days, Wagyu-spec):
#     Wagyu F1 carcase: A$16/kg cwt (Beef Central, 13 Mar 2025)
#     https://www.beefcentral.com/news/wagyu-feeder-steer-prices-stable-but-premium-over-angus-under-pressure/
#     16.00 / 0.684 = 23.39 → range: ~23.4 AUD/kg MEAT
# ============================================================
MEAT_YIELD = 0.684   # boneless meat yield (% of carcase weight); source: MLA

short_cr = 9.05 / MEAT_YIELD   # AUD/kg MEAT  (90-day,  midpoint 900–910 c/kg cwt; range: 13.2–13.3)
mid_cr   = 11.00 / MEAT_YIELD  # AUD/kg MEAT  (150-day, 1100 c/kg cwt;             range: ~16.1)
long_cr  = 16.00 / MEAT_YIELD  # AUD/kg MEAT  (300-day, Wagyu F1 A$16/kg cwt;      range: ~23.4)

cr_map = {
    "land":  land_cr,
    "short": short_cr,
    "mid":   mid_cr,
    "long":  long_cr,
}

print(f"\nRevenue CR (AUD/kg MEAT):")
for k, v in cr_map.items():
    print(f"  {k}: {v:.4f}")

# ============================================================
# Load cattle production by stage and compute ratio
# ============================================================
prod_df = pd.read_csv('../2_processed_data/cattle_production_by_stage.csv')
df = prod_df.copy()

df["Year"]       = df["Year"].astype(int)
df["production"] = pd.to_numeric(df["production"], errors="coerce")  # tonnes LW

# map revenue intensity by stage (AUD/kg MEAT)
df["cr_AUD_per_kg"] = df["stage"].map(cr_map)

unmapped = df.loc[df["cr_AUD_per_kg"].isna(), "stage"].unique()
if len(unmapped) > 0:
    raise ValueError(f"Unmapped stage values: {unmapped}")

# row-level revenue: production (t LW) × 1000 × CR (AUD/kg MEAT) → AUD
# DP factors cancel in ratio since all stages have similar dressing % (~0.54–0.56)
df["rev_AUD"] = df["production"] * 1000 * df["cr_AUD_per_kg"]

# ---- aggregate per Year & Scenario ----
rev_year_scen = (
    df.groupby(["Scenario_ag", "Year"], as_index=False)["rev_AUD"]
      .sum()
      .rename(columns={"rev_AUD": "rev_total_AUD"})
)

group_cols = ['Scenario_ag', 'Year']
total_production = (
    prod_df.groupby(group_cols, as_index=False)
           .agg(total_production_tonnes=('production', 'sum'))
)
merged_rev = rev_year_scen.merge(total_production, on=group_cols, how='left')

# Weighted average revenue intensity (AUD/kg MEAT, using LW production for weighting)
merged_rev['rev_avg_AUD_per_kg'] = (
    merged_rev["rev_total_AUD"] / (merged_rev['total_production_tonnes'] * 1000)
)

# Ratio vs. grass-fed baseline (both AUD/kg MEAT → dimensionless)
# >1: mixed system earns more revenue per kg than pure grass-fed
merged_rev['ratio'] = merged_rev['rev_avg_AUD_per_kg'] / land_cr

# Pivot: Year as index, Scenario_ag as columns
merged_rev = merged_rev.pivot(index='Year', columns='Scenario_ag', values='ratio')
merged_rev.to_csv('../3_Results/Feedlots_revenue_ratio_from_ag2050.csv')

print(f"\nRatio range: {merged_rev.values.min():.4f} – {merged_rev.values.max():.4f}")
print("Saved to ../3_Results/Feedlots_revenue_ratio_from_ag2050.csv")
