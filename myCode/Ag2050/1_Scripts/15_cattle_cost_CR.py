import pandas as pd
import numpy as np
import os

# ============================================================
# Cost conversion ratios (AUD / kg carcass weight = kg MEAT)
# All values expressed per kg MEAT (= LUTO's natural production unit, via Q1×F1)
#
# land_cr : derived from LUTO input data (QC+AC+FLC+FOC+FDC+drinking water)
#           Unit: cost_per_ha / (yield_pot × F1 × Q1 × 1000) = AUD/kg MEAT
#           No external dressing percentage needed — Q1 is already carcass weight.
#
# feedlot_cr: operational cost excl. feeder cattle purchase, expressed per kg MEAT
#             Source: MLA Agri-benchmark (2019), Australia-27K feedlot
#               Total long-run cost ~2.80 AUD/kg LW; feeder cattle purchase ~70%
#               Operational cost (feed+labour+depreciation+water): ~30% = 0.84 AUD/kg LW
#             Converted to AUD/kg MEAT using feedlot dressing % = 0.56
#               (grain-fed Australian feedlot; MLA 2022 Feedlot performance monitor)
#               0.84 / 0.56 = 1.50 AUD/kg MEAT
#
# Ratio uses production in tonnes LW for weighting; since all stages have similar
# dressing % (~0.54–0.56), the DP factor cancels in the weighted average ratio.
# ============================================================

input_dir = '../../../input'

# ---- Load LUTO input data ----
AGEC_LVSTK           = pd.read_hdf(os.path.join(input_dir, 'agec_lvstk.h5'))
FEED_REQ             = pd.read_hdf(os.path.join(input_dir, 'feed_req.h5')).to_numpy()
PASTURE_KG_DM_HA     = pd.read_hdf(os.path.join(input_dir, 'pasture_kg_dm_ha.h5')).to_numpy()
SAFE_PUR_NATL        = pd.read_hdf(os.path.join(input_dir, 'safe_pur_natl.h5')).to_numpy()
SAFE_PUR_MODL        = pd.read_hdf(os.path.join(input_dir, 'safe_pur_modl.h5')).to_numpy()
WATER_DELIVERY_PRICE = np.nan_to_num(pd.read_hdf(os.path.join(input_dir, 'water_delivery_price.h5')).to_numpy())

# ---- yield_pot (head/ha): same formula as quantity.py get_yield_pot() ----
# BEEF: dse_per_head=8, grassfed_factor=0.85
denominator    = 365 * 8 * 0.85
yield_pot_natl = FEED_REQ * PASTURE_KG_DM_HA / denominator * SAFE_PUR_NATL  # head/ha
yield_pot_modl = FEED_REQ * PASTURE_KG_DM_HA / denominator * SAFE_PUR_MODL  # head/ha

# ---- Cost components from AGEC_LVSTK ----
# QC  ($/head)  : variable cost per head (vet, supplements, livestock materials)
# AC  ($/ha)    : area-dependent variable cost (pasture maintenance, fertiliser)
# FLC ($/ha)    : fixed labour cost
# FOC ($/ha)    : fixed operating cost (fuel, repairs, administration)
# FDC ($/ha)    : fixed depreciation cost
# WR_DRN (ML/head): drinking water requirement per head
QC     = AGEC_LVSTK[('QC',     'BEEF')].to_numpy()
AC     = AGEC_LVSTK[('AC',     'BEEF')].to_numpy()
FLC    = AGEC_LVSTK[('FLC',    'BEEF')].to_numpy()
FOC    = AGEC_LVSTK[('FOC',    'BEEF')].to_numpy()
FDC    = AGEC_LVSTK[('FDC',    'BEEF')].to_numpy()
WR_DRN = AGEC_LVSTK[('WR_DRN', 'BEEF')].to_numpy()

F1 = AGEC_LVSTK[('F1', 'BEEF')].to_numpy()   # fraction of herd producing meat
Q1 = AGEC_LVSTK[('Q1', 'BEEF')].to_numpy()   # carcass weight per head (t/head)

# ---- land_cr: LUTO grass-fed cost per kg MEAT (cell-level, then averaged) ----
# Uses LUTO's natural production unit: yield_pot × F1 × Q1 = tonnes MEAT/ha
# No external dressing percentage required.
def cost_per_kg_meat(yield_pot):
    """AUD/kg MEAT — dryland grass-fed, same components as LUTO cost calculation"""
    costs_w      = WR_DRN * yield_pot * WATER_DELIVERY_PRICE          # $/ha (drinking water)
    cost_per_ha  = QC * yield_pot + AC + FLC + FOC + FDC + costs_w   # $/ha
    prod_meat_kg = yield_pot * F1 * Q1 * 1000                         # kg MEAT/ha (LUTO native)
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(prod_meat_kg > 0, cost_per_ha / prod_meat_kg, np.nan)

cost_natl = cost_per_kg_meat(yield_pot_natl)
cost_modl = cost_per_kg_meat(yield_pot_modl)
land_cr   = np.nanmean(np.concatenate([cost_natl, cost_modl]))  # AUD/kg MEAT  ≈ 0.96 AUD/kg MEAT

print(f"LUTO grass-fed cost (QC+AC+FLC+FOC+FDC+water, excl. irrigation): {land_cr:.4f} AUD/kg MEAT")
print(f"  Natural land:  {np.nanmean(cost_natl):.4f} AUD/kg MEAT")
print(f"  Modified land: {np.nanmean(cost_modl):.4f} AUD/kg MEAT")

# ============================================================
# Feedlot operational cost (AUD/kg MEAT), excl. feeder cattle purchase
# Source: MLA Agri-benchmark (2019)
#   Total ~2.80 AUD/kg LW; feeder cattle purchase ~70% → operational ~30% = 0.84 AUD/kg LW
#   Feedlot dressing %: 0.56 (grain-fed, MLA 2022 Feedlot performance monitor)
#   → 0.84 / 0.56 = 1.50 AUD/kg MEAT
# Components included: grain feed + labour + operating + depreciation + water
# Applied uniformly to short/mid/long (no per-type breakdown available)
# ============================================================
FEEDLOT_DP = 0.56   # grain-fed feedlot dressing % (MLA 2022)
short_cr = 0.84 / FEEDLOT_DP   # AUD/kg MEAT
mid_cr   = 0.84 / FEEDLOT_DP
long_cr  = 0.84 / FEEDLOT_DP

cr_map = {
    "land":  land_cr,
    "short": short_cr,
    "mid":   mid_cr,
    "long":  long_cr,
}

# ============================================================
# Load cattle production by stage and compute ratio
# ============================================================
prod_df = pd.read_csv('../2_processed_data/cattle_production_by_stage.csv')
df = prod_df.copy()

df["Year"]       = df["Year"].astype(int)
df["production"] = pd.to_numeric(df["production"], errors="coerce")  # tonnes LW

# map cost intensity by stage (AUD/kg MEAT)
df["cr_AUD_per_kg"] = df["stage"].map(cr_map)

unmapped = df.loc[df["cr_AUD_per_kg"].isna(), "stage"].unique()
if len(unmapped) > 0:
    raise ValueError(f"Unmapped stage values: {unmapped}")

# row-level cost: production (t LW) × 1000 (kg/t) × CR (AUD/kg MEAT) → AUD
# Note: all stages have similar dressing % (~0.54–0.56), so the DP factor
# cancels in the weighted ratio and production in LW units can be used directly.
df["cost_AUD"] = df["production"] * 1000 * df["cr_AUD_per_kg"]

# ---- aggregate per Year & Scenario ----
cost_year_scen = (
    df.groupby(["Scenario_ag", "Year"], as_index=False)["cost_AUD"]
      .sum()
      .rename(columns={"cost_AUD": "cost_total_AUD"})
)

group_cols = ['Scenario_ag', 'Year']
total_production = (
    prod_df.groupby(group_cols, as_index=False)
           .agg(total_production_tonnes=('production', 'sum'))
)
merged_cost = cost_year_scen.merge(total_production, on=group_cols, how='left')

# Weighted average cost intensity (AUD/kg MEAT, using LW production for weighting)
merged_cost['cost_avg_AUD_per_kg'] = (
    merged_cost["cost_total_AUD"] / (merged_cost['total_production_tonnes'] * 1000)
)

# Ratio vs. grass-fed baseline (both in AUD/kg MEAT → dimensionless)
# >1: feedlot operational cost is higher than grass-fed per kg MEAT
merged_cost['ratio'] = merged_cost['cost_avg_AUD_per_kg'] / land_cr

# Pivot: Year as index, Scenario_ag as columns
merged_cost = merged_cost.pivot(index='Year', columns='Scenario_ag', values='ratio')
merged_cost.to_csv('../3_Results/Feedlots_cost_ratio_from_ag2050.csv')

print(f"\nRatio range: {merged_cost.values.min():.4f} – {merged_cost.values.max():.4f}")
print("Saved to ../3_Results/Feedlots_cost_ratio_from_ag2050.csv")
