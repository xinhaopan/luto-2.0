# Copyright 2022 Fjalar J. de Haan and Brett A. Bryan at Deakin University
#
# This file is part of LUTO 2.0.
#
# LUTO 2.0 is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# LUTO 2.0 is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with
# LUTO 2.0. If not, see <https://www.gnu.org/licenses/>.

""" LUTO model settings. """

import pandas as pd


# ---------------------------------------------------------------------------- #
# LUTO model version.                                                                 #
# ---------------------------------------------------------------------------- #

VERSION = '2.3'


# ---------------------------------------------------------------------------- #
# Spyder options                                                            #
# ---------------------------------------------------------------------------- #

pd.set_option('display.width', 470)
pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 100)
pd.set_option('display.float_format', '{:,.4f}'.format)


# ---------------------------------------------------------------------------- #
# Directories.                                                                 #
# ---------------------------------------------------------------------------- #

INPUT_DIR = 'input'
OUTPUT_DIR = 'output'
DATA_DIR = 'input'
RAW_DATA = '../raw_data'


# ---------------------------------------------------------------------------- #
# Scenario parameters.                                                                  #
# ---------------------------------------------------------------------------- #

# Climate change assumptions. Options include '126', '245', '360', '585'
SSP = '245'
RCP = 'rcp' + SSP[1] + 'p' + SSP[2] # Representative Concentration Pathway string identifier e.g., 'rcp4p5'.

# Set demand parameters which define requirements for Australian production of agricultural commodities
SCENARIO = SSP_NUM = 'SSP' + SSP[0] # SSP1, SSP2, SSP3, SSP4, SSP5
DIET_DOM = 'BAU'                    # 'BAU', 'FLX', 'VEG', 'VGN' - domestic diets in Australia
DIET_GLOB = 'BAU'                   # 'BAU', 'FLX', 'VEG', 'VGN' - global diets
CONVERGENCE = 2050                  # 2050 or 2100 - date at which dietary transformation is completed (velocity of transformation)
IMPORT_TREND = 'Static'             # 'Static' (assumes 2010 shares of imports for each commodity) or 'Trend' (follows historical rate of change in shares of imports for each commodity)
WASTE = 1                           # 1 for full waste, 0.5 for half waste
FEED_EFFICIENCY = 'BAU'             # 'BAU' or 'High'

# Add CO2 fertilisation effects on agricultural production from GAEZ v4
CO2_FERT = 'on'   # or 'off'

# Fire impacts on carbon sequestration
RISK_OF_REVERSAL = 0.05  # Risk of reversal buffer under ERF (reasonable values range from 0.05 [100 years] to 0.25 [25 years]) https://www.cleanenergyregulator.gov.au/ERF/Choosing-a-project-type/Opportunities-for-the-land-sector/Risk-of-reversal-buffer
FIRE_RISK = 'med'   # Options are 'low', 'med', 'high'. Determines whether to take the 5th, 50th, or 95th percentile of modelled fire impacts.

""" Mean FIRE_RISK cell values (%)
    FD_RISK_PERC_5TH    80.3967
    FD_RISK_MEDIAN      89.2485
    FD_RISK_PERC_95TH   93.2735 """


# ---------------------------------------------------------------------------- #
# Economic parameters
# ---------------------------------------------------------------------------- #

# Amortise upfront (i.e., establishment and transitions) costs
AMORTISE_UPFRONT_COSTS = False

# Discount rate for amortisation
DISCOUNT_RATE = 0.07     # 0.05 = 5% pa.

# Set amortisation period
AMORTISATION_PERIOD = 30 # years



# ---------------------------------------------------------------------------- #
# Model parameters
# ---------------------------------------------------------------------------- #

# Optionally coarse-grain spatial domain (faster runs useful for testing). E.g. RESFACTOR 5 selects the middle cell in every 5 x 5 cell block
RESFACTOR = 5        # set to 1 to run at full spatial resolution, > 1 to run at reduced resolution.

# How does the model run over time
# MODE = 'snapshot'   # Runs for target year only
MODE = 'timeseries'   # Runs each year from base year to target year

# Define the objective function
OBJECTIVE = 'maxprofit'   # maximise profit (revenue - costs)  **** Requires soft demand constraints otherwise agriculture over-produces
# OBJECTIVE = 'mincost'  # minimise cost (transitions costs + annual production costs)

# Specify how demand for agricultural commodity production should be met in the solver
# DEMAND_CONSTRAINT_TYPE = 'hard'  # Adds demand as a constraint in the solver (linear programming approach)
DEMAND_CONSTRAINT_TYPE = 'soft'  # Adds demand as a type of slack variable in the solver (goal programming approach)

# Penalty in objective function to balance influence of demand versus cost when DEMAND_CONSTRAINT_TYPE = 'soft'
# 1e5 works well (i.e., demand are met), demands not met with anything less (i.e., large deviations)
# Don't set too high though otherwise it meets demand exactly (minimises deviations) even if the cost is ridiculously high
PENALTY = 1e5

# ---------------------------------------------------------------------------- #
# Geographical raster writing parameters
# ---------------------------------------------------------------------------- #

WRITE_OUTPUT_GEOTIFFS = True    # Write GeoTiffs to output directory: True or False
WRITE_FULL_RES_MAPS = False     # Write GeoTiffs at full or resfactored resolution: True or False
PARALLEL_WRITE = True           # If to use parallel processing to write GeoTiffs: True or False
WRITE_THREADS = 50              # The Threads to use for map making, only work with PARALLEL_WRITE = True

# ---------------------------------------------------------------------------- #
# Gurobi parameters
# ---------------------------------------------------------------------------- #

# Select Gurobi algorithm used to solve continuous models or the initial root relaxation of a MIP model. Default is automatic.
SOLVE_METHOD = 2  # 'automatic: -1, primal simplex: 0, dual simplex: 0, barrier: 2, concurrent: 3, deterministic concurrent: 4, deterministic concurrent simplex: 5

# Presolve parameters (switching both to 0 solves numerical problems)
PRESOLVE = 0     # automatic (-1), off (0), conservative (1), or aggressive (2)
AGGREGATE = 0    # Controls the aggregation level in presolve. The options are off (0), moderate (1), or aggressive (2). In rare instances, aggregation can lead to an accumulation of numerical errors. Turning it off can sometimes improve solution accuracy (it did not fix sub-optimal termination issue)

# Print detailed output to screen
VERBOSE = 1

# Relax the tolerances for feasibility and optimality
FEASIBILITY_TOLERANCE = 1e-2              # Primal feasility tolerance - Default: 1e-6, Min: 1e-9, Max: 1e-2
OPTIMALITY_TOLERANCE = 1e-2               # Dual feasility tolerance - Default: 1e-6, Min: 1e-9, Max: 1e-2
BARRIER_CONVERGENCE_TOLERANCE = 1e-5      # Range from 1e-2 to 1e-8 (default), that larger the number the faster but the less exact the solve. 1e-5 is a good compromise between optimality and speed.

# Whether to use crossover in barrier solve. 0 = off, -1 = automatic. Auto cleans up sub-optimal termination errors without much additional compute time (apart from 2050 when it sometimes never finishes).
CROSSOVER = 0

# Parameters for dealing with numerical issues. NUMERIC_FOCUS = 2 fixes most things but roughly doubles solve time.
SCALE_FLAG = -1     # Scales the rows and columns of the model to improve the numerical properties of the constraint matrix. -1: Auto, 0: No scaling, 1: equilibrium scaling (First scale each row to make its largest nonzero entry to be magnitude one, then scale each column to max-norm 1), 2: geometric scaling, 3: multi-pass equilibrium scaling. Testing revealed that 1 tripled solve time, 3 led to numerical problems.
NUMERIC_FOCUS = 0   # Controls the degree to which the code attempts to detect and manage numerical issues. Default (0) makes an automatic choice, with a slight preference for speed. Settings 1-3 increasingly shift the focus towards being more careful in numerical computations. NUMERIC_FOCUS = 1 is ok, but 2 increases solve time by ~4x
BARHOMOGENOUS = 1  # Useful for recognizing infeasibility or unboundedness. At the default setting (-1), it is only used when barrier solves a node relaxation for a MIP model. 0 = off, 1 = on. It is a bit slower than the default algorithm (3x slower in testing).

# Number of threads to use in parallel algorithms (e.g., barrier)
THREADS = 50


# ---------------------------------------------------------------------------- #
# Non-agricultural land usage parameters
# ---------------------------------------------------------------------------- #

"""
The dictionary below is the master list of all of the non agricultural land uses
and whether they are currently enabled in the solver (True/False).

To disable a non-agricultural land use, change the correpsonding value of the
NON_AG_LAND_USES dictionary to false.
"""
NON_AG_LAND_USES = {
    'Environmental Plantings': True,
    'Riparian Plantings': True,
    'Sheep Agroforestry': True,
    'Beef Agroforestry': True,
    'Carbon Plantings (Block)': True,
    'Sheep Carbon Plantings (Belt)': True,
    'Beef Carbon Plantings (Belt)': True,
    'BECCS': True,
}

"""
If settings.MODE == 'timeseries', the values of the below dictionary determine whether the model is allowed to abandon non-agr.
land uses on cells in the years after it chooses to utilise them. For example, if a cell has is using 'Environmental Plantings'
and the corresponding value in this dictionary is False, all cells using EP must also utilise this land use in all subsequent
years.

CAUTION: Setting reversibility == True can cause infeasibility issues in timeseries runs due to not being able to meet the water constraints.
With the net water yield limit set to say 80%, some catchments could be close to that yield then they experience some land use change to meet
GHG and biodiversity targets. This pushes the catchment close to the net yield constraint. Over time, climate change may reduce the amount of
water yield and if non-ag land uses are not reversible then a catchment may not be able to meet the net yield constraint.
This is expected behaviour and the user must choose how to deal with it.
"""
NON_AG_LAND_USES_REVERSIBLE = {
    'Environmental Plantings': False,
    'Riparian Plantings': False,
    'Sheep Agroforestry': False,
    'Beef Agroforestry': False,
    'Carbon Plantings (Block)': False,
    'Sheep Carbon Plantings (Belt)': False,
    'Beef Carbon Plantings (Belt)': False,
    'BECCS': False,
}

# Carbon price scenario: either '1.8C 67%', '1.5C 50%', '1.5C 67%', 'Default', '100', or None.
# Setting to None falls back to the 'Default' scenario.
CARBON_PRICES_FIELD = '1.8C 67%'

# Cost of fencing per linear metre
FENCING_COST_PER_M = 10

# Environmental Plantings Parameters
EP_ANNUAL_MAINTENANCE_COST_PER_HA_PER_YEAR = 100
EP_ANNUAL_ECOSYSTEM_SERVICES_BENEFIT_PER_HA_PER_YEAR = 0

# Carbon Plantings Block Parameters
CP_BLOCK_ANNUAL_MAINTENNANCE_COST_PER_HA_PER_YEAR = 100
CP_BLOCK_ANNUAL_ECOSYSTEM_SERVICES_BENEFIT_PER_HA_PER_YEAR = 0


# Carbon Plantings Belt Parameters
CP_BELT_ANNUAL_MAINTENNANCE_COST_PER_HA_PER_YEAR = 100
CP_BELT_ANNUAL_ECOSYSTEM_SERVICES_BENEFIT_PER_HA_PER_YEAR = 0

CP_BELT_ROW_WIDTH = 20
CP_BELT_ROW_SPACING = 40
CP_BELT_PROPORTION = CP_BELT_ROW_WIDTH / (CP_BELT_ROW_WIDTH + CP_BELT_ROW_SPACING)
cp_no_alleys_per_ha = 100 / (CP_BELT_ROW_WIDTH + CP_BELT_ROW_SPACING)
CP_BELT_FENCING_LENGTH = 100 * cp_no_alleys_per_ha * 2     # Length (average) of fencing required per ha in metres

# Riparian Planting Parameters
RP_ANNUAL_MAINTENNANCE_COST_PER_HA_PER_YEAR = 100
RP_ANNUAL_ECOSYSTEM_SERVICES_BENEFIT_PER_HA_PER_YEAR = 0

RIPARIAN_PLANTING_BUFFER_WIDTH = 30
RIPARIAN_PLANTING_TORTUOSITY_FACTOR = 0.5

# Agroforestry Parameters
AF_ANNUAL_MAINTENNANCE_COST_PER_HA_PER_YEAR = 100
AF_ANNUAL_ECOSYSTEM_SERVICES_BENEFIT_PER_HA_PER_YEAR = 0

AGROFORESTRY_ROW_WIDTH = 20
AGROFORESTRY_ROW_SPACING = 40
AF_PROPORTION = AGROFORESTRY_ROW_WIDTH / (AGROFORESTRY_ROW_WIDTH + AGROFORESTRY_ROW_SPACING)
no_belts_per_ha = 100 / (AGROFORESTRY_ROW_WIDTH + AGROFORESTRY_ROW_SPACING)
AF_FENCING_LENGTH = 100 * no_belts_per_ha * 2 # Length of fencing required per ha in metres



# ---------------------------------------------------------------------------- #
# Agricultural management parameters
# ---------------------------------------------------------------------------- #

"""
The dictionary below contains a master list of all agricultural management options and
which land uses they correspond to.

To disable an ag-mangement option, change the corresponding value in the AG_MANAGEMENTS dictionary to False.
"""
AG_MANAGEMENTS = {
    'Asparagopsis taxiformis': True,
    'Precision Agriculture': True,
    'Ecological Grazing': True,
    'Savanna Burning': True,
    'AgTech EI': True,
}

"""
If settings.MODE == 'timeseries', the values of the below dictionary determine whether the model is allowed to abandon agricultural
management options on cells in the years after it chooses to utilise them. For example, if a cell has is using 'Asparagopsis taxiformis',
and the corresponding value in this dictionary is False, all cells using Asparagopsis taxiformis must also utilise this land use
and agricultural management combination in all subsequent years.

WARNING: changing to False will result in 'locking in' land uses on cells that utilise the agricultural management option for
the rest of the simulation. This may be an unintended side effect.
"""
AG_MANAGEMENTS_REVERSIBLE = {
    'Asparagopsis taxiformis': True,
    'Precision Agriculture': True,
    'Ecological Grazing': True,
    'Savanna Burning': True,
    'AgTech EI': True,
}

# The cost for removing and establishing irrigation infrastructure ($ per hectare)
REMOVE_IRRIG_COST = 5000
NEW_IRRIG_COST = 10000

# Savanna burning cost per hectare per year ($/ha/yr)
SAVBURN_COST_HA_YR = 100

# The minimum value an agricultural management variable must take for the write_output function to consider it being used on a cell
AGRICULTURAL_MANAGEMENT_USE_THRESHOLD = 0.1


# ---------------------------------------------------------------------------- #
# Off-land commodity parameters
# ---------------------------------------------------------------------------- #

OFF_LAND_COMMODITIES = ['pork', 'chicken', 'eggs', 'aquaculture']
EGGS_AVG_WEIGHT = 60  # Average weight of an egg in grams


# ---------------------------------------------------------------------------- #
# Environmental parameters
# ---------------------------------------------------------------------------- #

# Greenhouse gas emissions limits and parameters *******************************
GHG_EMISSIONS_LIMITS = 'on'        # 'on' or 'off'

GHG_LIMITS_TYPE = 'file' # 'dict' or 'file'

# Set emissions limits in dictionary below (i.e., year: tonnes)
GHG_LIMITS = {
              2010: 90 * 1e6,    # Agricultural emissions in 2010 in tonnes CO2e
              2050: -100 * 1e6,  # GHG emissions target and year (can add more years/targets)
              2100: -100 * 1e6   # GHG emissions target and year (can add more years/targets)
             }

# Take data from 'GHG_targets.xlsx', options include: 'None', '1.5C (67%)', '1.5C (50%)', or '1.8C (67%)'
GHG_LIMITS_FIELD = '1.5C (67%) excl. avoided emis'

# Number of years over which to spread (average) soil carbon accumulation (from Mosnier et al. 2022 and Johnson et al. 2021)
SOC_AMORTISATION = 15

GHG_CONSTRAINT_TYPE = 'hard'  # Adds GHG limits as a constraint in the solver (linear programming approach)
# GHG_CONSTRAINT_TYPE = 'soft'  # Adds GHG usage as a type of slack variable in the solver (goal programming approach)

# Penalty for deviating from the GHG constraints when GHG_CONSTRAINT_TYPE is soft
GHG_PENALTY = 10e-3

# Water use yield and parameters *******************************
WATER_LIMITS = 'off'     # 'on' or 'off'. 'off' will turn off water net yield limit constraints in the solver.

RELAXED_WATER_LIMITS_FOR_INFEASIBILITY = 'on'



# Regionalisation to enforce water use limits by
WATER_REGION_DEF = 'Drainage Division'         # 'River Region' or 'Drainage Division' Bureau of Meteorology GeoFabric definition

# Water net yield targets: the value represents the proportion of the historical water yields
# that the net yield must exceed in a given year. Base year (2010) uses base year net yields as targets.
# Everything past the latest year specified uses the target figure for the latest year.

# Safe and just Earth system boundaries suggests a water stress of 0.2 (yield of 0.8). This is inclusive of
# domestic/industrial: https://www.nature.com/articles/s41586-023-06083-8, Approximately 70% of the total water use
# is used for agricultural purposes. This includes water used for irrigation, livestock, and domestic purposes on farms,
# with the rest used for domestic/industrial  https://soe.dcceew.gov.au/inland-water/pressures/population
# Hence, assuming that this proportion is uniform over all catchments and remains constant over time then if water
# stress is 0.2 then agriculture can use up 70% of this, leaving 30% for domestic/industrial. The water yield target for ag
# should then be historical net yield * (1 - water stress * agricultural share)

WATER_STRESS = 0.2
AG_SHARE_OF_WATER_USE = 1.0
WATER_YIELD_TARGET_AG_SHARE = 1 - WATER_STRESS * AG_SHARE_OF_WATER_USE

# Set a dictionary of water yield targets (i.e., the proportion of historical net annual water yield). LUTO will ensure that
# net annual water yield is >= this proportion of historical net annual water yield is met by the given date, leaving sufficient water
# for domestic and industrial use. The water yield target grades linearly from net water yield in 2010 to achieve the target by the target date
# for each catchment (river region or drainage division)
WATER_YIELD_TARGETS = {
                        2030: WATER_YIELD_TARGET_AG_SHARE,
                        2100: WATER_YIELD_TARGET_AG_SHARE,
                      }

# Consider livestock drinking water (0 [off] or 1 [on]) ***** Livestock drinking water turned off due to infeasibility issues with water constraint in Pilbara
LIVESTOCK_DRINKING_WATER = 1

# Consider water license costs (0 [off] or 1 [on]) of land-use transition ***** If on then there is a noticeable water sell-off by irrigators in the MDB when maximising profit
INCLUDE_WATER_LICENSE_COSTS = 0



# Biodiversity limits and parameters *******************************


# ------------------- Agricultural biodiversity parameters -------------------

# Connectivity source source
'''
    The connectivity source is the source of the connectivity score used to weigh the raw biodiversity priority score.
    This score is normalised between 0 (fartherst) and 1 (closest).
    Can be either 'NCI' or 'DWI'.
        - if 'NCI' is selected, the connectivity score is sourced from the DCCEEW's National Connectivity Index (v3.0).
        - if 'DWI' is selected, the connectivity score is calculated as distance to the nearest area of natural land as mapped
          by the National Land Use Map of Australia.
        - if 'NONE' is selected, the connectivity score is not used in the biodiversity calculation.
'''
CONNECTIVITY_SOURCE = 'NCI'                 # 'NCI', 'DWI' or 'NONE'


# Connectivity score importance
'''
    !!!!!   ONLY WORKS IF CONNECTIVITY_SOURCE IS NOT 'NONE'   !!!!!
    The relative importance of the connectivity score in the biodiversity calculation.
    I.e., the lower bound of the connectivity score for weighting the raw biodiversity priority score is CONNECTIVITY_LB.
'''
connect_importance = 0.3
CONNECTIVITY_LB = 1 - connect_importance    # Weighting of connectivity score in biodiversity calculation (0 - 1)



# Habitat condition data source
HABITAT_CONDITION = 'HCAS'                  # 'HCAS', 'USER_DEFINED', or 'NONE'
'''
    It is used to calculate the biodiversity benifits for aricultural landuses.
    - If 'HCAS' is selected, the habitat condition is calculated using the Habitat Condition Assessment System (HCAS)
    - If 'USER_DEFINED' is selected, the habitat condition is calculated using the user defined values in the 'HCAS_USER_DEFINED' dictionary.
'''


# HCAS percentile for each land-use type
''' Different land-use types have different biodiversity degradation impacts. We calculated the percentiles values of HCAS (indicating the
    suitability for wild animals ranging between 0-1) for each land-use type.

    Here is the parameter defining the percentile used to represent each land-use's degradation scale to biodiversity. Avaliable percentiles
    is one of [10, 25, 50, 75, 90].

    For example, the 50th percentile for 'Beef - Modified land' is 0.22, meaning this land has 22% biodiversity compared to if it was restored
    to a perfect natural land.
'''
HCAS_PERCENTILE = 50

# Biodiversity value under default late dry season savanna fire regime
''' For example, 0.8 means that all areas in the area eligible for savanna burning have a biodiversity value of 0.8 * the raw biodiv value
    (due to hot fires etc). When EDS sav burning is implemented the area is attributed the full biodiversity value.'''
LDS_BIODIVERSITY_VALUE = 0.8


# ------------------- Non-agricultural biodiversity parameters -------------------
''' The benefit of each non-agricultural land use to biodiversity is set as a proportion to the raw biodiversity priority value.
    For example, if the raw biodiversity priority value is 0.6 and the benefit is 0.8, then the biodiversity value
    will be 0.6 * 0.8 = 0.48.
'''
ENV_PLANTING_BIODIVERSITY_BENEFIT = 0.8
CARBON_PLANTING_BLOCK_BIODIV_BENEFIT = 0.1
CARBON_PLANTING_BELT_BIODIV_BENEFIT = 0.1
RIPARIAN_PLANTING_BIODIV_BENEFIT = 1
AGROFORESTRY_BIODIV_BENEFIT = 0.75
BECCS_BIODIVERSITY_BENEFIT = 0


# ------------------- Set biodiversity target (0 - 1 e.g., 0.3 = 30% of total achievable Zonation biodiversity benefit)
""" Kunming-Montreal Global Biodiversity Framework Target 2: Restore 30% of all Degraded Ecosystems
    Ensure that by 2030 at least 30 per cent of areas of degraded terrestrial, inland water, and coastal and marine ecosystems are under effective restoration,
    in order to enhance biodiversity and ecosystem functions and services, ecological integrity and connectivity.
"""
BIODIV_GBF_TARGET_2_DICT = {
              2010: 0,    # Proportion of degraded land restored in year 2010
              2030: 0.3,  # Proportion of degraded land restored in year 2030 - GBF Target 2
              2050: 0.3,  # Principle from GBF 2050 Goals and Vision and LeClere et al. Bending the Curve - need to arrest biodiversity decline then begin improving over time.
              2100: 0.3   # Stays at 2050 level
             }            # (can add more years/targets)\


# ------------------- Biodiversity contribution reporting -------------------
BIODIVERSITY_LIMITS = 'on'            # 'on' or 'off', if 'off' the biodiversity target will be set as zero.
BIODIVERSITY_CONTRIBUTION_REPORT = True  # True or False, report biodiversity contribution
BIO_CALC_LEVEL = 'group'  # 'group' or 'species' - determines whether to calculate biodiversity scores at the group or species level



# ---------------------------------------------------------------------------- #
# Other parameters
# ---------------------------------------------------------------------------- #

# Cell culling
CULL_MODE = 'absolute'      # cull to include at most MAX_LAND_USES_PER_CELL
# CULL_MODE = 'percentage'    # cull the LAND_USAGE_THRESHOLD_PERCENTAGE % most expensive options
# CULL_MODE = 'none'          # do no culling

MAX_LAND_USES_PER_CELL = 12
LAND_USAGE_CULL_PERCENTAGE = 0.15

# Non-ag output coding. Non-agricultural land uses will appear on the land use map offset by this amount (e.g. land use 0 will appear as 100)
NON_AGRICULTURAL_LU_BASE_CODE = 100

# Number of decimals to round the lower bound matrices to for non-agricultural land uses and agricultural management options.
LB_ROUND_DECMIALS = 6


""" NON-AGRICULTURAL LAND USES (indexed by k)
0: 'Environmental Plantings'
1: 'Riparian Plantings'
2: 'Sheep Agroforestry'
3: 'Beef Agroforestry'
4: 'Carbon Plantings (Block)'
5: 'Sheep Carbon Plantings (Belt)'
6: 'Beef Carbon Plantings (Belt)'
7: 'BECCS'


AGRICULTURAL MANAGEMENT OPTIONS (indexed by a)
0: (None)
1: 'Asparagopsis taxiformis'
2: 'Precision Agriculture'
3: 'Ecological Grazing'


DRAINAGE DIVISIONS
 1: 'Tanami-Timor Sea Coast',
 2: 'South Western Plateau',
 3: 'South West Coast',
 4: 'Tasmania',
 5: 'South East Coast (Victoria)',
 6: 'South Australian Gulf',
 7: 'Murray-Darling Basin',
 8: 'Pilbara-Gascoyne',
 9: 'North Western Plateau',
 10: 'South East Coast (NSW)',
 11: 'Carpentaria Coast',
 12: 'Lake Eyre Basin',
 13: 'North East Coast'


RIVER REGIONS
 1: 'ADELAIDE RIVER',
 2: 'ALBANY COAST',
 3: 'ARCHER-WATSON RIVERS',
 4: 'ARTHUR RIVER',
 5: 'ASHBURTON RIVER',
 6: 'AVOCA RIVER',
 7: 'AVON RIVER-TYRELL LAKE',
 8: 'BAFFLE CREEK',
 9: 'BARRON RIVER',
 10: 'BARWON RIVER-LAKE CORANGAMITE',
 11: 'BATHURST-MELVILLE ISLANDS',
 12: 'BEGA RIVER',
 13: 'BELLINGER RIVER',
 14: 'BENANEE-WILLANDRA CREEK',
 15: 'BILLABONG-YANCO CREEKS',
 16: 'BLACK RIVER',
 17: 'BLACKWOOD RIVER',
 18: 'BLYTH RIVER',
 19: 'BORDER RIVERS',
 20: 'BOYNE RIVER',
 21: 'BRISBANE RIVER',
 22: 'BROKEN RIVER',
 23: 'BROUGHTON RIVER',
 24: 'BRUNSWICK RIVER',
 25: 'BUCKINGHAM RIVER',
 26: 'BULLO RIVER-LAKE BANCANNIA',
 27: 'BUNYIP RIVER',
 28: 'BURDEKIN RIVER',
 29: 'BURNETT RIVER',
 30: 'BURRUM RIVER',
 31: 'BUSSELTON COAST',
 32: 'CALLIOPE RIVER',
 33: 'CALVERT RIVER',
 34: 'CAMPASPE RIVER',
 35: 'CAPE LEVEQUE COAST',
 36: 'CARDWELL COAST',
 37: 'CASTLEREAGH RIVER',
 38: 'CLARENCE RIVER',
 39: 'CLYDE RIVER-JERVIS BAY',
 40: 'COAL RIVER',
 41: 'COLLIE-PRESTON RIVERS',
 42: 'CONDAMINE-CULGOA RIVERS',
 43: 'COOPER CREEK',
 44: 'CURTIS ISLAND',
 45: 'DAINTREE RIVER',
 46: 'DALY RIVER',
 47: 'DARLING RIVER',
 48: 'DE GREY RIVER',
 49: 'DENMARK RIVER',
 50: 'DERWENT RIVER',
 51: 'DIAMANTINA-GEORGINA RIVERS',
 52: 'DON RIVER',
 53: 'DONNELLY RIVER',
 54: 'DRYSDALE RIVER',
 55: 'DUCIE RIVER',
 56: 'EAST ALLIGATOR RIVER',
 57: 'EAST COAST',
 58: 'EAST GIPPSLAND',
 59: 'EMBLEY RIVER',
 60: 'ENDEAVOUR RIVER',
 61: 'ESPERANCE COAST',
 62: 'EYRE PENINSULA',
 63: 'FINNISS RIVER',
 64: 'FITZMAURICE RIVER',
 65: 'FITZROY RIVER (QLD)',
 66: 'FITZROY RIVER (WA)',
 67: 'FLEURIEU PENINSULA',
 68: 'FLINDERS-CAPE BARREN ISLANDS',
 69: 'FLINDERS-NORMAN RIVERS',
 70: 'FORTESCUE RIVER',
 71: 'FORTH RIVER',
 72: 'FRANKLAND-DEEP RIVERS',
 73: 'FRASER ISLAND',
 74: 'GAIRDNER',
 75: 'GASCOYNE RIVER',
 76: 'GAWLER RIVER',
 77: 'GLENELG RIVER',
 78: 'GOOMADEER RIVER',
 79: 'GORDON RIVER',
 80: 'GOULBURN RIVER',
 81: 'GOYDER RIVER',
 82: 'GREENOUGH RIVER',
 83: 'GROOTE EYLANDT',
 84: 'GWYDIR RIVER',
 85: 'HASTINGS RIVER',
 86: 'HAUGHTON RIVER',
 87: 'HAWKESBURY RIVER',
 88: 'HERBERT RIVER',
 89: 'HINCHINBROOK ISLAND',
 90: 'HOLROYD RIVER',
 91: 'HOPKINS RIVER',
 92: 'HUNTER RIVER',
 93: 'HUON RIVER',
 94: 'ISDELL RIVER',
 95: 'JARDINE RIVER',
 96: 'JEANNIE RIVER',
 97: 'JOHNSTONE RIVER',
 98: 'KANGAROO ISLAND',
 99: 'KARUAH RIVER',
 100: 'KEEP RIVER',
 101: 'KENT RIVER',
 102: 'KIEWA RIVER',
 103: 'KING EDWARD RIVER',
 104: 'KING ISAND',
 105: 'KING-HENTY RIVERS',
 106: 'KINGSTON COAST',
 107: 'KOLAN RIVER',
 108: 'KOOLATONG RIVER',
 109: 'LACHLAN RIVER',
 110: 'LAKE EYRE',
 111: 'LAKE TORRENS-MAMBRAY COAST',
 112: 'LENNARD RIVER',
 113: 'LIMMEN BIGHT RIVER',
 114: 'LITTLE RIVER',
 115: 'LIVERPOOL RIVER',
 116: 'LOCKHART RIVER',
 117: 'LODDON RIVER',
 118: 'LOGAN-ALBERT RIVERS',
 119: 'LOWER MALLEE',
 120: 'LOWER MURRAY RIVER',
 121: 'MACLEAY RIVER',
 122: 'MACQUARIE-BOGAN RIVERS',
 123: 'MACQUARIE-TUGGERAH LAKES',
 124: 'MANNING RIVER',
 125: 'MAROOCHY RIVER',
 126: 'MARY RIVER (NT)',
 127: 'MARY RIVER (QLD)',
 128: 'MERSEY RIVER',
 129: 'MILLICENT COAST',
 130: 'MITCHELL-COLEMAN RIVERS (QLD)',
 131: 'MITCHELL-THOMSON RIVERS',
 132: 'MOONIE RIVER',
 133: 'MOORE-HILL RIVERS',
 134: 'MORNING INLET',
 135: 'MORNINGTON ISLAND',
 136: 'MORUYA RIVER',
 137: 'MOSSMAN RIVER',
 138: 'MOYLE RIVER',
 139: 'MULGRAVE-RUSSELL RIVERS',
 140: 'MURCHISON RIVER',
 141: 'MURRAY RIVER (WA)',
 142: 'MURRAY RIVERINA',
 143: 'MURRUMBIDGEE RIVER',
 144: 'MYPONGA RIVER',
 145: 'McARTHUR RIVER',
 146: 'NAMOI RIVER',
 147: 'NICHOLSON-LEICHHARDT RIVERS',
 148: 'NOOSA RIVER',
 149: 'NORMANBY RIVER',
 150: 'NULLARBOR',
 151: "O'CONNELL RIVER",
 152: 'OLIVE-PASCOE RIVERS',
 153: 'ONKAPARINGA RIVER',
 154: 'ONSLOW COAST',
 155: 'ORD-PENTECOST RIVERS',
 156: 'OTWAY COAST',
 157: 'OVENS RIVER',
 158: 'PAROO RIVER',
 159: 'PIEMAN RIVER',
 160: 'PINE RIVER',
 161: 'PIONEER RIVER',
 162: 'PIPER-RINGAROOMA RIVERS',
 163: 'PLANE CREEK',
 164: 'PORT HEDLAND COAST',
 165: 'PORTLAND COAST',
 166: 'PRINCE REGENT RIVER',
 167: 'PROSERPINE RIVER',
 168: 'RICHMOND RIVER',
 169: 'ROBINSON RIVER',
 170: 'ROPER RIVER',
 171: 'ROSIE RIVER',
 172: 'ROSS RIVER',
 173: 'RUBICON RIVER',
 174: 'SALT LAKE',
 175: 'SANDY CAPE COAST',
 176: 'SANDY DESERT',
 177: 'SETTLEMENT CREEK',
 178: 'SHANNON RIVER',
 179: 'SHOALHAVEN RIVER',
 180: 'SHOALWATER CREEK',
 181: 'SMITHTON-BURNIE COAST',
 182: 'SNOWY RIVER',
 183: 'SOUTH ALLIGATOR RIVER',
 184: 'SOUTH COAST',
 185: 'SOUTH GIPPSLAND',
 186: 'SOUTH-WEST COAST',
 187: 'SPENCER GULF',
 188: 'STEWART RIVER',
 189: 'STRADBROKE ISLAND',
 190: 'STYX RIVER',
 191: 'SWAN COAST-AVON RIVER',
 192: 'SYDNEY COAST-GEORGES RIVER',
 193: 'TAMAR RIVER',
 194: 'TORRENS RIVER',
 195: 'TORRES STRAIT ISLANDS',
 196: 'TOWAMBA RIVER',
 197: 'TOWNS RIVER',
 198: 'TULLY-MURRAY RIVERS',
 199: 'TUROSS RIVER',
 200: 'TWEED RIVER',
 201: 'UPPER MALLEE',
 202: 'UPPER MURRAY RIVER',
 203: 'VICTORIA RIVER-WISO',
 204: 'WAKEFIELD RIVER',
 205: 'WALKER RIVER',
 206: 'WARD RIVER',
 207: 'WARREGO RIVER',
 208: 'WARREN RIVER',
 209: 'WATER PARK CREEK',
 210: 'WENLOCK RIVER',
 211: 'WERRIBEE RIVER',
 212: 'WHITSUNDAY ISLANDS',
 213: 'WILDMAN RIVER',
 214: 'WIMMERA RIVER',
 215: 'WOLLONGONG COAST',
 216: 'WOORAMEL RIVER',
 217: 'YANNARIE RIVER',
 218: 'YARRA RIVER'}
"""