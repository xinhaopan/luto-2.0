# Copyright 2025 Bryan, B.A., Williams, N., Archibald, C.L., de Haan, F., Wang, J., 
# van Schoten, N., Hadjikakou, M., Sanson, J.,  Zyngier, R., Marcos-Martinez, R.,  
# Navarro, J.,  Gao, L., Aghighi, H., Armstrong, T., Bohl, H., Jaffe, P., Khan, M.S., 
# Moallemi, E.A., Nazari, A., Pan, X., Steyl, D., and Thiruvady, D.R.
#
# This file is part of LUTO2 - Version 2 of the Australian Land-Use Trade-Offs model
#
# LUTO2 is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# LUTO2 is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with
# LUTO2. If not, see <https://www.gnu.org/licenses/>.



""" LUTO model settings. """

import os
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
RAW_DATA = '../raw_data'


# ---------------------------------------------------------------------------- #
# Scenario parameters.                                                                  #
# ---------------------------------------------------------------------------- #

# Climate change assumptions. Options include '126', '245', '360', '585'
SSP = '245'
RCP = 'rcp' + SSP[1] + 'p' + SSP[2] # Representative Concentration Pathway string identifier e.g., 'rcp4p5'.

# Set demand parameters which define requirements for Australian production of agricultural commodities
SCENARIO = 'SSP' + SSP[0] # SSP1, SSP2, SSP3, SSP4, SSP5
DIET_DOM = 'BAU'                    # 'BAU', 'FLX', 'VEG', 'VGN' - domestic diets in Australia
DIET_GLOB = 'BAU'                   # 'BAU', 'FLX', 'VEG', 'VGN' - global diets
CONVERGENCE = 2050                  # 2050 or 2100 - date at which dietary transformation is completed (velocity of transformation)
IMPORT_TREND = 'Static'             # 'Static' (assumes 2010 shares of imports for each commodity) or 'Trend' (follows historical rate of change in shares of imports for each commodity)
WASTE = 1                           # 1 for full waste, 0.5 for half waste
FEED_EFFICIENCY = 'BAU'             # 'BAU' or 'High'

# Add CO2 fertilisation effects on agricultural production from GAEZ v4
CO2_FERT = 'off'   # or 'off'

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
RESFACTOR = 13      # set to 1 to run at full spatial resolution, > 1 to run at reduced resolution.

# The step size for the temporal domain (years)
SIM_YEARS = list(range(2010,2051,5)) # range(2020,2050)


# Define the objective function
OBJECTIVE = 'maxprofit'   # maximise profit (revenue - costs)  **** Requires soft demand constraints otherwise agriculture over-produces
# OBJECTIVE = 'mincost'  # minimise cost (transitions costs + annual production costs)



"""
If any of the targets are set to 'soft':
    Then they will have a deviation from target (normalised to near 1
    by dividing their BASE_YR (2010) sum) in the objective function.

    Here the weights determine the relative importance of each target 
    in the objective function.E.g., if SOLVER_WEIGHT_GHG = 2 and the 
    rest are 1, then reducing GHG deviation from target will be twice 
    as important as the other targets in the objective function.

If the target is set to 'hard' or 'off': 
    Then the deviation from target will be 0 and the weight will not be used.
"""
SOLVER_WEIGHT_DEMAND = 1
SOLVER_WEIGHT_GHG = 1
SOLVER_WEIGHT_WATER = 1

RESCALE_FACTOR = 1e3
'''
All input data before feeding into the solver is rescaled in the range between 0 and this factor.
This is to avoid numerical issues with the solver when dealing with very small/large numbers. 
E.g., the water yield for some cells is 10t but the Biodiversity-score is 1e-7, making the 
the model sensitive to variations in input data. 
'''



# ---------------------------------------------------------------------------- #
# Geographical raster writing parameters
# ---------------------------------------------------------------------------- #
WRITE_OUTPUT_GEOTIFFS = True               # Write GeoTiffs to output directory: True or False
PARALLEL_WRITE = True                       # If to use parallel processing to write GeoTiffs: True or False
WRITE_THREADS = min(32, os.cpu_count())     # The Threads to use for map making, only work with PARALLEL_WRITE = True

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
BARHOMOGENOUS = 1   # Useful for recognizing infeasibility or unboundedness. At the default setting (-1), it is only used when barrier solves a node relaxation for a MIP model. 0 = off, 1 = on. It is a bit slower than the default algorithm (3x slower in testing).

# Number of threads to use in parallel algorithms (e.g., barrier)
THREADS = min(32, os.cpu_count())



# ---------------------------------------------------------------------------- #
# No-Go areas; Regional adoption constraints
# ---------------------------------------------------------------------------- #

EXCLUDE_NO_GO_LU = False
NO_GO_VECTORS = {
    'Winter cereals':           os.path.join(INPUT_DIR, 'no_go_areas', 'no_go_Winter_cereals.shp'),
    'Environmental Plantings':  os.path.join(INPUT_DIR, 'no_go_areas', 'no_go_Enviornmental_Plantings.shp')
}
'''
Land-use and vector file pairs to exclude land-use from being utilised in that area. 
 - The key is the land-use name. 
 - The value is the path to the ESRI shapefile.
'''

REGIONAL_ADOPTION_CONSTRAINTS = 'off'  # 'on' or 'off'
REGIONAL_ADOPTION_ZONE = 'ABARES_AAGIS'   # One of 'ABARES_AAGIS', 'LGA_CODE', 'NRM_CODE', 'IBRA_ID', 'SLA_5DIGIT'
'''
The regional adoption zone is the spatial unit used to enforce regional adoption constraints.
The options are:
  - 'ABARES_AAGIS': Australian Bureau of Agricultural and Resource Economics and Sciences (ABARES) Agricultural and Agribusiness Geographic Information System (AAGIS) regions.
  - 'LGA_CODE': Local Government Area code.
  - 'NRM_CODE': Natural Resource Management code.
  - 'IBRA_ID': Interim Biogeographic Regionalisation of Australia (IBRA) region code.
  - 'SLA_5DIGIT': Statistical Local Area (SLA) 5-digit code.
'''



# ---------------------------------------------------------------------------- #
# Non-agricultural land usage parameters
# ---------------------------------------------------------------------------- #

NON_AG_LAND_USES = {
    'Environmental Plantings': True,
    'Riparian Plantings': True,
    'Sheep Agroforestry': True,
    'Beef Agroforestry': True,
    'Carbon Plantings (Block)': True,
    'Sheep Carbon Plantings (Belt)': True,
    'Beef Carbon Plantings (Belt)': True,
    'BECCS': False,
    'Destocked - natural land': True,
}
"""
The dictionary here is the master list of all of the non agricultural land uses
and whether they are currently enabled in the solver (True/False).

To disable a non-agricultural land use, change the correpsonding value of the
NON_AG_LAND_USES dictionary to false.
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
    'Destocked - natural land': True,
}
"""
The values of the below dictionary determine whether the model is allowed to abandon non-agr.
land uses on cells in the years after it chooses to utilise them. For example, if a cell has is using 'Environmental Plantings'
and the corresponding value in this dictionary is False, all cells using EP must also utilise this land use in all subsequent
years.

CAUTION: Setting reversibility == True can cause infeasibility issues in timeseries runs due to not being able to meet the water constraints.
With the net water yield limit set to say 80%, some catchments could be close to that yield then they experience some land use change to meet
GHG and biodiversity targets. This pushes the catchment close to the net yield constraint. Over time, climate change may reduce the amount of
water yield and if non-ag land uses are not reversible then a catchment may not be able to meet the net yield constraint.
This is expected behaviour and the user must choose how to deal with it.
"""

# Cost of fencing per linear metre
FENCING_COST_PER_M = 2

# Environmental Plantings Parameters
EP_ANNUAL_MAINTENANCE_COST_PER_HA_PER_YEAR = 100
EP_ANNUAL_ECOSYSTEM_SERVICES_BENEFIT_PER_HA_PER_YEAR = 0

# Carbon Plantings Block Parameters
CP_BLOCK_ANNUAL_MAINTENANCE_COST_PER_HA_PER_YEAR = 100
CP_BLOCK_ANNUAL_ECOSYSTEM_SERVICES_BENEFIT_PER_HA_PER_YEAR = 0

# Carbon Plantings Belt Parameters
CP_BELT_ANNUAL_MAINTENANCE_COST_PER_HA_PER_YEAR = 100
CP_BELT_ANNUAL_ECOSYSTEM_SERVICES_BENEFIT_PER_HA_PER_YEAR = 0

CP_BELT_ROW_WIDTH = 20
CP_BELT_ROW_SPACING = 40
CP_BELT_PROPORTION = CP_BELT_ROW_WIDTH / (CP_BELT_ROW_WIDTH + CP_BELT_ROW_SPACING)
cp_no_alleys_per_ha = 100 / (CP_BELT_ROW_WIDTH + CP_BELT_ROW_SPACING)
CP_BELT_FENCING_LENGTH = 100 * cp_no_alleys_per_ha * 2     # Length (average) of fencing required per ha in metres

# Riparian Planting Parameters
RP_ANNUAL_MAINTENANCE_COST_PER_HA_PER_YEAR = 100
RP_ANNUAL_ECOSYSTEM_SERVICES_BENEFIT_PER_HA_PER_YEAR = 0

RIPARIAN_PLANTING_BUFFER_WIDTH = 30
RIPARIAN_PLANTING_TORTUOSITY_FACTOR = 0.5

# Agroforestry Parameters
AF_ANNUAL_MAINTENANCE_COST_PER_HA_PER_YEAR = 100
AF_ANNUAL_ECOSYSTEM_SERVICES_BENEFIT_PER_HA_PER_YEAR = 0

AGROFORESTRY_ROW_WIDTH = 20
AGROFORESTRY_ROW_SPACING = 40
AF_PROPORTION = AGROFORESTRY_ROW_WIDTH / (AGROFORESTRY_ROW_WIDTH + AGROFORESTRY_ROW_SPACING)
no_belts_per_ha = 100 / (AGROFORESTRY_ROW_WIDTH + AGROFORESTRY_ROW_SPACING)
AF_FENCING_LENGTH_HA = 100 * no_belts_per_ha * 2 # Length of fencing required per ha in metres


# ---------------------------------------------------------------------------- #
# Agricultural management parameters
# ---------------------------------------------------------------------------- #


AG_MANAGEMENTS_TO_LAND_USES = {
    'Asparagopsis taxiformis':  ['Beef - modified land', 'Sheep - modified land', 'Dairy - natural land', 'Dairy - modified land'],
    
    'Precision Agriculture':    [# Cropping:
                                'Hay', 'Summer cereals', 'Summer legumes', 'Summer oilseeds', 'Winter cereals', 'Winter legumes', 'Winter oilseeds',
                                # Intensive Cropping:
                                'Cotton', 'Other non-cereal crops', 'Rice', 'Sugar', 'Vegetables',
                                # Horticulture:
                                'Apples', 'Citrus', 'Grapes', 'Nuts', 'Pears', 'Plantation fruit', 'Stone fruit', 'Tropical stone fruit'],
    
    'Ecological Grazing':       ['Beef - modified land', 'Sheep - modified land', 'Dairy - modified land'],
    
    'Savanna Burning': [        'Beef - natural land', 'Dairy - natural land', 'Sheep - natural land', 'Unallocated - natural land'],
    
    'AgTech EI': [              # Cropping:
                                'Hay', 'Summer cereals', 'Summer legumes', 'Summer oilseeds', 'Winter cereals', 'Winter legumes', 'Winter oilseeds',
                                # Intensive Cropping:
                                'Cotton', 'Other non-cereal crops', 'Rice', 'Sugar', 'Vegetables',
                                # Horticulture:
                                'Apples', 'Citrus', 'Grapes', 'Nuts', 'Pears', 'Plantation fruit', 'Stone fruit', 'Tropical stone fruit'],
    
    'Biochar':                  [# Cropping
                                'Hay', 'Summer cereals', 'Summer legumes', 'Summer oilseeds', 'Winter cereals', 'Winter legumes', 'Winter oilseeds',
                                # Horticulture:
                                'Apples', 'Citrus', 'Grapes', 'Nuts', 'Pears', 'Plantation fruit', 'Stone fruit', 'Tropical stone fruit'],

    'HIR - Beef':               ['Beef - natural land'],
    'HIR - Sheep':              ['Sheep - natural land'],
}


AG_MANAGEMENTS = {
    'Asparagopsis taxiformis': True,
    'Precision Agriculture': True,
    'Ecological Grazing': False,
    'Savanna Burning': True,
    'AgTech EI': True,
    'Biochar': True,
    'HIR - Beef': True,
    'HIR - Sheep': True,
}
"""
The dictionary below contains a master list of all agricultural management options and
which land uses they correspond to.

To disable an ag-mangement option, change the corresponding value in the AG_MANAGEMENTS dictionary to False.
"""

AG_MANAGEMENTS_REVERSIBLE = {
    'Asparagopsis taxiformis': True,
    'Precision Agriculture': True,
    'Ecological Grazing': True,
    'Savanna Burning': True,
    'AgTech EI': True,
    'Biochar': True,
    'HIR - Beef': True,
    'HIR - Sheep': True,
}
"""
The values of the below dictionary determine whether the model is allowed to abandon agricultural
management options on cells in the years after it chooses to utilise them. For example, if a cell has is using 'Asparagopsis taxiformis',
and the corresponding value in this dictionary is False, all cells using Asparagopsis taxiformis must also utilise this land use
and agricultural management combination in all subsequent years.

WARNING: changing to False will result in 'locking in' land uses on cells that utilise the agricultural management option for
the rest of the simulation. This may be an unintended side effect.
"""


# The cost for removing and establishing irrigation infrastructure ($ per hectare)
REMOVE_IRRIG_COST = 5000
NEW_IRRIG_COST = 10000

# Savanna burning cost per hectare per year ($/ha/yr)
SAVBURN_COST_HA_YR = 10

# The minimum value an agricultural management variable must take for the write_output function to consider it being used on a cell
AGRICULTURAL_MANAGEMENT_USE_THRESHOLD = 0.1

# Productivity contribution of HIR compared to not implementing HIR
HIR_PRODUCTIVITY_CONTRIBUTION = 0.5

# Maintainace cost for HIR
BEEF_HIR_MAINTENANCE_COST_PER_HA_PER_YEAR = 0
SHEEP_HIR_MAINTENANCE_COST_PER_HA_PER_YEAR = 0

# HIR effecting years
HIR_EFFECT_YEARS = 91



# ---------------------------------------------------------------------------- #
# Off-land commodity parameters
# ---------------------------------------------------------------------------- #

OFF_LAND_COMMODITIES = ['pork', 'chicken', 'eggs', 'aquaculture']
EGGS_AVG_WEIGHT = 60  # Average weight of an egg in grams


# ---------------------------------------------------------------------------- #
# Environmental parameters
# ---------------------------------------------------------------------------- #

# Take data from 'GHG_targets.xlsx', 
GHG_TARGETS_DICT = {
    'off':      None,
    'low':      '1.8C (67%) excl. avoided emis SCOPE1',
    'medium':   '1.5C (50%) excl. avoided emis SCOPE1',
    'high':     '1.5C (67%) excl. avoided emis SCOPE1',
}

# Greenhouse gas emissions limits and parameters *******************************
GHG_EMISSIONS_LIMITS = 'high'        # 'off', 'low', 'medium', or 'high'
'''
`GHG_EMISSIONS_LIMITS` options include: 
- Assuming agriculture is responsible to sequester 100% of the carbon emissions
    - '1.5C (67%)', '1.5C (50%)', or '1.8C (67%)' 
- Assuming agriculture is responsible to sequester carbon emissions not including electricity emissions and  off-land emissions 
    - '1.5C (67%) excl. avoided emis', '1.5C (50%) excl. avoided emis', or '1.8C (67%) excl. avoided emis'
- Assuming agriculture is responsible to sequester carbon emissions only in the scope 1 emissions (i.e., direct emissions from land-use and livestock types)
    - '1.5C (67%) excl. avoided emis SCOPE1', '1.5C (50%) excl. avoided emis SCOPE1', or '1.8C (67%) excl. avoided emis SCOPE1'
'''
GHG_percent = 1



# Carbon price scenario: either 'AS_GHG', 'Default', '100', or 'CONSTANT', or NONE.
# Setting to None falls back to the 'Default' scenario.
CARBON_PRICES_FIELD = 'CONSTANT'

# Automatically update the carbon price field if it is set to 'AS_GHG'
if CARBON_PRICES_FIELD == 'AS_GHG':
    CARBON_PRICES_FIELD = GHG_TARGETS_DICT[GHG_EMISSIONS_LIMITS][:9].replace('(','')  # '1.5C (67%) excl. avoided emis' -> '1.5C 67%'

if CARBON_PRICES_FIELD == 'CONSTANT':
    CARBON_PRICE_COSTANT = 0.0  # The constant value to add to the carbon price (e.g., $10/tonne CO2e).
'''
Only works when CARBON_PRICES_FIELD is set to 'CONSTANT'.
'''


USE_GHG_SCOPE_1 = True  # If True, only considers the basic GHG types (i.e., CO2E_KG_HA_SOIL, CO2E_KG_HEAD_DUNG_URINE, CO2E_KG_HEAD_ENTERIC, CO2E_KG_HEAD_FODDER, CO2E_KG_HEAD_IND_LEACH_RUNOFF, CO2E_KG_HEAD_SEED).
'''
Basic GHG types are the direct emissions from the land-use and livestock types, excluding
indirect emissions such as fertiliser, irrigation, land management, etc.
'''

CROP_GHG_SCOPE_1 = ['CO2E_KG_HA_SOIL']
LVSTK_GHG_SCOPE_1 = ['CO2E_KG_HEAD_DUNG_URINE', 'CO2E_KG_HEAD_ENTERIC', 'CO2E_KG_HEAD_IND_LEACH_RUNOFF', 'CO2E_KG_HEAD_MANURE_MGT']


# Number of years over which to spread (average) soil carbon accumulation (from Mosnier et al. 2022 and Johnson et al. 2021)
SOC_AMORTISATION = 91   # (2025/05/05) Change from 15 -> 91; This makes sure BIO_CHAR has the same GHG effect span as HIR

GHG_CONSTRAINT_TYPE = 'hard'  # Adds GHG limits as a constraint in the solver (linear programming approach)
# GHG_CONSTRAINT_TYPE = 'soft'  # Adds GHG usage as a type of slack variable in the solver (goal programming approach)

# Weight for the GHG/Demand deviation in the objective function
SOLVE_WEIGHT_ALPHA = 1  
''' 
Range from 0 to 1 that balances the relative important between economic values and biodiversity scores.
 - if approaching 0, the model will focus on maximising biodiversity scores.
 - if approaching 1, the model will focus on maximising prifit (or minimising cost).
'''

SOLVE_WEIGHT_BETA = 0.5
'''
The weight of the deviations from target in the objective function.
 - if approaching 0, the model will ignore the deviations from target.
 - if approaching 1, the model will try harder to meet the target.
'''


# Water use yield and parameters *******************************
WATER_LIMITS = 'on'     # 'on' or 'off'. 'off' will turn off water net yield limit constraints in the solver.

WATER_CONSTRAINT_TYPE = 'hard'  # Adds water limits as a constraint in the solver (linear programming approach)
# WATER_CONSTRAINT_TYPE = 'soft'  # Adds water usage as a type of slack variable in the solver (goal programming approach)


# Regionalisation to enforce water use limits by
WATER_REGION_DEF = 'Drainage Division'         # 'River Region' or 'Drainage Division' Bureau of Meteorology GeoFabric definition
"""
    Water net yield targets: the value represents the proportion of the historical water yields
    that the net yield must exceed in a given year. Base year (2010) uses base year net yields as targets.
    Everything past the latest year specified uses the target figure for the latest year.
    
    Safe and just Earth system boundaries suggests a water stress of 0.2 (yield of 0.8). This is inclusive of
    domestic/industrial: https://www.nature.com/articles/s41586-023-06083-8, Approximately 70% of the total water use
    is used for agricultural purposes. This includes water used for irrigation, livestock, and domestic purposes on farms,
    with the rest used for domestic/industrial  https://soe.dcceew.gov.au/inland-water/pressures/population
    Hence, assuming that this proportion is uniform over all catchments and remains constant over time then if water
    stress is 0.2 then agriculture can use up 70% of this, leaving 30% for domestic/industrial. The water yield target for ag
    should then be historical net yield * (1 - water stress * agricultural share)
    
    Aqueduct water stress levels:
    Low stress < 10% of the water available is withdrawn annually
    Low to medium stress 10-20% of the water available is withdrawn annually
    Medium to high stress 20-40% 10% of the water available is withdrawn annually
    High stress 40-80% of the water available is withdrawn annually
    Extremely high stress > 80% of the water available is withdrawn annually
    
    https://chinawaterrisk.org/resources/analysis-reviews/aqueduct-global-water-stress-rankings/ 
"""

WATER_STRESS = 0.6                                      # Aqueduct limit catchments, 0.6 means the water yield in a region must be >= 60% of the historical water yield
WATER_USE_SHARE_AG = 0.7                                # Ag share is 70% across all catchments, could be updated for each specific catchment based on actual data
WATER_USE_SHARE_DOMESTIC  = 1 - WATER_USE_SHARE_AG      # Domestic share is 30% across all catchments, could be updated for each specific catchment based on actual data


# Consider livestock drinking water (0 [off] or 1 [on]) ***** Livestock drinking water can cause infeasibility issues with water constraint in Pilbara
LIVESTOCK_DRINKING_WATER = 1

# Consider water license costs (0 [off] or 1 [on]) of land-use transition ***** If on then there is a noticeable water sell-off by irrigators in the MDB when maximising profit
INCLUDE_WATER_LICENSE_COSTS = 1



# Biodiversity limits and parameters *******************************


# ------------------- Agricultural biodiversity parameters -------------------

GBF2_CONSTRAINT_TYPE = 'hard' # Adds biodiversity limits as a constraint in the solver (linear programming approach)
# GBF2_CONSTRAINT_TYPE = 'soft'  # Adds biodiversity usage as a type of slack variable in the solver (goal programming approach)
'''
The constraint type for the biodiversity target.
- 'hard' adds biodiversity limits as a constraint in the solver (linear programming approach)
- 'soft' adds biodiversity usage as a type of slack variable in the solver (goal programming approach)
'''

# Set biodiversity target (0 - 1 e.g., 0.3 = 30% of total achievable Zonation biodiversity benefit)
GBF2_TARGETS_DICT = {
    'off':     None,
    'low':    {2030: 0,    2050: 0,    2100: 0},
    'medium': {2030: 0.30, 2050: 0.30, 2100: 0.30},
    'high':   {2030: 0.30, 2050: 0.50, 2100: 0.50},
}

# Global Biodiversity Framework Target 2: Restore 30% of all Degraded Ecosystems
BIODIVERSITY_TARGET_GBF_2 = 'high'            # 'off', 'low', 'medium', or 'high'
'''
Kunming-Montreal Global Biodiversity Framework Target 2: Restore 30% of all Degraded Ecosystems
Ensure that by 2030 at least 30 per cent of areas of degraded terrestrial, inland water, and coastal and marine ecosystems are under effective restoration,
in order to enhance biodiversity and ecosystem functions and services, ecological integrity and connectivity.
 - 'off' will turn off the GBF-3 target. 
 - 'low' is the low level of biodiversity target (i.e., restore 0% of degreaded biodiversity socore in the 'priority degraded land').
 - 'medium' is the medium level of biodiversity target (i.e., restore 15% of degreaded biodiversity socore in the 'priority degraded land').
 - 'high' is the high level of biodiversity target (i.e., restore 25% of degreaded biodiversity socore in the 'priority degraded land').
'''


GBF2_PRIORITY_DEGRADED_AREAS_PERCENTAGE_CUT = 20
'''
Based on Zonation alogrithm, the biodiversity feature coverage (an indicator of overall biodiversity benifits) is 
more attached to high rank cells (rank is an indicator of importance/priority in biodiversity conservation). 
For example, cells with rank between 0.9-1.0 only cover 20% of the areas but contribute to 40% of the biodiversity benefits.

By sorting the rank values from high to low and plot the cumulative area and cumulative biodiversity benefits,
we can get the a curve that shows the relationship between the area and the biodiversity benefits. In LUTO, we normalise
the area and biodiversity benefits between 0-100, and use the `GBF2_PRIORITY_DEGRADED_AREAS_PERCENTAGE_CUT` as the threshold
to identify the priority degraded areas that should be conserved to achieve the biodiversity target.

If set to 0, no cells will be considered as priority degraded areas, equal to not setting any GBF2 target.
If set to 100, all cells will be considered as priority degraded areas, equal to setting the GBF2 to the LUTO study area.
'''


# Connectivity source source
CONNECTIVITY_SOURCE = 'NCI'                 # 'DCCEEW_NCI', 'NATURAL_AREA_CONNECTIVITY' or 'NONE'
'''
The connectivity source is the source of the connectivity score used to weigh the raw biodiversity priority score.
This score is normalised between 0 (fartherst) and 1 (closest).
Can be either 'NCI' or 'DWI'.
- if 'NCI' is selected, the connectivity score is sourced from the DCCEEW's National Connectivity Index (v3.0).
- if 'DWI' is selected, the connectivity score is calculated as distance to the nearest area of natural land as mapped
        by the National Land Use Map of Australia.
- if 'NONE' is selected, the connectivity score is not used in the biodiversity calculation.
'''

# Connectivity score importance
CONNECTIVITY_LB = 0.7                       # Avaliable values are [0.5, 0.6, 0.7, 0.8, 0.9]
'''
The relative importance of the connectivity score in the biodiversity calculation. Used to scale the raw biodiversity score.
I.e., the lower bound of the connectivity score for weighting the raw biodiversity priority score is CONNECTIVITY_LB.
'''


# Habitat condition data source
HABITAT_CONDITION = 'USER_DEFINED'                  # One of [10, 25, 50, 75, 90], or 'USER_DEFINED'
'''
Different land-use types have different biodiversity degradation impacts. We calculated the percentiles values of HCAS (indicating the
suitability for wild animals ranging between 0-1) for each land-use type.Avaliable percentiles is one of [10, 25, 50, 75, 90].

For example, the 50th percentile for 'Beef - Modified land' is 0.22, meaning this land retains 22% biodiversity score compared
to undisturbed natural land.
'''


# Biodiversity value under default late dry season savanna fire regime
BIO_CONTRIBUTION_LDS = 0.8
''' For example, 0.8 means that all areas in the area eligible for savanna burning have a biodiversity value of 0.8 * the raw biodiv value
    (due to hot fires etc). When EDS sav burning is implemented the area is attributed the full biodiversity value (i.e., 1.0).
'''

# Non-agricultural biodiversity parameters 
BIO_CONTRIBUTION_ENV_PLANTING = 0.8
BIO_CONTRIBUTION_CARBON_PLANTING_BLOCK = 0.1
BIO_CONTRIBUTION_CARBON_PLANTING_BELT = 0.1
BIO_CONTRIBUTION_RIPARIAN_PLANTING = 1.2
BIO_CONTRIBUTION_AGROFORESTRY = 0.75       
BIO_CONTRIBUTION_BECCS = 0
''' 
The benefit of each non-agricultural land use to biodiversity is set as a proportion to the raw biodiversity priority value.
For example, if the raw biodiversity priority value is 0.6 and the benefit is 0.8, then the biodiversity value
will be 0.6 * 0.8 = 0.48.
'''




# ---------------------- Vegetation parameters ----------------------

GBF3_TARGET_CLASS  = 'MVS'                  # 'MVG', 'MVS', 'MVG_IBRA', 'MVS_IBRA'
'''
The National Vegetation Information System (NVIS) provides the 100m resolution information on
the distribution of vegetation (~30 primary group layers, or ~90 subgroup layers) across Australia.

- If 'MVG/MVS' is selected, use need to define conservation target for each NVIS group across the whole study area.
- If 'MVS_IBRA/MVG_IBRA' is selected, use need to define conservation target for each NVIS group for selected the IBRA region.
'''

GBF3_TARGETS_DICT = {
    'off':     None,
    'medium':  30,
    'high':    50,
    'USER_DEFINED': None
}

BIODIVERSITY_TARGET_GBF_3  = 'off'           # 'off', 'medium', 'high', or 'USER_DEFINED'
'''
Target 3 of the Kunming-Montreal Global Biodiversity Framework:
protect and manage 30% of the world's land, water, and coastal areas by 2030.

- if 'off' is selected, turn off the GBF-3 target for biodiversity.
- if 'medium' is selected, the conservation target is set to 30% for each NVIS group at 2050.
- if 'high' is selected, the conservation target is set to 50% for each NVIS group at 2050.
- if 'USER_DEFINED' is selected, the conservation target is reading from `input.BIODIVERSITY_GBF3_SCORES_AND_TARGETS.xlsx`.
'''



# ------------------------------- Species parameters -------------------------------
BIODIVERSITY_TARGET_GBF_4_SNES =  'off'           # 'on' or 'off'.
BIODIVERSITY_TARGET_GBF_4_ECNES = 'off'           # 'on' or 'off'.

'''
Target 4 of the Kunming-Montreal Global Biodiversity Framework (GBF) aims to 
halt the extinction of known threatened species, protect genetic diversity, 
and manage human-wildlife interactions
'''



# -------------------------------- Climate change impacts on biodiversity -------------------------------
BIODIVERSITY_TARGET_GBF_8 = 'off'           # 'on' or 'off'.
'''
Target 8 of the Kunming-Montreal Global Biodiversity Framework (GBF) aims to 
reduce the impacts of climate change on biodiversity and ecosystems.
'''




# ---------------------------------------------------------------------------- #
# Other parameters
# ---------------------------------------------------------------------------- #

# Cell culling
CULL_MODE = 'absolute'      # cull to include at most MAX_LAND_USES_PER_CELL
# CULL_MODE = 'percentage'    # cull the LAND_USAGE_THRESHOLD_PERCENTAGE % most expensive options
# CULL_MODE = 'none'          # do no culling

MAX_LAND_USES_PER_CELL = 12         if CULL_MODE == 'absolute' else 'Not used'
LAND_USAGE_CULL_PERCENTAGE = 0.15   if CULL_MODE == 'percentage' else 'Not used'

# Non-ag output coding. Non-agricultural land uses will appear on the land use map offset by this amount (e.g. land use 0 will appear as 100)
NON_AGRICULTURAL_LU_BASE_CODE = 100

# Number of decimals to round the lower bound matrices to for non-agricultural land uses and agricultural management options.
ROUND_DECMIALS = 6


""" NON-AGRICULTURAL LAND USES (indexed by k)
0: 'Environmental Plantings'
1: 'Riparian Plantings'
2: 'Sheep Agroforestry'
3: 'Beef Agroforestry'
4: 'Carbon Plantings (Block)'
5: 'Sheep Carbon Plantings (Belt)'
6: 'Beef Carbon Plantings (Belt)'
7: 'BECCS'
8: 'Destocked - natural land'


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