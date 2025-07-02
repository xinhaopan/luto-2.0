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

import luto.settings as settings


# Get the root directory of the data
YR_BASE = 2010

# Colors for reporting HTML to loop through
COLORS = [
    "#8085e9",
    "#f15c80",
    "#e4d354",
    "#2b908f",
    "#f45b5b",
    "#7cb5ec",
    "#434348",
    "#90ed7d",
    "#f7a35c",
    "#91e8e1",
]


# Define crop-lvstk land uses
LU_CROPS = ['Apples','Citrus','Cotton','Grapes','Hay','Nuts','Other non-cereal crops',
            'Pears','Plantation fruit','Rice','Stone fruit','Sugar','Summer cereals',
            'Summer legumes','Summer oilseeds','Tropical stone fruit','Vegetables',
            'Winter cereals','Winter legumes','Winter oilseeds']

LVSTK_NATURAL = ['Beef - natural land','Dairy - natural land','Sheep - natural land']

LVSTK_MODIFIED = ['Beef - modified land','Dairy - modified land','Sheep - modified land']

LU_LVSTKS = LVSTK_NATURAL + LVSTK_MODIFIED

LU_UNALLOW = ['Unallocated - modified land','Unallocated - natural land']


LU_NATURAL = ['Beef - natural land',
              'Dairy - natural land',
              'Sheep - natural land',
              'Unallocated - natural land']


# Define the commodity categories
COMMODITIES_ON_LAND = ['Apples','Beef live export','Beef meat','Citrus','Cotton','Dairy','Grapes',
                       'Hay','Nuts','Other non-cereal crops', 'Pears', 'Plantation fruit',
                       'Rice', 'Sheep live export', 'Sheep meat', 'Sheep wool', 'Stone fruit', 'Sugar',
                       'Summer cereals', 'Summer legumes', 'Summer oilseeds', 'Tropical stone fruit',
                       'Vegetables','Winter cereals','Winter legumes','Winter oilseeds']

COMMODITIES_OFF_LAND = ['Aquaculture', 'Chicken', 'Eggs', 'Pork' ]

COMMODITIES_ALL = COMMODITIES_ON_LAND + COMMODITIES_OFF_LAND


# Define land use code for am and non-ag land uses
AM_SELECT = [i for i in settings.AG_MANAGEMENTS if settings.AG_MANAGEMENTS[i]]
AM_DESELECT = [i for i in settings.AG_MANAGEMENTS if not settings.AG_MANAGEMENTS[i]]
AM_MAP_CODES = {i:(AM_SELECT.index(i) + 1) for i in AM_SELECT}

NON_AG_SELECT = [i for i in settings.NON_AG_LAND_USES if settings.NON_AG_LAND_USES[i]]
NON_AG_DESELECT = [i for i in settings.NON_AG_LAND_USES if not settings.NON_AG_LAND_USES[i]]
NON_AG_MAP_CODES = {i:(NON_AG_SELECT.index(i) + 1) for i in NON_AG_SELECT}

AM_NON_AG_CODES = {**AM_MAP_CODES, **NON_AG_MAP_CODES}
AM_NON_AG_REMOVED_DESC = AM_DESELECT + NON_AG_DESELECT


# Define the file name patterns for each category
GHG_FNAME2TYPE = {'GHG_emissions_separate_agricultural_landuse': 'Agricultural Landuse',
                  'GHG_emissions_separate_agricultural_management': 'Agricultural Management',
                  'GHG_emissions_separate_no_ag_reduction': 'Non-Agricultural Landuse',
                  'GHG_emissions_separate_transition_penalty': 'Transition Penalty',
                  'GHG_emissions_offland_commodity': 'Offland Commodity',}


AG_LANDUSE_MERGE_LANDTYPE = ['Apples', 'Beef', 'Citrus', 'Cotton', 'Dairy', 'Grapes', 'Hay', 'Nuts', 'Other non-cereal crops',
                             'Pears', 'Plantation fruit', 'Rice', 'Sheep', 'Stone fruit', 'Sugar', 'Summer cereals',
                             'Summer legumes', 'Summer oilseeds', 'Tropical stone fruit', 'Unallocated - modified land', 
                             'Unallocated - natural land', 'Vegetables', 'Winter cereals', 'Winter legumes', 'Winter oilseeds']


# Define the renaming of the Agricultural-Managment and Non-Agricultural 
RENAME_AM = {
    "Asparagopsis taxiformis": "Methane reduction (livestock)",
    "Precision Agriculture": "Agricultural technology (fertiliser)", 
    "Ecological Grazing": "Regenerative agriculture (livestock)", 
    "Savanna Burning": "Early dry-season savanna burning",
    "AgTech EI": "Agricultural technology (energy)",
}

RENAME_NON_AG = {
    "Environmental Plantings": "Environmental plantings (mixed species)",
    "Riparian Plantings": "Riparian buffer restoration (mixed species)",
    "Sheep Agroforestry": "Agroforestry (mixed species + sheep)",
    "Beef Agroforestry": "Agroforestry (mixed species + beef)",
    "Carbon Plantings (Block)": "Carbon plantings (monoculture)",
    "Sheep Carbon Plantings (Belt)": "Farm forestry (hardwood timber + sheep)",
    "Beef Carbon Plantings (Belt)": "Farm forestry (hardwood timber + beef)",
    "BECCS": "BECCS (Bioenergy with Carbon Capture and Storage)",
    "Destocked - natural land": "Destocked - natural land",
}

RENAME_AM_NON_AG = {**RENAME_AM, **RENAME_NON_AG}

# Read the land uses from the file
with open(f'{settings.INPUT_DIR}/ag_landuses.csv') as f:
    AG_LANDUSE = [line.strip() for line in f]
    
    
# This will be used in the HTML for reporting spatial maps
SPATIAL_MAP_DICT = {
    'Int_Map': ['lumap', 'non_ag', 'ammap', 'lmmap'],       # Each cell is an integer, representing a land-use for [AG, AM, Non-AG]
    'Ag_LU': AG_LANDUSE,                                    # Percentage of Agricultural Landuse to a cell
    'Ag_Mgt': list(settings.AG_MANAGEMENTS.keys()),         # Percentage of Agricultural Management to a cell                 
    'Non-Ag_LU': list(settings.NON_AG_LAND_USES.keys())     # Percentage of Non-Agricultural Landuse to a cell
}


# Get the non-agricultural land uses raw names
NON_AG_LANDUSE_RAW = list(settings.NON_AG_LAND_USES.keys())
NON_AG_LANDUSE_RAW = [i for i in NON_AG_LANDUSE_RAW if settings.NON_AG_LAND_USES[i]]


# Merge the land uses
LANDUSE_ALL_RAW = AG_LANDUSE + NON_AG_LANDUSE_RAW
LANDUSE_ALL_RENAMED = AG_LANDUSE + list(RENAME_NON_AG.values()) 




# Define the GHG categories
GHG_NAMES = {
    # Agricultural Landuse
    'TCO2E_CHEM_APPL': 'Chemical Application',
    'TCO2E_CROP_MGT': 'Crop Management',
    'TCO2E_CULTIV': 'Cultivation',
    'TCO2E_FERT_PROD': 'Fertiliser production',
    'TCO2E_HARVEST': 'Harvesting',
    'TCO2E_IRRIG': 'Irrigation',
    'TCO2E_PEST_PROD': 'Pesticide production',
    'TCO2E_SOWING': 'Sowing',
    'TCO2E_ELEC': 'Electricity Use livestock',
    'TCO2E_FODDER': 'Fodder production',
    'TCO2E_FUEL': 'Fuel Use livestock',
    'TCO2E_IND_LEACH_RUNOFF': 'Agricultural soils: Indirect leaching and runoff',
    'TCO2E_MANURE_MGT': 'Livestock Manure Management (biogenic)',
    'TCO2E_SEED': 'Pasture Seed production',
    'TCO2E_SOIL': 'Agricultural soils: Direct Soil Emissions (biogenic)',
    'TCO2E_DUNG_URINE': 'Agricultural soils: Animal production, dung and urine',
    'TCO2E_ENTERIC': 'Livestock Enteric Fermentation (biogenic)',
    # Agricultural Management
    'TCO2E_Asparagopsis taxiformis': 'Asparagopsis taxiformis', 
    'TCO2E_Precision Agriculture': 'Precision Agriculture',
    'TCO2E_Ecological Grazing': 'Ecological Grazing',
    # Non-Agricultural Landuse
    'TCO2E_Agroforestry': 'Agroforestry', 
    'TCO2E_Environmental Plantings': 'Environmental Plantings',
    'TCO2E_Riparian Plantings': 'Riparian Plantings',
    'TCO2E_Carbon Plantings (Belt)': 'Carbon Plantings (Belt)',
    'TCO2E_Carbon Plantings (Block)': 'Carbon Plantings (Block)',
    'TCO2E_BECCS': 'BECCS',
    'TCO2E_Savanna Burning': 'Savanna Burning',
    'TCO2E_AgTech EI': 'AgTech EI',
}

GHG_CATEGORY = {'Agricultural soils: Animal production, dung and urine': {"CH4":0.5,"CO2":0.5},
                'Livestock Enteric Fermentation (biogenic)':{'CH4':1},
                'Agricultural soils: Direct Soil Emissions (biogenic)':{"N2O":1},
                
                'Asparagopsis taxiformis':{'Asparagopsis taxiformis':1},
                'Precision Agriculture':{'Precision Agriculture':1},
                'Ecological Grazing':{'Ecological Grazing':1}}


