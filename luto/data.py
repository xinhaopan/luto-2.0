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



import os
import xarray as xr
import numpy as np
import pandas as pd
import rasterio
import rasterio.features
import geopandas as gpd
import netCDF4 # necessary for running luto in Denethor

from luto import tools
import luto.settings as settings
import luto.economics.agricultural.quantity as ag_quantity
import luto.economics.non_agricultural.quantity as non_ag_quantity

from math import ceil
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Literal, Optional
from affine import Affine
from scipy.interpolate import interp1d
from scipy.ndimage import distance_transform_edt

from luto.tools.spatializers import upsample_array


def dict2matrix(d, fromlist, tolist):
    """Return 0-1 matrix mapping 'from-vectors' to 'to-vectors' using dict d."""
    A = np.zeros((len(tolist), len(fromlist)), dtype=np.int8)
    for j, jstr in enumerate(fromlist):
        for istr in d[jstr]:
            i = tolist.index(istr)
            A[i, j] = True
    return A


def get_base_am_vars(ncells, ncms, n_ag_lus):
    """
    Get the 2010 agricultural management option vars.
    It is assumed that no agricultural management options were used in 2010,
    so get zero arrays in the correct format.
    """
    am_vars = {}
    for am in settings.AG_MANAGEMENTS_TO_LAND_USES:
        am_vars[am] = np.zeros((ncms, ncells, n_ag_lus))

    return am_vars



def lumap2non_ag_l_mk(lumap, num_non_ag_land_uses: int):
    """
    Convert the land-use map to a decision variable X_rk, where 'r' indexes cell and
    'k' indexes non-agricultural land use.

    Cells used for agricultural purposes have value 0 for all k.
    """
    base_code = settings.NON_AGRICULTURAL_LU_BASE_CODE
    non_ag_lu_codes = list(range(base_code, base_code + num_non_ag_land_uses))

    # Set up a container array of shape r, k.
    x_rk = np.zeros((lumap.shape[0], num_non_ag_land_uses), dtype=bool)

    for i,k in enumerate(non_ag_lu_codes):
        kmap = np.where(lumap == k, True, False)
        x_rk[:, i] = kmap

    return x_rk.astype(bool)


@dataclass
class Data:
    """
    Contains all data required for the LUTO model to run. Loads all data upon initialisation.
    """

    def __init__(self) -> None:
        """
        Sets up output containers (lumaps, lmmaps, etc) and loads all LUTO data, adjusted
        for resfactor.
        """
        # Path for write module - overwrite when provided with a base and target year
        self.path = None
        self.path_begin_end_compare = None
        
        # Timestamp of simulation to which this object belongs.
        with open(os.path.join(settings.OUTPUT_DIR, '.timestamp'), 'r') as f:
            self.timestamp = f.read().strip()

        # Setup output containers
        self.lumaps = {}
        self.lmmaps = {}
        self.ammaps = {}
        self.ag_dvars = {}
        self.non_ag_dvars = {}
        self.ag_man_dvars = {}
        self.prod_data = {}
        self.obj_vals = {}

        print('')
        print(f'Beginning data initialisation at RES{settings.RESFACTOR}...')

        self.YR_CAL_BASE = 2010  # The base year, i.e. where year index yr_idx == 0.

        ###############################################################
        # Masking and spatial coarse graining.
        ###############################################################
        print("\tSetting up masking and spatial course graining data...", flush=True)

        # Set resfactor multiplier
        self.RESMULT = settings.RESFACTOR ** 2

        # Set the nodata and non-ag code
        self.NODATA = -9999
        self.MASK_LU_CODE = -1

        # Load LUMAP without resfactor
        self.LUMAP_NO_RESFACTOR = pd.read_hdf(os.path.join(settings.INPUT_DIR, "lumap.h5")).to_numpy().astype(np.int8)   # 1D (ij flattend),  0-27 for land uses; -1 for non-agricultural land uses; All cells in Australia (land only)

        # NLUM mask.
        with rasterio.open(os.path.join(settings.INPUT_DIR, "NLUM_2010-11_mask.tif")) as rst:
            self.NLUM_MASK = rst.read(1).astype(np.int8)                                                                # 2D map,  0 for ocean, 1 for land
            self.LUMAP_2D_FULLRES = np.full_like(self.NLUM_MASK, self.NODATA, dtype=np.int16)                           # 2D map,  full of nodata (-9999)
            np.place(self.LUMAP_2D_FULLRES, self.NLUM_MASK == 1, self.LUMAP_NO_RESFACTOR)                               # 2D map,  -9999 for ocean; -1 for desert, urban, water, etc; 0-27 for land uses
            self.GEO_META_FULLRES = rst.meta                                                                            # dict,  key-value pairs of geospatial metadata for the full resolution land-use map
            self.GEO_META_FULLRES['dtype'] = 'float32'                                                                  # Set the data type to float32
            self.GEO_META_FULLRES['nodata'] = self.NODATA                                                               # Set the nodata value to -9999

        # Mask out non-agricultural, non-environmental plantings land (i.e., -1) from lumap 
        # (True means included cells. Boolean dtype.)
        self.LUMASK = self.LUMAP_NO_RESFACTOR != self.MASK_LU_CODE                                                      # 1D (ij flattend);  `True` for land uses; `False` for desert, urban, water, etc

        # Return combined land-use and resfactor mask
        if settings.RESFACTOR > 1:
            rf_mask = self.NLUM_MASK.copy()
            nonzeroes = np.nonzero(rf_mask)
            rf_mask[int(settings.RESFACTOR/2)::settings.RESFACTOR, int(settings.RESFACTOR/2)::settings.RESFACTOR] = 0
            resmask = np.where(rf_mask[nonzeroes] == 0, True, False)
            self.MASK = self.LUMASK * resmask
            self.LUMAP_2D_RESFACTORED = self.LUMAP_2D_FULLRES[int(settings.RESFACTOR/2)::settings.RESFACTOR, int(settings.RESFACTOR/2)::settings.RESFACTOR]
            self.GEO_META = self.update_geo_meta()
        elif settings.RESFACTOR == 1:
            self.MASK = self.LUMASK
            self.GEO_META = self.GEO_META_FULLRES
            self.LUMAP_2D_RESFACTORED = self.LUMAP_2D_FULLRES
        else:
            raise KeyError("Resfactor setting invalid")
        
        
        # Get the lon/lat coordinates.
        self.COORD_LON_LAT_2D_FULLRES = self.get_coord(np.nonzero(self.NLUM_MASK), self.GEO_META_FULLRES['transform'])     # 2D array([lon, ...], [lat, ...]);  lon/lat coordinates for each cell in Australia (land only)
        self.COORD_LON_LAT = [i[self.MASK] for i in self.COORD_LON_LAT_2D_FULLRES]  # Only keep the coordinates for the cells that are not masked out (i.e., land uses). 2D array([lon, ...], [lat, ...]);  lon/lat coordinates for each cell in Australia (land only) and not masked out
        
        

        ###############################################################
        # Load agricultural crop and livestock data.
        ###############################################################
        print("\tLoading agricultural crop and livestock data...", flush=True)
        self.AGEC_CROPS = pd.read_hdf(os.path.join(settings.INPUT_DIR, "agec_crops.h5"), where=self.MASK)
        self.AGEC_LVSTK = pd.read_hdf(os.path.join(settings.INPUT_DIR, "agec_lvstk.h5"), where=self.MASK)
        
        # Price multipliers for livestock and crops over the years.
        self.CROP_PRICE_MULTIPLIERS = pd.read_excel(os.path.join(settings.INPUT_DIR, "ag_price_multipliers.xlsx"), sheet_name="AGEC_CROPS", index_col="Year")
        self.LVSTK_PRICE_MULTIPLIERS = pd.read_excel(os.path.join(settings.INPUT_DIR, "ag_price_multipliers.xlsx"), sheet_name="AGEC_LVSTK", index_col="Year")



        ###############################################################
        # Set up lists of land-uses, commodities etc.
        ###############################################################
        print("\tSetting up lists of land uses, commodities, etc...", flush=True)

        # Read in lexicographically ordered list of land-uses.
        self.AGRICULTURAL_LANDUSES = pd.read_csv((os.path.join(settings.INPUT_DIR, 'ag_landuses.csv')), header = None)[0].to_list()
        self.NON_AGRICULTURAL_LANDUSES = list(settings.NON_AG_LAND_USES.keys())

        self.NONAGLU2DESC = dict(zip(range(settings.NON_AGRICULTURAL_LU_BASE_CODE, settings.NON_AGRICULTURAL_LU_BASE_CODE + len(self.NON_AGRICULTURAL_LANDUSES)), self.NON_AGRICULTURAL_LANDUSES))
        self.DESC2NONAGLU = {value: key for key, value in self.NONAGLU2DESC.items()}
 
        # Get number of land-uses
        self.N_AG_LUS = len(self.AGRICULTURAL_LANDUSES)
        self.N_NON_AG_LUS = len(self.NON_AGRICULTURAL_LANDUSES)

        # Construct land-use index dictionary (distinct from LU_IDs!)
        self.AGLU2DESC = {i: lu for i, lu in enumerate(self.AGRICULTURAL_LANDUSES)}
        self.DESC2AGLU = {value: key for key, value in self.AGLU2DESC.items()}
        self.AGLU2DESC[-1] = 'Non-agricultural land'
        
        # Combine ag and non-ag landuses
        self.ALL_LANDUSES = self.AGRICULTURAL_LANDUSES + self.NON_AGRICULTURAL_LANDUSES
        self.ALLDESC2LU = {**self.DESC2AGLU, **self.DESC2NONAGLU}
        self.ALLLU2DESC = {**self.AGLU2DESC, **self.NONAGLU2DESC}

        # Some useful sub-sets of the land uses.
        self.LU_CROPS = [ lu for lu in self.AGRICULTURAL_LANDUSES if 'Beef' not in lu
                                                                  and 'Sheep' not in lu
                                                                  and 'Dairy' not in lu
                                                                  and 'Unallocated' not in lu
                                                                  and 'Non-agricultural' not in lu ]
        self.LU_LVSTK = [ lu for lu in self.AGRICULTURAL_LANDUSES if 'Beef' in lu
                                                        or 'Sheep' in lu
                                                        or 'Dairy' in lu ]
        self.LU_UNALL = [ lu for lu in self.AGRICULTURAL_LANDUSES if 'Unallocated' in lu ]
        self.LU_NATURAL = [
            self.DESC2AGLU["Beef - natural land"],
            self.DESC2AGLU["Dairy - natural land"],
            self.DESC2AGLU["Sheep - natural land"],
            self.DESC2AGLU["Unallocated - natural land"],
        ]
        self.LU_LVSTK_NATURAL = [lu for lu in self.LU_NATURAL if self.AGLU2DESC[lu] != 'Unallocated - natural land']
        self.LU_LVSTK_NATURAL_DESC = [self.AGLU2DESC[lu] for lu in self.LU_LVSTK_NATURAL]
        self.LU_MODIFIED_LAND = [self.DESC2AGLU[lu] for lu in self.AGRICULTURAL_LANDUSES if self.DESC2AGLU[lu] not in self.LU_NATURAL]
        
        self.LU_CROPS_INDICES = [self.AGRICULTURAL_LANDUSES.index(lu) for lu in self.AGRICULTURAL_LANDUSES if lu in self.LU_CROPS]
        self.LU_LVSTK_INDICES = [self.AGRICULTURAL_LANDUSES.index(lu) for lu in self.AGRICULTURAL_LANDUSES if lu in self.LU_LVSTK]
        self.LU_UNALL_INDICES = [self.AGRICULTURAL_LANDUSES.index(lu) for lu in self.AGRICULTURAL_LANDUSES if lu in self.LU_UNALL]

        self.NON_AG_LU_NATURAL = [
            self.DESC2NONAGLU["Environmental Plantings"],
            self.DESC2NONAGLU["Riparian Plantings"],
            self.DESC2NONAGLU["Sheep Agroforestry"],
            self.DESC2NONAGLU["Beef Agroforestry"],
            self.DESC2NONAGLU["Carbon Plantings (Block)"],
            self.DESC2NONAGLU["Sheep Carbon Plantings (Belt)"],
            self.DESC2NONAGLU["Beef Carbon Plantings (Belt)"],
            self.DESC2NONAGLU["BECCS"],
        ]

        # Define which land uses correspond to deep/shallow rooted water yield.
        self.LU_SHALLOW_ROOTED = [
            self.DESC2AGLU["Hay"], self.DESC2AGLU["Summer cereals"], self.DESC2AGLU["Summer legumes"],
            self.DESC2AGLU["Summer oilseeds"], self.DESC2AGLU["Winter cereals"], self.DESC2AGLU["Winter legumes"],
            self.DESC2AGLU["Winter oilseeds"], self.DESC2AGLU["Cotton"], self.DESC2AGLU["Other non-cereal crops"],
            self.DESC2AGLU["Rice"], self.DESC2AGLU["Vegetables"], self.DESC2AGLU["Dairy - modified land"],
            self.DESC2AGLU["Beef - modified land"], self.DESC2AGLU["Sheep - modified land"],
            self.DESC2AGLU["Unallocated - modified land"],
        ]
        self.LU_DEEP_ROOTED = [
            self.DESC2AGLU["Apples"], self.DESC2AGLU["Citrus"], self.DESC2AGLU["Grapes"], self.DESC2AGLU["Nuts"],
            self.DESC2AGLU["Pears"], self.DESC2AGLU["Plantation fruit"], self.DESC2AGLU["Stone fruit"],
            self.DESC2AGLU["Sugar"], self.DESC2AGLU["Tropical stone fruit"],
        ]

        # Derive land management types from AGEC.
        self.LANDMANS = {t[1] for t in self.AGEC_CROPS.columns}  # Set comp., unique entries.
        self.LANDMANS = list(self.LANDMANS)  # Turn into list.
        self.LANDMANS.sort()  # Ensure lexicographic order.

        # Get number of land management types
        self.NLMS = len(self.LANDMANS)

        # List of products. Everything upper case to avoid mistakes.
        self.PR_CROPS = [s.upper() for s in self.LU_CROPS]
        self.PR_LVSTK = [
            'BEEF - MODIFIED LAND LEXP',
            'BEEF - MODIFIED LAND MEAT',
            'BEEF - NATURAL LAND LEXP',
            'BEEF - NATURAL LAND MEAT',
            
            'DAIRY - MODIFIED LAND',
            'DAIRY - NATURAL LAND',
            
            'SHEEP - MODIFIED LAND LEXP',
            'SHEEP - MODIFIED LAND MEAT',
            'SHEEP - MODIFIED LAND WOOL',
            'SHEEP - NATURAL LAND LEXP',
            'SHEEP - NATURAL LAND MEAT',
            'SHEEP - NATURAL LAND WOOL'
        ]
        self.PRODUCTS = self.PR_CROPS + self.PR_LVSTK
        self.PRODUCTS.sort() # Ensure lexicographic order.

        # Get number of products
        self.NPRS = len(self.PRODUCTS)

        # Some land-uses map to multiple products -- a dict and matrix to capture this.
        # Crops land-uses and crop products are one-one. Livestock is more complicated.
        self.LU2PR_DICT = {key: [key.upper()] if key in self.LU_CROPS else [] for key in self.AGRICULTURAL_LANDUSES}
        for lu in self.LU_LVSTK:
            for PR in self.PR_LVSTK:
                if lu.upper() in PR:
                    self.LU2PR_DICT[lu] = self.LU2PR_DICT[lu] + [PR]

        # A reverse dictionary for convenience.
        self.PR2LU_DICT = {pr: key for key, val in self.LU2PR_DICT.items() for pr in val}

        self.LU2PR = dict2matrix(self.LU2PR_DICT, self.AGRICULTURAL_LANDUSES, self.PRODUCTS)


        # List of commodities. Everything lower case to avoid mistakes.
        # Basically collapse 'NATURAL LAND' and 'MODIFIED LAND' products and remove duplicates.
        self.COMMODITIES = { ( s.replace(' - NATURAL LAND', '')
                                .replace(' - MODIFIED LAND', '')
                                .lower() )
                                for s in self.PRODUCTS }
        self.COMMODITIES = list(self.COMMODITIES)
        self.COMMODITIES.sort()
        self.CM_CROPS = [s for s in self.COMMODITIES if s in [k.lower() for k in self.LU_CROPS]]

        # Get number of commodities
        self.NCMS = len(self.COMMODITIES)


        # Some commodities map to multiple products -- dict and matrix to capture this.
        # Crops commodities and products are one-one. Livestock is more complicated.
        self.CM2PR_DICT = { key.lower(): [key.upper()] if key in self.CM_CROPS else []
                    for key in self.COMMODITIES }
        for key, _ in self.CM2PR_DICT.items():
            if len(key.split())==1:
                head = key.split()[0]
                tail = 0
            else:
                head = key.split()[0]
                tail = key.split()[1]
            for PR in self.PR_LVSTK:
                if tail==0 and head.upper() in PR:
                    self.CM2PR_DICT[key] = self.CM2PR_DICT[key] + [PR]
                elif (head.upper()) in PR and (tail.upper() in PR):
                    self.CM2PR_DICT[key] = self.CM2PR_DICT[key] + [PR]
                else:
                    ... # Do nothing, this should be a crop.

        self.PR2CM = dict2matrix(self.CM2PR_DICT, self.COMMODITIES, self.PRODUCTS).T # Note the transpose.
        
        
        # Get the land-use indices for each commodity.
        self.CM2LU_IDX = defaultdict(list)
        for c in self.COMMODITIES:
            for lu in self.AGRICULTURAL_LANDUSES:
                if lu.split(' -')[0].lower() in c:
                    self.CM2LU_IDX[c].append(self.AGRICULTURAL_LANDUSES.index(lu))
                    
                    
        ###############################################################
        # Cost multiplier data.
        ###############################################################
        cost_mult_excel = pd.ExcelFile(os.path.join(settings.INPUT_DIR, 'cost_multipliers.xlsx'))
        self.AC_COST_MULTS = pd.read_excel(cost_mult_excel, "AC_multiplier", index_col="Year")
        self.QC_COST_MULTS = pd.read_excel(cost_mult_excel, "QC_multiplier", index_col="Year")
        self.FOC_COST_MULTS = pd.read_excel(cost_mult_excel, "FOC_multiplier", index_col="Year")
        self.FLC_COST_MULTS = pd.read_excel(cost_mult_excel, "FLC_multiplier", index_col="Year")
        self.FDC_COST_MULTS = pd.read_excel(cost_mult_excel, "FDC_multiplier", index_col="Year")
        self.WP_COST_MULTS = pd.read_excel(cost_mult_excel, "WP_multiplier", index_col="Year")["Water_delivery_price_multiplier"].to_dict()
        self.WATER_LICENSE_COST_MULTS = pd.read_excel(cost_mult_excel, "Water License Cost multiplier", index_col="Year")["Water_license_cost_multiplier"].to_dict()
        self.EST_COST_MULTS = pd.read_excel(cost_mult_excel, "Establishment cost multiplier", index_col="Year")["Establishment_cost_multiplier"].to_dict()
        self.MAINT_COST_MULTS = pd.read_excel(cost_mult_excel, "Maintennance cost multiplier", index_col="Year")["Maintennance_cost_multiplier"].to_dict()
        self.TRANS_COST_MULTS = pd.read_excel(cost_mult_excel, "Transitions cost multiplier", index_col="Year")["Transitions_cost_multiplier"].to_dict()
        self.SAVBURN_COST_MULTS = pd.read_excel(cost_mult_excel, "Savanna burning cost multiplier", index_col="Year")["Savanna_burning_cost_multiplier"].to_dict()
        self.IRRIG_COST_MULTS = pd.read_excel(cost_mult_excel, "Irrigation cost multiplier", index_col="Year")["Irrigation_cost_multiplier"].to_dict()
        self.BECCS_COST_MULTS = pd.read_excel(cost_mult_excel, "BECCS cost multiplier", index_col="Year")["BECCS_cost_multiplier"].to_dict()
        self.BECCS_REV_MULTS = pd.read_excel(cost_mult_excel, "BECCS revenue multiplier", index_col="Year")["BECCS_revenue_multiplier"].to_dict()
        self.FENCE_COST_MULTS = pd.read_excel(cost_mult_excel, "Fencing cost multiplier", index_col="Year")["Fencing_cost_multiplier"].to_dict()



        ###############################################################
        # Spatial layers.
        ###############################################################
        print("\tSetting up spatial layers data...", flush=True)

        # Actual hectares per cell, including projection corrections.
        self.REAL_AREA_NO_RESFACTOR = pd.read_hdf(os.path.join(settings.INPUT_DIR, "real_area.h5")).to_numpy()
        self.REAL_AREA = self.REAL_AREA_NO_RESFACTOR[self.MASK] * self.RESMULT  # TODO: adjusting using 

        # Derive NCELLS (number of spatial cells) from the area array.
        self.NCELLS = self.REAL_AREA.shape[0]
        
        # Initial (2010) ag decision variable (X_mrj).
        self.LMMAP_NO_RESFACTOR = pd.read_hdf(os.path.join(settings.INPUT_DIR, "lmmap.h5")).to_numpy()
        self.AG_L_MRJ = self.get_exact_resfactored_lumap_mrj() 
        self.add_ag_dvars(self.YR_CAL_BASE, self.AG_L_MRJ)

        # Initial (2010) land-use map, mapped as lexicographic land-use class indices.
        self.LU_RESFACTOR_CELLS = pd.DataFrame({
            'lu_code': list(self.DESC2AGLU.values()),
            'res_size': [ceil((self.LUMAP_NO_RESFACTOR == lu_code).sum() / self.RESMULT) for _,lu_code in self.DESC2AGLU.items()]
        }).sort_values('res_size').reset_index(drop=True)
        
        self.LUMAP = self.get_resfactored_lumap() if settings.RESFACTOR > 1 else self.LUMAP_NO_RESFACTOR[self.MASK]
        self.add_lumap(self.YR_CAL_BASE, self.LUMAP)

        # Initial (2010) land management map.
        self.LMMAP = self.LMMAP_NO_RESFACTOR[self.MASK]
        self.add_lmmap(self.YR_CAL_BASE, self.LMMAP)

        # Initial (2010) agricutural management maps - no cells are used for alternative agricultural management options.
        # Includes a separate AM map for each agricultural management option, because they can be stacked.
        self.AG_MAN_DESC = [am for am in settings.AG_MANAGEMENTS if settings.AG_MANAGEMENTS[am]]
        self.AG_MAN_LU_DESC = {am:settings.AG_MANAGEMENTS_TO_LAND_USES[am] for am in self.AG_MAN_DESC}
        self.AG_MAN_MAP = {am: np.zeros(self.NCELLS).astype("int8") for am in self.AG_MAN_DESC}
        self.N_AG_MANS = len(self.AG_MAN_DESC)
        self.add_ammaps(self.YR_CAL_BASE, self.AG_MAN_MAP)

        

        self.NON_AG_L_RK = lumap2non_ag_l_mk(
            self.LUMAP, len(self.NON_AGRICULTURAL_LANDUSES)     # Int8
        )
        self.add_non_ag_dvars(self.YR_CAL_BASE, self.NON_AG_L_RK)

        ###############################################################
        # Climate change impact data.
        ###############################################################
        print("\tLoading climate change data...", flush=True)

        self.CLIMATE_CHANGE_IMPACT = pd.read_hdf(
            os.path.join(settings.INPUT_DIR, "climate_change_impacts_" + settings.RCP + "_CO2_FERT_" + settings.CO2_FERT.upper() + ".h5"), where=self.MASK
        )

        ###############################################################
        # No-Go areas; Regional adoption constraints.
        ###############################################################
        print("\tLoading no-go areas and regional adoption zones...", flush=True)
   
        ##################### No-go areas
        self.NO_GO_LANDUSE_AG = []
        self.NO_GO_LANDUSE_NON_AG = []

        for lu in settings.NO_GO_VECTORS.keys():
            if lu in self.AGRICULTURAL_LANDUSES:
                self.NO_GO_LANDUSE_AG.append(lu)
            elif lu in self.NON_AGRICULTURAL_LANDUSES:
                self.NO_GO_LANDUSE_NON_AG.append(lu)
            else:
                raise KeyError(f"Land use '{lu}' in no-go area vector does not match any land use in the model.")

        no_go_arrs_ag = []
        no_go_arrs_non_ag = []

        for lu, no_go_path in settings.NO_GO_VECTORS.items():
            # Read the no-go area shapefile
            no_go_shp = gpd.read_file(no_go_path)
            # Check if the CRS is defined
            if no_go_shp.crs is None:
                raise ValueError(f"{no_go_path} does not have a CRS defined")
            # Rasterize the reforestation vector; Fill with 1.  0 is no-go, 1 is 'free' cells.
            with rasterio.open(settings.INPUT_DIR + '/NLUM_2010-11_mask.tif') as src:
                src_arr = src.read(1)
                src_meta = src.meta.copy()
                no_go_shp = no_go_shp.to_crs(src_meta['crs'])
                no_go_arr = rasterio.features.rasterize(
                    ((row['geometry'], 0) for _,row in no_go_shp.iterrows()),
                    out_shape=(src_meta['height'], src_meta['width']),
                    transform=src_meta['transform'],
                    fill=1,
                    dtype=np.int16
                )
                # Add the no-go area to the ag or non_ag list.
                if lu in self.NO_GO_LANDUSE_AG:
                    no_go_arrs_ag.append(no_go_arr[np.nonzero(src_arr)].astype(np.bool_))
                elif lu in self.NO_GO_LANDUSE_NON_AG:
                    no_go_arrs_non_ag.append(no_go_arr[np.nonzero(src_arr)].astype(np.bool_))

        self.NO_GO_REGION_AG = np.stack(no_go_arrs_ag, axis=0)[:, self.MASK]
        self.NO_GO_REGION_NON_AG = np.stack(no_go_arrs_non_ag, axis=0)[:, self.MASK]
        

        
        ##################### Regional adoption zones
        if settings.REGIONAL_ADOPTION_CONSTRAINTS != "on":
            self.REGIONAL_ADOPTION_ZONES = None
            self.REGIONAL_ADOPTION_TARGETS = None
        else:
            self.REGIONAL_ADOPTION_ZONES = pd.read_hdf(
                os.path.join(settings.INPUT_DIR, "regional_adoption_zones.h5"), where=self.MASK
            )[settings.REGIONAL_ADOPTION_ZONE].to_numpy()
        
            regional_adoption_targets = pd.read_excel(os.path.join(settings.INPUT_DIR, "regional_adoption_zones.xlsx"), sheet_name=settings.REGIONAL_ADOPTION_ZONE)
            self.REGIONAL_ADOPTION_TARGETS = regional_adoption_targets.iloc[
                [idx for idx, row in regional_adoption_targets.iterrows() if
                    all([row['ADOPTION_PERCENTAGE_2030']>=0, 
                        row['ADOPTION_PERCENTAGE_2050']>=0, 
                        row['ADOPTION_PERCENTAGE_2100']>=0])
                ]
            ]



        ###############################################################
        # Livestock related data.
        ###############################################################
        print("\tLoading livestock related data...", flush=True)

        self.FEED_REQ = np.nan_to_num(
            pd.read_hdf(os.path.join(settings.INPUT_DIR, "feed_req.h5"), where=self.MASK).to_numpy()
        )
        self.PASTURE_KG_DM_HA = pd.read_hdf(
            os.path.join(settings.INPUT_DIR, "pasture_kg_dm_ha.h5"), where=self.MASK
        ).to_numpy()
        self.SAFE_PUR_NATL = pd.read_hdf(
            os.path.join(settings.INPUT_DIR, "safe_pur_natl.h5"), where=self.MASK
        ).to_numpy()
        self.SAFE_PUR_MODL = pd.read_hdf(
            os.path.join(settings.INPUT_DIR, "safe_pur_modl.h5"), where=self.MASK
        ).to_numpy()



        ###############################################################
        # Agricultural management options data.
        ###############################################################
        print("\tLoading agricultural management options' data...", flush=True)

        # Asparagopsis taxiformis data
        asparagopsis_file = os.path.join(settings.INPUT_DIR, "20250415_Bundle_MR.xlsx")
        self.ASPARAGOPSIS_DATA = {}
        self.ASPARAGOPSIS_DATA["Beef - modified land"] = pd.read_excel(
            asparagopsis_file, sheet_name="MR bundle (ext cattle)", index_col="Year"
        )
        self.ASPARAGOPSIS_DATA["Sheep - modified land"] = pd.read_excel(
            asparagopsis_file, sheet_name="MR bundle (sheep)", index_col="Year"
        )
        self.ASPARAGOPSIS_DATA["Dairy - natural land"] = pd.read_excel(
            asparagopsis_file, sheet_name="MR bundle (dairy)", index_col="Year"
        )
        self.ASPARAGOPSIS_DATA["Dairy - modified land"] = self.ASPARAGOPSIS_DATA[
            "Dairy - natural land"
        ]

        # Precision agriculture data
        prec_agr_file = os.path.join(settings.INPUT_DIR, "20231101_Bundle_AgTech_NE.xlsx")
        self.PRECISION_AGRICULTURE_DATA = {}
        int_cropping_data = pd.read_excel(
            prec_agr_file, sheet_name="AgTech NE bundle (int cropping)", index_col="Year"
        )
        cropping_data = pd.read_excel(
            prec_agr_file, sheet_name="AgTech NE bundle (cropping)", index_col="Year"
        )
        horticulture_data = pd.read_excel(
            prec_agr_file, sheet_name="AgTech NE bundle (horticulture)", index_col="Year"
        )

        for lu in [
            "Hay",
            "Summer cereals",
            "Summer legumes",
            "Summer oilseeds",
            "Winter cereals",
            "Winter legumes",
            "Winter oilseeds",
        ]:
            # Cropping land uses
            self.PRECISION_AGRICULTURE_DATA[lu] = cropping_data

        for lu in ["Cotton", "Other non-cereal crops", "Rice", "Sugar", "Vegetables"]:
            # Intensive Cropping land uses
            self.PRECISION_AGRICULTURE_DATA[lu] = int_cropping_data

        for lu in [
            "Apples",
            "Citrus",
            "Grapes",
            "Nuts",
            "Pears",
            "Plantation fruit",
            "Stone fruit",
            "Tropical stone fruit",
        ]:
            # Horticulture land uses
            self.PRECISION_AGRICULTURE_DATA[lu] = horticulture_data

        # Ecological grazing data
        eco_grazing_file = os.path.join(settings.INPUT_DIR, "20231107_ECOGRAZE_Bundle.xlsx")
        self.ECOLOGICAL_GRAZING_DATA = {}
        self.ECOLOGICAL_GRAZING_DATA["Beef - modified land"] = pd.read_excel(
            eco_grazing_file, sheet_name="Ecograze bundle (ext cattle)", index_col="Year"
        )
        self.ECOLOGICAL_GRAZING_DATA["Sheep - modified land"] = pd.read_excel(
            eco_grazing_file, sheet_name="Ecograze bundle (sheep)", index_col="Year"
        )
        self.ECOLOGICAL_GRAZING_DATA["Dairy - modified land"] = pd.read_excel(
            eco_grazing_file, sheet_name="Ecograze bundle (dairy)", index_col="Year"
        )

        # Load soil carbon data, convert C to CO2e (x 44/12), and average over years
        self.SOIL_CARBON_AVG_T_CO2_HA = (
            pd.read_hdf(os.path.join(settings.INPUT_DIR, "soil_carbon_t_ha.h5"), where=self.MASK).to_numpy(dtype=np.float32) 
            * (44 / 12) 
            / settings.SOC_AMORTISATION
        )


        # Load AgTech EI data
        prec_agr_file = os.path.join(settings.INPUT_DIR, '20231107_Bundle_AgTech_EI.xlsx')
        self.AGTECH_EI_DATA = {}
        int_cropping_data = pd.read_excel( prec_agr_file, sheet_name='AgTech EI bundle (int cropping)', index_col='Year' )
        cropping_data = pd.read_excel( prec_agr_file, sheet_name='AgTech EI bundle (cropping)', index_col='Year' )
        horticulture_data = pd.read_excel( prec_agr_file, sheet_name='AgTech EI bundle (horticulture)', index_col='Year' )

        for lu in ['Hay', 'Summer cereals', 'Summer legumes', 'Summer oilseeds',
                'Winter cereals', 'Winter legumes', 'Winter oilseeds']:
            # Cropping land uses
            self.AGTECH_EI_DATA[lu] = cropping_data

        for lu in ['Cotton', 'Other non-cereal crops', 'Rice', 'Sugar', 'Vegetables']:
            # Intensive Cropping land uses
            self.AGTECH_EI_DATA[lu] = int_cropping_data

        for lu in ['Apples', 'Citrus', 'Grapes', 'Nuts', 'Pears',
                'Plantation fruit', 'Stone fruit', 'Tropical stone fruit']:
            # Horticulture land uses
            self.AGTECH_EI_DATA[lu] = horticulture_data

        # Load BioChar data
        biochar_file = os.path.join(settings.INPUT_DIR, '20240918_Bundle_BC.xlsx')
        self.BIOCHAR_DATA = {}
        cropping_data = pd.read_excel(biochar_file, sheet_name='Biochar (cropping)', index_col='Year' )
        horticulture_data = pd.read_excel(biochar_file, sheet_name='Biochar (horticulture)', index_col='Year' )

        for lu in ['Hay', 'Summer cereals', 'Summer legumes', 'Summer oilseeds',
                'Winter cereals', 'Winter legumes', 'Winter oilseeds']:
            # Cropping land uses
            self.BIOCHAR_DATA[lu] = cropping_data

        for lu in ['Apples', 'Citrus', 'Grapes', 'Nuts', 'Pears',
                'Plantation fruit', 'Stone fruit', 'Tropical stone fruit']:
            # Horticulture land uses
            self.BIOCHAR_DATA[lu] = horticulture_data



        ###############################################################
        # Productivity data.
        ###############################################################
        print("\tLoading productivity data...", flush=True)

        # Yield increases.
        fpath = os.path.join(settings.INPUT_DIR, "yieldincreases_bau2022.csv")
        self.BAU_PROD_INCR = pd.read_csv(fpath, header=[0, 1]).astype(np.float32)




        ###############################################################
        # Auxiliary Spatial Layers
        # (spatial layers not required for production calculation)
        ###############################################################
        print("\tLoading auxiliary spatial layers data...", flush=True)

        # Load stream length data in metres of stream per cell
        self.STREAM_LENGTH = pd.read_hdf(
            os.path.join(settings.INPUT_DIR, "stream_length_m_cell.h5"), where=self.MASK
        ).to_numpy()

        # Calculate the proportion of the area of each cell within stream buffer (convert REAL_AREA from ha to m2 and divide m2 by m2)
        self.RP_PROPORTION =  (
            (2 * settings.RIPARIAN_PLANTING_BUFFER_WIDTH * self.STREAM_LENGTH) / (self.REAL_AREA_NO_RESFACTOR[self.MASK] * 10000)
        ).astype(np.float32)
        # Calculate the length of fencing required for each cell in per hectare terms for riparian plantings
        self.RP_FENCING_LENGTH = (
            (2 * settings.RIPARIAN_PLANTING_TORTUOSITY_FACTOR * self.STREAM_LENGTH) / self.REAL_AREA_NO_RESFACTOR[self.MASK]
        ).astype(np.float32)



        ###############################################################
        # Additional agricultural economic data.
        ###############################################################
        print("\tLoading additional agricultural economic data...", flush=True)


        # Load greenhouse gas emissions from agriculture
        self.AGGHG_CROPS = pd.read_hdf(os.path.join(settings.INPUT_DIR, "agGHG_crops.h5"), where=self.MASK)
        self.AGGHG_LVSTK = pd.read_hdf(os.path.join(settings.INPUT_DIR, "agGHG_lvstk.h5"), where=self.MASK)
        self.AGGHG_IRRPAST = pd.read_hdf(os.path.join(settings.INPUT_DIR, "agGHG_irrpast.h5"), where=self.MASK)


        # Raw transition cost matrix. In AUD/ha and ordered lexicographically.
        self.AG_TMATRIX = np.load(os.path.join(settings.INPUT_DIR, "ag_tmatrix.npy"))
        self.AG_TO_DESTOCKED_NATURAL_COSTS_HA = np.load(os.path.join(settings.INPUT_DIR, "ag_to_destock_tmatrix.npy"))
        
  
        # Boolean x_mrj matrix with allowed land uses j for each cell r under lm.
        self.EXCLUDE = np.load(os.path.join(settings.INPUT_DIR, "x_mrj.npy"))
        self.EXCLUDE = self.EXCLUDE[:, self.MASK, :]  # Apply resfactor specially for the exclude matrix



        ###############################################################
        # Non-agricultural data.
        ###############################################################
        print("\tLoading non-agricultural data...", flush=True)

        # Load plantings economic data
        self.EP_EST_COST_HA = pd.read_hdf(os.path.join(settings.INPUT_DIR, "ep_est_cost_ha.h5"), where=self.MASK).to_numpy(dtype=np.float32)
        self.RP_EST_COST_HA = self.EP_EST_COST_HA.copy()  # Riparian plantings have the same establishment cost as environmental plantings
        self.AF_EST_COST_HA = self.EP_EST_COST_HA.copy()  # Agroforestry plantings have the same establishment cost as environmental plantings
        self.CP_EST_COST_HA = pd.read_hdf(os.path.join(settings.INPUT_DIR, "cp_est_cost_ha.h5"), where=self.MASK).to_numpy(dtype=np.float32)

        # Load fire risk data (reduced carbon sequestration by this amount)
        fr_df = pd.read_hdf(os.path.join(settings.INPUT_DIR, "fire_risk.h5"), where=self.MASK)
        fr_dict = {"low": "FD_RISK_PERC_5TH", "med": "FD_RISK_MEDIAN", "high": "FD_RISK_PERC_95TH"}
        fire_risk = fr_df[fr_dict[settings.FIRE_RISK]]

        # Load environmental plantings (block) GHG sequestration (aboveground carbon discounted by settings.RISK_OF_REVERSAL and settings.FIRE_RISK)
        ep_df = pd.read_hdf(os.path.join(settings.INPUT_DIR, "ep_block_avg_t_co2_ha_yr.h5"), where=self.MASK)
        self.EP_BLOCK_AVG_T_CO2_HA = (
            ep_df.EP_BLOCK_AG_AVG_T_CO2_HA_YR * (fire_risk / 100) * (1 - settings.RISK_OF_REVERSAL)
            + ep_df.EP_BLOCK_BG_AVG_T_CO2_HA_YR
        ).to_numpy(dtype=np.float32)


        # Load environmental plantings (belt) GHG sequestration (aboveground carbon discounted by settings.RISK_OF_REVERSAL and settings.FIRE_RISK)
        ep_df = pd.read_hdf(os.path.join(settings.INPUT_DIR, "ep_belt_avg_t_co2_ha_yr.h5"), where=self.MASK)
        self.EP_BELT_AVG_T_CO2_HA = (
            (ep_df.EP_BELT_AG_AVG_T_CO2_HA_YR * (fire_risk / 100) * (1 - settings.RISK_OF_REVERSAL))
            + ep_df.EP_BELT_BG_AVG_T_CO2_HA_YR
        ).to_numpy(dtype=np.float32)

        # Load environmental plantings (riparian) GHG sequestration (aboveground carbon discounted by settings.RISK_OF_REVERSAL and settings.FIRE_RISK)
        ep_df = pd.read_hdf(os.path.join(settings.INPUT_DIR, "ep_rip_avg_t_co2_ha_yr.h5"), where=self.MASK)
        self.EP_RIP_AVG_T_CO2_HA = (
            (ep_df.EP_RIP_AG_AVG_T_CO2_HA_YR * (fire_risk / 100) * (1 - settings.RISK_OF_REVERSAL))
            + ep_df.EP_RIP_BG_AVG_T_CO2_HA_YR
        ).to_numpy(dtype=np.float32)

        # Load carbon plantings (block) GHG sequestration (aboveground carbon discounted by settings.RISK_OF_REVERSAL and settings.FIRE_RISK)
        cp_df = pd.read_hdf(os.path.join(settings.INPUT_DIR, "cp_block_avg_t_co2_ha_yr.h5"), where=self.MASK)
        self.CP_BLOCK_AVG_T_CO2_HA = (
            (cp_df.CP_BLOCK_AG_AVG_T_CO2_HA_YR * (fire_risk / 100) * (1 - settings.RISK_OF_REVERSAL))
            + cp_df.CP_BLOCK_BG_AVG_T_CO2_HA_YR
        ).to_numpy(dtype=np.float32)


        # Load farm forestry [i.e. carbon plantings (belt)] GHG sequestration (aboveground carbon discounted by settings.RISK_OF_REVERSAL and settings.FIRE_RISK)
        cp_df = pd.read_hdf(os.path.join(settings.INPUT_DIR, "cp_belt_avg_t_co2_ha_yr.h5"), where=self.MASK)
        self.CP_BELT_AVG_T_CO2_HA = (
            (cp_df.CP_BELT_AG_AVG_T_CO2_HA_YR * (fire_risk / 100) * (1 - settings.RISK_OF_REVERSAL))
            + cp_df.CP_BELT_BG_AVG_T_CO2_HA_YR
        ).to_numpy(dtype=np.float32)

        # Agricultural land use to plantings raw transition costs:
        self.AG2EP_TRANSITION_COSTS_HA = np.load(
            os.path.join(settings.INPUT_DIR, "ag_to_ep_tmatrix.npy")
        )  # shape: (28,)

        # EP to agricultural land use transition costs:
        self.EP2AG_TRANSITION_COSTS_HA = np.load(
            os.path.join(settings.INPUT_DIR, "ep_to_ag_tmatrix.npy")
        )  # shape: (28,)
        
        
        ##############################################################
        # Transition cost for all land use
        #############################################################
        
        # Transition matrix from ag
        tmat_ag2ag_xr = xr.DataArray(
            self.AG_TMATRIX,
            dims=['from_lu','to_lu'],
            coords={'from_lu':self.AGRICULTURAL_LANDUSES, 'to_lu':self.AGRICULTURAL_LANDUSES }
        )
        tmat_ag2non_ag_xr = xr.DataArray(
            np.repeat(self.AG2EP_TRANSITION_COSTS_HA.reshape(-1,1), len(self.NON_AGRICULTURAL_LANDUSES), axis=1),
            dims=['from_lu','to_lu'],
            coords={'from_lu':self.AGRICULTURAL_LANDUSES, 'to_lu':self.NON_AGRICULTURAL_LANDUSES}
        )
        tmat_from_ag_xr = xr.concat([tmat_ag2ag_xr, tmat_ag2non_ag_xr], dim='to_lu')                        # Combine ag2ag and ag2non-ag
        tmat_from_ag_xr.loc[:,'Destocked - natural land'] = self.AG_TO_DESTOCKED_NATURAL_COSTS_HA           # Ag to Destock-natural has its own values
        
        
        # Transition matrix of non-ag to unallocated-modified land (land clearing)
        tmat_wood_clear = np.load(os.path.join(settings.INPUT_DIR, 'transition_cost_clearing_forest.npz'))
        
        tmat_clear_EP = tmat_wood_clear['tmat_clear_wood_barrier'] + tmat_wood_clear['tmat_clear_dense_wood']
        tmat_clear_RP = tmat_wood_clear['tmat_clear_wood_barrier'] + tmat_wood_clear['tmat_clear_dense_wood']
        tmat_clear_sheep_ag_forest = (tmat_wood_clear['tmat_clear_wood_barrier'] + tmat_wood_clear['tmat_clear_dense_wood']) * settings.AF_PROPORTION
        tmat_clear_beef_ag_forest = (tmat_wood_clear['tmat_clear_wood_barrier'] + tmat_wood_clear['tmat_clear_dense_wood']) * settings.AF_PROPORTION
        tmat_clear_CP = tmat_wood_clear['tmat_clear_wood_barrier'] + tmat_wood_clear['tmat_clear_dense_wood']
        tmat_clear_sheep_CP = (tmat_wood_clear['tmat_clear_wood_barrier'] + tmat_wood_clear['tmat_clear_dense_wood']) * settings.CP_BELT_PROPORTION
        tmat_clear_beef_CP = (tmat_wood_clear['tmat_clear_wood_barrier'] + tmat_wood_clear['tmat_clear_dense_wood']) * settings.CP_BELT_PROPORTION
        tmat_clear_BECCS = tmat_wood_clear['tmat_clear_wood_barrier'] + tmat_wood_clear['tmat_clear_dense_wood']
        tmat_clear_destocked_nat = tmat_wood_clear['tmat_clear_light_wood'] + tmat_wood_clear['tmat_clear_dense_wood']
        
        tmat_costs = np.array([
            tmat_clear_EP, tmat_clear_RP, tmat_clear_sheep_ag_forest, tmat_clear_beef_ag_forest,
            tmat_clear_CP, tmat_clear_sheep_CP, tmat_clear_beef_CP, tmat_clear_BECCS, tmat_clear_destocked_nat
        ]).T
        
        
        
        
        # Transition matrix from non-ag
        tmat_non_ag2ag_xr = xr.DataArray(
            np.repeat(self.EP2AG_TRANSITION_COSTS_HA.reshape(1,-1), len(self.NON_AGRICULTURAL_LANDUSES), axis=0),
            dims=['from_lu','to_lu'],
            coords={'from_lu':self.NON_AGRICULTURAL_LANDUSES, 'to_lu':self.AGRICULTURAL_LANDUSES }
        )
        tmat_non_ag2non_ag_xr = xr.DataArray(
            np.full((len(self.NON_AGRICULTURAL_LANDUSES), len(self.NON_AGRICULTURAL_LANDUSES)), np.nan),
            dims=['from_lu','to_lu'],
            coords={'from_lu':self.NON_AGRICULTURAL_LANDUSES, 'to_lu':self.NON_AGRICULTURAL_LANDUSES }
        )


        np.fill_diagonal(tmat_non_ag2non_ag_xr.values, 0)                                                   # Lu staty the same has 0 cost
        tmat_from_non_ag_xr = xr.concat([tmat_non_ag2ag_xr, tmat_non_ag2non_ag_xr], dim='to_lu')            # Combine non-ag2ag and non-ag2non-ag
        tmat_from_non_ag_xr.loc['Destocked - natural land', 'Unallocated - natural land'] = np.nan          # Destocked-natural can not transit to unallow-natural
        
   
        # Get the full transition cost matrix
        self.T_MAT = xr.concat([tmat_from_ag_xr, tmat_from_non_ag_xr], dim='from_lu')
        self.T_MAT.loc[self.NON_AGRICULTURAL_LANDUSES, [self.AGLU2DESC[i] for i in self.LU_NATURAL]] = np.nan       # non-ag2natural is not allowed
        self.T_MAT.loc[self.NON_AGRICULTURAL_LANDUSES, 'Unallocated - modified land'] = tmat_costs                  # Clearing non-ag land requires such cost
        self.T_MAT.loc['Destocked - natural land', self.LU_LVSTK_NATURAL_DESC] = self.T_MAT.loc['Unallocated - natural land', self.LU_LVSTK_NATURAL_DESC]   # Destocked-natural transits to LVSTK-natural has the same cost as unallocated-natural to LVSTK-natural


        # tools.plot_t_mat(self.T_MAT)
        
        

        ###############################################################
        # Water data.
        ###############################################################
        print("\tLoading water data...", flush=True)
        
        # Initialize water constraints to avoid recalculating them every time.
        self.WATER_YIELD_LIMITS = None

        # Water requirements by land use -- LVSTK.
        wreq_lvstk_dry = pd.DataFrame()
        wreq_lvstk_irr = pd.DataFrame()

        # The rj-indexed arrays have zeroes where j is not livestock.
        for lu in self.AGRICULTURAL_LANDUSES:
            if lu in self.LU_LVSTK:
                # First find out which animal is involved.
                animal, _ = ag_quantity.lvs_veg_types(lu)
                # Water requirements per head are for drinking and irrigation.
                wreq_lvstk_dry[lu] = self.AGEC_LVSTK["WR_DRN", animal] * settings.LIVESTOCK_DRINKING_WATER
                wreq_lvstk_irr[lu] = (
                    self.AGEC_LVSTK["WR_IRR", animal] + self.AGEC_LVSTK["WR_DRN", animal] * settings.LIVESTOCK_DRINKING_WATER
                )
            else:
                wreq_lvstk_dry[lu] = 0.0
                wreq_lvstk_irr[lu] = 0.0

        # Water requirements by land use -- CROPS.
        wreq_crops_irr = pd.DataFrame()

        # The rj-indexed arrays have zeroes where j is not a crop.
        for lu in self.AGRICULTURAL_LANDUSES:
            if lu in self.LU_CROPS:
                wreq_crops_irr[lu] = self.AGEC_CROPS["WR", "irr", lu]
            else:
                wreq_crops_irr[lu] = 0.0

        # Add together as they have nans where not lvstk/crops
        self.WREQ_DRY_RJ = np.nan_to_num(wreq_lvstk_dry.to_numpy(dtype=np.float32))
        self.WREQ_IRR_RJ = np.nan_to_num(wreq_crops_irr.to_numpy(dtype=np.float32)) + np.nan_to_num(
            wreq_lvstk_irr.to_numpy(dtype=np.float32)
        )

        # Spatially explicit costs of a water licence per ML.
        self.WATER_LICENCE_PRICE = np.nan_to_num(
                pd.read_hdf(os.path.join(settings.INPUT_DIR, "water_licence_price.h5"), where=self.MASK).to_numpy()
            )

        # Spatially explicit costs of water delivery per ML.
        self.WATER_DELIVERY_PRICE = np.nan_to_num(
                pd.read_hdf(os.path.join(settings.INPUT_DIR, "water_delivery_price.h5"), where=self.MASK).to_numpy()
            )
       

        # River regions.
        self.RIVREG_ID = pd.read_hdf(os.path.join(settings.INPUT_DIR, "rivreg_id.h5"), where=self.MASK).to_numpy()  # River region ID mapped.
 
        rr = pd.read_hdf(os.path.join(settings.INPUT_DIR, "rivreg_lut.h5"))
        self.RIVREG_DICT = dict(
            zip(rr.HR_RIVREG_ID, rr.HR_RIVREG_NAME)
        )  # River region ID to Name lookup table
        self.RIVREG_HIST_LEVEL = dict(
            zip(rr.HR_RIVREG_ID, rr.WATER_YIELD_HIST_BASELINE_ML)
        )  # River region ID and water use limits

        # Drainage divisions
        self.DRAINDIV_ID = pd.read_hdf(os.path.join(settings.INPUT_DIR, "draindiv_id.h5"), where=self.MASK).to_numpy()  # Drainage div ID mapped.

        dd = pd.read_hdf(os.path.join(settings.INPUT_DIR, "draindiv_lut.h5"))
        self.DRAINDIV_DICT = dict(
            zip(dd.HR_DRAINDIV_ID, dd.HR_DRAINDIV_NAME)
        )  # Drainage div ID to Name lookup table
        self.DRAINDIV_HIST_LEVEL = dict(
            zip(dd.HR_DRAINDIV_ID, dd.WATER_YIELD_HIST_BASELINE_ML)
        )  # Drainage div ID and water use limits


        # Water yields -- run off from a cell into catchment by deep-rooted, shallow-rooted, and natural land
        water_yield_baselines = pd.read_hdf(os.path.join(settings.INPUT_DIR, "water_yield_baselines.h5"), where=self.MASK)
        self.WATER_YIELD_HIST_DR = water_yield_baselines['WATER_YIELD_HIST_DR_ML_HA'].to_numpy(dtype = np.float32)
        self.WATER_YIELD_HIST_SR = water_yield_baselines["WATER_YIELD_HIST_SR_ML_HA"].to_numpy(dtype = np.float32)
        self.DEEP_ROOTED_PROPORTION = water_yield_baselines['DEEP_ROOTED_PROPORTION'].to_numpy(dtype = np.float32)
        self.WATER_YIELD_HIST_NL = water_yield_baselines.eval(
            'WATER_YIELD_HIST_DR_ML_HA * DEEP_ROOTED_PROPORTION + WATER_YIELD_HIST_SR_ML_HA * (1 - DEEP_ROOTED_PROPORTION)'
        ).to_numpy(dtype = np.float32)

        wyield_fname_dr = os.path.join(settings.INPUT_DIR, 'water_yield_ssp' + str(settings.SSP) + '_2010-2100_dr_ml_ha.h5')
        wyield_fname_sr = os.path.join(settings.INPUT_DIR, 'water_yield_ssp' + str(settings.SSP) + '_2010-2100_sr_ml_ha.h5')
        
        # Read water yield data
        self.WATER_YIELD_DR_FILE = pd.read_hdf(wyield_fname_dr, where=self.MASK).T.values
        self.WATER_YIELD_SR_FILE = pd.read_hdf(wyield_fname_sr, where=self.MASK).T.values
        

        # Water yield from outside LUTO study area.
        water_yield_oustide_luto_hist = pd.read_hdf(os.path.join(settings.INPUT_DIR, 'water_yield_outside_LUTO_study_area_hist_1970_2000.h5'))
        
        if settings.WATER_REGION_DEF == 'River Region':
            rr_outside_luto = pd.read_hdf(os.path.join(settings.INPUT_DIR, 'water_yield_outside_LUTO_study_area_2010_2100_rr_ml.h5'))
            rr_outside_luto = rr_outside_luto.loc[:, pd.IndexSlice[:, settings.SSP]]
            rr_outside_luto.columns = rr_outside_luto.columns.droplevel('ssp')

            rr_natural_land = pd.read_hdf(os.path.join(settings.INPUT_DIR, 'water_yield_natural_land_2010_2100_rr_ml.h5'))
            rr_natural_land = rr_natural_land.loc[:, pd.IndexSlice[:, settings.SSP]]
            rr_natural_land.columns = rr_natural_land.columns.droplevel('ssp')

            self.WATER_OUTSIDE_LUTO_RR = rr_outside_luto
            self.WATER_OUTSIDE_LUTO_RR_HIST = water_yield_oustide_luto_hist.query('Region_Type == "River Region"').set_index('Region_ID')['Water Yield (ML)'].to_dict()
            self.WATER_UNDER_NATURAL_LAND_RR = rr_natural_land

        if settings.WATER_REGION_DEF == 'Drainage Division':
            dd_outside_luto = pd.read_hdf(os.path.join(settings.INPUT_DIR, 'water_yield_outside_LUTO_study_area_2010_2100_dd_ml.h5'))
            dd_outside_luto = dd_outside_luto.loc[:, pd.IndexSlice[:, settings.SSP]]
            dd_outside_luto.columns = dd_outside_luto.columns.droplevel('ssp')

            dd_natural_land = pd.read_hdf(os.path.join(settings.INPUT_DIR, 'water_yield_natural_land_2010_2100_dd_ml.h5'))
            dd_natural_land = dd_natural_land.loc[:, pd.IndexSlice[:, settings.SSP]]
            dd_natural_land.columns = dd_natural_land.columns.droplevel('ssp')

            self.WATER_OUTSIDE_LUTO_DD = dd_outside_luto
            self.WATER_OUTSIDE_LUTO_DD_HIST = water_yield_oustide_luto_hist.query('Region_Type == "Drainage Division"').set_index('Region_ID')['Water Yield (ML)'].to_dict()
            self.WATER_UNDER_NATURAL_LAND_DD = dd_natural_land
        
        
        # Get historical yields of regions
        if settings.WATER_REGION_DEF == 'River Region':
            self.WATER_REGION_NAMES = self.RIVREG_DICT
            self.WATER_REGION_HIST_LEVEL = self.RIVREG_HIST_LEVEL
            self.WATER_REGION_ID = self.RIVREG_ID
            
        elif settings.WATER_REGION_DEF == 'Drainage Division':
            self.WATER_REGION_NAMES = self.DRAINDIV_DICT
            self.WATER_REGION_HIST_LEVEL = self.DRAINDIV_HIST_LEVEL
            self.WATER_REGION_ID = self.DRAINDIV_ID
            

        # Get the water region index for each region
        self.WATER_REGION_INDEX_R = {k:(self.WATER_REGION_ID == k) for k in self.WATER_REGION_NAMES.keys()}


        # Place holder for Water Yield to avoid recalculating it every time.
        self.water_yield_regions_BASE_YR = None
        
        # Water use for domestic and industrial sectors.
        water_use_domestic = pd.read_csv(os.path.join(settings.INPUT_DIR, "Water_Use_Domestic.csv")).query('REGION_TYPE == @settings.WATER_REGION_DEF')
        self.WATER_USE_DOMESTIC = water_use_domestic.set_index('REGION_ID')['DOMESTIC_INDUSTRIAL_WATER_USE_ML'].to_dict()
        
        

        
        ###############################################################
        # Carbon sequestration by natural lands.
        ###############################################################
        print("\tLoading carbon sequestration by natural lands data...", flush=True)

        '''
        ['NATURAL_LAND_AGB_TCO2_HA']
            CO2 in aboveground living biomass in natural land i.e., the part impacted by livestock 
        ['NATURAL_LAND_AGB_DEBRIS_TCO2_HA']
            CO2 in aboveground living biomass and debris in natural land i.e., the part impacted by fire
        ['NATURAL_LAND_TREES_DEBRIS_SOIL_TCO2_HA']
            CO2 in aboveground living biomass and debris and soil in natural land i.e., the part impacted by land clearance
        '''
    
        # Load the natural land carbon data.
        nat_land_CO2 = pd.read_hdf(os.path.join(settings.INPUT_DIR, "natural_land_t_co2_ha.h5"), where=self.MASK)
        
        # Get the carbon stock of unallowcated natural land
        self.CO2E_STOCK_UNALL_NATURAL = np.array(
            nat_land_CO2['NATURAL_LAND_TREES_DEBRIS_SOIL_TCO2_HA'] - (nat_land_CO2['NATURAL_LAND_AGB_DEBRIS_TCO2_HA'] * (100 - fire_risk).to_numpy() / 100),  # everyting minus the fire DAMAGE
        )
        
        
        ###############################################################
        # Calculate base year production 
        ###############################################################

        self.AG_MAN_L_MRJ_DICT = get_base_am_vars(self.NCELLS, self.NLMS, self.N_AG_LUS)
        self.add_ag_man_dvars(self.YR_CAL_BASE, self.AG_MAN_L_MRJ_DICT)
        
        print(f"\tCalculating base year productivity...", flush=True)
        yr_cal_base_prod_data = self.get_production(self.YR_CAL_BASE, self.LUMAP, self.LMMAP)        
        self.add_production_data(self.YR_CAL_BASE, "Production", yr_cal_base_prod_data)
        
        
        
        # Place holders for base year values; will be filled in the input_data module.
        self.BASE_YR_economic_value = None
        self.BASE_YR_production_t = yr_cal_base_prod_data
        self.BASE_YR_GHG_t = None
        self.BASE_YR_water_ML = None
        self.BASE_YR_overall_bio_value = None
        self.BASE_YR_GBF2_score = None

        ###############################################################
        # Demand data.
        ###############################################################
        print("\tLoading demand data...", flush=True)

        # Load demand data (actual production (tonnes, ML) by commodity) - from demand model
        dd = pd.read_hdf(os.path.join(settings.INPUT_DIR, 'demand_projections.h5'))

        # Select the demand data under the running scenariobbryan-January
        self.DEMAND_DATA = dd.loc[(settings.SCENARIO,
                                   settings.DIET_DOM,
                                   settings.DIET_GLOB,
                                   settings.CONVERGENCE,
                                   settings.IMPORT_TREND,
                                   settings.WASTE,
                                   settings.FEED_EFFICIENCY)].copy()

        # Convert eggs from count to tonnes
        self.DEMAND_DATA.loc['eggs'] = self.DEMAND_DATA.loc['eggs'] * settings.EGGS_AVG_WEIGHT / 1000 / 1000

        # Get the off-land commodities
        self.DEMAND_OFFLAND = self.DEMAND_DATA.loc[self.DEMAND_DATA.query("COMMODITY in @settings.OFF_LAND_COMMODITIES").index, 'PRODUCTION'].copy()

        # Remove off-land commodities
        self.DEMAND_C = self.DEMAND_DATA.loc[self.DEMAND_DATA.query("COMMODITY not in @settings.OFF_LAND_COMMODITIES").index, 'PRODUCTION'].copy()

        # Convert to numpy array of shape (91, 26)
        self.D_CY = self.DEMAND_C.to_numpy(dtype = np.float32).T
        
        # Adjust demand data to the production data calculated using the base year layers;
        # The mismatch is caused by resfactoring spatial layers. Land uses of small size (i.e., other non-cereal crops) 
        # are distorted more under higher resfactoring.
        self.D_CY *= (yr_cal_base_prod_data / self.D_CY[0])[None, :]


        ###############################################################
        # Carbon emissions from off-land commodities.
        ###############################################################
        print("\tLoading off-land commodities' carbon emissions data...", flush=True)

        # Read the greenhouse gas intensity data
        off_land_ghg_intensity = pd.read_csv(f'{settings.INPUT_DIR}/agGHG_lvstk_off_land.csv')
        # Split the Emission Source column into two columns
        off_land_ghg_intensity[['Emission Type', 'Emission Source']] = off_land_ghg_intensity['Emission Source'].str.extract(r'^(.*?)\s*\((.*?)\)')

        # Get the emissions from the off-land commodities
        demand_offland_long = self.DEMAND_OFFLAND.stack().reset_index()
        demand_offland_long = demand_offland_long.rename(columns={ 0: 'DEMAND (tonnes)'})

        # Merge the demand and GHG intensity, and calculate the total GHG emissions
        off_land_ghg_emissions = demand_offland_long.merge(off_land_ghg_intensity, on='COMMODITY')
        off_land_ghg_emissions['Total GHG Emissions (tCO2e)'] = off_land_ghg_emissions.eval('`DEMAND (tonnes)` * `Emission Intensity [ kg CO2eq / kg ]`')

        # Keep only the relevant columns
        self.OFF_LAND_GHG_EMISSION = off_land_ghg_emissions[['YEAR',
                                                             'COMMODITY',
                                                             'Emission Type',
                                                             'Emission Source',
                                                             'Total GHG Emissions (tCO2e)']]

        # Get the GHG constraints for luto, shape is (91, 1)
        self.OFF_LAND_GHG_EMISSION_C = self.OFF_LAND_GHG_EMISSION.groupby(['YEAR']).sum(numeric_only=True).values

        # Read the carbon price per tonne over the years (indexed by the relevant year)
        if settings.CARBON_PRICES_FIELD == 'CONSTANT':
            self.CARBON_PRICES = {yr: settings.CARBON_PRICE_COSTANT for yr in range(2010, 2101)}
        else:
            carbon_price_sheet = settings.CARBON_PRICES_FIELD or "Default"
            carbon_price_usecols = "A,B"
            carbon_price_col_names = ["Year", "Carbon_price_$_tCO2e"]
            carbon_price_sheet_index_col = "Year" # if carbon_price_sheet != "Default" else 0
            carbon_price_sheet_header = 0         # if carbon_price_sheet != "Default" else None

            self.CARBON_PRICES: dict[int, float] = pd.read_excel(
                os.path.join(settings.INPUT_DIR, 'carbon_prices.xlsx'),
                sheet_name=carbon_price_sheet,
                usecols=carbon_price_usecols,
                names=carbon_price_col_names,
                header=carbon_price_sheet_header,
                index_col=carbon_price_sheet_index_col,
            )["Carbon_price_$_tCO2e"].to_dict()
            


        ###############################################################
        # GHG targets data.
        ###############################################################
        print("\tLoading GHG targets data...", flush=True)
        if settings.GHG_EMISSIONS_LIMITS != 'off':
            self.GHG_TARGETS = pd.read_excel(
                os.path.join(settings.INPUT_DIR, "GHG_targets.xlsx"), sheet_name="Data", index_col="YEAR"
            )
            self.GHG_TARGETS = self.GHG_TARGETS[settings.GHG_TARGETS_DICT[settings.GHG_EMISSIONS_LIMITS]].to_dict()
            self.GHG_TARGETS.update({k: v * float(settings.GHG_percent) for k, v in self.GHG_TARGETS.items()})


        ###############################################################
        # Savanna burning data.
        ###############################################################
        print("\tLoading savanna burning data...", flush=True)

        # Read in the dataframe
        savburn_df = pd.read_hdf(os.path.join(settings.INPUT_DIR, 'cell_savanna_burning.h5'), where=self.MASK)

        # Load the columns as numpy arrays
        self.SAVBURN_ELIGIBLE =  savburn_df.ELIGIBLE_AREA.to_numpy()                    # 1 = areas eligible for early dry season savanna burning under the ERF, 0 = ineligible          
        self.SAVBURN_TOTAL_TCO2E_HA = savburn_df.AEA_TOTAL_TCO2E_HA.to_numpy()
        
        # # Avoided emissions from savanna burning
        # self.SAVBURN_AVEM_CH4_TCO2E_HA = savburn_df.SAV_AVEM_CH4_TCO2E_HA.to_numpy()  # Avoided emissions - methane
        # self.SAVBURN_AVEM_N2O_TCO2E_HA = savburn_df.SAV_AVEM_N2O_TCO2E_HA.to_numpy()  # Avoided emissions - nitrous oxide
        # self.SAVBURN_SEQ_CO2_TCO2E_HA = savburn_df.SAV_SEQ_CO2_TCO2E_HA.to_numpy()    # Additional carbon sequestration - carbon dioxide

        # Cost per hectare in dollars from settings
        self.SAVBURN_COST_HA = settings.SAVBURN_COST_HA_YR



        ###############################################################
        # Biodiversity priority conservation data. (GBF Target 2)
        ###############################################################
        
        print("\tLoading biodiversity data...", flush=True)
        """
        Kunming-Montreal Biodiversity Framework Target 2: Restore 30% of all Degraded Ecosystems
        Ensure that by 2030 at least 30 per cent of areas of degraded terrestrial, inland water, and coastal and marine ecosystems are under effective restoration,
        in order to enhance biodiversity and ecosystem functions and services, ecological integrity and connectivity.
        """

        biodiv_raw = pd.read_hdf(os.path.join(settings.INPUT_DIR, 'bio_OVERALL_PRIORITY_RANK_AND_AREA_CONNECTIVITY.h5'), where=self.MASK)
        biodiv_contribution_lookup = pd.read_csv(os.path.join(settings.INPUT_DIR, 'bio_OVERALL_CONTRIBUTION_OF_LANDUSES.csv'))                              
        

        # ------------- Biodiversity priority scores for maximising overall biodiversity conservation in Australia ----------------------------
        
        # Get connectivity score
        match settings.CONNECTIVITY_SOURCE:
            case 'NCI':
                connectivity_score = biodiv_raw['DCCEEW_NCI'].to_numpy(dtype=np.float32)
                connectivity_score = np.interp( biodiv_raw['DCCEEW_NCI'], (connectivity_score.min(), connectivity_score.max()), (settings.CONNECTIVITY_LB, 1)).astype('float32')
            case 'DWI':
                connectivity_score = biodiv_raw['NATURAL_AREA_CONNECTIVITY'].to_numpy(dtype=np.float32)
                connectivity_score = np.interp(connectivity_score, (connectivity_score.min(), connectivity_score.max()), (1, settings.CONNECTIVITY_LB)).astype('float32')
            case 'NONE':
                connectivity_score = np.ones(self.NCELLS, dtype=np.float32)
            case _:
                raise ValueError(f"Invalid connectivity source: {settings.CONNECTIVITY_SOURCE}, must be 'NCI', 'DWI' or 'NONE'")
            
        self.CONNECTIVITY_SCORE = connectivity_score

        # Get the HCAS contribution scale (0-1)
        match settings.HABITAT_CONDITION:
            case 10 | 25 | 50 | 75 | 90:
                bio_HCAS_contribution_lookup = biodiv_contribution_lookup.set_index('lu')[f'PERCENTILE_{settings.HABITAT_CONDITION}'].to_dict()         # Get the biodiversity degradation score at specified percentile (pd.DataFrame)
                unallow_nat_scale = bio_HCAS_contribution_lookup[self.DESC2AGLU['Unallocated - natural land']]                                          # Get the biodiversity degradation score for unallocated natural land (float)
                bio_HCAS_contribution_lookup = {int(k): v * (1 / unallow_nat_scale) for k, v in bio_HCAS_contribution_lookup.items()}                   # Normalise the biodiversity degradation score to the unallocated natural land score
            case 'USER_DEFINED':
                bio_HCAS_contribution_lookup = biodiv_contribution_lookup.set_index('lu')['USER_DEFINED'].to_dict()
            case _:
                print(f"WARNING!! Invalid habitat condition source: {settings.HABITAT_CONDITION}, must be one of [10, 25, 50, 75, 90], or 'USER_DEFINED'")
        
        self.BIO_HABITAT_CONTRIBUTION_LOOK_UP = {j: round(x, settings.ROUND_DECMIALS) for j, x in bio_HCAS_contribution_lookup.items()}             # Round to the specified decimal places to avoid numerical issues in the GUROBI solver
        
        
        # Get the biodiversity contribution score 
        bio_contribution_raw = biodiv_raw[f'BIODIV_PRIORITY_SSP{settings.SSP}'].values
        self.BIO_CONNECTIVITY_RAW = bio_contribution_raw * connectivity_score                                          
        self.BIO_CONNECTIVITY_LDS = np.where(                                                                     
            self.SAVBURN_ELIGIBLE, 
            self.BIO_CONNECTIVITY_RAW * settings.BIO_CONTRIBUTION_LDS, 
            self.BIO_CONNECTIVITY_RAW
        )
        
  
        # ------------------ Habitat condition impacts for habitat conservation (GBF2) in 'priority degraded areas' regions ---------------
        if settings.BIODIVERSITY_TARGET_GBF_2 != 'off':
        
            # Get the mask of 'priority degraded areas' for habitat conservation
            conservation_performance_curve = pd.read_excel(os.path.join(settings.INPUT_DIR, 'BIODIVERSITY_GBF2_conservation_performance.xlsx'), sheet_name=f'ssp{settings.SSP}'
            ).set_index('AREA_COVERAGE_PERCENT')['PRIORITY_RANK'].to_dict()
            
            priority_degraded_areas_mask = bio_contribution_raw >= conservation_performance_curve[settings.GBF2_PRIORITY_DEGRADED_AREAS_PERCENTAGE_CUT]
            
            self.BIO_PRIORITY_DEGRADED_AREAS_R = np.where(
                self.SAVBURN_ELIGIBLE,
                priority_degraded_areas_mask * self.REAL_AREA * settings.BIO_CONTRIBUTION_LDS,
                priority_degraded_areas_mask * self.REAL_AREA
            )
            
            self.BIO_PRIORITY_DEGRADED_CONTRIBUTION_WEIGHTED_AREAS_BASE_YR_R = np.einsum(
                'j,mrj,r->r',
                np.array(list(self.BIO_HABITAT_CONTRIBUTION_LOOK_UP.values())),
                self.AG_L_MRJ,
                self.BIO_PRIORITY_DEGRADED_AREAS_R
            )

        
        ###############################################################
        # Vegetation data (GBF3).
        ###############################################################
        if settings.BIODIVERSITY_TARGET_GBF_3 != 'off':
        
            print("\tLoading vegetation data...", flush=True)
            
            # Read in the pre-1750 vegetation statistics, and get NVIS class names and areas
            GBF3_targets_df = pd.read_excel(
                settings.INPUT_DIR + '/BIODIVERSITY_GBF3_SCORES_AND_TARGETS.xlsx',
                sheet_name = f'NVIS_{settings.GBF3_TARGET_CLASS}'
            ).sort_values(by='group', ascending=True)
            
            
            if settings.BIODIVERSITY_TARGET_GBF_3 == 'USER_DEFINED':
                self.GBF3_GROUPS_SEL = [row['group'] for _,row in GBF3_targets_df.iterrows()
                    if all([
                        row['USER_DEFINED_TARGET_PERCENT_2030']>0,
                        row['USER_DEFINED_TARGET_PERCENT_2050']>0,
                        row['USER_DEFINED_TARGET_PERCENT_2100']>0]
                    )]
                self.GBF3_BASELINE_AREA_AND_USERDEFINE_TARGETS = GBF3_targets_df.query('group.isin(@self.GBF3_GROUPS_SEL)')
            else:
                self.GBF3_GROUPS_SEL = GBF3_targets_df['group'].tolist()
                self.GBF3_BASELINE_AREA_AND_USERDEFINE_TARGETS = GBF3_targets_df.query('group.isin(@self.GBF3_GROUPS_SEL)')
                self.GBF3_BASELINE_AREA_AND_USERDEFINE_TARGETS[[
                    'USER_DEFINED_TARGET_PERCENT_2030',
                    'USER_DEFINED_TARGET_PERCENT_2050', 
                    'USER_DEFINED_TARGET_PERCENT_2100']] = settings.GBF3_TARGETS_DICT[settings.BIODIVERSITY_TARGET_GBF_3]
                

            self.BIO_GBF3_BASELINE_SCORE_ALL_AUSTRALIA = self.GBF3_BASELINE_AREA_AND_USERDEFINE_TARGETS['AREA_WEIGHTED_SCORE_ALL_AUSTRALIA_HA'].to_numpy()
            self.BIO_GBF3_BASELINE_SCORE_OUTSIDE_LUTO = self.GBF3_BASELINE_AREA_AND_USERDEFINE_TARGETS['AREA_WEIGHTED_SCORE_OUTSIDE_LUTO_NATURAL_HA'].to_numpy()
            self.BIO_GBF3_ID2DESC = dict(enumerate(self.GBF3_GROUPS_SEL))
            self.BIO_GBF3_N_CLASSES = len(self.GBF3_GROUPS_SEL) 
            
            
            
            # Read in vegetation layer data
            NVIS_layers = xr.open_dataarray(settings.INPUT_DIR + f"/NVIS_{settings.GBF3_TARGET_CLASS.split('_')[0]}.nc").sel(group=self.GBF3_GROUPS_SEL)
            NVIS_layers = np.array([self.get_exact_resfactored_average_arr_without_lu_mask(arr) for arr in NVIS_layers], dtype=np.float32) / 100.0  # divide by 100 to get the percentage of the area in each cell that is covered by the vegetation type

            # Apply Savanna Burning penalties
            self.NVIS_LAYERS_LDS = np.where(
                self.SAVBURN_ELIGIBLE,
                NVIS_layers * settings.BIO_CONTRIBUTION_LDS,
                NVIS_layers
            )
            
            # Container storing which cells apply to each major vegetation group
            epsilon = 1e-5
            self.MAJOR_VEG_INDECES = {
                v: np.where(NVIS_layers[v] > epsilon)[0]
                for v in range(NVIS_layers.shape[0])
            }
            
            
 
        
        ##########################################################################
        #  Biodiersity environmental significance (GBF4)                         #
        ##########################################################################
        if settings.BIODIVERSITY_TARGET_GBF_4_SNES != 'off':

            print("\tLoading environmental significance data (SNES)...", flush=True)
            
            # Read in the species data from DCCEEW National Environmental Significance (noted as GBF-4)
            BIO_GBF4_SNES_score = pd.read_csv(settings.INPUT_DIR + '/BIODIVERSITY_GBF4_TARGET_SNES.csv').sort_values(by='SCIENTIFIC_NAME', ascending=True)
            
            self.BIO_GBF4_SNES_LIKELY_SEL = [row['SCIENTIFIC_NAME'] for _,row in BIO_GBF4_SNES_score.iterrows()
                                                    if all([row['USER_DEFINED_TARGET_PERCENT_2030_LIKELY']>0,
                                                            row['USER_DEFINED_TARGET_PERCENT_2050_LIKELY']>0,
                                                            row['USER_DEFINED_TARGET_PERCENT_2100_LIKELY']>0])]
            
            self.BIO_GBF4_SNES_LIKELY_AND_MAYBE_SEL = [row['SCIENTIFIC_NAME'] for _,row in BIO_GBF4_SNES_score.iterrows()
                                                    if all([row['USER_DEFINED_TARGET_PERCENT_2030_LIKELY_MAYBE']>0,
                                                            row['USER_DEFINED_TARGET_PERCENT_2050_LIKELY_MAYBE']>0,
                                                            row['USER_DEFINED_TARGET_PERCENT_2100_LIKELY_MAYBE']>0])]
            
            if len(self.BIO_GBF4_SNES_LIKELY_SEL) == 0:
                raise ValueError("At least one of 'LIKELY' layers should be selected!")

            likely_maybe_union = set(self.BIO_GBF4_SNES_LIKELY_SEL).intersection(self.BIO_GBF4_SNES_LIKELY_AND_MAYBE_SEL)
            if likely_maybe_union:
                print(f"\tWARNING: {len(likely_maybe_union)} duplicate SNE species targets found, using 'LIKELY' targets only:")
                print("\n".join(f"    {idx+1}) {name}" for idx, name in enumerate(likely_maybe_union)))
                self.BIO_GBF4_SNES_LIKELY_AND_MAYBE_SEL = list(set(self.BIO_GBF4_SNES_LIKELY_AND_MAYBE_SEL) - likely_maybe_union)
                
            self.BIO_GBF4_SNES_SEL_ALL = self.BIO_GBF4_SNES_LIKELY_SEL + self.BIO_GBF4_SNES_LIKELY_AND_MAYBE_SEL
            self.BIO_GBF4_PRESENCE_SNES_SEL = ['LIKELY'] * len(self.BIO_GBF4_SNES_LIKELY_SEL) + ['LIKELY_MAYBE'] * len(self.BIO_GBF4_SNES_LIKELY_AND_MAYBE_SEL)  
            self.BIO_GBF4_SNES_BASELINE_SCORE_TARGET_PERCENT_LIKELY = BIO_GBF4_SNES_score.query(f'SCIENTIFIC_NAME in {self.BIO_GBF4_SNES_LIKELY_SEL}')
            self.BIO_GBF4_SNES_BASELINE_SCORE_TARGET_PERCENT_LIKELY_AND_MAYBE = BIO_GBF4_SNES_score.query(f'SCIENTIFIC_NAME in {self.BIO_GBF4_SNES_LIKELY_AND_MAYBE_SEL}') 
            
            BIO_GBF4_SPECIES_raw = xr.open_dataarray(f'{settings.INPUT_DIR}/bio_GBF4_SNES.nc', chunks={'species':1})
            snes_arr_likely = BIO_GBF4_SPECIES_raw.sel(species=self.BIO_GBF4_SNES_LIKELY_SEL, presence='LIKELY')
            snes_arr_likely_maybe = BIO_GBF4_SPECIES_raw.sel(species=self.BIO_GBF4_SNES_LIKELY_AND_MAYBE_SEL, presence='LIKELY_AND_MAYBE')
            snes_arr = xr.concat([snes_arr_likely, snes_arr_likely_maybe], dim='species')
            self.BIO_GBF4_SPECIES_LAYERS = np.array([self.get_exact_resfactored_average_arr_without_lu_mask(arr) for arr in snes_arr]) 
        
        
        if settings.BIODIVERSITY_TARGET_GBF_4_SNES != 'off':
            print("\tLoading environmental significance data (ECNES)...", flush=True)
        
        
            BIO_GBF4_ECNES_score = pd.read_csv(settings.INPUT_DIR + '/BIODIVERSITY_GBF4_TARGET_ECNES.csv').sort_values(by='COMMUNITY', ascending=True)
       
            self.BIO_GBF4_ECNES_LIKELY_SEL = [row['COMMUNITY'] for _,row in BIO_GBF4_ECNES_score.iterrows()
                                                    if all([row['USER_DEFINED_TARGET_PERCENT_2030_LIKELY']>0,
                                                            row['USER_DEFINED_TARGET_PERCENT_2050_LIKELY']>0,
                                                            row['USER_DEFINED_TARGET_PERCENT_2100_LIKELY']>0])]
            
            self.BIO_GBF4_ECNES_LIKELY_AND_MAYBE_SEL = [row['COMMUNITY'] for _,row in BIO_GBF4_ECNES_score.iterrows()
                                                    if all([row['USER_DEFINED_TARGET_PERCENT_2030_LIKELY_MAYBE']>0,
                                                            row['USER_DEFINED_TARGET_PERCENT_2050_LIKELY_MAYBE']>0,
                                                            row['USER_DEFINED_TARGET_PERCENT_2100_LIKELY_MAYBE']>0])]
            
            if len(self.BIO_GBF4_ECNES_LIKELY_SEL) == 0:
                raise ValueError("At least one of 'LIKELY' layers should be selected!")
  
            likely_maybe_union = set(self.BIO_GBF4_ECNES_LIKELY_SEL).intersection(self.BIO_GBF4_ECNES_LIKELY_AND_MAYBE_SEL)
            if likely_maybe_union:
                print(f"\tWARNING: {len(likely_maybe_union)} duplicate ECNES species targets found, using 'LIKELY' targets only:")
                print("\n".join(f"    {idx+1}) {name}" for idx, name in enumerate(likely_maybe_union)))
                self.BIO_GBF4_ECNES_LIKELY_AND_MAYBE_SEL = list(set(self.BIO_GBF4_ECNES_LIKELY_AND_MAYBE_SEL) - likely_maybe_union)
                 
            self.BIO_GBF4_ECNES_SEL_ALL = self.BIO_GBF4_ECNES_LIKELY_SEL + self.BIO_GBF4_ECNES_LIKELY_AND_MAYBE_SEL
            self.BIO_GBF4_PRESENCE_ECNES_SEL = ['LIKELY'] * len(self.BIO_GBF4_ECNES_LIKELY_SEL) + ['LIKELY_MAYBE'] * len(self.BIO_GBF4_ECNES_LIKELY_AND_MAYBE_SEL)
            self.BIO_GBF4_ECNES_BASELINE_SCORE_TARGET_PERCENT_LIKELY = BIO_GBF4_ECNES_score.query(f'COMMUNITY in {self.BIO_GBF4_ECNES_LIKELY_SEL}')
            self.BIO_GBF4_ECNES_BASELINE_SCORE_TARGET_PERCENT_LIKELY_AND_MAYBE = BIO_GBF4_ECNES_score.query(f'COMMUNITY in {self.BIO_GBF4_ECNES_LIKELY_AND_MAYBE_SEL}')
    
            BIO_GBF4_COMUNITY_raw = xr.open_dataarray(f'{settings.INPUT_DIR}/bio_GBF4_ECNES.nc', chunks={'species':1})
            ecnes_arr_likely = BIO_GBF4_COMUNITY_raw.sel(species=self.BIO_GBF4_ECNES_LIKELY_SEL, presence='LIKELY').compute()
            ecnes_arr_likely_maybe = BIO_GBF4_COMUNITY_raw.sel(species=self.BIO_GBF4_ECNES_LIKELY_AND_MAYBE_SEL, presence='LIKELY_AND_MAYBE').compute()
            ecnes_arr = xr.concat([ecnes_arr_likely, ecnes_arr_likely_maybe], dim='species')
            self.BIO_GBF4_COMUNITY_LAYERS = np.array([self.get_exact_resfactored_average_arr_without_lu_mask(arr) for arr in ecnes_arr])
        
  
        
        ##########################################################################
        # Biodiersity species suitability under climate change (GBF8)            #
        ##########################################################################
        
        if settings.BIODIVERSITY_TARGET_GBF_8 != 'off':
            
            print("\tLoading Species suitability data...", flush=True)
            
            # Read in the species data from Carla Archibald (noted as GBF-8)
            BIO_GBF8_SPECIES_raw = xr.open_dataset(f'{settings.INPUT_DIR}/bio_GBF8_ssp{settings.SSP}_EnviroSuit.nc', chunks={'year':1,'species':1})['data']        
            bio_GBF8_baseline_score = pd.read_csv(settings.INPUT_DIR + '/BIODIVERSITY_GBF8_SCORES.csv').sort_values(by='species', ascending=True)
            bio_GBF8_target_percent = pd.read_csv(settings.INPUT_DIR + '/BIODIVERSITY_GBF8_TARGET.csv').sort_values(by='species', ascending=True)
            
            self.BIO_GBF8_SEL_SPECIES = [row['species'] for _,row in bio_GBF8_target_percent.iterrows() 
                                        if all([row['USER_DEFINED_TARGET_PERCENT_2030']>0,
                                                row['USER_DEFINED_TARGET_PERCENT_2050']>0,
                                                row['USER_DEFINED_TARGET_PERCENT_2100']>0])]
            
            self.BIO_GBF8_OUTSDIE_LUTO_SCORE_SPECIES = bio_GBF8_baseline_score.query(f'species in {self.BIO_GBF8_SEL_SPECIES}')[['species', 'year', f'OUTSIDE_LUTO_NATURAL_SUITABILITY_AREA_WEIGHTED_HA_SSP{settings.SSP}']]
            self.BIO_GBF8_OUTSDIE_LUTO_SCORE_GROUPS = pd.read_csv(settings.INPUT_DIR + '/BIODIVERSITY_GBF8_SCORES_group.csv')[['group', 'year', f'OUTSIDE_LUTO_NATURAL_SUITABILITY_AREA_WEIGHTED_HA_SSP{settings.SSP}']]
            
            self.BIO_GBF8_BASELINE_SCORE_AND_TARGET_PERCENT_SPECIES = bio_GBF8_target_percent.query(f'species in {self.BIO_GBF8_SEL_SPECIES}')
            self.BIO_GBF8_BASELINE_SCORE_GROUPS = pd.read_csv(settings.INPUT_DIR + '/BIODIVERSITY_GBF8_TARGET_group.csv')
            
            self.BIO_GBF8_SPECIES_LAYER = BIO_GBF8_SPECIES_raw.sel(species=self.BIO_GBF8_SEL_SPECIES).compute()
            self.N_GBF8_SPECIES = len(self.BIO_GBF8_SEL_SPECIES)
            
            self.BIO_GBF8_GROUPS_LAYER = xr.load_dataset(f'{settings.INPUT_DIR}/bio_GBF8_ssp{settings.SSP}_EnviroSuit_group.nc')['data']
            self.BIO_GBF8_GROUPS_NAMES = [i.capitalize() for i in self.BIO_GBF8_GROUPS_LAYER['group'].values]
        

        ###############################################################
        # BECCS data.
        ###############################################################
        print("\tLoading BECCS data...", flush=True)

        # Load dataframe
        beccs_df = pd.read_hdf(os.path.join(settings.INPUT_DIR, 'cell_BECCS_df.h5'), where=self.MASK)

        # Capture as numpy arrays
        self.BECCS_COSTS_AUD_HA_YR = beccs_df['BECCS_COSTS_AUD_HA_YR'].to_numpy()
        self.BECCS_REV_AUD_HA_YR = beccs_df['BECCS_REV_AUD_HA_YR'].to_numpy()
        self.BECCS_TCO2E_HA_YR = beccs_df['BECCS_TCO2E_HA_YR'].to_numpy()
        self.BECCS_MWH_HA_YR = beccs_df['BECCS_MWH_HA_YR'].to_numpy()

 

    def get_coord(self, index_ij: np.ndarray, trans):
        """
        Calculate the coordinates [[lon,...],[lat,...]] based on
        the given index [[row,...],[col,...]] and transformation matrix.

        Parameters
        index_ij (np.ndarray): A numpy array containing the row and column indices.
        trans (affin): An instance of the Transformation class.
        resfactor (int, optional): The resolution factor. Defaults to 1.

        Returns
        tuple: A tuple containing the x and y coordinates.
        """
        coord_x = trans.c + trans.a * (index_ij[1] + 0.5)    # Move to the center of the cell
        coord_y = trans.f + trans.e * (index_ij[0] + 0.5)    # Move to the center of the cell
        return coord_x, coord_y


    def update_geo_meta(self):
        """
        Update the geographic metadata based on the current settings.

        Note: When this function is called, the RESFACTOR is assumend to be > 1,
        because there is no need to update the metadata if the RESFACTOR is 1.

        Returns
            dict: The updated geographic metadata.
        """
        meta = self.GEO_META_FULLRES.copy()
        height, width =  self.LUMAP_2D_RESFACTORED.shape
        trans = list(self.GEO_META_FULLRES['transform'])
        trans[0] = trans[0] * settings.RESFACTOR    # Adjust the X resolution
        trans[4] = trans[4] * settings.RESFACTOR    # Adjust the Y resolution
        trans = Affine(*trans)
        meta.update(width=width, height=height, compress='lzw', driver='GTiff', transform=trans, nodata=self.NODATA, dtype='float32')
        return meta


    def add_lumap(self, yr: int, lumap: np.ndarray):
        """
        Safely adds a land-use map to the the Data object.
        """
        self.lumaps[yr] = lumap

    def add_lmmap(self, yr: int, lmmap: np.ndarray):
        """
        Safely adds a land-management map to the Data object.
        """
        self.lmmaps[yr] = lmmap

    def add_ammaps(self, yr: int, ammap: np.ndarray):
        """
        Safely adds an agricultural management map to the Data object.
        """
        self.ammaps[yr] = ammap

    def add_ag_dvars(self, yr: int, ag_dvars: np.ndarray):
        """
        Safely adds agricultural decision variables' values to the Data object.
        """
        self.ag_dvars[yr] = ag_dvars

    def add_non_ag_dvars(self, yr: int, non_ag_dvars: np.ndarray):
        """
        Safely adds non-agricultural decision variables' values to the Data object.
        """
        self.non_ag_dvars[yr] = non_ag_dvars

    def add_ag_man_dvars(self, yr: int, ag_man_dvars: dict[str, np.ndarray]):
        """
        Safely adds agricultural management decision variables' values to the Data object.
        """
        self.ag_man_dvars[yr] = ag_man_dvars
        
        
    def get_exact_resfactored_lumap_mrj(self):
        """
        Rather than picking the center cell when resfactoring the lumap, this function
        calculate the exact value of each land-use cell based from lumap to create dvars.
        
        E.g., given a resfactor of 5, then each resfactored dvar cell will cover a 5x5 area.
        If there are 9 Apple cells in the 5x5 area, then the dvar cell for it will be 9/25. 
        
        """
        if settings.RESFACTOR == 1:
            return tools.lumap2ag_l_mrj(self.LUMAP_NO_RESFACTOR, self.LMMAP_NO_RESFACTOR)[:, self.MASK, :]


        lumap_resample_avg = np.zeros((len(self.LANDMANS), self.NCELLS, self.N_AG_LUS), dtype=np.float32)  
        for idx_lu in self.DESC2AGLU.values():
            for idx_w, _ in enumerate(self.LANDMANS):
                arr_lu_lm = self.LUMAP_NO_RESFACTOR * self.LMMAP_NO_RESFACTOR
                lumap_resample_avg[idx_w, :, idx_lu] = self.get_exact_resfactored_average_arr_consider_lu_mask(arr_lu_lm)
                
        return lumap_resample_avg
    
    
    def get_exact_resfactored_lumap_mrj(self):
        """
        Rather than picking the center cell when resfactoring the lumap, this function
        calculate the exact value of each land-use cell based from lumap to create dvars.
        
        E.g., given a resfactor of 5, then each resfactored dvar cell will cover a 5x5 area.
        If there are 9 Apple cells in the 5x5 area, then the dvar cell for it will be 9/25. 
        
        """
        if settings.RESFACTOR == 1:
            return tools.lumap2ag_l_mrj(self.LUMAP_NO_RESFACTOR, self.LMMAP_NO_RESFACTOR)[:, self.MASK, :]

        lumap_mrj = np.zeros((self.NLMS, self.NCELLS, self.N_AG_LUS), dtype=np.float32)
        for idx_lu in self.DESC2AGLU.values():
            for idx_w, _ in enumerate(self.LANDMANS):
                # Get the cells with the same ID and water supply
                lu_arr = (self.LUMAP_NO_RESFACTOR == idx_lu) * (self.LMMAP_NO_RESFACTOR == idx_w)
                lumap_mrj[idx_w, :, idx_lu] = self.get_exact_resfactored_average_arr_consider_lu_mask(lu_arr)        
                    
        return lumap_mrj
    
    
    def get_exact_resfactored_average_arr_consider_lu_mask(self, arr: np.ndarray) -> np.ndarray:
            
        arr_2d = np.zeros_like(self.LUMAP_2D_FULLRES, dtype=np.float32)      # Create a 2D array of zeros with the same shape as the LUMAP_2D_FULLRES
        np.place(arr_2d, self.NLUM_MASK == 1, arr)                           # Place the values of arr in the 2D array where the LUMAP_2D_RESFACTORED is equal to idx_lu

        mask_arr_2d_resfactor = (self.LUMAP_2D_RESFACTORED != self.NODATA) & (self.LUMAP_2D_RESFACTORED != self.MASK_LU_CODE) 
        mask_arr_2d_fullres = (self.LUMAP_2D_FULLRES != self.NODATA) & (self.LUMAP_2D_FULLRES != self.MASK_LU_CODE)

        # Create a 2D array of IDs for the LUMAP_2D_RESFACTORED
        id_arr_2d_resfactored = np.arange(self.LUMAP_2D_RESFACTORED.size).reshape(self.LUMAP_2D_RESFACTORED.shape)
        id_arr_2d_fullres = upsample_array(self, id_arr_2d_resfactored, settings.RESFACTOR)

        # Calculate the average value for each cell in the resfactored array
        cell_count = np.bincount(id_arr_2d_fullres.flatten(), mask_arr_2d_fullres.flatten(), minlength=self.LUMAP_2D_RESFACTORED.size)
        cell_sum = np.bincount(id_arr_2d_fullres.flatten(), arr_2d.flatten(), minlength=self.LUMAP_2D_RESFACTORED.size)
        with np.errstate(divide='ignore', invalid='ignore'):                    # Ignore the division by zero warning
            cell_avg = cell_sum / cell_count
            cell_avg[~np.isfinite(cell_avg)] = 0                                # Set the NaN and Inf to 0
            
        # Reshape the 1D avg array to 2D array
        cell_avg_2d = cell_avg.reshape(self.LUMAP_2D_RESFACTORED.shape)
        return cell_avg_2d[mask_arr_2d_resfactor]
    
    
    def get_exact_resfactored_average_arr_without_lu_mask(self, arr: np.ndarray) -> np.ndarray:
        
        arr_2d = np.zeros_like(self.LUMAP_2D_FULLRES, dtype=np.float32)      # Create a 2D array of zeros with the same shape as the LUMAP_2D_FULLRES
        np.place(arr_2d, self.NLUM_MASK == 1, arr)                           # Place the values of arr in the 2D array where the LUMAP_2D_RESFACTORED is equal to idx_lu
        arr_2d = np.pad(arr_2d, ((0, settings.RESFACTOR), (0, settings.RESFACTOR)), mode='reflect')  

        arr_2d_xr = xr.DataArray(arr_2d, dims=['y', 'x'])
        arr_2d_xr_resfactored = arr_2d_xr.coarsen(x=settings.RESFACTOR, y=settings.RESFACTOR, boundary='trim').mean()
        arr_2d_xr_resfactored = arr_2d_xr_resfactored.values[0:self.LUMAP_2D_RESFACTORED.shape[0], 0:self.LUMAP_2D_RESFACTORED.shape[1]]  

        mask_arr_2d_resfactor = (self.LUMAP_2D_RESFACTORED != self.NODATA) & (self.LUMAP_2D_RESFACTORED != self.MASK_LU_CODE) 
        return arr_2d_xr_resfactored[mask_arr_2d_resfactor]
    

    
    def get_resfactored_lumap(self) -> np.ndarray:
        """
        Coarsens the LUMAP to the specified resolution factor.
        """

        lumap_resfactored = np.zeros(self.NCELLS, dtype=np.int8) - 1
        fill_mask = np.ones(self.NCELLS, dtype=bool)

        # Fill resfactored land-use map with the land-use codes given their resfactored size
        for _,(lu_code, res_size) in self.LU_RESFACTOR_CELLS.iterrows():
            
            lu_avg = self.AG_L_MRJ[:,:,lu_code].sum(0) * fill_mask
            res_size = min(res_size, (lu_avg > 0).sum())
            
            # Assign the n-largets cells with the land-use code
            lu_idx = np.argsort(lu_avg)[-res_size:]  
            lumap_resfactored[lu_idx] = lu_code
            fill_mask[lu_idx] = False
            
        # Fill -1 with nearest neighbour values
        nearst_ind = distance_transform_edt(
            (lumap_resfactored == -1),
            return_distances=False,
            return_indices=True
        )
      
        return lumap_resfactored[*nearst_ind]
    
    
    def get_GBF2_target_for_yr_cal(self, yr_cal:int) -> float:
        """
        Get the target score for priority degrade areas conservation.
        
        Parameters
        ----------
        yr_cal : int
            The year for which to get the habitat condition score.
            
        Returns
        -------
        float
            The priority degrade areas conservation target for the given year.
        """
 
        bio_habitat_score_baseline_sum = self.BIO_PRIORITY_DEGRADED_AREAS_R.sum()
        bio_habitat_score_base_yr_sum = self.BIO_PRIORITY_DEGRADED_CONTRIBUTION_WEIGHTED_AREAS_BASE_YR_R.sum()
        bio_habitat_score_base_yr_proportion = bio_habitat_score_base_yr_sum / bio_habitat_score_baseline_sum

        bio_habitat_target_proportion = [
            bio_habitat_score_base_yr_proportion + ((1 - bio_habitat_score_base_yr_proportion) * i)
            for i in settings.GBF2_TARGETS_DICT[settings.BIODIVERSITY_TARGET_GBF_2].values()
        ]

        targets_key_years = {
            self.YR_CAL_BASE: bio_habitat_score_base_yr_sum, 
            **dict(zip(settings.GBF2_TARGETS_DICT[settings.BIODIVERSITY_TARGET_GBF_2].keys(), bio_habitat_score_baseline_sum * np.array(bio_habitat_target_proportion)))
        }

        f = interp1d(
            list(targets_key_years.keys()),
            list(targets_key_years.values()),
            kind = "linear",
            fill_value = "extrapolate"
        )

        return f(yr_cal).item()  # Convert the interpolated value to a scalar
    
    
    def get_GBF3_limit_score_inside_LUTO_by_yr(self, yr:int) -> np.ndarray:
        '''
        Interpolate the user-defined targets to get target at the given year
        '''
        
        GBF3_target_percents = []
        for _,row in self.GBF3_BASELINE_AREA_AND_USERDEFINE_TARGETS.iterrows():
            f = interp1d(
                [2010, 2030, 2050, 2100],
                [min(row['BASE_YR_PERCENT'],row['USER_DEFINED_TARGET_PERCENT_2030']), 
                 row['USER_DEFINED_TARGET_PERCENT_2030'], 
                 row['USER_DEFINED_TARGET_PERCENT_2050'], 
                 row['USER_DEFINED_TARGET_PERCENT_2100']
                ],
                kind="linear",
                fill_value="extrapolate",
            )
            GBF3_target_percents.append(f(yr).item())
        
        limit_score_all_AUS = self.BIO_GBF3_BASELINE_SCORE_ALL_AUSTRALIA * (np.array(GBF3_target_percents) / 100)  # Convert the percentage to proportion
        limit_score_inside_LUTO = limit_score_all_AUS - self.BIO_GBF3_BASELINE_SCORE_OUTSIDE_LUTO
            
        return np.where(limit_score_inside_LUTO < 0, 0, limit_score_inside_LUTO)

    
    def get_GBF4_SNES_target_inside_LUTO_by_year(self, yr:int) -> np.ndarray:
        
        # Check the layer name
        snes_df_likely = self.BIO_GBF4_SNES_BASELINE_SCORE_TARGET_PERCENT_LIKELY
        snes_df_likely_maybe = self.BIO_GBF4_SNES_BASELINE_SCORE_TARGET_PERCENT_LIKELY_AND_MAYBE
        snes_df = pd.concat([snes_df_likely, snes_df_likely_maybe], ignore_index=True)

        targets = []
        for idx,row in snes_df.iterrows():
            layer = self.BIO_GBF4_PRESENCE_SNES_SEL[idx]
            f = interp1d(
                [2010, 2030, 2050, 2100],
                [row[f'HABITAT_SIGNIFICANCE_BASELINE_PERCENT_{layer}'], row[f'USER_DEFINED_TARGET_PERCENT_2030_{layer}'], row[f'USER_DEFINED_TARGET_PERCENT_2050_{layer}'], row[f'USER_DEFINED_TARGET_PERCENT_2100_{layer}']],
                kind = "linear",
                fill_value = "extrapolate",
            )
            
            score_all_aus = row[f'HABITAT_SIGNIFICANCE_BASELINE_ALL_AUSTRALIA_{layer}'] * f(yr) / 100  # Convert the percentage to proportion
            score_out_LUTO = row[f'HABITAT_SIGNIFICANCE_BASELINE_OUT_LUTO_NATURAL_{layer}'] 
            targets.append(score_all_aus - score_out_LUTO)

        return np.array(targets).astype(np.float32)

        
    def get_GBF4_ECNES_target_inside_LUTO_by_year(self, yr:int) -> np.ndarray:
        # Check the layer name
        ecnes_df_likely = self.BIO_GBF4_ECNES_BASELINE_SCORE_TARGET_PERCENT_LIKELY
        ecnes_df_likely_maybe = self.BIO_GBF4_ECNES_BASELINE_SCORE_TARGET_PERCENT_LIKELY_AND_MAYBE
        ecnes_df = pd.concat([ecnes_df_likely, ecnes_df_likely_maybe], ignore_index=True)

        targets = []
        for idx,row in ecnes_df.iterrows():
            layer = self.BIO_GBF4_PRESENCE_ECNES_SEL[idx]
            f = interp1d(
                [2010, 2030, 2050, 2100],
                [row[f'HABITAT_SIGNIFICANCE_BASELINE_PERCENT_{layer}'], row[f'USER_DEFINED_TARGET_PERCENT_2030_{layer}'], row[f'USER_DEFINED_TARGET_PERCENT_2050_{layer}'], row[f'USER_DEFINED_TARGET_PERCENT_2100_{layer}']],
                kind = "linear",
                fill_value = "extrapolate",
            )
            
            score_all_aus = row[f'HABITAT_SIGNIFICANCE_BASELINE_ALL_AUSTRALIA_{layer}'] * f(yr) / 100  # Convert the percentage to proportion
            score_out_LUTO = row[f'HABITAT_SIGNIFICANCE_BASELINE_OUT_LUTO_NATURAL_{layer}']
            targets.append(score_all_aus - score_out_LUTO)  
                      
        return np.array(targets).astype(np.float32)
    
    
    def get_GBF8_bio_layers_by_yr(self, yr: int, level:Literal['species', 'group']='species'):
        '''
        Get the biodiversity suitability score [hectare weighted] for each species at the given year.
        
        The raw biodiversity suitability score [2D (shape, 808*978), (dtype, uint8, 0-100)] represents the 
        suitability of each cell for each species/group.  Here it is LINEARLY interpolated to the given year,
        then LINEARLY interpolated to the given spatial coordinates.
        
        Because the coordinates are the controid of the `self.MASK` array, so the spatial interpolation is 
        simultaneously a masking process. 
        
        The suitability score is then weighted by the area (ha) of each cell. The area weighting is necessary 
        to ensure that the biodiversity suitability score will not be affected by different RESFACTOR (i.e., cell size) values.
        
        Parameters
        ----------
        yr : int
            The year for which to get the biodiversity suitability score.
        level : str, optional
            The level of the biodiversity suitability score, either 'species' or 'group'. The default is 'species'.
            
        Returns
        -------
        np.ndarray
            The biodiversity suitability score for each species at the given year.
        '''
        
        input_lr = self.BIO_GBF8_SPECIES_LAYER if level == 'species' else self.BIO_GBF8_GROUPS_LAYER
        
        current_species_val = input_lr.interp(                          # Here the year interpolation is done first                      
            year=yr,
            method='linear', 
            kwargs={'fill_value': 'extrapolate'}
        ).interp(                                                       # Then the spatial interpolation and masking is done
            x=xr.DataArray(self.COORD_LON_LAT[0], dims='cell'),
            y=xr.DataArray(self.COORD_LON_LAT[1], dims='cell'),
            method='linear'                                             # Use LINEAR interpolation
        ).drop_vars(['year']).values
        
        # Apply Savanna Burning penalties
        current_species_val = np.where(
            self.SAVBURN_ELIGIBLE,
            current_species_val * settings.BIO_CONTRIBUTION_LDS,
            current_species_val
        )
        
        return current_species_val.astype(np.float32)
    

    def get_GBF8_target_inside_LUTO_by_yr(self, yr: int) -> np.ndarray:
        '''
        Get the biodiversity suitability score (area weighted [ha]) for each species at the given year for the Inside LUTO natural land.
        '''
        target_scores = self.get_GBF8_score_all_Australia_by_yr(yr) - self.get_GBF8_score_outside_natural_LUTO_by_yr(yr)
        return target_scores
    

    
    def get_GBF8_score_all_Australia_by_yr(self, yr: int):
        '''
        Get the biodiversity suitability score (area weighted [ha]) for each species at the given year for all Australia.
        '''
        # Get the target percentage for each species at the given year
        target_pct = []
        for _,row in self.BIO_GBF8_BASELINE_SCORE_AND_TARGET_PERCENT_SPECIES.iterrows():
            f = interp1d(
                [2010, 2030, 2050, 2100],
                [row['HABITAT_SUITABILITY_BASELINE_PERCENT'], row[f'USER_DEFINED_TARGET_PERCENT_2030'], row[f'USER_DEFINED_TARGET_PERCENT_2050'], row[f'USER_DEFINED_TARGET_PERCENT_2100']],
                kind="linear",
                fill_value="extrapolate",
            )
            target_pct.append(f(yr).item()) 
            
        # Calculate the target biodiversity suitability score for each species at the given year for all Australia
        target_scores_all_AUS = self.BIO_GBF8_BASELINE_SCORE_AND_TARGET_PERCENT_SPECIES['HABITAT_SUITABILITY_BASELINE_SCORE_ALL_AUSTRALIA'] * (np.array(target_pct) / 100) # Convert the percentage to proportion
        return target_scores_all_AUS.values
    
    
    def get_GBF8_score_outside_natural_LUTO_by_yr(self, yr: int, level:Literal['species', 'group']='species'):
        '''
        Get the biodiversity suitability score (area weighted [ha]) for each species at the given year for the Outside LUTO natural land.
        '''
        
        if level == 'species':
            base_score = self.BIO_GBF8_BASELINE_SCORE_AND_TARGET_PERCENT_SPECIES['HABITAT_SUITABILITY_BASELINE_SCORE_OUTSIDE_LUTO']
            proj_score = self.BIO_GBF8_OUTSDIE_LUTO_SCORE_SPECIES.pivot(index='species', columns='year').reset_index()
        elif level == 'group':
            base_score = self.BIO_GBF8_BASELINE_SCORE_GROUPS['HABITAT_SUITABILITY_BASELINE_SCORE_OUTSIDE_LUTO']
            proj_score = self.BIO_GBF8_OUTSDIE_LUTO_SCORE_GROUPS.pivot(index='group', columns='year').reset_index()
        else:
            raise ValueError("Invalid level. Must be 'species' or 'group'")
        
        # Put the base score to the proj_score
        proj_score.columns = proj_score.columns.droplevel() 
        proj_score[1990] = base_score.values
        
        # Interpolate the suitability score for each species/group at the given year
        outside_natural_scores = []
        for _,row in proj_score.iterrows():
            f = interp1d(
                [1990, 2030, 2050, 2070, 2090],
                [row[1990], row[2030], row[2050], row[2070], row[2090]],
                kind="linear",
                fill_value="extrapolate",
            )
            outside_natural_scores.append(f(yr).item())
        
        return  outside_natural_scores


    
    def get_regional_adoption_percent_by_year(self, yr: int):
        """
        Get the regional adoption percentage for each region for the given year.
        
        Return a list of tuples where each tuple contains 
        - the region ID, 
        - landuse name, 
        - the adoption percentage.
        
        """
        if settings.REGIONAL_ADOPTION_CONSTRAINTS != "on":
            return ()
        
        reg_adop_limits = []
        for _,row in self.REGIONAL_ADOPTION_TARGETS.iterrows():
            f = interp1d(
                [2010, 2030, 2050, 2100],
                [row['BASE_LANDUSE_AREA_PERCENT'], row['ADOPTION_PERCENTAGE_2030'], row['ADOPTION_PERCENTAGE_2050'], row['ADOPTION_PERCENTAGE_2100']],
                kind="linear",
                fill_value="extrapolate",
            )
            reg_adop_limits.append((row[settings.REGIONAL_ADOPTION_ZONE], row['TARGET_LANDUSE'], f(yr).item()))
            
        return reg_adop_limits
    
    def get_regional_adoption_limit_ha_by_year(self, yr: int):
        """
        Get the regional adoption area for each region for the given year.
        
        Return a list of tuples where each tuple contains
        - the region ID,
        - landuse name,
        - the adoption area (ha).
        """
        if settings.REGIONAL_ADOPTION_CONSTRAINTS != "on":
            return ()
        
        reg_adop_limits = self.get_regional_adoption_percent_by_year(yr)
        reg_adop_limits_ha = []
        for reg, landuse, pct in reg_adop_limits:
            reg_total_area_ha = ((self.REGIONAL_ADOPTION_ZONES == reg) * self.REAL_AREA).sum()
            reg_adop_limits_ha.append((reg, landuse, reg_total_area_ha * pct / 100))
            
        return reg_adop_limits_ha
    
    
    def add_production_data(self, yr: int, data_type: str, prod_data: Any):
        """
        Safely save production data for a given year to the Data object.

        Parameters
        ----
        yr: int
            Year of production data being saved.
        data_type: str
            Type of production data being saved. Typically either 'Production', 'GHG Emissions' or 'Biodiversity'.
        prod_data: Any
            Actual production data to save.
        """
        if yr not in self.prod_data:
            self.prod_data[yr] = {}
        self.prod_data[yr][data_type] = prod_data

    def add_obj_vals(self, yr: int, obj_val: float):
        """
        Safely save objective value for a given year to the Data object
        """
        self.obj_vals[yr] = obj_val

    def set_path(self) -> str:
        """Create a folder for storing outputs and return folder name."""

        # Create path name
        years = [i for i in settings.SIM_YEARS if i<=self.last_year]
        self.path = f"{settings.OUTPUT_DIR}/{self.timestamp}_RF{settings.RESFACTOR}_{years[0]}-{years[-1]}"

        # Get all paths
        paths = (
            [self.path]
            + [f"{self.path}/out_{yr}" for yr in years]
            + [f"{self.path}/out_{yr}/lucc_separate" for yr in years[1:]]
        )  # Skip creating lucc_separate for base year

        # Add the path for the comparison between base-year and target-year if in the timeseries mode
        self.path_begin_end_compare = f"{self.path}/begin_end_compare_{years[0]}_{years[-1]}"
        paths = (
            paths
            + [self.path_begin_end_compare]
            + [
                f"{self.path_begin_end_compare}/out_{years[0]}",
                f"{self.path_begin_end_compare}/out_{years[-1]}",
                f"{self.path_begin_end_compare}/out_{years[-1]}/lucc_separate",
            ]
        )

        # Create all paths
        for p in paths:
            if not os.path.exists(p):
                os.mkdir(p)

        return self.path

    def get_production(
        self,
        yr_cal: int,
        lumap: np.ndarray,
        lmmap: np.ndarray,
    ) -> np.ndarray:
        """
        Return total production of commodities for a specific year...

        'yr_cal' is calendar year

        Can return base year production (e.g., year = 2010) or can return production for
        a simulated year if one exists (i.e., year = 2030).

        Includes the impacts of land-use change, productivity increases, and
        climate change on yield.
        """
        if yr_cal == self.YR_CAL_BASE:
            ag_X_mrj = self.AG_L_MRJ
            non_ag_X_rk = self.NON_AG_L_RK
            ag_man_X_mrj = self.AG_MAN_L_MRJ_DICT
            
        else:
            ag_X_mrj = tools.lumap2ag_l_mrj(lumap, lmmap)
            non_ag_X_rk = lumap2non_ag_l_mk(lumap, len(settings.NON_AG_LAND_USES.keys()))
            ag_man_X_mrj = get_base_am_vars(self.NCELLS, self.NLMS, self.N_AG_LUS)

        # Calculate year index (i.e., number of years since 2010)
        yr_idx = yr_cal - self.YR_CAL_BASE

        # Get the quantity of each commodity produced by agricultural land uses
        ag_q_mrp = ag_quantity.get_quantity_matrices(self, yr_idx)

        # Convert map of land-use in mrj format to mrp format using vectorization
        ag_X_mrp = np.einsum('mrj,pj->mrp', ag_X_mrj, self.LU2PR.astype(bool))

        # Sum quantities in product (PR/p) representation.
        ag_q_p = np.einsum('mrp,mrp->p', ag_q_mrp, ag_X_mrp)

        # Transform quantities to commodity (CM/c) representation.
        ag_q_c = np.einsum('cp,p->c', self.PR2CM.astype(bool), ag_q_p)

        # Get the quantity of each commodity produced by non-agricultural land uses
        q_crk = non_ag_quantity.get_quantity_matrix(self, ag_q_mrp, lumap)
        non_ag_q_c = np.einsum('crk,rk->c', q_crk, non_ag_X_rk)

        # Get quantities produced by agricultural management options
        ag_man_q_mrp = ag_quantity.get_agricultural_management_quantity_matrices(self, ag_q_mrp, yr_idx)
        ag_man_q_c = np.zeros(self.NCMS)

        j2p = {j: [p for p in range(self.NPRS) if self.LU2PR[p, j]]
                        for j in range(self.N_AG_LUS)}
        for am, am_lus in settings.AG_MANAGEMENTS_TO_LAND_USES.items():
            if not settings.AG_MANAGEMENTS[am]:
                continue
            
            am_j_list = [self.DESC2AGLU[lu] for lu in am_lus]
            current_ag_man_X_mrp = np.zeros(ag_q_mrp.shape, dtype=np.float32)
            for j in am_j_list:
                for p in j2p[j]:
                    current_ag_man_X_mrp[:, :, p] = ag_man_X_mrj[am][:, :, j]

            ag_man_q_p = np.einsum('mrp,mrp->p', ag_man_q_mrp[am], current_ag_man_X_mrp)
            ag_man_q_c += np.einsum('cp,p->c', self.PR2CM.astype(bool), ag_man_q_p)

        # Return total commodity production as numpy array.
        total_q_c = ag_q_c + non_ag_q_c + ag_man_q_c
        return total_q_c


    def get_carbon_price_by_yr_idx(self, yr_idx: int) -> float:
        """
        Return the price of carbon per tonne for a given year index (since 2010).
        The resulting year should be between 2010 - 2100
        """
        yr_cal = yr_idx + self.YR_CAL_BASE
        return self.get_carbon_price_by_year(yr_cal)

    def get_carbon_price_by_year(self, yr_cal: int) -> float:
        """
        Return the price of carbon per tonne for a given year.
        The resulting year should be between 2010 - 2100
        """
        if yr_cal not in self.CARBON_PRICES:
            raise ValueError(
                f"Carbon price data not given for the given year: {yr_cal}. "
                f"Year should be between {self.YR_CAL_BASE} and 2100."
            )
        return self.CARBON_PRICES[yr_cal]

    def get_water_nl_yield_for_yr_idx(
        self,
        yr_idx: int,
        water_dr_yield: Optional[np.ndarray] = None,
        water_sr_yield: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Get the net land water yield array, inclusive of all cells that LUTO does not look at.

        Returns
        -------
        np.ndarray: shape (NCELLS,)
        """
        water_dr_yield = (
            water_dr_yield if water_dr_yield is not None
            else self.WATER_YIELD_DR_FILE[yr_idx]
        )
        water_sr_yield = (
            water_sr_yield if water_sr_yield is not None
            else self.WATER_YIELD_SR_FILE[yr_idx]
        )
        dr_prop = self.DEEP_ROOTED_PROPORTION

        return (dr_prop * water_dr_yield + (1 - dr_prop) * water_sr_yield)
