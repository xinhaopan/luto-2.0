# Copyright 2023 Fjalar J. de Haan and Brett A. Bryan at Deakin University
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

"""
Pure functions to calculate economic profit from land use.
"""

import numpy as np

from luto.economics.quantity import get_yield_pot, lvs_veg_types, get_quantity

def get_rev_crop( data # Data object or module.
                , lu   # Land use.
                , lm   # Land management.
                , year # Number of years post base-year ('annum').
                ):
    """Return crop profit [AUD/cell] of `lu`+`lm` in `year` as np array.

    `data`: data object/module -- assumes fields like in `luto.data`.
    `lu`: land use (e.g. 'Winter cereals' or 'Beef - natural land').
    `lm`: land management (e.g. 'dry', 'irr').
    `year`: number of years from base year, counting from zero.
    """
    
    # Check if land-use exists in AGEC_CROPS (e.g., dryland Pears/Rice do not occur), if not return zeros
    if lu not in data.AGEC_CROPS['P1', lm].columns:
        rev_t = np.zeros((data.NCELLS))
        
    else: # Calculate the total revenue 
        
        # Revenue in $ per cell (includes REAL_AREA via get_quantity)
        rev_t = ( data.AGEC_CROPS['P1', lm, lu]
                * get_quantity( data, lu.upper(), lm, year )  # lu.upper() only for crops as needs to be in product format in get_quantity().
                )
    
    # Return revenue as numpy array.
    return rev_t


def get_rev_lvstk( data # Data object or module.
                 , lu   # Land use.
                 , lm   # Land management.
                 , year # Number of years post base-year ('annum').
                 ):
    """Return livestock revenue [AUD/cell] of `lu`+`lm` in `year` as np array.

    `data`: data object/module -- assumes fields like in `luto.data`.
    `lu`: land use (e.g. 'Winter cereals' or 'Beef - natural land').
    `lm`: land management (e.g. 'dry', 'irr').
    `year`: number of years from base year, counting from zero."""
    
    # Get livestock and vegetation type.
    lvstype, vegtype = lvs_veg_types(lu)

    # Get the yield potential, i.e. the total number of heads per hectare.
    yield_pot = get_yield_pot(data, lvstype, vegtype, lm, year)
    
    # Revenue in $ per cell (includes RESMULT via get_quantity)
    if lvstype == 'BEEF':
        rev = yield_pot * (  # Meat                           # Stocking density (head/ha)
                           ( data.AGEC_LVSTK['F1', lvstype]   # Fraction of herd producing (0 - 1)
                           * data.AGEC_LVSTK['Q1', lvstype]   # Quantity produced per head (meat tonnes/head)
                           * data.AGEC_LVSTK['P1', lvstype] ) # Price per unit quantity ($/tonne of meat)
                           + # Live exports
                           ( data.AGEC_LVSTK['F3', lvstype]   # Fraction of herd producing (0 - 1) 
                           * data.AGEC_LVSTK['Q3', lvstype]   # Quantity produced per head (animal weight tonnes/head)
                           * data.AGEC_LVSTK['P3', lvstype] ) # Price per unit quantity ($/tonne of animal)
                          )    

    elif lvstype == 'SHEEP':
        rev = yield_pot * (  # Meat                           # Stocking density (head/ha)
                           ( data.AGEC_LVSTK['F1', lvstype]   # Fraction of herd producing (0 - 1) 
                           * data.AGEC_LVSTK['Q1', lvstype]   # Quantity produced per head (meat tonnes/head)
                           * data.AGEC_LVSTK['P1', lvstype] ) # Price per unit quantity ($/tonne of meat)
                           + # Wool
                           ( data.AGEC_LVSTK['F2', lvstype]   # Fraction of herd producing (0 - 1) 
                           * data.AGEC_LVSTK['Q2', lvstype]   # Quantity produced per head (wool tonnes/head)
                           * data.AGEC_LVSTK['P2', lvstype] ) # Price per unit quantity ($/tonne wool)
                           + # Live exports
                           ( data.AGEC_LVSTK['F3', lvstype]   # Fraction of herd producing (0 - 1) 
                           * data.AGEC_LVSTK['Q3', lvstype]   # Quantity produced per head (animal weight tonnes/head)
                           * data.AGEC_LVSTK['P3', lvstype] ) # Price per unit quantity ($/tonne of whole animal)
                          )    

    elif lvstype == 'DAIRY':
        rev = yield_pot * (  # Milk                           # Stocking density (head/ha)
                           ( data.AGEC_LVSTK['F1', lvstype]   # Fraction of herd producing (0 - 1) 
                           * data.AGEC_LVSTK['Q1', lvstype]   # Quantity produced per head (milk litres/head)
                           * data.AGEC_LVSTK['P1', lvstype] ) # Price per unit quantity
                          )    

    else:  # Livestock type is unknown.
        raise KeyError("Unknown %s livestock type. Check `lvstype`." % lvstype)
        
    # Revenue so far in AUD/ha. Now convert to AUD/cell including resfactor.
    rev *= data.REAL_AREA
    
    # Return revenue as numpy array.
    return rev


def get_rev( data # Data object or module.
            , lu   # Land use.
            , lm   # Land management.
            , year # Number of years post base-year ('annum').
            ):
    """Return revenue from production [AUD/cell] of `lu`+`lm` in `year` as np array.

    `data`: data object/module -- assumes fields like in `luto.data`.
    `lu`: land use (e.g. 'Winter cereals').
    `lm`: land management (e.g. 'dry', 'irr').
    `year`: number of years from base year, counting from zero.
    """
    # If it is a crop, it is known how to get the revenue.
    if lu in data.LU_CROPS:
        return get_rev_crop(data, lu, lm, year)
    
    # If it is livestock, it is known how to get the revenue.
    elif lu in data.LU_LVSTK:
        return get_rev_lvstk(data, lu, lm, year)
    
    # If neither crop nor livestock but in LANDUSES it is unallocated land.
    elif lu in data.LANDUSES:
        return np.zeros(data.NCELLS)
    
    # If it is none of the above, it is not known how to get the revenue.
    else:
        raise KeyError("Land-use '%s' not found in data.LANDUSES" % lu)


def get_rev_matrix(data, lm, year):
    """Return r_rj matrix of revenue/cell per lu under `lm` in `year`."""
    
    r_rj = np.zeros((data.NCELLS, len(data.LANDUSES)))
    for j, lu in enumerate(data.LANDUSES):
        r_rj[:, j] = get_rev(data, lu, lm, year)
        
    # Make sure all NaNs are replaced by zeroes.
    return np.nan_to_num(r_rj)


def get_rev_matrices(data, year):
    """Return r_mrj matrix of revenue per cell as 3D Numpy array."""
    
    return np.stack(tuple( get_rev_matrix(data, lm, year)
                           for lm in data.LANDMANS )
                    ).astype(np.float32)