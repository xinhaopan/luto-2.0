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



"""
To compare LUTO 2.0 spatial arrays.
"""

import numpy as np
import pandas as pd
import luto.settings as settings


def lumap_crossmap(oldmap, newmap, ag_landuses, non_ag_landuses, real_area):
    # Produce the cross-tabulation matrix with optional labels.
    crosstab = pd.crosstab(oldmap, 
                           newmap, 
                           values = real_area, 
                           aggfunc = lambda x:x.sum() , 
                           margins = False)                 

    # Need to make sure each land use (both agricultural and non-agricultural) appears in the index/columns
    reindex = (
        list(range(len(ag_landuses)))
        + [settings.NON_AGRICULTURAL_LU_BASE_CODE + lu for lu in range(len(non_ag_landuses))]
    )
    
    crosstab = crosstab.reindex(reindex, axis = 0, fill_value = 0)
    crosstab = crosstab.reindex(reindex, axis = 1, fill_value = 0)
    
    lus = list(ag_landuses) + list(non_ag_landuses)
    crosstab.columns = lus
    crosstab.index = lus
    crosstab = crosstab.fillna(0)
    
    # Calculate net switches to land use (negative means switch away).
    switches = crosstab.sum(0) - crosstab.sum(1)
    switches = pd.DataFrame(switches).reset_index()
    switches.columns = ['Land-use', 'Area (ha)']
        
    # Stack the crosstab to a long format.
    crosstab = crosstab.stack().reset_index()
    crosstab.columns = ['From land-use', 'To land-use', 'Area (ha)']

    return crosstab, switches

def lmmap_crossmap(oldmap, newmap, real_area, lm):
    # Produce the cross-tabulation matrix with optional labels.
    crosstab = pd.crosstab(oldmap, 
                            newmap,
                            values = real_area, 
                            aggfunc = lambda x:x.sum() , 
                            margins = False)

    crosstab = crosstab.reindex(range(len(lm)), axis=0, fill_value=0)
    crosstab = crosstab.reindex(range(len(lm)), axis=1, fill_value=0)

    # Calculate net switches to land use (negative means switch away).
    switches = crosstab.sum(0) - crosstab.sum(1)
    switches = pd.DataFrame(switches).reset_index()
    switches.columns = ['Land-use', 'Area (ha)']

    # Stack the crosstab to a long format.
    crosstab = crosstab.stack().reset_index()
    crosstab.columns = ['From water_supply', 'To water_supply', 'Area (ha)']
    crosstab = crosstab.replace({0:'Dryland',1:'Irrigated'})

    return crosstab, switches





def crossmap_irrstat( lumap_old
                    , lmmap_old
                    , lumap_new
                    , lmmap_new
                    , ag_landuses
                    , non_ag_landuses
                    , real_area):

    ludict = {-2: 'Non-agricultural land (dry)',
            -1: 'Non-agricultural land (irr)'}

    for j, lu in enumerate(list(ag_landuses)):
        ludict[j*2] = f'{lu} (dry)'
        ludict[j*2 + 1] = f'{lu} (irr)'

    base = settings.NON_AGRICULTURAL_LU_BASE_CODE
    for k, lu in enumerate(list(non_ag_landuses)):
        ludict[(k + base)*2] = f'{lu} (dry)'
        ludict[(k + base)*2 + 1] = f'{lu} (irr)'

    # Expand the value range to avoid mem-spill-out
    lumap_old, lmmap_old , lumap_new , lmmap_new = (i.astype(np.int16) for i in  [lumap_old, lmmap_old , lumap_new , lmmap_new])

    highpos_old = np.where(lmmap_old == 0, (lumap_old * 2).astype(np.int16), lumap_old * 2 + 1)
    highpos_new = np.where(lmmap_new == 0, (lumap_new * 2).astype(np.int16), lumap_new * 2 + 1)

    # Produce the cross-tabulation matrix with labels.
    crosstab = pd.crosstab(highpos_old,         
                            highpos_new,        
                            values = real_area,        
                            aggfunc = lambda x:x.sum(),         
                            margins=False)

    # Make sure the cross-tabulation matrix is square.
    crosstab = crosstab.rename(index=ludict,columns=ludict)
    crosstab.index = crosstab.index.tolist()
    crosstab.columns = crosstab.columns.tolist()
    crosstab = crosstab.fillna(0)

    # Calculate net switches to land use (negative means switch away).
    df = pd.DataFrame()
    area_km2_prior = crosstab.sum(axis=1)
    area_km2_after = crosstab.sum(axis=0)
    df['Area prior [ km2 ]'] = area_km2_prior
    df['Area after [ km2 ]'] = area_km2_after
    df.fillna(0, inplace=True)
    switches = df['Area after [ km2 ]'] - df['Area prior [ km2 ]']
    nswitches = np.abs(switches).sum()
    pswitches = np.around(100 * nswitches / (real_area.sum()/100), decimals=2)

    df['Switches [ km2 ]'] = switches
    df['Switches [ % ]'] = 100 * switches / area_km2_prior
    df.loc['Total'] = area_km2_prior.sum(), area_km2_after.sum(), nswitches, pswitches

    # Rearrange crosstab to long format
    crosstab = crosstab.stack().reset_index().rename(columns={0: 'Area (km2)'})
    crosstab[['From land-use','From water_supply','tmp' ]] = crosstab['level_0'].str.split(r'\((dry|irr)', expand=True)
    crosstab[['To land-use', 'To water_supply', 'tmp']] = crosstab['level_1'].str.split(r'\((dry|irr)', expand=True)
    crosstab = crosstab.replace('irr', 'Irrigated').replace('dry', 'Dryland').drop(['level_0','level_1', 'tmp' ], axis=1)

    return crosstab, df


def crossmap_amstat( am
                   , lumap_old
                   , ammap_old
                   , lumap_new
                   , ammap_new
                   , ag_landuses
                   , non_ag_landuses
                   , real_area):
    """
    For a given agricultural management option, make a crossmap showing cell changes at a land use level.
    """
    ludict = {-2: 'Non-agricultural land (None)'}
    
    for j, lu in enumerate(list(ag_landuses)):
        ludict[j*2] = f"{lu} (None)"
        ludict[j*2 + 1] = f"{lu} ({am})"

    base = settings.NON_AGRICULTURAL_LU_BASE_CODE
    for k, lu in enumerate(list(non_ag_landuses)):
        ludict[(k + base)*2] = f"{lu} (None)"
        ludict[(k + base)*2 + 1] = f"{lu} ({am})"

    # Collect all j values that apply to the given AM option
    am_j = [j for j, lu_name in enumerate(ag_landuses) if lu_name in settings.AG_MANAGEMENTS_TO_LAND_USES[am]]
    am_j = np.array(am_j)

    # Encode information about the given AM usage by doubling up land uses
    highpos_old = np.where(ammap_old == 0, lumap_old.astype(np.int16) * 2, lumap_old * 2 + 1)
    highpos_new = np.where(ammap_new == 0, lumap_new.astype(np.int16) * 2, lumap_new * 2 + 1)

    # Produce the cross-tabulation matrix with labels.
    crosstab = pd.crosstab(highpos_old, 
                           highpos_new,
                           values=real_area, 
                           aggfunc=lambda x:x.sum() , 
                           margins=False)

    # Include all land uses and AM versions of land uses in the crosstab
    crosstab = crosstab.rename(index=ludict, columns=ludict)
    crosstab = crosstab.reindex(index=list(ludict.values()), columns=list(ludict.values()),fill_value=0)

    crosstab.index = crosstab.index.tolist()
    crosstab.columns = crosstab.columns.tolist()
    crosstab = crosstab.fillna(0)

    # Calculate net switches to land use (negative means switch away).
    df = pd.DataFrame()
    am_idx = [2*i + 1 for i in am_j]
    am_lus = [ludict[i] for i in am_idx]
    area_km2_prior = crosstab.iloc[:, -1].loc[am_lus]
    area_km2_after = crosstab.iloc[-1, :].loc[am_lus]
    df['Area prior [ km2 ]'] = area_km2_prior
    df['Area after [ km2 ]'] = area_km2_after
    df.fillna(0, inplace=True)
    switches = df['Area after [ km2 ]']-  df['Area prior [ km2 ]']
    nswitches = np.abs(switches).sum()
    pswitches = np.around(100 * nswitches / (real_area.sum()/100),decimals=2)

    df['Switches [ km2 ]'] = switches
    df['Switches [ % ]'] = 100 * switches / area_km2_prior
    df.loc['Total'] = area_km2_prior.sum(), area_km2_after.sum(), nswitches, pswitches
    
    # Rearrange crosstab to long format
    crosstab = crosstab.stack().reset_index().rename(columns={0: 'Area (km2)'})
    crosstab[['From land-use','From am','tmp']] = crosstab['level_0'].str.split(r' \(', expand=True)
    crosstab[['To land-use', 'To am', 'tmp']] = crosstab['level_1'].str.split(r' \(', expand=True)
    crosstab['From am'] = crosstab['From am'].str.replace(')', '', regex=False)
    crosstab['To am'] = crosstab['To am'].str.replace(')', '', regex=False)
    crosstab = crosstab.drop(['level_0','level_1','tmp'], axis=1)

    return crosstab, df
