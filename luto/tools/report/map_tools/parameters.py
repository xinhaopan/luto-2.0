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

from luto.settings import AG_MANAGEMENTS_TO_LAND_USES
from luto.tools.report.data_tools.parameters import AG_LANDUSE, LANDUSE_ALL_RAW, RENAME_AM_NON_AG


# The figure size and DPI for PNG map
FIG_SIZE = (11.2, 13.6)
DPI = 300


# The val-color(HEX) records for each map type
color_types ={
            # Integer rasters
            'lumap':  'luto/tools/report/Assets/lumap_colors_grouped.csv',
            'lmmap': 'luto/tools/report/Assets/lm_colors.csv',
            'ammap': 'luto/tools/report/Assets/ammap_colors.csv',
            'non_ag':'luto/tools/report/Assets/non_ag_colors.csv',
            # Float rasters
            'Ag_LU': 'luto/tools/report/Assets/float_img_colors.csv',
            'Ag_Mgt': 'luto/tools/report/Assets/float_img_colors.csv',
            'Land_Mgt':'luto/tools/report/Assets/float_img_colors.csv',
            'Non-Ag':'luto/tools/report/Assets/float_img_colors.csv'
            }


map_multiple_lucc = {
             'lumap': 'Land-use all category',
             'lmmap': 'Dryland/Irrigated Land-use',
             'ammap': 'Agricultural Management',
             'non_ag': 'Non-Agricultural Land-use',
             'dry': 'Dryland',
             'irr': 'Irrigated Land',
             }

map_single_lucc = AG_LANDUSE + list(AG_MANAGEMENTS_TO_LAND_USES.keys()) + LANDUSE_ALL_RAW
map_single_lucc = {k:k for k in map_single_lucc}

# Dictionary {k:v} for renaming the map names
# if <k> exists in the map name, the map full name will be <v>
map_basename_rename = map_multiple_lucc | map_single_lucc | RENAME_AM_NON_AG


# The extra colors for the float rasters
extra_color_float_tif = {
  0:(200, 200, 200, 255),     # 0 is the non-Agriculture land in the raw tif file
  -100:(225, 225, 225, 255)   # -100 refers to the nodata pixels in the raw tif file
} 

extra_desc_float_tif = {
  0: 'Agricultural land',
  -100: 'Non-Agriculture land'
}  


# The data types for each map type
data_types = {'lumap': 'integer',
              'lmmap': 'integer',
              'ammap': 'integer',
              'non_ag': 'integer',
              'Ag_LU': 'float',
              'Ag_Mgt': 'float',
              'Land_Mgt': 'float',
              'Non-Ag': 'float'
            }


# The parameters for legend
legend_params = {'lumap': {'bbox_to_anchor': (0.02, 0.19),
                            'loc': 'upper left',
                            'ncol': 2,
                            'fontsize': 10,
                            'framealpha': 0,
                            'columnspacing': 1},

                 'lmmap': {'bbox_to_anchor': (0.1, 0.25),
                            'loc': 'upper left',
                            'ncol': 1,
                            'fontsize': 10,
                            'framealpha': 0,
                            'columnspacing': 1},

                 'ammap': {'bbox_to_anchor': (0.02, 0.19),
                            'loc': 'upper left',
                            'ncol': 2,
                            'fontsize': 10,
                            'framealpha': 0,
                            'columnspacing': 1},

                 'non_ag': {'bbox_to_anchor': (0.02, 0.19),
                            'loc': 'upper left',
                            'ncol': 2,
                            'fontsize': 10,
                            'framealpha': 0,
                            'columnspacing': 1},

                 'Ag_LU': {'bbox_to_anchor': (0.05, 0.23),
                            'loc': 'upper left',
                            'ncol': 1,
                            'labelspacing': 2.0,
                            'fontsize': 15,
                            'framealpha': 0},

                 'Ag_Mgt': {'bbox_to_anchor': (0.05, 0.23),
                            'loc': 'upper left',
                            'ncol': 1,
                            'labelspacing': 2.0,
                            'fontsize': 15,
                            'framealpha': 0},

                 'Land_Mgt': {'bbox_to_anchor': (0.05, 0.23),
                            'loc': 'upper left',
                            'ncol': 1,
                            'labelspacing': 2.0,
                            'fontsize': 15,
                            'framealpha': 0},

                 'Non-Ag': {'bbox_to_anchor': (0.05, 0.23),
                            'loc': 'upper left',
                            'ncol': 1,
                            'labelspacing': 2.0,
                            'fontsize': 15,
                            'framealpha': 0},
                }
