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


import pandas as pd
import contextily as ctx
import matplotlib as mpl

from branca.colormap import LinearColormap

from luto.tools.report.map_tools.parameters import (
    COLOR_TYPES,
    DATA_TYPES,
    LEGEND_PARAMS,
    MAP_BASENAME_RENAME,
    MAP_BACKGROUND_COLORS_FLOAT
)


# Function to download a basemap image
def download_basemap(bounds_mercator: list[str]):
    """
    Downloads a basemap image within the specified bounds in Mercator projection.

    Args:
        bounds_mercator (BoundingBox): The bounding box in Mercator projection. Defaults to None.

    Returns
        tuple: A tuple containing the downloaded basemap image and its extent.
    """

    base_map, extent = ctx.bounds2raster(*bounds_mercator, 
                                        path='luto/ools/report/Assets/basemap.tif',
                                        source=ctx.providers.OpenStreetMap.Mapnik,
                                        zoom=7,
                                        n_connections=16,
                                        max_retries=4)
    return base_map, extent
                          

def create_color_csv_1_100(color_scheme:str='YlOrRd',
                           save_path:str='luto/tools/report/Assets/float_img_colors.csv',
                           extra_color:dict=MAP_BACKGROUND_COLORS_FLOAT):
    """
    Create a CSV file contains the value(1-100)-color(HEX) records.

    Parameters
    - color_scheme (str): 
        The name of the color scheme to use. Default is 'YlOrRd'.
    - save_path (str): 
        The file path to save the color dictionary as a CSV file. Default is 'Assets/float_img_colors.csv'.
    - extra_color (dict): 
        Additional colors to include in the dictionary. Default is {-100:(225, 225, 225, 255)}.

    Returns
        None
    """
    colors = mpl.colormaps[color_scheme]
    val_colors_dict = {i: colors(i/100) for i in range(1,101)}
    var_colors_dict = {k:tuple(int(num*255) for num in v) for k,v in val_colors_dict.items()}
    
    
    # If extra colors are specified, add them to the dictionary
    if extra_color:
        var_colors_dict.update(extra_color) 
    
    # Convert the RGBA values to HEX color codes
    var_colors_dict = {k: f"#{''.join(f'{c:02X}' for c in v)}" 
                       for k, v in var_colors_dict.items()}
    
    # Save the color dictionary to a CSV file
    color_df = pd.DataFrame(var_colors_dict.items(), columns=['lu_code', 'lu_color_HEX'])
    color_df.to_csv(save_path, index=False)
    
    
def get_map_meta():
    """
    Get the map making metadata.

    Returns
        map_meta (DataFrame): DataFrame containing map metadata with columns 'category', 'csv_path', 'legend_type', and 'legend_position'.
    """
    
    # Create a DataFrame from the COLOR_TYPES dictionary
    map_meta = pd.DataFrame(
        COLOR_TYPES.items(), 
        columns=['category', 'color_csv']
    )
 
    # Add other metadata columns to the DataFrame
    map_meta['data_type'] = map_meta['category'].map(DATA_TYPES)
    map_meta['legend_params'] = map_meta['category'].map(LEGEND_PARAMS)
    
    
    return map_meta.reset_index(drop=True)


def get_map_fullname(path:str):
    """
    Get the full name of a map based on its path.

    Args:
        path (str): The path of the map.

    Returns
        str: The full name of the map.
    """
    for k,v in MAP_BASENAME_RENAME.items():
        if k in path:
            return v


def get_scenario(data_root_dir:str):
    """
    Get the scenario name from the data root directory.

    Args:
        data_root_dir (str): The data root directory.

    Returns
        str: The scenario name.
    """
    GHG_scenario = ''
    bio_scenario = ''
    with open(f'{data_root_dir}/model_run_settings.txt', 'r') as f:
        for line in f:
            if 'GHG_EMISSIONS_LIMITS' in line:
                GHG_scenario = line.split(':')[-1].strip()
            if 'BIODIVERSITY_TARGET_GBF_2' in line:
                bio_scenario = line.split(':')[-1].strip()
        
    return f'GHG {GHG_scenario} - Biodiversity {bio_scenario}'


def get_legend_elemet(color_desc_dict:dict, map_dtype:str='float'):
    
    if map_dtype == 'integer':

        legend_css_list = [
            f'<p><a style="color:transparent;background-color:rgba{color_rgba};">__   </a>&emsp;{color_desc}</p>\n'
            for color_rgba, color_desc in color_desc_dict.items()
        ]
        
        legend_css = "".join(legend_css_list)

        
        # Create a custom HTML template for the legend
        template = f"""
        {{% macro html(this, kwargs) %}}
        <div id='legend'
            style="position: fixed; 
                   padding: 10px;
                   bottom: 30px;
                   left: 30px;
                   width: auto;
                   height: auto;
                   z-index:9999;
                   font-size:14px;
                   background-color: rgba(255, 255, 255, 0.7);
                   border-radius: 10px;">
            {legend_css}
        </div>
        {{% endmacro %}}
        """
        
    elif map_dtype == 'float':
    
        # Sort the dictionary by values
        color_desc_dict_sorted = dict(sorted(color_desc_dict.items(), key=lambda item: item[1]))

        # Filter the dictionary to include only values between 1 and 100
        color_desc_dict_filtered = {k: v for k, v in color_desc_dict_sorted.items() if 1 <= v <= 100}

        # Create a color map
        colors = [k for k in color_desc_dict_filtered.keys()]
        index = [v/100 for v in color_desc_dict_filtered.values()]
        color_map = LinearColormap(colors, index=index, caption= "Proportion of grid cell")
        color_map_str = color_map._repr_html_()
        
        # Add the color map to the folium map as a legend
        template = f"""
        {{% macro html(this, kwargs) %}}
        <div id='legend' 
            style= 'position: fixed; 
                    padding: 6px;
                    bottom: 30px; 
                    left: 50px; 
                    width: auto; 
                    height: auto; 
                    z-index:9999;
                    background-color: rgba(255, 255, 255, 0.7);
                    border-radius: 10px;'>

            { color_map_str }

        </div>
        {{% endmacro %}}
        """
        
    else:
        raise ValueError('Invalid map_dtype. Must be either "integer" or "float"')
    
    return template