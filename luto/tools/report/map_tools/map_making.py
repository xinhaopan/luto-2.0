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
import rasterio
from rasterio.merge import merge

import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from luto.tools.report.map_tools.helper import download_basemap
from luto.tools.report.map_tools.parameters import  DPI, FIG_SIZE, MAP_BACKGROUND_DESC_FLOAT





def create_png_map(tif_path: str, 
                   data_type: str,
                   color_desc_dict: dict,
                   basemap_path: str = 'luto/tools/report/Assets/basemap.tif', 
                   shapefile_path: str = 'luto/tools/report/Assets/AUS_adm/STE11aAust_mercator_simplified.shp',
                   anno_text: str = None,
                   mercator_bbox: tuple[int] = None,
                   legend_params: dict = None):

    """
    Creates a PNG map by overlaying a raster image with a basemap, shapefile, annotation, scale bar, north arrow, and legend.

    Parameters
    - tif_path (str): 
        The path to the input raster image.
    - data_type (str):
        The type of the data. It can be either 'integer' or 'float'.
    - color_desc_dict (dict): 
        A dictionary mapping color values to their descriptions for the legend.
    - basemap_path (str): 
        The path to the basemap image. Default is 'Assets/basemap.tif'.
    - shapefile_path (str): 
        The path to the shapefile for overlaying. Default is 'luto/tools/report/Assets/AUS_adm/STE11aAust_mercator_simplified.shp'.
    - anno_text (str): 
        The annotation text to be displayed on the map. Default is None.
    - mercator_bbox (tuple[int]):
        The bounding box in Mercator projection (west, south, east, north). Default is None.
    - legend_params (dict):
        The parameters for the legend. Default is None.

    Returns
    - None
    """

    
    # Download basemap if it does not exist
    if not os.path.exists(basemap_path):
        if mercator_bbox is None:
            raise ValueError("The bounding box in Mercator projection (w,s,e,n) is required to download the basemap.")
        print("Downloading basemap...")
        print('This Could take a while ...')
        print('Only download once ...')
        download_basemap(mercator_bbox)


    # Get the mercator input image
    out_base = os.path.splitext(tif_path)[0]
    in_mercator_path = f"{out_base}_mercator.tif"
    png_out_path = f"{out_base}_basemap.png"

    # Mosaic the raster with the basemap
    with rasterio.open(in_mercator_path) as src, rasterio.open(basemap_path) as base:
        # Mosaic the raster with the basemap
        mosaic, out_transform = merge([src, base], res=base.res)

    # Get the shape of the mosaic array
    array_shape = mosaic.shape[-2:]  # (height, width)  
    # Calculate the extent
    left, bottom, right, top = rasterio.transform.array_bounds(array_shape[0], array_shape[1], out_transform)


    # Create the figure and axis
    fig, ax = plt.subplots(figsize=FIG_SIZE, dpi=DPI)

    # Display the array with the correct extent
    ax.imshow(mosaic.transpose(1,2,0), extent=[left, right, bottom, top], interpolation='none')

    # Add annotation
    plt.annotate(anno_text, 
        xy=(0.05, 0.92), 
        xycoords='axes fraction',
        fontsize=15,
        #  fontweight = 'bold',
        ha='left', 
        va='center')

    # Overlay the shapefile
    # Load the shapefile with GeoPandas
    gdf = gpd.read_file(shapefile_path)
    gdf.boundary.plot(ax=ax, 
              color='grey', 
              linewidth=0.5, 
              edgecolor='grey', 
              facecolor='none')

    # Create legend
    if data_type == 'float':
        decorate_float_plot(color_desc_dict, legend_params, fig)
    elif data_type == 'integer':
        patches = [mpatches.Patch(color=tuple(value / 255 for value in k), label=v) 
                for k, v in color_desc_dict.items()]

        plt.legend(handles=patches, **legend_params)

    # Optionally remove axis
    ax.set_axis_off()
    plt.savefig(png_out_path, dpi=DPI, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

    # Delete the input raster
    os.remove(in_mercator_path)



def decorate_float_plot(color_desc_dict, legend_params, fig):
    # Add a legend
    legend_val = {
        MAP_BACKGROUND_DESC_FLOAT[v]: tuple(value / 255 for value in k)
        for k, v in color_desc_dict.items()
        if v < 1 or v > 100
    }

    patches = [mpatches.Patch(color=v, label=k) 
               for k,v in legend_val.items()]

    plt.legend(handles=patches, **legend_params)


    # Get the value for colorbar
    color_bar_val = [tuple(value / 255 for value in k) for k,v in color_desc_dict.items()  
                    if (v >= 1 and v <= 100)]

    # Create a colormap from your colors
    cmap = mpl.colors.ListedColormap(color_bar_val)

    # Each pixel in the float-datatype map is a float value between 0 and 1
    # Meaning the proportion of the land-use within this pixel 
    vmin, vmax = 0, 1
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax) 

    # Create a ScalarMappable object which will use the colormap and normalization
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    # Create a new axes at the desired position
    cbar_ax = fig.add_axes([0.48, 0.21, 0.05, 0.08])
    # Create the colorbar on the new axes and make it horizontal
    cbar = plt.colorbar(sm, cax=cbar_ax, orientation='vertical')

    # Add a label to the colorbar
    cbar.set_label('Proportion of grid cell', fontsize=12, labelpad=-30, y=1.25, rotation=0)