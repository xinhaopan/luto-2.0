write_png = False
intermediate_files = True
add_legend = False
add_scalebar = False
add_north_arrow = False

basemap_geo_tif = "Assets/basemap.tif"
basemap_rgb_tif = "basemap_rgb_tif.tif"
overlay_reproject_geo_tif = "overlay_reproject_geo_tif.tif"
overlay_reproject_rgb_tif = "overlay_reproject_rgb_tif.tif"
overlay_rgb_tif = "overlay_rgb_tif.tif"
crop_rbg_tif = "crop_rbg_tif.tif"
shp_add_rgb_tif = "shp_add_rgb_tif.tif"
scalebar_rgb_tif = "scalebar_rgb_tif.tif"
north_png = "north_png.png"
legend_png = "legend_png.png"

# Reproject overlay parameters
reproject_overlay_background_value = -9999

# Assign colors to overlay parameters
assign_colors_background_value = -9999

# Crop map parameters
crop_map_buffer_size = 1

# Add shapefile (SHP) overlay parameters
shp_path = "Assets/shp/GCCSA_line.shp"
shapefile_color = '#8F8F8F'
shapefile_linewidth = 1

# Scalebar parameters
scalebar_unit = 'km'
scalebar_length = 500
scalebar_location = (0.1, 0.95)
scalebar_lw = 1
scalebar_fontsize = 4
scalebar_fontname = 'Arial'

# Legend parameters
legend_png_name = "legend.png"
legend_title = ""
legend_fontsize = 8
legend_title_fontsize = 8
legend_ncol = 4
legend_figsize = (6, 2)

# Add legend to image parameters
legend_location = (0.07, 0.67)
legend_scale = 0.7

# North arrow parameters
north_arrow_zoom = 0.15
north_arrow_location = (0.9, 0.9)
arrow_image_path = "Assets/north_arrow.png"
