# write_png = False
# intermediate_files = True
# add_legend = False
# add_scalebar = False
# add_north_arrow = False
#
# basemap_geo_tif = "Assets/basemap.tif"
# # output file
# path = '../output'
# basemap_rgb_tif = f"{path}/basemap_rgb_tif.tif"
# overlay_reproject_geo_tif = f"{path}/overlay_reproject_geo_tif.tif"
# overlay_reproject_rgb_tif = f"{path}/overlay_reproject_rgb_tif.tif"
# overlay_rgb_tif = f"{path}/overlay_rgb_tif.tif"
# crop_rbg_tif = f"{path}/crop_rbg_tif.tif"
# shp_add_rgb_tif = f"{path}/shp_add_rgb_tif.tif"
# scalebar_rgb_tif = f"{path}/scalebar_rgb_tif.tif"
# north_png = f"{path}/north_png.png"
# legend_png = f"{path}/legend_png.png"
#
# # Reproject overlay parameters
# reproject_overlay_background_value = -9999
#
# # Assign colors to overlay parameters
# assign_colors_background_value = -9999
#
# # Crop map parameters
# crop_map_buffer_size = 1
#
# # Add shapefile (SHP) overlay parameters
# shp_path = "Assets/shp/GCCSA_line.shp"
# shapefile_color = '#8F8F8F'
# shapefile_linewidth = 1
#
# # Scalebar parameters
# scalebar_unit = 'km'
# scalebar_length = 500
# scalebar_location = (0.1, 0.95)
# scalebar_lw = 1
# scalebar_fontsize = 4
# scalebar_fontname = 'Arial'
#
# # Legend parameters
# legend_png_name = "legend.png"
# legend_title = ""
# legend_fontsize = 8
# legend_title_fontsize = 8
# legend_ncol = 4
# legend_figsize = (6, 2)
#
# # Add legend to image parameters
# legend_location = (0.07, 0.67)
# legend_scale = 0.7
#
# # North arrow parameters
# north_arrow_zoom = 0.15
# north_arrow_location = (0.9, 0.9)
# arrow_image_path = "Assets/north_arrow.png"

class MapConfig:
    def __init__(self, path='../output', input_name='', source_name=''):
        self.write_png = False
        self.remove_intermediate_files = True
        self.add_legend = False
        self.add_scalebar = False
        self.add_north_arrow = False
        self.reproject_overlay_background_value = -9999
        self.assign_colors_background_value = -9999
        self.crop_map_buffer_size = 1

        # 基础路径
        self.path = path
        self.input_name = input_name
        self.source_name = source_name

        # 更新后的路径和文件
        self.update_paths()

        # 地图参数
        self.shp_path = "Assets/shp/GCCSA_line.shp"
        self.shapefile_color = '#8F8F8F'
        self.shapefile_linewidth = 1

        self.scalebar_unit = 'km'
        self.scalebar_length = 500
        self.scalebar_location = (0.1, 0.95)
        self.scalebar_lw = 1
        self.scalebar_fontsize = 4
        self.scalebar_fontname = 'Arial'

        self.legend_title = ""
        self.legend_fontsize = 8
        self.legend_title_fontsize = 8
        self.legend_ncol = 4
        self.legend_figsize = (6, 2)
        self.legend_location = (0.07, 0.67)
        self.legend_scale = 0.7

        self.north_arrow_zoom = 0.15
        self.north_arrow_location = (0.9, 0.9)
        self.arrow_image_path = "Assets/north_arrow.png"

    def update_paths(self):
        """根据输入名称和来源名称更新文件路径"""
        base_name = f"{self.path}/{self.input_name}_{self.source_name}"

        self.basemap_geo_tif = "Assets/basemap.tif"
        self.basemap_rgb_tif = f"{base_name}_basemap_rgb_tif.tif"
        self.overlay_reproject_geo_tif = f"{base_name}_overlay_reproject_geo_tif.tif"
        self.overlay_reproject_rgb_tif = f"{base_name}_overlay_reproject_rgb_tif.tif"
        self.overlay_rgb_tif = f"{base_name}_overlay_rgb_tif.tif"
        self.crop_rbg_tif = f"{base_name}_crop_rbg_tif.tif"
        self.shp_add_rgb_tif = f"{base_name}_shp_add_rgb_tif.tif"
        self.scalebar_rgb_tif = f"{base_name}_scalebar_rgb_tif.tif"
        self.north_png = f"{base_name}_north_png.png"
        self.legend_png = f"{base_name}_legend_png.png"

    def get_intermediate_files(self):
        """返回所有中间文件路径的列表"""
        return [
            self.basemap_rgb_tif, self.overlay_reproject_geo_tif, self.overlay_reproject_rgb_tif,
            self.overlay_rgb_tif, self.crop_rbg_tif, self.shp_add_rgb_tif, self.scalebar_rgb_tif,
            self.north_png, self.legend_png
        ]
