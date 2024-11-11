from .map_helper import *

def draw_maps(overlay_geo_tif, colors_sheet, output_png, legend_png_name, cfg):
    intermediate_files = cfg.get_intermediate_files()

    # 1. 生成基础地图
    generate_basemap(cfg.basemap_geo_tif, cfg.basemap_rgb_tif, write_png=cfg.write_png)

    # 2. 重投影叠加图
    reproject_overlay(overlay_geo_tif, cfg.basemap_geo_tif, cfg.overlay_reproject_geo_tif, write_png=cfg.write_png)

    # 3. 给叠加图分配颜色
    assign_colors_to_overlay(cfg.overlay_reproject_geo_tif, colors_sheet, cfg.overlay_reproject_rgb_tif, write_png=cfg.write_png)

    # 4. 叠加底图和重投影的叠加图
    overlay_map(cfg.basemap_rgb_tif, cfg.overlay_reproject_rgb_tif, cfg.overlay_rgb_tif, write_png=cfg.write_png)

    # 5. 根据叠加图范围裁剪底图
    crop_map(cfg.overlay_rgb_tif, overlay_geo_tif, cfg.crop_rbg_tif, write_png=cfg.write_png)

    # 6. 添加 Shapefile
    add_shp(cfg.crop_rbg_tif, cfg.shp_path, cfg.shp_add_rgb_tif, shapefile_color=cfg.shapefile_color,
            linewidth=cfg.shapefile_linewidth, write_png=cfg.write_png)

    # 7. 添加比例尺（如需要）
    final_tif = cfg.shp_add_rgb_tif  # 默认最后的 TIFF 文件路径
    if cfg.add_scalebar:
        add_scalebar(final_tif, cfg.scalebar_rgb_tif, unit=cfg.scalebar_unit, length=cfg.scalebar_length,
                     location=cfg.scalebar_location, lw=cfg.scalebar_lw, fontsize=cfg.scalebar_fontsize, write_png=cfg.write_png)
        final_tif = cfg.scalebar_rgb_tif  # 更新最终文件路径

    # 8. 始终将 TIFF 转换为 `output_png`
    convert_tif_to_png(final_tif, output_png)

    # 9. 创建图例（无论是否添加到最终图像）
    create_legend(colors_sheet, legend_png_name=legend_png_name, legend_title=cfg.legend_title,
                  legend_fontsize=cfg.legend_fontsize, legend_title_fontsize=cfg.legend_title_fontsize,
                  legend_ncol=cfg.legend_ncol, legend_figsize=cfg.legend_figsize)

    # 10. 添加指北针（如需要）
    if cfg.add_north_arrow:
        add_north_arrow_to_png(output_png, output_png, cfg.arrow_image_path,
                               location=cfg.north_arrow_location, zoom=cfg.north_arrow_zoom)

    # 11. 添加图例（如需要）
    if cfg.add_legend:
        add_legend_to_image(output_png, legend_png_name, output_png, legend_location=cfg.legend_location,
                            legend_scale=cfg.legend_scale)
        intermediate_files.append(legend_png_name)  # 添加到中间文件列表

    # 删除中间文件
    if cfg.remove_intermediate_files:
        for file in intermediate_files:
            if os.path.exists(file):
                os.remove(file)
            png_file = os.path.splitext(file)[0] + ".png"
            if os.path.exists(png_file):
                os.remove(png_file)

    # print(f"Finish: {overlay_geo_tif}")
