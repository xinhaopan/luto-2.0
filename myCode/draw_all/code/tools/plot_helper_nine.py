import pandas as pd
import numpy as np
import math
import re
from lxml import etree
import cairosvg
from joblib import Parallel, delayed
from plotnine import (
    ggplot, aes, geom_area, geom_col, geom_line, geom_point,
    theme, element_text, element_blank, element_rect, element_line,
    scale_fill_manual, scale_color_manual, scale_x_continuous, scale_y_continuous,
    labs, xlim, ylim, guides, guide_legend, facet_wrap, facet_grid,
    theme_minimal, ggsave, annotate, coord_cartesian
)
from mizani.breaks import extended_breaks
import matplotlib.pyplot as plt
from io import BytesIO
import os

from tools.parameters import COLUMN_WIDTH, X_OFFSET, axis_linewidth


def get_colors(merged_dict, mapping_file, sheet_name=None):
    """
    根据映射文件处理 merged_dict 中的每个表，忽略 `-`，并返回 data_dict 和 legend_colors。

    参数:
    merged_dict (dict): 包含多个 DataFrame 的字典。
    mapping_file (str): 映射文件路径，包含 desc 和 color 列。
    sheet_name (str, 可选): 如果指定，读取映射文件中的指定 sheet。

    返回:
    tuple: data_dict（处理后的 DataFrame 字典）和 legend_colors（图例名称与颜色的对应关系）。
    """
    # 读取映射文件
    if sheet_name:
        mapping_df = pd.read_excel(mapping_file, sheet_name=sheet_name)
    else:
        mapping_df = pd.read_csv(mapping_file)

    # 处理 merged_dict 中的每个 DataFrame，获取处理后的 DataFrame 字典和 legend_colors
    data_dict = {}
    legend_colors = {}
    for key, df in merged_dict.items():
        processed_df, color_dict = process_single_df(df, mapping_df)
        data_dict[key] = processed_df
        legend_colors.update(color_dict)  # 合并每个表的颜色映射

    return data_dict, legend_colors


def process_single_df(df, mapping_df):
    """
    根据映射文件过滤和处理 DataFrame，仅保留能匹配上的列名。

    参数:
    df (pd.DataFrame): 需要处理的 DataFrame。
    mapping_df (pd.DataFrame): 包含 desc 和 color 列的映射文件 DataFrame。

    返回:
    tuple: 处理后的 DataFrame 和 legend_colors（仅包含匹配的列）。
    """
    # 创建映射字典，忽略 desc 中的 `-` 和大小写
    mapping_df['desc_processed'] = mapping_df['desc'].str.replace('-', '', regex=True).str.lower()
    column_mapping = {row['desc_processed']: row['desc'] for _, row in mapping_df.iterrows()}
    color_mapping = {row['desc']: row['color'] for _, row in mapping_df.iterrows()}

    # 获取并处理 DataFrame 列名，忽略 `-` 和大小写
    original_columns = list(df.columns)
    processed_columns = [re.sub(r'-', '', col).lower() for col in original_columns]

    # 匹配列名并过滤掉无法匹配的列
    matched_indices = [i for i, col in enumerate(processed_columns) if col in column_mapping]
    matched_original_columns = [original_columns[i] for i in matched_indices]
    matched_renamed_columns = [column_mapping[processed_columns[i]] for i in matched_indices]

    # 检查长度是否一致
    if len(matched_indices) != len(matched_renamed_columns):
        raise ValueError("Mismatch between matched indices and renamed columns.")

    # 筛选 legend_colors 只包含匹配的列
    legend_colors = {column_mapping[processed_columns[i]]: color_mapping[column_mapping[processed_columns[i]]]
                     for i in matched_indices}

    # 重命名和筛选 DataFrame
    filtered_df = df[matched_original_columns].rename(
        columns={matched_original_columns[i]: matched_renamed_columns[i] for i in range(len(matched_indices))}
    )
    if 'Year' in filtered_df.columns:
        filtered_df = filtered_df.set_index('Year')

    # 检查是否有 'desc_new' 列
    if 'desc_new' in mapping_df.columns:
        # 创建映射字典 {旧列名: 新列名}
        rename_dict = dict(zip(mapping_df['desc'], mapping_df['desc_new']))
        # 仅重命名 filtered_df 中存在的列
        filtered_df = filtered_df.rename(
            columns={col: rename_dict[col] for col in filtered_df.columns if col in rename_dict})
        legend_colors = {rename_dict.get(key, key): value for key, value in legend_colors.items()}

    return filtered_df, legend_colors



def plot_Combination_figures(merged_dict, output_png, input_names, plot_func, legend_colors,
                                      point_dict=None, point_colors=None, n_rows=3, n_cols=3,
                                      font_size=10, x_range=(2010, 2050), y_range=(-600, 100),
                                      x_ticks=5, y_ticks=[-200, 0, 200],
                                      legend_position="bottom", show_legend='last', legend_n_rows=1):
    """
    Creates a grid of plots using plotnine and saves them.

    Parameters:
        merged_dict (dict): Dictionary containing data frames.
        output_png (str): Path to save the output.
        input_names (list): List of keys in merged_dict to plot.
        plot_func (function): Function to create individual plots.
        legend_colors (dict): Dictionary mapping categories to colors.
        point_dict (dict, optional): Dictionary with point data for line plots.
        point_colors (list, optional): List of colors for points.
        n_rows (int): Number of rows in the grid.
        n_cols (int): Number of columns in the grid.
        font_size (int): Font size for text elements.
        x_range (tuple): Range for x-axis.
        y_range (tuple): Range for y-axis.
        x_ticks (int/list): Ticks for x-axis.
        y_ticks (list): Ticks for y-axis.
        legend_position (str): Position of the legend.
        show_legend (str): When to show the legend ('last' or 'all').
        legend_n_rows (int): Number of rows in the legend.
    """
    total_plots = len(input_names)
    fig_width = 12
    fig_height = 8

    # Create individual plots for each input name
    all_plots = []

    for i, input_name in enumerate(input_names):
        if i < total_plots:
            # Determine if this is a left column or bottom row plot
            is_left_col = (i % n_cols == 0)
            is_bottom_row = (i >= (n_rows - 1) * n_cols)

            # Set legend visibility based on show_legend parameter
            show_legend_for_plot = (show_legend == 'all') or (show_legend == 'last' and i == total_plots - 1)

            # Check which parameters the plot_func accepts and create a parameter dictionary
            import inspect
            func_params = inspect.signature(plot_func).parameters.keys()

            # Base parameters that all plot functions should accept
            plot_params = {
                'merged_dict': merged_dict,
                'input_name': input_name,
                'legend_colors': legend_colors,
                'font_size': font_size,
                'x_range': x_range,
                'y_range': y_range,
                'show_legend': show_legend_for_plot
            }

            # Add optional parameters only if the function accepts them
            if 'x_ticks' in func_params:
                plot_params['x_ticks'] = x_ticks
            if 'y_ticks' in func_params:
                plot_params['y_ticks'] = y_ticks
            if 'point_dict' in func_params and point_dict is not None:
                plot_params['point_dict'] = point_dict
            if 'point_colors' in func_params and point_colors is not None:
                plot_params['point_colors'] = point_colors

            # Create the plot using the provided function with appropriate parameters
            p = plot_func(**plot_params)

            # Add custom theming for consistent grid appearance
            # 添加自定义主题设置
            p = p + theme(
                # 控制轴文本显示
                axis_text_x=element_text(size=font_size) if is_bottom_row else element_blank(),
                axis_text_y=element_text(size=font_size) if is_left_col else element_blank(),

                # 移除轴标题
                axis_title_x=element_blank(),
                axis_title_y=element_blank(),

                # 网格线设置
                panel_grid_minor=element_blank(),
                panel_grid_major_x=element_blank(),
                panel_grid_major_y=element_line(linetype='dashed', size=1.5),

                # 边框设置
                panel_border=element_blank(),

                # 坐标轴线设置 - 使用plotnine支持的元素名称
                axis_line_x=element_line(size=axis_linewidth) if is_bottom_row else element_blank(),
                axis_line_y=element_line(size=axis_linewidth) if is_left_col else element_blank(),

                # 图表边距和图例位置
                plot_margin=0.01,
                legend_position='none'  # 稍后会创建单独的图例
            )

            all_plots.append(p)

    # Add empty plots if needed to fill the grid
    while len(all_plots) < n_rows * n_cols:
        empty_plot = ggplot() + theme_minimal() + theme(panel_border=element_blank())
        all_plots.append(empty_plot)

    # Save individual plots and combine them using facet_wrap later
    plot_paths = []
    for i, p in enumerate(all_plots[:total_plots]):
        temp_path = f"{output_png}_temp_{i}.png"
        ggsave(temp_path, p, width=fig_width / n_cols, height=fig_height / n_rows, dpi=300)
        plot_paths.append(temp_path)

    # Create a combined figure using a layout manager or directly saving to a grid
    # For simplicity, we'll use matplotlib to arrange the saved plots
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height))
    axs = axs.flatten()

    for i, path in enumerate(plot_paths):
        if i < total_plots:
            img = plt.imread(path)
            axs[i].imshow(img)
            axs[i].axis('off')

    # Hide any unused subplots
    for i in range(total_plots, n_rows * n_cols):
        axs[i].axis('off')

    plt.tight_layout()

    # Save the combined figure
    plt.savefig(f"{output_png}.png", dpi=300, bbox_inches='tight')

    # Create and save a separate legend file
    if legend_colors:
        create_plotnine_legend(legend_colors, point_colors, f"{output_png}_legend.png",
                               legend_n_rows=legend_n_rows, font_size=font_size)

    # Clean up temporary files
    for path in plot_paths:
        if os.path.exists(path):
            os.remove(path)

    # Apply the SVG processing function if needed
    save_figure_plotnine(f"{output_png}.png", output_png)

    plt.close(fig)


def create_plotnine_legend(legend_colors, point_colors=None, output_file="legend.png",
                           legend_n_rows=1, font_size=10):
    """
    Creates a standalone legend using plotnine and saves it as an image.

    Parameters:
        legend_colors (dict): Dictionary mapping categories to colors for fill legend.
        point_colors (list, optional): Colors for point/line legend items.
        output_file (str): Path to save the legend.
        legend_n_rows (int): Number of rows for the legend.
        font_size (int): Font size for legend text.
    """
    # Create a dummy dataframe with all legend items
    legend_data = pd.DataFrame({
        'x': [1] * len(legend_colors),
        'y': list(range(len(legend_colors))),
        'category': list(legend_colors.keys())
    })

    # Create a basic plot with both geom_col and geom_line if needed
    p = (ggplot(legend_data, aes(x='x', y='y', fill='category'))
         + geom_col(show_legend=True)
         + scale_fill_manual(values=legend_colors)
         + theme_minimal()
         + theme(
                legend_position="bottom",
                legend_box="horizontal",
                legend_title=element_blank(),
                legend_text=element_text(size=font_size),
                panel_grid=element_blank(),
                axis_text=element_blank(),
                axis_title=element_blank(),
                panel_border=element_blank(),
                panel_background=element_blank(),
                plot_background=element_blank(),
            )
         + guides(fill=guide_legend(nrow=legend_n_rows)))

    # Add point legend if point_colors are provided
    if point_colors is not None and len(point_colors) > 0:
        point_data = pd.DataFrame({
            'x': [1] * len(point_colors),
            'y': list(range(len(point_colors))),
            'point_category': [f"Point {i + 1}" for i in range(len(point_colors))]
        })

        p = p + geom_point(
            data=point_data,
            mapping=aes(x='x', y='y', color='point_category'),
            size=3, show_legend=True
        ) + scale_color_manual(values=point_colors)

    # Adjust figure size to show only the legend
    ggsave(output_file, p, width=6, height=1, dpi=300)


def save_figure_plotnine(input_png, output_prefix):
    """
    Processes a plotnine-generated PNG file to create SVG variants.

    Parameters:
        input_png (str): Path to input PNG file.
        output_prefix (str): Prefix for output files.
    """
    # Convert the PNG to SVG
    svg_path = f"{output_prefix}.svg"

    # Use a library like cairosvg to convert PNG to SVG
    # This is a simplified example and might need additional libraries
    import subprocess
    try:
        # Try using ImageMagick's convert tool if available
        subprocess.run(['convert', input_png, svg_path], check=True)
    except (subprocess.SubprocessError, FileNotFoundError):
        # If ImageMagick isn't available, we'll keep only the PNG
        print("Warning: ImageMagick not found. Only PNG file will be available.")
        return

    # Process the SVG similar to the original function
    try:
        with open(svg_path, "r", encoding="utf-8") as f:
            svg_content = f.read()

        parser = etree.XMLParser(remove_blank_text=True)
        tree = etree.fromstring(svg_content.encode("utf-8"), parser)

        # Generate no-text PDF
        pdf_path = f"{output_prefix}_no_text.pdf"
        tree_no_text = etree.fromstring(svg_content.encode("utf-8"), parser)

        for text_element in tree_no_text.xpath("//svg:text", namespaces={"svg": "http://www.w3.org/2000/svg"}):
            text_element.getparent().remove(text_element)

        svg_no_text = etree.tostring(tree_no_text, encoding="utf-8")
        cairosvg.svg2pdf(bytestring=svg_no_text, write_to=pdf_path, dpi=300)

        # Generate no-plot SVG
        svg_no_plot_path = f"{output_prefix}_no_plot.svg"
        tree_no_plot = etree.fromstring(svg_content.encode("utf-8"), parser)

        for element in tree_no_plot.xpath("//svg:path | //svg:polygon | //svg:circle",
                                          namespaces={"svg": "http://www.w3.org/2000/svg"}):
            element.getparent().remove(element)

        with open(svg_no_plot_path, "wb") as f:
            f.write(etree.tostring(tree_no_plot, pretty_print=True, encoding="utf-8"))

    except Exception as e:
        print(f"Error processing SVG: {e}")


def plot_stacked_bar_and_line(merged_dict, input_name, legend_colors, point_dict=None, point_colors=None,
                                       font_size=10, x_range=(2010, 2050), y_range=(-600, 100),
                                       x_ticks=None, y_ticks=None, show_legend=False):
    """
    Creates a stacked bar chart with optional line overlays using plotnine.

    Parameters:
        merged_dict (dict): Dictionary containing data frames.
        input_name (str): Key in merged_dict for the data to plot.
        legend_colors (dict): Dictionary mapping categories to colors.
        point_dict (dict, optional): Dictionary with point data for line plots.
        point_colors (list, optional): List of colors for points.
        font_size (int): Font size for text elements.
        x_range (tuple): Range for x-axis.
        y_range (tuple): Range for y-axis.
        x_ticks (int/list): Ticks for x-axis.
        y_ticks (list): Ticks for y-axis.
        show_legend (bool): Whether to show the legend.

    Returns:
        plotnine.ggplot: The created plot.
    """
    # Get the data
    merged_df = merged_dict[input_name].copy()
    merged_df.index.name = 'Year'
    merged_df = merged_df.reset_index()

    # Categories and colors
    categories = list(legend_colors.keys())

    # Convert to long format for both positive and negative values
    long_data = pd.melt(
        merged_df,
        id_vars='Year',
        value_vars=categories,
        var_name='Category',
        value_name='Value'
    )

    # Split into positive and negative dataframes
    pos_data = long_data.copy()
    pos_data['Value'] = pos_data['Value'].clip(lower=0)

    neg_data = long_data.copy()
    neg_data['Value'] = neg_data['Value'].clip(upper=0)

    # Create the base plot
    p = (ggplot() +
         geom_col(pos_data, aes(x='Year', y='Value', fill='Category'),
                  position='stack', width=COLUMN_WIDTH) +
         geom_col(neg_data, aes(x='Year', y='Value', fill='Category'),
                  position='stack', width=COLUMN_WIDTH) +
         scale_fill_manual(values=legend_colors) +
         scale_x_continuous(limits=(x_range[0] - X_OFFSET, x_range[1] + X_OFFSET),
                            breaks=np.arange(x_range[0], x_range[1] + 1, x_ticks if isinstance(x_ticks, int) else 5)) +
         scale_y_continuous(limits=y_range, breaks=y_ticks) +
         theme_minimal() +
         theme(
             text=element_text(size=font_size),
             panel_grid_minor=element_blank(),
             panel_grid_major_x=element_blank(),
             legend_position='none' if not show_legend else 'bottom'
         ))

    # Add line plots if point_dict is provided
    if point_dict is not None:
        point_df = point_dict[input_name].copy()
        point_df.index.name = 'Year'
        point_df = point_df.reset_index()

        for idx, column in enumerate(point_df.columns[1:]):  # Skip 'Year' column
            color = point_colors[idx] if point_colors and idx < len(point_colors) else 'black'

            point_long = point_df[['Year', column]].rename(columns={column: 'Value'})
            point_long['Series'] = column

            p = p + geom_line(
                data=point_long,
                mapping=aes(x='Year', y='Value', group='Series'),
                color=color,
                size=1.5
            ) + geom_point(
                data=point_long,
                mapping=aes(x='Year', y='Value', group='Series'),
                color=color,
                size=3
            )

    return p


def plot_stacked_bar(merged_dict, input_name, legend_colors, font_size=10,
                              x_range=(2010, 2050), y_range=(-600, 100),
                              x_ticks=None, y_ticks=None, show_legend=False):
    """
    Creates a stacked bar chart using plotnine.

    Parameters:
        merged_dict (dict): Dictionary containing data frames.
        input_name (str): Key in merged_dict for the data to plot.
        legend_colors (dict): Dictionary mapping categories to colors.
        font_size (int): Font size for text elements.
        x_range (tuple): Range for x-axis.
        y_range (tuple): Range for y-axis.
        x_ticks (int/list): Ticks for x-axis.
        y_ticks (list): Ticks for y-axis.
        show_legend (bool): Whether to show the legend.

    Returns:
        plotnine.ggplot: The created plot.
    """
    # Get the data
    merged_df = merged_dict[input_name].copy()
    merged_df.index.name = 'Year'
    merged_df = merged_df.reset_index()

    # Categories and colors
    categories = list(legend_colors.keys())

    # Convert to long format for both positive and negative values
    long_data = pd.melt(
        merged_df,
        id_vars='Year',
        value_vars=categories,
        var_name='Category',
        value_name='Value'
    )

    # Split into positive and negative dataframes
    pos_data = long_data.copy()
    pos_data['Value'] = pos_data['Value'].clip(lower=0)

    neg_data = long_data.copy()
    neg_data['Value'] = neg_data['Value'].clip(upper=0)

    # Create the plot
    p = (ggplot() +
         geom_col(pos_data, aes(x='Year', y='Value', fill='Category'),
                  position='stack', width=COLUMN_WIDTH) +
         geom_col(neg_data, aes(x='Year', y='Value', fill='Category'),
                  position='stack', width=COLUMN_WIDTH) +
         scale_fill_manual(values=legend_colors) +
         scale_x_continuous(limits=(x_range[0] - X_OFFSET, x_range[1] + X_OFFSET),
                            breaks=np.arange(x_range[0], x_range[1] + 1, x_ticks if isinstance(x_ticks, int) else 5)) +
         scale_y_continuous(limits=y_range, breaks=y_ticks) +
         theme_minimal() +
         theme(
             text=element_text(size=font_size),
             panel_grid_minor=element_blank(),
             panel_grid_major_x=element_blank(),
             legend_position='none' if not show_legend else 'bottom'
         ))

    return p


def plot_line_chart(merged_dict, input_name, legend_colors, font_size=10,
                             x_range=(2010, 2050), y_range=(-600, 100),
                             x_ticks=None, y_ticks=None, show_legend=False):
    """
    Creates a line chart using plotnine.

    Parameters:
        merged_dict (dict): Dictionary containing data frames.
        input_name (str): Key in merged_dict for the data to plot.
        legend_colors (dict): Dictionary mapping categories to colors.
        font_size (int): Font size for text elements.
        x_range (tuple): Range for x-axis.
        y_range (tuple): Range for y-axis.
        x_ticks (int/list): Ticks for x-axis.
        y_ticks (list): Ticks for y-axis.
        show_legend (bool): Whether to show the legend.

    Returns:
        plotnine.ggplot: The created plot.
    """
    # Get the data
    merged_df = merged_dict[input_name].copy()
    merged_df.index.name = 'Year'
    merged_df = merged_df.reset_index()

    # Categories and colors
    categories = list(legend_colors.keys())

    # Convert to long format
    long_data = pd.melt(
        merged_df,
        id_vars='Year',
        value_vars=categories,
        var_name='Category',
        value_name='Value'
    )

    # Create the plot
    p = (ggplot(long_data, aes(x='Year', y='Value', color='Category', group='Category')) +
         geom_line(size=1) +
         geom_point(size=3) +
         scale_color_manual(values=legend_colors) +
         scale_x_continuous(limits=x_range,
                            breaks=np.arange(x_range[0], x_range[1] + 1, x_ticks if isinstance(x_ticks, int) else 5)) +
         scale_y_continuous(limits=y_range, breaks=y_ticks) +
         theme_minimal() +
         theme(
             text=element_text(size=font_size),
             panel_grid_minor=element_blank(),
             panel_grid_major_x=element_blank(),
             legend_position='none' if not show_legend else 'bottom'
         ))

    return p


def plot_stacked_area(merged_dict, input_name, legend_colors, font_size=10,
                               x_range=(2010, 2050), y_range=(-600, 100), show_legend=False):
    """
    Creates a stacked area chart using plotnine.

    Parameters:
        merged_dict (dict): Dictionary containing data frames.
        input_name (str): Key in merged_dict for the data to plot.
        legend_colors (dict): Dictionary mapping categories to colors.
        font_size (int): Font size for text elements.
        x_range (tuple): Range for x-axis.
        y_range (tuple): Range for y-axis.
        show_legend (bool): Whether to show the legend.

    Returns:
        plotnine.ggplot: The created plot.
    """
    # Get the data
    merged_df = merged_dict[input_name].copy()
    merged_df.index.name = 'Year'
    merged_df = merged_df.reset_index()

    # Categories and colors
    categories = list(legend_colors.keys())

    # Convert to long format
    long_data = pd.melt(
        merged_df,
        id_vars='Year',
        value_vars=categories,
        var_name='Category',
        value_name='Value'
    )

    # Create the plot
    p = (ggplot(long_data, aes(x='Year', y='Value', fill='Category')) +
         geom_area() +
         scale_fill_manual(values=legend_colors) +
         scale_x_continuous(limits=x_range) +
         scale_y_continuous(limits=y_range) +
         theme_minimal() +
         theme(
             text=element_text(size=font_size),
             panel_grid_minor=element_blank(),
             legend_position='none' if not show_legend else 'bottom'
         ))

    return p




def get_max_min(df):
    """
    高效计算 df 的最大最小值：
    - 避免逐行 `cumsum(axis=1)`，直接计算正数最大累积和 & 负数最小累积和
    - 直接转换为 NumPy 数组，加速运算
    """
    arr = df.to_numpy()  # 直接转换为 NumPy 数组，加快计算速度

    # 计算每行正数累积和的最大值（优化版）
    pos_cumsum = np.where(arr > 0, arr, 0).cumsum(axis=1)
    pos_max = np.max(pos_cumsum, axis=1)  # 仅计算每行最大值
    overall_pos_max = np.max(pos_max)  # 取整个 df 的最大正数累积值

    # 计算每行负数累积和的最小值（优化版）
    neg_cumsum = np.where(arr < 0, arr, 0).cumsum(axis=1)
    neg_min = np.min(neg_cumsum, axis=1)  # 仅计算每行最小值
    overall_neg_min = np.min(neg_min)  # 取整个 df 的最小负数累积值

    # 计算单个值的最大最小值
    overall_min = np.min(arr)
    overall_max = np.max(arr)

    return max(overall_pos_max, overall_max), min(overall_neg_min, overall_min)

    range_value = max_value - min_value
    ideal_interval = range_value / (desired_ticks - 1)

    e = math.floor(math.log10(ideal_interval))
    nice_numbers = [1, 2, 5, 10]
    nice_numbers_scaled = [n * (10 ** e) for n in nice_numbers]
    interval = min(nice_numbers_scaled, key=lambda x: abs(x - ideal_interval))

    if min_value < 0 and min_value >= -1:
        min_tick = 0
        max_tick = math.ceil(max_value / interval) * interval
        ticks = list(np.arange(min_tick, max_tick + interval, interval))
        print("Space left below 0 for small negative values like -1.")
    else:
        min_tick = math.floor(min_value / interval) * interval
        max_tick = math.ceil(max_value / interval) * interval
        ticks = list(np.arange(min_tick, max_tick + interval, interval))
        if 0 not in ticks and min_value < 0 < max_value:
            ticks.append(0)
            ticks.sort()

    return interval, ticks

# def get_y_axis_ticks(min_value, max_value, desired_ticks=4):
#     range_value = max_value - min_value
#     if range_value <= 0:
#         return 0, 1, [0, 0.5, 1]
#
#     ideal_interval = range_value / (desired_ticks - 1)
#     e = math.floor(math.log10(ideal_interval))
#     nice_numbers = [1, 2, 5, 10]
#     nice_numbers_scaled = [n * (10 ** e) for n in nice_numbers]
#     interval = min(nice_numbers_scaled, key=lambda x: abs(x - ideal_interval))
#
#     # 统一逻辑：从 min_value 向下取整（不是硬设为 0）
#     min_tick = math.floor(min_value / interval) * interval
#     max_tick = math.ceil(max_value / interval) * interval
#
#     ticks = list(np.arange(min_tick, max_tick + interval, interval))
#
#     # 可选：强制包含 0
#     if 0 not in ticks and min_value < 0 < max_value:
#         ticks.append(0)
#         ticks = sorted(set(ticks))
#
#     # 统一逻辑：从 min_value 向下取整（不是硬设为 0）
#     if abs(min_value) < interval:
#         min_v = math.floor(min_value)
#     else:
#         min_v = math.floor(min_value / interval) * interval
#     max_v = math.ceil(max_value / interval) * interval
#
#     return min_v, max_v, ticks

def get_y_axis_ticks(min_value, max_value, desired_ticks=5):
    """
    生成Y轴刻度，根据数据范围智能调整刻度间隔和范围。
    优化版本，提高运行速度，并处理0-100特殊情况。
    """
    # 1. 快速处理特殊情况
    if min_value > 0 and max_value > 0:
        min_value = 0
    elif min_value < 0 and max_value < 0:
        max_value = 0

    range_value = max_value - min_value
    if range_value <= 0:
        return 0, 1, np.array([0, 0.5, 1])  # 使用numpy数组替代列表

    # 2. 一次性计算间隔
    ideal_interval = range_value / (desired_ticks - 1)
    # 根据理想间隔选择“nice”间隔
    e = math.floor(math.log10(ideal_interval))  # 计算数量级
    base = 10 ** e
    normalized_interval = ideal_interval / base

    # 定义“nice”间隔选项
    nice_intervals = [1, 2, 5, 10]
    # 选择最接近理想间隔的“nice”值
    interval = min(nice_intervals, key=lambda x: abs(x - normalized_interval)) * base

    # 3. 整合计算，减少中间变量
    min_tick = math.floor(min_value / interval) * interval
    max_tick = math.ceil(max_value / interval) * interval

    # 4. 使用numpy直接生成数组，避免Python列表操作
    tick_count = int((max_tick - min_tick) / interval) + 1
    ticks = np.linspace(min_tick, max_tick, tick_count)

    # 5. 高效处理0的插入
    if min_value < 0 < max_value and 0 not in ticks:
        # numpy的searchsorted比Python的排序更高效
        zero_idx = np.searchsorted(ticks, 0)
        ticks = np.insert(ticks, zero_idx, 0)

    # 6. 预计算共享变量，避免重复计算
    close_threshold = 0.3 * interval

    # 7. 简化逻辑，减少条件分支
    max_v = max_tick
    min_v = min_tick

    # 处理刻度和范围调整（仅当有足够刻度且最值不是0时）
    if len(ticks) >= 2:
        # 处理最大值
        if ticks[-1] != 0 and (max_value - ticks[-2]) < close_threshold and (ticks[-1] - max_value) > close_threshold:
            ticks = ticks[:-1]  # 移除最后一个刻度
            max_v = max_value + 0.1 * interval

        # 处理最小值
        if ticks[0] != 0 and (ticks[1] - min_value) < close_threshold and (min_value - ticks[0]) > close_threshold:
            ticks = ticks[1:]  # 移除第一个刻度
            min_v = min_value - 0.1 * interval
        elif abs(min_value) < interval:
            min_v = math.floor(min_value)

    # 8. 特殊情况：当刻度范围是0到100时，使用规则的25间隔
    if (abs(ticks[0]) < 1e-10 and abs(ticks[-1] - 100) < 1e-10) or (min_tick == 0 and max_tick == 100):
        ticks = np.array([0, 25, 50, 75, 100])
        min_v = 0
        max_v = 100

    return min_v, max_v, ticks.tolist()  # 根据需要转回列表


def calculate_y_axis_range(data_dict, desired_ticks=5,use_parallel=True, n_jobs=-1):
    """
    并行计算所有 DataFrame 的 (max, min) 值，并计算 y 轴范围和间隔
    """
    # 并行计算每个 DataFrame 的最大最小值
    if use_parallel:
        results = Parallel(n_jobs=n_jobs, prefer="threads")(delayed(get_max_min)(df) for df in data_dict.values())
    else:
        results = [get_max_min(df) for df in data_dict.values()]

    # 获取所有 DataFrame 的全局最大最小值
    max_value = max(res[0] for res in results)
    min_value = min(res[1] for res in results)

    range_value = max_value - min_value
    if range_value == 0:
        return (0, 1), [0, 1]
    else:
        scale_factor = 1

        if range_value > -1 and range_value < 1:
            scale_exponent = int(abs(math.floor(math.log10(range_value))))
            scale_factor = 10 ** scale_exponent
            max_value *= scale_factor
            min_value *= scale_factor

        min_v, max_v, ticks = get_y_axis_ticks(min_value, max_value, desired_ticks=desired_ticks)

        # 缩小结果回原比例
        if scale_factor != 1:
            min_v /= scale_factor
            max_v /= scale_factor
            ticks = [t / scale_factor for t in ticks]

        if ticks==[0,20,40,60,80,100]:
            ticks=[0,25,50,75,100]
        return ((min_v, max_v), ticks)


def plot_land_use_polar(input_file,output_file=None, result_file="../output/12_land_use_movement_all.xlsx", yticks=None, fontsize=30,x_offset=1.3):
    """
    绘制土地利用变化的极坐标图。

    参数：
      input_file: 用于生成输出文件名的字符串（例如 "mydata"）。
      result_file: 包含土地利用变化数据的 Excel 文件名，默认 "../output/12_land_use_movement_all.xlsx"。
                   Excel 文件中地类是第一列，列为 MultiIndex(年份,指标)。
      yticks: y 轴（径向轴）的刻度列表。如果为 None，则根据数据自动计算。
    """
    # 读取 Excel 文件，地类是第一列
    df = pd.read_excel(result_file, sheet_name=input_file, header=[0, 1])

    # 确保列是多级索引
    if not isinstance(df.columns, pd.MultiIndex):
        raise ValueError("Excel文件的列必须是多级索引，第一级为年份，第二级为指标")

    # 假设地类列是第一列，获取其列名
    land_use_col = df.columns[0]
    # 排除指定的地类
    excluded_land_uses = [
        "Unallocated - modified land",
        "Unallocated - natural land"
    ]
    # 提取地类名称列表
    names = df[land_use_col].unique().tolist()
    names = [land_use for land_use in names if land_use not in excluded_land_uses]

    # 自动提取年份（列索引第一级，从第二列开始）
    years = list(df.columns.levels[0])  # 使用levels获取第一级唯一值

    color_df = pd.read_excel('tools/land use colors.xlsx', sheet_name='ag')
    color_mapping = dict(zip(color_df['desc'], color_df['color']))

    # 更新绘图字体设置
    plt.rcParams.update({'font.size': 10, 'font.family': 'Arial'})

    # 创建极坐标图
    fig, ax = plt.subplots(figsize=(8, 6), subplot_kw={'projection': 'polar'})

    # 绘制每个 land_use 的移动轨迹
    for land_use in names:
        # 筛选当前地类的数据
        land_use_data = df[df[land_use_col] == land_use]

        # 提取距离和角度数据
        distances = []
        angles_deg = []

        for year in years:
            try:
                distance = land_use_data.loc[:, (year, 'Area×Distance')].values[0]
                angle = land_use_data.loc[:, (year, 'Angle (degrees)')].values[0]

                distances.append(distance)
                angles_deg.append(angle)
            except (KeyError, IndexError):
                # 如果某年份数据缺失，跳过
                continue

        if not distances:  # 如果没有数据，跳过该地类
            continue

        angles_rad = np.radians(angles_deg)
        color = color_mapping.get(land_use, '#C8C8C8')
        # 设置线条的粗细和标记点的大小
        ax.plot(angles_rad, distances, marker='o', label=land_use, color=color, linewidth=5, markersize=10)

    # 设置极坐标的角度刻度与标签
    angles_deg_fixed = np.arange(0, 360, 45)
    labels = ['North', 'Northeast', 'East', 'Southeast',
              'South', 'Southwest', 'West', 'Northwest']
    offset_angles = [45, 90, 135, 225, 270, 315]
    # 对于 offset_angles 的位置先不显示标签，后续用 ax.text 标注
    masked_labels = [label if angle not in offset_angles else "" for angle, label in zip(angles_deg_fixed, labels)]

    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.set_xticks(np.radians(angles_deg_fixed))
    ax.set_xticklabels(masked_labels, fontname='Arial', fontsize=fontsize)

    # 计算所有距离的最大值，用于设置y轴刻度
    all_distances = []
    for land_use in names:
        land_use_data = df[df[land_use_col] == land_use]
        for year in years:
            try:
                distance = land_use_data.loc[:, (year, 'Area×Distance')].values[0]
                if not pd.isna(distance):
                    all_distances.append(distance)
            except (KeyError, IndexError):
                continue

    # 如果未提供 yticks，则根据所有距离值自动计算
    if yticks is None:
        if all_distances:
            max_distance = max(all_distances)
            min_v, max_v, yticks = get_y_axis_ticks(0, max_distance)
        else:
            min_v, max_v, yticks = 0, 1, [0, 0.25, 0.5, 0.75, 1]
    else:
        min_v = yticks[0]
        max_v = yticks[-1]

    ax.set_yticks(yticks)
    ax.set_ylim(min_v, max_v)
    # 如果你还想更改字体，可以使用：
    labels_y = ax.get_yticklabels()
    ax.set_yticklabels(labels_y, fontname='Arial', fontsize=fontsize)

    # 在指定角度处添加完整标签
    # 定义不同方向的偏移乘数
    cardinal_offset = 1.0  # 控制四个主方向（东南西北）的偏移
    east_west_offset = 0.92  # 控制东西方向的额外偏移

    # 在指定角度处添加完整标签
    for angle_deg in offset_angles:
        angle_rad = np.radians(angle_deg)
        # 查找对应的标签
        label = labels[angles_deg_fixed.tolist().index(angle_deg)]

        # 设置基础偏移量
        current_offset = x_offset * cardinal_offset

        # 为东西方向设置额外偏移
        if angle_deg == 90 or angle_deg == 270:  # 东西方向
            current_offset = x_offset * east_west_offset

        # 添加标签
        ax.text(angle_rad,
                ax.get_rmax() * current_offset,
                label,
                ha='center',
                va='center',
                fontsize=fontsize,
                fontname='Arial')

        # 调试信息，确保正确的偏移量
        # print(f'angle_deg: {angle_deg}, current_offset: {current_offset}, label: {label}')

    ax.set_rlabel_position(180)  # 将径向标签移动至180度方向
    # 设置网格线的样式和粗细
    ax.grid(True, linestyle='--', linewidth=2, alpha=0.99)

    # 设置径向网格线的粗细
    ax.xaxis.set_tick_params(width=3)
    ax.yaxis.set_tick_params(width=3)
    # 添加图例
    # ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.0))

    # 保存图像，构造输出文件名
    if output_file is None:
        output_file = f"../output/12_{input_file}_polar"
    else:
        output_file = f"../output/{output_file}"
    fig.savefig(f'{output_file}.pdf', dpi=300, bbox_inches='tight')
    save_figure(fig, output_file)

    plt.show()

def plot_stacked_area(merged_dict, input_name, legend_colors, font_size=10,
                      x_range=(2010, 2050), y_range=(-600, 100), show_legend=False):
    """
    画plotnine风格的堆叠面积图
    """
    # 获取原始数据
    merged_df = merged_dict[input_name].copy()
    merged_df.index = merged_df.index.astype(int)
    merged_df = merged_df.reset_index().rename(columns={'index': 'Year'})

    # 宽表转长表
    categories = list(legend_colors.keys())
    data_long = merged_df.melt(id_vars='Year', value_vars=categories,
                               var_name='Category', value_name='Value')

    # 画图
    p = (
        ggplot(data_long, aes('Year', 'Value', fill='Category'))
        + geom_area()
        + scale_fill_manual(values=legend_colors)
        + labs(x='Year', y='Value')
        + xlim(x_range[0], x_range[1])
        + ylim(y_range[0], y_range[1])
        + theme(
            axis_text_x=element_text(size=font_size),
            axis_text_y=element_text(size=font_size),
            legend_position='right' if show_legend else 'none'
        )
    )
    return p