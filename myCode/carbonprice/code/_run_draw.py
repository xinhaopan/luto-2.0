import os
from collections import namedtuple
import tools.config as config
from tools.helper_plot import set_plot_style, xarray_to_dict, get_global_ylim, plot_22_layout, get_colors

set_plot_style(font_size=20, font_family='Arial')
task_name = config.TASK_NAME

PlotSpec = namedtuple("PlotSpec", ["file", "sheet", "scale", "ylabel", "column_spacing", "total_name", "negative"])

plot_specs = [
    PlotSpec("xr_total_cost",                               "cost",               1e3, r"Cost (Billion AU\$ yr$^{-1}$)",                                           1, "Total", False),
    PlotSpec("xr_cost_agricultural_management",             "am",                 1e3, r"Cost (Billion AU\$ yr$^{-1}$)",                                           1, None,    False),
    PlotSpec("xr_cost_non_ag",                              "non_ag",             1e3, r"Cost (Billion AU\$ yr$^{-1}$)",                                           1, None,    False),
    PlotSpec("xr_transition_cost_ag2non_ag_amortised_diff", "non_ag",             1e3, r"Cost (Billion AU\$ yr$^{-1}$)",                                           1, None,    False),
    PlotSpec("xr_total_carbon",                             "carbon_total",       1,   r"Change in net GHG emissions (MtCO$_2$e yr$^{-1}$)",                       1, "Total", True),
    PlotSpec("xr_total_bio",                                "biodiversity_total", 1,   r"Change in biodiversity (contribution-weighted area, Mha yr$^{-1}$)",       1, "Total", False),
    PlotSpec("xr_GHG_ag_management",                        "am",                 1,   r"Change in net GHG emissions (MtCO$_2$e yr$^{-1}$)",                       1, None,    True),
    PlotSpec("xr_GHG_non_ag",                               "non_ag",             1,   r"Change in net GHG emissions (MtCO$_2$e yr$^{-1}$)",                       1, None,    True),
    PlotSpec("xr_biodiversity_GBF2_priority_ag_management", "am",                 1,   r"Change in biodiversity (contribution-weighted area, Mha yr$^{-1}$)",       1, None,    False),
    PlotSpec("xr_biodiversity_GBF2_priority_non_ag",        "non_ag",             1,   r"Change in biodiversity (contribution-weighted area, Mha yr$^{-1}$)",       1, None,    False),
]

for rate in ["0.03", "0.05", "0.07"]:
    input_dir  = f"../../../output/{task_name}/carbon_price_{rate}/1_draw_data"
    output_dir = f"../../../output/{task_name}/carbon_price_{rate}/3_Paper_figure"
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n===== rate={rate} =====")
    for spec in plot_specs:
        input_file, sheet_name, scale, ylabel, column_spacing, total_name, negative = (
            spec.file, spec.sheet, spec.scale, spec.ylabel,
            spec.column_spacing, spec.total_name, spec.negative
        )
        nc_path = f"{input_dir}/{input_file}.nc"
        if not os.path.exists(nc_path):
            print(f"  SKIP (missing): {input_file}")
            continue
        print(f"  {input_file}")
        data_dict = xarray_to_dict(nc_path, scale, total_name=total_name, negative=negative)
        data_dict, colors = get_colors(data_dict, 'tools/land use colors.xlsx', sheet_name=sheet_name)
        summary_ylim = get_global_ylim(data_dict)
        output_path = os.path.join(output_dir, f'04_{input_file}.png')
        plot_22_layout(
            all_dfs=data_dict,
            title_map=config.CP_TITLE_MAP,
            colors=colors,
            output_path=output_path,
            summary_ylim=summary_ylim,
            ylabel=ylabel,
            column_spacing=column_spacing,
            total_name=total_name
        )

print("\nDone.")
