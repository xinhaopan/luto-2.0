from .config import input_files
from .helper_data import (
    amortize_costs,
    calculate_baseline_costs,
    compute_unit_prices,
    calculate_shadow_price
)
from .helper_plot import (
    plot_cost,
    plot_price,
    plot_revenue_cost_stacked,
    plot_specified_columns
)


def run_analysis_pipeline(input_file, use_parallel=False):
    amortize_costs(input_file)
    calculate_baseline_costs(input_file, use_parallel)
    compute_unit_prices(input_file)
    # calculate_shadow_price(input_file, percentile_num=97, mask_use=True)

def draw_plots(input_file):
    plot_cost(input_file)
    # plot_price(input_file)
    # plot_revenue_cost_stacked(input_file)
    # plot_specified_columns(input_file, columns_to_plot=['GHG Abatement(MtCOe2)'])
    # plot_specified_columns(input_file, columns_to_plot=['BIO(Mha)'])

