from .config import input_files
from .helper_data import (
    amortize_costs,
    calculate_baseline_costs,
    compute_unit_prices,
    calculate_shadow_price
)
from .helper_plot import (
    plot_cost_vs_price,
    plot_ghg_stack,
)

def run_analysis_pipeline(input_file, use_parallel=False):
    amortize_costs(input_file)
    calculate_baseline_costs(input_file, use_parallel)
    compute_unit_prices(input_file)
    calculate_shadow_price(input_file, percentile_num=97, mask_use=True)
    plot_cost_vs_price(input_file)
    plot_ghg_stack(input_file)
