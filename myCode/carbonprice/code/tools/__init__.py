from .config import input_files
from .helper_data import (
    amortize_costs,
    calculate_baseline_costs,
    compute_unit_prices,
    calculate_shadow_price
)
from .helper_plot import (
    plot_combined_with_facets
)

def run_analysis_pipeline(input_file, use_parallel=False):
    amortize_costs(input_file)
    calculate_baseline_costs(input_file, use_parallel)
    compute_unit_prices(input_file)
    calculate_shadow_price(input_file, percentile_num=97, mask_use=True)
    plot_combined_with_facets(input_file)
