import tools.config as config
from tools.helper_data import summarize_to_type,summarize_to_category,build_profit_and_cost_nc, make_prices_nc


if __name__ == "__main__":
    economic_files = config.economic_files
    carbon_files = config.carbon_files
    bio_files = config.bio_files

    input_files_0 = config.input_files_0
    input_files_1 = config.input_files_1
    input_files_2 = config.input_files_2
    input_files = config.input_files

    carbon_names = config.carbon_names
    carbon_bio_names = config.carbon_bio_names
    counter_carbon_bio_names = config.counter_carbon_bio_names
    output_names = carbon_names + carbon_bio_names + counter_carbon_bio_names

    years = [y for y in range(2011,2051)]
    summarize_to_category(output_names,years,carbon_files,'xr_total_carbon',n_jobs=41)
    summarize_to_category(output_names,years,bio_files,'xr_total_bio',n_jobs=41)

    profit_da = summarize_to_category(input_files,years, economic_files,'xr_original_economic',n_jobs=41)
    build_profit_and_cost_nc(profit_da, input_files_0, input_files_1, input_files_2, carbon_names, carbon_bio_names, counter_carbon_bio_names)

    files = ['xr_transition_cost_ag2non_ag_amortised_diff','xr_cost_agricultural_management','xr_cost_non_ag']
    dim_names = ['To land-use','am','lu']

    for file,dim_name in zip(files,dim_names):
        summarize_to_type(
            scenarios=output_names,
            years=years,
            file=file,
            keep_dim=dim_name,
            output_file=f'{file}',
            var_name='data',
            scale=1e6,
            dtype='float32',
            chunks = {'scenario': 1, 'year': len(years), 'type': 'auto'}  # 根据实际情况调整分块大小
        )

    make_prices_nc(output_names)