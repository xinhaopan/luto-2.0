import os
import tools.config as config

rename_dict = {

    '04_xr_total_cost': 'Figure 03 Cost',
    '04_xr_total_carbon': 'Figure 04 Change in GHG emissions',
    '04_xr_total_bio': 'Figure 05 Change in Biodiversity',
    '05_Carbon_price_all_long': 'Figure 06 Shadow carbon price',
    '05_biodiversity_price_long': 'Figure 07 Shadow biodiversity price',
    '06_cost_maps_line_clip': 'Figure 08 Cost maps',
    '06_Sol_price_maps_line': 'Figure 09 Shadow solution price maps',

    '06_biodiversity_contribution_curve': 'Figure S01 Biodiversity contribution curve',
    '03_Profit': 'Figure S02 Profit',
    '03_xr_total_carbon': 'Figure S03 GHG emissions',
    '03_xr_total_bio': 'Figure S04 Biodiversity',
    '03_xr_GHG_ag_management': 'Figure S05 GHG emissions from agricultural management',
    '03_xr_GHG_non_ag': 'Figure S06 GHG emissions from non-agriculture',
    '03_xr_biodiversity_GBF2_priority_ag_management': 'Figure S07 Biodiversity from agricultural management',
    '03_xr_biodiversity_GBF2_priority_non_ag': 'Figure S08 Biodiversity from non-agriculture',
    '03_xr_area_agricultural_management': 'Figure S09 Agricultural management area',
    '03_xr_area_non_agricultural_landuse': 'Figure S10 Non-agricultural land use area',
    '04_xr_cost_agricultural_management': 'Figure S11 Cost of agricultural management',
    '04_xr_cost_non_ag': 'Figure S12 Cost of non-agriculture',
    '04_xr_transition_cost_ag2non_ag_amortised_diff': 'Figure S13 Change in Transition(ag→non-ag) cost',
    '04_xr_GHG_ag_management': 'Figure S14 Change in GHG emissions from agricultural management',
    '04_xr_GHG_non_ag': 'Figure S15 Change in GHG emissions from non-agriculture',
    '04_xr_biodiversity_GBF2_priority_ag_management': 'Figure S16 Change in Biodiversity from agricultural management',
    '04_xr_biodiversity_GBF2_priority_non_ag': 'Figure S17 Change in Biodiversity from non-agriculture',
    '05_Carbon_solution_price_all': 'Figure S18 Shadow solution carbon price',
    '05_biodiversity_solution_price': 'Figure S19 Shadow solution biodiversity price',
    '06_GHG_maps_line': 'Figure S20 Change in Non-agricultural GHG emissions maps',
    '06_BIO_maps_line': 'Figure S21 Change in Non-agricultural Biodiversity maps',
    '06_area_agmgt_maps_line': 'Figure S22 Agricultural management area maps',
    '06_area_non_ag_maps_line': 'Figure S23 Non-agricultural land use area maps',
}

for old, new in rename_dict.items():
    old_file = f"../../../output/{config.TASK_NAME}/carbon_price/3_Paper_figure/{old}.png"
    new_file = f"../../../output/{config.TASK_NAME}/carbon_price/3_Paper_figure/{new}.png"
    if os.path.exists(old_file):
        os.replace(old_file, new_file)
    else:
        print(f"{old_file} not found, skipped.")