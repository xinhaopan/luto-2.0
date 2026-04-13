"""
07_GHG.py
Combined stacked-area figure for land use and agricultural management GHG.
"""
import _path_setup  # noqa: F401

import pandas as pd

from tools.two_row_figure import (
    LU_COLORS,
    RENAME_NON_AG,
    RENAME_AM_NON_AG,
    classify_land_use,
    export_long_tables,
    get_am_colors,
    input_files,
    load_report_source_csv,
    save_three_row_figure,
)

OVERVIEW_COLORS = {
    'Agricultural land-use': '#f39b8b',
    'Agricultural management': '#9A8AB3',
    'Non-agricultural land-use': '#6eabb1',
    'Transition': '#eb9132',
}


def prepare_overview():
    rows = []

    for scenario in input_files:
        ghg_ag = load_report_source_csv(scenario, 'GHG_emissions_separate_agricultural_landuse')
        if not ghg_ag.empty:
            ghg_ag = ghg_ag.query('region == "AUSTRALIA" and Water_supply != "ALL" and Source != "ALL"').copy()
            ghg_ag = ghg_ag.groupby('Year', as_index=False)['Value (t CO2e)'].sum()
            for _, row in ghg_ag.iterrows():
                rows.append({'year': int(row['Year']), 'scenario': scenario, 'category': 'Agricultural land-use', 'value': float(row['Value (t CO2e)']) / 1e6})

        ghg_am = load_report_source_csv(scenario, 'GHG_emissions_separate_agricultural_management')
        if not ghg_am.empty:
            ghg_am = ghg_am.query('region == "AUSTRALIA" and Water_supply != "ALL" and `Land-use` != "ALL"').copy()
            ghg_am = ghg_am.groupby('Year', as_index=False)['Value (t CO2e)'].sum()
            for _, row in ghg_am.iterrows():
                rows.append({'year': int(row['Year']), 'scenario': scenario, 'category': 'Agricultural management', 'value': float(row['Value (t CO2e)']) / 1e6})

        ghg_non_ag = load_report_source_csv(scenario, 'GHG_emissions_separate_no_ag_reduction')
        if not ghg_non_ag.empty:
            ghg_non_ag = ghg_non_ag.query('region == "AUSTRALIA" and `Land-use` != "ALL"').copy()
            ghg_non_ag = ghg_non_ag.groupby('Year', as_index=False)['Value (t CO2e)'].sum()
            for _, row in ghg_non_ag.iterrows():
                rows.append({'year': int(row['Year']), 'scenario': scenario, 'category': 'Non-agricultural land-use', 'value': float(row['Value (t CO2e)']) / 1e6})

        ghg_transition = load_report_source_csv(scenario, 'GHG_emissions_separate_transition_penalty')
        if not ghg_transition.empty:
            ghg_transition = ghg_transition.query(
                'region == "AUSTRALIA" and Type != "ALL" and Water_supply != "ALL"'
            ).copy()
            ghg_transition = ghg_transition.groupby('Year', as_index=False)['Value (t CO2e)'].sum()
            for _, row in ghg_transition.iterrows():
                rows.append({'year': int(row['Year']), 'scenario': scenario, 'category': 'Transition', 'value': float(row['Value (t CO2e)']) / 1e6})

    return pd.DataFrame(rows)


def prepare_land_use():
    rows = []

    for scenario in input_files:
        ghg_ag = load_report_source_csv(scenario, 'GHG_emissions_separate_agricultural_landuse')
        if not ghg_ag.empty:
            ghg_ag = ghg_ag.query(
                'region == "AUSTRALIA" and Water_supply != "ALL" and Source != "ALL"'
            ).copy()
            ghg_ag['category'] = ghg_ag['Land-use'].map(classify_land_use)
            ghg_ag = ghg_ag.dropna(subset=['category'])
            ghg_ag = ghg_ag.groupby(['Year', 'category'], as_index=False)['Value (t CO2e)'].sum()
            for _, row in ghg_ag.iterrows():
                rows.append({
                    'year': int(row['Year']),
                    'scenario': scenario,
                    'category': row['category'],
                    'value': float(row['Value (t CO2e)']) / 1e6,
                })

        ghg_non_ag = load_report_source_csv(scenario, 'GHG_emissions_separate_no_ag_reduction')
        if not ghg_non_ag.empty:
            ghg_non_ag = ghg_non_ag.copy()
            ghg_non_ag['Land-use'] = ghg_non_ag['Land-use'].replace(RENAME_NON_AG)
            ghg_non_ag = ghg_non_ag.query('region == "AUSTRALIA" and `Land-use` != "ALL"').copy()
            ghg_non_ag = ghg_non_ag.groupby(['Year'], as_index=False)['Value (t CO2e)'].sum()
            for _, row in ghg_non_ag.iterrows():
                rows.append({
                    'year': int(row['Year']),
                    'scenario': scenario,
                    'category': 'Non-agricultural land',
                    'value': float(row['Value (t CO2e)']) / 1e6,
                })

    return pd.DataFrame(rows)


def prepare_am():
    rows = []

    for scenario in input_files:
        ghg_am = load_report_source_csv(scenario, 'GHG_emissions_separate_agricultural_management')
        if ghg_am.empty:
            continue
        ghg_am = ghg_am.copy()
        ghg_am['Agricultural Management Type'] = ghg_am['Agricultural Management Type'].replace(RENAME_AM_NON_AG)
        ghg_am = ghg_am.query(
            'region == "AUSTRALIA" and Water_supply != "ALL" and `Land-use` != "ALL"'
        ).copy()
        ghg_am = ghg_am.groupby(['Year', 'Agricultural Management Type'], as_index=False)['Value (t CO2e)'].sum()
        for _, row in ghg_am.iterrows():
            rows.append({
                'year': int(row['Year']),
                'scenario': scenario,
                'category': row['Agricultural Management Type'],
                'value': float(row['Value (t CO2e)']) / 1e6,
            })

    return pd.DataFrame(rows)


def main():
    overview_df = prepare_overview()
    land_use_df = prepare_land_use()
    am_df = prepare_am()
    am_colors = {
        label: color
        for label, color in get_am_colors().items()
        if label != 'Other land-use'
    }
    export_long_tables(
        '07_ghg_long_tables.xlsx',
        overview=overview_df,
        land_use=land_use_df,
        agricultural_management=am_df,
    )
    save_three_row_figure(
        overview_df,
        land_use_df,
        am_df,
        OVERVIEW_COLORS,
        LU_COLORS,
        am_colors,
        'GHG (Mt CO\u2082e yr\u207b\u00b9)',
        '07_ghg.svg',
        y_label_x=0.030,
    )


if __name__ == '__main__':
    main()
