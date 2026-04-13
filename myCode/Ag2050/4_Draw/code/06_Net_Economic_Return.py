"""
06_Net_Economic_Return.py
Combined stacked-area figure for net economic return by land use and
agricultural management.
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
    'Ag cost': '#fab431',
    'Agmgt cost': '#ec7951',
    'Non-ag cost': '#cd4975',
    'Transition(ag->non-ag) cost': '#6200ac',
    'Transition(ag->ag) cost': '#9f0e9e',
    'Ag revenue': '#2d688f',
    'Agmgt revenue': '#19928e',
    'Non-ag revenue': '#35b876',
}


def prepare_overview():
    rows = []

    for scenario in input_files:
        revenue_ag = load_report_source_csv(scenario, 'economics_ag_revenue')
        cost_ag = load_report_source_csv(scenario, 'economics_ag_cost')
        if not revenue_ag.empty or not cost_ag.empty:
            revenue_ag = revenue_ag.query('region == "AUSTRALIA" and Water_supply != "ALL" and Type != "ALL"').copy()
            cost_ag = cost_ag.query('region == "AUSTRALIA" and Water_supply != "ALL" and Type != "ALL"').copy()
            if not revenue_ag.empty:
                data = revenue_ag.groupby('Year', as_index=False)['Value ($)'].sum()
                for _, row in data.iterrows():
                    rows.append({'year': int(row['Year']), 'scenario': scenario, 'category': 'Ag revenue', 'value': float(row['Value ($)']) / 1e9})
            if not cost_ag.empty:
                data = cost_ag.groupby('Year', as_index=False)['Value ($)'].sum()
                for _, row in data.iterrows():
                    rows.append({'year': int(row['Year']), 'scenario': scenario, 'category': 'Ag cost', 'value': -float(row['Value ($)']) / 1e9})

        revenue_am = load_report_source_csv(scenario, 'economics_am_revenue')
        cost_am = load_report_source_csv(scenario, 'economics_am_cost')
        if not revenue_am.empty or not cost_am.empty:
            revenue_am = revenue_am.query('region == "AUSTRALIA" and Water_supply != "ALL" and `Land-use` != "ALL"').copy()
            cost_am = cost_am.query('region == "AUSTRALIA" and Water_supply != "ALL" and `Land-use` != "ALL"').copy()
            if not revenue_am.empty:
                data = revenue_am.groupby('Year', as_index=False)['Value ($)'].sum()
                for _, row in data.iterrows():
                    rows.append({'year': int(row['Year']), 'scenario': scenario, 'category': 'Agmgt revenue', 'value': float(row['Value ($)']) / 1e9})
            if not cost_am.empty:
                data = cost_am.groupby('Year', as_index=False)['Value ($)'].sum()
                for _, row in data.iterrows():
                    rows.append({'year': int(row['Year']), 'scenario': scenario, 'category': 'Agmgt cost', 'value': -float(row['Value ($)']) / 1e9})

        revenue_non_ag = load_report_source_csv(scenario, 'economics_non_ag_revenue')
        cost_non_ag = load_report_source_csv(scenario, 'economics_non_ag_cost')
        if not revenue_non_ag.empty or not cost_non_ag.empty:
            revenue_non_ag = revenue_non_ag.query('region == "AUSTRALIA"').copy()
            cost_non_ag = cost_non_ag.query('region == "AUSTRALIA"').copy()
            if not revenue_non_ag.empty:
                data = revenue_non_ag.groupby('Year', as_index=False)['Value ($)'].sum()
                for _, row in data.iterrows():
                    rows.append({'year': int(row['Year']), 'scenario': scenario, 'category': 'Non-ag revenue', 'value': float(row['Value ($)']) / 1e9})
            if not cost_non_ag.empty:
                data = cost_non_ag.groupby('Year', as_index=False)['Value ($)'].sum()
                for _, row in data.iterrows():
                    rows.append({'year': int(row['Year']), 'scenario': scenario, 'category': 'Non-ag cost', 'value': -float(row['Value ($)']) / 1e9})

        cost_transition_ag2ag = load_report_source_csv(scenario, 'transition_cost_ag2ag')
        if not cost_transition_ag2ag.empty:
            cost_transition_ag2ag = cost_transition_ag2ag.query(
                '`From-land-use` != "ALL" and `To-land-use` != "ALL" and Type != "ALL"'
            ).copy()
            cost_transition_ag2ag = cost_transition_ag2ag.groupby('Year', as_index=False)['Cost ($)'].sum()
            for _, row in cost_transition_ag2ag.iterrows():
                rows.append({
                    'year': int(row['Year']),
                    'scenario': scenario,
                    'category': 'Transition(ag->ag) cost',
                    'value': -float(row['Cost ($)']) / 1e9,
                })

        cost_transition_ag2non = load_report_source_csv(scenario, 'transition_cost_ag2non_ag')
        if not cost_transition_ag2non.empty:
            cost_transition_ag2non = cost_transition_ag2non.query(
                '`From-land-use` != "ALL" and `To-land-use` != "ALL" and `Cost-type` != "ALL"'
            ).copy()
            cost_transition_ag2non = cost_transition_ag2non.groupby('Year', as_index=False)['Cost ($)'].sum()
            for _, row in cost_transition_ag2non.iterrows():
                rows.append({
                    'year': int(row['Year']),
                    'scenario': scenario,
                    'category': 'Transition(ag->non-ag) cost',
                    'value': -float(row['Cost ($)']) / 1e9,
                })

    return pd.DataFrame(rows)


def prepare_land_use():
    rows = []

    for scenario in input_files:
        revenue_ag = load_report_source_csv(scenario, 'economics_ag_revenue')
        cost_ag = load_report_source_csv(scenario, 'economics_ag_cost')
        if not revenue_ag.empty or not cost_ag.empty:
            revenue_ag = revenue_ag.query(
                'region == "AUSTRALIA" and Water_supply != "ALL" and Type != "ALL"'
            ).copy()
            cost_ag = cost_ag.query(
                'region == "AUSTRALIA" and Water_supply != "ALL" and Type != "ALL"'
            ).copy()
            if not cost_ag.empty:
                cost_ag['Value ($)'] = cost_ag['Value ($)'] * -1
            economics_ag = pd.concat([revenue_ag, cost_ag], ignore_index=True)
            economics_ag['category'] = economics_ag['Land-use'].map(classify_land_use)
            economics_ag = economics_ag.dropna(subset=['category'])
            economics_ag = economics_ag.groupby(['Year', 'category'], as_index=False)['Value ($)'].sum()
            for _, row in economics_ag.iterrows():
                rows.append({
                    'year': int(row['Year']),
                    'scenario': scenario,
                    'category': row['category'],
                    'value': float(row['Value ($)']) / 1e9,
                })

        revenue_non_ag = load_report_source_csv(scenario, 'economics_non_ag_revenue')
        cost_non_ag = load_report_source_csv(scenario, 'economics_non_ag_cost')
        if not revenue_non_ag.empty or not cost_non_ag.empty:
            revenue_non_ag = revenue_non_ag.copy()
            cost_non_ag = cost_non_ag.copy()
            if not revenue_non_ag.empty:
                revenue_non_ag['Land-use'] = revenue_non_ag['Land-use'].replace(RENAME_NON_AG)
                revenue_non_ag = revenue_non_ag.query('region == "AUSTRALIA"').copy()
            if not cost_non_ag.empty:
                cost_non_ag['Land-use'] = cost_non_ag['Land-use'].replace(RENAME_NON_AG)
                cost_non_ag = cost_non_ag.query('region == "AUSTRALIA"').copy()
                cost_non_ag['Value ($)'] = cost_non_ag['Value ($)'] * -1
            economics_non_ag = pd.concat([revenue_non_ag, cost_non_ag], ignore_index=True)
            economics_non_ag = economics_non_ag.groupby(['Year'], as_index=False)['Value ($)'].sum()
            for _, row in economics_non_ag.iterrows():
                rows.append({
                    'year': int(row['Year']),
                    'scenario': scenario,
                    'category': 'Non-agricultural land',
                    'value': float(row['Value ($)']) / 1e9,
                })

    return pd.DataFrame(rows)


def prepare_am():
    rows = []

    for scenario in input_files:
        revenue_am = load_report_source_csv(scenario, 'economics_am_revenue')
        cost_am = load_report_source_csv(scenario, 'economics_am_cost')
        if revenue_am.empty and cost_am.empty:
            continue

        revenue_am = revenue_am.copy()
        cost_am = cost_am.copy()
        if not revenue_am.empty:
            revenue_am['Management Type'] = revenue_am['Management Type'].replace(RENAME_AM_NON_AG)
            revenue_am = revenue_am.query(
                'region == "AUSTRALIA" and Water_supply != "ALL" and `Land-use` != "ALL"'
            ).copy()
        if not cost_am.empty:
            cost_am['Management Type'] = cost_am['Management Type'].replace(RENAME_AM_NON_AG)
            cost_am = cost_am.query(
                'region == "AUSTRALIA" and Water_supply != "ALL" and `Land-use` != "ALL"'
            ).copy()
            cost_am['Value ($)'] = cost_am['Value ($)'] * -1

        economics_am = pd.concat([revenue_am, cost_am], ignore_index=True)
        economics_am = economics_am.groupby(['Year', 'Management Type'], as_index=False)['Value ($)'].sum()
        for _, row in economics_am.iterrows():
            rows.append({
                'year': int(row['Year']),
                'scenario': scenario,
                'category': row['Management Type'],
                'value': float(row['Value ($)']) / 1e9,
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
        '06_net_economic_return_long_tables.xlsx',
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
        'Net economic return (Billion AU$ yr\u207b\u00b9)',
        '06_net_economic_return.svg',
        total_legend_label='Net economic return',
        y_label_x=0.028,
    )


if __name__ == '__main__':
    main()
