"""
07_Biodiversity.py
Combined stacked-area figure for biodiversity contribution by land use and
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

VALUE_COL = 'Area Weighted Score (ha)'
OVERVIEW_COLORS = {
    'Agricultural land-use': '#f39b8b',
    'Agricultural management': '#9A8AB3',
    'Non-agricultural land-use': '#6eabb1',
}


def prepare_overview():
    rows = []

    for scenario in input_files:
        bio = load_report_source_csv(scenario, 'biodiversity_GBF2_priority_scores')
        if bio.empty:
            continue
        bio = bio.replace(RENAME_AM_NON_AG).query(
            'region == "AUSTRALIA" and Water_supply != "ALL" and Landuse != "ALL" and abs(`Area Weighted Score (ha)`) > 1e-4'
        ).copy()
        bio = bio.groupby(['Year', 'Type'], as_index=False)[VALUE_COL].sum()
        for _, row in bio.iterrows():
            category = row['Type']
            if category == 'Agricultural land-use':
                category = 'Agricultural land-use'
            elif category == 'Agricultural Management':
                category = 'Agricultural management'
            elif category == 'Non-Agricultural Land-use':
                category = 'Non-agricultural land-use'
            else:
                continue
            rows.append({
                'year': int(row['Year']),
                'scenario': scenario,
                'category': category,
                'value': float(row[VALUE_COL]) / 1e6,
            })

    return pd.DataFrame(rows)


def prepare_land_use():
    rows = []

    for scenario in input_files:
        bio = load_report_source_csv(scenario, 'biodiversity_GBF2_priority_scores')
        if bio.empty:
            continue
        bio = bio.replace(RENAME_AM_NON_AG).query(
            'region == "AUSTRALIA" and Water_supply != "ALL" and Landuse != "ALL" and abs(`Area Weighted Score (ha)`) > 1e-4'
        ).copy()

        bio_ag = bio.query('Type == "Agricultural land-use"').copy()
        bio_ag['category'] = bio_ag['Landuse'].map(classify_land_use)
        bio_ag = bio_ag.dropna(subset=['category'])
        bio_ag = bio_ag.groupby(['Year', 'category'], as_index=False)[VALUE_COL].sum()
        for _, row in bio_ag.iterrows():
            rows.append({
                'year': int(row['Year']),
                'scenario': scenario,
                'category': row['category'],
                'value': float(row[VALUE_COL]) / 1e6,
            })

        bio_non_ag = bio.query('Type == "Non-Agricultural Land-use"').copy()
        if not bio_non_ag.empty:
            bio_non_ag = bio_non_ag.groupby(['Year'], as_index=False)[VALUE_COL].sum()
            for _, row in bio_non_ag.iterrows():
                rows.append({
                    'year': int(row['Year']),
                    'scenario': scenario,
                    'category': 'Non-agricultural land',
                    'value': float(row[VALUE_COL]) / 1e6,
                })

    return pd.DataFrame(rows)


def prepare_am():
    rows = []

    for scenario in input_files:
        bio = load_report_source_csv(scenario, 'biodiversity_GBF2_priority_scores')
        if bio.empty:
            continue
        bio = bio.replace(RENAME_AM_NON_AG).query(
            'region == "AUSTRALIA" and Water_supply != "ALL" and Landuse != "ALL" and Type == "Agricultural Management" and abs(`Area Weighted Score (ha)`) > 1e-4'
        ).copy()
        bio = bio.groupby(['Year', 'Agricultural Management'], as_index=False)[VALUE_COL].sum()
        for _, row in bio.iterrows():
            rows.append({
                'year': int(row['Year']),
                'scenario': scenario,
                'category': row['Agricultural Management'],
                'value': float(row[VALUE_COL]) / 1e6,
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
        '07_biodiversity_long_tables.xlsx',
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
        'Biodiversity (Mha)',
        '07_biodiversity.svg',
        y_label_x=0.030,
    )


if __name__ == '__main__':
    main()
