"""
08_Food.py
Combined stacked-area figure for food production by land use and agricultural
management.
"""
import _path_setup  # noqa: F401

import pandas as pd

from tools.two_row_figure import (
    LU_COLORS,
    RENAME_NON_AG,
    RENAME_AM,
    classify_land_use,
    export_long_tables,
    get_am_colors,
    input_files,
    load_report_source_csv,
    save_three_row_figure,
)

VALUE_COL = 'Production (tonnes, KL)'
OVERVIEW_COLORS = {
    'Agricultural land-use': '#f39b8b',
    'Agricultural management': '#9A8AB3',
    'Non-agricultural land-use': '#6eabb1',
}


def prepare_overview():
    rows = []

    for scenario in input_files:
        food = load_report_source_csv(scenario, 'quantity_production_kt_separate')
        if food.empty:
            continue
        for raw_name, name in [
            ('Agricultural Landuse', 'Agricultural land-use'),
            ('Agricultural Management', 'Agricultural management'),
            ('Non-agricultural Landuse', 'Non-agricultural land-use'),
        ]:
            data = food.query('`Landuse Type` == @raw_name').copy()
            if data.empty:
                continue
            data = data.groupby('Year', as_index=False)[VALUE_COL].sum()
            for _, row in data.iterrows():
                rows.append({
                    'year': int(row['Year']),
                    'scenario': scenario,
                    'category': name,
                    'value': float(row[VALUE_COL]) / 1e6,
                })

    return pd.DataFrame(rows)


def prepare_land_use():
    rows = []

    for scenario in input_files:
        food = load_report_source_csv(scenario, 'quantity_production_kt_separate')
        if food.empty:
            continue
        food = food.copy()

        food_ag = food.query('`Landuse Type` == "Agricultural Landuse"').copy()
        food_ag['category'] = food_ag['Landuse'].map(classify_land_use)
        food_ag = food_ag.dropna(subset=['category'])
        food_ag = food_ag.groupby(['Year', 'category'], as_index=False)[VALUE_COL].sum()
        for _, row in food_ag.iterrows():
            rows.append({
                'year': int(row['Year']),
                'scenario': scenario,
                'category': row['category'],
                'value': float(row[VALUE_COL]) / 1e6,
            })

        food_non_ag = food.query('`Landuse Type` == "Non-agricultural Landuse"').copy()
        if not food_non_ag.empty:
            food_non_ag['Landuse'] = food_non_ag['Landuse'].replace(RENAME_NON_AG)
            food_non_ag = food_non_ag.groupby(['Year'], as_index=False)[VALUE_COL].sum()
            for _, row in food_non_ag.iterrows():
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
        food = load_report_source_csv(scenario, 'quantity_production_kt_separate')
        if food.empty:
            continue
        food = food.query('`Landuse Type` == "Agricultural Management"').copy()
        food['Landuse subtype'] = food['Landuse subtype'].replace(RENAME_AM)
        food = food.groupby(['Year', 'Landuse subtype'], as_index=False)[VALUE_COL].sum()
        for _, row in food.iterrows():
            rows.append({
                'year': int(row['Year']),
                'scenario': scenario,
                'category': row['Landuse subtype'],
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
        '08_food_long_tables.xlsx',
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
        'Food production (Mt, GL)',
        '08_food.svg',
        y_label_x=0.030,
    )


if __name__ == '__main__':
    main()
