"""
09_Food.py
Combined stacked-area figure for food production by source type, commodity
group, and agricultural management.
"""
import _path_setup  # noqa: F401

import pandas as pd

from tools.parameters import GENERATE_TABLES
from tools.two_row_figure import (
    COMMODITY_TO_FOOD_GROUP,
    FOOD_PRODUCTION_CSV,
    FOOD_PRODUCTION_VALUE_COL,
    RENAME_AM,
    export_long_tables,
    filter_food_detail_rows,
    get_am_colors,
    input_files,
    load_long_tables,
    load_report_source_csv,
    save_three_row_figure,
)

FOOD_COLORS = {
    'Meat & live animals': '#9B4528',
    'Livestock products': '#CA927E',
    'Grains & oilseeds': '#EB8500',
    'All other crops': '#F3BD8B',
    'Fruit & vegetables': '#5A8529',
    'All other horticulture': '#D1D9BF',
}
FOOD_LEGEND_ORDER = list(FOOD_COLORS)
OVERVIEW_COLORS = {
    'Agricultural land-use': '#f39b8b',
    'Agricultural management': '#9A8AB3',
    'Non-agricultural land-use': '#6eabb1',
}


def _load_food(scenario):
    return filter_food_detail_rows(
        load_report_source_csv(scenario, FOOD_PRODUCTION_CSV)
    )


def prepare_overview():
    rows = []

    for scenario in input_files:
        food = _load_food(scenario)
        if food.empty:
            continue
        for raw_name, name in [
            ('Agricultural', 'Agricultural land-use'),
            ('Agricultural_Management', 'Agricultural management'),
            ('Non_Agricultural', 'Non-agricultural land-use'),
        ]:
            data = food.query('Type == @raw_name').copy()
            if data.empty:
                continue
            data = data.groupby('Year', as_index=False)[FOOD_PRODUCTION_VALUE_COL].sum()
            for _, row in data.iterrows():
                rows.append({
                    'year': int(row['Year']),
                    'scenario': scenario,
                    'category': name,
                    'value': float(row[FOOD_PRODUCTION_VALUE_COL]) / 1e6,
                })

    return pd.DataFrame(rows)


def prepare_land_use():
    rows = []

    for scenario in input_files:
        food = _load_food(scenario)
        if food.empty:
            continue
        food = food.query('Type != "Agricultural_Management"').copy()
        food['category'] = food['Commodity'].map(COMMODITY_TO_FOOD_GROUP)
        food = food.dropna(subset=['category'])
        food = food.groupby(['Year', 'category'], as_index=False)[FOOD_PRODUCTION_VALUE_COL].sum()
        for _, row in food.iterrows():
            rows.append({
                'year': int(row['Year']),
                'scenario': scenario,
                'category': row['category'],
                'value': float(row[FOOD_PRODUCTION_VALUE_COL]) / 1e6,
            })

    return pd.DataFrame(rows)


def prepare_am():
    rows = []

    for scenario in input_files:
        food = _load_food(scenario)
        if food.empty:
            continue
        food = food.query('Type == "Agricultural_Management"').dropna(subset=['am']).copy()
        food['am'] = food['am'].replace(RENAME_AM)
        food = food.groupby(['Year', 'am'], as_index=False)[FOOD_PRODUCTION_VALUE_COL].sum()
        for _, row in food.iterrows():
            rows.append({
                'year': int(row['Year']),
                'scenario': scenario,
                'category': row['am'],
                'value': float(row[FOOD_PRODUCTION_VALUE_COL]) / 1e6,
            })

    return pd.DataFrame(rows)


def main():
    workbook = '18_food_long_tables.xlsx'
    if GENERATE_TABLES:
        export_long_tables(
            workbook,
            overview=prepare_overview(),
            land_use=prepare_land_use(),
            agricultural_management=prepare_am(),
        )
    tables = load_long_tables(workbook, 'overview', 'land_use', 'agricultural_management')
    overview_df = tables['overview']
    land_use_df = tables['land_use']
    am_df = tables['agricultural_management']
    am_colors = {
        label: color
        for label, color in get_am_colors().items()
        if label != 'Other land-use'
    }
    save_three_row_figure(
        overview_df,
        land_use_df,
        am_df,
        OVERVIEW_COLORS,
        FOOD_COLORS,
        am_colors,
        'Agri-food production (Mt yr⁻¹)',
        '18_food.svg',
        y_label_x=0.020,
        top_legend_order=FOOD_LEGEND_ORDER,
    )


if __name__ == '__main__':
    main()
