"""
09_Water.py
Combined stacked-area figure for water yield by land use and agricultural
management.
"""
import _path_setup  # noqa: F401

import pandas as pd

from tools.two_row_figure import (
    LU_COLORS,
    RENAME_AM_NON_AG,
    classify_land_use,
    export_long_tables,
    get_am_colors,
    input_files,
    load_report_source_csv,
    save_three_row_figure,
)

VALUE_COL = 'Water Net Yield (ML)'
OVERVIEW_COLORS = {
    'Agricultural land-use': '#f39b8b',
    'Agricultural management': '#9A8AB3',
    'Non-agricultural land-use': '#6eabb1',
}


def load_water_australia(scenario):
    water = load_report_source_csv(scenario, 'water_yield_separate_watershed')
    if water.empty:
        return water
    water = (
        water.groupby(
            ['Water Supply', 'Landuse', 'Type', 'Agricultural Management', 'Year'],
            dropna=False,
            as_index=False,
        )[VALUE_COL]
        .sum()
    )
    water = water.replace(RENAME_AM_NON_AG)
    return water.query('`Water Supply` != "ALL" and Landuse != "ALL"').copy()


def prepare_land_use():
    rows = []

    for scenario in input_files:
        water = load_water_australia(scenario)
        if water.empty:
            continue

        water_ag = water.query('Type == "Agricultural land-use"').copy()
        water_ag['category'] = water_ag['Landuse'].map(classify_land_use)
        water_ag = water_ag.dropna(subset=['category'])
        water_ag = water_ag.groupby(['Year', 'category'], as_index=False)[VALUE_COL].sum()
        for _, row in water_ag.iterrows():
            rows.append({
                'year': int(row['Year']),
                'scenario': scenario,
                'category': row['category'],
                'value': float(row[VALUE_COL]) / 1e3,
            })

        water_non_ag = water.query('Type == "Non-Agricultural Land-use"').copy()
        if not water_non_ag.empty:
            water_non_ag = water_non_ag.groupby(['Year'], as_index=False)[VALUE_COL].sum()
            for _, row in water_non_ag.iterrows():
                rows.append({
                    'year': int(row['Year']),
                    'scenario': scenario,
                    'category': 'Non-agricultural land',
                    'value': float(row[VALUE_COL]) / 1e3,
                })

    return pd.DataFrame(rows)


def prepare_am():
    rows = []

    for scenario in input_files:
        water = load_water_australia(scenario)
        if water.empty:
            continue
        water = water.query('Type == "Agricultural Management"').copy()
        water = water.groupby(['Year', 'Agricultural Management'], as_index=False)[VALUE_COL].sum()
        for _, row in water.iterrows():
            rows.append({
                'year': int(row['Year']),
                'scenario': scenario,
                'category': row['Agricultural Management'],
                'value': float(row[VALUE_COL]) / 1e3,
            })

    return pd.DataFrame(rows)


def prepare_overview():
    rows = []

    for scenario in input_files:
        water = load_water_australia(scenario)
        if water.empty:
            continue
        for raw_name, name in [
            ('Agricultural land-use', 'Agricultural land-use'),
            ('Agricultural Management', 'Agricultural management'),
            ('Non-Agricultural Land-use', 'Non-agricultural land-use'),
        ]:
            data = water.query('Type == @raw_name').copy()
            if data.empty:
                continue
            data = data.groupby('Year', as_index=False)[VALUE_COL].sum()
            for _, row in data.iterrows():
                rows.append({
                    'year': int(row['Year']),
                    'scenario': scenario,
                    'category': name,
                    'value': float(row[VALUE_COL]) / 1e3,
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
        '09_water_long_tables.xlsx',
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
        'Water yield (GL)',
        '09_water.svg',
        y_label_x=0.015,
    )


if __name__ == '__main__':
    main()
