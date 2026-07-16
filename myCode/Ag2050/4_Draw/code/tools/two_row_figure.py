import os

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.gridspec as gridspec
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tools.data_helper import (
    _list_years,
    _list_years_zip,
    _read_csv_from_zip,
    get_path,
    get_zip_info,
)
from tools.parameters import EXCEL_DIR, OUTPUT_DIR, SCENARIO_LABELS, font_size, input_files
from tools.plot_helper import calc_y_range, get_colors, set_plot_style, stacked_area_pos_neg

RENAME_AM = {
    "Asparagopsis taxiformis": "Methane reduction (livestock)",
    "Precision Agriculture": "Agricultural technology (fertiliser)",
    "Ecological Grazing": "Regenerative agriculture (livestock)",
    "Savanna Burning": "Early dry-season savanna burning",
    "AgTech EI": "Agricultural technology (energy)",
    "Biochar": "Biochar (soil amendment)",
    "HIR - Beef": "Managed regeneration (beef)",
    "HIR - Sheep": "Managed regeneration (sheep)",
    "Utility Solar PV": "Utility Solar PV",
    "Onshore Wind": "Onshore wind",
}

RENAME_NON_AG = {
    "Environmental Plantings": "Environmental plantings (mixed species)",
    "Riparian Plantings": "Riparian buffer restoration (mixed species)",
    "Sheep Agroforestry": "Agroforestry (mixed species + sheep)",
    "Beef Agroforestry": "Agroforestry (mixed species + beef)",
    "Carbon Plantings (Block)": "Carbon plantings (monoculture)",
    "Sheep Carbon Plantings (Belt)": "Farm forestry (hardwood timber + sheep)",
    "Beef Carbon Plantings (Belt)": "Farm forestry (hardwood timber + beef)",
    "BECCS": "BECCS (Bioenergy with Carbon Capture and Storage)",
    "Destocked - natural land": "Destocked - natural land",
}

RENAME_AM_NON_AG = {**RENAME_AM, **RENAME_NON_AG}

LU_COLORS = {
    "Dryland cropland and horticulture": "#aecb75",
    "Irrigated cropland and horticulture": "#83b5ff",
    "Dryland grazing (modified pastures)": "#762400",
    "Irrigated grazing (modified pastures)": "#c4669b",
    "Grazing (native vegetation)": "#c4996b",
    "Unallocated land": "#e5d8a8",
    "Non-agricultural land-use": "#3A7F4A",
}

CROPLAND_LUS = {
    "Apples", "Citrus", "Cotton", "Grapes", "Hay", "Nuts",
    "Other non-cereal crops", "Pears", "Plantation fruit", "Rice",
    "Stone fruit", "Sugar", "Summer cereals", "Summer legumes",
    "Summer oilseeds", "Tropical stone fruit", "Vegetables",
    "Winter cereals", "Winter legumes", "Winter oilseeds",
}

MODIFIED_PASTURE_LUS = {
    "Beef - modified land", "Dairy - modified land", "Sheep - modified land",
}

NATIVE_PASTURE_LUS = {
    "Beef - natural land", "Dairy - natural land", "Sheep - natural land",
}

UNALLOCATED_LUS = {
    "Unallocated - modified land", "Unallocated - natural land",
}

FOOD_PRODUCTION_CSV = 'quantity_production_t_separate'
FOOD_PRODUCTION_VALUE_COL = 'Production (t/KL)'
COMMODITY_TO_FOOD_GROUP = {
    'beef lexp': 'Meat & live animals',
    'beef meat': 'Meat & live animals',
    'sheep lexp': 'Meat & live animals',
    'sheep meat': 'Meat & live animals',
    'dairy': 'Livestock products',
    'sheep wool': 'Livestock products',
    'summer cereals': 'Grains & oilseeds',
    'winter cereals': 'Grains & oilseeds',
    'summer legumes': 'Grains & oilseeds',
    'winter legumes': 'Grains & oilseeds',
    'summer oilseeds': 'Grains & oilseeds',
    'winter oilseeds': 'Grains & oilseeds',
    'rice': 'Grains & oilseeds',
    'cotton': 'All other crops',
    'hay': 'All other crops',
    'other non-cereal crops': 'All other crops',
    'sugar': 'All other crops',
    'apples': 'Fruit & vegetables',
    'citrus': 'Fruit & vegetables',
    'grapes': 'Fruit & vegetables',
    'pears': 'Fruit & vegetables',
    'plantation fruit': 'Fruit & vegetables',
    'stone fruit': 'Fruit & vegetables',
    'tropical stone fruit': 'Fruit & vegetables',
    'vegetables': 'Fruit & vegetables',
    'nuts': 'All other horticulture',
}

BIODIVERSITY_BACKEND = 'Suitability'

COLORS_FILE = 'tools/land use colors.xlsx'
UNIT_DEJAVU_CHARS = {'₂', '⁻', '¹'}


def _split_unit_font_runs(text):
    runs = []
    current_text = []
    current_family = None
    for char in text:
        family = 'DejaVu Sans' if char in UNIT_DEJAVU_CHARS else 'Arial'
        if family != current_family and current_text:
            runs.append((''.join(current_text), current_family))
            current_text = []
        current_text.append(char)
        current_family = family
    if current_text:
        runs.append((''.join(current_text), current_family))
    return runs


def _measure_text_runs(fig, runs, fontsize, fontweight):
    temp_artists = [
        fig.text(
            0, 0, text,
            fontsize=fontsize,
            fontfamily=family,
            fontweight=fontweight,
            alpha=0,
        )
        for text, family in runs
    ]
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    widths = [
        artist.get_window_extent(renderer=renderer).width
        for artist in temp_artists
    ]
    for artist in temp_artists:
        artist.remove()
    return widths


def _add_vertical_unit_label(fig, x, y, text, fontsize, fontweight='normal'):
    runs = _split_unit_font_runs(text)
    if len(runs) == 1:
        text_run, family = runs[0]
        fig.text(
            x, y, text_run,
            ha='center', va='center', rotation=90,
            fontsize=fontsize, fontfamily=family, fontweight=fontweight,
        )
        return

    widths = _measure_text_runs(fig, runs, fontsize, fontweight)
    total_width = sum(widths)
    cursor = -total_width / 2
    fig_height_px = fig.bbox.height
    for (text_run, family), width in zip(runs, widths):
        offset_px = cursor + width / 2
        fig.text(
            x, y + offset_px / fig_height_px, text_run,
            ha='center', va='center', rotation=90,
            fontsize=fontsize, fontfamily=family, fontweight=fontweight,
        )
        cursor += width


def _water_supply_prefix(water_supply):
    if water_supply is None or pd.isna(water_supply):
        return None
    value = str(water_supply).strip().lower()
    if value in {'dryland', 'dry'}:
        return 'Dryland'
    if value in {'irrigated', 'irr'}:
        return 'Irrigated'
    return None


def classify_land_use(name, water_supply=None):
    prefix = _water_supply_prefix(water_supply)
    if name in CROPLAND_LUS:
        return f"{prefix} cropland and horticulture" if prefix else None
    if name in MODIFIED_PASTURE_LUS:
        return f"{prefix} grazing (modified pastures)" if prefix else None
    if name in NATIVE_PASTURE_LUS:
        return "Grazing (native vegetation)"
    if name in UNALLOCATED_LUS:
        return "Unallocated land"
    return None


def _filter_region_level(df: pd.DataFrame, level: str = 'region_NRM') -> pd.DataFrame:
    """Keep only one region_level to avoid double-counting with dual-region output.

    Backward-compatible: returns df unchanged if 'region_level' column is absent
    (old single-region format).
    """
    if 'region_level' not in df.columns:
        return df
    return df[df['region_level'] == level].drop(columns='region_level').reset_index(drop=True)


def load_report_source_csv(scenario, csv_name):
    info = get_zip_info(scenario)
    frames = []
    if info is not None:
        zip_path, prefix = info
        years = _list_years_zip(zip_path, prefix)
        for year in years:
            df = _read_csv_from_zip(zip_path, prefix, year, csv_name)
            if df is not None and not df.empty:
                if 'Year' not in df.columns:
                    df = df.copy()
                    df['Year'] = int(year)
                frames.append(_filter_region_level(df))
    else:
        base = get_path(scenario)
        for year in _list_years(base):
            path = os.path.join(base, f'out_{year}', f'{csv_name}_{year}.csv')
            if os.path.exists(path):
                df = pd.read_csv(path)
                if not df.empty:
                    if 'Year' not in df.columns:
                        df = df.copy()
                        df['Year'] = int(year)
                    frames.append(_filter_region_level(df))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def filter_transition_cost_rows(df):
    """Select detailed transition costs without double-counting regional output."""
    if df.empty:
        return df
    cost_type_col = 'Cost-type' if 'Cost-type' in df.columns else 'Type'
    mask = (
        df['From-land-use'].ne('ALL')
        & df['To-land-use'].ne('ALL')
        & df[cost_type_col].ne('ALL')
    )
    detailed = df.loc[mask].copy()
    if 'region' in detailed.columns and detailed['region'].eq('AUSTRALIA').any():
        detailed = detailed[detailed['region'].eq('AUSTRALIA')].copy()
    return detailed


def require_columns(df, columns, source_name):
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f'{source_name} is missing required columns: {missing}')


def assert_series_close(actual, expected, label, atol=2e-3, rtol=1e-6):
    """Fail loudly when plotted totals diverge from an independent aggregate."""
    actual, expected = actual.align(expected, join='inner')
    if actual.empty:
        return
    close = np.isclose(
        actual.astype(float).to_numpy(),
        expected.astype(float).to_numpy(),
        atol=atol,
        rtol=rtol,
    )
    if close.all():
        return
    mismatch = pd.DataFrame({'actual': actual, 'expected': expected})[~close]
    raise ValueError(f'{label} total validation failed:\n{mismatch.to_string()}')


def filter_am_cost_rows(df):
    """Keep AM cost detail and remove the duplicate Cost_type=ALL hierarchy."""
    if df.empty:
        return df
    require_columns(df, ['Cost_type'], 'economics_am_cost')
    return df[df['Cost_type'] != 'ALL'].copy()


def filter_area_ag_rows(df):
    """Keep national agricultural area detail and verify its ALL-water total."""
    if df.empty:
        return df
    required = ['region', 'Water_supply', 'Land-use', 'Year', 'Area (ha)']
    require_columns(df, required, 'area_agricultural_landuse')
    national = df[df['region'].eq('AUSTRALIA')].copy()
    detail = national[
        national['Water_supply'].ne('ALL')
        & national['Land-use'].ne('ALL')
    ].copy()
    unknown = detail[
        detail.apply(
            lambda row: classify_land_use(row['Land-use'], row['Water_supply']) is None,
            axis=1,
        )
    ]['Land-use'].drop_duplicates()
    if not unknown.empty:
        raise ValueError(f'Unclassified agricultural land uses: {sorted(unknown)}')
    expected = (
        national[
            national['Water_supply'].eq('ALL')
            & national['Land-use'].ne('ALL')
        ]
        .groupby('Year')['Area (ha)']
        .sum()
    )
    actual = detail.groupby('Year')['Area (ha)'].sum()
    assert_series_close(actual, expected, 'Agricultural area hierarchy', atol=10.0)
    return detail


def filter_area_am_rows(df):
    """Keep AM area detail and verify its independent ALL-water hierarchy."""
    if df.empty:
        return df
    required = ['region', 'Water_supply', 'Land-use', 'Year', 'Area (ha)']
    require_columns(df, required, 'area_agricultural_management')
    national = df[df['region'].eq('AUSTRALIA')].copy()
    detail = national[
        national['Water_supply'].ne('ALL')
        & national['Land-use'].ne('ALL')
    ].copy()
    expected = (
        national[
            national['Water_supply'].eq('ALL')
            & national['Land-use'].ne('ALL')
        ]
        .groupby('Year')['Area (ha)']
        .sum()
    )
    actual = detail.groupby('Year')['Area (ha)'].sum()
    assert_series_close(actual, expected, 'Agricultural-management area hierarchy', atol=10.0)
    return detail


def filter_food_detail_rows(df):
    """Keep one food-production hierarchy and verify it against aggregate rows."""
    if df.empty:
        return df
    required = [
        'region', 'Water_supply', 'Commodity', 'Type', 'Year',
        FOOD_PRODUCTION_VALUE_COL,
    ]
    require_columns(df, required, FOOD_PRODUCTION_CSV)
    national = df[df['region'].eq('AUSTRALIA')].copy()
    detail = national[
        national['Water_supply'].ne('ALL')
        & national['Commodity'].ne('ALL')
    ].copy()
    unknown_types = sorted(set(detail['Type'].dropna()) - {
        'Agricultural', 'Agricultural_Management', 'Non_Agricultural',
    })
    if unknown_types:
        raise ValueError(f'Unclassified food-production types: {unknown_types}')
    unknown_commodities = sorted(
        set(detail['Commodity'].dropna()) - set(COMMODITY_TO_FOOD_GROUP)
    )
    if unknown_commodities:
        raise ValueError(f'Unclassified food commodities: {unknown_commodities}')

    expected_parts = [
        national[
            national['Type'].eq('Agricultural')
            & national['Water_supply'].eq('ALL')
            & national['Commodity'].ne('ALL')
        ],
        national[
            national['Type'].eq('Agricultural_Management')
            & national['Water_supply'].eq('ALL')
            & national['Commodity'].eq('ALL')
        ],
        national[national['Type'].eq('Non_Agricultural')],
    ]
    expected = (
        pd.concat(expected_parts, ignore_index=True)
        .groupby('Year')[FOOD_PRODUCTION_VALUE_COL]
        .sum()
    )
    actual = detail.groupby('Year')[FOOD_PRODUCTION_VALUE_COL].sum()
    assert_series_close(actual, expected, 'Food-production hierarchy', atol=10.0)
    return detail


def filter_biodiversity_rows(df):
    """Return national all-cell Suitability rows at one non-aggregate hierarchy."""
    if df.empty:
        return df
    require_columns(
        df,
        [
            'region', 'backend', 'Water_supply', 'Landuse', 'Type',
            'Agricultural Management', 'Area Weighted Score (ha)',
        ],
        'biodiversity_overall_priority_scores',
    )
    if BIODIVERSITY_BACKEND not in set(df['backend'].dropna()):
        raise ValueError(
            f'Biodiversity backend {BIODIVERSITY_BACKEND!r} is unavailable; '
            f'found {sorted(df["backend"].dropna().unique())}'
        )
    detail = df[
        (df['region'] == 'AUSTRALIA')
        & (df['backend'] == BIODIVERSITY_BACKEND)
        & (df['Water_supply'] != 'ALL')
        & (df['Landuse'] != 'ALL')
    ].copy()
    duplicate_am_total = (
        detail['Type'].eq('Agricultural Management')
        & detail['Agricultural Management'].eq('ALL')
    )
    return detail[~duplicate_am_total].copy()


def filter_water_detail_rows(df):
    """Collapse watersheds and keep one detailed water-yield hierarchy."""
    if df.empty:
        return df
    required = [
        'Water Supply', 'Landuse', 'Type', 'Agricultural Management',
        'Year', 'Water Net Yield (ML)',
    ]
    require_columns(df, required, 'water_yield_separate_watershed')
    grouped = (
        df.groupby(
            ['Water Supply', 'Landuse', 'Type', 'Agricultural Management', 'Year'],
            dropna=False,
            as_index=False,
        )['Water Net Yield (ML)']
        .sum()
    )

    detail = grouped[
        (grouped['Water Supply'] != 'ALL')
        & (grouped['Landuse'] != 'ALL')
    ].copy()
    duplicate_am_total = (
        detail['Type'].eq('Agricultural Management')
        & detail['Agricultural Management'].eq('ALL')
    )
    detail = detail[~duplicate_am_total].copy()

    expected_parts = [
        grouped[
            grouped['Type'].eq('Agricultural Land-use')
            & grouped['Water Supply'].eq('ALL')
            & grouped['Landuse'].eq('ALL')
        ],
        grouped[
            grouped['Type'].eq('Agricultural Management')
            & grouped['Water Supply'].eq('ALL')
            & grouped['Landuse'].eq('ALL')
            & grouped['Agricultural Management'].eq('ALL')
        ],
        grouped[
            grouped['Type'].eq('Non-Agricultural Land-use')
            & grouped['Water Supply'].ne('ALL')
            & grouped['Landuse'].eq('ALL')
        ],
    ]
    expected = (
        pd.concat(expected_parts, ignore_index=True)
        .groupby('Year')['Water Net Yield (ML)']
        .sum()
    )
    actual = detail.groupby('Year')['Water Net Yield (ML)'].sum()
    # Watershed CSV layers are independently rounded; their national hierarchies
    # can differ by roughly 0.2 GL after summing thousands of rows.
    assert_series_close(actual, expected, 'Water-yield hierarchy', atol=250.0)
    return detail


def prepare_net_economic_return_overview():
    rows = []
    for scenario in input_files:
        for csv_name, category, sign, query in [
            (
                'economics_ag_revenue', 'Ag revenue', 1.0,
                'region == "AUSTRALIA" and Water_supply != "ALL" and '
                'Type != "ALL" and `Land-use` != "ALL"',
            ),
            (
                'economics_ag_cost', 'Ag cost', -1.0,
                'region == "AUSTRALIA" and Water_supply != "ALL" and '
                'Type != "ALL" and `Land-use` != "ALL"',
            ),
            (
                'economics_am_revenue', 'Agmgt revenue', 1.0,
                'region == "AUSTRALIA" and Water_supply != "ALL" and '
                '`Land-use` != "ALL" and `Management Type` != "ALL"',
            ),
            (
                'economics_am_cost', 'Agmgt cost', -1.0,
                'region == "AUSTRALIA" and Water_supply != "ALL" and '
                '`Land-use` != "ALL" and `Management Type` != "ALL"',
            ),
            (
                'economics_non_ag_revenue', 'Non-ag revenue', 1.0,
                'region == "AUSTRALIA" and `Land-use` != "ALL"',
            ),
            (
                'economics_non_ag_cost', 'Non-ag cost', -1.0,
                'region == "AUSTRALIA" and `Land-use` != "ALL"',
            ),
        ]:
            data = load_report_source_csv(scenario, csv_name)
            if data.empty:
                continue
            data = data.query(query).copy()
            if csv_name == 'economics_am_cost':
                data = filter_am_cost_rows(data)
            for _, row in data.groupby('Year', as_index=False)['Value ($)'].sum().iterrows():
                rows.append({
                    'year': int(row['Year']),
                    'scenario': scenario,
                    'category': category,
                    'value': sign * float(row['Value ($)']) / 1e9,
                })

        for csv_name, category in [
            ('transition_ag2ag_cost', 'Transition(ag->ag) cost'),
            ('transition_nonag2ag_cost', 'Transition(non-ag->ag) cost'),
            ('transition_ag2nonag_cost', 'Transition(ag->non-ag) cost'),
        ]:
            data = filter_transition_cost_rows(load_report_source_csv(scenario, csv_name))
            if data.empty:
                continue
            for _, row in data.groupby('Year', as_index=False)['Cost ($)'].sum().iterrows():
                rows.append({
                    'year': int(row['Year']),
                    'scenario': scenario,
                    'category': category,
                    'value': -float(row['Cost ($)']) / 1e9,
                })

    result = pd.DataFrame(rows, columns=['year', 'scenario', 'category', 'value'])
    _validate_net_economic_return(result)
    return result


def _validate_net_economic_return(result):
    for scenario in input_files:
        expected_parts = []
        for csv_name, query in [
            (
                'economics_ag_profit',
                'region == "AUSTRALIA" and Water_supply == "ALL" and `Land-use` == "ALL"',
            ),
            (
                'economics_am_profit',
                'region == "AUSTRALIA" and Water_supply == "ALL" and '
                '`Land-use` == "ALL" and `Management Type` == "ALL"',
            ),
            (
                'economics_non_ag_profit',
                'region == "AUSTRALIA" and `Land-use` == "ALL"',
            ),
        ]:
            data = load_report_source_csv(scenario, csv_name)
            if not data.empty:
                expected_parts.append(data.query(query)[['Year', 'Value ($)']])
        if not expected_parts:
            continue
        expected = (
            pd.concat(expected_parts, ignore_index=True)
            .groupby('Year')['Value ($)']
            .sum()
            / 1e9
        )
        actual = (
            result[result['scenario'] == scenario]
            .groupby('year')['value']
            .sum()
        )
        assert_series_close(actual, expected, f'{scenario} net economic return')


def prepare_ghg_overview():
    rows = []
    sources = [
        (
            'GHG_emissions_separate_agricultural_landuse',
            'Agricultural land-use',
            'region == "AUSTRALIA" and Water_supply != "ALL" and '
            'Source != "ALL" and `Land-use` != "ALL"',
            'region == "AUSTRALIA" and Water_supply == "ALL" and '
            'Source == "ALL" and `Land-use` == "ALL"',
        ),
        (
            'GHG_emissions_separate_agricultural_management',
            'Agricultural management',
            'region == "AUSTRALIA" and Water_supply != "ALL" and '
            '`Land-use` != "ALL" and `Agricultural Management Type` != "ALL"',
            'region == "AUSTRALIA" and Water_supply == "ALL" and '
            '`Land-use` == "ALL" and `Agricultural Management Type` == "ALL"',
        ),
        (
            'GHG_emissions_separate_no_ag_reduction',
            'Non-agricultural land-use',
            'region == "AUSTRALIA" and `Land-use` != "ALL"',
            'region == "AUSTRALIA" and `Land-use` == "ALL"',
        ),
        (
            'GHG_emissions_separate_transition_penalty',
            'Transition',
            'region == "AUSTRALIA" and Type != "ALL" and Water_supply != "ALL"',
            'region == "AUSTRALIA" and Type == "ALL" and Water_supply == "ALL"',
        ),
    ]
    for scenario in input_files:
        for csv_name, category, query, aggregate_query in sources:
            data = load_report_source_csv(scenario, csv_name)
            if data.empty:
                continue
            detail = data.query(query).copy()
            actual = detail.groupby('Year')['Value (t CO2e)'].sum()
            expected = data.query(aggregate_query).groupby('Year')['Value (t CO2e)'].sum()
            assert_series_close(
                actual,
                expected,
                f'{scenario} {category} GHG hierarchy',
                atol=100.0,
            )
            for _, row in detail.groupby('Year', as_index=False)['Value (t CO2e)'].sum().iterrows():
                rows.append({
                    'year': int(row['Year']),
                    'scenario': scenario,
                    'category': category,
                    'value': float(row['Value (t CO2e)']) / 1e6,
                })

        offland = load_report_source_csv(scenario, 'GHG_emissions_offland_commodity')
        if not offland.empty:
            for _, row in offland.groupby('Year', as_index=False)['Total GHG Emissions (tCO2e)'].sum().iterrows():
                rows.append({
                    'year': int(row['Year']),
                    'scenario': scenario,
                    'category': 'Off-land commodities',
                    'value': float(row['Total GHG Emissions (tCO2e)']) / 1e6,
                })

    return pd.DataFrame(rows, columns=['year', 'scenario', 'category', 'value'])


def prepare_biodiversity_overview():
    category_map = {
        'Agricultural Land-use': 'Agricultural land-use',
        'Agricultural Management': 'Agricultural management',
        'Non-Agricultural Land-use': 'Non-agricultural land-use',
    }
    rows = []
    for scenario in input_files:
        data = filter_biodiversity_rows(
            load_report_source_csv(scenario, 'biodiversity_overall_priority_scores')
        )
        for _, row in data.groupby(['Year', 'Type'], as_index=False)['Area Weighted Score (ha)'].sum().iterrows():
            category = category_map.get(row['Type'])
            if category is not None:
                rows.append({
                    'year': int(row['Year']),
                    'scenario': scenario,
                    'category': category,
                    'value': float(row['Area Weighted Score (ha)']) / 1e6,
                })

        totals = load_report_source_csv(scenario, 'biodiversity_overall_priority_scores_all')
        if totals.empty:
            continue
        expected = (
            totals[
                totals['region'].eq('AUSTRALIA')
                & totals['backend'].eq(BIODIVERSITY_BACKEND)
                & totals['Type'].eq('ALL')
            ]
            .set_index('Year')['Area Weighted Score (ha)']
            / 1e6
        )
        scenario_rows = pd.DataFrame(rows)
        actual = (
            scenario_rows[scenario_rows['scenario'] == scenario]
            .groupby('year')['value']
            .sum()
        )
        assert_series_close(actual, expected, f'{scenario} biodiversity')
    return pd.DataFrame(rows, columns=['year', 'scenario', 'category', 'value'])


def get_am_colors():
    am_colors_all = get_colors(COLORS_FILE, 'am')
    return {
        label: color
        for label, color in am_colors_all.items()
        if label not in {
            'Public and indigenous land, urban land, plantation forestry, and water bodies',
            'No agricultural management',
        }
    }


def _draw_row(fig, gs, row_idx, df_long, colors, y_range, show_titles):
    years = list(range(2010, 2051))
    cats = sorted(colors.keys(), reverse=True)  # reverse-alpha: Z=bottom, A=top of positive stack
    axes = []

    for col_idx, scenario in enumerate(input_files):
        if col_idx == 0:
            ax = fig.add_subplot(gs[row_idx, col_idx])
        else:
            ax = fig.add_subplot(gs[row_idx, col_idx], sharey=axes[0])
        axes.append(ax)

        df_s = df_long[df_long['scenario'] == scenario]
        df_wide = (
            df_s.pivot_table(index='year', columns='category', values='value', aggfunc='sum')
            .reindex(index=years)
            .reindex(columns=cats, fill_value=0)
        )

        title = SCENARIO_LABELS.get(scenario, scenario).split('\n')[0] if show_titles else ''
        stacked_area_pos_neg(
            ax,
            df_wide,
            colors=colors,
            alpha=0.60,
            title_name=title,
            ylabel='',
            y_ticks_all=y_range,
            show_legend=False,
        )
        ax.plot(
            [0, 1], [0, 0],
            transform=ax.transAxes,
            color='black',
            linewidth=1.2,
            zorder=50,
            clip_on=False,
        )

        ax.set_xticks([year for year in range(2010, 2051, 10)])
        ax.tick_params(axis='x', labelrotation=45, labelbottom=True)
        if row_idx == 0:
            ax.tick_params(axis='x', labelbottom=False)

        if col_idx != 0:
            ax.tick_params(axis='y', labelleft=False)

        if row_idx != 0 and col_idx != 0:
            fig.canvas.draw()
            labels = ax.get_xticklabels()
            if labels:
                labels[0].set_visible(False)

    return axes


def _add_patch_legend(ax, colors, legend_order=None):
    if legend_order:
        ordered_labels = [label for label in legend_order if label in colors]
        ordered_labels += sorted(label for label in colors if label not in ordered_labels)
    else:
        ordered_labels = sorted(colors)
    handles = [
        mpatches.Patch(facecolor=colors[label], edgecolor='none', label=label)
        for label in ordered_labels
    ]
    ax.axis('off')
    ax.legend(
        handles,
        [handle.get_label() for handle in handles],
        loc='center left',
        ncol=1,
        frameon=False,
        handlelength=0.7,
        handleheight=0.7,
        handletextpad=0.4,
        labelspacing=0.6,
        columnspacing=1.0,
    )


def _add_mixed_legend(ax, patch_colors, line_label=None, line_color='black'):
    handles = sorted(
        [mpatches.Patch(facecolor=color, edgecolor='none', label=label)
         for label, color in patch_colors.items()],
        key=lambda h: h.get_label(),
    )  # A-Z alphabetical order
    if line_label:
        handles.append(mlines.Line2D([], [], color=line_color, linewidth=1.5, label=line_label))
    ax.axis('off')
    ax.legend(
        handles,
        [handle.get_label() for handle in handles],
        loc='center left',
        ncol=1,
        frameon=False,
        handlelength=0.7,
        handleheight=0.7,
        handletextpad=0.4,
        labelspacing=0.6,
        columnspacing=1.0,
    )


def save_two_row_figure(df_top, df_bottom, top_colors, bottom_colors, y_label, output_name):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    set_plot_style(font_size=font_size)

    df_top = df_top[df_top['category'].isin(top_colors)].copy()
    df_bottom = df_bottom[df_bottom['category'].isin(bottom_colors)].copy()
    y_range_top = calc_y_range(df_top, 5)
    y_range_bottom = calc_y_range(df_bottom, 5)

    fig = plt.figure(figsize=(21.5, 9.2))
    gs = gridspec.GridSpec(
        2, 5, figure=fig,
        width_ratios=[1, 1, 1, 1, 0.78],
        hspace=0.26, wspace=0.10,
    )
    fig.subplots_adjust(left=0.08, right=0.95, top=0.90, bottom=0.12)

    axes_top = _draw_row(fig, gs, 0, df_top, top_colors, y_range_top, show_titles=False)
    axes_bottom = _draw_row(fig, gs, 1, df_bottom, bottom_colors, y_range_bottom, show_titles=False)

    for ax in axes_top + axes_bottom:
        ax.set_xlabel('')

    # top=0.90, bottom=0.12, hspace=0.26, 2 rows:
    #   row_h = 0.78/2.26 ≈ 0.345,  gap_h = 0.090
    #   Row 0: 0.555–0.900,  gap: 0.465–0.555,  Row 1: 0.120–0.465
    fig.text(
        0.43, 0.935, 'Land-use',
        ha='center', va='bottom',
        fontsize=font_size, fontfamily='Arial', fontweight='normal',
    )
    fig.text(
        0.43, 0.503, 'Agricultural management',
        ha='center', va='bottom',
        fontsize=font_size, fontfamily='Arial', fontweight='normal',
    )
    _add_vertical_unit_label(fig, 0.038, 0.50, y_label, font_size)

    # Bold scenario column headers — placed just above the section label
    fig.canvas.draw()
    for ax, scenario in zip(axes_top, input_files):
        pos = ax.get_position()
        cx = (pos.x0 + pos.x1) / 2
        fig.text(cx, 0.968, SCENARIO_LABELS.get(scenario, scenario),
                 ha='center', va='bottom',
                 fontsize=font_size, fontweight='bold', fontfamily='Arial')

    _add_patch_legend(fig.add_subplot(gs[0, 4]), top_colors)
    _add_patch_legend(fig.add_subplot(gs[1, 4]), bottom_colors)

    out = os.path.join(OUTPUT_DIR, output_name)
    fig.savefig(out, dpi=600, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out}")


def save_three_row_figure(
    df_overview,
    df_top,
    df_bottom,
    overview_colors,
    top_colors,
    bottom_colors,
    y_label,
    output_name,
    total_legend_label='Sum',
    y_label_x=0.038,
    overview_required_ticks=None,
    top_legend_order=None,
):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    set_plot_style(font_size=font_size)

    df_overview = df_overview[df_overview['category'].isin(overview_colors)].copy()
    df_top = df_top[df_top['category'].isin(top_colors)].copy()
    df_bottom = df_bottom[df_bottom['category'].isin(bottom_colors)].copy()

    y_range_overview = calc_y_range(df_overview, 5)
    if overview_required_ticks:
        min_v, max_v, ticks = y_range_overview
        ticks = sorted(set(ticks) | set(overview_required_ticks))
        min_v = min(min_v, ticks[0])
        max_v = max(max_v, ticks[-1])
        y_range_overview = (min_v, max_v, ticks)
    y_range_top = calc_y_range(df_top, 5)
    y_range_bottom = calc_y_range(df_bottom, 5)

    fig = plt.figure(figsize=(21.5, 19.5))
    gs = gridspec.GridSpec(
        3, 5, figure=fig,
        width_ratios=[1, 1, 1, 1, 0.82],
        hspace=0.24, wspace=0.10,
    )
    fig.subplots_adjust(left=0.07, right=0.97, top=0.84, bottom=0.10)

    axes_overview = _draw_row(fig, gs, 0, df_overview, overview_colors, y_range_overview, show_titles=False)
    axes_top = _draw_row(fig, gs, 1, df_top, top_colors, y_range_top, show_titles=False)
    axes_bottom = _draw_row(fig, gs, 2, df_bottom, bottom_colors, y_range_bottom, show_titles=False)

    years = list(range(2010, 2051))
    for ax, scenario in zip(axes_overview, input_files):
        df_s = df_overview[df_overview['scenario'] == scenario]
        total = (
            df_s.groupby('year', as_index=False)['value']
            .sum()
            .set_index('year')
            .reindex(years, fill_value=0)['value']
        )
        ax.plot(years, total.values, color='black', linewidth=1.5, zorder=60)
        ax.tick_params(axis='x', labelbottom=False)

    for ax in axes_top:
        ax.tick_params(axis='x', labelbottom=False)

    for ax in axes_overview + axes_top + axes_bottom:
        ax.set_xlabel('')

    # Use actual axes positions so labels sit just above each row without overlap
    fig.canvas.draw()
    pos_ov = axes_overview[0].get_position()
    pos_tp = axes_top[0].get_position()
    pos_bt = axes_bottom[0].get_position()

    _section_gap = 0.008   # gap between row top edge and section-label bottom
    _header_gap  = 0.020   # additional gap from section label to column-header bottom

    fig.text(
        0.43, pos_ov.y1 + _section_gap, 'Total',
        ha='center', va='bottom',
        fontsize=font_size, fontfamily='Arial', fontweight='normal',
    )
    fig.text(
        0.43, pos_tp.y1 + _section_gap, 'Land-use',
        ha='center', va='bottom',
        fontsize=font_size, fontfamily='Arial', fontweight='normal',
    )
    fig.text(
        0.43, pos_bt.y1 + _section_gap, 'Agricultural management',
        ha='center', va='bottom',
        fontsize=font_size, fontfamily='Arial', fontweight='normal',
    )
    _add_vertical_unit_label(fig, y_label_x, (pos_bt.y0 + pos_ov.y1) / 2, y_label, font_size)

    # Bold scenario column headers just above the Total section label
    header_y = pos_ov.y1 + _section_gap + _header_gap
    for ax, scenario in zip(axes_overview, input_files):
        pos = ax.get_position()
        cx = (pos.x0 + pos.x1) / 2
        fig.text(cx, header_y, SCENARIO_LABELS.get(scenario, scenario),
                 ha='center', va='bottom',
                 fontsize=font_size, fontweight='bold', fontfamily='Arial')

    _add_mixed_legend(fig.add_subplot(gs[0, 4]), overview_colors, line_label=total_legend_label)
    _add_patch_legend(fig.add_subplot(gs[1, 4]), top_colors, top_legend_order)
    _add_patch_legend(fig.add_subplot(gs[2, 4]), bottom_colors)

    out = os.path.join(OUTPUT_DIR, output_name)
    fig.savefig(out, dpi=600, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out}")


def export_long_tables(workbook_name, **tables):
    os.makedirs(EXCEL_DIR, exist_ok=True)
    out = os.path.join(EXCEL_DIR, workbook_name)
    with pd.ExcelWriter(out, engine='openpyxl') as writer:
        for sheet_name, df in tables.items():
            export_df = df.copy()
            if not export_df.empty:
                if 'scenario' in export_df.columns:
                    export_df['scenario_label'] = export_df['scenario'].map(SCENARIO_LABELS).fillna(export_df['scenario'])
                sort_cols = [col for col in ['scenario', 'year', 'category'] if col in export_df.columns]
                if sort_cols:
                    export_df = export_df.sort_values(sort_cols).reset_index(drop=True)
            else:
                export_df = pd.DataFrame({'note': ['No data']})
            export_df.to_excel(writer, sheet_name=sheet_name[:31], index=False)
    print(f"Saved: {out}")
    return out


def table_cache_path(workbook_name):
    return os.path.join(EXCEL_DIR, workbook_name)


def missing_table_error(path, sheet_name=None):
    target = f'{path}' if sheet_name is None else f'{path} [{sheet_name}]'
    return FileNotFoundError(
        f'Missing table cache: {target}\n'
        'Set GENERATE_TABLES = True in tools/parameters.py and run the script once '
        'before plotting with GENERATE_TABLES = False.'
    )


def load_long_tables(workbook_name, *sheet_names):
    path = table_cache_path(workbook_name)
    if not os.path.exists(path):
        raise missing_table_error(path)

    tables = {}
    for sheet_name in sheet_names:
        excel_sheet = sheet_name[:31]
        try:
            df = pd.read_excel(path, sheet_name=excel_sheet)
        except ValueError as exc:
            raise missing_table_error(path, excel_sheet) from exc
        if list(df.columns) == ['note']:
            df = pd.DataFrame()
        if 'scenario_label' in df.columns:
            df = df.drop(columns='scenario_label')
        tables[sheet_name] = df
    return tables
