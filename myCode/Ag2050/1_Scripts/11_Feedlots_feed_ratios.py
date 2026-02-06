import pandas as pd
from plotnine import (
    ggplot, aes, geom_line, geom_point,
    facet_wrap, labs, theme_minimal, theme,
    element_text,expand_limits
)

beef_demand = pd.read_csv('../0_original_data/beef_demand.csv')
beef_demand['Year'] = pd.to_numeric(beef_demand['Year'])
feedlots_percent = pd.read_csv('../2_processed_data/cattle_percent_scenarios.csv')
beef_demand = beef_demand.dropna(subset=['New_demand']).query("Imports=='Static'").copy()

# # 创建分面图
# plot = (
#     ggplot(beef_demand, aes(x='Year', y='New_demand', group='Scenario', color='Scenario')) +
#     # geom_line(size=1) +
#     geom_point(size=2) +
#     expand_limits(y=0) +
#     facet_wrap('~Scenario', ncol=2) +  # 按Scenario分面，2列布局（2x2）
#     labs(
#         x='Year',
#         y='Beef Demand (t)',
#         # title='New Demand by Scenario over Years'
#     ) +
#     theme_minimal() +
#     theme(
#         figure_size=(12, 10),
#         axis_text_x=element_text(rotation=45, hjust=1),
#         strip_text=element_text(size=11, face='bold')
#     )
# )
#
# # 显示图形
# print(plot)


short_land_weight = 347 * 0.6817 * (1-0.008)
mid_land_weight = 421 * 0.6817 * (1-0.007)
long_land_weight = 441 * 0.6817 * (1-0.021)

all_land_weight = 233.24
short_closing_wight = 468 * 0.6817 * (1-0.008)
mid_closing_weight = 652 * 0.6817 * (1-0.007)
long_closing_weight = 784 * 0.6817 * (1-0.021)

short_finishing_days = 69
mid_finishing_days = 125
long_finishing_days = 346

short_number_percent = 0.2
medium_number_percent = 0.662
long_number_percent = 0.138

land_feed_df = pd.read_csv('../0_original_data/Feed_ratios_BAU.csv')
feedlots_feed_df = pd.read_excel('../0_original_data/Cattle_feed_parameter.xlsx')
# feedlots_percent_df = pd.read_excel('../0_original_data/Feedlots_percent.xlsx')
# =========================
# 4. LAND stage feed demand
# =========================

# 只保留 beef
beef_land_feed = land_feed_df[land_feed_df['Livestock'].str.lower() == 'beef'].copy()

# 体重（转为 tonnes）
land_weights = {
    'short': short_land_weight / 1000,
    'mid':   mid_land_weight   / 1000,
    'long':  long_land_weight  / 1000
}

land_feed_results = []

for stage, weight_t in land_weights.items():
    tmp = beef_land_feed.copy()
    tmp['stage'] = stage
    tmp['feed_tonnes'] = tmp['FCR_scaled'] * weight_t
    land_feed_results.append(tmp[['stage', 'SPREAD', 'feed_tonnes']])

land_feed_result_df = pd.concat(land_feed_results, ignore_index=True)

# 汇总每类饲料
land_feed_summary = (
    land_feed_result_df
    .groupby(['stage', 'SPREAD'], as_index=False)
    .agg(total_feed_tonnes=('feed_tonnes', 'sum'))
)

# =========================
# 5. FEEDLOT stage feed demand
# =========================

finishing_config = {
    'short': {
        'fed_col': 'short_fed',
        'days': short_finishing_days
    },
    'mid': {
        'fed_col': 'mid_fed',
        'days': mid_finishing_days
    },
    'long': {
        'fed_col': 'long_fed',
        'days': long_finishing_days
    }
}

feedlot_results = []

for stage, cfg in finishing_config.items():
    tmp = feedlots_feed_df.copy()
    tmp['stage'] = stage
    tmp['feed_tonnes'] = (
        tmp[cfg['fed_col']] *        # kg/day
        cfg['days'] *
        tmp['Multiplier'] / 1000     # → tonnes
    )
    feedlot_results.append(
        tmp[['stage', 'LUTO2_commodity', 'feed_tonnes']]
    )

feedlot_feed_result_df = pd.concat(feedlot_results, ignore_index=True)

feedlot_feed_summary = (
    feedlot_feed_result_df
    .groupby(['stage', 'LUTO2_commodity'], as_index=False)
    .agg(total_feed_tonnes=('feed_tonnes', 'sum'))
)

land_feed_summary['phase'] = 'land'
feedlot_feed_summary = feedlot_feed_summary.rename(
    columns={'LUTO2_commodity': 'commodity'}
)
feedlot_feed_summary['phase'] = 'feedlot'
land_feed_summary = land_feed_summary.rename(columns={'SPREAD': 'commodity'})

total_feed_summary = pd.concat(
    [land_feed_summary[['stage','commodity','total_feed_tonnes','phase']],
     feedlot_feed_summary[['stage','commodity','total_feed_tonnes','phase']]],
    ignore_index=True
)
long_feed_df = (
    total_feed_summary
    .groupby(['stage', 'commodity'], as_index=False)
    .agg(total_feed_tonnes=('total_feed_tonnes', 'sum'))
)
closing_weights = {
    'short': 468 * 0.6817 * (1 - 0.008) / 1000,
    'mid': 652 * 0.6817 * (1 - 0.007) / 1000,
    'long': 784 * 0.6817 * (1 - 0.021) / 1000
}

# 直接计算 FCR_scaled_feedlots
long_feed_df['FCR_scaled'] = (
    long_feed_df['total_feed_tonnes'] / long_feed_df['stage'].map(closing_weights)
)
long_feed_df_out = (
    long_feed_df
    .drop(columns=['total_feed_tonnes'])
    .rename(columns={'commodity': 'SPREAD'})
)

land_fcr_df = (
    beef_land_feed[['SPREAD', 'FCR_scaled']]
    .drop_duplicates()          # 如果同一个SPREAD有多行，避免重复；如你需要保留可删掉这一行
    .assign(stage='land')
)
long_feed_df_out = long_feed_df_out[['stage', 'SPREAD', 'FCR_scaled']]
land_fcr_df = land_fcr_df[['stage', 'SPREAD', 'FCR_scaled']]
final_df = pd.concat([long_feed_df_out, land_fcr_df], ignore_index=True)
final_df.to_csv('../2_processed_data/All_FCR_scaled.csv', index=False)