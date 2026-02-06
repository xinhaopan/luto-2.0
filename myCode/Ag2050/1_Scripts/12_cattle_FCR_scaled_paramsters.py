import pandas as pd
from plotnine import (
    ggplot, aes, geom_col, facet_wrap, labs,
    theme_minimal, theme, element_text, scale_y_continuous,
    scale_fill_brewer, scale_x_continuous
)

# -------------------------
# 单头产肉量（kg / head）
# -------------------------
meat_yield = {
    "land": 233.24,
    "short": 468 * 0.6817 * (1 - 0.008) ,
    "mid":   652 * 0.6817 * (1 - 0.007),
    "long":  784 * 0.6817 * (1 - 0.021)
}

# feedlot 内部结构
feedlot_split = {
    "short": 0.2,
    "mid":   0.662,
    "long":  0.138
}

# -------------------------
# 读入 feedlot 占比（%）
# -------------------------
feedlots_percent = pd.read_csv(
    "../2_processed_data/cattle_percent_scenarios.csv"
)

# wide → long
feedlot_long = feedlots_percent.melt(
    id_vars="Year",
    var_name="Scenario",
    value_name="feedlot_percent"
)

feedlot_long["feedlot_share"] = feedlot_long["feedlot_percent"] / 100

# -------------------------
# 计算每年产肉量占比
# -------------------------
rows = []

for _, r in feedlot_long.iterrows():
    p_feedlot = r["feedlot_share"]

    # 头数占比
    head_share = {
        "land":  1 - p_feedlot,
        "short": p_feedlot * feedlot_split["short"],
        "mid":   p_feedlot * feedlot_split["mid"],
        "long":  p_feedlot * feedlot_split["long"]
    }

    # 产肉贡献值
    meat_contrib = {
        k: head_share[k] * meat_yield[k]
        for k in head_share
    }

    total_meat = sum(meat_contrib.values())

    for k, v in meat_contrib.items():
        rows.append({
            "Year": r["Year"],
            "Scenario": r["Scenario"],
            "cattle_type": k,
            "meat_share": v / total_meat
        })

meat_share_long = pd.DataFrame(rows)

# plot = (
#     ggplot(meat_share_long, aes(x='factor(Year)', y='meat_share', fill='cattle_type')) +
#     geom_col(position='stack', width=0.8) +  # 堆叠柱形图
#     facet_wrap('~Scenario', ncol=1) +  # 按Scenario分面，垂直排列
#     scale_y_continuous(
#         labels=lambda l: [f'{int(v * 100)}%' for v in l],  # 转换为百分比
#         limits=[0, 1]
#     ) +
#     scale_fill_brewer(
#         type='qual',
#         palette='Set2',
#         labels=['Land', 'Short', 'Mid', 'Long']  # 自定义图例标签
#     ) +
#     labs(
#         x='Year',
#         y='Meat Share (%)',
#         fill='Cattle Type',
#         # title='Meat Share by Cattle Type across Scenarios (2010-2050)'
#     ) +
#     theme_minimal() +
#     theme(
#         figure_size=(14, 10),
#         axis_text_x=element_text(rotation=45, hjust=1, size=8),
#         axis_text_y=element_text(size=9),
#         strip_text=element_text(size=12, face='bold'),
#         legend_position='right',
#         legend_title=element_text(size=10, face='bold'),
#         plot_title=element_text(size=14, face='bold')
#     )
# )
# print(plot)

scenario_map = {
    "AgS4": "Medium",
    "AgS3": "Medium",
    "AgS2": "High",
    "AgS1": "Very_High"
}
beef_demand = pd.read_csv('../0_original_data/beef_demand.csv')
beef_demand = beef_demand.dropna(subset=['New_demand']).query("Imports=='Static'").copy()

# 1. 映射需求表的Scenario
beef_demand['Scenario_mapped'] = beef_demand['Scenario'].map(scenario_map)

# 2. 选择需要的列
cattle_production = meat_share_long.merge(
    beef_demand[['Year', 'Scenario', 'Scenario_mapped', 'New_demand']].rename(columns={'Scenario': 'Scenario_ag'}),
    left_on=['Year', 'Scenario'],           # 2个匹配列
    right_on=['Year', 'Scenario_mapped'],   # 2个匹配列（数量必须相同）
    how='right'
)
cattle_production['production'] = cattle_production['meat_share'] * cattle_production['New_demand']
cattle_production['Year'] = cattle_production['Year'].astype(int)
# plot = (
#     ggplot(cattle_production, aes(x='Year', y='production', fill='cattle_type')) +
#     geom_col(position='stack', width=0.8) +
#     facet_wrap('~Scenario_ag', ncol=2) +
#     scale_fill_brewer(type='qual', palette='Set2', name='Cattle Type') +
#     scale_x_continuous(
#         breaks=list(range(2010, 2051, 5))   # 只显示每5年刻度
#         # 可选：limits=(2010, 2050)
#     ) +
#     scale_y_continuous(labels=lambda l: [f'{v/1e6:.1f}M' for v in l]) +
#     labs(
#         x='Year',
#         y='Production (tonnes)',
#         # title='Cattle Production by Type and Scenario (2010-2050)'
#     ) +
#     theme_minimal() +
#     theme(
#         figure_size=(14, 10),
#         axis_text_x=element_text(rotation=45, hjust=1, size=10),
#         axis_text_y=element_text(size=9),
#         strip_text=element_text(size=12, face='bold'),
#         legend_position='right',
#         legend_title=element_text(size=10, face='bold'),
#         plot_title=element_text(size=14, face='bold')
#     )
# )
#
# print(plot)
fcr_scaled_df = pd.read_csv('../2_processed_data/All_FCR_scaled.csv')
# 1) 统一 join key：stage <-> cattle_type
prod_df = cattle_production.rename(columns={'cattle_type': 'stage'}).copy()
prod_df.to_csv('../2_processed_data/cattle_production_by_stage.csv', index=False)
# 你要按什么维度算“总需求量/合并FCR”
# A: 每年、每个情景分别算：
group_cols = ['Scenario_ag', 'Year']
# B: 如果只想整体算一次，改成：
# group_cols = []

# 2) 把 FCR 系数并到产量表（会得到 每个stage的产量 * 每个SPREAD 的FCR）
tmp = prod_df.merge(fcr_scaled_df, on='stage', how='inner')

# 3) 计算该 stage 对该 SPREAD 的饲料需求量贡献：FCR * 产量
tmp['feed_demand_tonnes'] = tmp['FCR_scaled'] * tmp['production']

# 4) 对每个 SPREAD 求总饲料需求量
feed_by_spread = (
    tmp.groupby(group_cols + ['SPREAD'], as_index=False)
       .agg(total_feed_demand_tonnes=('feed_demand_tonnes', 'sum'))
)

# 5) 同一维度下的总产量（所有 stage 加总）
total_production = (
    prod_df.groupby(group_cols, as_index=False)
           .agg(total_production_tonnes=('production', 'sum'))
)

# 6) 合并并计算“合并后的FCR”
merged_fcr = feed_by_spread.merge(total_production, on=group_cols, how='left')
merged_fcr['FCR_scaled_merged'] = (
    merged_fcr['total_feed_demand_tonnes'] / merged_fcr['total_production_tonnes']
)
merged_fcr.to_csv('../2_processed_data/cattle_FCR_scaled_merged.csv', index=False)
print()