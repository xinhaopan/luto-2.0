import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FuncFormatter

# 参数
ghg_per_weight = 9.67  # kg CO₂-e per kg liveweight
cost_per_weight = 2.7  # $ per kg liveweight
retail_meat = 0.6817  # retail meat per kg liveweight
growth_time = 0.8  # 年份校正系数
cattle_weight = 634.67  # kg/head
export_live_percent = 0.116  # 11.6% of meat weight is exported live

# 读取数据
df = pd.read_csv('../2_processed_data/cattle_number_forecast.csv')

scenarios = ['Low', 'Medium', 'High', 'Very_High']
results = []

for _, row in df.iterrows():
    year = row['Year']

    for scenario in scenarios:
        cattle_num = row[scenario]

        # 步骤1: 应用年份校正系数
        adjusted_cattle = cattle_num * growth_time

        # 步骤2: 计算总活重 (kg)
        total_liveweight = adjusted_cattle * cattle_weight

        # 步骤3: 计算活畜出口重量 (kg)
        live_export_weight = total_liveweight * export_live_percent

        # 步骤4: 计算剩余用于屠宰的重量 (kg)
        slaughter_weight = total_liveweight * (1 - export_live_percent)

        # 步骤5: 计算零售肉重量 (kg)
        retail_meat_weight = slaughter_weight * retail_meat

        # 步骤6: 计算温室气体排放 (tonnes CO₂-e)
        ghg_emissions = total_liveweight * ghg_per_weight / 1000

        # 步骤7: 计算总成本 (Million $)
        total_cost = total_liveweight * cost_per_weight / 1e6

        # 汇总结果
        indicators = {
            'Year': year,
            'Scenario': scenario,
            'Original_Cattle_Number': cattle_num,
            'Adjusted_Cattle_Number': adjusted_cattle,
            'Total_Liveweight_kg': total_liveweight,
            'Live_Export_Weight_kg': live_export_weight,
            'Slaughter_Weight_kg': slaughter_weight,
            'Retail_Meat_kg': retail_meat_weight,
            'GHG_Emissions_tonnes': ghg_emissions,
            'Total_Cost_Million_Dollar': total_cost
        }

        results.append(indicators)

# 创建结果 DataFrame
df_indicators = pd.DataFrame(results)

# 保存
df_indicators.to_csv('../2_processed_data/cattle_all_indicators.csv', index=False)

print("计算完成！")
print("\n计算逻辑验证 (以第一行数据为例):")
example = df_indicators.iloc[0]
print(f"原始牛数量: {example['Original_Cattle_Number']: ,.0f} head")
print(f"校正后数量: {example['Adjusted_Cattle_Number']: ,.2f} head (×0.8)")
print(f"总活重: {example['Total_Liveweight_kg']: ,.2f} kg (校正数量 × {cattle_weight})")
print(f"活畜出口重量: {example['Live_Export_Weight_kg']:,.2f} kg (总活重 × {export_live_percent})")
print(f"屠宰重量:  {example['Slaughter_Weight_kg']:,.2f} kg (总活重 × {1 - export_live_percent})")
print(f"零售肉重量: {example['Retail_Meat_kg']:,.2f} kg (屠宰重量 × {retail_meat})")
print(f"温室气体:  {example['GHG_Emissions_tonnes']:,.2f} tonnes CO₂-e")
print(f"总成本:  ${example['Total_Cost_Million_Dollar']:,.2f} million")

print("\n各情景2050年指标预览：")
df_2050 = df_indicators[df_indicators['Year'] == 2050]
for scenario in scenarios:
    data = df_2050[df_2050['Scenario'] == scenario].iloc[0]
    print(f"\n{scenario} 情景:")
    print(f"  原始牛数量: {data['Original_Cattle_Number']: ,.0f} head")
    print(f"  校正后数量: {data['Adjusted_Cattle_Number']:,.0f} head")
    print(f"  总活重: {data['Total_Liveweight_kg']:,.0f} kg")
    print(f"  活畜出口:  {data['Live_Export_Weight_kg']:,.0f} kg")
    print(f"  零售肉产量: {data['Retail_Meat_kg']:,.0f} kg")
    print(f"  温室气体: {data['GHG_Emissions_tonnes']:,.0f} tonnes CO₂-e")
    print(f"  总成本: ${data['Total_Cost_Million_Dollar']:,.2f} million")

# 设置科研风格
sns.set_theme(style="whitegrid", context="paper")

# 创建 2x2 子图
fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=300)
axes = axes.flatten()

# 定义要画的指标
indicators = [
    ('Live_Export_Weight_kg', 'Live Export Weight (t)', 1e3),
    ('Retail_Meat_kg', 'Retail Meat Production (t)', 1e3),
    ('GHG_Emissions_tonnes', 'GHG Emissions (tonnes CO2-e)', 1),
    ('Total_Cost_Million_Dollar', 'Total Cost (Million $)', 1)
]

scenarios = ['Low', 'Medium', 'High', 'Very_High']

for idx, (indicator, title, scale) in enumerate(indicators):
    ax = axes[idx]

    for scenario in scenarios:
        # 筛选数据
        df_scenario = df_indicators[df_indicators['Scenario'] == scenario]

        # 分割历史和未来数据
        df_history = df_scenario[df_scenario['Year'] <= 2024]
        df_future = df_scenario[df_scenario['Year'] >= 2024]

        # 画历史数据（黑色）- 只画一次
        if scenario == 'Low':
            ax.plot(df_history['Year'], df_history[indicator] / scale,
                    color='black', linewidth=2.5, marker='o',
                    label='Historical', markersize=5)

        # 画未来情景 - 这里要取消缩进！
        markers = {'Low': 'o', 'Medium': 's', 'High': '^', 'Very_High': 'd'}
        ax.plot(df_future['Year'], df_future[indicator] / scale,
                marker=markers[scenario], label=scenario.replace('_', ' '),
                linewidth=2, markersize=5)

    # 2024年分界线 - 这些设置应该在 scenario 循环外面
    ax.axvline(x=2024, color='gray', linestyle='--', alpha=0.5, linewidth=1.5)

    # 千分位分隔符
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{x: ,.0f}'))

    # 设置坐标轴范围
    ax.set_xlim(2010, 2050)

    # 标签
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_xlabel('Year', fontsize=11)
    # ax.set_ylabel('Value', fontsize=11)
    ax.legend(fontsize=9, loc='best')
    ax.grid(alpha=0.3)

# 总标题 - 应该在所有子图循环外面
plt.suptitle('Feedlot Cattle Indicators by Scenario',
             fontsize=15, fontweight='bold', y=0.995)

plt.tight_layout()

# 保存
plt.savefig('cattle_indicators_all.png', dpi=300, bbox_inches='tight')
plt.show()

print("图表已保存！")