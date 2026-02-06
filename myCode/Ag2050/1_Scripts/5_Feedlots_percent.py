import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.exponential_smoothing.ets import ETSModel
import warnings

warnings.filterwarnings('ignore')

# 设置绘图风格
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 1. 准备数据
years = [2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014,
         2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024]

percent = [2.993448832, 3.224167607, 2.697157709, 2.427415254, 2.638308551,
           2.83266584, 2.703190004, 2.704179349, 2.821993961, 3.117485935,
           3.572668318, 3.339503754, 3.668430112, 3.891641129, 4.363652,
           4.1007007, 4.230847465, 4.388109456, 4.464715645, 5.004877969]

# 创建DataFrame
df = pd.DataFrame({'year': years, 'percent': percent})
df.set_index('year', inplace=True)

print("原始数据:")
print(df)

# 2. 建立ETS模型
model = ETSModel(df['percent'],
                 error='add',
                 trend='add',
                 seasonal=None,
                 damped_trend=True)

fitted_model = model.fit()

print("\n模型摘要:")
print(fitted_model.summary())

# 3. 预测到2050年
forecast_years = 2050 - 2024
future_years = list(range(2025, 2051))

# 获取点预测和标准误差
forecast = fitted_model.forecast(steps=forecast_years)
forecast_df = fitted_model.get_prediction(start=len(df), end=len(df) + forecast_years - 1)

# 4. 创建四个场景
forecast_summary_50 = forecast_df.summary_frame(alpha=0.50)
forecast_summary_80 = forecast_df.summary_frame(alpha=0.20)
forecast_summary_95 = forecast_df.summary_frame(alpha=0.05)

# 定义四个预测场景
scenarios = {
    'Low': forecast_summary_95['pi_lower'].values,
    'Medium': forecast.values,
    'High': forecast_summary_80['pi_upper'].values,
    'Very High': forecast_summary_95['pi_upper'].values
}

# 创建预测DataFrame
prediction_df = pd.DataFrame({
    'Year': future_years,
    'Low': scenarios['Low'],
    'Medium': scenarios['Medium'],
    'High': scenarios['High'],
    'Very_High': scenarios['Very High']
})

# ========== 新增：合并历史数据和预测数据 ==========
# 创建历史数据DataFrame（所有场景列值相同）
historical_df = pd.DataFrame({
    'Year': years,
    'Low': percent,
    'Medium': percent,
    'High': percent,
    'Very_High': percent
})

# 合并历史和预测数据
complete_df = pd.concat([historical_df, prediction_df], ignore_index=True)

print("\n完整数据（历史+预测）前10行:")
print(complete_df.head(10))

print("\n完整数据（历史+预测）后10行:")
print(complete_df.tail(10))

# 5. 绘图
fig, ax = plt.subplots(figsize=(14, 8))

ax.set_facecolor('#E8E8F0')
fig.patch.set_facecolor('white')

# 绘制历史数据
ax.plot(df.index, df['percent'],
        'o-', color='black', linewidth=2.5,
        markersize=7, label='Historical',
        markerfacecolor='black', zorder=5)

# 连接点
last_year = df.index[-1]
last_value = df['percent'].iloc[-1]

# 场景颜色
scenario_colors = {
    'Low': '#4477AA',
    'Medium': '#EE8833',
    'High': '#228833',
    'Very High': '#CC3333'
}

# 绘制预测场景
for scenario_name, col_name in [('Low', 'Low'), ('Medium', 'Medium'),
                                  ('High', 'High'), ('Very High', 'Very_High')]:
    full_years = [last_year] + future_years
    full_values = [last_value] + list(prediction_df[col_name])

    ax.plot(full_years, full_values,
            'o-', color=scenario_colors[scenario_name],
            linewidth=2.5, markersize=6,
            label=scenario_name,
            markerfacecolor=scenario_colors[scenario_name],
            markeredgewidth=0.5,
            markeredgecolor='white',
            zorder=4)

# 图表设置
ax.set_xlabel('Year', fontsize=13, fontweight='bold')
ax.set_ylabel('Percent (%)', fontsize=13, fontweight='bold')
ax.set_title('Feedlots Cattle Percent Forecast',
             fontsize=15, fontweight='bold', pad=20)

ax.set_xlim(2005, 2050)
ax.set_xticks(range(2005, 2055, 5))

ax.grid(True, alpha=0.4, linestyle='-', linewidth=0.5, color='white')
ax.set_axisbelow(True)

ax.legend(loc='upper left', fontsize=11, framealpha=0.9,
          edgecolor='gray', fancybox=True)

plt.tight_layout()
plt.savefig('../2_processed_data/ets_forecast_scenarios.png', dpi=300, bbox_inches='tight')
print("\n图表已保存为 '../2_processed_data/ets_forecast_scenarios.png'")
plt.show()

# 6. 输出关键年份对比
print("\n关键年份预测对比:")
print("-" * 80)
print(f"{'Year':<8} {'Low':<12} {'Medium':<12} {'High':<12} {'Very High':<12}")
print("-" * 80)

key_years = [2005, 2010, 2015, 2020, 2024, 2025, 2030, 2035, 2040, 2045, 2050]
for year in key_years:
    row = complete_df[complete_df['Year'] == year]
    if not row.empty:
        print(f"{year:<8} {row['Low'].values[0]:<12.4f} "
              f"{row['Medium'].values[0]:<12.4f} "
              f"{row['High'].values[0]:<12.4f} "
              f"{row['Very_High'].values[0]:<12.4f}")

# 7. 保存完整结果（历史+预测）到CSV
complete_df.to_csv('../2_processed_data/cattle_percent_scenarios.csv', index=False)
print("\n完整数据（历史+预测）已保存为 '../2_processed_data/cattle_percent_scenarios.csv'")

# 8. 2050年预测场景
print("\n2050年预测场景:")
print("-" * 50)
for scenario, col in [('Low', 'Low'), ('Medium', 'Medium'),
                       ('High', 'High'), ('Very High', 'Very_High')]:
    value_2050 = complete_df[complete_df['Year'] == 2050][col].values[0]
    print(f"{scenario:<12}: {value_2050:.4f}%")

# 9. 显示历史数据统计
print("\n历史数据统计 (2005-2024):")
print("-" * 50)
hist_data = complete_df[complete_df['Year'] <= 2024]['Low']
print(f"平均值: {hist_data.mean():.4f}%")
print(f"标准差: {hist_data.std():.4f}%")
print(f"最小值: {hist_data.min():.4f}% (年份: {complete_df[complete_df['Low'] == hist_data.min()]['Year'].values[0]})")
print(f"最大值: {hist_data.max():.4f}% (年份: {complete_df[complete_df['Low'] == hist_data.max()]['Year'].values[0]})")