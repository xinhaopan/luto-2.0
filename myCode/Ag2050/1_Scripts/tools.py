import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns

def predit_growth_index(df, var_name='labour cost'):
    draw_base_year = df['Year'][0]
    base_year = 2010
    end_year = 2050

    # 清洗 & 排序
    df = df[['Year', 'Cost']].dropna().sort_values('Year').reset_index(drop=True)
    df['Year'] = df['Year'].astype(int)

    # ============ 模型训练 ============
    df_model = df.copy()
    X = sm.add_constant(df_model['Year'])
    y = df_model['Cost']
    model = sm.OLS(y.values.astype(float), X.values.astype(float)).fit()

    # ============ 模型预测 ============
    years_pred = np.arange(df['Year'].min(), end_year + 1)
    X_pred = sm.add_constant(pd.Series(years_pred, name='Year'))
    summary = model.get_prediction(X_pred).summary_frame(alpha=0.05)

    df_pred = pd.DataFrame({
        'Year': years_pred,
        'Mean': summary['mean'],
        'CI_Lower': summary['mean_ci_lower'],
        'CI_Upper': summary['mean_ci_upper'],
    })
    df_pred['Very_High'] = df_pred['CI_Upper'] + (df_pred['CI_Upper'] - df_pred['Mean'])

    # ============ 标准化为指数 ============
    base_val = df_pred.loc[df_pred['Year'] == base_year, 'Mean'].values[0]
    for col in ['Mean', 'CI_Lower', 'CI_Upper', 'Very_High']:
        df_pred[col + '_Index'] = df_pred[col] / base_val

    # ============ 只保留 base_year 之后的年份 ============
    df_pred = df_pred[df_pred['Year'] >= draw_base_year].copy()

    # ============ 历史数据 ============
    df_hist = df_model[df_model['Year'] >= draw_base_year].copy()
    df_hist['Growth_Index'] = df_hist['Cost'] / df_model.loc[df_model['Year'] == base_year, 'Cost'].values[0]
    df_hist['Scenario'] = 'Historical'

    last_hist_year = df_hist['Year'].max()

    # ============ 构造各个情景 DataFrame ============
    def make_scenario_df(df, col, scenario, start_year=draw_base_year):
        sub = df[df['Year'] >= start_year][['Year', col]].copy()
        sub.columns = ['Year', 'Growth_Index']
        sub['Scenario'] = scenario
        return sub

    df_medium = make_scenario_df(df_pred, 'Mean_Index', 'Medium')
    df_low = make_scenario_df(df_pred, 'CI_Lower_Index', 'Low', last_hist_year + 1)
    df_high = make_scenario_df(df_pred, 'CI_Upper_Index', 'High', last_hist_year + 1)
    df_vhigh = make_scenario_df(df_pred, 'Very_High_Index', 'Very High', last_hist_year + 1)

    # ============ 合并绘图数据 ============
    plot_base = pd.concat([df_hist, df_medium], ignore_index=True)
    plot_extra = pd.concat([df_low, df_high, df_vhigh], ignore_index=True)

    # ============ 绘图 ============
    sns.set(style='whitegrid')
    fig, ax = plt.subplots(figsize=(12, 6))
    color_dict = {
        'Historical': 'black',
        'Medium': '#2ca02c',
        'Low': '#ff7f0e',
        'High': '#d62728',
        'Very High': '#9467bd'
    }
    legend_labels = {
        'Historical': 'Historical (points only)',
        'Medium': 'Medium = mean (OLS prediction)',
        'Low': 'Low = mean_95%ci_lower',
        'High': 'High = mean_95%ci_upper',
        'Very High': 'Very High = 95%CI_upper + (95%CI_upper − mean)'
    }
    legend_order = list(legend_labels.keys())

    sns.scatterplot(data=plot_base[plot_base['Scenario'] == 'Historical'],
                    x='Year', y='Growth_Index', color=color_dict['Historical'], s=40,
                    label=legend_labels['Historical'], ax=ax)

    sns.lineplot(data=plot_base[plot_base['Scenario'] == 'Medium'],
                 x='Year', y='Growth_Index', color=color_dict['Medium'],
                 linewidth=2, label=legend_labels['Medium'], ax=ax)

    for scen in ['Low', 'High', 'Very High']:
        df_scen = plot_extra[plot_extra['Scenario'] == scen]
        if not df_scen.empty:
            sns.lineplot(data=df_scen, x='Year', y='Growth_Index',
                         color=color_dict[scen], linewidth=1.5,
                         label=legend_labels[scen], ax=ax)

    ax.axhline(1.0, linestyle='dotted', color='gray', linewidth=1)
    ax.set_title(f'{var_name} Growth Index (Base Year = {base_year})', fontsize=14, weight='bold')
    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('Growth Index', fontsize=12)
    ax.set_xlim(draw_base_year, end_year)
    ax.set_xticks(range(base_year, end_year + 1, 5))

    # 图例顺序
    handles, labels = ax.get_legend_handles_labels()
    ordered_handles = [handles[legend_order.index(k)] for k in legend_order if k in labels]
    ordered_labels = [legend_labels[k] for k in legend_order if k in labels]
    ax.legend(ordered_handles, ordered_labels, loc='upper left', frameon=True)

    # ============ 输出表格 ============
    df_return = df_pred[['Year', 'CI_Lower_Index', 'Mean_Index', 'CI_Upper_Index', 'Very_High_Index']].copy()
    df_return.columns = ['Year', 'Low', 'Medium', 'High', 'Very_High']
    df_return = df_return.set_index('Year').reindex(np.arange(base_year, end_year + 1))
    df_return.fillna(1.0, inplace=True)

    # 用历史真实值替代预测
    hist_map = df_hist.set_index('Year')['Growth_Index']
    for y in hist_map.index.intersection(df_return.index):
        for col in df_return.columns:
            df_return.loc[y, col] = hist_map.loc[y]

    return ax, df_return