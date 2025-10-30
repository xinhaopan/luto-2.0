import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.statespace.structural import UnobservedComponents
import warnings

warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.statespace.structural import UnobservedComponents
import warnings

warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.statespace.structural import UnobservedComponents
import warnings

warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.statespace.structural import UnobservedComponents
import warnings

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.statespace.structural import UnobservedComponents

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.statespace.structural import UnobservedComponents
import warnings

warnings.filterwarnings('ignore')


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from statsmodels.tsa.exponential_smoothing.ets import ETSModel

def predict_growth_index(df, var_name='labour cost', pi_level=0.75,base_year = 2010,model='UnobservedComponents'):
    """
    ETS (additive trend, damped) 作为中心预测；绘制深蓝色整段拟合曲线；
    同时绘制 95% 与 80% 预测区间阴影（95% 更深、更底层），并保留四种情景彩色线。
    返回: ax, df_return(index=Year, cols=['Low','Medium','High','Very_High'])
    """
    # ---------------- 基本设置 ----------------
    draw_base_year = 2010

    end_year = 2050

    # ---- 清洗 & 排序
    df = df[['Year', 'Cost']].dropna().sort_values('Year').reset_index(drop=True)
    df['Year'] = df['Year'].astype(int)

    # 建立带日期索引
    y_ts = pd.Series(
        df['Cost'].values.astype(float),
        index=pd.to_datetime(df['Year'].astype(str) + '-01-01'),
        name='Cost'
    )

    # ---------------- 拟合 ETS（阻尼趋势） ----------------
    if model == 'ETS':
        mod = ETSModel(y_ts, error='add', trend='add', damped_trend=True)
        res = mod.fit(disp=False)
    elif model == 'UnobservedComponents':
        mod = UnobservedComponents(
            y_ts,
            # level='llevel',  # local level
            trend=True,  # 加一个 slope（趋势项）
            stochastic_level=True,  # 水平项随机
            stochastic_trend=True,  # 趋势项随机（关键！要有 sigma2.trend）
            irregular=True  # 显式观测噪声 epsilon_t
            # concentrate_scale=False  # 若想让 scale 非 1.0，更直观可打开
        )
        res = mod.fit(disp=False, maxiter=2000)

        print("UCM params:", res.params)



    # ---------------- 历史末年 ----------------
    last_hist_year = int(df['Year'].max())
    y_last = float(df.loc[df['Year'] == last_hist_year, 'Cost'].values[0])

    # ---------------- 预测（含区间） ----------------
    start_dt = y_ts.index.min()
    end_dt   = pd.to_datetime(str(end_year) + '-01-01')
    pred_80  = res.get_prediction(start=start_dt, end=end_dt)
    pred_95  = res.get_prediction(start=start_dt, end=end_dt)

    def _to_pi_df(pred, level):
        """将 prediction -> DataFrame(Year, Mean, Lower, Upper)；若缺少 PI 列则用均值SE+残差方差合成"""
        alpha = 1 - level
        sf = pred.summary_frame(alpha=alpha)
        mean = sf['mean'].to_numpy()

        if {'pi_lower','pi_upper'}.issubset(sf.columns):
            lower = sf['pi_lower'].to_numpy()
            upper = sf['pi_upper'].to_numpy()
        elif {'obs_ci_lower','obs_ci_upper'}.issubset(sf.columns):
            lower = sf['obs_ci_lower'].to_numpy()
            upper = sf['obs_ci_upper'].to_numpy()
        else:
            z = norm.ppf(1 - alpha/2.0)
            if {'mean_ci_lower','mean_ci_upper'}.issubset(sf.columns):
                mean_se = (sf['mean_ci_upper'] - sf['mean_ci_lower']).to_numpy() / (2.0*z)
            elif 'mean_se' in sf.columns:
                mean_se = sf['mean_se'].to_numpy()
            elif hasattr(pred, 'var_pred_mean'):
                mean_se = np.sqrt(np.asarray(pred.var_pred_mean))
            else:
                mean_se = np.zeros_like(mean)
            sigma2 = float(res.sigma2) if hasattr(res, 'sigma2') else float(res.scale)
            half_w = z * np.sqrt(mean_se**2 + sigma2)
            lower = mean - half_w
            upper = mean + half_w

        out = pd.DataFrame({
            'Year': sf.index.year,
            'Mean': mean,
            'Lower': lower,
            'Upper': upper
        }).drop_duplicates(subset=['Year']).reset_index(drop=True)
        return out

    df80 = _to_pi_df(pred_80, pi_level)  # 主：80%
    df95 = _to_pi_df(pred_95, 0.95)      # 95%

    # Very High：上界再加一段带宽（基于 80% 区间）
    # df80['Very_High'] = df80['Upper'] + (df80['Upper'] - df80['Mean'])
    df80['Very_High'] = df95['Upper']

    # 保留未经对齐的均线（画整段深蓝线）
    df80['Raw_SES_Mean'] = df80['Mean'].copy()

    # ---------------- offset 对齐（用历史末年真实值） ----------------
    pred_mean_last = float(df80.loc[df80['Year'] == last_hist_year, 'Mean'].values[0])
    offset_mean = y_last - pred_mean_last
    offset_mean = 0  # <--- 如需关闭 offset 对齐，请取消注释此行
    for dfp in (df80, df95):
        for col in ['Mean', 'Lower', 'Upper']:
            dfp[col] = dfp[col] + offset_mean
    df80['Very_High'] = df80['Very_High'] + offset_mean

    # ---------------- 标准化（以 base_year 的 Mean 作为 1） ----------------
    if draw_base_year not in df80['Year'].values:
        raise ValueError(f"base_year={draw_base_year} 不在预测范围内，请检查数据年份。")
    base_val = float(df.loc[df['Year'] == draw_base_year, 'Cost'].values[0])

    for col in ['Mean', 'Lower', 'Upper', 'Very_High', 'Raw_SES_Mean']:
        df80[col + '_Index'] = df80[col] / base_val
    for col in ['Mean', 'Lower', 'Upper']:
        df95[col + '_Index'] = df95[col] / base_val

    # 只保留绘图起始年份之后
    df80 = df80[df80['Year'] >= base_year].copy()
    df95 = df95[df95['Year'] >= base_year].copy()

    # 历史数据（指数）
    df_hist = df[df['Year'] >= base_year].copy()
    df_hist['Growth_Index'] = df_hist['Cost'] / base_val
    df_hist['Scenario'] = 'Historical'

    # 场景表（从 last_hist_year 起）
    def make_scenario_df(df_in, col, scenario, start_year=base_year):
        sub = df_in[df_in['Year'] >= start_year][['Year', col]].copy()
        sub.columns = ['Year', 'Growth_Index']
        sub['Scenario'] = scenario
        return sub

    df_medium  = make_scenario_df(df80, 'Mean_Index',       'Medium',    last_hist_year)
    df_low     = make_scenario_df(df80, 'Lower_Index',      'Low',       last_hist_year)
    df_high    = make_scenario_df(df80, 'Upper_Index',      'High',      last_hist_year)
    df_vhigh   = make_scenario_df(df80, 'Very_High_Index',  'Very High', last_hist_year)
    df_raw_ses = make_scenario_df(df80, 'Raw_SES_Mean_Index','Raw_SES_Fit', base_year)
    df_raw_ses = df_raw_ses[df_raw_ses['Year'] <= last_hist_year].copy()

    # ---------------- 绘图 ----------------
    sns.set(style='whitegrid')
    fig, ax = plt.subplots(figsize=(12, 6))

    deep_blue   = '#0b3d91'  # 80% 阴影
    deeper_blue = '#082b6a'  # 95% 阴影（更深）
    color_dict = {
        'Historical': 'black',
        'Raw_SES_Fit': deep_blue,
        'Medium': '#2ca02c',
        'Low': '#ff7f0e',
        'High': '#d62728',
        'Very High': '#9467bd'
    }

    # 历史点
    min_hist_year = int(df_hist['Year'].min())
    if min_hist_year > base_year:
        # 生成需要补齐的年份区间
        missing_years = list(range(base_year, min_hist_year))

        # 取第一个观测年的数值（例如 2014）
        first_row = df_hist.iloc[0]
        first_year = int(first_row['Year'])
        fill_val = float(first_row['Growth_Index'])
        fill_cost = float(first_row['Cost'])

        # 生成补齐的 DataFrame
        df_fill = pd.DataFrame({
            'Year': missing_years,
            'Cost': fill_cost,
            'Growth_Index': fill_val,
            'Scenario': 'Historical'
        })

        # 合并并重新排序
        df_hist = pd.concat([df_fill, df_hist], ignore_index=True)
        df_hist = df_hist.sort_values('Year').reset_index(drop=True)
    ax.scatter(df_hist['Year'], df_hist['Growth_Index'],
               s=40, c=color_dict['Historical'],
               label='Historical (actual data)', zorder=6)

    # 深蓝实线：整段拟合到 2050
    ax.plot(df_raw_ses['Year'], df_raw_ses['Growth_Index'],
            color=color_dict['Raw_SES_Fit'], linewidth=2.8,
            label=f'{model} mean', zorder=5)

    # 先画 95% 阴影，再画 80% 阴影（后者覆盖，浅一些）
    env95 = df95[df95['Year'] >= last_hist_year].copy()
    if not env95.empty:
        ax.fill_between(
            env95['Year'].to_numpy(),
            env95['Lower_Index'].to_numpy(),
            env95['Upper_Index'].to_numpy(),
            color=deeper_blue, alpha=0.12,
            label='Prediction interval (95%)',
            zorder=2
        )

    env80 = df80[df80['Year'] >= last_hist_year].copy()
    if not env80.empty:
        ax.fill_between(
            env80['Year'].to_numpy(),
            env80['Lower_Index'].to_numpy(),
            env80['Upper_Index'].to_numpy(),
            color=deep_blue, alpha=0.18,
            label=f'Prediction interval ({int(pi_level*100)}%)',
            zorder=3
        )
        # （可选）80% 上/下界虚线
        ax.plot(env80['Year'], env80['Lower_Index'], color=deep_blue, linewidth=1.0, alpha=0.6, linestyle='--', zorder=4)
        ax.plot(env80['Year'], env80['Upper_Index'], color=deep_blue, linewidth=1.0, alpha=0.6, linestyle='--', zorder=4)

    # 四种情景彩色线（从 last_hist_year 起）
    for dfi, lab, col, lw in [
        (df_low,    f'Low = {int(pi_level*100)}% PI lower',  color_dict['Low'],  1.6),
        (df_medium, 'Medium',                           color_dict['Raw_SES_Fit'], 2.8),
        (df_high,   f'High = {int(pi_level*100)}% PI upper', color_dict['High'], 1.6),
        (df_vhigh,  'Very High = 95% PI upper',            color_dict['Very High'], 1.6),
    ]:
        if not dfi.empty:
            ax.plot(dfi['Year'], dfi['Growth_Index'], color=col, linewidth=lw,linestyle='--', label=lab, zorder=7)

    ax.axhline(1.0, linestyle='dotted', color='gray', linewidth=1, zorder=1)
    ax.set_title(f'{var_name} Growth Index',
                 fontsize=14, weight='bold')
    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('Growth Index', fontsize=12)
    ax.set_xlim(base_year, end_year)
    ax.set_xticks(range(base_year, end_year + 1, 5))
    ax.legend(loc='upper left', frameon=True, fontsize=9, ncol=1)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    # ---------------- 输出表（基于 80% 中/上下界 + Very_High） ----------------
    df_return = df80[['Year', 'Lower_Index', 'Mean_Index',
                      'Upper_Index', 'Very_High_Index']].copy()
    df_return.columns = ['Year', 'Low', 'Medium', 'High', 'Very_High']
    df_return = df_return.set_index('Year').reindex(np.arange(draw_base_year, end_year + 1))

    # 历史真实值覆盖
    hist_map = (df[df['Year'] >= base_year]
                .set_index('Year')['Cost']
                .astype(float) / base_val)
    common_years = hist_map.index.intersection(df_return.index)
    if len(common_years) > 0:
        vals = hist_map.loc[common_years].values.reshape(-1, 1)
        df_return.loc[common_years, ['Low', 'Medium', 'High', 'Very_High']] = vals

    df_return.fillna(1.0, inplace=True)

    return ax, df_return


# def predict_growth_index(df, var_name='labour cost', pi_level=0.75, base_year=2010):
#     """
#     使用 UCM 生成发散的预测区间（使用整数年份索引，不转换为日期）
#     """
#     # ---------------- 基本设置 ----------------
#     draw_base_year = int(df['Year'].iloc[0])
#     end_year = 2050
#
#     # ---- 清洗 & 排序
#     df = df[['Year', 'Cost']].dropna().sort_values('Year').reset_index(drop=True)
#     df['Year'] = df['Year'].astype(int)
#
#     # 直接使用整数年份作为索引（不转换为日期）
#     y_ts = pd.Series(
#         df['Cost'].values.astype(float),
#         index=df['Year'].values,  # 直接使用年份
#         name='Cost'
#     )
#
#     # ---------------- 拟合 UCM ----------------
#     mod = UnobservedComponents(
#         y_ts,
#         level='llevel',
#         trend=True,
#         stochastic_level=True,
#         stochastic_trend=True,
#         irregular=True
#     )
#     res = mod.fit(disp=False, maxiter=2000)
#
#     print("UCM params:", res.params)
#     print("scale(残差方差):", res.scale)
#
#     # ---------------- 历史末年 ----------------
#     last_hist_year = int(df['Year'].max())
#     y_last = float(df.loc[df['Year'] == last_hist_year, 'Cost'].values[0])
#
#     # ---------------- 预测 ----------------
#     n_forecast = end_year - last_hist_year
#
#     # 方法1：使用 forecast 获取点预测
#     forecast_result = res.get_forecast(steps=n_forecast)
#     forecast_mean = forecast_result.predicted_mean
#
#     # 提取方差参数
#     sigma_level = float(res.params.get('sigma2.level', 0))
#     sigma_trend = float(res.params.get('sigma2.trend', 0)) if 'sigma2.trend' in res.params else 0
#     sigma_irreg = float(res.params.get('sigma2.irregular', res.scale))
#
#     print(f"方差参数: level={sigma_level:.2f}, trend={sigma_trend:.6f}, irregular={sigma_irreg:.2f}")
#
#     # 手动构造发散的预测区间
#     # 预测方差随时间增长（状态空间模型理论）
#     forecast_var = np.zeros(n_forecast)
#     for h in range(n_forecast):
#         # 方差累积：每步增加
#         # - 水平方差贡献：线性增长
#         # - 趋势方差贡献：二次增长（因为趋势影响累积）
#         # - 观测噪声：固定
#         forecast_var[h] = (h + 1) * sigma_level + \
#                           ((h + 1) * (h + 2) / 2) * sigma_trend + \
#                           sigma_irreg
#
#     forecast_std = np.sqrt(forecast_var)
#
#     # 计算预测区间
#     from scipy.stats import norm
#     z_80 = norm.ppf(1 - (1 - pi_level) / 2)
#     z_95 = norm.ppf(0.975)
#
#     forecast_years = np.arange(last_hist_year + 1, end_year + 1)
#     lower_80 = forecast_mean.values - z_80 * forecast_std
#     upper_80 = forecast_mean.values + z_80 * forecast_std
#     lower_95 = forecast_mean.values - z_95 * forecast_std
#     upper_95 = forecast_mean.values + z_95 * forecast_std
#
#     print(f"预测区间宽度 (第1年): {upper_80[0] - lower_80[0]:.2f}")
#     print(f"预测区间宽度 (最后一年): {upper_80[-1] - lower_80[-1]:.2f}")
#
#     # 获取历史拟合值
#     fitted_values = res.fittedvalues
#     hist_years = df['Year'].values
#     hist_mean = fitted_values.values
#
#     # 创建预测区间 DataFrame（仅预测部分）
#     df80_forecast = pd.DataFrame({
#         'Year': forecast_years,
#         'Mean': forecast_mean.values,
#         'Lower': lower_80,
#         'Upper': upper_80
#     })
#
#     df95_forecast = pd.DataFrame({
#         'Year': forecast_years,
#         'Mean': forecast_mean.values,
#         'Lower': lower_95,
#         'Upper': upper_95
#     })
#
#     # 添加历史部分（区间等于均值）
#     hist_df = pd.DataFrame({
#         'Year': hist_years,
#         'Mean': hist_mean,
#         'Lower': hist_mean,
#         'Upper': hist_mean
#     })
#
#     df80 = pd.concat([hist_df, df80_forecast], ignore_index=True)
#     df95 = pd.concat([hist_df, df95_forecast], ignore_index=True)
#
#     # Very High：95% 上界
#     df80['Very_High'] = np.concatenate([hist_mean, upper_95])
#     df80['Raw_SES_Mean'] = df80['Mean'].copy()
#
#     # 为 df95 也添加 Very_High
#     df95['Very_High'] = df95['Upper'].copy()
#
#     # ---------------- 标准化（以 base_year 的 Mean 作为 1） ----------------
#     if base_year not in df80['Year'].values:
#         raise ValueError(f"base_year={base_year} 不在预测范围内，请检查数据年份。")
#     base_val = float(df80.loc[df80['Year'] == base_year, 'Mean'].values[0])
#
#     for col in ['Mean', 'Lower', 'Upper', 'Very_High', 'Raw_SES_Mean']:
#         df80[col + '_Index'] = df80[col] / base_val
#     for col in ['Mean', 'Lower', 'Upper']:
#         df95[col + '_Index'] = df95[col] / base_val
#
#     # 只保留绘图起始年份之后
#     df80 = df80[df80['Year'] >= draw_base_year].copy()
#     df95 = df95[df95['Year'] >= draw_base_year].copy()
#
#     # 历史数据（指数）
#     df_hist = df[df['Year'] >= draw_base_year].copy()
#     df_hist['Growth_Index'] = df_hist['Cost'] / base_val
#     df_hist['Scenario'] = 'Historical'
#
#     # 场景表（从 last_hist_year 起）
#     def make_scenario_df(df_in, col, scenario, start_year=draw_base_year):
#         sub = df_in[df_in['Year'] >= start_year][['Year', col]].copy()
#         sub.columns = ['Year', 'Growth_Index']
#         sub['Scenario'] = scenario
#         return sub
#
#     df_medium = make_scenario_df(df80, 'Mean_Index', 'Medium', last_hist_year)
#     df_low = make_scenario_df(df80, 'Lower_Index', 'Low', last_hist_year)
#     df_high = make_scenario_df(df80, 'Upper_Index', 'High', last_hist_year)
#     df_vhigh = make_scenario_df(df80, 'Very_High_Index', 'Very High', last_hist_year)
#     df_raw_ses = make_scenario_df(df80, 'Raw_SES_Mean_Index', 'Raw_UCM_Fit', draw_base_year)
#     df_raw_ses = df_raw_ses[df_raw_ses['Year'] <= last_hist_year].copy()
#
#     # ---------------- 绘图 ----------------
#     sns.set(style='whitegrid')
#     fig, ax = plt.subplots(figsize=(12, 6))
#
#     deep_blue = '#0b3d91'
#     deeper_blue = '#082b6a'
#     color_dict = {
#         'Historical': 'black',
#         'Raw_UCM_Fit': deep_blue,
#         'Medium': '#2ca02c',
#         'Low': '#ff7f0e',
#         'High': '#d62728',
#         'Very High': '#9467bd'
#     }
#
#     # 历史点
#     ax.scatter(df_hist['Year'], df_hist['Growth_Index'],
#                s=40, c=color_dict['Historical'],
#                label='Historical (actual data)', zorder=6)
#
#     # 拟合线（历史部分）
#     ax.plot(df_raw_ses['Year'], df_raw_ses['Growth_Index'],
#             color=color_dict['Raw_UCM_Fit'], linewidth=2.8,
#             label='UCM fitted mean', zorder=5)
#
#     # 95% 阴影
#     env95 = df95[df95['Year'] >= last_hist_year].copy()
#     if not env95.empty:
#         ax.fill_between(
#             env95['Year'].to_numpy(),
#             env95['Lower_Index'].to_numpy(),
#             env95['Upper_Index'].to_numpy(),
#             color=deeper_blue, alpha=0.12,
#             label='Prediction interval (95%)',
#             zorder=2
#         )
#
#     # 80% 阴影
#     env80 = df80[df80['Year'] >= last_hist_year].copy()
#     if not env80.empty:
#         ax.fill_between(
#             env80['Year'].to_numpy(),
#             env80['Lower_Index'].to_numpy(),
#             env80['Upper_Index'].to_numpy(),
#             color=deep_blue, alpha=0.18,
#             label=f'Prediction interval ({int(pi_level * 100)}%)',
#             zorder=3
#         )
#         ax.plot(env80['Year'], env80['Lower_Index'], color=deep_blue,
#                 linewidth=1.0, alpha=0.6, linestyle='--', zorder=4)
#         ax.plot(env80['Year'], env80['Upper_Index'], color=deep_blue,
#                 linewidth=1.0, alpha=0.6, linestyle='--', zorder=4)
#
#     # 四种情景线
#     for dfi, lab, col, lw in [
#         (df_low, f'Low = {int(pi_level * 100)}% PI lower', color_dict['Low'], 1.6),
#         (df_medium, 'Medium (UCM forecast)', color_dict['Medium'], 2.8),
#         (df_high, f'High = {int(pi_level * 100)}% PI upper', color_dict['High'], 1.6),
#         (df_vhigh, 'Very High = 95% PI upper', color_dict['Very High'], 1.6),
#     ]:
#         if not dfi.empty:
#             ax.plot(dfi['Year'], dfi['Growth_Index'], color=col,
#                     linewidth=lw, linestyle='--', label=lab, zorder=7)
#
#     ax.axhline(1.0, linestyle='dotted', color='gray', linewidth=1, zorder=1)
#     ax.set_title(f'{var_name} Growth Index (Base Year = {base_year})',
#                  fontsize=14, weight='bold')
#     ax.set_xlabel('Year', fontsize=12)
#     ax.set_ylabel('Growth Index', fontsize=12)
#     ax.set_xlim(draw_base_year, end_year)
#     ax.set_xticks(range(base_year, end_year + 1, 5))
#     ax.legend(loc='upper left', frameon=True, fontsize=9, ncol=1)
#     ax.grid(True, alpha=0.3)
#     plt.tight_layout()
#
#     # ---------------- 输出表 ----------------
#     df_return = df80[['Year', 'Lower_Index', 'Mean_Index',
#                       'Upper_Index', 'Very_High_Index']].copy()
#     df_return.columns = ['Year', 'Low', 'Medium', 'High', 'Very_High']
#     df_return = df_return.set_index('Year').reindex(np.arange(base_year, end_year + 1))
#
#     # 历史真实值覆盖
#     hist_map = (df[df['Year'] >= base_year]
#                 .set_index('Year')['Cost']
#                 .astype(float) / base_val)
#     common_years = hist_map.index.intersection(df_return.index)
#     if len(common_years) > 0:
#         vals = hist_map.loc[common_years].values.reshape(-1, 1)
#         df_return.loc[common_years, ['Low', 'Medium', 'High', 'Very_High']] = vals
#
#     df_return.fillna(1.0, inplace=True)
#
#     return ax, df_return