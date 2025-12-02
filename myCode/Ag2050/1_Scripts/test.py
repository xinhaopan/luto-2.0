import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression
import warnings
import pandas as pd
import chardet
import re

warnings.filterwarnings('ignore')

def read_csv_auto(filepath, **kwargs):
    """自动检测编码并读取CSV"""
    with open(filepath, 'rb') as f:
        result = chardet.detect(f.read(100000))
        encoding = result['encoding']

    print(f"Detected: {encoding} (confidence: {result['confidence']:.2%})")

    try:
        return pd.read_csv(filepath, encoding=encoding, **kwargs)
    except:
        return pd.read_csv(filepath, encoding='latin1', **kwargs)


def forecast_data(df, group_col, year_col, value_cols, target_years, method='linear'):
    """
    通用数据预测填充函数

    参数:
    - df: 原始数据框
    - group_col: 分组列名（如国家代码）
    - year_col: 年份列名
    - value_cols: 需要预测的列名列表
    - target_years: 目标年份列表
    - method: 预测方法 'linear' 或 'exponential'
    """
    result_list = []

    for group_val in df[group_col].unique():
        group_data = df[df[group_col] == group_val].copy()
        existing_years = set(group_data[year_col].unique())
        missing_years = [y for y in target_years if y not in existing_years]

        if not missing_years:
            continue

        forecast_dict = {year_col: missing_years, group_col: group_val}

        for col in value_cols:
            clean_data = group_data.dropna(subset=[col])

            if len(clean_data) < 3:
                forecast_dict[col] = [None] * len(missing_years)
                continue

            # 使用最近10年数据
            recent = clean_data[clean_data[year_col] >= (clean_data[year_col].max() - 10)]
            if len(recent) >= 3:
                clean_data = recent

            X = clean_data[year_col].values.reshape(-1, 1)
            y = clean_data[col].values

            # 指数增长（适合人口）
            if method == 'exponential':
                y = np.log(np.maximum(y, 1))

            model = LinearRegression()
            model.fit(X, y)

            X_future = np.array(missing_years).reshape(-1, 1)
            y_pred = model.predict(X_future)

            if method == 'exponential':
                y_pred = np.exp(y_pred)

            # 根据列类型设置约束
            if 'population' in col.lower() or 'gdp' in col.lower():
                y_pred = np.maximum(y_pred, 0)
            elif 'pct' in col.lower() or 'urban' in col.lower():
                y_pred = np.clip(y_pred, 0, 100)

            forecast_dict[col] = y_pred

        result_list.append(pd.DataFrame(forecast_dict))

    if result_list:
        forecast_df = pd.concat(result_list, ignore_index=True)
        return pd.concat([df, forecast_df], ignore_index=True)
    return df
# ---------------------------
# 创建2010-2050年预测表（带数据预测功能）
# ---------------------------
print("创建2010-2050年预测表...")

# ---- 1) 从 tradeF 获取所有唯一的贸易对组合 ----
tradeF = pd.read_csv("../2_processed_data/trade_model_data_all.csv", low_memory=False)
trade_pairs = tradeF[['group', 'Report ISO', 'Partner ISO', 'country', 'Element']].drop_duplicates()

SSP2_PATH = "../0_original_data/data.ssp2.csv"
GDPPC_PATH = "../0_original_data/gdp_pc_ppp_2005_wdi.csv"
GRAVITY_PATH = "../0_original_data/dynamic gravity data AU od do.csv"
ssp2 = read_csv_auto(SSP2_PATH, low_memory=False)
gdp_pc = read_csv_auto(GDPPC_PATH, low_memory=False)
gravity = read_csv_auto(GRAVITY_PATH, low_memory=False)
print(f"唯一的贸易对数量: {len(trade_pairs)}")


future_years = list(range(2010, 2051))

trade_pairs['_key'] = 1
years_df = pd.DataFrame({'year': future_years, '_key': 1})
future_trade = trade_pairs.merge(years_df, on='_key').drop('_key', axis=1)
future_trade['year'] = future_trade['year'].astype(int)

print(f"future_trade: {future_trade.shape[0]} 行")

# 2) 准备并预测 GDP
gdp_year_cols = [c for c in gdp_pc.columns if re.match(r'^\d{4}$', str(c))]
gdp_pc_long = gdp_pc.melt(
    id_vars=['Country Name', 'Country Code'],
    value_vars=gdp_year_cols,
    var_name='year',
    value_name='gdp_pc'
)
gdp_pc_long['year'] = pd.to_numeric(gdp_pc_long['year'], errors='coerce').astype(int)
gdp_pc_long['gdp_pc'] = pd.to_numeric(gdp_pc_long['gdp_pc'], errors='coerce')

gdp_pc_full = forecast_data(
    gdp_pc_long,
    group_col='Country Code',
    year_col='year',
    value_cols=['gdp_pc'],
    target_years=future_years,
    method='linear'
)
gdp_pc_full = gdp_pc_full[(gdp_pc_full['year'] >= 2010) & (gdp_pc_full['year'] <= 2050)]
print(f"GDP: {gdp_pc_full.shape[0]} 行")

# 3) 准备并预测 SSP2
ssp2['country.code'] = ssp2['country.code'].astype(str).str.strip().str.upper()
ssp2['year'] = pd.to_numeric(ssp2['year'], errors='coerce').astype(int)
ssp2['Population.WB'] = pd.to_numeric(ssp2['Population.WB'], errors='coerce')
ssp2['Urban.population.pct.WB'] = pd.to_numeric(ssp2['Urban.population.pct.WB'], errors='coerce')

ssp2_full = forecast_data(
    ssp2,
    group_col='country.code',
    year_col='year',
    value_cols=['Population.WB', 'Urban.population.pct.WB'],
    target_years=future_years,
    method='exponential'  # 人口用指数增长
)
ssp2_full = ssp2_full[(ssp2_full['year'] >= 2010) & (ssp2_full['year'] <= 2050)]
print(f"SSP2: {ssp2_full.shape[0]} 行")

# 4) 准备 Gravity（距离不变）
gravity_au = gravity[['iso3_o', 'iso3_d', 'distance', 'lat_d', 'lng_d']].drop_duplicates()
gravity_au['_key'] = 1
gravity_au_full = gravity_au.merge(years_df, on='_key').drop('_key', axis=1)
print(f"Gravity: {gravity_au_full.shape[0]} 行")

# 5) 合并所有数据
merged = (future_trade
    .merge(gdp_pc_full[['Country Code', 'year', 'gdp_pc']],
           left_on=['Partner ISO', 'year'], right_on=['Country Code', 'year'], how='left')
    .merge(gravity_au_full[['iso3_o', 'iso3_d', 'year', 'distance', 'lat_d', 'lng_d']],
           left_on=['Report ISO', 'Partner ISO', 'year'], right_on=['iso3_o', 'iso3_d', 'year'], how='left')
    .merge(ssp2_full[['country.code', 'year', 'Population.WB', 'Urban.population.pct.WB']],
           left_on=['Partner ISO', 'year'], right_on=['country.code', 'year'], how='left')
    .drop(columns=['Country Code', 'iso3_o', 'iso3_d', 'country.code'])
)

print(f"\n最终数据: {merged.shape[0]} 行")
print(f"缺失值:\n{merged[['gdp_pc', 'Population.WB', 'Urban.population.pct.WB', 'distance']].isnull().sum()}")

# 6) 保存
df_complete = merged.dropna(subset=['gdp_pc', 'Population.WB', 'Urban.population.pct.WB', 'distance'])
df_complete.to_csv('../2_processed_data/future_trade_2010_2050.csv', index=False)
