import pandas as pd
import chardet
import re
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import pickle
import chardet
import warnings
warnings.filterwarnings("ignore")


# ---------------------------
# 辅助函数
# ---------------------------
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

def detect_year_columns(df):
    """自动识别像 '1990' 或 'X1990' 的年份列名"""
    year_cols = [c for c in df.columns if re.match(r'^X?\d{4}$', str(c))]
    return year_cols

def melt_gdp_pc(gdp_pc_df):
    """把 gdp_pc 宽表转成长表，返回列 ['Country Code','year','gdp_pc']"""
    # 识别年份列
    year_cols = detect_year_columns(gdp_pc_df)
    id_cols = [c for c in gdp_pc_df.columns if c not in year_cols]
    # melt
    long = gdp_pc_df.melt(id_vars=id_cols, value_vars=year_cols, var_name="year", value_name="gdp_pc")
    # 清理 year
    long["year"] = long["year"].astype(str).str.replace("^X", "", regex=True)
    long["year"] = pd.to_numeric(long["year"], errors="coerce").astype("Int64")
    # 强制数值
    long["gdp_pc"] = pd.to_numeric(long["gdp_pc"], errors="coerce")
    # 规范列名，确保有 Country Code 列
    possible_country_cols = [c for c in id_cols if re.search(r'country.*code', c, flags=re.I)]
    if possible_country_cols:
        # 保留第一个匹配到的
        if possible_country_cols[0] != "Country Code":
            long = long.rename(columns={possible_country_cols[0]: "Country Code"})
    return long


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


# =============================================================================
# 加载和合并数据
# =============================================================================
print("\n=== Loading data ===")

tradeF = read_csv_auto("../0_original_data/Trade_DetailedTradeMatrix_E_All.csv")
commodity_map = read_csv_auto("../0_original_data/FAOSTAT_data_groups_factors_cleaned.csv")

TRADEF_PATH = "../2_processed_data/tradeF_final.csv"
SSP2_PATH = "../0_original_data/data.ssp2.csv"
GDPPC_PATH = "../0_original_data/gdp_pc_ppp_2005_wdi.csv"
GRAVITY_PATH = "../0_original_data/dynamic gravity data AU od do.csv"

OUT_CSV = "../2_processed_data/trade_model_data_AUS_all.csv"


# Check if Factor exists
if 'Factor' not in commodity_map.columns:
    print("WARNING: Factor column not found, adding default Factor=1")
    commodity_map['Factor'] = 1

tradeF = tradeF.merge(
    commodity_map[['Item Code', 'Item', 'LUTO', 'Factor']],
    on=['Item Code', 'Item'],
    how='left'
)

print(f"✓ Trade data shape after merge: {tradeF.shape}")
print(f"  Rows with LUTO=NA: {tradeF['LUTO'].isna().sum()}")

# =============================================================================
# 转换为2005年实际价格
# =============================================================================
print("\n=== Converting to real 2005 values ===")

deflator = read_csv_auto("../0_original_data/US GDP deflator.csv")
year_cols = [f"{year}" for year in range(1986, 2017)]

for year_col in year_cols:
    if year_col in tradeF.columns:
        year = int(year_col.replace('X', ''))
        deflator_value = deflator.loc[deflator['Year'] == year, 'GDP_def_2005'].values

        if len(deflator_value) > 0:
            mask = tradeF['Unit'] == "1000 US$"
            tradeF.loc[mask, year_col] = tradeF.loc[mask, year_col] / deflator_value[0]

print("✓ Converted to real 2005 values")

# =============================================================================
# 添加ISO代码
# =============================================================================
print("\n=== Adding ISO codes ===")

iso = read_csv_auto("../0_original_data/FAOSTAT_data_10-15-2020_ISO.csv")
iso['Country'] = iso['Country'].replace(
    "United Kingdom of Great Britain and Northern Ireland",
    "United Kingdom"
)
tradeF['Reporter Country Code'] = tradeF['Reporter Country Code'].astype(str)
tradeF['Partner Country Code'] = tradeF['Partner Country Code'].astype(str)
iso['Country.Code'] = iso['Country.Code'].astype(str)

iso_map = iso.set_index('Country.Code')['ISO3 Code']
tradeF['Report ISO'] = tradeF['Reporter Country Code'].map(iso_map)
tradeF['Partner ISO'] = tradeF['Partner Country Code'].map(iso_map)
print("✓ Added ISO codes")


# 第二步：过滤掉不需要的数据
tradeF_filtered = tradeF[
    (tradeF['Partner Countries'] != 'Australia') &      # ✅ 排除对自己的贸易
    (tradeF['Partner Countries'] != 'Antarctica') &     # 排除南极洲
    (~tradeF['Element'].isin(['Export Value', 'Import Value']))
].copy()

# =============================================================================
# 定义商品组（排除特定组和Unknown）
# =============================================================================
print("\n=== Defining commodity groups ===")

exclude_groups = [
    "Animals, live, non-food",
    "Animals live nes (1000 heads)",
    "Unknown",  # 排除 NA 转换的 Unknown
]


# 导出最终用于建模的数据（过滤后，不含 Unknown）
tradeF_final = tradeF_filtered[
    tradeF_filtered['LUTO'].notna() &
    (~tradeF_filtered['LUTO'].isin(exclude_groups))
    ].copy()




tradeF_final.to_excel("../2_processed_data/tradeF_final.xlsx", index=False)

# ---------------------------
# Step 1: 读取数据
# ---------------------------
print("1) 读取输入数据...")
tradeF = tradeF_final
ssp2 = read_csv_auto(SSP2_PATH, low_memory=False)
gdp_pc = read_csv_auto(GDPPC_PATH, low_memory=False)
gravity = read_csv_auto(GRAVITY_PATH, low_memory=False)

# ---- 1) Extract exports for ALL groups and long-format trade table ----
# year columns in tradeF are numeric strings like '1986'..'2016'
trade_year_cols = [c for c in tradeF.columns if re.match(r'^\d{4}$', str(c))]
# If you only want 1986-2016, filter:
trade_year_cols = [c for c in trade_year_cols if 1986 <= int(c) <= 2016]

trades_long = tradeF.melt(
    id_vars=['LUTO', 'Report ISO', 'Partner ISO', 'Factor',  'Reporter Countries','Element'],
    value_vars=trade_year_cols,
    var_name='year',
    value_name='value'
)
trades_long['trade'] = trades_long['value'] * trades_long['Factor']

# aggregate: group (LUTO), ISO3 Code, year
trade_data = (
    trades_long
    .groupby(['LUTO', 'Report ISO', 'Partner ISO', 'year','Element'], as_index=False)['trade']
    .sum()
    .rename(columns={'LUTO': 'group', 'ISO3 Code': 'ISO'})
)

# restrict to 1990-2014 for model building (same window as your R code)
trade_data['year'] = pd.to_numeric(trade_data['year'], errors='coerce').astype(int)
trade_data = trade_data[(trade_data['year'] >= 1990) & (trade_data['year'] <= 2014)].copy()

# 步骤1: 计算包含所有目的地的总量（含Unspecified Area）
print("\n步骤1: 计算总量（包括未指定目的地）...")

# 使用原始tradeF_aus（在应用Partner ISO过滤之前）
trade_year_cols_all = [c for c in tradeF.columns if re.match(r'^\d{4}$', str(c))]
trade_year_cols_all = [c for c in trade_year_cols_all if 1986 <= int(c) <= 2016]

# 转换为长格式（包含所有伙伴国，含Unspecified Area）
trades_all_long = tradeF.melt(
    id_vars=['LUTO', 'Factor', 'Element', 'Partner Countries'],
    value_vars=trade_year_cols_all,
    var_name='year',
    value_name='value'
)
trades_all_long['trade'] = trades_all_long['value'] * trades_all_long['Factor']
trades_all_long['year'] = pd.to_numeric(trades_all_long['year'], errors='coerce').astype(int)

# 排除对Australia自己的贸易，但保留Unspecified Area
trades_all_long = trades_all_long[
    trades_all_long['Partner Countries'] != 'Australia'
    ].copy()

# 计算总量（1990-2014）
total_exports = (
    trades_all_long[
        (trades_all_long['Element'] == 'Export Quantity') &
        (trades_all_long['year'] >= 1990) &
        (trades_all_long['year'] <= 2014)
        ]
    .groupby(['LUTO', 'year'], as_index=False)['trade']
    .sum()
    .rename(columns={'trade': 'total_exports'})
)

print(f"  ✓ 总量（含未指定地区）: {len(total_exports):,} 行")
if not total_exports.empty:
    total_2010 = total_exports[total_exports['year'] == 2010]['total_exports'].sum()
    print(f"  ✓ 2010年总出口: {total_2010:,.0f}")

# 步骤2: 计算子总量（只有明确ISO的）
print("\n步骤2: 计算子总量（只有明确ISO）...")

subtotal_exports = (
    trade_data[trade_data['Element'] == 'Export Quantity']
    .groupby(['group', 'year'], as_index=False)['trade']
    .sum()
    .rename(columns={'trade': 'subtotal_exports', 'group': 'LUTO'})
)

print(f"  ✓ 子总量（有明确ISO）: {len(subtotal_exports):,} 行")
if not subtotal_exports.empty:
    subtotal_2010 = subtotal_exports[subtotal_exports['year'] == 2010]['subtotal_exports'].sum()
    print(f"  ✓ 2010年子总出口: {subtotal_2010:,.0f}")
    print(f"  ✓ 2010年差异: {total_2010 - subtotal_2010:,.0f} ({(total_2010 - subtotal_2010) / total_2010 * 100:.1f}%)")

# 步骤3: 计算缩放因子
print("\n步骤3: 计算缩放因子...")

scale_factors = total_exports.merge(
    subtotal_exports,
    on=['LUTO', 'year'],
    how='left'
)

scale_factors['scale_factor'] = (
        scale_factors['total_exports'] / scale_factors['subtotal_exports']
)

# 处理特殊情况
scale_factors['scale_factor'] = scale_factors['scale_factor'].replace([np.inf, -np.inf], 1)
scale_factors['scale_factor'] = scale_factors['scale_factor'].fillna(1)

print(f"\n  缩放因子统计:")
print(f"    平均值: {scale_factors['scale_factor'].mean():.4f}")
print(f"    中位数: {scale_factors['scale_factor'].median():.4f}")
print(f"    最小值: {scale_factors['scale_factor'].min():.4f}")
print(f"    最大值: {scale_factors['scale_factor'].max():.4f}")

# 显示2010年未指定地区占比最大的商品
scale_2010 = scale_factors[scale_factors['year'] == 2010].copy()
scale_2010['diff'] = scale_2010['total_exports'] - scale_2010['subtotal_exports']
scale_2010['diff_pct'] = (scale_2010['diff'] / scale_2010['total_exports'] * 100).fillna(0)

print(f"\n  2010年未指定地区占比最大的商品（前10）:")
top_diff = scale_2010.nlargest(10, 'diff')
for i, (_, row) in enumerate(top_diff.iterrows(), 1):
    if row['diff'] > 0:
        print(f"    {i:2d}. {row['LUTO']:45s}: {row['diff']:>12,.0f} "
              f"({row['diff_pct']:5.1f}%, scale={row['scale_factor']:.4f})")

# 步骤4: 应用缩放因子
print("\n步骤4: 应用缩放因子...")

trade_data_scaled = trade_data.merge(
    scale_factors[['LUTO', 'year', 'scale_factor']],
    left_on=['group', 'year'],
    right_on=['LUTO', 'year'],
    how='left'
)

trade_data_scaled['scale_factor'] = trade_data_scaled['scale_factor'].fillna(1)

# 只对Export Quantity应用缩放
export_mask = trade_data_scaled['Element'] == 'Export Quantity'
original_sum = trade_data_scaled.loc[export_mask, 'trade'].sum()

trade_data_scaled.loc[export_mask, 'trade'] = (
        trade_data_scaled.loc[export_mask, 'trade'] *
        trade_data_scaled.loc[export_mask, 'scale_factor']
)

scaled_sum = trade_data_scaled.loc[export_mask, 'trade'].sum()

print(f"  ✓ 缩放前出口总量: {original_sum:,.0f}")
print(f"  ✓ 缩放后出口总量: {scaled_sum:,.0f}")
print(f"  ✓ 增加量: {scaled_sum - original_sum:,.0f} ({(scaled_sum / original_sum - 1) * 100:.2f}%)")

# 清理并替换原数据
trade_data_scaled = trade_data_scaled.drop(columns=['LUTO', 'scale_factor'], errors='ignore')
trade_data = trade_data_scaled.copy()

# 验证Sugar
print("\n步骤5: 验证Sugar数据...")
sugar_2010 = trade_data[
    (trade_data['year'] == 2010) &
    (trade_data['Element'] == 'Export Quantity') &
    (trade_data['group'].str.contains('Sugar', case=False, na=False))
    ]

if not sugar_2010.empty:
    sugar_total = sugar_2010['trade'].sum()
    print(f"  ✓ Sugar 2010年出口总量（缩放后）: {sugar_total:,.2f}")

    sugar_by_group = sugar_2010.groupby('group')['trade'].sum()
    for group, value in sugar_by_group.items():
        print(f"      {group}: {value:,.2f}")

print("\n" + "=" * 80)
print("✓ 未指定地区数据重新分配完成！")
print("=" * 80 + "\n")

print("trade_data rows:", trade_data.shape)

# ---- 2) Prepare GDP per capita long table (gdp_pc has columns 'Country Name','Country Code','1990'..'2020') ----
gdp_year_cols = [c for c in gdp_pc.columns if re.match(r'^\d{4}$', str(c))]
gdp_pc_long = gdp_pc.melt(
    id_vars=['Country Name', 'Country Code'],
    value_vars=gdp_year_cols,
    var_name='year',
    value_name='gdp_pc'
)

# ---- 3) Prepare gravity subset for AUS origin ----
# gravity columns include 'iso3_o' and 'iso3_d' and 'year' and 'distance' and 'lat_d','lng_d'
gravity_au = gravity[['iso3_o','iso3_d', 'year', 'distance', 'lat_d', 'lng_d']].copy()

# ---- 4) Prepare SSP2 columns: pick the first matching Population.WB and Urban.population.pct.WB if duplicates ----
# ssp2 columns include lowercase 'country.code' in your print; keep as-is
# get Population.WB column (if duplicated, pick first)

gdp_pc_long['year'] = pd.to_numeric(gdp_pc_long['year'], errors='coerce').astype(int)
# merge gdp_pc
merged = trade_data.merge(
    gdp_pc_long[['Country Code', 'year', 'gdp_pc']],
    left_on=['Partner ISO', 'year'],
    right_on=['Country Code', 'year'],
    how='left',
)

# merge gravity (left join: add distance/lat/lng)
gravity_au['year'] = pd.to_numeric(gravity_au['year'], errors='coerce').astype(int)
merged = merged.merge(
    gravity_au[['iso3_o','iso3_d', 'year', 'distance']],
    left_on=['Report ISO','Partner ISO', 'year'],
    right_on=['iso3_o','iso3_d', 'year'],
    how='left'
)

# merge ssp2 (Population & Urban)
ssp2['country.code'] = ssp2['country.code'].astype(str).str.strip().str.upper()
ssp2['year'] = pd.to_numeric(ssp2['year'], errors='coerce').astype(int)
merged = merged.merge(
    ssp2[['country.code', 'year', 'country', 'Population.WB', 'Urban.population.pct.WB']],
    left_on=['Partner ISO', 'year'],
    right_on=['country.code', 'year'],
    how='left',
)

# if merged has duplicate column names, drop the right-hand key columns
delete_columns = ['Country Code', 'iso3_o', 'iso3_d', 'country.code']
for col in delete_columns:
    if col in merged.columns:
        merged = merged.drop(columns=[col])

# rename ssp2 cols to fixed names
# merged = merged.rename(columns={pop_col: 'Population.WB', urban_col: 'Urban.population.pct.WB'})
df = merged.dropna(subset=['trade', 'gdp_pc', 'Population.WB', 'Urban.population.pct.WB', 'distance']).copy()

# 9) save results and scaler
df.to_csv(OUT_CSV, index=False)



print("创建2010-2050年预测表...")

# ---- 1) 从 tradeF 获取所有唯一的贸易对组合 ----
tradeF = pd.read_csv(OUT_CSV, low_memory=False)
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
merged = merged.drop_duplicates()
print(f"\n最终数据: {merged.shape[0]} 行")
print(f"缺失值:\n{merged[['gdp_pc', 'Population.WB', 'Urban.population.pct.WB', 'distance']].isnull().sum()}")

# 6) 保存
df_complete = merged.dropna(subset=['gdp_pc', 'Population.WB', 'Urban.population.pct.WB', 'distance'])
df_complete.to_csv('../2_processed_data/future_trade_AUS_2010_2050.csv', index=False)

df_saved = pd.read_csv("../2_processed_data/trade_model_data_AUS_all.csv")

# 筛选2010年、Export Quantity、Sugar相关商品
sugar_2010 = df_saved[
    (df_saved['year'] == 2010) &
    (df_saved['Element'] == 'Export Quantity') &
    (df_saved['group'].str.contains('Sugar', case=False, na=False))
]

# 计算总量
total_sugar_export = sugar_2010['trade'].sum()

print(f"\n2010年澳大利亚Sugar出口总量: {total_sugar_export:,.2f} 吨")
print(f"\nSugar相关商品明细:")
sugar_by_group = sugar_2010.groupby('group')['trade'].sum().sort_values(ascending=False)
for group, value in sugar_by_group.items():
    print(f"  {group}: {value:,.2f}")