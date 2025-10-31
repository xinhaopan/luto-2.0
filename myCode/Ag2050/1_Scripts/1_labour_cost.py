import numpy as np
import pandas as pd
from pathlib import Path
from tools import predict_growth_index

# ---------- 配置 ----------
import pandas as pd
import numpy as np

# 读取原始数据
df = pd.read_excel('../0_original_data/labour_cost.xlsx', usecols="A,G")
col0, col1 = df.columns[0], df.columns[1]

# 解析 "Aug-14" -> 2014
df['Year'] = pd.to_datetime(df[col0], format='%b-%y', errors='coerce').dt.year
df.rename(columns={col1: 'Cost'}, inplace=True)
df['Cost'] = pd.to_numeric(df['Cost'], errors='coerce')
# 提取 Year 和 Cost，排序
df = df[['Year', 'Cost']].dropna(subset=['Year']).sort_values('Year').reset_index(drop=True)
# 获取2014年的成本值（第一个非NaN值）
cost_2014 = df.loc[df['Year'] == 2014, 'Cost'].values[0]
# 构造 2010–2013 年份的数据，用2014年的Cost填充
pre_years = pd.DataFrame({'Year': [2010, 2011, 2012, 2013], 'Cost': cost_2014})
# 合并补全的和原始数据，并排序
df = pd.concat([pre_years, df], ignore_index=True).drop_duplicates('Year').sort_values('Year').reset_index(drop=True)

# ---------- 预测增长指数并绘图 ----------
ax, df_result = predict_growth_index(df, var_name='Labour Cost',base_year = 2014)
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels, loc='upper left', frameon=True)

fig = ax.get_figure()
fig.savefig('labour_cost_growth_index.png', dpi=300)
fig.show()
excel_path = "../0_original_data/FLC_cost_multipliers.xlsx"   # 把此处改为你的 Excel 文件路径
template_sheet = "FLC_multiplier"            # 模板表所在 sheet（0 表示第一个 sheet）

# ---------- 读取模板 Excel ----------
p = Path(excel_path)
if not p.exists():
    raise FileNotFoundError(f"指定的文件不存在：{excel_path}")

df_template = pd.read_excel(excel_path, sheet_name=template_sheet)
# 假定 Year 是一列
if 'Year' not in df_template.columns:
    raise KeyError("找不到 'Year' 列，请确认模板表中包含 'Year' 列。")

# 获取年份（行名）和作物列名（排除 'Year'）
years = df_template['Year'].astype(int).tolist()
crop_cols = [c for c in df_template.columns if c != 'Year']

print("读取到年份（行名）：", years)
print("读取到作物列名：", crop_cols)

# 统一索引为整数年份
df_result = df_result.copy()
df_result.index = df_result.index.astype(int)

# 容错查找情景列名（尝试常见变体）
def find_col(df, targets):
    """在 df.columns 中按大小写不敏感匹配 targets（列表），返回第一个存在的列名或 None。"""
    cols_lower = {c.lower(): c for c in df.columns}
    for t in targets:
        tl = t.lower()
        if tl in cols_lower:
            return cols_lower[tl]
    # 尝试更宽松匹配（去下划线和空格）
    norm_map = {c.replace('_','').replace(' ','').lower(): c for c in df.columns}
    for t in targets:
        key = t.replace('_','').replace(' ','').lower()
        if key in norm_map:
            return norm_map[key]
    return None

scenario_map = {
    'FLC_multiplier_low':    ['Low', 'low'],
    'FLC_multiplier_medium': ['Medium', 'medium'],
    'FLC_multiplier_high':   ['High', 'high'],
    'FLC_multiplier_very_high': ['Very_High', 'Very High', 'VeryHigh', 'very_high', 'very high', 'veryhigh']
}

found_cols = {}
for sheet, candidates in scenario_map.items():
    col = find_col(df_result, candidates)
    if col is None:
        print(f"警告：在 df_result 中未找到与 {sheet} 对应的情景列（尝试了 {candidates}）。该 sheet 中的值将全部为 NaN。")
    else:
        print(f"情景 '{sheet}' 对应 df_result 列名为: {col}")
    found_cols[sheet] = col  # 可能为 None

# ---------- 为每个情景构建表格 ----------
sheets_to_write = {}
for sheet_name, col_name in found_cols.items():
    # DataFrame：index 为 years，columns 为 crop_cols
    df_sheet = pd.DataFrame(index=years, columns=crop_cols, dtype=float)
    df_sheet.index.name = 'Year'
    # 按年份填充：该年份在 df_result 中有值则取该值，否则为 NaN
    for y in years:
        if col_name is not None and (y in df_result.index):
            val = df_result.at[y, col_name]
        else:
            val = np.nan
        # 把该年份所有作物列都设为 val
        df_sheet.loc[y, :] = val
    sheets_to_write[sheet_name] = df_sheet.reset_index()  # 写 Excel 时通常希望把 Year 作为列

# ---------- 写入 Excel（追加或替换 sheet） ----------
# 如果文件存在，使用 mode='a' 并替换同名 sheet；否则新建文件
mode = 'a' if p.exists() else 'w'
# pandas >= 1.3 支持 if_sheet_exists 参数
with pd.ExcelWriter(excel_path, engine='openpyxl', mode=mode, if_sheet_exists='replace') as writer:
    for sheet_name, df_sheet in sheets_to_write.items():
        df_sheet.to_excel(writer, sheet_name=sheet_name, index=False)
print(f"已将 {len(sheets_to_write)} 个 sheet 写入到 {excel_path}（同名 sheet 将被替换）。")