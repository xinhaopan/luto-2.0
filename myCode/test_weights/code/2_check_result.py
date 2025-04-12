import os
import pandas as pd

df_runs = pd.read_csv('../../tasks_run/Custom_runs/setting_0410_test_weight1.csv')
file_names = [name.replace('.', '_') for name in df_runs.columns[2:]]
output_dir = '../output'

# 要排除的 Production 组
exclude_prod = {'beef lexp','beef meat','sheep lexp','sheep meat','sheep wool'}

# 用于收集结果的容器
missing = []
econ    = []
gbf2    = []
ghg     = []
water   = []
prod    = []

existing = []

for path_name in file_names:
    df_path = os.path.join(output_dir, f'{path_name}.xlsx')
    if not os.path.exists(df_path):
        missing.append({'path_name': path_name})
        continue

    existing.append(path_name)
    df    = pd.read_excel(df_path)
    df2050 = df[df['Year'] == 2050]

    # Economy
    sub = df2050[
        (df2050['Indicator']=="Economy Total Value (Billion AUD)") &
        (df2050['Value'] < 0)
    ]
    for _, row in sub.iterrows():
        econ.append({'path_name': path_name, 'Value': row['Value']})

    # GBF2
    sub = df2050[
        (df2050['Indicator']=="GBF2 Deviation (ha)") &
        (df2050['Value'] < -0.1)
    ]
    for _, row in sub.iterrows():
        gbf2.append({'path_name': path_name, 'Value': row['Value']})

    # GHG
    sub = df2050[
        (df2050['Indicator']=="GHG Oversheet (MtCO2e)") &
        (df2050['Value'] < -0.1)
    ]
    for _, row in sub.iterrows():
        ghg.append({'path_name': path_name, 'Value': row['Value']})

    # Water (可能多行)
    sub = df2050[
        (df2050['Indicator']=="Water Deviation (TL)") &
        (df2050['Value'] < -0.1)
    ]
    for _, row in sub.iterrows():
        water.append({
            'path_name': path_name,
            'Group': row['Group'],
            'Value': row['Value']
        })

    # Production（排除指定组）
    sub = df2050[
        (df2050['Indicator']=="Production Deviation (%)") &
        (df2050['Value'] < -0.1) &
        (~df2050['Group'].isin(exclude_prod))
    ]
    for _, row in sub.iterrows():
        prod.append({
            'path_name': path_name,
            'Group': row['Group'],
            'Value': row['Value']
        })

# 构造 DataFrame
df_missing = pd.DataFrame(missing)
df_econ = pd.DataFrame(econ, columns=['path_name','Value']).drop_duplicates()
df_gbf2 = pd.DataFrame(gbf2, columns=['path_name','Value']).drop_duplicates()
df_ghg  = pd.DataFrame(ghg,  columns=['path_name','Value']).drop_duplicates()


df_water_wide = (
    pd.DataFrame(water)
      .pivot_table(index='path_name', columns='Group', values='Value', aggfunc='first')
      .reset_index()
)

df_prod_wide = (
    pd.DataFrame(prod)
      .pivot_table(index='path_name', columns='Group', values='Value', aggfunc='first')
      .reset_index()
)
# 计算没有问题的 path_name
bad = set(df_econ['path_name']) \
      | set(df_gbf2['path_name']) \
      | set(df_ghg['path_name']) \
      | set(df_water_wide['path_name']) \
      | set(df_prod_wide['path_name'])
good = sorted(set(existing) - bad)
df_good = pd.DataFrame({'path_name': good})

# 统计各组数量
counts = {
    'Missing Files': len(df_missing),
    'Economy Negative': len(df_econ),
    'GBF2 Negative': len(df_gbf2),
    'GHG Negative': len(df_ghg),
    'Water Negative': len(df_water_wide),
    'Production Negative': len(df_prod_wide),
    'No Issues': len(df_good),
}
df_counts = pd.DataFrame.from_dict(counts, orient='index', columns=['Count']).reset_index()
df_counts.rename(columns={'index': 'Group'}, inplace=True)

# 写入多 sheet 的 Excel
summary_path = '../result/summary.xlsx'
with pd.ExcelWriter(summary_path) as writer:
    df_good.to_excel(writer, sheet_name='No Issues', index=False)
    df_counts.to_excel(writer, sheet_name='Counts', index=False)
    df_missing.to_excel(writer, sheet_name='Missing Files', index=False)
    df_econ.to_excel(writer, sheet_name='Economy Negative', index=False)
    df_gbf2.to_excel(writer, sheet_name='GBF2 Negative', index=False)
    df_ghg.to_excel(writer, sheet_name='GHG Negative', index=False)
    df_water_wide.to_excel(writer, sheet_name='Water Negative', index=False)
    df_prod_wide.to_excel(writer, sheet_name='Production Negative', index=False)
print(df_counts)
print(f"Summary written to {summary_path}")