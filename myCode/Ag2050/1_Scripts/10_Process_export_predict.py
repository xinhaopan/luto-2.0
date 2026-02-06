import pandas as pd
from pathlib import Path
import numpy as np

# ========== 配置 ==========
input_path = "../2_processed_data/trained_models_ets"
his_file = "historical_data_aggregated.xlsx"
pre_file = "all_predictions_long_ets_raw.xlsx"
output_path = "../2_processed_data/trained_models_ets/export_projections"

Path(output_path).mkdir(parents=True, exist_ok=True)

# ========== 完整分类映射 ==========
category_mapping = {
    # Ruminants/Monogastrics/Dairy (反刍动物/单胃动物/乳制品)
    'Animals live nes (heads)': 'Ruminants/Monogastrics/Dairy',
    'Bovine Meat': 'Ruminants/Monogastrics/Dairy',
    'Cattle (heads)': 'Ruminants/Monogastrics/Dairy',
    'Cheese': 'Ruminants/Monogastrics/Dairy',
    'Eggs': 'Ruminants/Monogastrics/Dairy',
    'Fats, Animals, Raw': 'Ruminants/Monogastrics/Dairy',
    'Meat, Other': 'Ruminants/Monogastrics/Dairy',
    'Milk - Excluding Butter': 'Ruminants/Monogastrics/Dairy',
    'Milk products (nec)': 'Ruminants/Monogastrics/Dairy',
    'Mutton & Goat Meat': 'Ruminants/Monogastrics/Dairy',
    'Offals, Edible': 'Ruminants/Monogastrics/Dairy',
    'Pigmeat': 'Ruminants/Monogastrics/Dairy',
    'Pigs (heads)': 'Ruminants/Monogastrics/Dairy',
    'Poultry (1000 heads)': 'Ruminants/Monogastrics/Dairy',
    'Poultry Meat': 'Ruminants/Monogastrics/Dairy',
    'Sheep and goats  (heads)': 'Ruminants/Monogastrics/Dairy',

    # Food & Fibre (食品与纤维)
    'Beverages, Alcoholic': 'Food & Fibre',
    'Cereals (nec)': 'Food & Fibre',
    'Coffee and tea': 'Food & Fibre',
    'Cottonseed Oil': 'Food & Fibre',
    'Fodder and Feeding Stuff': 'Food & Fibre',
    'Hides and skins': 'Food & Fibre',
    'Miscellaneous': 'Food & Fibre',
    'Non-alcoholic Beverages': 'Food & Fibre',
    'Non-edible Crude Materials': 'Food & Fibre',
    'Non-edible Fats and Oils': 'Food & Fibre',
    'Rice and products': 'Food & Fibre',
    'Sugar ': 'Food & Fibre',
    'Wheat and products': 'Food & Fibre',
    'Wine': 'Food & Fibre',
    'Wool': 'Food & Fibre',

    # Plant-based proteins (植物蛋白)
    'Beans': 'Plant-based proteins',
    'Groundnuts (Shelled Eq)': 'Plant-based proteins',
    'Oilseeds': 'Plant-based proteins',
    'Peas': 'Plant-based proteins',
    'Pulses, Other and products': 'Plant-based proteins',
    'Soyabeans': 'Plant-based proteins',

    # Horticulture (园艺)
    'Apples and products': 'Horticulture',
    'Bananas': 'Horticulture',
    'Fruits (nec)': 'Horticulture',
    'Grapes and products (excl wine)': 'Horticulture',
    'Nuts and products': 'Horticulture',
    'Potatoes and products': 'Horticulture',
    'Seafood': 'Horticulture',
    'Vegetables (nec)': 'Horticulture',
}

# ========== 场景映射规则 ==========
# Domestic demand
scenario_rules = {
    'Ruminants/Monogastrics/Dairy': {
        'AgS1': 'CI80_Upper',  # ↑
        'AgS2': 'CI95_Lower',  # ↓
        'AgS3': 'CI80_Upper',  # ↑
        'AgS4': 'CI95_Lower',  # ↓
    },
    'Food & Fibre': {
        'AgS1': 'CI80_Upper',  # ↑
        'AgS2': 'Mean',  # ↔
        'AgS3': 'CI80_Upper',  # ↑
        'AgS4': 'CI95_Lower',  # ↓
    },
    'Plant-based proteins': {
        'AgS1': 'Mean',  # ↔
        'AgS2': 'CI95_Upper',  # ↑↑
        'AgS3': 'Mean',  # ↔
        'AgS4': 'CI80_Upper',  # ↑
    },
    'Horticulture': {
        'AgS1': 'Mean',  # ↔
        'AgS2': 'CI80_Upper',  # ↑
        'AgS3': 'Mean',  # ↔
        'AgS4': 'Mean',  # ↔
    },
}

# export
scenario_rules = {
    'Ruminants/Monogastrics/Dairy': {
        'AgS1': 'CI95_Upper',
        'AgS2': 'CI80_Upper',
        'AgS3': 'Mean',
        'AgS4': 'Static'
    },
    'Food & Fibre': {
        'AgS1': 'CI95_Upper',
        'AgS2': 'CI80_Upper',
        'AgS3': 'Mean',
        'AgS4': 'Static'
    },
    'Plant-based proteins': {
        'AgS1': 'CI95_Upper',
        'AgS2': 'CI80_Upper',
        'AgS3': 'Mean',
        'AgS4': 'Static'
    },
    'Horticulture': {
        'AgS1': 'CI95_Upper',
        'AgS2': 'CI80_Upper',
        'AgS3': 'Mean',
        'AgS4': 'Static'
    },
}

# ========== 加载数据 ==========
print("加载数据...")
df_his = pd.read_excel(f"{input_path}/{his_file}")
df_pre = pd.read_excel(f"{input_path}/{pre_file}")

print(f"历史数据: {len(df_his)} 行")
print(f"预测数据: {len(df_pre)} 行")

# 只处理 Export Quantity
df_his = df_his[df_his['Element'] == 'Export Quantity'].copy()
df_pre = df_pre[df_pre['Element'] == 'Export Quantity'].copy()

print(f"筛选后历史数据: {len(df_his)} 行")
print(f"筛选后预测数据: {len(df_pre)} 行")


# ========== 处理函数 ==========
def process_group(group_name, df_his_group, df_pre_group):
    """处理单个商品的数据"""

    # 确定类别
    if group_name not in category_mapping:
        print(f"  ⚠️  警告: {group_name} 未分类，默认归为 'Food & Fibre'")
        category = 'Food & Fibre'
    else:
        category = category_mapping[group_name]

    rules = scenario_rules[category]

    print(f"\n处理: {group_name}")
    print(f"  类别: {category}")

    # 获取 2010 年的值作为 Static
    static_value = df_his_group[df_his_group['year'] == 2010]['trade'].values
    if len(static_value) > 0:
        static_value = float(static_value[0])
        print(f"  2010年值 (Static): {static_value:.2f}")
    else:
        static_value = np.nan
        print(f"  2010年值 (Static): 无数据")

    # 创建年份范围 (1990-2050)
    years = list(range(1990, 2051))

    result_data = []

    for idx, year in enumerate(years, start=1):
        row = {'...1': idx, 'year': year, 'Static': static_value}

        # 历史数据 (<=2010)
        if year <= 2010:
            his_row = df_his_group[df_his_group['year'] == year]
            if not his_row.empty:
                hist_value = float(his_row['trade'].values[0])
                hist_value = max(0, hist_value)  # 小于0设为0

                row['historical'] = hist_value
                row['AgS1'] = hist_value
                row['AgS2'] = hist_value
                row['AgS3'] = hist_value
                row['AgS4'] = hist_value
            else:
                row['historical'] = np.nan
                row['AgS1'] = np.nan
                row['AgS2'] = np.nan
                row['AgS3'] = np.nan
                row['AgS4'] = np.nan

            # fitted 值 (从预测数据获取，如果有的话)
            pre_row = df_pre_group[df_pre_group['Year'] == year]
            if not pre_row.empty:
                row['fitted'] = max(0, float(pre_row['Mean'].values[0]))
            else:
                row['fitted'] = np.nan

            row['Trend'] = np.nan

        # 预测数据 (>2010)
        else:
            row['historical'] = np.nan

            pre_row = df_pre_group[df_pre_group['Year'] == year]
            if not pre_row.empty:
                # fitted 和 Trend 都使用 Mean
                mean_value = float(pre_row['Mean'].values[0])
                row['fitted'] = max(0, mean_value)
                row['Trend'] = max(0, mean_value)

                # 根据场景规则设置 AgS1-AgS4
                for scenario, col_name in rules.items():
                    if col_name == 'Static':
                        # Static: use 2010 value
                        value = static_value
                    else:
                        value = float(pre_row[col_name].values[0])

                    # 小于0设为0；static_value 可能是 nan，这里也会保留 nan
                    row[scenario] = max(0, value) if not pd.isna(value) else np.nan
            else:
                row['fitted'] = np.nan
                row['Trend'] = np.nan
                row['AgS1'] = np.nan
                row['AgS2'] = np.nan
                row['AgS3'] = np.nan
                row['AgS4'] = np.nan

        result_data.append(row)

    # 创建 DataFrame
    df_result = pd.DataFrame(result_data)

    # 调整列顺序
    df_result = df_result[['...1', 'historical', 'fitted', 'year',
                           'Trend', 'Static', 'AgS1', 'AgS2', 'AgS3', 'AgS4']]

    return df_result, category


# ========== 处理所有商品 ==========
print("\n" + "=" * 80)
print("开始处理所有商品...")
print("=" * 80)

# 获取所有需要处理的商品
groups_from_his = set(df_his['group'].unique())
groups_from_pre = set(df_pre['Group'].unique())
all_groups = groups_from_his.union(groups_from_pre)

print(f"历史数据中的商品: {len(groups_from_his)}")
print(f"预测数据中的商品: {len(groups_from_pre)}")
print(f"总共需要处理: {len(all_groups)} 个商品")

success_count = 0
error_count = 0
category_stats = {cat: 0 for cat in scenario_rules.keys()}

for group_name in sorted(all_groups):
    df_his_group = df_his[df_his['group'] == group_name].copy()
    df_pre_group = df_pre[df_pre['Group'] == group_name].copy()

    try:
        df_result, category = process_group(group_name, df_his_group, df_pre_group)

        # 保存文件（使用原始商品名）
        output_file = f"{output_path}/{group_name} export projections_ag2050.csv"
        df_result.to_csv(output_file, index=False)

        # 统计
        success_count += 1
        category_stats[category] += 1

        # 显示关键信息
        year_2010 = df_result[df_result['year'] == 2010]
        year_2050 = df_result[df_result['year'] == 2050]

        if not year_2010.empty:
            val_2010 = year_2010['historical'].values[0]
            print(f"  2010年历史值: {val_2010 if not pd.isna(val_2010) else 'N/A'}")

        if not year_2050.empty:
            scenarios = year_2050[['AgS1', 'AgS2', 'AgS3', 'AgS4']].values[0]
            print(f"  2050年预测 (S1/S2/S3/S4): {scenarios}")

        print(f"  ✓ 已保存: {group_name} export projections_ag2050.csv")

    except Exception as e:
        error_count += 1
        print(f"\n[ERROR] {group_name}: {e}")
        import traceback

        traceback.print_exc()

print("\n" + "=" * 80)
print("处理完成！")
print("=" * 80)
print(f"成功: {success_count} 个")
print(f"失败: {error_count} 个")
print(f"\n各类别统计:")
for cat, count in category_stats.items():
    print(f"  {cat}: {count} 个商品")

# ========== 生成分类汇总 ==========
print("\n生成分类汇总...")
summary_data = []
for group_name in sorted(all_groups):
    category = category_mapping.get(group_name, 'Food & Fibre')
    rules = scenario_rules[category]
    summary_data.append({
        'Group': group_name,
        'Category': category,
        'AgS1_Rule': rules['AgS1'],
        'AgS2_Rule': rules['AgS2'],
        'AgS3_Rule': rules['AgS3'],
        'AgS4_Rule': rules['AgS4'],
    })

df_summary = pd.DataFrame(summary_data)
summary_file = f"{output_path}/category_summary.csv"
df_summary.to_csv(summary_file, index=False)
print(f"✓ 分类汇总已保存: {summary_file}")

# 显示未分类的商品
unmapped = [g for g in all_groups if g not in category_mapping]
if unmapped:
    print(f"\n⚠️  未分类的商品 ({len(unmapped)} 个):")
    for g in sorted(unmapped):
        print(f"  - {g}")

print("\n" + "=" * 80)
print("全部完成！")
print("=" * 80)

