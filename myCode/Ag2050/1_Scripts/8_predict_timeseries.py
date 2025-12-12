import numpy as np
import pandas as pd
import pickle
from pathlib import Path


def predict_with_intervals(result: dict,
                           start_year: int = 1990,
                           end_year: int = 2050) -> pd.DataFrame:
    """
    使用后验样本进行预测，计算预测区间
    """
    alpha_samples = result['alpha_samples']
    beta_samples = result['beta_samples']
    group_name = result['group']
    element_name = result['element']

    # 预测年份
    years = np.arange(start_year, end_year + 1)

    predictions = []

    for year in years:
        # 用所有后验样本预测
        pred_samples = alpha_samples + beta_samples * year

        predictions.append({
            'group': group_name,
            'element': element_name,
            'year': int(year),
            'prediction': float(pred_samples.mean()),  # 预测值
            'lower_95': float(np.percentile(pred_samples, 2.5)),  # 95% 下界
            'upper_95': float(np.percentile(pred_samples, 97.5)),  # 95% 上界
            'lower_80': float(np.percentile(pred_samples, 10)),  # 80% 下界
            'upper_80': float(np.percentile(pred_samples, 90)),  # 80% 上界
        })

    return pd.DataFrame(predictions)


def main():
    # ========== 配置 ==========
    MODEL_FILE = "../2_processed_data/trained_models/all_trained_models.pkl"
    OUTPUT_FILE = "../2_processed_data/predictions_1990_2050.csv"

    START_YEAR = 1990
    END_YEAR = 2050

    # ========== 加载训练结果 ==========
    print(f"加载训练结果:  {MODEL_FILE}")

    if not Path(MODEL_FILE).exists():
        print(f"错误: 找不到模型文件 {MODEL_FILE}")
        print("请先运行训练代码!")
        return

    with open(MODEL_FILE, 'rb') as f:
        trained_models = pickle.load(f)

    print(f"总共 {len(trained_models)} 个模型\n")

    if len(trained_models) == 0:
        print("错误: 没有训练好的模型!")
        return

    # ========== 对每个商品进行预测 ==========
    all_predictions = []
    failed_count = 0

    for i, result in enumerate(trained_models, 1):
        group_name = result['group']
        element_name = result['element']

        print(f"[{i}/{len(trained_models)}] 预测:  {group_name} - {element_name}")

        try:
            # 预测
            pred_df = predict_with_intervals(result, START_YEAR, END_YEAR)
            all_predictions.append(pred_df)
        except Exception as e:
            print(f"  ✗ 预测失败: {e}")
            failed_count += 1

    # ========== 合并所有预测 ==========
    if not all_predictions:
        print("\n错误: 没有成功的预测!")
        return

    print(f"\n合并所有预测结果...  (成功: {len(all_predictions)}, 失败: {failed_count})")
    df_all_pred = pd.concat(all_predictions, ignore_index=True)

    # ========== 保存结果 ==========
    df_all_pred.to_csv(OUTPUT_FILE, index=False)
    print(f"\n✓ 预测结果已保存到: {OUTPUT_FILE}")
    print(f"  总行数: {len(df_all_pred)}")
    print(f"  列: {list(df_all_pred.columns)}")

    # ========== 显示样例 ==========
    print("\n样例预测结果 (前20行):")
    print(df_all_pred.head(20).to_string(index=False))

    # ========== 统计信息 ==========
    print(f"\n{'=' * 80}")
    print("统计信息:")
    print(f"{'=' * 80}")

    n_unique_pairs = df_all_pred[['group', 'element']].drop_duplicates().shape[0]
    print(f"  商品-类型组合数量: {n_unique_pairs}")
    print(f"  年份范围: {df_all_pred['year'].min()} - {df_all_pred['year'].max()}")

    # 按 element 统计
    print(f"\n按类型统计:")
    for element in sorted(df_all_pred['element'].unique()):
        n_groups = df_all_pred[df_all_pred['element'] == element]['group'].nunique()
        print(f"  {element}: {n_groups} 个商品")

    # 预测值范围
    print(f"\n预测值范围:")
    print(f"  最小值: {df_all_pred['prediction'].min():.2f}")
    print(f"  最大值:  {df_all_pred['prediction'].max():.2f}")
    print(f"  均值: {df_all_pred['prediction'].mean():.2f}")

    # 检查是否有异常值（负预测值）
    negative_count = (df_all_pred['prediction'] < 0).sum()
    if negative_count > 0:
        print(f"\n⚠️  警告: 有 {negative_count} 个负预测值")
        print("负预测值的商品:")
        neg_data = df_all_pred[df_all_pred['prediction'] < 0][
            ['group', 'element', 'year', 'prediction']].drop_duplicates(['group', 'element'])
        print(neg_data.head(10).to_string(index=False))

    print(f"\n{'=' * 80}")
    print("预测完成!  (仅澳大利亚 AUS)")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()