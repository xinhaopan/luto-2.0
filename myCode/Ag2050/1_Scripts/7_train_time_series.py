import numpy as np
import pandas as pd
import pymc as pm
import pickle
from pathlib import Path
from joblib import Parallel, delayed


def train_single_pair(df_pair: pd.DataFrame,
                      group_name: str,
                      element_name: str) -> dict:
    """
    贝叶斯线性回归:  trade = alpha + beta * year

    返回后验样本，用于后续预测和画图
    """

    # 1.准备训练数据 (1990-2014)
    train_df = df_pair[(df_pair['year'] >= 1990) & (df_pair['year'] <= 2014)].copy()

    if len(train_df) < 5:
        print(f"[SKIP] {group_name} - {element_name}: 数据不足")
        return None

    x_train = train_df['year'].values.astype(float)
    y_train = train_df['trade'].values.astype(float)

    print(f"[TRAIN] {group_name} - {element_name}")

    # 2.贝叶斯线性回归
    with pm.Model() as model:
        alpha = pm.Normal('alpha', mu=y_train.mean(), sigma=y_train.std() * 10)
        beta = pm.Normal('beta', mu=0, sigma=y_train.std())
        sigma = pm.HalfNormal('sigma', sigma=y_train.std() * 2)

        mu = alpha + beta * x_train
        likelihood = pm.Normal('likelihood', mu=mu, sigma=sigma, observed=y_train)

        trace = pm.sample(1000, tune=1000, chains=4, cores=1,
                          return_inferencedata=True, progressbar=False)

    # 3.提取后验样本
    alpha_samples = trace.posterior['alpha'].values.flatten()
    beta_samples = trace.posterior['beta'].values.flatten()

    return {
        'group': group_name,
        'element': element_name,
        'alpha_samples': alpha_samples,
        'beta_samples': beta_samples,
    }


def train_one_pair_wrapper(group_name, element_name, df_pair):
    """
    包装函数，用于并行处理
    """
    try:
        result = train_single_pair(df_pair, group_name, element_name)
        return result
    except Exception as e:
        print(f"[ERROR] {group_name} - {element_name}: {e}")
        return None


def main():
    # ========== 配置 ==========
    DATA_PATH = "../2_processed_data/trade_model_data_all.csv"
    OUTPUT_DIR = "../2_processed_data/trained_models"
    N_JOBS = 100  # 并行任务数

    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    # ========== 加载数据 ==========
    print("加载数据...")
    df = pd.read_csv(DATA_PATH)
    df = df[df["Report ISO"].str.upper() == "AUS"].copy()
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df["trade"] = pd.to_numeric(df["trade"], errors="coerce")
    df = df.dropna(subset=["year", "trade"])

    # 聚合数据
    df_agg = df.groupby(["group", "Element", "year"], as_index=False)["trade"].sum()
    print(f"聚合后数据: {len(df_agg)} 行")

    # 获取所有 (group, element) 组合
    pairs = df_agg[["group", "Element"]].drop_duplicates()
    print(f"总共 {len(pairs)} 个商品需要训练\n")

    # ========== 准备任务 ==========
    tasks = []
    for _, row in pairs.iterrows():
        group_name = row["group"]
        element_name = row["Element"]
        df_pair = df_agg[
            (df_agg["group"] == group_name) &
            (df_agg["Element"] == element_name)
            ].copy()
        tasks.append((group_name, element_name, df_pair))

    # ========== 并行训练 ==========
    print(f"开始并行训练 (n_jobs={N_JOBS})...\n")
    results = Parallel(n_jobs=N_JOBS, verbose=10)(
        delayed(train_one_pair_wrapper)(g, e, df) for g, e, df in tasks
    )

    # ========== 保存结果 ==========
    print("\n保存结果...")
    valid_results = [r for r in results if r is not None]
    print(f"成功训练:  {len(valid_results)}/{len(tasks)}")

    # 保存为单个文件
    output_file = Path(OUTPUT_DIR) / "all_trained_models.pkl"
    with open(output_file, 'wb') as f:
        pickle.dump(valid_results, f)

    print(f"\n✓ 所有结果已保存到:  {output_file}")

    # 显示样例
    print("\n样例结果:")
    for i, result in enumerate(valid_results[:3]):
        print(f"\n[{i + 1}] {result['group']} - {result['element']}")
        print(f"    Alpha 样本数: {len(result['alpha_samples'])}")
        print(f"    Beta 样本数:  {len(result['beta_samples'])}")
        print(f"    Beta 均值: {result['beta_samples'].mean():.4f}")
        print(f"    Beta 80% CI: [{np.percentile(result['beta_samples'], 10):.4f}, "
              f"{np.percentile(result['beta_samples'], 90):.4f}]")


if __name__ == "__main__":
    main()