import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, Any, Tuple, List

from statsmodels.tsa.exponential_smoothing.ets import ETSModel
from statsmodels.tsa.statespace.structural import UnobservedComponents
import statsmodels.api as sm

# ---------------- Metrics ----------------
def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred))**2)))

def mae(y_true, y_pred):
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))

def mape(y_true, y_pred, eps=1e-9):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    mask = np.abs(y_true) > eps    # 防止除零
    if mask.sum() == 0:
        return np.nan
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0)

# ------------- Rolling-origin CV -------------
@dataclass
class CVConfig:
    horizon: int = 1              # 预测步长（一步预测）
    min_train_size: int = 8       # 最小训练样本点数
    step: int = 1                 # 每次向前滚动的步长

def rolling_origin_splits(n_obs: int, cfg: CVConfig):
    """
    生成滚动起点的 (train_end_idx, test_end_idx) 对。
    train: [0 : train_end_idx] ; test: 预测下一个 horizon 的点到 test_end_idx
    """
    for train_end in range(cfg.min_train_size - 1, n_obs - cfg.horizon, cfg.step):
        test_end = train_end + cfg.horizon
        yield train_end, test_end

# ------------- 模型工厂（候选趋势）-------------
def fit_predict_ets(y_train: pd.Series, y_all_index: pd.DatetimeIndex,
                    trend=None, damped_trend=False, horizon=1) -> float:
    model = ETSModel(y_train, error='add', trend=trend, damped_trend=damped_trend)
    res = model.fit(disp=False)
    # 直接预测未来 horizon 步
    fc = res.forecast(horizon)
    return float(fc.iloc[-1])  # 返回 horizon 的末端点预测

def fit_predict_ucm(y_train: pd.Series, y_all_index: pd.DatetimeIndex,
                    trend=True, damped_trend=False, horizon=1) -> float:
    mod = UnobservedComponents(y_train, level='llevel', trend=trend, damped_trend=damped_trend)
    res = mod.fit(disp=False)
    fc = res.forecast(horizon)
    return float(fc.iloc[-1])

def fit_predict_ols(y_train: pd.Series, y_all_index: pd.DatetimeIndex,
                    horizon=1) -> float:
    # 线性回归：y ~ const + t
    t = np.arange(len(y_train))
    X = sm.add_constant(t)
    res = sm.OLS(y_train.values.astype(float), X).fit()
    t_future = len(y_train) + horizon - 1
    x_new = sm.add_constant(np.array([t_future]))
    pred = res.predict(x_new)
    return float(pred[0])

# 候选模型列表（可按需增减）
CANDIDATES: Dict[str, Dict[str, Any]] = {
    # ETS 家族
    "ETS_SES":            dict(kind="ets",  trend=None, damped_trend=False),
    "ETS_Holt":           dict(kind="ets",  trend='add', damped_trend=False),
    "ETS_DampedHolt":     dict(kind="ets",  trend='add', damped_trend=True),
    # UCM / State Space
    "UCM_LLT":            dict(kind="ucm",  trend=True,  damped_trend=False),   # local linear trend
    "UCM_LLT_Damped":     dict(kind="ucm",  trend=True,  damped_trend=True),
    # OLS 线性趋势
    "OLS_Linear":         dict(kind="ols"),
}

# ------------- 主评估函数 -------------
def evaluate_trend_models(df: pd.DataFrame,
                          value_col: str = "Cost",
                          year_col: str = "Year",
                          cfg: CVConfig = CVConfig(horizon=1, min_train_size=8, step=1),
                          models: Dict[str, Dict[str, Any]] = None
                          ) -> Tuple[pd.DataFrame, str]:
    """
    对多种趋势模型做 rolling-origin CV，返回每个模型的 RMSE/MAE/MAPE，
    并给出最佳模型（以 RMSE 最小为准；可改为 MAE/MAPE）。
    """
    if models is None:
        models = CANDIDATES

    # 准备时间序列
    df = df[[year_col, value_col]].dropna().sort_values(year_col).reset_index(drop=True)
    # 构造日期索引（每年 1 月 1 日）
    idx = pd.to_datetime(df[year_col].astype(str) + "-01-01")
    y = pd.Series(df[value_col].astype(float).values, index=idx, name=value_col)
    n = len(y)

    # 度量容器
    records: List[Dict[str, Any]] = []

    # 对每个模型做 CV
    for name, spec in models.items():
        preds = []
        trues = []

        for tr_end, te_end in rolling_origin_splits(n_obs=n, cfg=cfg):
            y_train = y.iloc[:tr_end+1]
            y_true  = y.iloc[te_end]          # horizon 的末端真值

            try:
                if spec["kind"] == "ets":
                    pred = fit_predict_ets(y_train, y.index,
                                           trend=spec.get("trend"),
                                           damped_trend=spec.get("damped_trend", False),
                                           horizon=cfg.horizon)
                elif spec["kind"] == "ucm":
                    pred = fit_predict_ucm(y_train, y.index,
                                           trend=spec.get("trend", True),
                                           damped_trend=spec.get("damped_trend", False),
                                           horizon=cfg.horizon)
                elif spec["kind"] == "ols":
                    pred = fit_predict_ols(y_train, y.index, horizon=cfg.horizon)
                else:
                    raise ValueError(f"Unknown model kind: {spec['kind']}")
            except Exception as e:
                # 某些模型在极短样本或特定数据下可能失败；跳过该折
                # 也可选择把这次折记作 np.nan
                continue

            preds.append(pred)
            trues.append(float(y_true))

        # 少量失败会导致 preds/trues 为空；做保护
        if len(preds) == 0:
            metrics = dict(RMSE=np.nan, MAE=np.nan, MAPE=np.nan, N=0)
        else:
            metrics = dict(RMSE=rmse(trues, preds),
                           MAE=mae(trues, preds),
                           MAPE=mape(trues, preds),
                           N=len(trues))
        metrics["Model"] = name
        records.append(metrics)

    result = pd.DataFrame.from_records(records).set_index("Model").sort_values("RMSE", ascending=True)
    best_model = result.index[0] if len(result) else None
    return result, best_model

def preprocessed_df(df):
    col0 = df.columns[0]
    col1 = df.columns[1]

    # 1) 提取 '-' 前面的年份并转为整数
    df['Year'] = (
        df[col0]
        .astype(str)  # 确保为字符串
        .str.strip()  # 去掉前后空白
        .str.split('-', n=1)  # 按第一个 '-' 拆分
        .str[0]  # 取左侧部分
    )

    # 尝试把 Year 转成整数，无法转换的会变成 NaN
    df['Year'] = pd.to_numeric(df['Year'], errors='coerce').astype('Int64')
    # 2) 重命名并把第二列转为数值型
    df['Cost'] = pd.to_numeric(df[col1], errors='coerce')
    df.drop(df.columns[:2], axis=1, inplace=True)
    return df


# ---------- 主程序 ----------
# sheet_configs = [
#     ('table_c2', 'Crop Productivity'),
#     ('table_18', 'Dairy Productivity'),
#     ('table_c4', 'Sheep Productivity'),
#     ('table_c5', 'Beef Productivity')
# ]

sheet_configs = [
    ('Sheet1', 'label_cost'),
]

all_plots = []
all_results = []

for sheet_name, var_name in sheet_configs:
    df = pd.read_excel('../0_original_data/labour_cost.xlsx', usecols="A,G")
    col0, col1 = df.columns[0], df.columns[1]

    # 解析 "Aug-14" -> 2014
    df['Year'] = pd.to_datetime(df[col0], format='%b-%y', errors='coerce').dt.year
    df.rename(columns={col1: 'Cost'}, inplace=True)
    df['Cost'] = pd.to_numeric(df['Cost'], errors='coerce')
    # 提取 Year 和 Cost，排序
    df_processed = df[['Year', 'Cost']].dropna(subset=['Year']).sort_values('Year').reset_index(drop=True)

    cfg = CVConfig(horizon=1, min_train_size=8, step=1)
    summary, best = evaluate_trend_models(df_processed, value_col="Cost", year_col="Year", cfg=cfg)
    print(summary)
    print(f"\n {sheet_name} best model by RMSE:", best)