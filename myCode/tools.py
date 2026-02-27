import pandas as pd
from pathlib import Path

def print_memory_summary(file_path):
    """
    读取内存日志并直接打印：
    - 总耗时
    - 最大内存（GB）
    - 峰值出现时间
    """
    file_path = Path(file_path)

    df = pd.read_csv(
        file_path,
        sep=r"\s+|\t",
        header=None,
        names=["time", "mem_gb"],
        engine="python"
    )

    df["time"] = pd.to_datetime(df["time"])

    time_cost = df["time"].iloc[-1] - df["time"].iloc[0]
    max_mem = df["mem_gb"].max()
    peak_time = df.loc[df["mem_gb"].idxmax(), "time"]

    print("===== Memory Usage Summary =====")
    print(f"Start time : {df['time'].iloc[0]}")
    print(f"End time   : {df['time'].iloc[-1]}")
    print(f"Time cost  : {time_cost}")
    print(f"Max memory : {max_mem:.3f} GB")
    print(f"Peak time  : {peak_time}")
    print("================================")