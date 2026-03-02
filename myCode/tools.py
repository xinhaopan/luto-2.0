import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
from pathlib import Path


def _read_mem_log(file_path):
    """Read a mem_log.txt file into a DataFrame with datetime and GB columns."""
    df = pd.read_csv(
        Path(file_path),
        sep=r"\s+|\t",
        header=None,
        names=["date", "time_str", "mem_gb"],
        engine="python",
    )
    # Support both "YYYY-MM-DD HH:MM:SS" (two columns) and single-column timestamp
    if df["time_str"].isna().all():
        df = pd.read_csv(
            Path(file_path),
            sep=r"\t",
            header=None,
            names=["timestamp", "mem_gb"],
            engine="python",
        )
        df["time"] = pd.to_datetime(df["timestamp"])
    else:
        df["time"] = pd.to_datetime(df["date"] + " " + df["time_str"])
    df["elapsed_min"] = (df["time"] - df["time"].iloc[0]).dt.total_seconds() / 60
    return df[["time", "elapsed_min", "mem_gb"]]


def print_memory_summary(file_path):
    """
    读取内存日志并直接打印：
    - 总耗时
    - 最大内存（GB）
    - 峰值出现时间
    """
    df = _read_mem_log(file_path)

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


def plot_memory_usage(
    file_path,
    output_path=None,
    title=None,
    x_unit="min",          # "min" | "hour" | "datetime"
    color="#2196F3",
    figsize=(12, 4),
    dpi=150,
    annotate_peak=True,
    hlines=None,           # list of (y_value, label, color) to draw horizontal reference lines
):
    """
    Plot memory-usage-over-time from a mem_log.txt file.

    Parameters
    ----------
    file_path   : path to mem_log.txt  (format: "YYYY-MM-DD HH:MM:SS\\tGB")
    output_path : where to save the PNG; if None, saves next to the log file
    title       : figure title; auto-generated if None
    x_unit      : "min" → elapsed minutes, "hour" → elapsed hours, "datetime" → wall-clock
    color       : line colour
    hlines      : e.g. [(128, "RAM limit", "red")] draws a red dashed line at y=128
    """
    df = _read_mem_log(file_path)
    file_path = Path(file_path)

    if x_unit == "min":
        x = df["elapsed_min"]
        xlabel = "Elapsed time (min)"
    elif x_unit == "hour":
        x = df["elapsed_min"] / 60
        xlabel = "Elapsed time (h)"
    else:
        x = df["time"]
        xlabel = "Wall-clock time"

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(x, df["mem_gb"], color=color, linewidth=1.2, label="Memory (GB)")
    ax.fill_between(x, df["mem_gb"], alpha=0.15, color=color)

    # Peak annotation
    if annotate_peak:
        peak_idx = df["mem_gb"].idxmax()
        px = x.iloc[peak_idx]
        py = df["mem_gb"].iloc[peak_idx]
        ax.axhline(py, linestyle="--", linewidth=0.8, color="gray", alpha=0.6)
        ax.annotate(
            f"Peak: {py:.1f} GB",
            xy=(px, py),
            xytext=(0.02, 0.95),
            textcoords="axes fraction",
            fontsize=9,
            va="top",
            color="dimgray",
            arrowprops=dict(arrowstyle="->", color="dimgray", lw=0.8),
        )

    # Optional horizontal reference lines
    if hlines:
        for yval, label, lc in hlines:
            ax.axhline(yval, linestyle=":", linewidth=1, color=lc, label=label)
        ax.legend(fontsize=9, loc="upper right")

    # x-axis formatting
    if x_unit == "datetime":
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        fig.autofmt_xdate()

    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel("Memory (GB)", fontsize=11)
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.spines[["top", "right"]].set_visible(False)

    if title is None:
        title = f"Memory usage — {file_path.parent.name}/{file_path.name}"
    ax.set_title(title, fontsize=12, pad=8)

    # Duration label in corner
    total_min = df["elapsed_min"].iloc[-1]
    ax.text(
        0.99, 0.02,
        f"Duration: {total_min:.0f} min  |  Max: {df['mem_gb'].max():.1f} GB",
        transform=ax.transAxes,
        ha="right", va="bottom",
        fontsize=8, color="gray",
    )

    plt.tight_layout()

    if output_path is None:
        output_path = file_path.with_suffix(".png")
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")
    return output_path


if __name__ == "__main__":
    log = "F:/Users/s222552331/Work/LUTO2_XH/luto-2.0/output/20260301_Paper2_Results_test/carbon_price/mem_log.txt"
    print_memory_summary(log)
    plot_memory_usage(log)
    
