import os
import time
from memory_profiler import memory_usage


def main():
    os.chdir('../..')  # 改变工作目录
    from myCode.tasks_run_in_windows.tools.helpers import create_task_runs
    file_path = 'myCode/tasks_run_in_windows/Custom_runs/settings_template4.csv'
    create_task_runs(from_path=file_path,use_multithreading=False, num_workers=1)


if __name__ == "__main__":
    # 定义输出日志文件
    log_file = "performance_log.txt"

    # 记录开始时间
    start_time = time.time()

    # 监控内存使用
    mem_usage = memory_usage(main, interval=0.1)  # 调用 main，并监控内存
    max_memory = max(mem_usage)  # 获取最大内存占用

    # 记录结束时间
    end_time = time.time()

    # 计算运行时间
    run_time = (end_time - start_time) / 60

    print(f"time: {run_time:.2f} min")
    print(f"mem: {max_memory:.2f} MiB")

    # 将结果写入日志文件
    with open(log_file, "w") as f:
        f.write(f"time: {run_time:.2f} min\n")
        f.write(f"mem: {max_memory:.2f} MiB\n")

    print(f"Finish: {log_file}")
