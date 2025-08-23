import os
import luto.simulation as sim
import luto.settings as settings
import traceback
import time
from memory_profiler import memory_usage
import shutil, zipfile



# 定义日志文件
log_file = 'output/simulation_log.txt'  # 自定义路径
error_log_file = 'output/error_log.txt'  # 错误日志路径

def write_log(message, file=log_file):
    """写入日志并附加时间戳"""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    with open(file, 'a', encoding="utf-8") as log:
        log.write(f"[{timestamp}] {message}\n")

def monitor_memory(func, *args, **kwargs):
    """监控函数的内存使用并返回结果和峰值内存"""
    mem_usage, result = memory_usage((func, args, kwargs), retval=True, interval=0.1)
    peak_memory = max(mem_usage) / 1024  # 转换为 GB
    return result, peak_memory

def main():

    try:
        # 记录模拟开始的时间
        write_log("Simulation started")
        overall_start_time = time.time()

        # 监控加载数据
        data, load_data_memory = monitor_memory(sim.load_data)
        write_log(f"Data loaded. Peak memory usage: {load_data_memory:.2f} GB")

        # 监控运行模拟
        _, run_memory = monitor_memory(sim.run, data=data)
        write_log(f"Simulation completed. Peak memory usage: {run_memory:.2f} GB")

        # 总结束时间
        overall_end_time = time.time()
        total_duration = overall_end_time - overall_start_time

        # 跨天友好显示
        days = int(total_duration // (24 * 3600))
        hours = int((total_duration % (24 * 3600)) // 3600)
        minutes = int((total_duration % 3600) // 60)
        seconds = int(total_duration % 60)

        if days > 0:
            formatted_duration = f"{days}d {hours:02}:{minutes:02}:{seconds:02}"
        else:
            formatted_duration = f"{hours:02}:{minutes:02}:{seconds:02}"

        # 记录日志
        write_log(f"Total run time: {formatted_duration}")
        write_log(f"Overall peak memory usage: {max(load_data_memory, run_memory):.2f} GB")


    except Exception as e:
        # 记录错误到日志文件
        error_message = f"An error occurred during simulation:\n{str(e)}\n{traceback.format_exc()}"
        write_log(error_message, file=error_log_file)

        # 打印错误信息，便于调试
        print(f"Error in simulation: {e}")
        print("Full traceback written to error_log.txt")



if __name__ == "__main__":
    main()

