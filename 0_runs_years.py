import os
import dill
import luto.simulation as sim
import traceback
import time
from memory_profiler import memory_usage

# 定义日志文件
log_file = 'output/simulation_log.txt'  # 自定义路径
error_log_file = 'output/error_log.txt'  # 错误日志路径

def write_log(message, file=log_file):
    """写入日志并附加时间戳"""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    with open(file, 'a') as log:
        log.write(f"[{timestamp}] {message}\n")

def monitor_memory(func, *args, **kwargs):
    """监控函数的内存使用并返回结果和峰值内存"""
    mem_usage, result = memory_usage((func, args, kwargs), retval=True, interval=0.1)
    peak_memory = max(mem_usage) / 1024  # 转换为 GB
    return result, peak_memory

def main():
    # 设置参数
    start_year = 2010

    try:
        # 确保日志目录存在
        os.makedirs('output', exist_ok=True)

        # 记录模拟开始的时间
        overall_start_time = time.time()
        write_log("Simulation started")

        # 监控加载数据
        data, load_data_memory = monitor_memory(sim.load_data)
        write_log(f"Data loaded. Peak memory usage: {load_data_memory:.2f} GB")

        for target_year in range(2015, 2051,5):
            print(f"Running simulation for target year {target_year}")
            enum_start_time = time.time()
            # 监控运行模拟
            write_log(f"Start simulation for target year {target_year}")
            _, simulation_memory = monitor_memory(sim.run, data=data, base=start_year, target=target_year)
            write_log(f"Simulation completed. Peak memory usage: {simulation_memory:.2f} GB")

            # 保存数据
            pkl_path = f'{data.path}/data_with_solution.pkl'

            with open(pkl_path, 'wb') as f:
                dill.dump(data, f)
            write_log(f"{pkl_path} has been saved.")

            # 监控写输出结果
            write_log(f"start write_outputs")
            from luto.tools.write import write_outputs
            _, write_output_memory = monitor_memory(write_outputs, data)
            write_log(f"Outputs written. Peak memory usage: {write_output_memory:.2f} GB")
            enum_end_time = time.time()
            enum_time = (enum_end_time - enum_start_time) / 3600
            write_log(f"Total time for {target_year} is {enum_time:.2f} h")
            write_log(
                f"Run {target_year} overall peak memory usage: {max(load_data_memory, simulation_memory, write_output_memory):.2f} GB")
            write_log("---------------------------------------------")
        # 总结束时间
        overall_end_time = time.time()
        total_duration = (overall_end_time - overall_start_time) / 3600  # 总用时

        # 记录模拟过程的详细信息
        write_log(f"Total run time: {total_duration:.2f} h")


    except Exception as e:
        # 记录错误到日志文件
        error_message = f"An error occurred during simulation:\n{str(e)}\n{traceback.format_exc()}"
        write_log(error_message, file=error_log_file)

        # 打印错误信息，便于调试
        print(f"Error in simulation: {e}")
        print("Full traceback written to error_log.txt")

if __name__ == "__main__":
    main()
