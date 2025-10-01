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
        # 确保日志目录存在
        os.makedirs('output', exist_ok=True)

        # 记录模拟开始的时间
        write_log("Simulation started")
        overall_start_time = time.time()

        # 监控加载数据
        data, load_data_memory = monitor_memory(sim.load_data)
        write_log(f"Data loaded. Peak memory usage: {load_data_memory:.2f} GB")

        # 监控运行模拟
        _, simulation_memory = monitor_memory(sim.run, data=data)
        write_log(f"Run completed. Peak memory usage: {simulation_memory:.2f} GB")

        write_log(f"Model finished in {data.last_year}")

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
        write_log(f"Overall peak memory usage: {max(load_data_memory, simulation_memory):.2f} GB")

        # Remove all files except the report directory if settings.KEEP_OUTPUTS is False
        '''
        KEEP_OUTPUTS is not originally defined in the settings, but will be added in the `luto/tools/create_task_runs/create_running_tasks.py` file.
        '''

        if settings.KEEP_OUTPUTS:

            # Save the data object to disk
            pass

        else:
            report_dir = f"{data.path}/DATA_REPORT"
            archive_path = './DATA_REPORT.zip'

            # Zip the output directory, and remove the original directory
            with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for root, dirs, files in os.walk(report_dir):
                    for file in files:
                        abs_path = os.path.join(root, file)
                        rel_path = os.path.relpath(abs_path, start=report_dir)
                        zipf.write(abs_path, arcname=rel_path)

            # —— 只保留 ZIP 和指定的 RES gz 文件 ——
            keep = {
                os.path.basename(archive_path),  # 'DATA_REPORT.zip'
                f"Data_RES{settings.RESFACTOR}.gz",
            }

            for k in keep:
                if k == os.path.basename(archive_path):
                    continue  # DATA_REPORT.zip 已经在当前目录，无需复制
                src = os.path.join(data.path, k)
                dst = os.path.join('.', k)
                if os.path.exists(src):
                    if not os.path.exists(dst):
                        shutil.copy2(src, dst)
                else:
                    print(f"[WARN] 源文件不存在: {src}")

            for item in os.listdir('.'):
                if item in keep:
                    continue
                try:
                    if os.path.isfile(item) or os.path.islink(item):
                        os.unlink(item)
                    elif os.path.isdir(item):
                        shutil.rmtree(item)
                except Exception as e:
                    print(f"Failed to delete {item}. Reason: {e}")



    except Exception as e:
        # 记录错误到日志文件
        write_log(f"Run failed.", file=log_file)
        write_log(f"Model finished in {data.last_year}", file=log_file)

        error_message = f"An error occurred during simulation:\n{str(e)}\n{traceback.format_exc()}"
        write_log(error_message, file=error_log_file)

        # 打印错误信息，便于调试
        print(f"Error in simulation: {e}")
        print("Full traceback written to error_log.txt")



if __name__ == "__main__":
    main()

