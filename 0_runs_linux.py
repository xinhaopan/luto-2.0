import os
import dill
import luto.simulation as sim
import traceback
import time

# 定义日志文件
log_file = 'output/simulation_log.txt'  # 自定义路径
error_log_file = 'output/error_log.txt'  # 错误日志路径

def write_log(message, file=log_file):
    """写入日志并附加时间戳"""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    with open(file, 'a') as log:
        log.write(f"[{timestamp}] {message}\n")

def main(start_year, end_year):
    try:
        # 确保日志目录存在
        os.makedirs('output', exist_ok=True)

        # 记录模拟开始的时间
        overall_start_time = time.time()
        write_log("Simulation started")

        # 加载数据
        data = sim.load_data()
        write_log("Data loaded")

        # 运行模拟
        sim.run(data=data, base=start_year, target=end_year)
        write_log("Simulation completed")

        # 保存数据
        pkl_path = f'{data.path}/data_with_solution.gz'

        sim.save_data_to_disk(data,pkl_path)
        write_log(f"Data with solution saved in {data.path}.")

        # 写输出结果
        write_log(f"start write_outputs")
        from luto.tools.write import write_outputs
        write_outputs(data)
        write_log(f"Outputs written")

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
    main(2010,2050)  # 设置开始和结束年份
