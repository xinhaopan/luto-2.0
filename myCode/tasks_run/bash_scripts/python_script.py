import os, pathlib
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

def _find_lz4(output_path="output"):
    """Find Data_RES*.lz4 in the first '2010-2050' output subfolder.
    Returns (lz4_path, subfolder_name)."""
    import glob
    subfolders = [f for f in os.listdir(output_path)
                  if os.path.isdir(os.path.join(output_path, f)) and '2010-2050' in f]
    if not subfolders:
        raise FileNotFoundError(f"No output subfolder found in '{output_path}'")
    found = glob.glob(os.path.join(output_path, subfolders[0], "Data_RES*.lz4"))
    if not found:
        raise FileNotFoundError(f"No Data_RES*.lz4 found in '{subfolders[0]}'")
    return found[0], subfolders[0]

def is_zip_valid(path):
    """Quick check: zip file exists and its central directory is readable."""
    import zipfile
    if not os.path.exists(path):
        return False
    try:
        with zipfile.ZipFile(path, 'r') as zf:
            zf.infolist()
    except zipfile.BadZipFile:
        return False
    return True

def zip_only():
    """Load data from .lz4 and run create_zip (no simulation)."""
    import joblib
    lz4_path, _ = _find_lz4()
    print(f"Loading data from: {lz4_path}")
    data = joblib.load(lz4_path)
    report_zip_path = create_zip(data)
    print(f"Archiving complete. Report zip: {report_zip_path}")

def write_only():
    """Load .lz4, run write_outputs, then create_zip (no simulation)."""
    import joblib, sys
    from luto.tools.write import write_outputs
    import luto.settings as settings

    # If a valid archive already exists, clean up and exit
    archive_path = 'Run_Archive.zip'
    if is_zip_valid(archive_path):
        print(f"'{archive_path}' exists and is valid. Cleaning up other files.")
        for item in os.listdir('.'):
            if item != archive_path:
                try:
                    p = pathlib.Path(item)
                    if p.is_file() or p.is_symlink():
                        p.unlink()
                    elif p.is_dir():
                        shutil.rmtree(p)
                except Exception as e:
                    print(f"Failed to delete {item}. Reason: {e}")
        print("Cleanup complete.")
        sys.exit(0)
    elif os.path.exists(archive_path):
        print(f"'{archive_path}' is corrupted. Deleting and regenerating.")
        try:
            os.remove(archive_path)
        except Exception as e:
            print(f"Failed to delete corrupted archive: {e}")
            sys.exit(1)

    lz4_path, _ = _find_lz4()
    print(f"Loading data from: {lz4_path}")
    data = joblib.load(lz4_path)

    print("Writing outputs...")
    write_outputs(data)
    print("Write complete.")

    if settings.KEEP_OUTPUTS:
        print("KEEP_OUTPUTS is True. Skipping archiving.")
    else:
        report_zip_path = create_zip(data)
        print(f"Archiving complete. Report zip: {report_zip_path}")

def report_only():
    """Load .lz4 and run create_report (no simulation, no archiving)."""
    import joblib, shutil
    from luto import tools, settings
    from luto.tools.report.create_report_layers import save_report_layer
    from luto.tools.report.create_report_data import save_report_data

    lz4_path, subfolder = _find_lz4()

    # Skip if report already exists
    html_path = os.path.join("output", subfolder, "DATA_REPORT", "data",
                             "map_layers", "map_water_yield_NonAg.js")
    if os.path.exists(html_path):
        print(f"Report already exists: {html_path}")
        return

    print(f"Loading data from: {lz4_path}")
    data = joblib.load(lz4_path)

    print("Creating report...")
    save_dir = (f"{settings.OUTPUT_DIR}/{tools.read_timestamp()}"
                f"_RF{settings.RESFACTOR}_{settings.SIM_YEARS[0]}-{settings.SIM_YEARS[-1]}")

    @tools.LogToFile(f"{save_dir}/LUTO_RUN_", mode='a')
    def _create_report():
        shutil.copytree('luto/tools/report/VUE_modules',
                        f"{data.path}/DATA_REPORT", dirs_exist_ok=True)
        save_report_data(data.path)
        save_report_layer(data)

    _create_report()
    print("Report created successfully.")

def create_zip(data):
    """Zip results into two archives: full archive + report-only archive, then clean up."""
    import os, pathlib, shutil, zipfile

    output_dir = pathlib.Path(data.path).absolute()
    simulation_root = output_dir.parent.parent  # .../Run_XXXX/

    run_idx = simulation_root.name
    report_data_dir = simulation_root.parent / 'Report_Data'
    report_data_dir.mkdir(parents=True, exist_ok=True)

    report_zip_path = report_data_dir / f'{run_idx}.zip'
    archive_path = simulation_root / 'Run_Archive.zip'

    with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as run_zip, \
         zipfile.ZipFile(report_zip_path, 'w', zipfile.ZIP_DEFLATED) as report_zip:
        for root, dirs, files in os.walk(simulation_root):
            files = [f for f in files if f != 'Run_Archive.zip']
            for file in files:
                abs_path = pathlib.Path(root) / file
                if 'DATA_REPORT' in abs_path.as_posix():
                    report_zip.write(abs_path, arcname=abs_path.relative_to(output_dir))
                else:
                    run_zip.write(abs_path, arcname=abs_path.relative_to(simulation_root))

    for item in os.listdir(simulation_root):
        if item != 'Run_Archive.zip':
            item_path = simulation_root / item
            try:
                if item_path.is_file() or item_path.is_symlink():
                    item_path.unlink()
                elif item_path.is_dir():
                    shutil.rmtree(item_path)
            except Exception as e:
                print(f"Failed to delete {item}. Reason: {e}")

    return report_zip_path

def main():
    import luto.simulation as sim
    import luto.settings as settings

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
            write_log("Archiving results...")
            report_zip_path = create_zip(data)
            write_log(f"Archiving complete. Report zip: {report_zip_path}")
            write_log("Cleanup complete.")

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
    if os.path.exists('zip_mode.flag'):
        zip_only()
    elif os.path.exists('write_mode.flag'):
        write_only()
    elif os.path.exists('report_mode.flag'):
        report_only()
    else:
        main()

