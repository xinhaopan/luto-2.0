import os
import re
import pandas as pd
import glob
import zipfile
import io
import time
import subprocess
from datetime import datetime


def get_max_memory_from_stream(stream):
    """从文件流中解析内存使用情况，返回第二列的最大值。"""
    max_mem = 0.0
    for line in stream:
        try:
            parts = line.strip().split()
            if len(parts) >= 2:
                mem_val = float(parts[-1])
                if mem_val > max_mem:
                    max_mem = mem_val
        except (ValueError, IndexError):
            continue  # 跳过格式不正确的行
    return f"{max_mem:.3f} GB" if max_mem > 0 else None


def get_memory_summary_from_stream(stream):
    """Parse runtime and peak memory from a RES_*_mem_log.txt stream."""
    start_time = None
    end_time = None
    peak_time = None
    max_mem = None

    for line in stream:
        parts = line.strip().split()
        if len(parts) < 3:
            continue

        try:
            timestamp = datetime.strptime(f"{parts[0]} {parts[1]}", "%Y-%m-%d %H:%M:%S")
            mem_val = float(parts[-1])
        except (ValueError, IndexError):
            continue

        if start_time is None:
            start_time = timestamp
        end_time = timestamp

        if max_mem is None or mem_val > max_mem:
            max_mem = mem_val
            peak_time = timestamp

    runtime = end_time - start_time if start_time and end_time else None

    return {
        "StartTime": start_time.strftime("%Y-%m-%d %H:%M:%S") if start_time else None,
        "EndTime": end_time.strftime("%Y-%m-%d %H:%M:%S") if end_time else None,
        "Runtime": str(runtime) if runtime else None,
        "RuntimeHours": round(runtime.total_seconds() / 3600, 3) if runtime else None,
        "Memory": f"{max_mem:.3f} GB" if max_mem is not None else None,
        "MaxMemoryGB": round(max_mem, 3) if max_mem is not None else None,
        "PeakTime": peak_time.strftime("%Y-%m-%d %H:%M:%S") if peak_time else None,
    }


def parse_year_from_stream(stream):
    """从文件流中解析 'Running for year XXXX' 并返回最大年份。"""
    years = []
    pat = re.compile(r"Running for year\s+(\d{4})")
    for line in stream:
        m = pat.search(line)
        if m:
            years.append(int(m.group(1)))
    return max(years) if years else None


def parse_solver_state_from_stream(stream):
    """从 stdout 文本流中解析求解器最终状态。

    关键：求解器带重试逻辑（"Non-optimal status ... retrying with next attempt"），
    某一年可能先报 Infeasible、重试后又求解成功。因此不能"见到 infeasible 就返回"，
    必须扫完整个日志、以【最后一个决定性事件】为准。

    决定性事件（按出现顺序记录，取最后一个）:
      - "Solver status for year XXXX: INFEASIBLE"  -> 所有重试均失败，确定不可行
      - "Infeasible or unbounded model"            -> 不可行/无界（求解直接崩在这里）
      - "Optimal solution found"                   -> 该年求解成功
    """
    final_state = None          # 最后一个决定性事件
    error_state = None
    timeout_state = None

    pat_status = re.compile(r"Solver status for year\s+\d{4}\s*:\s*([A-Z_]+)", re.IGNORECASE)

    for line in stream:
        low = line.lower()

        # 权威终态：某年所有重试都失败后打印的最终状态
        m = pat_status.search(line)
        if m:
            status = m.group(1).upper()
            if status == 'INFEASIBLE':
                final_state = 'Infeasible'
            elif 'UNBOUNDED' in status:
                final_state = 'Unbounded'
            elif status == 'OPTIMAL':
                final_state = 'Optimal'
            else:
                final_state = status.capitalize()
            continue

        # 求解直接崩在不可行/无界（不会再打印上面的最终状态行）
        if 'infeasible or unbounded' in low:
            final_state = 'Infeasible'
            continue

        # 单次尝试成功（重试成功也走这里，会覆盖之前的 infeasible 尝试）
        if 'optimal solution found' in low or 'optimal objective' in low or 'barrier solved model' in low:
            final_state = 'Optimal'
            continue

        if 'time limit' in low or 'timed out' in low or 'aborted' in low:
            timeout_state = 'Timeout'

        if 'traceback' in low or 'exception' in low:
            error_state = 'Error'

    # 不可行/无界优先于后续的 Python 报错（崩溃往往只是不可行的后果）
    if final_state in ('Infeasible', 'Unbounded'):
        return final_state
    if timeout_state:
        return timeout_state
    if error_state:
        return error_state
    return final_state


def parse_output_stage_from_stream(stream):
    """判断写出（write_outputs）阶段是否【已开始】。

    写出阶段的标记：write_outputs 会逐年打印 "Mosaic maps written for year XXXX"。
    注意：'Report created successfully' 只在独立的 Report 模式下才会打印，普通 Run
    模式跑完 write_outputs 也不会有，所以不能拿它当写出完成的判据。
    """
    for line in stream:
        if 'Mosaic maps written' in line or 'Writing outputs' in line or 'Report created successfully' in line:
            return 'Writing'
    return None


def parse_run_result(output_dir):
    """从 output/simulation_log.txt 读取整轮运行的最终结果（python_script.py 写入）。

    成功 -> "Run completed. Peak memory usage: ..."
    失败 -> "Run failed."
    返回 'Completed' / 'Failed' / None（仍在跑或日志缺失）。
    """
    log_path = os.path.join(output_dir, 'simulation_log.txt')
    if not os.path.exists(log_path):
        return None
    try:
        with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read()
    except Exception:
        return None
    if 'Run failed' in text:
        return 'Failed'
    if 'Run completed' in text:
        return 'Completed'
    return None

def check_report_created(runing_file):
    """检查 LUTO_RUN__stdout.log 文件中是否包含 'Report created successfully'。"""
    if not runing_file or not os.path.exists(runing_file):
        return False
    try:
        with open(runing_file, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if "Report created successfully" in line:
                    return True
    except Exception as e:
        print(f"Warning: Could not read {runing_file}. Reason: {e}")
    return False


def get_archived_run_info(archive_path):
    """从 Run_Archive.zip 中提取信息，如果文件损坏则报告失败。"""
    try:
        running_year = None
        solver_state = None
        memory_summary = {
            "StartTime": None,
            "EndTime": None,
            "Runtime": None,
            "RuntimeHours": None,
            "Memory": None,
            "MaxMemoryGB": None,
            "PeakTime": None,
        }
        archive_size_mb = None
        num_data_lz4 = 0
        stderr_error_summary = None
        with zipfile.ZipFile(archive_path, 'r') as zf:
            archive_size_mb = os.path.getsize(archive_path) / 1e6
            names = zf.namelist()
            stdout_log_path = next((name for name in names if 'LUTO_RUN__stdout.log' in name), None)
            mem_log_path = next((name for name in names if
                                 os.path.basename(name).startswith('RES_') and name.endswith('_mem_log.txt')), None)
            stderr_log_path = next((name for name in names if 'LUTO_RUN__stderr.log' in name), None)
            num_data_lz4 = sum(1 for name in names if os.path.basename(name).startswith('Data_RES') and name.endswith('.lz4'))

            if stdout_log_path:
                with zf.open(stdout_log_path) as log_file:
                    with io.TextIOWrapper(log_file, encoding='utf-8', errors='ignore') as text_stream:
                        full_text = text_stream.read()
                        running_year = parse_year_from_stream(full_text.splitlines())
                        solver_state = parse_solver_state_from_stream(full_text.splitlines())
            if mem_log_path:
                with zf.open(mem_log_path) as mem_file:
                    with io.TextIOWrapper(mem_file, encoding='utf-8', errors='ignore') as text_stream:
                        memory_summary = get_memory_summary_from_stream(text_stream)
            if stderr_log_path:
                with zf.open(stderr_log_path) as err_file:
                    with io.TextIOWrapper(err_file, encoding='utf-8', errors='ignore') as text_stream:
                        for line in text_stream:
                            if 'ERROR' in line or 'Traceback' in line or 'Exception' in line:
                                stderr_error_summary = line.strip()
                                break

        return {
            "RunningYear": running_year,
            **memory_summary,
            "Solver Status": solver_state or ("Solved" if num_data_lz4 > 0 else "Unknown"),
            "Output Status": "Success",
            "ArchiveExists": True,
            "ArchiveSize_MB": round(archive_size_mb, 3) if archive_size_mb is not None else None,
            "Num_Data_RES_lz4": num_data_lz4,
            "StdErr_Errors": stderr_error_summary,
        }
    except (zipfile.BadZipFile, FileNotFoundError) as e:
        print(f"Warning: Could not process archive {archive_path}. Reason: {e}")
        return {
            "RunningYear": None,
            "StartTime": None,
            "EndTime": None,
            "Runtime": None,
            "RuntimeHours": None,
            "Memory": None,
            "MaxMemoryGB": None,
            "PeakTime": None,
            "Solver Status": "Error",
            "Output Status": "Failed",
            "ArchiveExists": True,
            "ArchiveSize_MB": None,
            "Num_Data_RES_lz4": 0,
            "StdErr_Errors": None,
        }


def get_first_subfolder(output_dir):
    """获取 output/ 目录下的第一个子文件夹。"""
    if not os.path.isdir(output_dir): return None
    subfolders = sorted(f for f in os.listdir(output_dir) if os.path.isdir(os.path.join(output_dir, f)))
    return subfolders[0] if subfolders else None


def parse_running_year(runing_file):
    """从 LUTO_RUN__stdout.log 文件路径解析当前运行年份。"""
    if not runing_file or not os.path.exists(runing_file): return None
    with open(runing_file, "r", encoding="utf-8", errors="ignore") as f:
        return parse_year_from_stream(f)


def get_dir_size_mb(directory):
    """用 du -sb 快速计算目录大小（MB），适合 HPC Lustre 文件系统。"""
    if not directory or not os.path.isdir(directory):
        return None
    try:
        result = subprocess.run(
            ['du', '-sb', directory],
            capture_output=True, text=True, timeout=30
        )
        if result.returncode == 0:
            return round(int(result.stdout.split()[0]) / 1e6, 1)
    except Exception:
        pass
    return None


def check_stderr_errors(stderr_file):
    """检查 stderr 文件中是否包含 ERROR / Traceback 等信息，返回首条摘要。"""
    if not stderr_file or not os.path.exists(stderr_file):
        return None
    try:
        with open(stderr_file, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                if 'ERROR' in line or 'Traceback' in line or 'Exception' in line:
                    return line.strip()
    except Exception:
        return None
    return None


if __name__ == "__main__":
    # 输出运行时间
    print("="*80)
    print(f"脚本运行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    # 支持检查多个任务
    task_names = [
        '20260710_Paper3_aquila',
        '20260711_Paper3_HPC',
        # 可以在此添加更多任务名称
    ]
    target_year = 2050

    # 遍历每个任务
    for task_name in task_names:
        print(f"\n{'='*80}")
        print(f"正在检查任务: {task_name}")
        print(f"{'='*80}\n")
        
        base_dir = f'../../output/{task_name}'
        
        # 检查任务目录是否存在
        if not os.path.exists(base_dir):
            print(f"⚠️  任务目录不存在: {base_dir}")
            continue
        
        template_file = os.path.join(base_dir, 'grid_search_template.csv')
        if not os.path.exists(template_file):
            print(f"⚠️  未找到模板文件: {template_file}")
            continue
            
        template_df = pd.read_csv(template_file, index_col=0)
        run_dirs = [col for col in template_df.columns if col.startswith('Run_')]

        rows = []
        for run_dir in run_dirs:
            archive_path = os.path.join(base_dir, run_dir, 'Run_Archive.zip')

            if os.path.exists(archive_path):
                row = get_archived_run_info(archive_path)
                row["Name"] = run_dir
                rows.append(row)
                continue

            # --- 如果归档不存在，执行新的状态检查逻辑 ---
            row = {
                "Name": run_dir,
                "RunningYear": None,
                "StartTime": None,
                "EndTime": None,
                "Runtime": None,
                "RuntimeHours": None,
                "Memory": None,
                "MaxMemoryGB": None,
                "PeakTime": None,
                "Solver Status": "Running",  # 默认状态
                "Output Status": "Running",  # 默认状态
                "ArchiveExists": False,
                "ArchiveSize_MB": None,
                "Num_Data_RES_lz4": 0,
                "StdErr_Errors": None,
                "Output_Files_MB": None,
                "Stdout_LastModified": None,
            }

            output_dir = os.path.join(base_dir, run_dir, 'output')
            subfolder = get_first_subfolder(output_dir)

            # 搜集候选检查目录：output 根目录以及第一个子文件夹（如果有）
            candidate_dirs = [output_dir]
            if subfolder:
                candidate_dirs.append(os.path.join(output_dir, subfolder))

            # 在所有候选目录中查找 Data_RES*.lz4 文件（累计计数）
            total_lz4 = 0
            for d in candidate_dirs:
                lz4_pattern = os.path.join(d, 'Data_RES*.lz4')
                lz4_files = glob.glob(lz4_pattern)
                total_lz4 += len(lz4_files)
            if total_lz4 > 0:
                row["Num_Data_RES_lz4"] = total_lz4

            # 检查 error_log（根目录或子目录）以判断是否失败
            error_found = False
            for d in candidate_dirs:
                error_log_pattern = os.path.join(d, 'error_log.txt')
                if glob.glob(error_log_pattern):
                    error_found = True
                    break

            # 查找 stdout：优先使用子文件夹（如果存在），否则在根目录查找
            runing_file = None
            for d in candidate_dirs[::-1]:
                candidate_stdout = os.path.join(d, 'LUTO_RUN__stdout.log')
                if os.path.exists(candidate_stdout):
                    runing_file = candidate_stdout
                    break

            # 一次性读入 stdout，解析：运行年份 / 求解器终态 / 写出阶段
            solver_state = None
            output_stage = None
            if runing_file:
                try:
                    with open(runing_file, 'r', encoding='utf-8', errors='ignore') as rf:
                        txt = rf.read().splitlines()
                    row["RunningYear"] = parse_year_from_stream(txt)
                    solver_state = parse_solver_state_from_stream(txt)
                    output_stage = parse_output_stage_from_stream(txt)
                except Exception:
                    pass

            # ---- Solver Status ----
            if solver_state:
                row["Solver Status"] = solver_state
            elif total_lz4 > 0:
                # 已存出 Data_RES*.lz4 → 求解已完成
                row["Solver Status"] = 'Optimal'
            elif error_found:
                row["Solver Status"] = 'Error'
            elif runing_file:
                row["Solver Status"] = 'Running'
            else:
                row["Solver Status"] = 'Unknown'

            # ---- Output Status ----
            # 权威判据：simulation_log.txt 里的 "Run completed" / "Run failed."
            run_result = parse_run_result(output_dir)

            if row["Solver Status"] in ('Infeasible', 'Unbounded'):
                # 求解就没过，根本走不到写出阶段
                row["Output Status"] = 'Not started'
            elif run_result == 'Completed':
                row["Output Status"] = 'Success'
            elif run_result == 'Failed' or error_found:
                # 已进入写出阶段才算写出失败；否则是求解阶段就失败了
                row["Output Status"] = 'Failed' if output_stage == 'Writing' else 'Not started'
            elif output_stage == 'Writing':
                row["Output Status"] = 'Writing'
            else:
                row["Output Status"] = 'Running'

            # mem log
            mem_found = False
            for d in candidate_dirs:
                mem_log_pattern = os.path.join(d, 'RES_*_mem_log.txt')
                mem_log_files = glob.glob(mem_log_pattern)
                if mem_log_files:
                    with open(mem_log_files[0], 'r', encoding='utf-8', errors='ignore') as f:
                        row.update(get_memory_summary_from_stream(f))
                    mem_found = True
                    break

            # stderr 错误摘要（在候选目录中查找）
            stderr_summary = None
            for d in candidate_dirs:
                stderr_file = os.path.join(d, 'LUTO_RUN__stderr.log')
                stderr_summary = check_stderr_errors(stderr_file)
                if stderr_summary:
                    row["StdErr_Errors"] = stderr_summary
                    break

            # 输出文件夹大小（优先子文件夹），以及 stdout 最后修改时间
            out_dir_for_size = candidate_dirs[-1] if len(candidate_dirs) > 1 else candidate_dirs[0]
            row["Output_Files_MB"] = get_dir_size_mb(out_dir_for_size)
            if runing_file and os.path.exists(runing_file):
                try:
                    mtime = os.path.getmtime(runing_file)
                    row["Stdout_LastModified"] = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(mtime))
                except Exception:
                    row["Stdout_LastModified"] = None

            rows.append(row)

        results_df = pd.DataFrame(rows, columns=[
            "Name",
            "RunningYear",
            "Solver Status",
            "Output Status",
            "Runtime",
            "RuntimeHours",
            "Memory",
            "MaxMemoryGB",
            "PeakTime",
            # "ArchiveExists",
            # "ArchiveSize_MB",
            # "Num_Data_RES_lz4",
            # "StdErr_Errors",
            # "Output_Files_MB",
            # "Stdout_LastModified",
        ])
        out_excel = os.path.join(base_dir, f'{task_name}_run_status_report.xlsx')
        results_df.to_excel(out_excel, index=False)

        # 设置 pandas 显示选项以完整显示 DataFrame
        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        pd.set_option('display.max_colwidth', None)

        print(results_df)
        print(f"✅ Run 状态表已保存: {out_excel}")
    
    print(f"\n{'='*80}")
    print(f"✅ 所有任务检查完成！共检查了 {len(task_names)} 个任务")
    print(f"{'='*80}")
