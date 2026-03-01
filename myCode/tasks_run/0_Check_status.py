import os
import re
import pandas as pd
import glob
import zipfile
import io
import time
import subprocess


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
    """从 stdout/stderr 文本流中解析求解器状态（Optimal, Infeasible, Unbounded, Timeout, Error, Running）。
    
    优先级顺序（从高到低）:
    1. Infeasible ("Infeasible model", "INFEASIBLE", "no feasible", "Status: 3")
    2. Unbounded
    3. Timeout ("time limit", "timed out", "interrupted")
    4. Error (error, exception, traceback)
    5. Optimal (complete solve success)
    """
    error_state = None
    found_optimal = False
    
    for line in stream:
        low = line.lower()
        
        # Highest priority: Infeasible status (multiple patterns)
        if 'infeasible model' in low or 'infeasible' in low and 're-solving' not in low:
            return 'Infeasible'
        if 'barrier reported infeasible' in low:
            return 'Infeasible'
        if 'status: 3' in low:  # Gurobi status code 3 = infeasible
            return 'Infeasible'
        if 'no feasible' in low:
            return 'Infeasible'
        
        # Second priority: Unbounded
        if 'unbounded' in low:
            return 'Unbounded'
        
        # Third priority: Timeout
        if 'time limit' in low or 'timed out' in low or 'interrupted' in low or 'aborted' in low:
            return 'Timeout'
        
        # Fourth priority: Error
        if ('error' in line and 'no error' not in low) or 'exception' in low or 'traceback' in low:
            error_state = 'Error'
        
        # Fifth priority: Optimal (but don't return yet, keep checking for errors)
        if 'optimal objective' in low or 'completed solve' in low or 'barrier solved model' in low:
            found_optimal = True
    
    # Return in priority order
    if error_state:
        return error_state
    if found_optimal:
        return 'Optimal'
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
        running_year, memory = None, None
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
                        memory = get_max_memory_from_stream(text_stream)
            if stderr_log_path:
                with zf.open(stderr_log_path) as err_file:
                    with io.TextIOWrapper(err_file, encoding='utf-8', errors='ignore') as text_stream:
                        for line in text_stream:
                            if 'ERROR' in line or 'Traceback' in line or 'Exception' in line:
                                stderr_error_summary = line.strip()
                                break

        return {
            "RunningYear": running_year,
            "Memory": memory,
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
            "Memory": None,
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
    task_name = '20260226_Paper2_Results_aquila'
    base_dir = f'../../output/{task_name}'
    target_year = 2050
    template_df = pd.read_csv(os.path.join(base_dir, 'grid_search_template.csv'), index_col=0)
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
            "Memory": None,
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

        # 检查是否存在 "Report created successfully"
        if runing_file and check_report_created(runing_file):
            row["Output Status"] = "Success"

        # 获取运行年份和内存信息（优先子文件夹的 mem log）
        if runing_file:
            row["RunningYear"] = parse_running_year(runing_file)
            try:
                with open(runing_file, 'r', encoding='utf-8', errors='ignore') as rf:
                    txt = rf.read().splitlines()
                    solver_state = parse_solver_state_from_stream(txt)
                    if solver_state:
                        row["Solver Status"] = solver_state
                    elif total_lz4 > 0:
                        # Output files exist → assume solver completed successfully
                        row["Solver Status"] = 'Optimal'
                    elif error_found:
                        row["Solver Status"] = 'Error'
                    else:
                        row["Solver Status"] = 'Running'
            except Exception:
                if total_lz4 > 0:
                    row["Solver Status"] = 'Optimal'
                elif error_found:
                    row["Solver Status"] = 'Error'
                else:
                    row["Solver Status"] = 'Unknown'
        else:
            # 没有 stdout 的情况下，用 lz4 或 error 判定
            if total_lz4 > 0:
                row["Solver Status"] = 'Optimal'
            elif error_found:
                row["Solver Status"] = 'Error'
            else:
                row["Solver Status"] = 'Unknown'

        # mem log
        mem_found = False
        for d in candidate_dirs:
            mem_log_pattern = os.path.join(d, 'RES_*_mem_log.txt')
            mem_log_files = glob.glob(mem_log_pattern)
            if mem_log_files:
                with open(mem_log_files[0], 'r', encoding='utf-8', errors='ignore') as f:
                    row["Memory"] = get_max_memory_from_stream(f)
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
        "Memory",
        "Solver Status",
        "Output Status",
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