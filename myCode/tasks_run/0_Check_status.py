import os
import re
import pandas as pd
import glob
import zipfile
import io


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


def get_archived_run_info(archive_path):
    """从 Run_Archive.zip 中提取信息，如果文件损坏则报告失败。"""
    try:
        running_year, memory = None, None
        with zipfile.ZipFile(archive_path, 'r') as zf:
            stdout_log_path = next((name for name in zf.namelist() if 'LUTO_RUN__stdout.log' in name), None)
            mem_log_path = next((name for name in zf.namelist() if
                                 os.path.basename(name).startswith('RES_') and name.endswith('_mem_log.txt')), None)

            if stdout_log_path:
                with zf.open(stdout_log_path) as log_file:
                    with io.TextIOWrapper(log_file, encoding='utf-8', errors='ignore') as text_stream:
                        running_year = parse_year_from_stream(text_stream)
            if mem_log_path:
                with zf.open(mem_log_path) as mem_file:
                    with io.TextIOWrapper(mem_file, encoding='utf-8', errors='ignore') as text_stream:
                        memory = get_max_memory_from_stream(text_stream)

        # 成功读取归档文件
        return {
            "RunningYear": running_year,
            "Memory": memory,
            "Simulation Status": "Success",
            "Output Status": "Success"
        }
    except (zipfile.BadZipFile, FileNotFoundError) as e:
        print(f"Warning: Could not process archive {archive_path}. Reason: {e}")
        # 无法打开或处理归档文件，返回失败状态
        return {
            "RunningYear": None,
            "Memory": None,
            "Simulation Status": "Failed",
            "Output Status": "Failed"
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


if __name__ == "__main__":
    task_name = '20260210_Paper1_Results_aquila'
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
            "Simulation Status": "Running",  # 默认状态
            "Output Status": "Running"  # 默认状态
        }

        output_dir = os.path.join(base_dir, run_dir, 'output')
        subfolder = get_first_subfolder(output_dir)

        if subfolder:
            # 检查 lz4 文件是否存在以判断模拟是否完成
            lz4_pattern = os.path.join(output_dir, subfolder, 'Data_RES*.lz4')
            if glob.glob(lz4_pattern):
                row["Simulation Status"] = "Success"
                # 此情况下 Output Status 保持默认的 Failed
            else:
                error_log_pattern = os.path.join(output_dir, 'error_log.txt')
                if glob.glob(error_log_pattern):
                    row["Simulation Status"] = "Failed"

            # 无论状态如何，都尝试获取运行年份和内存信息
            runing_file = os.path.join(output_dir, subfolder, 'LUTO_RUN__stdout.log')
            row["RunningYear"] = parse_running_year(runing_file)

            mem_log_pattern = os.path.join(output_dir, subfolder, 'RES_*_mem_log.txt')
            mem_log_files = glob.glob(mem_log_pattern)
            if mem_log_files:
                with open(mem_log_files[0], 'r', encoding='utf-8', errors='ignore') as f:
                    row["Memory"] = get_max_memory_from_stream(f)

        rows.append(row)

    results_df = pd.DataFrame(rows, columns=["Name", "RunningYear", "Memory", "Simulation Status", "Output Status"])
    out_excel = os.path.join(base_dir, f'{task_name}_run_status_report.xlsx')
    results_df.to_excel(out_excel, index=False)

    # 设置 pandas 显示选项以完整显示 DataFrame
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)

    print(results_df)
    print(f"✅ Run 状态表已保存: {out_excel}")

