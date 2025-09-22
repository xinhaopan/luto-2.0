import os
import re
import pandas as pd

def parse_log(base_dir, run_dir,runing_file):
    """
    解析单个 run 的 simulation_log.txt
    - EndYear: "Model finished in 2050"
    - RunTime: "Total run time: 13:03:04"
    - Memory : "Overall peak memory usage: 47.46 GB"
    - Status : Completed / Failed / Running / Log not found
    """
    txt_path = os.path.join(base_dir, run_dir, 'output', 'simulation_log.txt')
    result = {
        "Name": run_dir,
        "Status": "Running",  # 默认 Running
        "EndYear": None,
        "RunTime": None,
        "Memory": None
    }

    if not os.path.exists(txt_path):
        result["Status"] = "Log not found"
        return result

    with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()

            # 结束年份
            m = re.search(r"Model finished in (\d{4})", line)
            if m:
                result["EndYear"] = int(m.group(1))

            # 运行时间
            m = re.search(r"Total run time:\s*([\d:]+)", line)
            if m:
                result["RunTime"] = m.group(1)

            # 内存峰值
            m = re.search(r"Overall peak memory usage:\s*([\d.]+\s*GB)", line)
            if m:
                result["Memory"] = m.group(1)

            # 状态
            if "Run completed." in line:
                result["Status"] = "Completed"
            elif "Run failed." in line:
                result["Status"] = "Failed"
    with open(runing_file, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if "Warning: Gurobi solver did not find an optimal/suboptimal solution for year" in line:
                result["Status"] = "Infeasible"

    return result


def get_first_subfolder(output_dir):
    """获取 output/ 目录下的第一个子文件夹（按名称排序后取第一个）。"""
    if not os.path.isdir(output_dir):
        return None
    subfolders = sorted(
        f for f in os.listdir(output_dir) if os.path.isdir(os.path.join(output_dir, f))
    )
    return subfolders[0] if subfolders else None


def parse_running_year(runing_file):
    """
    从 LUTO_RUN__stdout.log 中抓取所有形如：
      "Running for year 2034"
    的年份，返回最大值。若没有匹配或文件不存在，返回 None。
    """
    if not runing_file or not os.path.exists(runing_file):
        return None

    years = []
    pat = re.compile(r"Running for year\s+(\d{4})")
    with open(runing_file, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = pat.search(line)
            if m:
                years.append(int(m.group(1)))
    return max(years) if years else None


if __name__ == "__main__":
    base_dir = '../../output/20250922_Paper2_Results_HPC_NUM0'
    # base_dir = '../../output/20250921_Paper2_Results'
    target_year = 2050

    # 从 grid_search_template.csv 的列名里找 Run_* 作为 run 目录
    template_df = pd.read_csv(os.path.join(base_dir, 'grid_search_template.csv'), index_col=0)
    run_dirs = [col for col in template_df.columns if col.startswith('Run_')]

    rows = []
    for run_dir in run_dirs:
        # 找 running 文件，解析 RunningYear
        output_dir = os.path.join(base_dir, run_dir, 'output')
        subfolder = get_first_subfolder(output_dir)
        # 解析主日志
        runing_file = (
            os.path.join(output_dir, subfolder, 'LUTO_RUN__stdout.log')
            if subfolder else None
        )
        row = parse_log(base_dir, run_dir,runing_file)

        row["RunningYear"] = parse_running_year(runing_file)

        rows.append(row)

    results_df = pd.DataFrame(rows, columns=["Name","RunningYear", "EndYear", "RunTime", "Memory", "Status"])

    # 保存状态表
    out_excel = os.path.join(base_dir, 'run_status_report.xlsx')
    results_df.to_excel(out_excel, index=False)
    print(results_df)
    print(f"✅ Run 状态表已保存: {out_excel}")

    # # 筛选 EndYear != target_year
    # not_target = results_df[results_df["EndYear"] != target_year]["Name"]
    # if not not_target.empty:
    #     print(f"\n以下运行的 EndYear 不是 {target_year}:")
    #     for name in not_target:
    #         print(" -", name)
    # else:
    #     print(f"\n所有运行的 EndYear 均为 {target_year} ✅")
