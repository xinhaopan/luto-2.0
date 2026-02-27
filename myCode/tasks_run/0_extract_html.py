import os
import pandas as pd
import shutil
import subprocess
from joblib import Parallel, delayed
from pathlib import Path


def get_7z_path():
    """检测 7z 路径，确保调用成功"""
    paths = [r"C:\Program Files\7-Zip\7z.exe", "7z"]
    for p in paths:
        if shutil.which(p) or os.path.exists(p):
            return p
    return None


def extract_one_scenario(name1, source_base, target_base):
    """阶段 1：并行提取（利用 SSD 的高并发能力）"""
    try:
        scenario_output_path = source_base / name1 / "output"
        if not scenario_output_path.exists(): return

        # 寻找 DATA_REPORT
        sub_dirs = [d for d in scenario_output_path.iterdir() if d.is_dir()]
        data_report_src = next((d / "DATA_REPORT" for d in sub_dirs if (d / "DATA_REPORT").exists()), None)

        if data_report_src:
            target_report_dir = target_base / name1 / "DATA_REPORT"
            target_report_dir.parent.mkdir(parents=True, exist_ok=True)

            if target_report_dir.exists(): shutil.rmtree(target_report_dir)
            # SSD 并行拷贝的关键：不锁死总线
            shutil.copytree(data_report_src, target_report_dir)
            return True
    except Exception as e:
        print(f"❌ {name1} 提取失败: {e}")
    return False


def run_extreme_pipeline(input_folder):
    source_base = Path(input_folder).resolve()
    target_root_name = f"{input_folder}_Report"
    target_root = Path(target_root_name).resolve()
    target_root.mkdir(parents=True, exist_ok=True)

    exe_7z = get_7z_path()
    if not exe_7z:
        print("❌ 错误：未找到 7-Zip，请确认已安装。")
        return

    # 1. 获取场景列表
    df = pd.read_csv(source_base / "grid_search_template.csv", index_col=0)
    name1_list = df.columns.tolist()

    # --- 阶段 1: 并行提取 (SSD 强项) ---
    print(f"🚀 SSD 并行模式：正在同时提取 {len(name1_list)} 个场景...")
    # n_jobs=-1 利用所有 CPU 核心同时发起 IO 请求
    Parallel(n_jobs=-1)(
        delayed(extract_one_scenario)(n, source_base, target_root)
        for n in name1_list
    )
    print("✅ 提取完成。")

    # --- 阶段 2: 7z 整体多线程压缩 ---
    zip_output = Path(f"{target_root_name}.zip").resolve()
    print(f"⚡ 7-Zip 多线程模式：正在整体打包压缩...")

    # 指令详解：
    # a: 添加
    # -mx1: 最快级别（SSD 写入瓶颈下，mx1 效率最高，mx0 不压缩反而可能因为体积太大卡 IO）
    # -mmt=on: 开启多线程压缩
    # -tzip: 采用 zip 格式（Windows 下处理小文件最快的容器格式）
    cmd = [exe_7z, "a", str(zip_output), str(target_root / "*"), "-mx1", "-mmt=on", "-tzip"]

    try:
        # 使用 shell=True 确保在 Windows 环境下正确加载
        subprocess.run(cmd, check=True, shell=True)
        print(f"🎉 任务圆满完成！\n总提取文件夹：{target_root}\n总压缩包：{zip_output}")
    except Exception as e:
        print(f"❌ 压缩阶段出错: {e}")


if __name__ == "__main__":
    # 使用你指定的 output 目录
    os.chdir(r'F:\Users\s222552331\Work\LUTO2_XH\luto-2.0\output')
    run_extreme_pipeline("20260226_Paper2_Results_NCI")