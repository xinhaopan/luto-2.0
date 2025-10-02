import os
import shutil
import zipfile

path = r"\\school-les-m.shares.deakin.edu.au\school-les-m\Planet-A\LUF-Modelling\LUTO2_XH\LUTO2\output\20251001_Nick_task\Run_01_GBF2_high_CUT_50_CarbonPrice_0\output\2025_09_30__23_42_01_RF5_2010-2050"

report_dir = f"{path}/DATA_REPORT"
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
    f"Data_RES5.gz",
}

for k in keep:
    if k == os.path.basename(archive_path):
        continue  # DATA_REPORT.zip 已经在当前目录，无需复制
    src = os.path.join(path, k)
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