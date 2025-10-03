import os
import pandas as pd

import os
import zipfile
import shutil

def clean_and_recompress(root_path, archive_root=True):
    # 1. 找到所有Run_Archive.zip
    run_zip_paths = []
    for dirpath, dirnames, filenames in os.walk(root_path):
        for filename in filenames:
            if filename == "Run_Archive.zip":
                run_zip_paths.append(os.path.join(dirpath, filename))

    # 2. 处理每个Run_Archive.zip，只保留DATA_REPORT相关内容
    for zip_path in run_zip_paths:
        print(f"Processing {zip_path}")
        # 收集需要保留的文件内容
        with zipfile.ZipFile(zip_path, "r") as zf:
            names_to_keep = [name for name in zf.namelist() if "DATA_REPORT" in os.path.basename(name)]
            keep_files = {}
            for name in names_to_keep:
                keep_files[name] = zf.read(name)

        # 覆盖原zip，只写保留文件
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for name, data in keep_files.items():
                zf.writestr(name, data)

    # 3. 可选：压缩整个root_path到上一级目录
    if archive_root:
        parent_dir = os.path.dirname(root_path)
        base_name = os.path.basename(root_path)
        archive_path = os.path.join(parent_dir, base_name)
        shutil.make_archive(archive_path, "zip", root_path)
        print(f"Recompressed whole directory to {archive_path}.zip")

def find_missing_reports_by_col(path):
    # 读取csv列名
    csv_path = os.path.join(path, "grid_search_template.csv")
    df = pd.read_csv(csv_path)
    col_names = df.columns.tolist()

    missing_cols = []
    for col_name in col_names:
        new_path = os.path.join(path, str(col_name))
        zip_path = os.path.join(new_path, "DATA_REPORT.zip")
        if not os.path.exists(zip_path):
            missing_cols.append(col_name)
    return missing_cols

path = "../../output/20251002_Cost_curve_task"
print(find_missing_reports_by_col(path))