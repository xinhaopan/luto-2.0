import os
import shutil
from joblib import Parallel, delayed
import tools.config as config
import tools.tools as tools

tar_path = os.path.join(config.TASK_DIR, "carbon_price", "Report")
os.makedirs(tar_path, exist_ok=True)

def copy_report(input_name):
    input_path = tools.get_path(input_name)
    report_dir = os.path.join(input_path, "DATA_REPORT","REPORT_HTML")
    target_dir = os.path.join(tar_path, input_name)

    if not os.path.exists(report_dir):
        return f"[skip] {input_name}: no DATA_REPORT"

    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)

    shutil.copytree(report_dir, target_dir)
    return f"[ok] {input_name}"

results = Parallel(n_jobs=-1, verbose=100)(   # verbose=10 会打印进度
    delayed(copy_report)(name) for name in config.INPUT_FILES
)

for r in results:
    print(r)
