import luto.simulation as sim
import luto.settings as settings
import os
import shutil

data = sim.load_data()
sim.run(data=data, base_year=2010, target_year=2050, step_size=settings.STEP_SIZE)
pkl_path = f'{data.path}/data_with_solution.gz'
settings_path = f'{data.path}/model_run_settings.txt'
sim.save_data_to_disk(data,pkl_path)

snapshot_dir = os.path.dirname(settings_path)
run_dir = os.path.dirname(snapshot_dir)

# 需要保留的两个文件的绝对路径
keep = {os.path.abspath(pkl_path), os.path.abspath(settings_path)}

# 1) 清理 run_dir，保留 snapshot_dir
for name in os.listdir(run_dir):
    full = os.path.join(run_dir, name)
    if os.path.abspath(full) == os.path.abspath(snapshot_dir):
        continue
    if os.path.isdir(full):
        shutil.rmtree(full)
    else:
        os.remove(full)

# 2) 清理 snapshot_dir，保留 pkl_path 和 settings_path
for name in os.listdir(snapshot_dir):
    full = os.path.join(snapshot_dir, name)
    if os.path.abspath(full) in keep:
        continue
    if os.path.isdir(full):
        shutil.rmtree(full)
    else:
        os.remove(full)