import luto.simulation as sim
import luto.settings as settings
import os
import shutil
import sys

data = sim.load_data()
sim.run(data=data)
pkl_path = f'{data.path}/data_with_solution.gz'
sim.save_data_to_disk(data,pkl_path)

snapshot_dir = os.path.abspath(data.path)
pkl_path      = os.path.join(snapshot_dir, 'data_with_solution.gz')
settings_path = os.path.join(snapshot_dir, 'settings.py')

# 计算上两级目录
output_dir = os.path.abspath(os.path.join(snapshot_dir, os.pardir))
run_dir    = os.path.abspath(os.path.join(snapshot_dir, os.pardir, os.pardir))

# 打印检查
print("snapshot_dir =", snapshot_dir)
print("output_dir   =", output_dir)
print("run_dir      =", run_dir)

src_path = os.path.join(run_dir,'luto','settings.py')
shutil.copy2(src_path, settings_path)

# 确保这几个目录都存在
for d in (run_dir, output_dir, snapshot_dir):
    if not os.path.isdir(d):
        print(f"错误：目录不存在：{d}")
        sys.exit(1)

deleted = {'run_dir': [], 'output_dir': [], 'snapshot_dir': []}

# 1) 在 run_dir 下，只保留 output_dir
for name in os.listdir(run_dir):
    full = os.path.join(run_dir, name)
    if os.path.abspath(full) != output_dir:
        deleted['run_dir'].append(full)
        if os.path.isdir(full):
            shutil.rmtree(full)
        else:
            os.remove(full)

# 2) 在 output_dir 下，只保留 snapshot_dir
for name in os.listdir(output_dir):
    full = os.path.join(output_dir, name)
    if os.path.abspath(full) != snapshot_dir:
        deleted['output_dir'].append(full)
        if os.path.isdir(full):
            shutil.rmtree(full)
        else:
            os.remove(full)

# 3) 在 snapshot_dir 下，只保留两个文件
keep = {os.path.abspath(pkl_path), os.path.abspath(settings_path)}
for name in os.listdir(snapshot_dir):
    full = os.path.join(snapshot_dir, name)
    if os.path.abspath(full) not in keep:
        deleted['snapshot_dir'].append(full)
        if os.path.isdir(full):
            shutil.rmtree(full)
        else:
            os.remove(full)

# 打印删除清单
print("\n删除清单：")
print(f"— run_dir ({run_dir}) 下删除：")
for path in deleted['run_dir']:
    print("    ", path)
print(f"\n— output_dir ({output_dir}) 下删除：")
for path in deleted['output_dir']:
    print("    ", path)
print(f"\n— snapshot_dir ({snapshot_dir}) 下删除：")
for path in deleted['snapshot_dir']:
    print("    ", path)