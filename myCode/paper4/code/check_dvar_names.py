import sys, zipfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from tools.price_slice_utils import build_run_map

run_map, cp_vals, bp_vals = build_run_map()
baseline_zip = run_map[(0.0, 0.0)]
with zipfile.ZipFile(baseline_zip) as arc:
    names = [n for n in arc.namelist() if 'dvar' in n.lower() and '2050' in n]
    print('\n'.join(names[:30]))
