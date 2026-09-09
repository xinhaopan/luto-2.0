"""
make_transition_cache.py -- one-off: cache the land-use transition matrix.

Why not rebuild it here
    T_MAT is assembled in luto/data.py from ag_tmatrix.npy, ag_to_ep_tmatrix.npy,
    ag_to_destock_tmatrix.npy, ep_to_ag_tmatrix.npy and
    transition_cost_clearing_forest.npz, plus a dozen special cases: non-ag land
    cannot go back to natural land, clearing non-ag costs a specific amount,
    destocked natural land inherits the unallocated-natural costs for livestock,
    and so on.  Mirroring those forty lines here would be a second copy that
    silently drifts from the model.  The Data object pickled inside each run
    archive already holds the finished matrix, so it is read from there.

    NaN means the transition is not permitted; a finite value is its cost per
    hectare.  Only the pattern matters for the figure.

What it writes
    EXCEL_DIR/34_transition_matrix.csv   from_lu x to_lu, NaN where not allowed

Run it from the 4_Draw/code directory with the xpluto environment:
    <xpluto>/python.exe tools/make_transition_cache.py
"""

import os
import sys
import zipfile

import joblib
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
os.chdir(os.path.dirname(_HERE))

from tools.data_helper import get_zip_info                  # noqa: E402
from tools.parameters import EXCEL_DIR, input_files         # noqa: E402

CACHE_NAME = '34_transition_matrix.csv'


def main() -> None:
    repo = os.path.abspath(os.path.join(_HERE, '../../../../..'))
    if repo not in sys.path:
        sys.path.insert(0, repo)
    import luto.settings as settings
    settings.AG2050_MODE = True          # Paper3 runs are Ag2050 mode
    import luto.data  # noqa: F401       # registers the classes for unpickling

    # The matrix is the same in every scenario, so the first run will do.
    scenario = input_files[0]
    info = get_zip_info(scenario)
    if info is None:
        raise FileNotFoundError(f'No Run_Archive.zip for {scenario}')
    zip_path, _prefix = info

    os.makedirs(EXCEL_DIR, exist_ok=True)
    tmp = os.path.join(EXCEL_DIR, '_tmp_Data_tmat.lz4')
    with zipfile.ZipFile(zip_path) as z:
        member = next((n for n in z.namelist()
                       if n.endswith('.lz4') and 'Data_RES' in n), None)
        if member is None:
            raise FileNotFoundError(f'No Data_RES*.lz4 inside {zip_path}')
        print(f'  extracting {member}')
        with z.open(member) as src, open(tmp, 'wb') as dst:
            while True:
                chunk = src.read(1 << 24)
                if not chunk:
                    break
                dst.write(chunk)

    print('  loading the Data object (this is the slow part)')
    data = joblib.load(tmp)
    t_mat = data.T_MAT
    frame = pd.DataFrame(
        t_mat.values,
        index=pd.Index([str(v) for v in t_mat['from_lu'].values], name='from_lu'),
        columns=[str(v) for v in t_mat['to_lu'].values],
    )
    os.remove(tmp)

    out = os.path.join(EXCEL_DIR, CACHE_NAME)
    frame.to_csv(out)
    allowed = int(frame.notna().sum().sum())
    total = frame.size
    print(f'  land uses : {frame.shape[0]} from x {frame.shape[1]} to')
    print(f'  allowed   : {allowed:,} of {total:,} '
          f'({100.0 * allowed / total:.1f}%)')
    print(f'  wrote {out}')


if __name__ == '__main__':
    main()
